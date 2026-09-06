"""P0 评测 v2：Recall@K + MRR + 分类统计（纯向量 vs 混合检索）
- 测试集：eval_set_v2.json（50 条，7 类）
- 指标：Recall@1/3/5/10、MRR（正确答案在检索结果中的位置）
- 对比：纯向量检索 vs 混合检索（RRF）
- 无答案问题不参与检索命中统计（单独计数）

用法: python eval_v2.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import jieba
jieba.setLogLevel(20)
from rank_bm25 import BM25Okapi

import rag_qa as rq

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. 构建
print("构建知识库...（约 1-2 分钟）")
loader = DirectoryLoader(rq.DOC_DIR, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""], chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name=rq.EMBEDDING_MODEL_PATH)
db = Chroma.from_documents(chunks, embeddings)
print(f"✓ {len(chunks)} 块")

chunk_texts = [c.page_content for c in chunks]
text2src = {}
for c in chunks:
    src = c.metadata.get("source", "")
    fn = src.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    text2src[c.page_content] = fn

bm25 = BM25Okapi([jieba.lcut(t) for t in chunk_texts])

# 2. 测试集
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set_v2.json"), encoding="utf-8") as f:
    eval_set = json.load(f)


def vector_retrieve(query, top_k=10):
    """纯向量检索：返回文档文本列表"""
    docs = db.similarity_search(query, k=top_k)
    return [d.page_content for d in docs]


def hybrid_retrieve(query, top_k=10):
    """混合检索（向量 + BM25，RRF）"""
    vec_docs = db.similarity_search(query, k=top_k * 2)
    vec_ranks = {d.page_content: i + 1 for i, d in enumerate(vec_docs)}
    bm25_scores = bm25.get_scores(jieba.lcut(query))
    bm25_order = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_ranks = {chunk_texts[i]: rank + 1 for rank, i in enumerate(bm25_order)}
    all_docs = set(vec_ranks) | set(bm25_ranks)
    rrf = {}
    for doc in all_docs:
        s = 0.0
        if doc in vec_ranks: s += 1.0 / (60 + vec_ranks[doc])
        if doc in bm25_ranks: s += 1.0 / (60 + bm25_ranks[doc])
        rrf[doc] = s
    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


def first_rank(texts, target_file):
    """返回正确答案首次出现的位置（1-based），未命中返回 None"""
    for i, t in enumerate(texts):
        if text2src.get(t, "") == target_file:
            return i + 1
    return None


def evaluate(retriever_fn):
    """评测一个检索器：按分类统计 Recall@K + MRR"""
    # 分类统计
    by_cat = {}
    for item in eval_set:
        if item["category"] == "无答案问题":
            continue  # 无答案不参与检索命中
        cat = item["category"]
        by_cat.setdefault(cat, []).append(item)

    results = {}
    for cat, items in by_cat.items():
        recall = {1: 0, 3: 0, 5: 0, 10: 0}
        mrr_sum = 0.0
        for item in items:
            texts = retriever_fn(item["question"], top_k=10)
            rank = first_rank(texts, item["source"])
            if rank:
                for k in (1, 3, 5, 10):
                    if rank <= k:
                        recall[k] += 1
                mrr_sum += 1.0 / rank
        n = len(items)
        results[cat] = {
            "count": n,
            "Recall@1": recall[1] / n,
            "Recall@3": recall[3] / n,
            "Recall@5": recall[5] / n,
            "Recall@10": recall[10] / n,
            "MRR": mrr_sum / n,
        }
    # 总体
    all_items = [i for i in eval_set if i["category"] != "无答案问题"]
    recall = {1: 0, 3: 0, 5: 0, 10: 0}
    mrr_sum = 0.0
    for item in all_items:
        texts = retriever_fn(item["question"], top_k=10)
        rank = first_rank(texts, item["source"])
        if rank:
            for k in (1, 3, 5, 10):
                if rank <= k: recall[k] += 1
            mrr_sum += 1.0 / rank
    n = len(all_items)
    results["总体"] = {"count": n, "Recall@1": recall[1]/n, "Recall@3": recall[3]/n,
                       "Recall@5": recall[5]/n, "Recall@10": recall[10]/n, "MRR": mrr_sum/n}
    return results


def print_results(title, results):
    print("\n" + "=" * 72)
    print(f"📊 {title}")
    print(f"{'分类':<8}{'条数':<5}{'R@1':<8}{'R@3':<8}{'R@5':<8}{'R@10':<8}{'MRR':<8}")
    print("-" * 72)
    for cat, r in results.items():
        print(f"{cat:<8}{r['count']:<5}{r['Recall@1']:.1%}    {r['Recall@3']:.1%}    "
              f"{r['Recall@5']:.1%}    {r['Recall@10']:.1%}    {r['MRR']:.3f}")
    print("=" * 72)


n_no_answer = sum(1 for i in eval_set if i["category"] == "无答案问题")
print(f"测试集：{len(eval_set)} 条（含无答案 {n_no_answer} 条，参与检索评测 {len(eval_set) - n_no_answer} 条）")

vec_results = evaluate(vector_retrieve)
print_results("纯向量检索（Baseline）", vec_results)

hyb_results = evaluate(hybrid_retrieve)
print_results("混合检索（向量 + BM25）", hyb_results)
