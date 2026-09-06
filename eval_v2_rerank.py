"""P1 验证：rerank 是否弥补 RRF 的跨文档短板
对比：混合检索(RRF) vs 混合+Rerank，在 43 条分类测试上的 Recall@K/MRR
用法: python eval_v2_rerank.py（加载 rerank 模型，约 2-3 分钟）
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
from rerank import rerank

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("构建知识库...（约 1-2 分钟）")
loader = DirectoryLoader(rq.DOC_DIR, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""], chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name=rq.EMBEDDING_MODEL_PATH)
db = Chroma.from_documents(chunks, embeddings)

chunk_texts = [c.page_content for c in chunks]
text2src = {}
for c in chunks:
    src = c.metadata.get("source", "")
    fn = src.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    text2src[c.page_content] = fn
bm25 = BM25Okapi([jieba.lcut(t) for t in chunk_texts])

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set_v2.json"), encoding="utf-8") as f:
    eval_set = json.load(f)


def hybrid_retrieve(query, top_k=10):
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


def rank_in(texts, target):
    for i, t in enumerate(texts):
        if text2src.get(t, "") == target:
            return i + 1
    return None


def evaluate(mode):
    """mode: 'hybrid' 或 'hybrid_rerank'"""
    items = [i for i in eval_set if i["category"] != "无答案问题"]
    recall = {1: 0, 3: 0, 5: 0, 10: 0}
    mrr = 0.0
    by_cat = {}
    for item in items:
        cat = item["category"]
        by_cat.setdefault(cat, []).append(item)
    cat_stats = {}
    for cat, cat_items in by_cat.items():
        r = {1: 0, 3: 0, 5: 0, 10: 0}
        m = 0.0
        for it in cat_items:
            texts = hybrid_retrieve(it["question"], top_k=10)
            if mode == "hybrid_rerank":
                texts = rerank(it["question"], texts, top_n=10)  # 精排但不截断，测 R@K
            rank = rank_in(texts, it["source"])
            if rank:
                for k in (1, 3, 5, 10):
                    if rank <= k: r[k] += 1
                m += 1.0 / rank
        n = len(cat_items)
        cat_stats[cat] = {"n": n, "R@1": r[1]/n, "R@3": r[3]/n, "R@5": r[5]/n, "R@10": r[10]/n, "MRR": m/n}
    # 总体
    for it in items:
        texts = hybrid_retrieve(it["question"], top_k=10)
        if mode == "hybrid_rerank":
            texts = rerank(it["question"], texts, top_n=10)
        rank = rank_in(texts, it["source"])
        if rank:
            for k in (1, 3, 5, 10):
                if rank <= k: recall[k] += 1
            mrr += 1.0 / rank
    n = len(items)
    return {"总体": {"n": n, "R@1": recall[1]/n, "R@3": recall[3]/n, "R@5": recall[5]/n, "R@10": recall[10]/n, "MRR": mrr/n}, "分类": cat_stats}


def show(title, result):
    print("\n" + "=" * 70)
    print(f"📊 {title}")
    ov = result["总体"]
    print(f"总体: R@1 {ov['R@1']:.1%} | R@3 {ov['R@3']:.1%} | R@5 {ov['R@5']:.1%} | R@10 {ov['R@10']:.1%} | MRR {ov['MRR']:.3f}")
    print("-" * 70)
    for cat, s in result["分类"].items():
        print(f"{cat:<7} R@1 {s['R@1']:.1%} | R@3 {s['R@3']:.1%} | R@10 {s['R@10']:.1%} | MRR {s['MRR']:.3f}")
    print("=" * 70)


hybrid = evaluate("hybrid")
show("混合检索（RRF，无 rerank）", hybrid)

hr = evaluate("hybrid_rerank")
show("混合检索 + Rerank", hr)

# 跨文档对比
print("\n🎯 跨文档问题对比（P1 核心）:")
print(f"  Hybrid only:  R@1 {hybrid['分类']['跨文档问题']['R@1']:.1%} | MRR {hybrid['分类']['跨文档问题']['MRR']:.3f}")
print(f"  +Rerank:      R@1 {hr['分类']['跨文档问题']['R@1']:.1%} | MRR {hr['分类']['跨文档问题']['MRR']:.3f}")
