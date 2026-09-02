"""Chunk 策略优化评测：RecursiveCharacterTextSplitter（按句子边界切） vs 原硬切
流程：新切块 → 混合检索 + rerank → Top-1/3/5 命中率（对比当前 95/100/100）
用法：python eval_chunk_opt.py
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

# 1. 用 RecursiveCharacterTextSplitter 切块（按中文句子边界）
print("构建知识库（RecursiveCharacterTextSplitter）...")
loader = DirectoryLoader(rq.DOC_DIR, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
documents = loader.load()
print(f"✓ 加载文档: {len(documents)} 份")

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    chunk_size=200,
    chunk_overlap=20,
)
chunks = splitter.split_documents(documents)
print(f"✓ 切块: {len(chunks)} 块（原硬切 47 块）")

embeddings = HuggingFaceEmbeddings(model_name=rq.EMBEDDING_MODEL_PATH)
db = Chroma.from_documents(chunks, embeddings)
print("✓ 向量库构建完成")

chunk_texts = [c.page_content for c in chunks]
text2src = {}
for c in chunks:
    src = c.metadata.get("source", "")
    fn = src.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    text2src[c.page_content] = fn

# BM25 索引
tokenized = [jieba.lcut(t) for t in chunk_texts]
bm25 = BM25Okapi(tokenized)


def hybrid_retrieve(query, top_k=10):
    """混合检索（向量 + BM25，RRF 融合）"""
    vec_docs = db.similarity_search(query, k=top_k * 2)
    vec_ranks = {d.page_content: i + 1 for i, d in enumerate(vec_docs)}
    bm25_scores = bm25.get_scores(jieba.lcut(query))
    bm25_order = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_ranks = {chunk_texts[i]: rank + 1 for rank, i in enumerate(bm25_order)}
    all_docs = set(vec_ranks.keys()) | set(bm25_ranks.keys())
    rrf = {}
    for doc in all_docs:
        s = 0.0
        if doc in vec_ranks:
            s += 1.0 / (60 + vec_ranks[doc])
        if doc in bm25_ranks:
            s += 1.0 / (60 + bm25_ranks[doc])
        rrf[doc] = s
    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


# 2. 测试集
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json"), encoding="utf-8") as f:
    eval_set = json.load(f)


def hit_at_k(texts, target_file, k):
    for t in texts[:k]:
        if text2src.get(t, "") == target_file:
            return True
    return False


# 3. 评测：混合检索 top-10 → rerank top-5
hits = {"top1": 0, "top3": 0, "top5": 0}
for item in eval_set:
    q = item["question"]
    target = item["source"]
    recall = hybrid_retrieve(q, top_k=10)
    ranked = rerank(q, recall, top_n=5)
    if hit_at_k(ranked, target, 1):
        hits["top1"] += 1
    if hit_at_k(ranked, target, 3):
        hits["top3"] += 1
    if hit_at_k(ranked, target, 5):
        hits["top5"] += 1

total = len(eval_set)
print("=" * 50)
print(f"评测集：{total} 条 | 切块：句子边界（200字/重叠20）")
print(f"Top-1 命中率：{hits['top1']/total:.1%}（{hits['top1']}/{total}）")
print(f"Top-3 命中率：{hits['top3']/total:.1%}（{hits['top3']}/{total}）")
print(f"Top-5 命中率：{hits['top5']/total:.1%}（{hits['top5']}/{total}）")
print("=" * 50)
print("\n对比（混合检索 + rerank 链路）：")
print(f"  原硬切块（47块）  Top-1 95.0% | Top-3 100.0% | Top-5 100.0%")
print(f"  句子边界切（{len(chunks)}块） Top-1 {hits['top1']/total:.1%} | Top-3 {hits['top3']/total:.1%} | Top-5 {hits['top5']/total:.1%}")
