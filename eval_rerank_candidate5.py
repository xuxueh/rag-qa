"""验证：rerank 候选从 10 减到 5 是否损失精度（性能优化依据）
假设：混合检索 Top-5 命中 100% → 正确答案必在 RRF 前 5 → rerank 5 个即可
用法：python eval_rerank_candidate5.py
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

print("构建知识库...")
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


def hybrid_retrieve(query, top_k):
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


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json"), encoding="utf-8") as f:
    eval_set = json.load(f)


def hit_at_k(texts, target, k):
    return any(text2src.get(t, "") == target for t in texts[:k])


# 方案对比：rerank 10 个候选 vs 5 个候选
for cand in (10, 5):
    hits = {"top1": 0, "top3": 0, "top5": 0}
    for item in eval_set:
        q, target = item["question"], item["source"]
        recall = hybrid_retrieve(q, top_k=cand)  # 召回 cand 个
        ranked = rerank(q, recall, top_n=5)
        if hit_at_k(ranked, target, 1): hits["top1"] += 1
        if hit_at_k(ranked, target, 3): hits["top3"] += 1
        if hit_at_k(ranked, target, 5): hits["top5"] += 1
    total = len(eval_set)
    print(f"\n候选数 {cand}: Top-1 {hits['top1']/total:.1%} | Top-3 {hits['top3']/total:.1%} | Top-5 {hits['top5']/total:.1%}")
