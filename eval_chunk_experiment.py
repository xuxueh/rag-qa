"""P2 Chunk 参数实验：chunk_size 200/400/600/800 × 句子边界切分
指标：Recall@1/5/10 + MRR（混合检索，无 rerank——chunk 影响的是召回质量）
记录：chunk 数量（chunk 越大数量越少 → 检索粒度越粗）
用法: python eval_chunk_experiment.py（4 次重建知识库，约 8-12 分钟）
"""
import sys
import os
import json
import time

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

print("加载文档 + 嵌入模型（复用一次）...")
loader = DirectoryLoader(rq.DOC_DIR, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
documents = loader.load()
embeddings = HuggingFaceEmbeddings(model_name=rq.EMBEDDING_MODEL_PATH)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set_v2.json"), encoding="utf-8") as f:
    eval_set = json.load(f)
items = [i for i in eval_set if i["category"] != "无答案问题"]


def evaluate_chunk(chunk_size):
    t0 = time.time()
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""], chunk_size=chunk_size, chunk_overlap=20)
    chunks = splitter.split_documents(documents)
    db = Chroma.from_documents(chunks, embeddings)
    chunk_texts = [c.page_content for c in chunks]
    text2src = {}
    for c in chunks:
        src = c.metadata.get("source", "")
        fn = src.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        text2src[c.page_content] = fn
    bm25 = BM25Okapi([jieba.lcut(t) for t in chunk_texts])
    build_time = time.time() - t0

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

    recall = {1: 0, 5: 0, 10: 0}
    mrr = 0.0
    retr_time = 0.0
    for item in items:
        t1 = time.time()
        texts = hybrid_retrieve(item["question"], top_k=10)
        retr_time += time.time() - t1
        rank = None
        for i, t in enumerate(texts):
            if text2src.get(t, "") == item["source"]:
                rank = i + 1
                break
        if rank:
            for k in (1, 5, 10):
                if rank <= k: recall[k] += 1
            mrr += 1.0 / rank
    n = len(items)
    return {
        "chunk_size": chunk_size,
        "n_chunks": len(chunks),
        "Recall@1": recall[1] / n,
        "Recall@5": recall[5] / n,
        "Recall@10": recall[10] / n,
        "MRR": mrr / n,
        "avg_retrieve_s": retr_time / n,
        "build_time_s": round(build_time, 1),
    }


results = []
for size in (200, 400, 600, 800):
    print(f"\n🔬 测试 chunk_size = {size}...")
    r = evaluate_chunk(size)
    results.append(r)
    print(f"  chunks: {r['n_chunks']} | R@1 {r['Recall@1']:.1%} | R@5 {r['Recall@5']:.1%} | "
          f"R@10 {r['Recall@10']:.1%} | MRR {r['MRR']:.3f} | 检索 {r['avg_retrieve_s']*1000:.0f}ms | 构建 {r['build_time_s']}s")

print("\n" + "=" * 78)
print(f"{'chunk_size':<12}{'chunks':<8}{'R@1':<9}{'R@5':<9}{'R@10':<9}{'MRR':<9}{'检索ms':<9}")
print("-" * 78)
for r in results:
    print(f"{r['chunk_size']:<12}{r['n_chunks']:<8}{r['Recall@1']:.1%}    {r['Recall@5']:.1%}    "
          f"{r['Recall@10']:.1%}    {r['MRR']:.3f}    {r['avg_retrieve_s']*1000:.0f}")
print("=" * 78)
