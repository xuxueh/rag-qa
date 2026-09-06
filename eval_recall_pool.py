"""GPT 评审第 1 点验证：召回池 5 vs 10 对 rerank 天花板的影响
核心问题：正确答案在 rerank 输入池里的覆盖率（rerank 不能找回没召回的）
对比：
  A. retrieve top5  → 正确项在池内比例（当前生产链路）
  B. retrieve top10 → 正确项在池内比例
  C. retrieve top20 → 正确项在池内比例
用法: python eval_recall_pool.py（rerank 43×3 较慢，约 8-12 分钟）
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

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set_v2.json"), encoding="utf-8") as f:
    eval_set = json.load(f)
items = [i for i in eval_set if i["category"] != "无答案问题"]


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


def rank_in(texts, target):
    for i, t in enumerate(texts):
        if text2src.get(t, "") == target:
            return i + 1
    return None


print("\n" + "=" * 64)
print("召回池大小对正确答案覆盖率的影响（rerank 天花板）")
print("=" * 64)
print(f"{'召回池':<8}{'池内命中率':<12}{'rerank后R@1':<12}{'rerank后R@3':<12}{'rerank后MRR':<10}")
print("-" * 64)

for pool in (5, 10, 20):
    pool_hit = 0
    r1 = r3 = 0
    mrr = 0.0
    for item in items:
        recall = hybrid_retrieve(item["question"], top_k=pool)
        # 池内命中（rerank 天花板）
        if rank_in(recall, item["source"]):
            pool_hit += 1
        # rerank 后（精排取 top10 看排序质量；实际生产取 top3）
        ranked = rerank(item["question"], recall, top_n=10)
        r = rank_in(ranked, item["source"])
        if r:
            if r <= 1: r1 += 1
            if r <= 3: r3 += 1
            mrr += 1.0 / r
    n = len(items)
    print(f"top{pool:<5}{pool_hit/n:<14.1%}{r1/n:<14.1%}{r3/n:<14.1%}{mrr/n:.3f}")

print("=" * 64)
print("解读: '池内命中率'是 rerank 的理论上限（rerank 不能找回没召回的）")
