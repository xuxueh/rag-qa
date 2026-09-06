"""P1 失败案例分析：跨文档问题在 Vector/BM25/RRF 各环节的表现
目标：找到"混合检索在跨文档问题上 R@1 反而下降"的原因
对每条跨文档问题，打印正确答案在三个检索器中的排名
用法: python eval_failure_analysis.py
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

cross_docs = [i for i in eval_set if i["category"] == "跨文档问题"]
print(f"跨文档问题 {len(cross_docs)} 条\n")


def rank_in(texts, target):
    """正确答案(target 文件)首次出现的位置，None 表示未命中"""
    for i, t in enumerate(texts):
        if text2src.get(t, "") == target:
            return i + 1
    return None


def bm25_rank_texts(query, top_k=10):
    scores = bm25.get_scores(jieba.lcut(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [chunk_texts[i] for i in order[:top_k]]


def hybrid_rank_texts(query, top_k=10):
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


for item in cross_docs:
    q = item["question"]
    target = item["source"]
    print("=" * 70)
    print(f"Q: {q}")
    print(f"目标文件: {target}")

    v = db.similarity_search(q, k=10)
    v_texts = [d.page_content for d in v]
    b_texts = bm25_rank_texts(q)
    h_texts = hybrid_rank_texts(q)

    r_v = rank_in(v_texts, target)
    r_b = rank_in(b_texts, target)
    r_h = rank_in(h_texts, target)

    print(f"Vector 中正确项排名: {r_v if r_v else '未命中(top10)'}")
    print(f"BM25   中正确项排名: {r_b if r_b else '未命中(top10)'}")
    print(f"RRF    中正确项排名: {r_h if r_h else '未命中(top10)'}")

    # 看 top-3 是什么文件（理解检索被什么"带偏"）
    def top_sources(texts):
        seen = []
        for t in texts[:3]:
            fn = text2src.get(t, "?")
            if fn not in seen:
                seen.append(fn)
        return seen

    print(f"Vector top3 来源: {top_sources(v_texts)}")
    print(f"BM25   top3 来源: {top_sources(b_texts)}")
    print(f"RRF    top3 来源: {top_sources(h_texts)}")
