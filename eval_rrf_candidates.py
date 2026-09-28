"""② RRF 候选集实验（GPT 评审 ②）
对比两种融合策略：
- A（当前实现）：Vector top-2K + BM25 全库排序 → RRF（全库参与融合）
- B（典型设计）：Vector top-K + BM25 top-K → 并集 → RRF → top-K

回答：Hybrid 的提升来自"BM25 提供额外候选"还是"BM25 全库 rank 参与 RRF"？

用法: python eval_rrf_candidates.py（无 rerank，只测检索融合策略，约 3-5 分钟）
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import jieba
jieba.setLogLevel(20)
from rank_bm25 import BM25Okapi

from eval_common import build_eval_retriever, first_rank, recall_mrr

retriever = build_eval_retriever()
db = retriever.db
chunks = retriever.chunks
bm25 = retriever.bm25

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set_v2.json"), encoding="utf-8") as f:
    eval_set = json.load(f)
items = [i for i in eval_set if i["category"] != "无答案问题"]


def rrf_full_corpus(query, top_k=10, rrf_k=60):
    """方案 A（当前）：BM25 全库排序参与 RRF"""
    vec_docs = db.similarity_search(query, k=top_k * 2)
    vec_ranks = {d.id: i + 1 for i, d in enumerate(vec_docs)}
    scores = bm25.get_scores(jieba.lcut(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    bm25_ranks = {chunks[i].chunk_id: r + 1 for r, i in enumerate(order)}
    all_ids = set(vec_ranks) | set(bm25_ranks)
    rrf = {}
    for cid in all_ids:
        s = 0.0
        if cid in vec_ranks: s += 1.0 / (rrf_k + vec_ranks[cid])
        if cid in bm25_ranks: s += 1.0 / (rrf_k + bm25_ranks[cid])
        rrf[cid] = s
    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    return [retriever.id2meta[c] for c, _ in ranked[:top_k]]


def rrf_topk_only(query, top_k=10, rrf_k=60):
    """方案 B（典型）：双路各取 Top-K 候选，并集后再 RRF"""
    vec_docs = db.similarity_search(query, k=top_k)
    vec_ranks = {d.id: i + 1 for i, d in enumerate(vec_docs)}
    scores = bm25.get_scores(jieba.lcut(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    bm25_ranks = {chunks[i].chunk_id: r + 1 for r, i in enumerate(order)}
    all_ids = set(vec_ranks) | set(bm25_ranks)
    rrf = {}
    for cid in all_ids:
        s = 0.0
        if cid in vec_ranks: s += 1.0 / (rrf_k + vec_ranks[cid])
        if cid in bm25_ranks: s += 1.0 / (rrf_k + bm25_ranks[cid])
        rrf[cid] = s
    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    return [retriever.id2meta[c] for c, _ in ranked[:top_k]]


print("=" * 68)
print("RRF 候选集策略对比（43 条，文件级命中，无 rerank）")
print("=" * 68)

for name, fn in (("A: BM25 全库参与 RRF（当前）", rrf_full_corpus),
                 ("B: 双路各 Top-K 后融合", rrf_topk_only)):
    ranks = []
    for item in items:
        metas = fn(item["question"], top_k=10)
        ranks.append(first_rank(metas, item["source"]))
    m = recall_mrr(ranks)
    print(f"\n{name}:")
    print(f"  R@1 {m['R@1']:.1%} | R@3 {m['R@3']:.1%} | R@5 {m['R@5']:.1%} | R@10 {m['R@10']:.1%} | MRR {m['MRR']:.3f}")

print("\n" + "=" * 68)
print("结论解读：若两者接近 → 提升主要来自 BM25 提供额外候选；")
print("          若 A 明显更好 → BM25 全库 rank 参与 RRF 有额外价值")
