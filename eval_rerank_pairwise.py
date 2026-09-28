"""第四组验证：rerank 链路的旧 RRF vs 新 RRF（同构建内，消除 HNSW 噪声）
目的：确认回归验证中"文件级 R@1 83.7% → 93.0%"是真实改善还是噪声
用法: python eval_rerank_pairwise.py（rerank 43×2，约 8-10 分钟）
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import jieba
jieba.setLogLevel(20)

from eval_common import build_eval_retriever, first_rank, recall_mrr
from rerank import rerank_objects

retriever = build_eval_retriever()
db = retriever.db
chunks = retriever.chunks
bm25 = retriever.bm25

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set_v2.json"), encoding="utf-8") as f:
    eval_set = json.load(f)
items = [i for i in eval_set if i["category"] != "无答案问题"]


def old_rrf_full(query, top_k=10, rrf_k=60):
    """旧实现：BM25 全库参与"""
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


print("=" * 72)
print("rerank 链路：旧 RRF vs 新 RRF（同构建内）")
print("=" * 72)

results = {}
for name, recall_fn in (("旧RRF(全库)", old_rrf_full), ("新RRF(双路Top-K)", retriever.retrieve_meta)):
    print(f"\n跑 {name} + rerank...")
    ranks = []
    for i, item in enumerate(items, 1):
        recall = recall_fn(item["question"], top_k=10)
        ranked, _ = rerank_objects(item["question"], recall, lambda m: m.text, top_n=10)
        ranks.append(first_rank(ranked, item["source"]))
        if i % 15 == 0:
            print(f"  进度 {i}/{len(items)}")
    m = recall_mrr(ranks)
    results[name] = m

print("\n" + "=" * 72)
print("对比结果（同构建内，消除 HNSW 噪声）")
print("=" * 72)
for name, m in results.items():
    print(f"  {name:<20} R@1 {m['R@1']:.1%} | R@3 {m['R@3']:.1%} | R@5 {m['R@5']:.1%} | MRR {m['MRR']:.3f}")

old_m = results["旧RRF(全库)"]
new_m = results["新RRF(双路Top-K)"]
d_r1 = new_m["R@1"] - old_m["R@1"]
d_mrr = new_m["MRR"] - old_m["MRR"]
print(f"\n  差异: R@1 {d_r1:+.1%} | MRR {d_mrr:+.3f}")
print(f"  结论: {'✅ 新RRF在rerank链路真实更优' if d_r1 > 0.02 or d_mrr > 0.02 else '⚠️ 差异在噪声范围内（需更多样本）'}")
