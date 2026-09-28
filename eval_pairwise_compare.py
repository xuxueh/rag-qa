"""第二组验证：旧 RRF（全库参与）vs 新 RRF（双路 Top-K）逐条对比
比聚合指标更硬的证据——看每条问题的排名变化（改善/持平/变差）
用法: python eval_pairwise_compare.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import jieba
jieba.setLogLevel(20)

from eval_common import build_eval_retriever, first_rank, recall_mrr

retriever = build_eval_retriever()
db = retriever.db
chunks = retriever.chunks
bm25 = retriever.bm25

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set_v2.json"), encoding="utf-8") as f:
    eval_set = json.load(f)
items = [i for i in eval_set if i["category"] != "无答案问题"]


def old_rrf_full(query, top_k=10, rrf_k=60):
    """旧实现：BM25 全库参与 RRF"""
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


def new_rrf_topk(query, top_k=10):
    """新实现：生产代码（双路 Top-K）"""
    return retriever.retrieve_meta(query, top_k=top_k)


print("=" * 76)
print("旧 RRF（全库）vs 新 RRF（双路Top-K）逐条对比")
print("=" * 76)
print(f"{'#':<4}{'问题':<30}{'旧排名':<9}{'新排名':<9}{'变化':<8}")
print("-" * 76)

old_ranks, new_ranks = [], []
improved = degraded = same = 0
for i, item in enumerate(items, 1):
    q = item["question"]
    src = item["source"]
    r_old = first_rank(old_rrf_full(q, top_k=10), src)
    r_new = first_rank(new_rrf_topk(q, top_k=10), src)
    old_ranks.append(r_old)
    new_ranks.append(r_new)

    def fmt(r):
        return str(r) if r else "未命中"

    # 变化判定
    if r_old and r_new:
        delta = r_old - r_new  # 正数=改善（排名变小）
        mark = f"↑{delta}" if delta > 0 else (f"↓{-delta}" if delta < 0 else "—")
        if delta > 0: improved += 1
        elif delta < 0: degraded += 1
        else: same += 1
    elif not r_old and r_new:
        mark = "新命中"
        improved += 1
    elif r_old and not r_new:
        mark = "变未命中"
        degraded += 1
    else:
        mark = "都未中"
        same += 1

    print(f"{i:<4}{q[:28]:<30}{fmt(r_old):<9}{fmt(r_new):<9}{mark:<8}")

print("-" * 76)
m_old = recall_mrr(old_ranks)
m_new = recall_mrr(new_ranks)
print(f"\n汇总:")
print(f"  改善 {improved} 条 | 持平 {same} 条 | 变差 {degraded} 条")
print(f"\n  旧 RRF: R@1 {m_old['R@1']:.1%} | R@3 {m_old['R@3']:.1%} | R@5 {m_old['R@5']:.1%} | MRR {m_old['MRR']:.3f}")
print(f"  新 RRF: R@1 {m_new['R@1']:.1%} | R@3 {m_new['R@3']:.1%} | R@5 {m_new['R@5']:.1%} | MRR {m_new['MRR']:.3f}")
print(f"\n  结论: {'✅ 新实现有效（改善 > 变差）' if improved > degraded else '⚠️ 需复查'}")
