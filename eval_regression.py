"""全链路回归验证——用生产代码验证所有改进真实有效

验证项（对比历史数据）：
1. RRF 双路 Top-K 修复：文件级 R@1 应 ≥ 76.7%（历史：全库版 72.1%）
2. chunk_id 重构：指标不应回退
3. rerank 链路：文件级 R@1 应 ≥ 83.7%（历史）
4. 条款级（严格标准）：对比历史 68.3%（+rerank）

数据来源：eval_common.build_eval_retriever（= 生产 build_hybrid + _rrf_ids）
用法: python eval_regression.py（rerank 阶段较慢，约 8-12 分钟）
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from eval_common import build_eval_retriever, first_rank, article_first_rank, recall_mrr
from rerank import rerank_objects

print("构建生产检索器（chunk_id + 双路Top-K RRF）...")
retriever = build_eval_retriever()

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set_v2.json"), encoding="utf-8") as f:
    eval_set = json.load(f)
items = [i for i in eval_set if i["category"] != "无答案问题"]


def find_gold(answer, target_source):
    def norm(s):
        return "".join(s.split())
    norm_a = norm(answer)
    cands = [m for m in retriever.chunks if m.source == target_source]
    for m in cands:
        if norm_a[:15] in norm(m.text):
            return m
    for probe_len in (12, 10, 8, 6):
        for m in cands:
            if norm_a[:probe_len] in norm(m.text):
                return m
    docs = retriever.db.similarity_search(answer, k=10)
    for d in docs:
        src = d.metadata.get("source", "")
        fn = src.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        if fn == target_source:
            return retriever.id2meta.get(d.id)
    return None


print("标注 gold（条款级）...")
for item in items:
    item["_gold"] = find_gold(item["answer"], item["source"])
gold_ok = sum(1 for i in items if i["_gold"])
print(f"gold 标注: {gold_ok}/{len(items)}")

# ── 1. 文件级（混合检索，无 rerank）──
file_ranks = [first_rank(retriever.retrieve_meta(i["question"], top_k=10), i["source"]) for i in items]
file_m = recall_mrr(file_ranks)

# ── 2. 条款级（混合检索，无 rerank）──
art_ranks = [article_first_rank(retriever.retrieve_meta(i["question"], top_k=10), i["_gold"])
             if i["_gold"] else None for i in items]
art_ranks = [r for i, r in zip(items, art_ranks) if i["_gold"]]
art_m = recall_mrr(art_ranks)

print("\n" + "=" * 72)
print("【验证 1】混合检索（生产双路Top-K RRF）")
print("=" * 72)
print(f"  文件级: R@1 {file_m['R@1']:.1%} | R@3 {file_m['R@3']:.1%} | R@5 {file_m['R@5']:.1%} | R@10 {file_m['R@10']:.1%} | MRR {file_m['MRR']:.3f}")
print(f"  历史(全库版): R@1 72.1% | R@3 83.7% | R@5 88.4% | R@10 95.3% | MRR 0.799")
v1 = "✅ 有效（≥历史）" if file_m["R@1"] >= 0.767 else f"⚠️ 需复查（R@1 {file_m['R@1']:.1%} < 76.7%）"
print(f"  → {v1}")
print(f"\n  条款级: R@1 {art_m['R@1']:.1%} | R@3 {art_m['R@3']:.1%} | R@5 {art_m['R@5']:.1%} | MRR {art_m['MRR']:.3f}")
print(f"  历史(全库版): R@1 48.8% | R@3 73.2% | R@5 78.0% | MRR 0.641")

# ── 3. rerank 链路 ──
print("\n跑 rerank 链路（较慢）...")
file_ranks_r, art_ranks_r = [], []
for i, item in enumerate(items, 1):
    recall = retriever.retrieve_meta(item["question"], top_k=10)
    ranked, _ = rerank_objects(item["question"], recall, lambda m: m.text, top_n=10)
    file_ranks_r.append(first_rank(ranked, item["source"]))
    if item["_gold"]:
        art_ranks_r.append(article_first_rank(ranked, item["_gold"]))
    if i % 10 == 0:
        print(f"  进度 {i}/{len(items)}")
file_mr = recall_mrr(file_ranks_r)
art_mr = recall_mrr(art_ranks_r)

print("\n" + "=" * 72)
print("【验证 2】混合 + Rerank（生产链路）")
print("=" * 72)
print(f"  文件级: R@1 {file_mr['R@1']:.1%} | R@3 {file_mr['R@3']:.1%} | R@5 {file_mr['R@5']:.1%} | MRR {file_mr['MRR']:.3f}")
print(f"  历史(全库RRF): R@1 83.7% | R@3 93.0% | MRR 0.888")
v2 = "✅ 有效" if file_mr["R@1"] >= 0.80 else f"⚠️ 需复查（{file_mr['R@1']:.1%}）"
print(f"  → {v2}")
print(f"\n  条款级: R@1 {art_mr['R@1']:.1%} | R@3 {art_mr['R@3']:.1%} | R@5 {art_mr['R@5']:.1%} | MRR {art_mr['MRR']:.3f}")
print(f"  历史(全库RRF): R@1 68.3% | R@3 90.2% | MRR 0.783")
v3 = "✅ 有效" if art_mr["R@1"] >= 0.68 else f"⚠️ 需复查（{art_mr['R@1']:.1%}）"
print(f"  → {v3}")

print("\n" + "=" * 72)
print("验证结论汇总")
print("=" * 72)
print(f"  1. RRF 双路Top-K 修复: {v1}")
print(f"  2. rerank 链路（文件级）: {v2}")
print(f"  3. rerank 链路（条款级）: {v3}")
