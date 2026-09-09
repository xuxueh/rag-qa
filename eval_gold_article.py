"""⑧ 条款级（gold）命中评测——升级标准：文件 + 条款 都命中
背景（GPT 评审 ⑧）：source 文件级命中可能有假阳性
（如问"婚假"但 Top-1 是 03-请假 讲"病假"的 chunk，仍算命中）

升级：gold = (source, articles)——标准答案对应的条款
命中条件：返回 chunk 的 source 匹配 且 条款号有交集

gold 自动标注：答案文本前 16 字在知识库中定位所在 chunk（答案从原文提取，可定位）
用法: python eval_gold_article.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import rag_qa as rq
from hybrid_retriever import build_hybrid
from rerank import rerank_objects

print("构建知识库（chunk_id 版）...（约 1-2 分钟）")
retriever = build_hybrid(rq.DOC_DIR, rq.EMBEDDING_MODEL_PATH)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set_v2.json"), encoding="utf-8") as f:
    eval_set = json.load(f)
items = [i for i in eval_set if i["category"] != "无答案问题"]


def find_gold(answer: str, target_source: str, retriever):
    """自动定位 gold：三级策略定位答案所在 chunk（限目标文件）"""
    # 归一化：去所有空白（答案与原文可能有空格差异）
    def norm(s):
        return "".join(s.split())

    norm_a = norm(answer)
    cands = [m for m in retriever.chunks if m.source == target_source]

    # ① 全答案归一后包含匹配
    for m in cands:
        if norm_a[:15] in norm(m.text):
            return m
    # ② 递减探针
    for probe_len in (12, 10, 8, 6):
        for m in cands:
            if norm_a[:probe_len] in norm(m.text):
                return m
    # ③ 语义兜底：用答案检索（同文件内取最高分 chunk）
    docs = retriever.db.similarity_search(answer, k=10)
    for d in docs:
        src = d.metadata.get("source", "")
        fn = src.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        if fn == target_source:
            return retriever.id2meta.get(d.id)
    return None


def article_hit(metas, gold_meta, k):
    """条款级命中：返回的 chunk 中，source 匹配 且 articles 与 gold 有交集"""
    for m in metas[:k]:
        if m.source == gold_meta.source and set(m.articles) & set(gold_meta.articles):
            return True
    return False


# 标注 gold（可缓存）
print("标注 gold（条款级）...")
gold_not_found = []
for item in items:
    gm = find_gold(item["answer"], item["source"], retriever)
    item["_gold_meta"] = gm
    if gm is None:
        gold_not_found.append(item["question"])

print(f"gold 标注成功: {len(items) - len(gold_not_found)}/{len(items)}")
if gold_not_found:
    print(f"⚠️ gold 未定位（答案片段不在知识库？）: {gold_not_found}")

# 评测（条款级命中）
print("\n" + "=" * 60)
print("条款级命中评测（source + 条款都命中）")
print("=" * 60)

for mode in ("hybrid_only", "hybrid_rerank"):
    r1 = r3 = r5 = 0
    mrr = 0.0
    n = 0
    for item in items:
        gm = item.get("_gold_meta")
        if gm is None:
            continue
        n += 1
        recall = retriever.retrieve_meta(item["question"], top_k=10)
        if mode == "hybrid_rerank":
            ranked, _ = rerank_objects(item["question"], recall, lambda m: m.text, top_n=10)
        else:
            ranked = recall
        # 找条款级命中的 rank
        rank = None
        for i, m in enumerate(ranked[:10]):
            if m.source == gm.source and set(m.articles) & set(gm.articles):
                rank = i + 1
                break
        if rank:
            if rank <= 1: r1 += 1
            if rank <= 3: r3 += 1
            if rank <= 5: r5 += 1
            mrr += 1.0 / rank
    if n:
        label = "混合检索(RRF)" if mode == "hybrid_only" else "混合+Rerank"
        print(f"\n{label}（条款级，n={n}）:")
        print(f"  R@1 {r1/n:.1%} | R@3 {r3/n:.1%} | R@5 {r5/n:.1%} | MRR {mrr/n:.3f}")

print("\n" + "=" * 60)
print("对比（文件级 vs 条款级，可看出假阳性程度）")
print("  文件级 R@1 约 83.7%（之前评测）；条款级见上方（应更低，因更严格）")
