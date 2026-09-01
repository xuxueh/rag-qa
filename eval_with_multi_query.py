"""Multi-Query 评测：改写问题 → 分别检索 → 合并 → 检查命中
流程：LLM 改写 3 变体 + 原问题 → 每变体混合检索 top-5 → 合并去重 → 检查 Top-1/3/5
用法：python eval_with_multi_query.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import rag_qa as rq
from hybrid_retriever import build_hybrid
from query_rewrite import multi_query

# 1. 构建混合检索器
print("构建混合检索器...（加载 GTE 模型，约 1-2 分钟）")
retriever = build_hybrid(rq.DOC_DIR, rq.EMBEDDING_MODEL_PATH)

# 2. LLM（用于改写）
llm = rq.ChatOpenAI(
    model="deepseek-chat",
    api_key=rq.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

# 3. text → 来源 映射（text2src 从 retriever 拿）
text2src = retriever.text2src

# 4. 测试集
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json"), encoding="utf-8") as f:
    eval_set = json.load(f)


def hit_at_k(texts, target_file, k):
    for t in texts[:k]:
        if text2src.get(t, "") == target_file:
            return True
    return False


# 5. 评测
hits = {"top1": 0, "top3": 0, "top5": 0}
for i, item in enumerate(eval_set):
    q = item["question"]
    target = item["source"]

    # ① 改写问题为多个变体
    variants = multi_query(q, llm)
    # ② 每个变体检索 top-5，合并去重
    all_candidates = []
    for v in variants:
        all_candidates += retriever.retrieve(v, top_k=5)
    seen, merged = set(), []
    for t in all_candidates:
        if t not in seen:
            seen.add(t)
            merged.append(t)
    top = merged[:5]

    if hit_at_k(top, target, 1):
        hits["top1"] += 1
    if hit_at_k(top, target, 3):
        hits["top3"] += 1
    if hit_at_k(top, target, 5):
        hits["top5"] += 1

    if (i + 1) % 5 == 0:
        print(f"  进度：{i + 1}/{len(eval_set)}")

total = len(eval_set)
print("=" * 50)
print(f"评测集：{total} 条 | 流程：Multi-Query → 混合检索 → 合并")
print(f"Top-1 命中率：{hits['top1']/total:.1%}（{hits['top1']}/{total}）")
print(f"Top-3 命中率：{hits['top3']/total:.1%}（{hits['top3']}/{total}）")
print(f"Top-5 命中率：{hits['top5']/total:.1%}（{hits['top5']}/{total}）")
print("=" * 50)
print("\n完整对比表：")
print("  基线（纯向量）       Top-1 75.0% | Top-3 95.0% | Top-5 100.0%")
print("  混合检索             Top-1 90.0% | Top-3 100.0% | Top-5 100.0%")
print("  Rerank               Top-1 95.0% | Top-3 100.0% | Top-5 100.0%")
print("  混合+Rerank          Top-1 95.0% | Top-3 100.0% | Top-5 100.0%")
print(f"  Multi-Query（本结果） Top-1 {hits['top1']/total:.1%} | Top-3 {hits['top3']/total:.1%} | Top-5 {hits['top5']/total:.1%}")
