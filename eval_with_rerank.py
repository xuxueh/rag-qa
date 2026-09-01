"""带 rerank 的评测：Top-1/3/5 命中率（对比 eval_baseline.py）
流程：similarity_search(k=10) 召回 → rerank 精排 → 检查命中
用法：python eval_with_rerank.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_qa as rq
from rerank import rerank

# 1. 构建知识库
print("构建知识库...（加载 GTE 模型，约 1-2 分钟）")
db, _ = rq.build_knowledge_base(rq.DOC_DIR, rq.EMBEDDING_MODEL_PATH)

# 2. 加载测试集
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json"), encoding="utf-8") as f:
    eval_set = json.load(f)


def hit_at_k(docs, target_file, k):
    for d in docs[:k]:
        src = d.metadata.get("source", "")
        fn = src.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        if target_file in fn:
            return True
    return False


# 3. 跑评测（召回 top-10 → rerank 取 top-5 → 检查命中）
hits = {"top1": 0, "top3": 0, "top5": 0}
RECALL_K = 10  # 召回数量
RERANK_K = 5   # 精排后保留数量

for item in eval_set:
    q = item["question"]
    target = item["source"]

    # ① 召回 top-10（带 metadata）
    docs = db.similarity_search(q, k=RECALL_K)
    texts = [d.page_content for d in docs]

    # ② rerank 精排
    ranked_texts = rerank(q, texts, top_n=RERANK_K)

    # ③ 按文本匹配回原 docs（拿回 metadata 来源）
    ranked_docs = []
    for t in ranked_texts:
        for d in docs:
            if d.page_content == t:
                ranked_docs.append(d)
                break

    # ④ 检查命中
    if hit_at_k(ranked_docs, target, 1):
        hits["top1"] += 1
    if hit_at_k(ranked_docs, target, 3):
        hits["top3"] += 1
    if hit_at_k(ranked_docs, target, 5):
        hits["top5"] += 1

total = len(eval_set)
print("=" * 50)
print(f"评测集：{total} 条 | 流程：召回 top-{RECALL_K} → rerank → top-{RERANK_K}")
print(f"Top-1 命中率：{hits['top1']/total:.1%}（{hits['top1']}/{total}）")
print(f"Top-3 命中率：{hits['top3']/total:.1%}（{hits['top3']}/{total}）")
print(f"Top-5 命中率：{hits['top5']/total:.1%}（{hits['top5']}/{total}）")
print("=" * 50)
print("\n对比基线（纯向量，无 rerank）：")
print("  基线    Top-1 75.0% | Top-3 95.0% | Top-5 100.0%")
print(f"  本结果  Top-1 {hits['top1']/total:.1%} | Top-3 {hits['top3']/total:.1%} | Top-5 {hits['top5']/total:.1%}")
