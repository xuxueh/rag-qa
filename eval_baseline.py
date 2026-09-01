"""RAG 基线评测：Top-1/3/5 检索命中率
用法：python eval_baseline.py
产出：命中率报告（所有升级的"升级前"基线）
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_qa as rq

# 1. 构建知识库
print("构建知识库...（加载 GTE 模型，约 1-2 分钟）")
db, _ = rq.build_knowledge_base(rq.DOC_DIR, rq.EMBEDDING_MODEL_PATH)

# 2. 加载测试集
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json"), encoding="utf-8") as f:
    eval_set = json.load(f)

# 3. 评测函数：判断检索结果里是否命中目标文件
def hit_at_k(docs, target_file, k):
    for d in docs[:k]:
        src = d.metadata.get("source", "")
        fn = src.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        if target_file in fn:
            return True
    return False

# 4. 跑评测
hits = {"top1": 0, "top3": 0, "top5": 0}
misses = []  # Top-5 都没命中的问题（分析用）

for item in eval_set:
    q = item["question"]
    target = item["source"]
    docs = db.similarity_search(q, k=5)
    if hit_at_k(docs, target, 1):
        hits["top1"] += 1
    if hit_at_k(docs, target, 3):
        hits["top3"] += 1
    if hit_at_k(docs, target, 5):
        hits["top5"] += 1
    else:
        misses.append(q)

total = len(eval_set)
print("=" * 50)
print(f"评测集：{total} 条（覆盖 12 份制度）")
print(f"Top-1 命中率：{hits['top1']/total:.1%}（{hits['top1']}/{total}）")
print(f"Top-3 命中率：{hits['top3']/total:.1%}（{hits['top3']}/{total}）")
print(f"Top-5 命中率：{hits['top5']/total:.1%}（{hits['top5']}/{total}）")
print("=" * 50)
if misses:
    print(f"\nTop-5 未命中的问题（{len(misses)} 条）：")
    for m in misses:
        print(f"  - {m}")
else:
    print("\n全部问题 Top-5 命中 ✅")
