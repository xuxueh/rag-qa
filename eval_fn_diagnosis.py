"""6 条 FN 诊断（⑨ 跟进）
3 条真漏答：看检索是否召回正确条款（区分"检索问题" vs "prompt/模型问题"）
3 条判定误伤：验证改进后的拒答判定（只有结论=不知道才算法拒答）
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from eval_common import build_eval_retriever
from rerank import rerank_objects

retriever = build_eval_retriever()

# 3 条真漏答（资料里有答案，模型却说未找到）
CASES = [
    ("员工多久有一次免费体检？", "05-薪酬福利制度.txt", "第六条：入职满一年员工每年一次免费健康体检"),
    ("公司有什么福利？", "05-薪酬福利制度.txt", "第六条：公司福利包括体检/生日礼金/团建/节日礼品"),
    ("调休会影响全勤奖吗？", "04-加班与调休制度.txt", "第六条：调休期间视为正常出勤，不影响全勤奖"),
]

print("=" * 70)
print("真漏答诊断：正确答案是否被召回？")
print("=" * 70)

for q, target_file, gold_text in CASES:
    print(f"\n❓ {q}")
    print(f"   资料中的答案: {gold_text}")

    recall = retriever.retrieve_meta(q, top_k=10)
    ranked, _ = rerank_objects(q, recall, lambda m: m.text, top_n=3)

    # 正确条款是否在召回池 / rerank 结果里
    gold_kw = gold_text.split("：")[1][:10]  # 答案关键片段
    in_pool = any(target_file == m.source and gold_kw.replace(" ", "")[:8] in m.text.replace(" ", "")
                  for m in recall)
    in_ranked = any(target_file == m.source and gold_kw.replace(" ", "")[:8] in m.text.replace(" ", "")
                    for m in ranked)

    print(f"   召回池 top10 含正确条款: {'✅ 是' if in_pool else '❌ 否'}")
    print(f"   rerank top3 含正确条款: {'✅ 是' if in_ranked else '❌ 否'}")
    print(f"   rerank top3 实际来源: {[m.citation() for m in ranked]}")

    # 结论
    if not in_pool:
        print("   🔍 诊断: 检索未召回（检索问题——需改切块/融合/召回池）")
    elif not in_ranked:
        print("   🔍 诊断: 召回了但 rerank 未保留（rerank 问题——top_n 或模型）")
    else:
        print("   🔍 诊断: 正确条款在 context 里，模型却答'未找到' → prompt/模型保守问题")
