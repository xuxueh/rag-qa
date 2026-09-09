"""⑨ 无答案检测评估——TP/FP/FN/TN 矩阵 + 拒答指标

有答案问题：应该回答（回答了=好，拒答=FN 漏答）
无答案问题：应该拒答（拒答=TN ✓，硬答编造=FP ✗）

矩阵：
                    实际有答案    实际无答案
系统回答(不拒答)      TP            FP (编造!)
系统拒答             FN (漏答!)     TN

指标：
- 正确回答率 = TP / 有答案总数（有答案时给出正确回答）
- 正确拒答率 = TN / 无答案总数（无答案时老实说不知道）
用法: python eval_abstention.py（50 条完整问答，约 15-20 分钟）
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import rag_qa as rq
from hybrid_retriever import build_hybrid

print("构建检索器...")
retriever = build_hybrid(rq.DOC_DIR, rq.EMBEDDING_MODEL_PATH)
llm = rq.ChatOpenAI(model="deepseek-chat", api_key=rq.DEEPSEEK_API_KEY,
                    base_url="https://api.deepseek.com", temperature=0)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set_v2.json"), encoding="utf-8") as f:
    eval_set = json.load(f)

REFUSE_MARKERS = ("不知道", "没有相关信息", "无法回答", "资料中没有", "未提及")


def is_refusal(answer: str) -> bool:
    """判断回答是否为拒答（说不知道）"""
    return any(m in answer for m in REFUSE_MARKERS)


# 跑全部 50 条（含 7 无答案）
matrix = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
details = []
for i, item in enumerate(eval_set):
    q = item["question"]
    has_answer = item["category"] != "无答案问题"
    try:
        answer = rq.ask(q, retriever, llm)
    except Exception as e:
        answer = f"[出错] {e}"
    refused = is_refusal(answer)

    if has_answer and not refused:
        matrix["tp"] += 1
    elif has_answer and refused:
        matrix["fn"] += 1
    elif not has_answer and not refused:
        matrix["fp"] += 1
    else:
        matrix["tn"] += 1

    details.append({"q": q, "has_answer": has_answer, "refused": refused, "answer_head": answer[:40]})
    print(f"[{i+1}/50] {'有答案' if has_answer else '无答案'} | {'拒答' if refused else '回答'} | {q[:25]}")

n_answerable = sum(1 for i in eval_set if i["category"] != "无答案问题")
n_unanswerable = len(eval_set) - n_answerable

print("\n" + "=" * 56)
print("无答案检测矩阵")
print("=" * 56)
print(f"                实际有答案({n_answerable})   实际无答案({n_unanswerable})")
print(f"系统回答(不拒答)   TP={matrix['tp']}            FP={matrix['fp']}")
print(f"系统拒答          FN={matrix['fn']}            TN={matrix['tn']}")
print("=" * 56)
tp, fp, fn, tn = matrix["tp"], matrix["fp"], matrix["fn"], matrix["tn"]
print(f"正确回答率 TP/(TP+FN) = {tp}/{tp+fn} = {tp/(tp+fn):.1%}" if tp+fn else "N/A")
print(f"正确拒答率 TN/(TN+FP) = {tn}/{tn+fp} = {tn/(tn+fp):.1%}" if tn+fp else "N/A")
print(f"拒答率(无答案时) = {tn}/({tn}+{fp}) = {tn/(tn+fp):.1%}" if tn+fp else "N/A")
print(f"FP 编造数 = {fp}（无答案却硬答——最危险）")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "abstention_result.json"), "w", encoding="utf-8") as f:
    json.dump({"matrix": matrix, "details": details}, f, ensure_ascii=False, indent=2)
print("\n明细已保存: abstention_result.json")
