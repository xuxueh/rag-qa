"""回答质量评测：生产链路完整问答 → LLM-as-judge 判分
与检索命中率评测的区别：
- eval_*.py: 测"检索环节"（Top-1/3/5 命中）
- 本脚本: 测"最终回答"（用户实际体验的指标）——回答正确率

对 20 条 QA：跑完整问答（混合检索+rerank+生成）→ DeepSeek 当裁判，
对比标准答案判定 正确/部分正确/错误，统计正确率。

用法: python eval_answer_quality.py（约 15-20 分钟：模型加载 + 40 次 LLM 调用）
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import rag_qa as rq
from hybrid_retriever import build_hybrid

# 1. 构建检索器 + LLM
print("构建混合检索器（生产链路）...（约 1-2 分钟）")
retriever = build_hybrid(rq.DOC_DIR, rq.EMBEDDING_MODEL_PATH)

llm = rq.ChatOpenAI(
    model="deepseek-chat",
    api_key=rq.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0,
)

# 2. 测试集
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json"), encoding="utf-8") as f:
    eval_set = json.load(f)


def judge(question, reference, answer):
    """DeepSeek 裁判：判定 AI 回答是否与标准答案一致"""
    prompt = f"""你是问答系统的质量评审。判断 AI 的回答是否正确回答了问题。

问题：{question}
标准答案：{reference}
AI 的回答：{answer}

请判定 AI 回答的等级：
- "正确"：回答包含了标准答案的关键信息，没有错误
- "部分正确"：方向对但遗漏了部分关键信息，或表述不够准确
- "错误"：回答错误或与标准答案不符

只输出一个词：正确 / 部分正确 / 错误"""
    try:
        result = llm.invoke(prompt).content.strip()
        if "正确" in result and "部分" not in result:
            return "正确"
        if "部分正确" in result:
            return "部分正确"
        return "错误"
    except Exception as e:
        print(f"  ⚠️ 判分失败：{e}")
        return "判分失败"


# 3. 跑完整问答 + 判分
stats = {"正确": 0, "部分正确": 0, "错误": 0, "判分失败": 0}
details = []

for i, item in enumerate(eval_set):
    q = item["question"]
    print(f"[{i + 1}/{len(eval_set)}] {q}")
    try:
        answer = rq.ask(q, retriever, llm)
    except Exception as e:
        print(f"  ❌ 问答失败：{e}")
        answer = f"[问答失败] {e}"
    grade = judge(q, item["answer"], answer)
    stats[grade] = stats.get(grade, 0) + 1
    details.append({"question": q, "reference": item["answer"], "answer": answer, "grade": grade})
    print(f"  → {grade}")

total = len(eval_set)
print("\n" + "=" * 50)
print(f"回答质量评测：{total} 条（生产链路完整问答）")
print(f"✅ 正确：{stats['正确']}/{total}（{stats['正确']/total:.0%}）")
print(f"🟡 部分正确：{stats['部分正确']}/{total}（{stats['部分正确']/total:.0%}）")
print(f"❌ 错误：{stats['错误']}/{total}（{stats['错误']/total:.0%}）")
correct_rate = (stats['正确'] + stats['部分正确']) / total
print(f"🎯 可接受率（正确+部分）：{correct_rate:.0%}")
print("=" * 50)

# 保存明细
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_quality_result.json"), "w", encoding="utf-8") as f:
    json.dump({"stats": stats, "details": details}, f, ensure_ascii=False, indent=2)
print("\n明细已保存: eval_quality_result.json")
