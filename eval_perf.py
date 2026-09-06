"""性能监控：统计生产链路每环节耗时 + Token 消耗
指标：
- 检索耗时（混合检索）
- rerank 耗时
- 生成耗时（DeepSeek）
- Token 消耗（prompt / completion）

用法: python eval_perf.py（先加载模型再测 5 条，排除首次加载）
"""
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import rag_qa as rq
from hybrid_retriever import build_hybrid
from rerank import rerank

# 1. 构建
print("构建混合检索器 + 预加载 rerank 模型...（约 1-2 分钟）")
retriever = build_hybrid(rq.DOC_DIR, rq.EMBEDDING_MODEL_PATH)
rerank("预热", ["预热"], top_n=1)  # 预加载 rerank，不计入统计

llm = rq.ChatOpenAI(
    model="deepseek-chat",
    api_key=rq.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0,
)

QUESTIONS = [
    "报销金额超过5000元需要谁审批？",
    "婚假有几天？",
    "迟到超过30分钟怎么处理？",
    "绩效考核S档对应的绩效系数是多少？",
    "员工辞职需要提前多少天？",
]


def ask_with_metrics(q, retriever, llm):
    """完整问答并记录各环节耗时/token"""
    t0 = time.time()
    recall = retriever.retrieve(q, top_k=5)  # top-5 召回（优化后：rerank 5 候选）
    t1 = time.time()

    ranked = rerank(q, recall, top_n=3)
    t2 = time.time()

    parts = []
    for t in ranked:
        parts.append(f"[来源：{retriever.get_source(t)}]\n{t}")
    context = "\n\n".join(parts)

    prompt = f"""根据以下资料回答问题。如果资料里没有答案，就说不知道，不要瞎编。

回答要求：
1. 先说明依据：指出答案来自资料的哪份文件、第几条
2. 再给出结论
3. 引用内容不得超出资料范围

资料：
{context}

问题：{q}

回答："""
    result = llm.invoke(prompt)
    t3 = time.time()

    usage = (result.response_metadata or {}).get("token_usage", {})
    return {
        "question": q,
        "retrieve_ms": round((t1 - t0) * 1000, 1),
        "rerank_ms": round((t2 - t1) * 1000, 1),
        "generate_ms": round((t3 - t2) * 1000, 1),
        "total_ms": round((t3 - t0) * 1000, 1),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "answer_head": result.content[:40],
    }


# 2. 跑评测
results = []
for q in QUESTIONS:
    m = ask_with_metrics(q, retriever, llm)
    results.append(m)
    print(f"✓ {q[:20]}... 总耗时 {m['total_ms']}ms（检索 {m['retrieve_ms']}ms + rerank {m['rerank_ms']}ms + 生成 {m['generate_ms']}ms）")

# 3. 统计
n = len(results)
avg = lambda k: sum(r[k] for r in results) / n
print("\n" + "=" * 55)
print(f"性能统计（{n} 条，模型已预热）")
print(f"平均总耗时：{avg('total_ms')}ms（约 {avg('total_ms')/1000:.1f} 秒）")
print(f"  其中 检索：{avg('retrieve_ms')}ms")
print(f"        rerank：{avg('rerank_ms')}ms")
print(f"        生成：{avg('generate_ms')}ms（占大头，LLM 网络延迟）")
print(f"平均 prompt tokens：{avg('prompt_tokens'):.0f}")
print(f"平均 completion tokens：{avg('completion_tokens'):.0f}")
print(f"单次问答平均 token 消耗：{avg('prompt_tokens') + avg('completion_tokens'):.0f}")
print("=" * 55)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf_report.json"), "w", encoding="utf-8") as f:
    json.dump({"results": results}, f, ensure_ascii=False, indent=2)
print("报告已保存: perf_report.json")
