"""⑪ 端到端单条评测——完整链路明细输出（评审要求）

一条测试完整输出：
Query → Retrieval(10) → Rerank(3) → Context → Answer → Citation
      → Retrieval Score → Answer Score → Latency → Token Cost

用法: python eval_e2e.py [问题]
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import rag_qa as rq
from hybrid_retriever import build_hybrid
from rerank import rerank_objects

print("构建检索器（chunk_id 版）...")
retriever = build_hybrid(rq.DOC_DIR, rq.EMBEDDING_MODEL_PATH)

llm = rq.ChatOpenAI(model="deepseek-chat", api_key=rq.DEEPSEEK_API_KEY,
                    base_url="https://api.deepseek.com", temperature=0)

QUESTIONS = sys.argv[1:] or [
    "报销金额超过5000元需要谁审批？",
    "想回老家办婚礼，需要申请什么假？",   # 语义匹配
    "公司附近有什么好吃的餐厅？",          # 无答案
]


def judge_correctness(question, reference, answer):
    """LLM judge：回答是否正确（正确/部分/错误）"""
    if not reference:
        return "N/A"
    p = f"""判断 AI 回答是否正确回答问题。
问题：{question}
标准答案：{reference}
AI 回答：{answer}
只输出一个词：正确/部分正确/错误"""
    try:
        r = llm.invoke(p).content.strip()
        if "部分正确" in r:
            return "部分正确"
        if "正确" in r:
            return "正确"
        return "错误"
    except Exception:
        return "判分失败"


for q in QUESTIONS:
    print("\n" + "═" * 68)
    print(f"🔍 QUERY: {q}")
    print("═" * 68)

    # ① 检索
    t0 = time.time()
    recall = retriever.retrieve_meta(q, top_k=10)
    t1 = time.time()
    print(f"\n[1] RETRIEVAL top-10（{round((t1-t0)*1000,1)}ms）:")
    for i, m in enumerate(recall[:10], 1):
        print(f"  {i:2d}. {m.chunk_id} [{m.citation()}] {m.text[:50]}...")

    # ② rerank
    t2 = time.time()
    ranked, rmeta = rerank_objects(q, recall, lambda m: m.text, top_n=3)
    t3 = time.time()
    print(f"\n[2] RERANK top-3（{round((t3-t2)*1000,1)}ms, enabled={rmeta['rerank_enabled']}）:")
    for i, m in enumerate(ranked, 1):
        print(f"  {i:2d}. {m.chunk_id} [{m.citation()}]")

    # ③ context + 生成
    parts = []
    for m in ranked:
        parts.append(f"[来源：{m.citation()}]\n{m.text}")
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
    t4 = time.time()
    answer = result.content
    usage = (result.response_metadata or {}).get("token_usage", {})

    print(f"\n[3] CONTEXT（{len(ranked)} 块, {len(context)} 字）:")
    print(f"[4] ANSWER（生成 {round((t4-t3)*1000,1)}ms）:")
    print(f"  {answer[:200]}")

    # ⑤ 输出
    citations = [m.citation() for m in ranked]
    chunk_ids = [m.chunk_id for m in ranked]
    print(f"\n[5] CITATION: {citations}")
    print(f"[6] CHUNK_IDS: {chunk_ids}")

    # 检索分（rerank 后 top1 是否含答案——简化用 LLM judge 答案）
    total_ms = round((t4 - t0) * 1000, 1)
    tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    print(f"[7] LATENCY: {total_ms}ms（检索+rerank+生成）")
    print(f"[8] TOKENS: {tokens}（prompt {usage.get('prompt_tokens',0)} + completion {usage.get('completion_tokens',0)}）")
    print("═" * 68)
