"""
query_rewrite.py - 查询改写（Multi-Query）
=========================================
用户问题往往表述模糊。Multi-Query：让 LLM 把问题改写成多个变体，
分别检索后合并结果，提升召回率。

例："报销怎么弄？" → ["报销流程是什么？", "费用报销需要什么手续？", "如何申请报销？"]
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")


def multi_query(question: str, llm, n: int = 3) -> list[str]:
    """把问题改写成 n 个变体 + 原问题，返回变体列表（供分别检索）"""
    prompt = f"""请将下面的问题改写成 {n} 个不同表述的变体，用于提高文档检索的召回率。
要求：
1. 意思与原文完全相同，但换用不同的词、句式表达
2. 每行一个变体
3. 只输出改写的变体，不要编号、不要解释、不要输出其他内容

原问题：{question}

改写结果："""

    try:
        resp = llm.invoke(prompt)
        lines = [l.strip() for l in resp.content.strip().split("\n") if l.strip()]
        variants = lines[:n]
        if not variants:
            return [question]
        return variants + [question]  # 变体 + 原问题
    except Exception as e:
        print(f"⚠️ 查询改写失败（{e}），使用原问题")
        return [question]


# ── 集成说明 ────────────────────────────────────────────
# 检索时：对每个变体分别检索，合并结果（去重）后再 rerank。
# 例：
# variants = multi_query(question, llm)
# all_candidates = []
# for v in variants:
#     all_candidates += retriever.retrieve(v, top_k=5)
# # 去重保序
# seen, merged = set(), []
# for t in all_candidates:
#     if t not in seen:
#         seen.add(t); merged.append(t)
# ranked = rerank(question, merged[:15], top_n=3)
