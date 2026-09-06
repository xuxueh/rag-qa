"""
rerank.py - 本地 bge-reranker-v2-m3 重排模块
=============================================
RAG 升级：召回（找得多）→ 重排（排得准）
- 输入：用户问题 + 候选文档列表
- 输出：按相关性分数降序的 top_n 文档
- 模型：BAAI/bge-reranker-v2-m3（本地，ModelScope 下载）
- 优雅降级：模型加载失败时返回原顺序，不影响主流程
"""
import os
import sys
import logging

# Windows 控制台中文乱码防护
sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger("rerank")

MODEL_PATH = r"C:\Users\xujin\ai-learning\data\models\bge-reranker-v2-m3"

# 惰性加载：第一次调用才加载模型（避免每次 import 都加载）
_model = None


def _get_model():
    """加载 reranker 模型（只加载一次）"""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        print("加载 reranker 模型...（首次约 10-30 秒）")
        _model = CrossEncoder(MODEL_PATH)
    return _model


def rerank_with_meta(query: str, documents: list[str], top_n: int = 3) -> tuple[list[str], dict]:
    """对候选文档按与 query 的相关性重排，返回 (top_n 文档, 元数据)。

    元数据含降级状态，供调用方记录（response metadata）：
    - {"rerank_enabled": True,  "fallback": False}：正常精排
    - {"rerank_enabled": False, "fallback": True}：模型加载/推理失败，降级返回原顺序
    """
    meta = {"rerank_enabled": True, "fallback": False}
    if not documents:
        return documents, meta

    try:
        model = _get_model()
    except Exception as e:
        logger.warning("reranker 模型加载失败（%s），降级返回原始顺序", e)
        print(f"⚠️ reranker 模型加载失败（{e}），降级返回原始顺序")
        meta = {"rerank_enabled": False, "fallback": True}
        return documents[:top_n], meta

    try:
        # CrossEncoder：对 (query, 文档) 成对打分
        pairs = [[query, doc] for doc in documents]
        scores = model.predict(pairs)
        # 按分数降序排序，取 top_n
        ordered = [d for _, d in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
        return ordered[:top_n], meta
    except Exception as e:
        logger.warning("rerank 推理失败（%s），降级返回原始顺序", e)
        print(f"⚠️ rerank 推理失败（{e}），降级返回原始顺序")
        meta = {"rerank_enabled": False, "fallback": True}
        return documents[:top_n], meta


def rerank(query: str, documents: list[str], top_n: int = 3) -> list[str]:
    """兼容版：只返回文档列表（旧调用方/评测脚本使用）"""
    docs, _ = rerank_with_meta(query, documents, top_n)
    return docs


if __name__ == "__main__":
    # 自测：直接运行本文件
    docs = [
        "第一条：单笔金额超过5000元的，需总经理审批。",
        "第一条：公司实行标准工作制，上班时间为上午9:00至下午18:00。",
        "第八条：报销款在审批通过后5个工作日内打款至员工工资卡。",
    ]
    result = rerank("报销金额超过5000元需要谁审批？", docs, top_n=2)
    print("重排结果：")
    for d in result:
        print(f"  - {d}")

# ── 集成到 rag_qa.py 的 ask() 建议 ──────────────────────────
# from rerank import rerank
#
# def ask(question, db, llm, k=3):
#     # ① 召回放宽到 top-10（先找得多）
#     docs = db.similarity_search(question, k=10)
#     texts = [d.page_content for d in docs]
#     # ② rerank 精排，取 top-3（再排得准）
#     ranked = rerank(question, texts, top_n=k)
#     # ③ 用 ranked 组装 context（原逻辑的 parts 循环换成遍历 ranked）
#     # 注意：溯源 metadata 需要 docs 与 ranked 对应，可按 rerank 返回顺序
#     # 重新组织 (文本, 来源) 对，或先用文本匹配找回 metadata
