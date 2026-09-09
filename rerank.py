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


def _score(query: str, documents: list[str]):
    """对候选文本打分，返回分数列表；模型不可用时返回 None（触发降级）"""
    if not documents:
        return []
    try:
        model = _get_model()
    except Exception as e:
        logger.warning("reranker 模型加载失败（%s），降级", e)
        print(f"⚠️ reranker 模型加载失败（{e}），降级返回原始顺序")
        return None
    try:
        pairs = [[query, doc] for doc in documents]
        return model.predict(pairs)
    except Exception as e:
        logger.warning("rerank 推理失败（%s），降级", e)
        print(f"⚠️ rerank 推理失败（{e}），降级返回原始顺序")
        return None


def rerank_with_meta(query: str, documents: list[str], top_n: int = 3) -> tuple[list[str], dict]:
    """对候选文本按与 query 的相关性重排，返回 (top_n 文档, 元数据)。

    元数据含降级状态：
    - {"rerank_enabled": True,  "fallback": False}：正常精排
    - {"rerank_enabled": False, "fallback": True}：模型不可用，降级返回原顺序
    """
    if not documents:
        return documents, {"rerank_enabled": True, "fallback": False}

    scores = _score(query, documents)
    if scores is None:
        return documents[:top_n], {"rerank_enabled": False, "fallback": True}

    ordered = [d for _, d in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
    return ordered[:top_n], {"rerank_enabled": True, "fallback": False}


def rerank_objects(query: str, objects: list, text_fn, top_n: int = 3):
    """对象级重排（评审 ③：id 贯穿 rerank，不丢 chunk_id）。

    - objects: 任意对象列表（如 ChunkMeta）
    - text_fn: obj -> 用于打分的文本（如 lambda m: m.text）
    - 返回 (排序后的对象列表, meta)
    """
    if not objects:
        return objects, {"rerank_enabled": True, "fallback": False}
    texts = [text_fn(o) for o in objects]
    scores = _score(query, texts)
    if scores is None:
        return objects[:top_n], {"rerank_enabled": False, "fallback": True}
    ordered = [o for _, o in sorted(zip(scores, objects), key=lambda x: x[0], reverse=True)]
    return ordered[:top_n], {"rerank_enabled": True, "fallback": False}


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
