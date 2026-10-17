"""
rerank.py — 本地 bge-reranker-v2-m3 精排模块
=============================================
RAG 的两段式：召回负责「找得全」（宁多勿漏），重排负责「排得准」（把真正相关的顶到前面）。

设计要点：
- 模型路径从 config 读（不写死），可用环境变量 RERANK_MODEL 覆盖
- 惰性加载：第一次调用才 import torch / 加载模型
- 优雅降级：模型缺失或推理失败 → 返回原顺序 + fallback 标记，主流程不中断
- 加载失败只记一次：避免每次问答都重试加载（那是几十秒的空转）
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import config

logger = logging.getLogger(__name__)

_model = None
_load_failed = False


class RerankUnavailable(RuntimeError):
    """精排模型不可用（缺模型/缺依赖）。调用方负责决定是否降级。"""


def get_model():
    """加载 reranker（进程内只成功加载一次）"""
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        raise RerankUnavailable("reranker 模型此前加载失败，本次直接降级（不会再重试）")
    model_path = config.resolve_rerank_model()
    try:
        from sentence_transformers import CrossEncoder

        logger.info("加载 reranker 模型：%s", model_path)
        _model = CrossEncoder(model_path)
        logger.info("reranker 就绪")
        return _model
    except Exception as exc:  # 缺模型/缺依赖/显存不足 都走这里
        _load_failed = True
        raise RerankUnavailable(
            f"reranker 模型加载失败（{exc}）。修复：确认 RERANK_MODEL={model_path} 存在，"
            f"或 pip install -U sentence-transformers；也可设 USE_RERANK=0 直接用混合检索排序"
        ) from exc


def rerank_enabled() -> bool:
    """配置开关：USE_RERANK=0 可整体关掉精排（用于对比实验）"""
    return os.environ.get("USE_RERANK", "1").strip().lower() not in ("0", "false", "no", "off")


def score_pairs(query: str, documents: list[str]) -> list[float] | None:
    """给 (query, doc) 打相关性分。模型不可用返回 None（调用方降级）。"""
    if not documents:
        return []
    try:
        model = get_model()
        return [float(s) for s in model.predict([[query, d] for d in documents])]
    except RerankUnavailable as exc:
        logger.warning("精排不可用，降级：%s", exc)
        return None
    except Exception as exc:
        logger.warning("精排推理失败，降级：%s", exc)
        return None


def rerank_with_meta(query: str, documents: list[str], top_n: int = 3) -> tuple[list[str], dict]:
    """文本精排：返回 (top_n 文本, meta)。

    meta: {"rerank_enabled": bool, "fallback": bool, "scores": list[float] | None}
    """
    if not documents:
        return documents, {"rerank_enabled": rerank_enabled(), "fallback": False, "scores": []}
    if not rerank_enabled():
        return documents[:top_n], {"rerank_enabled": False, "fallback": True, "scores": None}

    scores = score_pairs(query, documents)
    if scores is None:
        return documents[:top_n], {"rerank_enabled": False, "fallback": True, "scores": None}
    pairs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    ordered = [d for _, d in pairs][:top_n]
    ordered_scores = [float(s) for s, _ in pairs][:top_n]
    return ordered, {"rerank_enabled": True, "fallback": False, "scores": ordered_scores}


def rerank_objects(query: str, objects: list, text_fn, top_n: int = 3):
    """对象级精排：chunk_id / citation 等元数据全程不丢（不然溯源会错位）

    - objects: 任意对象列表（如 ChunkMeta）
    - text_fn: obj -> 打分用文本（如 lambda m: m.text）
    - 返回 (重排后的对象列表, meta)；meta["scores"] 与返回对象**同序**（可直接 zip 展示）
    """
    if not objects:
        return objects, {"rerank_enabled": rerank_enabled(), "fallback": False, "scores": []}
    if not rerank_enabled():
        return objects[:top_n], {"rerank_enabled": False, "fallback": True, "scores": None}

    scores = score_pairs(query, [text_fn(o) for o in objects])
    if scores is None:
        return objects[:top_n], {"rerank_enabled": False, "fallback": True, "scores": None}
    pairs = sorted(zip(scores, objects), key=lambda x: x[0], reverse=True)
    ordered = [o for _, o in pairs][:top_n]
    ordered_scores = [float(s) for s, _ in pairs][:top_n]
    return ordered, {"rerank_enabled": True, "fallback": False, "scores": ordered_scores}


def rerank(query: str, documents: list[str], top_n: int = 3) -> list[str]:
    """兼容版：只要文本列表"""
    return rerank_with_meta(query, documents, top_n)[0]


if __name__ == "__main__":
    config.setup_logging()
    docs = [
        "第一条：单笔金额超过5000元的，需总经理审批。",
        "第一条：公司实行标准工作制，上班时间为上午9:00至下午18:00。",
        "第八条：报销款在审批通过后5个工作日内打款至员工工资卡。",
    ]
    print("重排结果：")
    for d in rerank("报销金额超过5000元需要谁审批？", docs, top_n=2):
        print(f"  - {d}")
