"""
eval_common.py - 评测公共模块（评审 ⑤：收敛重复的构建逻辑）
=============================================================
背景：此前 eval_v2 / eval_failure_analysis / eval_chunk_experiment 等脚本
各自重复"加载→清洗→切块→Embedding→Chroma→BM25"逻辑，与生产代码分叉。

统一入口：
- build_eval_retriever(): 构建评测用检索器（与生产同一套 build_hybrid）
- first_rank(): 命中判断（文件级）

新评测脚本请用本模块，避免再次分叉。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import rag_qa as rq
from hybrid_retriever import build_hybrid


def build_eval_retriever(doc_dir: str | None = None, model_path: str | None = None):
    """构建评测检索器（复用生产构建，避免逻辑分叉）"""
    return build_hybrid(doc_dir or rq.DOC_DIR, model_path or rq.EMBEDDING_MODEL_PATH)


def first_rank(metas, target_source: str, k: int = 10):
    """文件级命中排名（1-based）；未命中返回 None"""
    for i, m in enumerate(metas[:k]):
        if m.source == target_source:
            return i + 1
    return None


def article_first_rank(metas, gold_meta, k: int = 10):
    """条款级命中排名：source 匹配且条款有交集"""
    for i, m in enumerate(metas[:k]):
        if m.source == gold_meta.source and set(m.articles) & set(gold_meta.articles):
            return i + 1
    return None


def recall_mrr(ranks: list, ks=(1, 3, 5, 10)):
    """由排名列表算 Recall@K + MRR"""
    n = len(ranks)
    if n == 0:
        return {}
    out = {f"R@{k}": sum(1 for r in ranks if r and r <= k) / n for k in ks}
    out["MRR"] = sum(1.0 / r for r in ranks if r) / n
    return out
