"""混合检索测试：RRF 融合逻辑（用假向量库，不加载任何模型）+ recall_mrr 口径"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_common import recall_mrr
from hybrid_retriever import ChunkMeta, HybridRetriever


class FakeDB:
    """假向量库：按给定顺序返回 chunk_id，模拟「向量检索」这一路"""

    def __init__(self, ids):
        self.ids = ids
        self.queries = []

    def similarity_search(self, query, k=4):
        self.queries.append(query)
        return [SimpleNamespace(metadata={"chunk_id": i}, page_content="") for i in self.ids[:k]]


CHUNKS = [
    ChunkMeta("d01_c01", "第一条：单笔金额超过5000元的，需总经理审批。", "02-费用报销制度.txt", [1]),
    ChunkMeta("d01_c02", "第一条：公司实行标准工作制，上班时间为上午9:00至下午18:00。", "01-考勤管理制度.txt", [1]),
    ChunkMeta("d02_c01", "第八条：报销款在审批通过后5个工作日内打款至员工工资卡。", "02-费用报销制度.txt", [8]),
]


def make_retriever(ids):
    return HybridRetriever(FakeDB(ids), CHUNKS)


def test_rrf_lifts_doc_hit_by_both_routes():
    # 向量把「考勤」排第一，BM25 看重「5000元审批」→ 双路都命中的 d01_c01 应被顶到第一
    r = make_retriever(["d01_c02", "d01_c01", "d02_c01"])
    top = r.retrieve_meta("5000元审批", top_k=3)
    assert top[0].chunk_id == "d01_c01"


def test_dense_only_keeps_vector_order():
    r = make_retriever(["d01_c02", "d01_c01", "d02_c01"])
    assert [m.chunk_id for m in r.dense_meta("5000元审批", top_k=3)] == ["d01_c02", "d01_c01", "d02_c01"]


def test_bm25_route_finds_keyword_match():
    r = make_retriever(["d01_c02", "d02_c01", "d01_c01"])
    assert r.bm25_meta("5000元审批", top_k=1)[0].chunk_id == "d01_c01"


def test_bm25_route_is_empty_when_no_keyword_overlap():
    r = make_retriever(["d01_c02", "d01_c01"])
    assert r.bm25_meta("婚假几天", top_k=3) == []


def test_hybrid_falls_back_to_dense_when_bm25_empty():
    r = make_retriever(["d01_c02", "d01_c01"])
    assert [m.chunk_id for m in r.retrieve_meta("婚假几天", top_k=3)] == ["d01_c02", "d01_c01"]


def test_top_k_is_respected():
    r = make_retriever(["d01_c02", "d01_c01", "d02_c01"])
    assert len(r.retrieve_meta("审批", top_k=2)) == 2


def test_empty_query_returns_nothing():
    r = make_retriever(["d01_c02"])
    assert r.retrieve_meta("   ", top_k=3) == []


def test_empty_knowledge_base_is_rejected():
    with pytest.raises(ValueError):
        HybridRetriever(FakeDB([]), [])


def test_doc_id_falls_back_to_document_id():
    doc = SimpleNamespace(metadata={}, id="d01_c01")
    assert HybridRetriever._doc_id(doc) == "d01_c01"


def test_recall_mrr_math():
    stats = recall_mrr([1, 3, None, 5], ks=(1, 3, 5))
    assert stats["R@1"] == pytest.approx(0.25)
    assert stats["R@3"] == pytest.approx(0.5)
    assert stats["R@5"] == pytest.approx(0.75)
    assert stats["MRR"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 5) / 4)
