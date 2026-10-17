"""精排模块测试：排序正确性、分数与输出同序、模型不可用时的降级"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rerank


def test_scores_align_with_returned_order(monkeypatch):
    monkeypatch.setattr(rerank, "rerank_enabled", lambda: True)
    monkeypatch.setattr(rerank, "score_pairs", lambda q, docs: [0.1, 0.9, 0.5])
    ordered, meta = rerank.rerank_objects("问题", ["a", "b", "c"], lambda o: o, top_n=3)
    assert ordered == ["b", "c", "a"]
    assert meta["scores"] == [0.9, 0.5, 0.1]          # 与 ordered 同序，可直接 zip
    assert meta["fallback"] is False


def test_top_n_truncates_after_sorting(monkeypatch):
    monkeypatch.setattr(rerank, "rerank_enabled", lambda: True)
    monkeypatch.setattr(rerank, "score_pairs", lambda q, docs: [0.1, 0.9, 0.5])
    ordered, meta = rerank.rerank_objects("问题", ["a", "b", "c"], lambda o: o, top_n=2)
    assert ordered == ["b", "c"]
    assert len(meta["scores"]) == 2


def test_falls_back_to_original_order_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(rerank, "rerank_enabled", lambda: True)
    monkeypatch.setattr(rerank, "score_pairs", lambda q, docs: None)
    ordered, meta = rerank.rerank_objects("问题", ["a", "b", "c"], lambda o: o, top_n=2)
    assert ordered == ["a", "b"]                       # 保持召回顺序
    assert meta["fallback"] is True
    assert meta["scores"] is None


def test_use_rerank_zero_disables_ranking(monkeypatch):
    monkeypatch.setenv("USE_RERANK", "0")

    def boom(q, docs):                                  # 关掉时不应该调用模型
        raise AssertionError("不应该调用精排模型")

    monkeypatch.setattr(rerank, "score_pairs", boom)
    ordered, meta = rerank.rerank_objects("问题", ["a", "b", "c"], lambda o: o, top_n=2)
    assert ordered == ["a", "b"]
    assert meta["rerank_enabled"] is False and meta["fallback"] is True


def test_empty_input_is_safe():
    ordered, meta = rerank.rerank_objects("问题", [], lambda o: o, top_n=3)
    assert ordered == [] and meta["scores"] == []
    assert rerank.rerank("问题", [], top_n=3) == []


def test_text_wrapper_matches_object_wrapper(monkeypatch):
    monkeypatch.setattr(rerank, "rerank_enabled", lambda: True)
    monkeypatch.setattr(rerank, "score_pairs", lambda q, docs: [0.2, 0.8])
    docs, meta = rerank.rerank_with_meta("问题", ["低", "高"], top_n=1)
    assert docs == ["高"] and meta["scores"] == [0.8]
