"""API 层测试：健康检查、文档管理、问答链路（检索/大模型用假件，不联网不加载模型）"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import rag_api
import rag_qa
from hybrid_retriever import ChunkMeta

FAKE_METAS = [ChunkMeta("d01_c01", "第五条：婚假为15天，需在领证后一年内使用。", "03-请假管理制度.txt", [5])]


@pytest.fixture
def kb(tmp_path, monkeypatch):
    """临时知识库目录 + 干净的全局状态"""
    knowledge = tmp_path / "知识库"
    knowledge.mkdir()
    (knowledge / "a.txt").write_text("第一条：婚假15天。", encoding="utf-8")
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(rag_api, "_retriever", None)
    monkeypatch.setattr(rag_api, "_llm", None)
    return knowledge


@pytest.fixture
def client(kb):
    return TestClient(rag_api.app)


def _fake_retrieve_and_rerank(retriever, question, recall_k=None, top_n=None):
    return FAKE_METAS, FAKE_METAS, 3.0


class FakeLLM:
    def invoke(self, prompt):
        assert "第五条" in prompt           # prompt 里必须带引用来源
        return SimpleNamespace(content="根据 03-请假管理制度.txt 第五条：婚假为15天。")


def test_health_reports_state(client):
    body = client.get("/health").json()
    assert set(["status", "retriever_ready", "documents", "config_issues", "config"]) <= set(body)
    assert body["documents"] == 1
    assert "api_key_set" in body["config"]["llm"]


def test_documents_listing(client):
    body = client.get("/documents").json()
    assert body["count"] == 1
    assert body["documents"] == ["a.txt"]
    assert body["items"][0]["size"] > 0


def test_root_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "html" in resp.text.lower()


def test_ask_returns_answer_with_sources(client, monkeypatch):
    monkeypatch.setattr(rag_api, "get_retriever", lambda force=False: SimpleNamespace(chunks=FAKE_METAS))
    monkeypatch.setattr(rag_api, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(rag_qa, "retrieve_and_rerank", _fake_retrieve_and_rerank)

    resp = client.post("/ask", json={"question": "婚假几天？"})
    assert resp.status_code == 200
    body = resp.json()
    assert "15天" in body["answer"]
    assert body["sources"][0]["source"] == "03-请假管理制度.txt · 第五条"
    assert body["timings"]["retrieval_ms"] == 3.0


def test_ask_requires_question_text(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_503_when_api_key_missing(client, monkeypatch):
    monkeypatch.setattr(rag_api, "get_retriever", lambda force=False: SimpleNamespace(chunks=FAKE_METAS))
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(rag_api, "_llm", None)
    resp = client.post("/ask", json={"question": "婚假几天？"})
    assert resp.status_code == 503
    assert "DEEPSEEK_API_KEY" in resp.json()["detail"]


def test_ask_503_when_retriever_broken(client, monkeypatch):
    def boom(force=False):
        raise RuntimeError("模型路径不存在")

    monkeypatch.setattr(rag_api, "get_retriever", boom)
    resp = client.post("/ask", json={"question": "婚假几天？"})
    assert resp.status_code == 503
    assert "知识库尚未就绪" in resp.json()["detail"]


def test_upload_rejects_unsupported_type(client):
    resp = client.post("/documents/upload", files={"file": ("bad.exe", b"x", "application/octet-stream")})
    assert resp.status_code == 400
    assert "不支持的文件类型" in resp.json()["detail"]


def test_upload_rejects_empty_file(client):
    resp = client.post("/documents/upload", files={"file": ("e.txt", b"", "text/plain")})
    assert resp.status_code == 400
    assert "空文件" in resp.json()["detail"]


def test_upload_success_rebuilds(client, kb, monkeypatch):
    monkeypatch.setattr(rag_api, "rebuild_retriever",
                        lambda: {"chunks": 12, "docs": 2, "elapsed_s": 0.1, "errors": []})
    resp = client.post("/documents/upload", files={"file": ("b.txt", "第二条：测试。".encode(), "text/plain")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] and body["chunks"] == 12
    assert (kb / "b.txt").exists()


def test_upload_rolls_back_when_rebuild_fails(client, kb, monkeypatch):
    def broken():
        raise RuntimeError("embedding 模型加载失败")

    monkeypatch.setattr(rag_api, "rebuild_retriever", broken)
    resp = client.post("/documents/upload", files={"file": ("c.txt", b"x", "text/plain")})
    assert resp.status_code == 500
    assert "回滚" in resp.json()["detail"]
    assert not (kb / "c.txt").exists()


def test_upload_restores_previous_version_on_rebuild_failure(client, kb, monkeypatch):
    target = kb / "a.txt"
    original = target.read_text(encoding="utf-8")

    def broken():
        raise RuntimeError("重建失败")

    monkeypatch.setattr(rag_api, "rebuild_retriever", broken)
    resp = client.post("/documents/upload",
                       files={"file": ("a.txt", "被覆盖的内容".encode(), "text/plain")})
    assert resp.status_code == 500
    assert target.read_text(encoding="utf-8") == original    # 原文件必须还在


def test_delete_missing_document_404(client):
    resp = client.delete("/documents/nope.txt")
    assert resp.status_code == 404
    assert "文档不存在" in resp.json()["detail"]


def test_delete_success(client, kb, monkeypatch):
    monkeypatch.setattr(rag_api, "rebuild_retriever",
                        lambda: {"chunks": 1, "docs": 0, "elapsed_s": 0.1, "errors": []})
    resp = client.delete("/documents/a.txt")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == "a.txt"
    assert not (kb / "a.txt").exists()


def test_delete_restores_file_when_rebuild_fails(client, kb, monkeypatch):
    def broken():
        raise RuntimeError("BM25 重建失败")

    monkeypatch.setattr(rag_api, "rebuild_retriever", broken)
    resp = client.delete("/documents/a.txt")
    assert resp.status_code == 500
    assert "已恢复" in resp.json()["detail"]
    assert (kb / "a.txt").exists()                            # 删了也要能还原
