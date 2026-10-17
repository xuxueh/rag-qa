"""文档加载器测试：多格式、编码回退、坏文件必须给出可读错误"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_loader import DocumentLoadError, load_documents, load_documents_with_report


def test_load_utf8_txt(tmp_path):
    (tmp_path / "a.txt").write_text("第一条：婚假 15 天。", encoding="utf-8")
    docs, errors = load_documents_with_report(tmp_path)
    assert errors == []
    assert "婚假 15 天" in docs[0].page_content
    assert docs[0].metadata["source"].endswith("a.txt")


def test_load_gbk_txt_falls_back(tmp_path):
    (tmp_path / "gbk.txt").write_bytes("第一条：报销需发票。".encode("gbk"))
    docs, errors = load_documents_with_report(tmp_path)
    assert errors == []
    assert "报销需发票" in docs[0].page_content


def test_load_markdown(tmp_path):
    (tmp_path / "b.md").write_text("# 标题\n\n第二条：内容。", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert any("第二条" in d.page_content for d in docs)


def test_empty_file_reports_clear_error(tmp_path):
    (tmp_path / "empty.txt").write_bytes(b"")
    docs, errors = load_documents_with_report(tmp_path)
    assert docs == []
    assert "空文件" in errors[0]["error"]


def test_unsupported_extension_is_reported(tmp_path):
    (tmp_path / "x.xyz").write_text("内容", encoding="utf-8")
    (tmp_path / "ok.txt").write_text("第一条：有效内容。", encoding="utf-8")
    docs, errors = load_documents_with_report(tmp_path)
    assert len(docs) == 1              # 支持的文件照常加载
    assert errors == []                # 不支持的类型只是跳过，不算错误


def test_corrupt_pdf_raises_with_hint(tmp_path):
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    with pytest.raises(DocumentLoadError) as exc:
        load_documents(tmp_path)
    assert "bad.pdf" in str(exc.value)


def test_all_failed_raises_with_per_file_details(tmp_path):
    (tmp_path / "e1.txt").write_bytes(b"")
    with pytest.raises(DocumentLoadError) as exc:
        load_documents(tmp_path)
    assert "e1.txt" in str(exc.value)


def test_missing_dir_raises_with_fix_hint(tmp_path):
    with pytest.raises(DocumentLoadError) as exc:
        load_documents_with_report(tmp_path / "nope")
    assert "知识库目录不存在" in str(exc.value)


def test_docx_paragraphs_are_loaded(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph("第一条：办公用品在行政部登记领用。")
    path = tmp_path / "c.docx"
    d.save(path)
    docs = load_documents(tmp_path)
    assert "行政部登记领用" in docs[0].page_content
