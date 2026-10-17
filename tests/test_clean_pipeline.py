"""文本清洗与引用格式化的单元测试（不依赖模型/网络）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clean_pipeline import clean_document, clean_text, dedupe_lines
from hybrid_retriever import ChunkMeta, extract_articles, format_citation


def test_clean_text_normalizes_whitespace_and_newlines():
    dirty = "第一条：上班时间 9:00\r\n\r\n\r\n  至 18:00\u3000。\x00"
    cleaned = clean_text(dirty)
    assert "\r" not in cleaned
    assert "\u3000" not in cleaned
    assert "\x00" not in cleaned
    assert "9:00" in cleaned and "18:00" in cleaned


def test_clean_text_collapses_blank_lines():
    assert "\n\n\n" not in clean_text("a\n\n\n\n\nb")


def test_dedupe_lines_keeps_first_occurrence():
    text = "第一条：A\n\n第一条：A\n第二条：B"
    assert dedupe_lines(text) == "第一条：A\n\n第二条：B"


def test_clean_document_pipeline():
    dirty = "  第一条：A 。\r\n\r\n\r\n第一条：A 。\u3000"
    assert clean_document(dirty) == "第一条：A 。"


def test_extract_articles_mixed_chinese_and_digits():
    assert extract_articles("第三条：…第十二条：…第 7 条：…") == [3, 12, 7]


def test_extract_articles_dedupes():
    assert extract_articles("第五条 … 第五条") == [5]


def test_citation_with_articles():
    meta = ChunkMeta("d01_c01", "文本", "01-考勤管理制度.txt", [3])
    assert meta.citation() == "01-考勤管理制度.txt · 第三条"


def test_citation_without_articles_is_filename_only():
    assert format_citation("02-费用报销制度.txt", []) == "02-费用报销制度.txt"


def test_citation_page_is_one_based():
    meta = ChunkMeta("d01_c01", "文本", "a.pdf", [2], page="0")
    assert meta.citation() == "a.pdf · 第二条 · 第1页"


def test_citation_article_above_ten_uses_digits():
    assert format_citation("x.txt", [12]) == "x.txt · 第12条"
