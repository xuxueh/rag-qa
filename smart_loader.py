"""
smart_loader.py — 多格式文档加载器（TXT / Markdown / PDF / Word）
=================================================================
按扩展名自动选解析方式：
- .txt / .md → TextLoader（UTF-8 优先，失败回退 GB18030 并记警告）
- .pdf       → PyPDFLoader（带页码 metadata，用于页码级引用）
- .docx      → python-docx（段落 + 表格都要，纯段落会漏掉表格式制度）

工程要求：解析失败不能静默吞掉。每个文件错误都会带文件名 + 原因 + 修复建议，
既记日志也汇总返回，接口层可以把失败清单直接回给上传者。
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

log = logging.getLogger(__name__)


class DocumentLoadError(RuntimeError):
    """单个文档解析失败（消息里已含文件名与修复建议）"""


def _read_text(path: str):
    """读纯文本：先 UTF-8，再回退 GB18030（Windows 记事本常见编码）"""
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read(), encoding
        except UnicodeDecodeError:
            continue
    raise DocumentLoadError(
        f"{os.path.basename(path)} 编码无法识别（已尝试 UTF-8 / GB18030）。"
        f"修复：用记事本另存为 UTF-8 后重新入库"
    )


def load_file(path: str):
    """加载单个文件 → list[Document]。失败抛 DocumentLoadError（带修复建议）。"""
    from langchain_core.documents import Document

    name = os.path.basename(path)
    if not os.path.exists(path):
        raise DocumentLoadError(f"文件不存在：{path}")
    if os.path.getsize(path) == 0:
        raise DocumentLoadError(f"{name} 是空文件（0 字节），没有内容可入库")
    ext = os.path.splitext(name)[1].lower()

    if ext in (".txt", ".md"):
        text, encoding = _read_text(path)
        if encoding != "utf-8":
            log.warning("%s 不是 UTF-8（实际 %s），已按该编码读取，建议统一转 UTF-8", name, encoding)
        return [Document(page_content=text, metadata={"source": path})]

    if ext == ".pdf":
        try:
            from langchain_community.document_loaders import PyPDFLoader

            docs = PyPDFLoader(path).load()
        except Exception as exc:
            raise DocumentLoadError(
                f"{name} 解析失败（{type(exc).__name__}: {exc}）。"
                f"常见原因：PDF 加密/损坏/被占用。修复：用 PDF 阅读器另存为新文件，或导出为 txt 再入库"
            ) from exc
        if not docs or not any((d.page_content or "").strip() for d in docs):
            raise DocumentLoadError(
                f"{name} 解析后没有文字（可能是扫描件/图片型 PDF）。"
                f"修复：先做 OCR 转成文字版 PDF 或 txt，再入库"
            )
        return docs

    if ext == ".docx":
        try:
            from docx import Document as DocxDocument

            docx = DocxDocument(path)
            lines = [p.text for p in docx.paragraphs if p.text and p.text.strip()]
            for table in docx.tables:                      # 制度文件常用表格，不能漏
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
                    if cells:
                        lines.append(" | ".join(cells))
        except Exception as exc:
            raise DocumentLoadError(
                f"{name} 解析失败（{type(exc).__name__}: {exc}）。"
                f"常见原因：文件损坏或其实是 .doc 旧格式。修复：用 Word 另存为 .docx 后重试"
            ) from exc
        if not lines:
            raise DocumentLoadError(f"{name} 里没有可提取的文字（空文档或全是图片）")
        return [Document(page_content="\n".join(lines), metadata={"source": path})]

    raise DocumentLoadError(
        f"{name} 类型不支持（{ext or '无扩展名'}）。支持：{', '.join(sorted(config.ALLOWED_EXT))}"
    )


def load_documents_with_report(doc_dir: str | None = None):
    """加载整个目录 → (documents, errors)。errors: [{"file":..., "error":...}]

    单个文件失败不影响其它文件（知识库要能容错），但失败原因一定带出来。
    """
    doc_dir = str(doc_dir or config.KNOWLEDGE_DIR)
    if not os.path.isdir(doc_dir):
        raise DocumentLoadError(f"知识库目录不存在：{doc_dir}。修复：创建该目录或设置 KNOWLEDGE_DIR")

    documents, errors = [], []
    files = sorted(os.listdir(doc_dir))
    candidate = [f for f in files
                 if os.path.isfile(os.path.join(doc_dir, f))
                 and os.path.splitext(f)[1].lower() in config.ALLOWED_EXT]
    for fname in files:
        path = os.path.join(doc_dir, fname)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(fname)[1].lower() not in config.ALLOWED_EXT:
            log.debug("跳过不支持的文件：%s", fname)
            continue
        try:
            loaded = load_file(path)
            documents.extend(loaded)
            log.info("已加载 %s（%d 段）", fname, len(loaded))
        except DocumentLoadError as exc:
            log.error("加载失败 %s：%s", fname, exc)
            errors.append({"file": fname, "error": str(exc)})
        except Exception as exc:  # 兜底：绝不让单个文件炸掉整次构建
            log.exception("加载 %s 时发生未预期错误", fname)
            errors.append({"file": fname, "error": f"未预期错误 {type(exc).__name__}: {exc}"})

    if not candidate:
        log.warning("目录里没有受支持的文档（%s）：%s", sorted(config.ALLOWED_EXT), doc_dir)
    log.info("加载完成：%d 份文档成功，%d 份失败", len(documents), len(errors))
    return documents, errors


def load_documents(doc_dir: str | None = None):
    """兼容版：只要文档列表；全部失败时抛 DocumentLoadError（消息含每个文件的失败原因）"""
    documents, errors = load_documents_with_report(doc_dir)
    if not documents:
        detail = "\n".join(f"  - {e['file']}: {e['error']}" for e in errors) or "  （目录为空）"
        raise DocumentLoadError(f"没有任何文档加载成功：{doc_dir}\n{detail}")
    return documents


if __name__ == "__main__":
    config.setup_logging()
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_docs")
    if not os.path.exists(test_dir):
        from make_test_docs import make_test_docs

        make_test_docs(test_dir)
    docs, errs = load_documents_with_report(test_dir)
    for d in docs:
        print(f"  [{os.path.basename(d.metadata.get('source', '?'))}] {d.page_content[:30]}...")
    print(f"成功 {len(docs)}，失败 {len(errs)}")
