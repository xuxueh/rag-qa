"""
📚 RAG 问答 API 服务 v4（生产链路 + 文档管理 + 健康检查）
=========================================================
链路：加载/清洗/切块 → 混合检索(RRF) → rerank 精排 → DeepSeek 生成 → Citation

接口：
- GET    /health                 健康检查（配置问题、检索器状态、文档数）
- POST   /ask                    问答（返回答案 + 引用 + 耗时）
- GET    /documents              列出知识库文档（大小/修改时间）
- POST   /documents/upload       上传文档 → 自动重建；重建失败自动回滚
- DELETE /documents/{filename}   删除文档 → 自动重建；重建失败自动恢复

工程约定：
- 配置全部来自 config.py（密钥/路径走环境变量，不写死）
- 重建串行化（锁），问答侧只做原子引用替换，不阻塞已就绪的服务
- 任何失败都返回「能看懂的中文原因 + 建议」，不裸抛 500
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ.pop("PYTHONPATH", None)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import config
import rag_qa as rq

log = logging.getLogger("rag.api")

# ═══════════════════════ 运行期状态 ═══════════════════════
_retriever = None            # 当前生效的检索器（原子替换）
_retriever_ready_at: float | None = None
_rebuild_lock = threading.Lock()
_llm = None
_llm_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.setup_logging()
    problems = config.check(require_api_key=False)
    for p in problems:
        log.warning("配置问题：%s", p)
    log.info("服务启动：知识库=%s 文档数=%d", config.KNOWLEDGE_DIR, len(list_document_files()))
    yield
    log.info("服务停止")


app = FastAPI(title="RAG 问答 API v4（生产链路 + 文档管理）", version="4.0", lifespan=lifespan)


# ═══════════════════════ 模型 / 检索器 ═══════════════════════
def get_retriever(force: bool = False):
    """懒加载检索器；构建失败把原因记下来并抛给调用方（不缓存坏的半成品）"""
    global _retriever, _retriever_ready_at
    if _retriever is not None and not force:
        return _retriever
    with _rebuild_lock:
        if _retriever is not None and not force:
            return _retriever
        problems = config.check(require_api_key=False)
        if problems:
            raise RuntimeError("配置不完整：" + "；".join(problems))
        log.info("构建检索器（首次约 1-2 分钟）...")
        t0 = time.perf_counter()
        from hybrid_retriever import build_hybrid

        _retriever = build_hybrid(config.KNOWLEDGE_DIR, config.EMBEDDING_MODEL)
        _retriever_ready_at = time.time()
        log.info("检索器就绪，用时 %.1fs", time.perf_counter() - t0)
    return _retriever


def rebuild_retriever() -> dict:
    """文档变更后重建。返回 report（含失败文件清单）；失败时旧检索器继续可用。"""
    global _retriever, _retriever_ready_at
    with _rebuild_lock:
        report: dict = {}
        log.info("🔄 文档变更，重建检索器...")
        t0 = time.perf_counter()
        from hybrid_retriever import build_hybrid

        new_retriever = build_hybrid(config.KNOWLEDGE_DIR, config.EMBEDDING_MODEL, report=report)
        _retriever = new_retriever          # 原子替换：构建成功才切换
        _retriever_ready_at = time.time()
        report["elapsed_s"] = round(time.perf_counter() - t0, 1)
        log.info("✅ 重建完成：%d chunk，失败 %d 个文件，用时 %.1fs",
                 report.get("chunks", 0), len(report.get("errors", [])), report["elapsed_s"])
        return report


def get_llm():
    global _llm, _llm_error
    if _llm is not None:
        return _llm
    try:
        _llm = rq.build_llm()
        _llm_error = None
    except config.ConfigError as exc:
        _llm_error = str(exc)
        raise
    return _llm


# ═══════════════════════ 数据模型 ═══════════════════════
class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")


class Answer(BaseModel):
    answer: str
    sources: list[dict]
    timings: dict = {}


# ═══════════════════════ 工具 ═══════════════════════
def list_document_files() -> list[Path]:
    if not config.KNOWLEDGE_DIR.exists():
        return []
    return sorted(
        p for p in config.KNOWLEDGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in config.ALLOWED_EXT
    )


# ═══════════════════════ 页面 / 健康检查 ═══════════════════════
@app.get("/", response_class=HTMLResponse)
def root():
    html_path = Path(__file__).parent / "static" / "index.html"
    if not html_path.exists():
        raise HTTPException(404, f"前端页面缺失：{html_path}")
    return html_path.read_text(encoding="utf-8")


@app.get("/health")
def health():
    problems = config.check(require_api_key=False)
    files = list_document_files()
    return {
        "status": "ok" if (_retriever is not None and not problems) else "degraded",
        "retriever_ready": _retriever is not None,
        "retriever_ready_at": _retriever_ready_at,
        "documents": len(files),
        "chunks": len(_retriever.chunks) if _retriever is not None else 0,
        "config_issues": problems,
        "config": config.describe(),
    }


# ═══════════════════════ 问答 ═══════════════════════
@app.post("/ask", response_model=Answer)
def ask(q: Question):
    try:
        retriever = get_retriever()
    except Exception as exc:
        log.error("检索器不可用：%s", exc)
        raise HTTPException(503, f"知识库尚未就绪：{exc}") from exc

    try:
        llm = get_llm()
    except config.ConfigError as exc:
        raise HTTPException(503, f"大模型未配置：{exc}") from exc

    t0 = time.perf_counter()
    try:
        result = rq.answer(q.question, retriever=retriever, llm=llm)
    except Exception as exc:
        log.exception("问答失败")
        raise HTTPException(502, f"问答失败（{type(exc).__name__}）：{exc}") from exc

    log.info("⏱️ [%.2fs] Q: %s", time.perf_counter() - t0, q.question)
    return Answer(answer=result["answer"], sources=result["sources"], timings=result["timings"])


# ═══════════════════════ 文档管理 ═══════════════════════
@app.get("/documents")
def list_documents():
    files = list_document_files()
    items = [
        {"name": p.name, "size": p.stat().st_size,
         "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))}
        for p in files
    ]
    return {"documents": [i["name"] for i in items], "items": items, "count": len(items),
            "knowledge_dir": str(config.KNOWLEDGE_DIR)}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    raw_name = file.filename or ""
    safe_name = Path(raw_name).name
    if not safe_name:
        raise HTTPException(400, "文件名无效（为空）")
    ext = Path(safe_name).suffix.lower()
    if ext not in config.ALLOWED_EXT:
        raise HTTPException(400, f"不支持的文件类型 {ext or '（无扩展名）'}，"
                                f"支持：{', '.join(sorted(config.ALLOWED_EXT))}")

    content = await file.read()
    if not content:
        raise HTTPException(400, f"{safe_name} 是空文件，上传无意义")
    if len(content) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"{safe_name} 超过 {config.MAX_UPLOAD_MB}MB 上限")

    config.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.KNOWLEDGE_DIR / safe_name
    backup = None
    if dest.exists():                     # 覆盖同名文件时先留底，重建失败可还原
        backup = dest.with_suffix(dest.suffix + ".bak")
        shutil.copy2(dest, backup)

    # 原子落盘：先写临时文件再替换，避免半截文件被其它进程读到
    fd, tmp_path = tempfile.mkstemp(dir=str(config.KNOWLEDGE_DIR), suffix=".uploading")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        os.replace(tmp_path, dest)
    except Exception as exc:
        Path(tmp_path).unlink(missing_ok=True)
        log.exception("写入上传文件失败")
        raise HTTPException(500, f"保存文件失败（{type(exc).__name__}）：{exc}") from exc

    log.info("📄 上传：%s（%d bytes）", safe_name, len(content))
    try:
        report = rebuild_retriever()
    except Exception as exc:
        log.exception("重建失败，回滚上传")
        dest.unlink(missing_ok=True)
        if backup and backup.exists():     # 覆盖场景：恢复原文件
            os.replace(backup, dest)
        raise HTTPException(
            500,
            f"文件已保存但重建知识库失败，已回滚（旧知识库仍在服务）：{exc}",
        ) from exc

    if backup:
        backup.unlink(missing_ok=True)
    file_errors = report.get("errors", [])
    return {
        "ok": True,
        "uploaded": safe_name,
        "chunks": report.get("chunks"),
        "documents": report.get("docs"),
        "elapsed_s": report.get("elapsed_s"),
        "file_errors": file_errors,
        "note": "检索器已重建" if not file_errors else "已重建，但有文件解析失败（见 file_errors）",
    }


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    safe_name = Path(filename).name
    path = config.KNOWLEDGE_DIR / safe_name
    if not path.exists() or path.suffix.lower() not in config.ALLOWED_EXT:
        raise HTTPException(404, f"文档不存在：{safe_name}")

    backup = path.with_suffix(path.suffix + ".deleted")
    shutil.copy2(path, backup)
    path.unlink()
    log.info("🗑️ 删除：%s", safe_name)
    try:
        report = rebuild_retriever()
    except Exception as exc:
        log.exception("重建失败，恢复被删文件")
        os.replace(backup, path)
        raise HTTPException(
            500,
            f"已删除但重建知识库失败，文档已恢复（旧知识库仍在服务）：{exc}",
        ) from exc
    backup.unlink(missing_ok=True)
    return {"ok": True, "deleted": safe_name, "chunks": report.get("chunks"),
            "elapsed_s": report.get("elapsed_s"), "file_errors": report.get("errors", []),
            "note": "检索器已重建"}
