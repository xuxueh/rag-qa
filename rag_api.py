"""
📚 RAG 问答 API 服务 v3（生产链路 + 文档管理）
================================================
生产链路：清洗 → 句子边界切块 → 混合检索(RRF) → rerank → DeepSeek 生成 → Citation

文档管理：
- GET    /documents              列出知识库文档
- POST   /documents/upload       上传文档（txt/md/pdf/docx）→ 自动重建检索器
- DELETE /documents/{name}       删除文档 → 自动重建检索器

启动: python run_server.py
"""
import os
import sys
import time
import threading
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
# ⚠️ 清理 PYTHONPATH 污染（Hermes 终端会注入 Python 3.11 的包路径）
os.environ.pop("PYTHONPATH", None)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

import rag_qa as rq
from hybrid_retriever import build_hybrid
from rerank import rerank

# ═══════════════ 配置 ═══════════════
DOC_DIR = rq.DOC_DIR
EMBEDDING_MODEL = rq.EMBEDDING_MODEL_PATH
ALLOWED_EXT = {".txt", ".md", ".pdf", ".docx"}
# ════════════════════════════════════

app = FastAPI(title="RAG 问答 API v3（生产链路 + 文档管理）", version="3.0")

# 全局检索器（懒加载 + 可重建）
_retriever = None
_rebuild_lock = threading.Lock()


def get_retriever():
    global _retriever
    if _retriever is None:
        print("构建混合检索器（首次约 1-2 分钟）...")
        _retriever = build_hybrid(DOC_DIR, EMBEDDING_MODEL)
        print("✅ 检索器就绪")
    return _retriever


def rebuild_retriever():
    """文档变更后重建检索器"""
    global _retriever
    with _rebuild_lock:
        print("🔄 文档变更，重建检索器...（约 1-2 分钟）")
        _retriever = build_hybrid(DOC_DIR, EMBEDDING_MODEL)
        print("✅ 重建完成")


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str
    sources: list[dict]


# ─── 页面 ───
@app.get("/", response_class=HTMLResponse)
def root():
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


# ─── 问答 ───
@app.post("/ask", response_model=Answer)
def ask(q: Question):
    retriever = get_retriever()
    start = time.time()

    recall = retriever.retrieve(q.question, top_k=10)  # 10池: 池内命中95.3%(5池仅86%)
    ranked = rerank(q.question, recall, top_n=3)

    parts, sources = [], []
    for t in ranked:
        fn = retriever.get_citation(t)
        parts.append(f"[来源：{fn}]\n{t}")
        sources.append({"text": t, "source": fn})
    context = "\n\n".join(parts)

    llm = ChatOpenAI(model="deepseek-chat", api_key=rq.DEEPSEEK_API_KEY,
                     base_url="https://api.deepseek.com", temperature=0)
    prompt = f"""根据以下资料回答问题。如果资料里没有答案，就说不知道，不要瞎编。

回答要求：
1. 先说明依据：指出答案来自资料的哪份文件、第几条
2. 再给出结论
3. 引用内容不得超出资料范围

资料：
{context}

问题：{q.question}

回答："""
    answer = llm.invoke(prompt).content
    elapsed = round(time.time() - start, 2)
    print(f"⏱️ [{elapsed}s] Q: {q.question}")
    return Answer(answer=answer, sources=sources)


# ─── 文档管理 ───
@app.get("/documents")
def list_documents():
    files = [f for f in sorted(os.listdir(DOC_DIR))
             if os.path.isfile(os.path.join(DOC_DIR, f))
             and Path(f).suffix.lower() in ALLOWED_EXT]
    return {"documents": files, "count": len(files)}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"不支持的文件类型 {ext}，支持: {ALLOWED_EXT}")
    # 防路径穿越
    safe_name = Path(file.filename).name
    dest = Path(DOC_DIR) / safe_name
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    print(f"📄 上传: {safe_name} ({len(content)} bytes)")
    # 重建（新文档入库后检索才生效）
    rebuild_retriever()
    return {"ok": True, "uploaded": safe_name, "note": "检索器已重建"}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    safe_name = Path(filename).name
    path = Path(DOC_DIR) / safe_name
    if not path.exists():
        raise HTTPException(404, f"文档不存在: {safe_name}")
    os.remove(path)
    print(f"🗑️ 删除: {safe_name}")
    rebuild_retriever()
    return {"ok": True, "deleted": safe_name, "note": "检索器已重建"}
