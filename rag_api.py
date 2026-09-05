"""
📚 RAG 问答 API 服务（生产链路版）
==================================
把 RAG 包成 API：任何人发 POST 请求就能问知识库
生产链路：文本清洗 → 句子边界切块 → 混合检索(BM25+向量 RRF) → rerank 精排 → DeepSeek 生成

启动: python run_server.py （或 uvicorn rag_api:app）
测试: curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"年假有几天？"}'
"""

import os
import sys
import time
from pathlib import Path

# Windows 控制台 UTF-8
sys.stdout.reconfigure(encoding="utf-8")

# ⚠️ 清理 PYTHONPATH 污染（Hermes 终端会注入 Python 3.11 的包路径）
os.environ.pop("PYTHONPATH", None)

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
# ⚠️ 导入顺序有讲究（Windows DLL 加载顺序）：
# 必须先加载 huggingface/chroma/openai，再加载 TextLoader/text_splitters
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 复用生产链路模块
import rag_qa as rq
from hybrid_retriever import build_hybrid
from rerank import rerank

# ═══════════════ 配置（从 rag_qa 读） ═══════════════
DOC_DIR = rq.DOC_DIR
EMBEDDING_MODEL = rq.EMBEDDING_MODEL_PATH
# ═══════════════════════════════════════════════════

# ─── 构建混合检索器（启动时加载一次）───
print("构建混合检索器（生产链路）...（约 1-2 分钟）")
retriever = build_hybrid(DOC_DIR, EMBEDDING_MODEL)
print(f"✅ 检索器构建完成（混合检索 + rerank 就绪）")

# ─── FastAPI 应用 ───
app = FastAPI(title="RAG 问答 API（生产链路）", version="2.0")


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str
    sources: list[dict]  # [{text, source}]


@app.get("/", response_class=HTMLResponse)
def root():
    """返回问答网页界面（自包含，无 CDN 依赖）"""
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.post("/ask", response_model=Answer)
def ask(q: Question):
    start = time.time()

    # 1. 混合检索召回 top-10（向量 + BM25）
    recall = retriever.retrieve(q.question, top_k=10)

    # 2. rerank 精排取 top-3
    ranked = rerank(q.question, recall, top_n=3)

    # 3. 组装上下文（带来源）
    parts = []
    sources = []
    for t in ranked:
        fn = retriever.get_source(t)
        parts.append(f"[来源：{fn}]\n{t}")
        sources.append({"text": t, "source": fn})
    context = "\n\n".join(parts)

    # 4. DeepSeek 生成
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=rq.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=0,
    )
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
