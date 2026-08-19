"""
📚 RAG 问答 API 服务
====================
把 RAG 包成 API：任何人发 POST 请求就能问知识库
启动: uvicorn rag_api:app --reload
测试: curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"年假有几天？"}'
"""

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
# ⚠️ 导入顺序有讲究（Windows DLL 加载顺序）：
# 必须先加载 huggingface/chroma/openai，再加载 TextLoader/text_splitters，
# 否则原生库 DLL 冲突导致进程崩溃（exit 127）
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter

# ═══════════════ 配置 ═══════════════
ENV_FILE = Path(r"C:\Users\xujin\ai-learning\项目\rag-qa\.env")
DOC_DIR = str(Path(r"C:\Users\xujin\ai-learning\项目\rag-qa") / "知识库")
EMBEDDING_MODEL = r"C:\Users\xujin\ai-learning\data\models\gte-small\models\iic--nlp_gte_sentence-embedding_chinese-small\snapshots\master"
# ═══════════════════════════════════


def load_key():
    try:
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DEEPSEEK_API_KEY="):
                    return line.split("=", 1)[1]
    except Exception:
        pass
    return None


DEEPSEEK_API_KEY = load_key()

# ─── 构建知识库（启动时加载一次）───
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
loader = DirectoryLoader(DOC_DIR, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)
db = Chroma.from_documents(chunks, embeddings)
print(f"✅ 知识库构建完成：{len(documents)} 份文档 → {len(chunks)} 块")

# ─── FastAPI 应用 ───
app = FastAPI(title="RAG 问答 API", version="1.0")


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str
    sources: list[str]


@app.get("/")
def root():
    return {"message": "RAG 问答 API 已启动", "docs": "/docs"}


@app.post("/ask", response_model=Answer)
def ask(q: Question):
    # 1. 检索最相关的 3 段（带来源文件名）
    docs = db.similarity_search_with_score(q.question, k=3)
    context = "\n\n".join([
        f"[来源：{Path(d[0].metadata.get('source', '未知')).stem}]\n{d[0].page_content}"
        for d in docs
    ])

    # 2. 调用 DeepSeek
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
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

    # 3. 返回回答 + 引用来源
    sources = [d[0].page_content for d in docs]
    return Answer(answer=answer, sources=sources)
