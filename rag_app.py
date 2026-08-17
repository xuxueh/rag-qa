"""
📚 企业知识库问答系统（RAG 网页版）
=================================
兼容本地（Windows/.env）与云端（Streamlit Cloud/Secrets）
运行: streamlit run rag_app.py
"""

import os
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

# ═══════════════ 配置 ═══════════════
ENV_FILE = Path(__file__).parent / ".env"
DOC_PATH = str(Path(__file__).parent / "公司制度.txt")

# 嵌入模型：默认用 HF 模型名（云端自动下载）
# 本地可用 .env 里的 EMBEDDING_MODEL 覆盖为本地路径（更快）
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# ═══════════════════════════════════

EXAMPLE_QUESTIONS = [
    "报销金额超过 5000 元需要谁审批？",
    "迟到超过 30 分钟怎么处理？",
    "年假有几天？",
    "加班怎么调休？",
]


def load_key():
    """读取 key：优先 Streamlit Secrets（云端），其次 .env（本地）"""
    # 云端：Streamlit Secrets
    try:
        if "DEEPSEEK_API_KEY" in st.secrets:
            return st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        pass
    # 本地：.env 文件
    try:
        if ENV_FILE.exists():
            for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("DEEPSEEK_API_KEY="):
                    return line.split("=", 1)[1]
    except Exception:
        pass
    return None


def load_embedding_model():
    """读取嵌入模型配置：.env 有就用本地路径，否则用 HF 模型名"""
    try:
        if ENV_FILE.exists():
            for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("EMBEDDING_MODEL="):
                    return line.split("=", 1)[1]
    except Exception:
        pass
    return DEFAULT_EMBEDDING_MODEL


DEEPSEEK_API_KEY = load_key()
EMBEDDING_MODEL = load_embedding_model()

st.set_page_config(page_title="智能知识库问答", page_icon="🤖")

st.title("🤖 智能知识库问答")
st.caption("基于 RAG：LangChain + Chroma + DeepSeek")


@st.cache_resource
def build_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    loader = TextLoader(DOC_PATH, encoding='utf-8')
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(documents)
    return Chroma.from_documents(chunks, embeddings)


def answer_question(question, db):
    """RAG 问答，返回 (回答, 片段列表)"""
    docs = db.similarity_search_with_score(question, k=3)
    context = "\n\n".join([d[0].page_content for d in docs])

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=0,
    )

    prompt = f"""根据以下资料回答问题。如果资料里没有答案，就说不知道，不要瞎编。

资料：
{context}

问题：{question}

回答："""

    answer = llm.invoke(prompt).content
    return answer, docs


# ─── 初始化 ───
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ─── 侧边栏 ───
with st.sidebar:
    st.header("📋 知识库")
    with st.spinner("加载知识库..."):
        db = build_db()
    st.success("✅ 知识库已加载")

    if not DEEPSEEK_API_KEY:
        st.error("❌ 未配置 DEEPSEEK_API_KEY（云端用 Secrets，本地用 .env）")

    st.divider()
    st.subheader("💡 示例问题")
    selected = None
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True, key=f"btn_{q}"):
            selected = q

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ─── 展示聊天记录 ───
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "docs" in msg and msg["docs"]:
            with st.expander("📄 引用来源"):
                for i, (doc, score) in enumerate(msg["docs"], 1):
                    st.markdown(f"**片段 {i}**（相关度 {score:.3f}）")
                    st.info(doc.page_content)

# ─── 输入 ───
user_input = st.chat_input("输入你的问题...")
question = selected if selected else user_input

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state["messages"].append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("🤔 思考中..."):
            answer, docs = answer_question(question, db)
        st.write(answer)
        if docs:
            with st.expander("📄 引用来源"):
                for i, (doc, score) in enumerate(docs, 1):
                    st.markdown(f"**片段 {i}**（相关度 {score:.3f}）")
                    st.info(doc.page_content)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "docs": docs}
    )
