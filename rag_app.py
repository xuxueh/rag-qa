"""
📚 企业知识库问答系统（RAG 网页版 - 聊天版 v2）
=========================================
修复：示例问题按钮直接触发提问
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
ENV_FILE = Path(r"C:\Users\xujin\ai-learning\项目\rag-qa\.env")
DOC_PATH = "公司制度.txt"
EMBEDDING_MODEL_PATH = r"C:\Users\xujin\ai-learning\data\models\gte-small\models\iic--nlp_gte_sentence-embedding_chinese-small\snapshots\master"
# ═══════════════════════════════════

EXAMPLE_QUESTIONS = [
    "报销金额超过 5000 元需要谁审批？",
    "迟到超过 30 分钟怎么处理？",
    "年假有几天？",
    "加班怎么调休？",
]


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

st.set_page_config(page_title="智能知识库问答", page_icon="🤖")

st.title("🤖 智能知识库问答")
st.caption("基于 RAG：LangChain + Chroma + DeepSeek")


@st.cache_resource
def build_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
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

# ─── 输入：底部输入框 或 示例按钮 ───
user_input = st.chat_input("输入你的问题...")
question = selected if selected else user_input

if question:
    # 用户气泡
    with st.chat_message("user"):
        st.write(question)
    st.session_state["messages"].append({"role": "user", "content": question})

    # AI 气泡
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
