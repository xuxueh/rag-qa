"""
📚 企业知识库问答系统（RAG 网页版 - 稳定版 v2）
=========================================
修复：示例问题按钮可用
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


# ─── 侧边栏：示例问题 ───
with st.sidebar:
    st.header("📋 知识库")
    with st.spinner("加载知识库..."):
        db = build_db()
    st.success("✅ 知识库已加载")

    st.divider()
    st.subheader("💡 示例问题")
    for q in [
        "报销金额超过 5000 元需要谁审批？",
        "迟到超过 30 分钟怎么处理？",
        "年假有几天？",
        "加班怎么调休？",
    ]:
        if st.button(q, use_container_width=True, key=f"btn_{q}"):
            st.session_state["question"] = q   # 点按钮 → 存问题
            st.rerun()                          # 立即刷新

# ─── 输入区 ───
question = st.text_input(
    "❓ 你的问题：",
    placeholder="输入问题后按回车，或点左侧示例",
    value=st.session_state.get("question", ""),
)

# ─── 提问处理（去重：同一问题只处理一次）───
if question and question != st.session_state.get("last_q"):
    st.session_state["last_q"] = question

    with st.spinner("🤔 思考中..."):
        answer, docs = answer_question(question, db)

    st.markdown("---")
    st.markdown(f"**❓ 问题：** {question}")
    st.markdown(f"**🤖 回答：**")
    st.write(answer)

    st.markdown("---")
    with st.expander(f"📄 引用来源（{len(docs)} 个片段）"):
        for i, (doc, score) in enumerate(docs, 1):
            st.markdown(f"**片段 {i}**（相关度 {score:.3f}）")
            st.info(doc.page_content)
