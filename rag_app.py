"""
📚 企业知识库问答系统（RAG 网页版）
=================================
Streamlit 界面 + LangChain RAG 流水线
运行: streamlit run rag_app.py
"""

import os

from dotenv import load_dotenv

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

load_dotenv()   # 读取 .env 文件

# ═══════════════ 配置 ═══════════════
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")   # 从 .env 读，不写死在代码里
DOC_PATH = "公司制度.txt"
EMBEDDING_MODEL_PATH = r"C:\Users\xujin\ai-learning\data\models\gte-embedding\models\iic--nlp_gte_sentence-embedding_chinese-base\snapshots\master"
DB_DIR = "chroma_db"   # 向量库持久化目录
# ═══════════════════════════════════


@st.cache_resource   # 缓存，刷新页面不重建
def build_db():
    """构建或加载向量知识库（已存在则直接读磁盘）"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

    # 库已存在 → 直接加载（快）
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print("✓ 加载已保存的向量库")
        return Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )

    # 库不存在 → 构建并保存（慢一次）
    loader = TextLoader(DOC_PATH, encoding='utf-8')
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(documents)

    print("✓ 构建新向量库...")
    return Chroma.from_documents(
        chunks, embeddings,
        persist_directory=DB_DIR
    )


def answer_question(question, db):
    """RAG 问答：检索 + 生成，返回 (回答, 相关片段列表)"""
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

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


# ═══════════════ 页面 ═══════════════
st.set_page_config(page_title="企业知识库问答", page_icon="📚")

st.title("📚 企业知识库问答系统")
st.caption("基于 RAG：LangChain + Chroma + DeepSeek")

with st.sidebar:
    st.header("💡 说明")
    st.write("输入问题，系统会：")
    st.write("1️⃣ 在知识库中语义检索最相关片段")
    st.write("2️⃣ 让 DeepSeek 基于片段回答")
    st.write("3️⃣ 展示引用来源")
    st.divider()
    st.write("**试试问：**")
    st.write("- 报销金额超过 5000 元需要谁审批？")
    st.write("- 迟到超过 30 分钟怎么处理？")
    st.write("- 年假有几天？")

with st.spinner("正在加载知识库..."):
    db = build_db()
st.success("✅ 知识库加载完成（考勤 + 报销制度）")

question = st.text_input("❓ 你的问题：", placeholder="例如：年假有几天？")

if question:
    with st.spinner("🤔 思考中..."):
        answer, docs = answer_question(question, db)

    st.divider()
    st.markdown(f"**回答：**\n\n{answer}")

    with st.expander("📄 查看引用来源（检索到的片段）"):
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**片段 {i}**")
            st.info(doc.page_content)
