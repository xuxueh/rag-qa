"""
📚 企业知识库问答系统（Streamlit 网页版 · 生产链路）
====================================================
支持本地与云端：
- 本地：混合检索 + bge-reranker 精排（生产链路完整版）
- 云端（Streamlit Cloud）：rerank 模型加载失败时自动降级（返回召回结果）

运行: streamlit run rag_app.py
"""

import os
import sys
from pathlib import Path

# Windows 控制台 UTF-8
sys.stdout.reconfigure(encoding="utf-8")

import streamlit as st

# ═══════════════ 配置 ═══════════════
ENV_FILE = Path(__file__).parent / ".env"
# 默认知识库目录（12 份公司制度）；云端可用 Secrets 覆盖
DEFAULT_DOC_DIR = r"C:\Users\xujin\ai-learning\项目\rag-qa\知识库"
# 嵌入模型：.env 的 EMBEDDING_MODEL（本地路径）优先，否则 HF 模型名（云端自动下载）
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# ═══════════════════════════════════

EXAMPLE_QUESTIONS = [
    "报销金额超过 5000 元需要谁审批？",
    "迟到超过 30 分钟怎么处理？",
    "婚假有几天？",
    "加班怎么调休？",
    "公司附近有什么好吃的餐厅？",
]


def load_key():
    """读取 key：优先 Streamlit Secrets（云端），其次 .env（本地）"""
    try:
        if "DEEPSEEK_API_KEY" in st.secrets:
            return st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        pass
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
    """嵌入模型：.env 本地路径优先，否则 HF 模型名"""
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
try:
    DOC_DIR = st.secrets.get("DOC_DIR", DEFAULT_DOC_DIR)
except Exception:
    DOC_DIR = DEFAULT_DOC_DIR  # 无 secrets.toml 时用默认知识库目录

st.set_page_config(page_title="智能知识库问答", page_icon="🤖")

st.title("🤖 智能知识库问答")
st.caption("生产链路：混合检索（BM25+向量）→ Rerank 精排 → DeepSeek")


@st.cache_resource
def build_retriever():
    """构建混合检索器（向量 + BM25 + 溯源）"""
    from hybrid_retriever import build_hybrid
    return build_hybrid(DOC_DIR, EMBEDDING_MODEL)


def answer_question(question, retriever):
    """RAG 问答（生产链路），返回 (回答, 来源列表)"""
    # 1. 混合检索召回 top-5（评测支撑：Top-5 命中 100%，rerank 5 候选精度无损）
    recall = retriever.retrieve(question, top_k=5)

    # 2. rerank 精排 top-3（模型不可用时自动降级返回原顺序）
    from rerank import rerank
    ranked = rerank(question, recall, top_n=3)

    # 3. 组装上下文（带来源）
    parts, sources = [], []
    for t in ranked:
        fn = retriever.get_source(t)
        parts.append(f"[来源：{fn}]\n{t}")
        sources.append({"text": t, "source": fn})
    context = "\n\n".join(parts)

    # 4. DeepSeek 生成
    from langchain_openai import ChatOpenAI
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

问题：{question}

回答："""
    answer = llm.invoke(prompt).content
    return answer, sources


# ─── 初始化 ───
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ─── 侧边栏 ───
with st.sidebar:
    st.header("📋 知识库")
    with st.spinner("构建混合检索器（首次约 1-2 分钟）..."):
        try:
            retriever = build_retriever()
            st.success("✅ 生产链路就绪（混合检索 + rerank）")
        except Exception as e:
            st.error(f"❌ 构建失败：{e}")
            st.stop()

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
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 引用来源"):
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(f"**来源 {i}**：`{s['source']}`")
                    st.info(s["text"])

# ─── 输入 ───
user_input = st.chat_input("输入你的问题...")
question = selected if selected else user_input

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state["messages"].append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("🤔 思考中（首次加载 rerank 模型约 30 秒）..."):
            try:
                answer, sources = answer_question(question, retriever)
            except Exception as e:
                answer = f"❌ 出错了：{type(e).__name__}: {e}"
                sources = []
        st.write(answer)
        if sources:
            with st.expander("📄 引用来源"):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"**来源 {i}**：`{s['source']}`")
                    st.info(s["text"])

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
