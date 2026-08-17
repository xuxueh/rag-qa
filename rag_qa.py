"""
企业知识库问答系统 (RAG)
========================
基于检索增强生成 (RAG) 的智能问答系统：
文档加载 → 文本切块 → 向量化 → 语义检索 → 大模型生成

技术栈: LangChain, Chroma, GTE中文嵌入, DeepSeek API
"""

import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

# ═══════════════ 配置区（改成你自己的） ═══════════════
DEEPSEEK_API_KEY = "sk-你的key"          # ← 填你的 DeepSeek API key
DOC_PATH = "公司制度.txt"                 # 知识库文档
EMBEDDING_MODEL_PATH = r"C:\Users\xujin\ai-learning\data\models\gte-embedding\models\iic--nlp_gte_sentence-embedding_chinese-base\snapshots\master"
# ══════════════════════════════════════════════════════


def build_knowledge_base(doc_path, embedding_model_path):
    """构建向量知识库：加载 → 切块 → 嵌入 → 存储"""
    # 1. 加载文档
    loader = TextLoader(doc_path, encoding='utf-8')
    documents = loader.load()
    print(f"✓ 加载文档: {len(documents)} 份")

    # 2. 切块（200字/块，重叠20字保持上下文连贯）
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(documents)
    print(f"✓ 切块: {len(chunks)} 块")

    # 3. 嵌入 + 4. 存储
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
    db = Chroma.from_documents(chunks, embeddings)
    print("✓ 向量知识库构建完成")
    return db, embeddings


def ask(question, db, llm, k=3):
    """RAG 问答：检索最相关片段 → 拼接 prompt → 大模型生成"""
    # 1. 语义检索最相关的 k 块
    docs = db.similarity_search(question, k=k)
    context = "\n\n".join([d.page_content for d in docs])

    # 2. 拼接 prompt（资料 + 问题）
    prompt = f"""根据以下资料回答问题。如果资料里没有答案，就说不知道，不要瞎编。

资料：
{context}

问题：{question}

回答："""

    # 3. 大模型生成
    result = llm.invoke(prompt)
    return result.content


def main():
    # 构建知识库
    db, _ = build_knowledge_base(DOC_PATH, EMBEDDING_MODEL_PATH)

    # 配置大模型
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=0,
    )

    # 命令行问答
    print("\n" + "=" * 40)
    print("知识库问答系统启动！输入问题，q 退出")
    print("=" * 40)
    while True:
        question = input("\n你问：").strip()
        if question.lower() in ("q", "quit", "退出"):
            break
        if not question:
            continue
        answer = ask(question, db, llm)
        print(f"\n回答：{answer}")


if __name__ == "__main__":
    main()
