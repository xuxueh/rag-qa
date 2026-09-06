"""
企业知识库问答系统 (RAG)
========================
基于检索增强生成 (RAG) 的智能问答系统：
文档加载 → 文本切块 → 向量化 → 语义检索 → 大模型生成

技术栈: LangChain, Chroma, GTE中文嵌入, DeepSeek API
"""

import os
import sys

# Windows 控制台默认 GBK 编码，print ✓ 等字符会报错；强制 UTF-8（Day 2 学过的坑）
sys.stdout.reconfigure(encoding="utf-8")


def load_env(path):
    """加载 .env 文件到环境变量（不依赖 python-dotenv）"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()
    except FileNotFoundError:
        pass


load_env(os.path.join(os.path.dirname(__file__), ".env"))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
# ⚠️ Windows DLL 加载顺序：TextLoader 必须在这三个之后导入
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter

# ═══════════════ 配置区（从 .env 读取） ═══════════════
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-你的key")
DOC_DIR = r"C:\Users\xujin\ai-learning\项目\rag-qa\知识库"   # 知识库文档目录（12 份公司制度）
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL", "")  # 从 .env 读 gte-small 路径
# ══════════════════════════════════════════════════════


def build_knowledge_base(doc_dir, embedding_model_path):
    """构建向量知识库：加载目录全部文档 → 切块 → 嵌入 → 存储"""
    # 1. 加载目录下所有文档
    loader = DirectoryLoader(doc_dir, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()
    print(f"✓ 加载文档: {len(documents)} 份")

    # 文本清洗（对干净文档无影响，对未来 PDF/脏文本有效）
    from clean_pipeline import clean_document
    for doc in documents:
        doc.page_content = clean_document(doc.page_content)

    # 2. 切块（200字/块，重叠20字；按中文句子边界递归切）
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        chunk_size=200,
        chunk_overlap=20,
    )
    chunks = splitter.split_documents(documents)
    print(f"✓ 切块: {len(chunks)} 块")

    # 3. 嵌入 + 4. 存储
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
    db = Chroma.from_documents(chunks, embeddings)
    print("✓ 向量知识库构建完成")
    return db, embeddings


def ask(question, retriever, llm, k=3):
    """RAG 问答：混合检索（召回 top-10）→ rerank 精排 → 拼接 prompt → 大模型生成"""
    # 1. 混合检索召回 top-10（向量 + BM25，先找得多）
    # 1. 混合检索召回 top-5（评测支撑：Top-5 命中 100% → 正确答案必在前 5，rerank 5 候选即可）
    recall = retriever.retrieve(question, top_k=5)

    # 2. rerank 精排，取 top-k（再排得准）
    from rerank import rerank
    ranked = rerank(question, recall, top_n=k)

    # 3. 组装上下文（用 retriever 溯源：块文本 → 来源文件名）
    parts = []
    for t in ranked:
        parts.append(f"[来源：{retriever.get_source(t)}]\n{t}")
    context = "\n\n".join(parts)

    # 2. 拼接 prompt（资料 + 问题）
    prompt = f"""根据以下资料回答问题。如果资料里没有答案，就说不知道，不要瞎编。

回答要求：
1. 先说明依据：指出答案来自资料的哪份文件、第几条
2. 再给出结论
3. 引用内容不得超出资料范围

资料：
{context}

问题：{question}

回答："""

    # 3. 大模型生成
    result = llm.invoke(prompt)
    return result.content


def main():
    # 构建混合检索器（向量 + BM25，含溯源）
    from hybrid_retriever import build_hybrid
    retriever = build_hybrid(DOC_DIR, EMBEDDING_MODEL_PATH)

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
        answer = ask(question, retriever, llm)
        print(f"\n回答：{answer}")


if __name__ == "__main__":
    main()
