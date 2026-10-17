"""
企业知识库问答系统 (RAG) — 生产链路公共函数 + 命令行入口
=========================================================
链路（一条问题从进来到出去）：
  问题 → 混合检索(向量+BM25, RRF) 召回 top-10 → rerank 精排 top-3
       → 拼 prompt（带 [来源：文件 · 第X条]）→ DeepSeek 生成 → 答案 + 引用

本模块只负责「链路编排」和「prompt / 上下文组装」，
检索细节在 hybrid_retriever.py，重排在 rerank.py，配置在 config.py。
"""
from __future__ import annotations

import logging
import sys

# Windows 控制台默认 GBK，print ✓ / emoji 会报错；强制 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # 被重定向到管道/文件时可能不支持
    pass

import config

log = logging.getLogger(__name__)

# 兼容旧调用方（历史评测脚本用 rag_qa.DOC_DIR / rag_qa.EMBEDDING_MODEL_PATH）
DOC_DIR = config.DOC_DIR
EMBEDDING_MODEL_PATH = config.EMBEDDING_MODEL_PATH
DEEPSEEK_API_KEY = config.DEEPSEEK_API_KEY

# 防幻觉 prompt：先给依据、再给结论、找不到就说不知道
PROMPT_TEMPLATE = """根据以下资料回答问题。

回答要求：
1. 先说明依据：指出答案来自资料的哪份文件、第几条
2. 再给出结论
3. 引用内容不得超出资料范围
4. 逐条核对资料中的每个条款：只要有任何一条与问题相关，就必须用它回答，不要遗漏
5. 只有当资料中确实找不到任何相关信息时，才回答"不知道"，不要瞎编

资料：
{context}

问题：{question}

回答："""


def build_llm():
    """构造大模型客户端（密钥来自环境变量，缺失直接报错说明怎么修）"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        api_key=config.require_api_key(),
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
    )


def format_context(metas) -> tuple[str, list[dict]]:
    """把检索结果（ChunkMeta 列表）拼成 prompt 上下文 + 结构化 sources"""
    parts, sources = [], []
    for m in metas:
        parts.append(f"[来源：{m.citation()}]\n{m.text}")
        sources.append({"chunk_id": m.chunk_id, "text": m.text, "source": m.citation()})
    return "\n\n".join(parts), sources


def build_prompt(context: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(context=context, question=question)


def retrieve_and_rerank(retriever, question: str, recall_k: int | None = None,
                        top_n: int | None = None):
    """召回 → 精排（生产链路第 2、3 步）。返回 (ranked_metas, recall_metas, elapsed_ms)"""
    import time

    from rerank import rerank_objects

    recall_k = recall_k or config.RECALL_TOP_K
    top_n = top_n or config.RERANK_TOP_N
    t0 = time.perf_counter()
    recall = retriever.retrieve_meta(question, top_k=recall_k)
    t1 = time.perf_counter()
    ranked, meta = rerank_objects(question, recall, lambda m: m.text, top_n=top_n)
    t2 = time.perf_counter()
    log.debug(
        "检索完成: recall=%d rerank_top=%d rerank_fallback=%s recall耗时=%.0fms rerank耗时=%.0fms",
        len(recall), len(ranked), meta.get("fallback", False), (t1 - t0) * 1000, (t2 - t1) * 1000,
    )
    return ranked, recall, round((t2 - t0) * 1000, 1)


def answer(question: str, retriever=None, llm=None) -> dict:
    """完整问答：检索 → 精排 → prompt → 生成。返回 {answer, sources, context, timings}"""
    import time

    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空")

    retriever = retriever or get_default_retriever()
    llm = llm or build_llm()

    ranked, _, retrieval_ms = retrieve_and_rerank(retriever, question)
    context, sources = format_context(ranked)
    if not context:
        # 知识库为空 / 检索全空：不调用模型，直接给可解释的空答案
        return {"answer": "知识库中没有检索到相关内容。", "sources": [],
                "context": "", "timings": {"retrieval_ms": retrieval_ms, "llm_ms": 0.0}}

    prompt = build_prompt(context, question)
    t0 = time.perf_counter()
    result = llm.invoke(prompt)
    llm_ms = round((time.perf_counter() - t0) * 1000, 1)
    log.info("问答完成: %.1fs（检索 %.0fms / 生成 %.0fms）", (retrieval_ms + llm_ms) / 1000,
             retrieval_ms, llm_ms)
    return {"answer": result.content, "sources": sources, "context": context,
            "timings": {"retrieval_ms": retrieval_ms, "llm_ms": llm_ms}}


# ═══════════════ 默认检索器（CLI/脚本共用，构建一次复用） ═══════════════
_default_retriever = None


def get_default_retriever():
    """构建或复用全局检索器（首次约 1-2 分钟：加载 embedding + 建向量库 + BM25）"""
    global _default_retriever
    if _default_retriever is None:
        from hybrid_retriever import build_hybrid

        config.assert_valid(require_api_key=False)
        log.info("构建检索器（首次约 1-2 分钟）...")
        _default_retriever = build_hybrid(str(config.KNOWLEDGE_DIR), config.EMBEDDING_MODEL)
        log.info("检索器就绪")
    return _default_retriever


def build_knowledge_base(doc_dir, embedding_model_path):
    """[DEPRECATED] 旧构建入口——只返回纯向量库，无 BM25 / Citation。

    生产链路与评测请统一用 hybrid_retriever.build_hybrid()（含 chunk_id + BM25 + 溯源）。
    保留仅供早期脚本兼容，勿在新代码中使用。
    """
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from clean_pipeline import clean_document
    from smart_loader import load_documents

    documents = load_documents(doc_dir)
    for doc in documents:
        doc.page_content = clean_document(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
    return Chroma.from_documents(chunks, embeddings), embeddings


def ask(question: str, retriever, llm, k: int | None = None) -> str:
    """兼容版：只返回答案文本（旧脚本/旧界面用）"""
    return answer(question, retriever=retriever, llm=llm)["answer"]


def main():
    config.setup_logging()
    config.assert_valid()

    retriever = get_default_retriever()
    llm = build_llm()

    print("\n" + "=" * 44)
    print("知识库问答系统启动！输入问题，q 退出")
    print(f"知识库：{config.KNOWLEDGE_DIR}")
    print("=" * 44)
    while True:
        try:
            question = input("\n你问：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("q", "quit", "退出"):
            break
        if not question:
            continue
        try:
            result = answer(question, retriever=retriever, llm=llm)
        except Exception as exc:  # 单次问答失败不退出程序
            log.error("问答失败：%s", exc)
            print(f"\n❌ 出错了：{exc}")
            continue
        print(f"\n回答：{result['answer']}")
        if result["sources"]:
            print("\n引用：")
            for s in result["sources"]:
                print(f"  - {s['source']}")


if __name__ == "__main__":
    main()
