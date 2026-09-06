"""
hybrid_retriever.py - 混合检索（BM25 + 向量，RRF 融合）
=========================================================
RAG 升级 Phase 2：向量管语义，BM25 管精确词（编号/专有名词），RRF 融合两者排序。

- BM25Okapi：关键词检索（jieba 中文分词）
- Chroma：语义向量检索
- RRF（Reciprocal Rank Fusion）：score = Σ 1/(k + rank)，k 通常取 60
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import jieba
from rank_bm25 import BM25Okapi

import rag_qa as rq

# jieba 首次加载会打印日志，静音
jieba.setLogLevel(20)

_CN_NUMS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
            "十": 10, "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
            "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}


def extract_articles(text: str) -> list[int]:
    """从 chunk 文本中提取所有条款号（第X条 → 数字列表）"""
    import re
    found = []
    for m in re.finditer(r"第\s*([一二三四五六七八九十0-9]+)\s*条", text):
        num_str = m.group(1)
        if num_str in _CN_NUMS:
            found.append(_CN_NUMS[num_str])
    # 去重保序
    seen, out = set(), []
    for n in found:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def format_citation(filename: str, articles: list[int]) -> str:
    """格式化引用：'01-考勤管理制度.txt · 第三条'"""
    if not articles:
        return filename
    parts = "、".join(f"第{_int_to_cn(a)}条" if a <= 10 else f"第{a}条" for a in articles)
    return f"{filename} · {parts}"


def _int_to_cn(n: int) -> str:
    cn = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}
    return cn.get(n, str(n))


class HybridRetriever:
    """混合检索器：向量检索 + BM25 关键词检索，RRF 融合排序"""

    def __init__(self, db, chunk_texts: list[str], text2src: dict | None = None,
                 text2articles: dict | None = None):
        self.db = db
        self.chunk_texts = chunk_texts  # 与 db 中的块一一对应
        self.text2src = text2src or {}  # 块文本 → 来源文件名（溯源用）
        self.text2articles = text2articles or {}  # 块文本 → 条款号列表（Citation）
        # 构建 BM25 索引（jieba 分词）
        tokenized = [jieba.lcut(t) for t in chunk_texts]
        self.bm25 = BM25Okapi(tokenized)

    def get_source(self, text: str) -> str:
        """返回某块文本的来源文件名（溯源）"""
        return self.text2src.get(text, "未知")

    def get_citation(self, text: str) -> str:
        """返回完整引用：'01-考勤管理制度.txt · 第三条'（Citation）"""
        fn = self.text2src.get(text, "未知")
        articles = self.text2articles.get(text, [])
        return format_citation(fn, articles)

    def retrieve(self, query: str, top_k: int = 5, rrf_k: int = 60) -> list[str]:
        """混合检索：返回 top_k 个文档文本（RRF 融合排序）"""
        # ① 向量检索（多召回一些，保证融合池子够大）
        vec_docs = self.db.similarity_search(query, k=top_k * 2)
        vec_ranks = {d.page_content: i + 1 for i, d in enumerate(vec_docs)}

        # ② BM25 检索（对全部块打分）
        bm25_scores = self.bm25.get_scores(jieba.lcut(query))
        bm25_order = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_ranks = {self.chunk_texts[i]: rank + 1 for rank, i in enumerate(bm25_order)}

        # ③ RRF 融合
        all_docs = set(vec_ranks.keys()) | set(bm25_ranks.keys())
        rrf_scores: dict[str, float] = {}
        for doc in all_docs:
            score = 0.0
            if doc in vec_ranks:
                score += 1.0 / (rrf_k + vec_ranks[doc])
            if doc in bm25_ranks:
                score += 1.0 / (rrf_k + bm25_ranks[doc])
            rrf_scores[doc] = score

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]


def build_hybrid(doc_dir: str, embedding_model_path: str) -> HybridRetriever:
    """构建知识库 + BM25 索引，返回 HybridRetriever"""
    # 复用 rag_qa 的加载/切块逻辑
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    loader = DirectoryLoader(doc_dir, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()
    print(f"✓ 加载文档: {len(documents)} 份")

    # 文本清洗（去空白/统一换行/去重；对 PDF 等脏文本有效，对干净文档无影响）
    from clean_pipeline import clean_document
    for doc in documents:
        doc.page_content = clean_document(doc.page_content)

    # 切块：按中文句子边界递归切（对长文档更鲁棒；评测 47 块与硬切无差异）
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        chunk_size=200,
        chunk_overlap=20,
    )
    chunks = splitter.split_documents(documents)
    print(f"✓ 切块: {len(chunks)} 块")

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
    db = Chroma.from_documents(chunks, embeddings)
    print("✓ 向量库构建完成")

    chunk_texts = [c.page_content for c in chunks]  # 与 db 块顺序一致

    # 构建 文本 → 来源文件名 + 条款号 映射（Citation 溯源用）
    text2src = {}
    text2articles = {}
    for c in chunks:
        src = c.metadata.get("source", "")
        fn = src.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        text2src[c.page_content] = fn
        text2articles[c.page_content] = extract_articles(c.page_content)

    print("✓ BM25 索引构建完成")
    return HybridRetriever(db, chunk_texts, text2src, text2articles)


if __name__ == "__main__":
    retriever = build_hybrid(rq.DOC_DIR, rq.EMBEDDING_MODEL_PATH)
    for q in ["报销金额超过5000元需要谁审批？", "迟到30分钟怎么处理？", "婚假几天？"]:
        top = retriever.retrieve(q, top_k=3)
        print(f"\n问题：{q}")
        for i, t in enumerate(top):
            print(f"  {i + 1}. {t[:45]}...")

# ── 集成说明 ────────────────────────────────────────────
# from hybrid_retriever import build_hybrid
# retriever = build_hybrid(DOC_DIR, EMBEDDING_MODEL_PATH)
# texts = retriever.retrieve(question, top_k=3)  # 替代 db.similarity_search
# 然后可再接 rerank 精排（可选）
