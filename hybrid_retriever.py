"""
hybrid_retriever.py — 混合检索（向量 + BM25，RRF 融合）· chunk_id 版
=====================================================================
为什么混合：单纯向量检索对「关键词/编号/金额」类问题不敏感（语义近但字面不匹配），
BM25 对「意思相近但换了说法」不敏感。两路各出候选，用 RRF 做「排名投票」融合。

关键设计：
- 每个 chunk 有稳定 chunk_id（d01_c03 形式），写进 Chroma 的 id 与 metadata
- 检索内部一律按 chunk_id 融合，避免「重复文本互相覆盖」
- 对外返回 ChunkMeta（chunk_id / text / source / articles / page），citation() 直接溯源
- 双路各取 Top-K 后融合（实验：比 BM25 全库参与更优，R@1 +4.6pp）

RRF 公式：score(d) = Σ_route 1 / (rrf_k + rank_route(d))
"""
from __future__ import annotations

import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import jieba
from rank_bm25 import BM25Okapi

import config

jieba.setLogLevel(logging.WARNING)
log = logging.getLogger(__name__)

_CN_NUMS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
            "十": 10, "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
            "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}
_CN_OUT = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}


def extract_articles(text: str) -> list[int]:
    """从 chunk 文本中提取所有条款号（第X条 → 数字列表，去重保序）"""
    found = []
    for m in re.finditer(r"第\s*([一二三四五六七八九十0-9]+)\s*条", text):
        num_str = m.group(1)
        if num_str in _CN_NUMS:
            found.append(_CN_NUMS[num_str])
    seen, out = set(), []
    for n in found:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _int_to_cn(n: int) -> str:
    return _CN_OUT.get(n, str(n))


def format_citation(filename: str, articles: list[int]) -> str:
    """'01-考勤管理制度.txt · 第三条'"""
    if not articles:
        return filename
    parts = "、".join(f"第{_int_to_cn(a)}条" if a <= 10 else f"第{a}条" for a in articles)
    return f"{filename} · {parts}"


class ChunkMeta:
    """chunk 结构化元数据：检索、重排、溯源全程用它，不再拿文本当 ID"""

    __slots__ = ("chunk_id", "text", "source", "articles", "page")

    def __init__(self, chunk_id: str, text: str, source: str, articles: list[int], page: str = ""):
        self.chunk_id = chunk_id
        self.text = text
        self.source = source
        self.articles = articles
        self.page = page

    def citation(self) -> str:
        """'文件 · 第X条'；PDF 有页码时追加页码（PyPDFLoader 的 page 是 0-based，展示转 1-based）"""
        base = format_citation(self.source, self.articles)
        if self.page != "":
            try:
                page_display = int(self.page) + 1
            except (ValueError, TypeError):
                page_display = self.page
            return f"{base} · 第{page_display}页"
        return base

    def __repr__(self):
        return f"<ChunkMeta {self.chunk_id} {self.source}>"


def _match_keyword(query: str) -> list[str]:
    """jieba 分词 + 去空白/标点，供 BM25 使用"""
    return [t for t in jieba.lcut(query) if t.strip()]


class HybridRetriever:
    """混合检索器：向量（Chroma）+ BM25（jieba），RRF 融合"""

    def __init__(self, db, chunks: list[ChunkMeta], rrf_k: int | None = None):
        if not chunks:
            raise ValueError("知识库为空：没有任何 chunk 可检索，请检查 KNOWLEDGE_DIR 与文档解析")
        self.db = db
        self.chunks = chunks
        self.rrf_k = rrf_k or config.RRF_K
        self.id2meta = {c.chunk_id: c for c in chunks}
        tokenized = [_match_keyword(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        log.info("检索器就绪：%d 个 chunk，%d 个文档，rrf_k=%d",
                 len(chunks), len({c.source for c in chunks}), self.rrf_k)

    # ── 单路检索 ──
    def _dense_ids(self, query: str, top_k: int) -> list[str]:
        docs = self.db.similarity_search(query, k=top_k)
        return [self._doc_id(d) for d in docs if self._doc_id(d) in self.id2meta]

    def _bm25_ids(self, query: str, top_k: int) -> list[str]:
        scores = self.bm25.get_scores(_match_keyword(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.chunks[i].chunk_id for i in order if scores[i] > 0]

    @staticmethod
    def _doc_id(doc) -> str:
        """Chroma 返回的 Document → chunk_id（metadata 优先，兼容老版本只有 id 的情况）"""
        return (doc.metadata or {}).get("chunk_id") or getattr(doc, "id", "") or ""

    # ── RRF 融合 ──
    def _rrf_ids(self, query: str, top_k: int, rrf_k: int | None = None) -> list[str]:
        rrf_k = rrf_k or self.rrf_k
        bm25_k = config.BM25_TOP_K or top_k   # 0 = 与召回池同宽
        vec_ranks = {cid: i + 1 for i, cid in enumerate(self._dense_ids(query, top_k))}
        bm25_ranks = {cid: i + 1 for i, cid in enumerate(self._bm25_ids(query, bm25_k))}

        rrf: dict[str, float] = {}
        for cid in set(vec_ranks) | set(bm25_ranks):
            score = 0.0
            if cid in vec_ranks:
                score += 1.0 / (rrf_k + vec_ranks[cid])
            if cid in bm25_ranks:
                score += 1.0 / (rrf_k + bm25_ranks[cid])
            rrf[cid] = score
        return [cid for cid, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]]

    # ── 对外接口 ──
    def retrieve_meta(self, query: str, top_k: int | None = None,
                      rrf_k: int | None = None) -> list[ChunkMeta]:
        """混合检索：返回 top_k 个 ChunkMeta（含溯源）"""
        top_k = top_k or config.RECALL_TOP_K
        if not (query or "").strip():
            return []
        return [self.id2meta[cid] for cid in self._rrf_ids(query, top_k, rrf_k)]

    def dense_meta(self, query: str, top_k: int | None = None) -> list[ChunkMeta]:
        """纯向量检索（评测基线用，不含 BM25）"""
        top_k = top_k or config.RECALL_TOP_K
        return [self.id2meta[cid] for cid in self._dense_ids(query, top_k)]

    def bm25_meta(self, query: str, top_k: int | None = None) -> list[ChunkMeta]:
        """纯 BM25（诊断用）"""
        top_k = top_k or config.RECALL_TOP_K
        return [self.id2meta[cid] for cid in self._bm25_ids(query, top_k)]

    def retrieve(self, query: str, top_k: int = 5, rrf_k: int | None = None) -> list[str]:
        """兼容版：返回文本列表（旧调用方）"""
        return [m.text for m in self.retrieve_meta(query, top_k, rrf_k)]

    def get_chunk(self, chunk_id: str) -> ChunkMeta | None:
        return self.id2meta.get(chunk_id)


def build_hybrid(doc_dir: str | None = None, embedding_model_path: str | None = None,
                 report: dict | None = None) -> HybridRetriever:
    """构建知识库：加载 → 清洗 → 句子边界切块 → chunk_id → 向量库 + BM25 索引

    report: 可选字典，回填 {"errors": [{"file","error"}], "chunks": n, "docs": n}
            —— 文档管理接口用它把「哪些文件没进来、为什么」回给调用方
    """
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from clean_pipeline import clean_document
    from smart_loader import load_documents_with_report

    doc_dir = str(doc_dir or config.KNOWLEDGE_DIR)
    embedding_model_path = embedding_model_path or config.resolve_embedding_model()

    documents, errors = load_documents_with_report(doc_dir)
    if report is not None:
        report["errors"] = errors
    if not documents:
        raise ValueError(
            f"知识库目录里没有解析成功的文档：{doc_dir}。"
            f"支持 {sorted(config.ALLOWED_EXT)}，请检查文件是否为空/加密/编码异常"
        )
    for doc in documents:
        doc.page_content = clean_document(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    # chunk_id：d<文档序号>_c<文档内块序号>
    chunk_metas: list[ChunkMeta] = []
    chunk_ids: list[str] = []
    doc_counter: dict[str, int] = {}
    chunk_counter: dict[str, int] = {}
    for c in chunks:
        src = c.metadata.get("source", "")
        fn = src.replace("\\", "/").rsplit("/", 1)[-1]
        if fn not in doc_counter:
            doc_counter[fn] = len(doc_counter) + 1
            chunk_counter[fn] = 0
        doc_no = doc_counter[fn]
        chunk_counter[fn] += 1
        cid = f"d{doc_no:02d}_c{chunk_counter[fn]:02d}"
        c.metadata["chunk_id"] = cid          # 写进 metadata，检索时按 id 还原 ChunkMeta
        chunk_ids.append(cid)
        chunk_metas.append(ChunkMeta(
            chunk_id=cid,
            text=c.page_content,
            source=fn,
            articles=extract_articles(c.page_content),
            page=str(c.metadata.get("page", "")),
        ))
    log.info("知识库构建：%d 份文档 → %d 个 chunk（size=%d, overlap=%d）",
             len(doc_counter), len(chunk_metas), config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
    db = Chroma.from_documents(chunks, embeddings, ids=chunk_ids)
    if report is not None:
        report["chunks"] = len(chunk_metas)
        report["docs"] = len(doc_counter)
    return HybridRetriever(db, chunk_metas)


if __name__ == "__main__":
    config.setup_logging()
    config.assert_valid(require_api_key=False)
    retriever = build_hybrid()
    for q in ["报销金额超过5000元需要谁审批？", "迟到30分钟怎么处理？", "婚假几天？"]:
        print(f"\n问题：{q}")
        for m in retriever.retrieve_meta(q, top_k=3):
            print(f"  {m.chunk_id} [{m.citation()}] {m.text[:40]}...")
