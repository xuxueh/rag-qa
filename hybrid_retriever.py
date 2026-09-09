"""
hybrid_retriever.py - 混合检索（BM25 + 向量，RRF 融合）· chunk_id 版
=====================================================================
v2 重构（GPT 评审 ③）：用 chunk_id 作为唯一标识，不再用 page_content 当 ID。
- 每个 chunk 拥有稳定 chunk_id（d<文档>_c<块>），Chroma 以 id 存储
- 检索内部按 chunk_id 融合（重复文本的两个 chunk 互不覆盖）
- 生产链路用 retrieve_meta() 拿结构化结果（chunk_id/text/source/citation/page）

- BM25Okapi：关键词检索（jieba 中文分词）
- Chroma：语义向量检索（id = chunk_id）
- RRF：score = Σ 1/(k + rank)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import jieba
from rank_bm25 import BM25Okapi

import rag_qa as rq

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
    seen, out = set(), []
    for n in found:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _int_to_cn(n: int) -> str:
    cn = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}
    return cn.get(n, str(n))


def format_citation(filename: str, articles: list[int]) -> str:
    """格式化引用：'01-考勤管理制度.txt · 第三条'"""
    if not articles:
        return filename
    parts = "、".join(f"第{_int_to_cn(a)}条" if a <= 10 else f"第{a}条" for a in articles)
    return f"{filename} · {parts}"


class ChunkMeta:
    """chunk 结构化元数据（评审 ③：替代文本作 ID 的隐患）"""

    __slots__ = ("chunk_id", "text", "source", "articles", "page")

    def __init__(self, chunk_id: str, text: str, source: str, articles: list[int], page: str = ""):
        self.chunk_id = chunk_id
        self.text = text
        self.source = source
        self.articles = articles
        self.page = page

    def citation(self) -> str:
        """'文件 · 第X条'（有页码则含页码）"""
        base = format_citation(self.source, self.articles)
        if self.page:
            return f"{base} · 第{self.page}页"
        return base

    def __repr__(self):
        return f"<ChunkMeta {self.chunk_id} {self.source}>"


class HybridRetriever:
    """混合检索器（chunk_id 索引）：向量检索 + BM25 关键词，RRF 融合"""

    def __init__(self, db, chunks: list[ChunkMeta]):
        self.db = db
        self.chunks = chunks  # list[ChunkMeta]，与 Chroma 存储顺序一致（id = chunk_id）
        self.id2meta = {c.chunk_id: c for c in chunks}
        # BM25 索引（按 chunks 顺序分词；id 平行映射）
        tokenized = [jieba.lcut(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    # ── 内部：RRF 融合（返回排序后的 chunk_id 列表）──
    def _rrf_ids(self, query: str, top_k: int, rrf_k: int = 60) -> list[str]:
        # ① 向量检索（Chroma 存 id=chunk_id，d.id 即稳定标识，重复文本不冲突）
        vec_docs = self.db.similarity_search(query, k=top_k * 2)
        vec_ranks = {d.id: i + 1 for i, d in enumerate(vec_docs)}

        # ② BM25 全库打分（id 按平行顺序映射）
        bm25_scores = self.bm25.get_scores(jieba.lcut(query))
        bm25_order = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_ranks = {self.chunks[i].chunk_id: rank + 1 for rank, i in enumerate(bm25_order)}

        # ③ RRF
        all_ids = set(vec_ranks) | set(bm25_ranks)
        rrf: dict[str, float] = {}
        for cid in all_ids:
            s = 0.0
            if cid in vec_ranks:
                s += 1.0 / (rrf_k + vec_ranks[cid])
            if cid in bm25_ranks:
                s += 1.0 / (rrf_k + bm25_ranks[cid])
            rrf[cid] = s
        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:top_k]]

    def retrieve(self, query: str, top_k: int = 5, rrf_k: int = 60) -> list[str]:
        """兼容版：返回 top_k 个文档文本（旧调用方/评测脚本）"""
        ids = self._rrf_ids(query, top_k, rrf_k)
        return [self.id2meta[cid].text for cid in ids]

    def retrieve_meta(self, query: str, top_k: int = 5, rrf_k: int = 60) -> list[ChunkMeta]:
        """生产链路：返回 top_k 个 ChunkMeta（chunk_id + 溯源 + citation）"""
        ids = self._rrf_ids(query, top_k, rrf_k)
        return [self.id2meta[cid] for cid in ids]

    def get_chunk(self, chunk_id: str) -> ChunkMeta:
        return self.id2meta.get(chunk_id)


def build_hybrid(doc_dir: str, embedding_model_path: str) -> HybridRetriever:
    """构建知识库（chunk_id 化）→ 返回 HybridRetriever"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from smart_loader import load_documents
    from clean_pipeline import clean_document

    documents = load_documents(doc_dir)
    print(f"✓ 加载文档: {len(documents)} 份")
    for doc in documents:
        doc.page_content = clean_document(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(documents)
    print(f"✓ 切块: {len(chunks)} 块")

    # 生成 chunk_id + 结构化元数据
    chunk_metas = []
    chunk_ids = []
    doc_counter = {}
    for i, c in enumerate(chunks):
        src = c.metadata.get("source", "")
        fn = src.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        # doc 序号：按文件名首次出现顺序
        if fn not in doc_counter:
            doc_counter[fn] = len(doc_counter) + 1
        doc_no = doc_counter[fn]
        chunk_no = sum(1 for m in chunk_metas if m.source == fn) + 1
        cid = f"d{doc_no:02d}_c{chunk_no:02d}"
        chunk_ids.append(cid)
        chunk_metas.append(ChunkMeta(
            chunk_id=cid,
            text=c.page_content,
            source=fn,
            articles=extract_articles(c.page_content),
            page=str(c.metadata.get("page", "")),
        ))
    print(f"✓ chunk_id 生成: {len(chunk_metas)} 块（{len(doc_counter)} 个文档）")

    # 向量库（以 chunk_id 为 Chroma id——杜绝文本冲突）
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
    db = Chroma.from_documents(chunks, embeddings, ids=chunk_ids)
    print("✓ 向量库构建完成（id=chunk_id）")

    print("✓ BM25 索引构建完成")
    return HybridRetriever(db, chunk_metas)


if __name__ == "__main__":
    retriever = build_hybrid(rq.DOC_DIR, rq.EMBEDDING_MODEL_PATH)
    for q in ["报销金额超过5000元需要谁审批？", "迟到30分钟怎么处理？", "婚假几天？"]:
        metas = retriever.retrieve_meta(q, top_k=3)
        print(f"\n问题：{q}")
        for m in metas:
            print(f"  {m.chunk_id} [{m.citation()}] {m.text[:40]}...")
