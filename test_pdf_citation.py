"""验证 ④ PDF 页码 Citation：PDF 加载 → chunk 带 page metadata → citation 显示页码"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from hybrid_retriever import build_hybrid

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_docs")
if not os.path.exists(TEST_DIR):
    from make_test_docs import make_test_docs
    make_test_docs(TEST_DIR)

# 用混合检索器构建测试目录（含 txt/md/pdf/docx）
retriever = build_hybrid(TEST_DIR, r"C:\Users\xujin\ai-learning\data\models\gte-small\models\iic--nlp_gte_sentence-embedding_chinese-small\snapshots\master")

print("\n=== 各 chunk 的 citation（看 PDF 是否带页码）===")
for m in retriever.chunks:
    print(f"  {m.chunk_id} | page='{m.page}' | citation='{m.citation()}' | {m.text[:30]}...")

print("\n=== 检索测试 ===")
for q in ["测试制度", "PDF 文档", "Word 文档"]:
    metas = retriever.retrieve_meta(q, top_k=2)
    print(f"\n问题：{q}")
    for m in metas:
        print(f"  {m.chunk_id} [{m.citation()}] {m.text[:35]}...")
