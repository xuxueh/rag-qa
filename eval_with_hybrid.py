"""混合检索评测：Top-1/3/5 命中率（对比基线 75/95/100 和 rerank 95/100/100）
流程：构建向量库 + BM25 → RRF 融合检索 → 检查命中
用法：python eval_with_hybrid.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import rag_qa as rq
from hybrid_retriever import build_hybrid

# 1. 构建混合检索器（向量 + BM25）
print("构建混合检索器...（加载 GTE 模型，约 1-2 分钟）")
retriever = build_hybrid(rq.DOC_DIR, rq.EMBEDDING_MODEL_PATH)

# 2. 重建 chunks 拿 text → 来源文件 映射（用于命中判断）
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = DirectoryLoader(rq.DOC_DIR, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)
text2src = {}
for c in chunks:
    src = c.metadata.get("source", "")
    fn = src.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    text2src[c.page_content] = fn

# 3. 加载测试集
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json"), encoding="utf-8") as f:
    eval_set = json.load(f)


def hit_at_k(texts, target_file, k):
    for t in texts[:k]:
        if text2src.get(t, "") == target_file:
            return True
    return False


# 4. 跑评测
hits = {"top1": 0, "top3": 0, "top5": 0}
for item in eval_set:
    q = item["question"]
    target = item["source"]
    top = retriever.retrieve(q, top_k=5)
    if hit_at_k(top, target, 1):
        hits["top1"] += 1
    if hit_at_k(top, target, 3):
        hits["top3"] += 1
    if hit_at_k(top, target, 5):
        hits["top5"] += 1

total = len(eval_set)
print("=" * 50)
print(f"评测集：{total} 条 | 流程：向量 + BM25 → RRF 融合 → top-5")
print(f"Top-1 命中率：{hits['top1']/total:.1%}（{hits['top1']}/{total}）")
print(f"Top-3 命中率：{hits['top3']/total:.1%}（{hits['top3']}/{total}）")
print(f"Top-5 命中率：{hits['top5']/total:.1%}（{hits['top5']}/{total}）")
print("=" * 50)
print("\n对比：")
print("  基线（纯向量）    Top-1 75.0% | Top-3 95.0% | Top-5 100.0%")
print("  Rerank            Top-1 95.0% | Top-3 100.0% | Top-5 100.0%")
print(f"  混合检索（本结果） Top-1 {hits['top1']/total:.1%} | Top-3 {hits['top3']/total:.1%} | Top-5 {hits['top5']/total:.1%}")
