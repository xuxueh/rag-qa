"""rag-qa 冒烟测试：构建知识库 + 问 3 个问题（非交互）"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_qa as rq

print("=" * 50)
print("构建混合检索器...（加载 GTE 模型，可能需要 1-2 分钟）")
from hybrid_retriever import build_hybrid
retriever = build_hybrid(rq.DOC_DIR, rq.EMBEDDING_MODEL_PATH)

llm = rq.ChatOpenAI(
    model="deepseek-chat",
    api_key=rq.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0,
)

questions = [
    "报销金额超过5000元需要谁审批？",
    "员工迟到超过30分钟怎么处理？",
    "公司附近有什么好吃的餐厅？",
]

for q in questions:
    print("\n" + "=" * 50)
    print(f"问：{q}")
    try:
        answer = rq.ask(q, retriever, llm)
        print(f"答：{answer}")
    except Exception as e:
        print(f"出错了：{type(e).__name__}: {e}")

print("\n" + "=" * 50)
print("测试完成")
