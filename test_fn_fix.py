"""验证 prompt 改进是否修复 3 条真漏答"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import rag_qa as rq
from eval_common import build_eval_retriever

retriever = build_eval_retriever()
llm = rq.ChatOpenAI(model="deepseek-chat", api_key=rq.DEEPSEEK_API_KEY,
                    base_url="https://api.deepseek.com", temperature=0)

CASES = [
    "员工多久有一次免费体检？",
    "公司有什么福利？",
    "调休会影响全勤奖吗？",
]

for q in CASES:
    print("\n" + "=" * 60)
    print(f"问：{q}")
    answer = rq.ask(q, retriever, llm)
    print(f"答：{answer[:220]}")
