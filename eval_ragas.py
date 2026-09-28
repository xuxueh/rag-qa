"""Ragas 深度评测（评审：Faithfulness / Answer Relevancy / Context Precision）
方案：DeepSeek 作 judge（LangchainLLMWrapper）+ 本地 GTE 作 embeddings

指标：
- faithfulness：答案是否忠于检索到的上下文（幻觉检测）
- answer_relevancy：答案与问题的相关度
- context_precision：检索上下文的精确度

用法: python eval_ragas.py [条数，默认8]
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import rag_qa as rq
from eval_common import build_eval_retriever
from rerank import rerank_objects

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8

print("构建检索器...")
retriever = build_eval_retriever()
llm = ChatOpenAI(model="deepseek-chat", api_key=rq.DEEPSEEK_API_KEY,
                 base_url="https://api.deepseek.com", temperature=0)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set_v2.json"), encoding="utf-8") as f:
    eval_set = json.load(f)
# 取前 N 条有答案的问题
items = [i for i in eval_set if i["category"] != "无答案问题"][:N]

print(f"跑 {len(items)} 条完整问答，收集 answer + contexts...")
samples = []
for idx, item in enumerate(items, 1):
    q = item["question"]
    recall = retriever.retrieve_meta(q, top_k=10)
    ranked, _ = rerank_objects(q, recall, lambda m: m.text, top_n=3)
    contexts = [m.text for m in ranked]

    parts = [f"[来源：{m.citation()}]\n{m.text}" for m in ranked]
    prompt = f"""根据以下资料回答问题。

回答要求：
1. 先说明依据：指出答案来自资料的哪份文件、第几条
2. 再给出结论
3. 引用内容不得超出资料范围
4. 逐条核对资料中的每个条款：只要有任何一条与问题相关，就必须用它回答，不要遗漏
5. 只有当资料中确实找不到任何相关信息时，才回答"不知道"，不要瞎编

资料：
{chr(10).join(chr(10).join(parts).split(chr(10)))}

问题：{q}

回答："""
    answer = llm.invoke(prompt).content
    samples.append({
        "user_input": q,
        "response": answer,
        "retrieved_contexts": contexts,
        "reference": item["answer"],
    })
    print(f"  [{idx}/{len(items)}] {q[:22]}...")

print("\n运行 Ragas 评测（DeepSeek judge + 本地 GTE embeddings）...")
dataset = EvaluationDataset.from_list(samples)

judge = LangchainLLMWrapper(ChatOpenAI(
    model="deepseek-chat", api_key=rq.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com", temperature=0))
emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=rq.EMBEDDING_MODEL_PATH))

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=judge,
    embeddings=emb,
)

print("\n" + "=" * 56)
print("Ragas 评测结果（0-1，越高越好）")
print("=" * 56)
print(result)
df = result.to_pandas()
print("\n各条明细:")
print(df.to_string())
df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ragas_result.csv"),
          index=False, encoding="utf-8-sig")
print("\n明细已保存: ragas_result.csv")
