# 📚 企业知识库问答系统（RAG）

基于**检索增强生成（RAG）**的智能问答系统。让大模型能回答企业私有文档的问题，并且**有据可查、不瞎编**。

## ✨ 功能

- 企业知识库问答（12 份公司制度，支持自定义文档）
- **混合检索**：向量语义检索 + BM25 关键词检索（RRF 融合）
- **Reranker 精排**：本地 bge-reranker-v2-m3 对召回结果精排
- 答案带**来源溯源**（标注来自哪份文件）
- 防幻觉：文档里没有答案时，模型会老实说「不知道」
- **评测体系**：20 条 QA 测试集 + Top-1/3/5 命中率评测脚本

## 🧠 原理（生产链路）

```
文档 → 清洗 → ① 句子边界切块 → ② 向量化(GTE) + BM25 索引 → ③ 混合检索(RRF) → ④ rerank 精排 → ⑤ DeepSeek 生成
```

1. **文本清洗**：统一换行、去空白/全角空格/重复段（`clean_pipeline.py`）
2. **切块**：按中文句子边界递归切块（200 字/块，重叠 20）
3. **召回**：GTE 语义向量 + BM25 关键词（jieba 分词），RRF 融合（`hybrid_retriever.py`）
4. **精排**：bge-reranker-v2-m3 对 top-10 候选打分，取 top-3（`rerank.py`）
5. **生成**：DeepSeek 基于检索片段回答（防幻觉 prompt + 来源标注）

## 📊 评测结果（20 条 QA，Top-K 检索命中率）

| 链路 | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| 基线（纯向量） | 75.0% | 95.0% | 100.0% |
| 混合检索（BM25+向量） | 90.0% | 100.0% | 100.0% |
| Rerank 精排 | 95.0% | 100.0% | 100.0% |
| **混合检索 + Rerank（生产链路）** | **95.0%** | **100.0%** | **100.0%** |

> 从基线到生产链路：**Top-1 提升 20 个百分点**（75% → 95%）。
> 查询改写（Multi-Query）已实现并评测：对标准表述问题无提升（Top-1 持平 75%），故未纳入主链路——模糊查询场景可按需启用。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型（ModelScope）

```python
from modelscope import snapshot_download
# GTE 中文嵌入模型
snapshot_download('iic/nlp_gte_sentence-embedding_chinese-small',
                  local_dir='data/models/gte-small')
# bge-reranker 重排模型
snapshot_download('BAAI/bge-reranker-v2-m3',
                  local_dir='data/models/bge-reranker-v2-m3')
```

### 3. 配置

创建 `.env`（已被 .gitignore 忽略，不会上传）：

```bash
DEEPSEEK_API_KEY=sk-你的key
EMBEDDING_MODEL=你的GTE模型本地路径
```

在 `rag_qa.py` 中修改 `DOC_DIR` 为你的知识库目录（.txt）。

### 4. 运行

```bash
# 命令行问答（混合检索 + rerank 生产链路）
python rag_qa.py

# 评测（对比各链路效果）
python eval_baseline.py          # 纯向量基线
python eval_with_rerank.py       # rerank 效果
python eval_with_hybrid.py       # 混合检索效果
python eval_hybrid_rerank.py     # 生产链路效果
```

## 📸 效果演示

```
知识库问答系统启动！输入问题，q 退出

问题：报销金额超过 5000 元需要谁审批？
回答：依据：[来源：02-费用报销制度.txt] 第三条："单笔金额超过 5000 元的，需总经理审批。"
结论：报销金额超过5000元需要总经理审批。

问题：公司附近有什么好吃的餐厅？
回答：依据：资料中没有任何文件提及公司附近餐厅信息。
结论：不知道。
```

## 🛠️ 技术栈

| 组件 | 用途 |
|---|---|
| LangChain | RAG 流水线编排 |
| Chroma | 向量数据库 |
| GTE (中文) | 语义向量化 |
| BM25 + jieba | 关键词检索（中文分词） |
| bge-reranker-v2-m3 | 检索结果精排 |
| DeepSeek API | 大模型生成 |
| Python | 主语言 |

## 📁 项目结构

```
rag-qa/
├── rag_qa.py               # 主程序（生产链路）
├── hybrid_retriever.py     # 混合检索（向量 + BM25，RRF 融合）
├── rerank.py               # bge-reranker 精排
├── query_rewrite.py        # Multi-Query 查询改写（可选）
├── clean_pipeline.py       # 文本清洗管道
├── eval_set.json           # 20 条评测集
├── eval_baseline.py        # 基线评测
├── eval_with_rerank.py     # rerank 评测
├── eval_with_hybrid.py     # 混合检索评测
├── eval_hybrid_rerank.py   # 生产链路评测
├── eval_with_multi_query.py# 查询改写评测
├── eval_chunk_opt.py       # 切块策略评测
├── requirements.txt        # 依赖
└── README.md               # 本文件
```

## 🔜 待优化

- [ ] 支持 PDF / Word / 网页文档上传
- [ ] Ragas 深度评测（Faithfulness / Context Precision）
- [ ] FastAPI 接口 + 网页界面
- [ ] Docker 部署
- [ ] 性能监控（响应时间 / Token 消耗）
