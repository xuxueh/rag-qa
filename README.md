# 📚 企业知识库问答系统（RAG）

基于**检索增强生成（RAG）**的智能问答系统。让大模型能回答企业私有文档的问题，并且**有据可查、不瞎编**。

## ✨ 功能

- 多格式知识库：**TXT / Markdown / PDF / Word** 自动解析（smart_loader）
- **文档管理 API**：上传 / 删除文档，变更后自动重建检索器
- **混合检索**：GTE 语义向量 + BM25 关键词（RRF 融合）
- **Reranker 精排**：bge-reranker-v2-m3（本地，优雅降级）
- **Citation 溯源**：答案标注"文件 · 第 X 条"
- 防幻觉：文档里没有答案时，模型老实说「不知道」
- **评测体系**：50 条 7 类分类测试集 + Recall@K / MRR / 回答正确率

## 🧠 生产链路

```
文档(上传/删除 → 自动重建)
  → 文本清洗 → 句子边界切块(200字,实验验证最优)
  → 混合检索（GTE语义 + BM25，RRF融合）
  → bge-reranker 精排
  → DeepSeek 生成（防幻觉 prompt）
  → Answer + Citation（文件 · 第X条）
```

## 📊 评测结果

**50 条 7 类分类测试集（简单事实/关键词/语义/多段/模糊/无答案/跨文档）**

| 检索器 | Recall@1 | Recall@3 | Recall@10 | MRR |
|---|---|---|---|---|
| 纯向量 | 62.8% | 79.1% | 88.4% | 0.716 |
| 混合检索（RRF） | 74.4% | 83.7% | 95.3% | 0.811 |
| **混合 + Rerank** | **83.7%** | **93.0%** | 95.3% | **0.888** |

**回答质量**：LLM-as-judge 评测 20/20 = **100% 正确率**

**性能**：单次问答平均 5.8s（rerank 候选 5 个，精度无损——实验验证）

**关键实验结论**（各实验脚本可复现）：
- RRF 会把"单路高排名"的正确项挤出候选 → **rerank 弥补**（跨文档 R@1 +28.5pp）
- chunk 大小权衡：chunk 越大召回越宽但排序越差 → **200 最优**
- Multi-Query 对标准表述无提升 → 未集成（评测驱动决策）

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载模型（ModelScope）
```python
from modelscope import snapshot_download
snapshot_download('iic/nlp_gte_sentence-embedding_chinese-small', local_dir='data/models/gte-small')
snapshot_download('BAAI/bge-reranker-v2-m3', local_dir='data/models/bge-reranker-v2-m3')
```

### 3. 配置 `.env`（已被 gitignore，不会上传）
```bash
DEEPSEEK_API_KEY=sk-你的key
EMBEDDING_MODEL=你的GTE模型本地路径
```
在 `rag_qa.py` 中修改 `DOC_DIR` 为知识库目录。

### 4. 运行
```bash
# 命令行问答
python rag_qa.py

# API + 网页界面（http://127.0.0.1:8000）
python run_server.py

# Streamlit 界面
streamlit run rag_app.py

# 评测
python eval_v2.py               # Recall@K/MRR 分类评测
python eval_v2_rerank.py        # rerank 效果对比
python eval_answer_quality.py   # 回答质量（LLM-as-judge）
```

## 🛠️ 文档管理 API

```
GET    /documents              列出知识库文档
POST   /documents/upload       上传（txt/md/pdf/docx）→ 自动重建
DELETE /documents/{name}       删除 → 自动重建
```

## 🧠 技术栈

| 组件 | 用途 |
|---|---|
| LangChain | RAG 流水线编排 |
| Chroma | 向量数据库 |
| GTE (中文) | 语义向量化 |
| BM25 + jieba | 关键词检索 |
| bge-reranker-v2-m3 | 精排 |
| DeepSeek API | 生成 |
| pypdf / python-docx | PDF / Word 解析 |
| FastAPI / Streamlit | API + 界面 |
| Docker | 部署（Dockerfile） |

## 📁 项目结构

```
rag-qa/
├── rag_qa.py               # 命令行主程序（生产链路）
├── rag_api.py              # FastAPI v3（问答 + 文档管理）
├── rag_app.py              # Streamlit 界面
├── run_server.py           # 启动脚本
├── hybrid_retriever.py     # 混合检索（RRF + Citation）
├── rerank.py               # bge-reranker 精排
├── smart_loader.py         # 多格式加载（txt/md/pdf/docx）
├── clean_pipeline.py       # 文本清洗
├── query_rewrite.py        # Multi-Query（评测后未集成，可选）
├── static/index.html       # 问答网页
├── eval_set.json           # 20 条基础评测集
├── eval_set_v2.json        # 50 条 7 类分类评测集
├── eval_v2.py              # Recall@K/MRR 评测
├── eval_v2_rerank.py       # rerank 对比
├── eval_answer_quality.py  # 回答质量（LLM-as-judge）
├── eval_failure_analysis.py# 失败案例诊断
├── eval_chunk_experiment.py# chunk 参数实验
├── eval_perf.py            # 性能监控
├── Dockerfile
└── README.md
```

## 🔜 待优化（诚实清单）

- [ ] **chunk-level 评测**：gold_chunk_id（当前 source 文件级命中可能假阳性）
- [ ] 无答案检测指标（正确回答率 + 正确拒答率）
- [ ] PDF 页码级 Citation（文件 + 页码 + 章节 + chunk_id）
- [ ] 检索数据结构重构（chunk_id / metadata，替代文本作 ID）
- [ ] Ragas 深度指标（Faithfulness / Context Precision）
- [ ] Docker 云端部署 + 网页上传界面
