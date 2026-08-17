# 📚 企业知识库问答系统（RAG）

基于**检索增强生成（RAG）**的智能问答系统。让大模型能回答企业私有文档的问题，并且**有据可查、不瞎编**。

## ✨ 功能

- 上传企业文档（当前支持 .txt）
- 语义检索：问「怎么报销」，能找到文档里写「费用如何申请」的片段
- 智能回答：DeepSeek 大模型基于检索内容生成答案
- 防幻觉：文档里没有答案时，模型会老实说「不知道」

## 🧠 原理（RAG 四步）

```
文档 → ① 切块 → ② 向量化 → ③ 存储(Chroma) → ④ 检索+生成
```

1. **切块**：把长文档切成 200 字的小段
2. **向量化**：用 GTE 中文嵌入模型把文字变成向量（语义相近的文字向量也相近）
3. **存储**：向量存入 Chroma 向量数据库
4. **检索+生成**：用户提问 → 找出最相关的 3 段 → 拼进 prompt → DeepSeek 生成答案

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载嵌入模型（ModelScope）

```python
from modelscope import snapshot_download
snapshot_download('iic/nlp_gte_sentence-embedding_chinese-base')
```

### 3. 配置 API Key

在 `rag_qa.py` 中填入你的 DeepSeek API key：

```python
DEEPSEEK_API_KEY = "sk-你的key"
```

### 4. 运行

```bash
python rag_qa.py
```

## 📸 效果演示

```
知识库问答系统启动！输入问题，q 退出

问题：报销金额超过 5000 元需要谁审批？
回答：根据资料中的公司报销制度第一条和第七条，报销金额超过 5000 元需要总经理审批。

问题：公司附近有什么好吃的餐厅？
回答：资料中没有相关信息，无法回答。
```

## 🛠️ 技术栈

| 组件 | 用途 |
|---|---|
| LangChain | RAG 流水线编排 |
| Chroma | 向量数据库 |
| GTE (中文) | 文本向量化 |
| DeepSeek API | 大模型生成 |
| Python | 主语言 |

## 📁 项目结构

```
rag-qa/
├── rag_qa.py          # 主程序
├── requirements.txt   # 依赖
├── 公司制度.txt        # 示例知识库文档
└── README.md          # 本文件
```

## 🔜 待优化

- [ ] 支持 PDF / Word / 网页文档
- [ ] 答案带引用溯源（标注来自文档哪一段）
- [ ] Streamlit 网页界面
- [ ] 检索结果重排序（rerank）
