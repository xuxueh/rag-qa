# RAG 问答系统 Docker 镜像
# 说明：
# - 模型（GTE ~2.1GB + reranker ~2.1GB）较大，不打进镜像，用 volume 挂载
# - .env（API key）用环境变量注入，不打进镜像
#
# 构建: docker build -t rag-qa .
# 运行: docker run -p 8000:8000 \
#   -v /path/to/知识库:/app/知识库 \
#   -v /path/to/models:/app/models \
#   -e DEEPSEEK_API_KEY=sk-xxx \
#   -e EMBEDDING_MODEL=/app/models/gte-small \
#   rag-qa
#   然后改 rag_api.py 的 DOC_DIR/EMBEDDING_MODEL 指向挂载路径

FROM python:3.11-slim

WORKDIR /app

# 依赖（清华镜像加速）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 代码
COPY . .

EXPOSE 8000

CMD ["python", "run_server.py"]
