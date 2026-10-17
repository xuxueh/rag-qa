"""启动 RAG API 服务
用法：
    python run_server.py                  # 默认 127.0.0.1:8000
    set API_HOST=0.0.0.0 & set API_PORT=8080 & python run_server.py
启动后：http://127.0.0.1:8000/docs 看接口，/health 看状态
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 关键：清理 PYTHONPATH 污染（某些终端会注入别的 Python 版本包路径，导致 DLL 加载失败）
os.environ.pop("PYTHONPATH", None)

import uvicorn

import config

if __name__ == "__main__":
    config.setup_logging()
    problems = config.check(require_api_key=False)
    for p in problems:
        print(f"⚠️  {p}")
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    print(f"启动服务：http://{host}:{port}  （文档 /docs，健康检查 /health）")
    uvicorn.run("rag_api:app", host=host, port=port)
