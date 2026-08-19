"""启动脚本：直接运行本文件即可启动 RAG API 服务
用法: python run_server.py
"""
import os

# 关键：清理 PYTHONPATH 污染（Hermes 终端会注入 Python 3.11 的包路径）
os.environ.pop("PYTHONPATH", None)

import uvicorn

if __name__ == "__main__":
    uvicorn.run("rag_api:app", host="127.0.0.1", port=8000)
