"""
demo_live.py — 端到端演示脚本（打真实接口，会调用大模型）
==========================================================
先启动服务：python run_server.py
再运行：    python demo_live.py

它会依次做：
  1. GET  /health            冷启动状态（检索器懒加载，此时还没构建）
  2. POST /ask ×4            正常事实题 / 金额审批题 / 相似条款题 / 无答案题
  3. POST /documents/upload  上传一份临时制度 → 重建索引（约 15-20 秒）
  4. POST /ask               问新文档里的内容
  5. DELETE /documents/...   删除该文档 → 重建（恢复原状）
  6. POST /ask               再问同一问题（应变成"不知道"）

注意：会真的往知识库目录写入并删除一个文件（脚本结束时会清理）。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8000")
KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "知识库")
TEMP_DOC = "99-演示用培训制度.txt"


def call(method: str, path: str, payload=None, timeout: int = 600):
    """统一请求：路径里的中文必须 percent-encode（HTTP 请求行只能 ASCII）"""
    url = BASE + urllib.parse.quote(path, safe="/?=&")
    data, headers = None, {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def upload(path: str, filename: str, timeout: int = 600):
    boundary = f"----ragdemo{int(time.time())}"
    with open(path, "rb") as f:
        content = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: text/plain\r\n\r\n"
    ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        BASE + "/documents/upload", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def show_ask(question: str) -> None:
    t0 = time.perf_counter()
    status, body = call("POST", "/ask", {"question": question})
    elapsed = time.perf_counter() - t0
    print(f"\n--- 问：{question}（HTTP {status}，总耗时 {elapsed:.1f}s）---")
    if status != 200:
        print("失败：", body)
        return
    print("答：", body["answer"].strip())
    print("耗时细分：", body["timings"])
    for i, s in enumerate(body["sources"], 1):
        print(f"  引用{i}：{s['source']}")


def main() -> int:
    print("===== 1. 健康检查（冷启动，检索器尚未构建）=====")
    status, health = call("GET", "/health")
    print(f"HTTP {status} | status={health['status']} retriever_ready={health['retriever_ready']} "
          f"documents={health['documents']} chunks={health['chunks']}")

    print("\n===== 2. 四类问题 =====")
    show_ask("婚假有几天？")
    show_ask("报销金额超过5000元需要谁审批？")
    show_ask("出差期间的餐饮补贴按什么计算？")
    show_ask("公司年会在什么时候举办？")

    print("\n===== 3. 上传新文档（触发重建）=====")
    tmp_path = os.path.join(KB_DIR, TEMP_DOC)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("演示用培训制度\n\n第一条：公司每年为员工提供不少于 40 小时的内部培训，"
                "培训费用由公司承担。\n")
    t0 = time.perf_counter()
    status, body = upload(tmp_path, TEMP_DOC)
    print(f"HTTP {status}（{time.perf_counter()-t0:.1f}s）：{json.dumps(body, ensure_ascii=False)}")

    print("\n===== 4. 问新文档内容 =====")
    show_ask("员工每年要参加多少小时培训？培训费用谁承担？")

    print("\n===== 5. 删除该文档（触发重建）=====")
    t0 = time.perf_counter()
    status, body = call("DELETE", f"/documents/{TEMP_DOC}")
    print(f"HTTP {status}（{time.perf_counter()-t0:.1f}s）：{json.dumps(body, ensure_ascii=False)}")

    print("\n===== 6. 删除后再问（应变成不知道）=====")
    show_ask("员工每年要参加多少小时培训？")

    status, health = call("GET", "/health")
    print(f"\n收尾：documents={health['documents']} chunks={health['chunks']} "
          f"retriever_ready={health['retriever_ready']}")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
