"""
verify_eval_set.py — 评测集质量校验（跑评测前先跑它）
=====================================================
检查项：
1. id 唯一、分类齐全、总数符合预期
2. 每条的 source 文件真实存在
3. 答案条款号在原文里确实存在（"第X条" 出现在该文件）
4. evidence 摘录能逐字（忽略空白差异）在原文中找到 —— 防止凭印象编 gold
5. 无答案题不带 source；有答案题必须带 source

用法：python verify_eval_set.py [eval_set_v3.json]
解析失败会打印到 stderr 并返回退出码 1，可直接放进 CI。
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

CN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
      "十": 10, "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15}


def _norm(text: str) -> str:
    """去掉所有空白，避免“原文空格/换行”造成假失败"""
    return re.sub(r"\s+", "", text or "")


def _doc_text(source: str) -> str:
    path = config.KNOWLEDGE_DIR / source
    return _norm(path.read_text(encoding="utf-8", errors="ignore"))


def evidence_segments(evidence: str) -> list[str]:
    """evidence 允许用 … 拼接多处原文（例如一条制度里不相邻的两句）"""
    return [s for s in re.split(r"…+|\.{3,}", evidence or "") if _norm(s)]


def _articles_in(text: str) -> set[int]:
    found = set()
    for m in re.finditer(r"第([一二三四五六七八九十]+)条", text):
        if m.group(1) in CN:
            found.add(CN[m.group(1)])
    for m in re.finditer(r"第(\d+)条", text):
        found.add(int(m.group(1)))
    return found


def verify(path: Path) -> list[str]:
    items = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []

    ids = Counter(i.get("id") for i in items)
    for dup, n in ids.items():
        if n > 1:
            errors.append(f"id 重复：{dup} ×{n}")

    expected_total = 100
    if len(items) != expected_total:
        errors.append(f"条目数 {len(items)} ≠ {expected_total}")

    cats = Counter(i.get("category") for i in items)
    print("分类分布：")
    for cat, n in cats.items():
        print(f"  {cat:<12} {n}")

    doc_cache: dict[str, str] = {}
    for item in items:
        iid = item.get("id", "?")
        source = item.get("source")
        if item.get("category") == "无答案问题":
            if source:
                errors.append(f"{iid} 是无答案题，不应指定 source")
            continue
        if not source:
            errors.append(f"{iid} 缺少 source")
            continue
        if not (config.KNOWLEDGE_DIR / source).exists():
            errors.append(f"{iid} source 文件不存在：{source}")
            continue
        text = doc_cache.setdefault(source, _doc_text(source))

        arts = item.get("articles") or []
        if not arts:
            errors.append(f"{iid} 有答案但没写 gold 条款号")
        present = _articles_in(text)
        for a in arts:
            if a not in present:
                errors.append(f"{iid} gold 条款 第{a}条 在 {source} 中不存在")

        evidence = item.get("evidence")
        if not evidence:
            errors.append(f"{iid} 缺少 evidence 原文摘录")
        else:
            # evidence 可以是「source + also」任一文件的拼接
            pool = text
            for extra in item.get("also") or []:
                if (config.KNOWLEDGE_DIR / extra).exists():
                    pool += doc_cache.setdefault(extra, _doc_text(extra))
            for seg in evidence_segments(evidence):
                if _norm(seg) not in pool:
                    errors.append(f"{iid} evidence 在原文中找不到（可能被改写）：{seg[:30]}…")

        for extra in item.get("also") or []:
            if not (config.KNOWLEDGE_DIR / extra).exists():
                errors.append(f"{iid} also 文件不存在：{extra}")

    return errors


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("eval_set_v3.json")
    print(f"校验：{target}")
    errors = verify(target)
    if errors:
        print(f"\n❌ 发现 {len(errors)} 处问题：", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("\n✅ 评测集校验通过：id 唯一、条款号存在、evidence 可在原文逐字命中")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
