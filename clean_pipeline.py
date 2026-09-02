"""
clean_pipeline.py - 文本清洗管道
=================================
数据工程：PDF/网页/OCR 提取的原始文本常带噪声（乱码、多余空白、
重复段落、非法字符）。清洗后再切块入库，保证数据质量。

- clean_text(): 单段文本清洗
- clean_file(): 文件清洗（保留结构）
"""
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")


def clean_text(text: str) -> str:
    """清洗单段文本：换行统一、去多余空白、去全角空格、去不可见字符"""
    if not text:
        return text
    # 1. 统一换行符（Windows/Mac → Unix）
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 2. 全角空格 → 普通空格
    text = text.replace("\u3000", " ")
    # 3. 去零宽字符等不可见字符（保留换行和普通空格）
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # 4. 连续空格/制表符 → 单空格
    text = re.sub(r"[ \t]+", " ", text)
    # 5. 行首尾空白去掉
    lines = [l.strip() for l in text.split("\n")]
    # 6. 连续 3+ 空行 → 2 空行
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return cleaned


def dedupe_lines(text: str) -> str:
    """去除完全重复的段落（保持首次出现顺序）"""
    seen = set()
    out = []
    for line in text.split("\n"):
        key = line.strip()
        if not key:
            if out and out[-1] != "":
                out.append("")
            continue
        if key not in seen:
            seen.add(key)
            out.append(line)
    return "\n".join(out).strip()


def clean_document(text: str) -> str:
    """完整清洗管道：clean_text → dedupe_lines"""
    return dedupe_lines(clean_text(text))


if __name__ == "__main__":
    # 自测：模拟一段带噪声的文本
    dirty = "  第一条：上班时间为 9:00  至 18:00 。\r\n\r\n\r\n第一条：上班时间为 9:00 至 18:00 。\u3000\u3000第二条：午休 12:00-13:00\x00\x07"
    print("清洗前：")
    print(repr(dirty))
    print("\n清洗后：")
    print(clean_document(dirty))
