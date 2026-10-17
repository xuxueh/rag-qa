"""
config.py — 统一配置入口（唯一读取环境变量/路径的地方）
==========================================================
解决的问题（本轮回访要求「第三步：修复工程问题」）：
1. 模型路径不再写死在某台电脑上：走 .env / 环境变量，默认相对仓库目录
2. API 密钥统一走环境变量，代码里不出现明文密钥，缺失时给出可执行的报错
3. 切块 / 召回 / rerank 等超参集中一处，改参数不用翻 5 个文件
4. 配置错误一次性列出全部问题（而不是跑 2 分钟后才崩在某一行）

约定：
- 环境变量 > .env > 仓库内默认值
- 相对路径一律按「仓库根目录」解析，换机器只要改 .env
- 本模块不导入任何重依赖（langchain/torch），可被测试单独导入
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


class ConfigError(RuntimeError):
    """配置缺失或非法。消息里必须写清「怎么修」。"""


# ═══════════════════════ .env 加载 ═══════════════════════
def load_env(path: Path | str = ENV_FILE, override: bool = False) -> None:
    """把 .env 读进 os.environ。已有系统环境变量优先（部署时直接注入即可）。

    只支持 KEY=VALUE / # 注释 / 空行，够用且无额外依赖。
    """
    path = Path(path)
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value


def _as_path(value: str, *, relative_to: Path = BASE_DIR) -> Path:
    """支持 ~ / 环境变量 / 相对仓库根的路径"""
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return expanded if expanded.is_absolute() else (relative_to / expanded)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} 不是整数，请改成数字（例如 {default}）") from exc
    if value < minimum:
        raise ConfigError(f"{name}={value} 太小，必须 >= {minimum}")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(str(raw).strip())
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} 不是数字，请改成小数（例如 {default}）") from exc


def _is_path_ref(value: str) -> bool:
    """判断是本地路径还是模型 ID（如 iic/nlp_gte_sentence-embedding_chinese-small）"""
    if not value:
        return False
    return (
        os.path.isabs(value)
        or value.startswith(".")
        or "\\" in value
        or value.startswith("~")
    )


load_env()

# ═══════════════════════ 路径与模型 ═══════════════════════
KNOWLEDGE_DIR = _as_path(os.environ.get("KNOWLEDGE_DIR") or "知识库")

# 可以是本地目录，也可以是 ModelScope / HuggingFace 模型 ID
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL") or str(BASE_DIR / "models" / "gte-small")
RERANK_MODEL = os.environ.get("RERANK_MODEL") or str(BASE_DIR / "models" / "bge-reranker-v2-m3")

# ═══════════════════════ 大模型 ═══════════════════════
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.0)

# ═══════════════════════ 检索超参（集中管理） ═══════════════════════
CHUNK_SIZE = _env_int("CHUNK_SIZE", 200)          # 实验验证 200 最优
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 20, minimum=0)
RECALL_TOP_K = _env_int("RECALL_TOP_K", 10)       # 混合检索召回池（10 时池内命中 95.3%）
RERANK_TOP_N = _env_int("RERANK_TOP_N", 3)        # 进 prompt 的条数
RRF_K = _env_int("RRF_K", 60)                     # RRF 平滑常数
BM25_TOP_K = _env_int("BM25_TOP_K", 0, minimum=0)  # 0 = 与召回池同宽（双路各取 Top-K）

# ═══════════════════════ 服务 ═══════════════════════
ALLOWED_EXT = {".txt", ".md", ".pdf", ".docx"}
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 20)
LOG_LEVEL = (os.environ.get("LOG_LEVEL") or "INFO").upper()

# 兼容旧代码里的变量名（历史脚本 import rag_qa.DOC_DIR / EMBEDDING_MODEL_PATH）
DOC_DIR = str(KNOWLEDGE_DIR)
EMBEDDING_MODEL_PATH = EMBEDDING_MODEL


# ═══════════════════════ 校验 ═══════════════════════
def _check_model_ref(name: str, value: str) -> str | None:
    """本地路径必须存在；模型 ID 无法离线校验，返回 None 放行"""
    if not value:
        return f"{name} 为空，请在 .env 里设置（本地目录或模型 ID）"
    if _is_path_ref(value) and not Path(value).exists():
        return (
            f"{name}={value} 不存在。\n"
            f"    修复：① 在 .env 里改成实际路径；或 ② 用 ModelScope 下载到仓库 models/ 下；"
            f"或 ③ 设成模型 ID（如 iic/nlp_gte_sentence-embedding_chinese-small）"
        )
    return None


def check(require_api_key: bool = True) -> list[str]:
    """返回全部配置问题（空列表=健康）。供 /health、启动自检、CLI 使用。"""
    problems: list[str] = []
    if not KNOWLEDGE_DIR.exists():
        problems.append(f"知识库目录不存在：{KNOWLEDGE_DIR}（设置 KNOWLEDGE_DIR 或创建该目录）")
    elif not any(p.is_file() and p.suffix.lower() in ALLOWED_EXT for p in KNOWLEDGE_DIR.iterdir()):
        problems.append(f"知识库目录里没有可用文档（支持 {sorted(ALLOWED_EXT)}）：{KNOWLEDGE_DIR}")
    for name, value in (("EMBEDDING_MODEL", EMBEDDING_MODEL), ("RERANK_MODEL", RERANK_MODEL)):
        msg = _check_model_ref(name, value)
        if msg:
            problems.append(msg)
    if require_api_key and not DEEPSEEK_API_KEY:
        problems.append(
            "DEEPSEEK_API_KEY 未设置：在 .env 写 DEEPSEEK_API_KEY=sk-xxx，"
            "或用系统环境变量注入（不要写进代码/提交到 git）"
        )
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        problems.append(f"CHUNK_OVERLAP({CHUNK_OVERLAP}) 必须小于 CHUNK_SIZE({CHUNK_SIZE})")
    return problems


def assert_valid(require_api_key: bool = True) -> None:
    """启动/评测前的硬校验：一次性把问题全抛出来"""
    problems = check(require_api_key)
    if problems:
        bullet = "\n".join(f"  - {p}" for p in problems)
        raise ConfigError(f"配置有 {len(problems)} 处问题，请先修复：\n{bullet}")


def require_api_key() -> str:
    if not DEEPSEEK_API_KEY:
        raise ConfigError(
            "缺少 DEEPSEEK_API_KEY：请在 .env 里写 DEEPSEEK_API_KEY=sk-xxx，"
            "或用系统环境变量注入；密钥不要写进代码"
        )
    return DEEPSEEK_API_KEY


def resolve_embedding_model() -> str:
    msg = _check_model_ref("EMBEDDING_MODEL", EMBEDDING_MODEL)
    if msg:
        raise ConfigError(msg)
    prepare_local_model_env(EMBEDDING_MODEL)
    return EMBEDDING_MODEL


def resolve_rerank_model() -> str:
    msg = _check_model_ref("RERANK_MODEL", RERANK_MODEL)
    if msg:
        raise ConfigError(msg)
    prepare_local_model_env(RERANK_MODEL)
    return RERANK_MODEL


def prepare_local_model_env(model_ref: str) -> None:
    """本地模型目录 → 关掉 HuggingFace 联网校验。

    踩过的坑：路径明明是本地目录，sentence-transformers 仍会尝试访问 huggingface.co
    校验/补文件；在无外网（或代理不稳）的机器上会卡十几分钟才超时。
    """
    if _is_path_ref(model_ref) and Path(model_ref).exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def describe() -> dict:
    """配置快照（不含密钥），给 /health、日志、README 用"""
    return {
        "knowledge_dir": str(KNOWLEDGE_DIR),
        "embedding_model": EMBEDDING_MODEL,
        "rerank_model": RERANK_MODEL,
        "llm": {"model": DEEPSEEK_MODEL, "base_url": DEEPSEEK_BASE_URL,
                "api_key_set": bool(DEEPSEEK_API_KEY)},
        "chunk": {"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
        "retrieval": {"recall_top_k": RECALL_TOP_K, "rerank_top_n": RERANK_TOP_N, "rrf_k": RRF_K},
        "log_level": LOG_LEVEL,
    }


# ═══════════════════════ 日志 ═══════════════════════
_LOG_CONFIGURED = False


def setup_logging(level: str | int | None = None) -> logging.Logger:
    """统一日志格式。入口处调用一次（CLI / API / 评测脚本）。"""
    global _LOG_CONFIGURED
    lvl = level if level is not None else LOG_LEVEL
    if not _LOG_CONFIGURED:
        logging.basicConfig(
            level=lvl,
            format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOG_CONFIGURED = True
    root = logging.getLogger()
    root.setLevel(lvl)
    return logging.getLogger("rag")


if __name__ == "__main__":
    # 自检：python config.py
    setup_logging()
    log = logging.getLogger("config")
    log.info("配置快照: %s", describe())
    issues = check()
    if issues:
        log.error("发现 %d 处配置问题：", len(issues))
        for i in issues:
            log.error("  - %s", i)
        raise SystemExit(1)
    log.info("配置自检通过 ✅")
