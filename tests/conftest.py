"""pytest 公共配置

- 仓库根目录进 sys.path
- 自定义 tmp_path：某些受限环境（沙箱/EDR）里 pytest 自带的临时目录
  （mkdtemp，0700）创建后无法再扫描，会把所有用 tmp_path 的用例变成
  PermissionError。这里改成在仓库 .tmp 下用 makedirs 建用例目录，稳定可用。
"""
import logging
import shutil
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TMP_ROOT = ROOT / ".tmp"

logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


@pytest.fixture
def tmp_path():
    """替代 pytest 内建 tmp_path（理由见文件头）"""
    TMP_ROOT.mkdir(exist_ok=True)
    path = TMP_ROOT / f"case-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
