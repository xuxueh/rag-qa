"""配置模块测试：环境变量优先级、路径解析、缺配置时的可执行报错"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config


def test_load_env_reads_key_value(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# 注释\nFOO_TEST_KEY=bar\n\nBAZ_TEST_KEY="quoted"\n', encoding="utf-8")
    monkeypatch.delenv("FOO_TEST_KEY", raising=False)
    monkeypatch.delenv("BAZ_TEST_KEY", raising=False)
    config.load_env(env)
    import os
    assert os.environ["FOO_TEST_KEY"] == "bar"
    assert os.environ["BAZ_TEST_KEY"] == "quoted"


def test_load_env_does_not_override_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DSH_TEST_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("DSH_TEST_KEY", "from_shell")
    config.load_env(env)
    import os
    assert os.environ["DSH_TEST_KEY"] == "from_shell"   # 系统环境变量优先


def test_as_path_resolves_relative_to_repo():
    assert config._as_path("知识库") == config.BASE_DIR / "知识库"
    assert config._as_path(str(config.BASE_DIR)).is_absolute()


def test_env_int_rejects_non_numeric(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE_TEST", "abc")
    with pytest.raises(config.ConfigError) as exc:
        config._env_int("CHUNK_SIZE_TEST", 200)
    assert "不是整数" in str(exc.value)


def test_env_int_rejects_too_small(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE_TEST", "0")
    with pytest.raises(config.ConfigError):
        config._env_int("CHUNK_SIZE_TEST", 200, minimum=1)


def test_check_reports_missing_knowledge_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path / "nope")
    problems = config.check(require_api_key=False)
    assert any("知识库目录不存在" in p for p in problems)


def test_check_reports_empty_knowledge_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path)
    problems = config.check(require_api_key=False)
    assert any("没有可用文档" in p for p in problems)


def test_check_reports_missing_endpoint_key(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")
    problems = config.check()
    assert any("DEEPSEEK_API_KEY" in p for p in problems)
    # 只做检索评测时不该因为缺密钥而失败
    assert not any("DEEPSEEK_API_KEY" in p for p in config.check(require_api_key=False))


def test_require_api_key_raises_with_fix_hint(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")
    with pytest.raises(config.ConfigError) as exc:
        config.require_api_key()
    assert ".env" in str(exc.value)


def test_assert_valid_lists_all_problems(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path / "nope")
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")
    with pytest.raises(config.ConfigError) as exc:
        config.assert_valid()
    msg = str(exc.value)
    assert "知识库目录不存在" in msg and "DEEPSEEK_API_KEY" in msg


def test_describe_never_leaks_key(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "sk-super-secret-value")
    snapshot = config.describe()
    assert snapshot["llm"]["api_key_set"] is True
    assert "sk-super-secret-value" not in str(snapshot)
