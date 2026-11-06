"""Native Windows cache path checks; no Paddle import, model download or API."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from src.ocr_pipeline import _prepare_paddle_cache
from src import ocr_pipeline


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows native path fix")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.delenv("PADDLE_PDX_CACHE_HOME", raising=False)
    monkeypatch.delitem(sys.modules, "paddlex.utils.cache", raising=False)
    runtime = tmp_path / "中文运行目录"
    target = runtime / "cache" / "paddleocr-vl-1.6"
    target.mkdir(parents=True)
    (target / "inference.json").write_bytes(b'{"unchanged":true}')
    return runtime, target


def test_unicode_cache_uses_same_bytes_through_reusable_ascii_junction(cache, monkeypatch):
    runtime, target = cache
    first = Path(_prepare_paddle_cache(runtime))
    assert str(first).isascii() and first.resolve() == target.resolve()
    assert (first / "inference.json").read_bytes() == (target / "inference.json").read_bytes()
    assert os.path.samefile(first, target)
    monkeypatch.delenv("PADDLE_PDX_CACHE_HOME")
    assert _prepare_paddle_cache(runtime) == str(first)


@pytest.mark.parametrize("unicode_override", [False, True])
def test_explicit_cache_target_is_preserved(cache, tmp_path, monkeypatch, unicode_override):
    runtime, default_target = cache
    explicit = tmp_path / ("自选缓存" if unicode_override else "custom-cache")
    explicit.mkdir()
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(explicit))
    selected = Path(_prepare_paddle_cache(runtime))
    assert selected.resolve() == explicit.resolve() and selected.resolve() != default_target.resolve()
    assert str(selected).isascii()
    if not unicode_override:
        assert selected == explicit


@pytest.mark.parametrize("wrong_kind", ["directory", "junction"])
def test_existing_wrong_alias_is_never_replaced(cache, tmp_path, monkeypatch, wrong_kind):
    import _winapi
    runtime, target = cache
    alias = Path(_prepare_paddle_cache(runtime))
    alias.rmdir()  # Remove only this test-created junction, never its target.
    other = tmp_path / "other"
    other.mkdir()
    if wrong_kind == "junction":
        _winapi.CreateJunction(str(other), str(alias))
    else:
        alias.mkdir()
    (alias / "keep.txt").write_bytes(b"user-owned")
    monkeypatch.delenv("PADDLE_PDX_CACHE_HOME")
    with pytest.raises(RuntimeError, match="未修改"):
        _prepare_paddle_cache(runtime)
    assert (alias / "keep.txt").read_bytes() == b"user-owned"
    assert (target / "inference.json").read_bytes() == b'{"unchanged":true}'


def test_preimported_unicode_cache_requires_restart(cache, monkeypatch):
    runtime, target = cache
    monkeypatch.setitem(sys.modules, "paddlex.utils.cache", SimpleNamespace(CACHE_DIR=str(target)))
    with pytest.raises(RuntimeError, match="重新启动"):
        _prepare_paddle_cache(runtime)
    assert "PADDLE_PDX_CACHE_HOME" not in os.environ


def test_no_controlled_ascii_project_root_does_not_create_link(cache, monkeypatch):
    runtime, target = cache
    runtime = target.parent
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(target))
    with pytest.raises(RuntimeError, match="配置英文"):
        _prepare_paddle_cache(runtime)
    assert os.environ["PADDLE_PDX_CACHE_HOME"] == str(target)


def test_preimported_different_ascii_cache_requires_restart(cache, tmp_path, monkeypatch):
    runtime, target = cache
    monkeypatch.setitem(sys.modules, "paddlex.utils.cache", SimpleNamespace(CACHE_DIR=str(tmp_path / "other-cache")))
    with pytest.raises(RuntimeError, match="重新启动"):
        _prepare_paddle_cache(runtime)
