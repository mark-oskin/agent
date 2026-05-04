"""Tests for ``tools.web`` URL policy and plugin registration."""

from __future__ import annotations

import importlib
import importlib.util

import pytest

import agentlib.tools.plugins as plugins


def test_url_allowed_accepts_https():
    from tools import web as w

    assert w._url_allowed("https://example.com/path") is None


def test_url_allowed_rejects_scheme():
    from tools import web as w

    assert w._url_allowed("file:///etc/passwd") is not None


def test_browser_navigate_rejects_bad_scheme():
    from tools import web as w

    assert "scheme" in w.browser_navigate({"url": "javascript:alert(1)"}).lower()


def test_plugin_toolset_registers_browser_tools(tmp_path):
    from pathlib import Path

    plugins.load_plugin_toolsets(tools_dir=str(tmp_path), default_tools_dir=str(tmp_path))
    assert "browser_navigate" not in plugins.PLUGIN_TOOL_HANDLERS

    import tools as tools_pkg

    base = str(Path(tools_pkg.__file__).resolve().parent)
    plugins.load_plugin_toolsets(tools_dir=None, default_tools_dir=base)
    assert "browser_navigate" in plugins.PLUGIN_TOOL_HANDLERS
    assert plugins.PLUGIN_TOOL_TO_TOOLSET.get("browser_navigate") == "browser"


def test_browser_close_resets_without_playwright(monkeypatch):
    from tools import web as w

    w._reset_browser("test")  # noqa: SLF001 - module under test
    assert w.browser_close({}) == "Browser session closed. Call browser_navigate to open a new session."


@pytest.mark.skipif(
    importlib.util.find_spec("playwright") is None,
    reason="Playwright not installed (uv sync --extra browser)",
)
def test_playwright_smoke_navigate_snapshot():
    from tools import web as w

    try:
        w._reset_browser("clean")  # noqa: SLF001
        out = w.browser_navigate({"url": "https://example.com/"})
        assert "Navigated OK" in out
        snap = w.browser_snapshot({"selector": "body", "max_chars": 2000})
        assert "Example Domain" in snap or "example" in snap.lower()
    finally:
        w._reset_browser("teardown")  # noqa: SLF001
