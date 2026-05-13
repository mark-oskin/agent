"""Tests for ``tools.queue`` global FIFO lists and tool handlers."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from tests.harness import build_test_session, run_session_lines
from tools import queue as Q


@pytest.fixture(autouse=True)
def _reset_queues():
    Q.reset_queues_for_testing()
    yield
    Q.reset_queues_for_testing()


def test_queue_toolset_in_default_tools_dir():
    import os

    from agentlib.tools.registry import ToolRegistry

    project = Path(__file__).resolve().parent.parent
    reg = ToolRegistry(default_tools_dir=os.path.join(project, "tools"))
    reg.load_plugin_toolsets(reg.default_tools_dir)
    reg.register_aliases()
    assert "queue" in reg.plugin_toolsets
    assert "list_add" in reg.plugin_tool_handlers
    assert "list_remove" in reg.plugin_tool_handlers
    assert "list_names" in reg.plugin_tool_handlers


def test_fifo_remove_and_peek():
    assert Q.queue_add("a", "1").startswith("list_add")
    assert Q.queue_add("a", "2").startswith("list_add")
    assert Q.queue_peek("a") == "1"
    assert Q.queue_remove("a") == "1"
    assert Q.queue_peek("a") == "2"
    assert Q.queue_remove("a") == "2"
    assert Q.queue_remove("a") == "<empty>"
    assert Q.queue_peek("a") == "<empty>"


def test_list_names_and_show_lists():
    Q.queue_add("z", "x")
    Q.queue_add("m", "y")
    assert "m\t1" in Q.queue_all_stats()
    assert "z\t1" in Q.queue_all_stats()
    out = Q.format_show_lists()
    assert "m" in out and "(1 items)" in out


def test_save_load_roundtrip(tmp_path):
    Q.queue_add("s", "hello")
    Q.queue_add("s", "world")
    p = tmp_path / "q.json"
    assert "wrote" in Q.queue_save("s", str(p)).lower()
    Q.reset_queues_for_testing()
    assert Q.queue_peek("s") == "<empty>"
    assert "loaded" in Q.queue_load("s", str(p)).lower()
    assert Q.queue_remove("s") == "hello"
    assert Q.queue_remove("s") == "world"


def test_tool_handlers_coerce_params():
    assert Q.tool_list_add({"listname": "L", "data": "hi"}).startswith("list_add")
    assert Q.tool_list_remove({"name": "L"}) == "hi"
    assert Q.tool_list_length({"name": "L"}) == "0"
    out = Q.tool_list_names({})
    assert "no lists" in out.lower() or "list_names" in out


def test_load_rejects_non_array(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"a": 1}', encoding="utf-8")
    assert "array" in Q.queue_load("x", str(p)).lower()


def test_repl_queue_show_lists(monkeypatch):
    Q.reset_queues_for_testing()
    Q.queue_add("demo", "x")
    ext = Path(__file__).resolve().parent.parent / "extensions" / "queue_control.py"
    lines = [
        f"/load {ext}",
        "/queue show lists",
        "/queue demo remove",
        "/unload",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "demo" in out and "1 items" in out
