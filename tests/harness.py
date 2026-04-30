"""
Shared test harness: mock Ollama + stub tools for agent.main().
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from typing import Any, Callable, List, Optional

import pytest


def build_test_app(monkeypatch: pytest.MonkeyPatch):
    """
    Create a fresh app instance for tests.

    Tests should not depend on `agent.py` internals; `agent.py` is a shim.
    """
    from agentlib.app import default_app
    from agentlib import AgentSettings

    app = default_app()
    app.settings = AgentSettings.defaults()
    # Keep tests deterministic: ignore the developer's ~/.agent.json entirely.
    monkeypatch.setattr(app, "load_prefs", lambda: None)
    return app


def run_main(
    monkeypatch: pytest.MonkeyPatch,
    argv: List[str],
    responses: List[str],
    *,
    route_web: Optional[str] = None,
    route_after_answer: Optional[str] = None,
    stub_search_web: Optional[Callable[..., str]] = None,
    stub_fetch_page: Optional[Callable[..., str]] = None,
    stub_write_file: Optional[Callable[..., str]] = None,
    stub_read_file: Optional[Callable[..., str]] = None,
    stub_run_command: Optional[Callable[..., str]] = None,
    stub_list_directory: Optional[Callable[..., str]] = None,
    stub_tail_file: Optional[Callable[..., str]] = None,
    stub_replace_text: Optional[Callable[..., str]] = None,
    stub_download_file: Optional[Callable[..., str]] = None,
    stub_call_python: Optional[Callable[..., str]] = None,
) -> str:
    """Run app.run(argv) capturing stdout."""
    call_i = {"i": 0}

    def fake_call_ollama_chat(
        messages, primary_profile=None, enabled_tools=None, verbose=0, **kwargs
    ):  # noqa: ARG001
        idx = call_i["i"]
        call_i["i"] += 1
        if idx >= len(responses):
            return json.dumps({"action": "error", "error": f"no ollama response #{idx}"})
        return responses[idx]

    def fake_route_requires_websearch(
        user_query,
        today_str,
        primary_profile=None,
        enabled_tools=None,
        transcript_messages=None,
        **kwargs,
    ):  # noqa: ARG001
        return route_web

    def fake_route_after_answer(
        user_query,
        today_str,
        proposed_answer,
        primary_profile=None,
        enabled_tools=None,
        transcript_messages=None,
        **kwargs,
    ):  # noqa: ARG001
        return route_after_answer

    app = build_test_app(monkeypatch)
    from agentlib.tools import builtins as tool_builtins
    monkeypatch.setattr(app, "call_ollama_chat", fake_call_ollama_chat)
    monkeypatch.setattr(app, "route_requires_websearch", fake_route_requires_websearch)
    monkeypatch.setattr(app, "route_requires_websearch_after_answer", fake_route_after_answer)
    if stub_search_web is not None:
        monkeypatch.setattr(
            tool_builtins,
            "search_web",
            lambda query, params=None, settings=None: stub_search_web(query),
        )
    if stub_fetch_page is not None:
        monkeypatch.setattr(tool_builtins, "fetch_page", lambda url: stub_fetch_page(url))
    if stub_write_file is not None:
        monkeypatch.setattr(tool_builtins, "write_file", lambda path, content: stub_write_file(path, content))
    if stub_read_file is not None:
        monkeypatch.setattr(tool_builtins, "read_file", lambda path: stub_read_file(path))
    if stub_run_command is not None:
        monkeypatch.setattr(tool_builtins, "run_command", lambda command: stub_run_command(command))
    if stub_list_directory is not None:
        monkeypatch.setattr(tool_builtins, "list_directory", lambda path: stub_list_directory(path))
    if stub_tail_file is not None:
        monkeypatch.setattr(tool_builtins, "tail_file", lambda path, lines=20: stub_tail_file(path, lines=lines))
    if stub_replace_text is not None:
        monkeypatch.setattr(
            tool_builtins,
            "replace_text",
            lambda path, pattern, replacement, replace_all=True: stub_replace_text(
                path, pattern, replacement, replace_all=replace_all
            ),
        )
    if stub_download_file is not None:
        monkeypatch.setattr(tool_builtins, "download_file", lambda url, path: stub_download_file(url, path))
    if stub_call_python is not None:
        monkeypatch.setattr(tool_builtins, "call_python", lambda code, globals=None: stub_call_python(code, globals=globals))

    from agentlib.app import main as app_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        app_main(argv, app=app)
    return buf.getvalue().strip()


def j(**kwargs: Any) -> str:
    """Shorthand for model JSON lines."""
    return json.dumps(kwargs, separators=(",", ":"))
