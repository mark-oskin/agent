"""
Shared test harness: mock Ollama + stub tools for agent.main().
"""

from __future__ import annotations

import importlib
import io
import json
import sys
from contextlib import redirect_stdout
from typing import Any, Callable, List, Optional

import pytest


def reload_agent(monkeypatch: pytest.MonkeyPatch, **patches: Callable[..., Any]):
    if "agent" in sys.modules:
        del sys.modules["agent"]
    import agent as d  # noqa: WPS433

    importlib.reload(d)
    for attr, fn in patches.items():
        monkeypatch.setattr(d, attr, fn)
    return d


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
    """Run agent.main() capturing stdout."""
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

    d = reload_agent(monkeypatch, call_ollama_chat=fake_call_ollama_chat)
    monkeypatch.setattr(d, "_route_requires_websearch", fake_route_requires_websearch)
    monkeypatch.setattr(d, "_route_requires_websearch_after_answer", fake_route_after_answer)
    if stub_search_web is not None:
        def _search_web_wrap(query, params=None):  # noqa: ARG001
            return stub_search_web(query)

        monkeypatch.setattr(d, "search_web", _search_web_wrap)
    if stub_fetch_page is not None:
        monkeypatch.setattr(d, "fetch_page", stub_fetch_page)
    if stub_write_file is not None:
        monkeypatch.setattr(d, "write_file", stub_write_file)
    if stub_read_file is not None:
        monkeypatch.setattr(d, "read_file", stub_read_file)
    if stub_run_command is not None:
        monkeypatch.setattr(d, "run_command", stub_run_command)
    if stub_list_directory is not None:
        monkeypatch.setattr(d, "list_directory", stub_list_directory)
    if stub_tail_file is not None:
        monkeypatch.setattr(d, "tail_file", stub_tail_file)
    if stub_replace_text is not None:
        monkeypatch.setattr(d, "replace_text", stub_replace_text)
    if stub_download_file is not None:
        monkeypatch.setattr(d, "download_file", stub_download_file)
    if stub_call_python is not None:
        monkeypatch.setattr(d, "call_python", stub_call_python)

    monkeypatch.setattr(sys, "argv", ["agent.py", *argv])
    buf = io.StringIO()
    with redirect_stdout(buf):
        d.main()
    return buf.getvalue().strip()


def j(**kwargs: Any) -> str:
    """Shorthand for model JSON lines."""
    return json.dumps(kwargs, separators=(",", ":"))
