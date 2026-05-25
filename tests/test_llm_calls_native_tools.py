"""Ollama native tools API (Phase 1) in call_ollama_chat."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from agentlib import agent_json
from agentlib.llm import streaming
from agentlib.llm.calls import call_ollama_chat, normalize_tool_call_mode
from agentlib.llm.profile import default_primary_llm_profile
from agentlib.tools import turn_support
from agentlib.tools.registry import ToolRegistry
import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REG = ToolRegistry(default_tools_dir=os.path.join(PROJECT_DIR, "tools"))
_REG.load_plugin_toolsets(_REG.default_tools_dir)
_REG.register_aliases()
_JSON_DEPS = agent_json.AgentJsonDeps(
    all_known_tools=_REG.all_known_tools,
    coerce_enabled_tools=_REG.coerce_enabled_tools,
    merge_tool_param_aliases=turn_support.merge_tool_param_aliases,
)


def _msg_to_json(msg, enabled_tools=None):
    return agent_json.message_to_agent_json_text(msg, enabled_tools, _JSON_DEPS)


class _StreamResp:
    def __init__(self, line: str):
        self._line = line

    def raise_for_status(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def iter_lines(self, decode_unicode=True):
        yield self._line

    def iter_content(self, chunk_size=65536, decode_unicode=False):
        yield self._line.encode("utf-8")


def _call_native(monkeypatch, *, line: str, enabled_tools, mode: str = "native"):
    from agentlib.llm import calls as calls_mod

    bodies: list[dict] = []

    def fake_post(url, json=None, **kwargs):
        bodies.append(dict(json or {}))
        return _StreamResp(line)

    monkeypatch.setattr(calls_mod.requests, "post", fake_post)

    raw = call_ollama_chat(
        [{"role": "user", "content": "search for cats"}],
        primary_profile=default_primary_llm_profile(),
        enabled_tools=enabled_tools,
        verbose=0,
        ollama_base_url="http://localhost:11434",
        ollama_model="dummy:latest",
        ollama_think_value=False,
        ollama_debug=False,
        merge_stream_message_chunks=lambda lines_iter, stream_chunks=False: streaming.merge_stream_message_chunks(
            lines_iter,
            stream_chunks=stream_chunks,
            agent_stream_thinking_enabled=lambda: False,
        ),
        ollama_usage_from_chat_response=streaming.ollama_usage_from_chat_response,
        message_to_agent_json_text=_msg_to_json,
        verbose_emit_final_agent_readable=lambda _t: None,
        format_ollama_usage_line=lambda _u: "",
        set_last_ollama_usage=lambda _u: None,
        call_hosted_agent_chat_impl=lambda *a, **k: "{}",
        ollama_tool_call_mode=mode,
    )
    return bodies, raw


def test_normalize_tool_call_mode():
    assert normalize_tool_call_mode("native") == "native"
    assert normalize_tool_call_mode("JSON") == "json"
    assert normalize_tool_call_mode(None) == "native"


def test_native_mode_sends_tools_without_format_json(monkeypatch):
    native_line = json.dumps(
        {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_web",
                            "arguments": json.dumps({"query": "cats"}),
                        }
                    }
                ]
            },
            "done": True,
        }
    )
    bodies, raw = _call_native(
        monkeypatch,
        line=native_line,
        enabled_tools=frozenset({"search_web"}),
    )
    assert len(bodies) == 1
    assert "tools" in bodies[0]
    assert "format" not in bodies[0]
    out = json.loads(raw)
    assert out["action"] == "tool_call"
    assert out["tool"] == "search_web"
    assert out["parameters"]["query"] == "cats"


def test_native_mode_falls_back_to_json_when_unusable(monkeypatch):
    empty_answer_line = json.dumps(
        {
            "message": {"content": '{"action":"answer","answer":""}'},
            "done": True,
        }
    )
    json_line = json.dumps(
        {
            "message": {"content": '{"action":"answer","answer":"ok"}'},
            "done": True,
        }
    )
    from agentlib.llm import calls as calls_mod

    bodies: list[dict] = []
    calls = iter([empty_answer_line, json_line])

    def fake_post(url, json=None, **kwargs):
        bodies.append(dict(json or {}))
        return _StreamResp(next(calls))

    monkeypatch.setattr(calls_mod.requests, "post", fake_post)

    raw = call_ollama_chat(
        [{"role": "user", "content": "hi"}],
        primary_profile=default_primary_llm_profile(),
        enabled_tools=frozenset({"search_web"}),
        verbose=0,
        ollama_base_url="http://localhost:11434",
        ollama_model="dummy:latest",
        ollama_think_value=False,
        ollama_debug=False,
        merge_stream_message_chunks=lambda lines_iter, stream_chunks=False: streaming.merge_stream_message_chunks(
            lines_iter,
            stream_chunks=stream_chunks,
            agent_stream_thinking_enabled=lambda: False,
        ),
        ollama_usage_from_chat_response=streaming.ollama_usage_from_chat_response,
        message_to_agent_json_text=_msg_to_json,
        verbose_emit_final_agent_readable=lambda _t: None,
        format_ollama_usage_line=lambda _u: "",
        set_last_ollama_usage=lambda _u: None,
        call_hosted_agent_chat_impl=lambda *a, **k: "{}",
        ollama_tool_call_mode="native",
    )
    assert len(bodies) == 2
    assert "tools" in bodies[0]
    assert "format" not in bodies[0]
    assert bodies[1].get("format") == "json"
    assert "tools" not in bodies[1]
    assert json.loads(raw).get("answer") == "ok"


def test_json_mode_unchanged(monkeypatch):
    json_line = json.dumps(
        {
            "message": {"content": '{"action":"answer","answer":"hi"}'},
            "done": True,
        }
    )
    bodies, raw = _call_native(
        monkeypatch,
        line=json_line,
        enabled_tools=frozenset({"search_web"}),
        mode="json",
    )
    assert len(bodies) == 1
    assert bodies[0].get("format") == "json"
    assert "tools" not in bodies[0]
    assert json.loads(raw).get("answer") == "hi"
