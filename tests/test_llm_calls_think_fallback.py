"""Ollama /api/chat retries without ``think`` when the server returns HTTP 400."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from requests.exceptions import HTTPError

from agentlib import agent_json
from agentlib.llm import streaming
from agentlib.llm.calls import (
    _should_retry_ollama_chat_without_think,
    call_ollama_chat,
    call_ollama_plaintext,
)
from agentlib.llm.profile import LlmProfile, default_primary_llm_profile
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


def test_should_retry_helper():
    body = {"think": True}
    exc = HTTPError()
    exc.response = MagicMock(status_code=400)
    assert _should_retry_ollama_chat_without_think(exc, body) is True
    assert _should_retry_ollama_chat_without_think(exc, {**body, "think": False}) is False
    exc.response.status_code = 500
    assert _should_retry_ollama_chat_without_think(exc, body) is False


class _StreamResp:
    """Minimal streaming response for ``requests.post(..., stream=True)``."""

    def __init__(self, fail_raise: bool, line: str):
        self._fail = fail_raise
        self._line = line
        self.status_code = 400 if fail_raise else 200

    def raise_for_status(self):
        if self._fail:
            e = HTTPError()
            e.response = MagicMock(status_code=400)
            raise e

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def iter_lines(self, decode_unicode=True):
        yield self._line


def test_call_ollama_chat_retries_without_think_on_400(monkeypatch):
    from agentlib.llm import calls as calls_mod

    bodies: list[dict] = []
    agent_line = json.dumps({"message": {"content": '{"action":"answer","answer":"ok"}'}, "done": True})

    def fake_post(url, json=None, **kwargs):
        assert kwargs.get("stream") is True
        bodies.append(dict(json or {}))
        fail = json.get("think") is not False
        return _StreamResp(fail_raise=fail, line=agent_line)

    monkeypatch.setattr(calls_mod.requests, "post", fake_post)

    raw = call_ollama_chat(
        [{"role": "user", "content": "hi"}],
        primary_profile=default_primary_llm_profile(),
        enabled_tools=None,
        verbose=0,
        ollama_base_url="http://localhost:11434",
        ollama_model="dummy:latest",
        ollama_think_value=True,
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
    )

    assert len(bodies) == 2
    assert bodies[0].get("think") is True
    assert bodies[1].get("think") is False
    out = json.loads(raw)
    assert out.get("answer") == "ok"


def test_call_ollama_chat_routes_hosted_profile_to_openai_chat_completions(monkeypatch):
    from agentlib.llm import calls as calls_mod

    got: dict = {}

    class _OkResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"action":"answer","answer":"hosted ok"}',
                        }
                    }
                ]
            }

    def fake_post(url, json=None, **kwargs):
        got["url"] = url
        got["json"] = dict(json or {})
        assert kwargs.get("stream") is not True
        return _OkResp()

    monkeypatch.setattr(calls_mod.requests, "post", fake_post)

    raw = call_ollama_chat(
        [{"role": "user", "content": "hi"}],
        primary_profile=LlmProfile(
            backend="hosted",
            base_url="https://api.x.ai/v1",
            model="grok-test:latest",
            api_key="xai-key",
        ),
        enabled_tools=None,
        verbose=0,
        ollama_base_url="http://localhost:11434",
        ollama_model="dummy:latest",
        ollama_think_value=True,
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
        call_hosted_agent_chat_impl=calls_mod.call_hosted_agent_chat,
    )

    assert got["url"] == "https://api.x.ai/v1/chat/completions"
    assert got["json"].get("model") == "grok-test:latest"
    out = json.loads(raw)
    assert out.get("answer") == "hosted ok"


def test_call_ollama_plaintext_retries_without_think_on_400(monkeypatch):
    from agentlib.llm import calls as calls_mod

    bodies: list[dict] = []
    agent_line = json.dumps({"message": {"content": "plain ok"}, "done": True})

    def fake_post(url, json=None, **kwargs):
        bodies.append(dict(json or {}))
        fail = json.get("think") is not False
        return _StreamResp(fail_raise=fail, line=agent_line)

    monkeypatch.setattr(calls_mod.requests, "post", fake_post)

    text = call_ollama_plaintext(
        base_url="http://localhost:11434",
        messages=[{"role": "user", "content": "hi"}],
        model="dummy:latest",
        think_value="medium",
        merge_stream_message_chunks=lambda lines_iter, stream_chunks=False: streaming.merge_stream_message_chunks(
            lines_iter,
            stream_chunks=stream_chunks,
            agent_stream_thinking_enabled=lambda: False,
        ),
    )

    assert len(bodies) == 2
    assert bodies[0].get("think") == "medium"
    assert bodies[1].get("think") is False
    assert text == "plain ok"
