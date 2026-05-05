from __future__ import annotations

import json
from unittest.mock import MagicMock

from agentlib import agent_json
from agentlib.llm import streaming
from agentlib.llm.calls import call_ollama_chat
from agentlib.llm.profile import default_primary_llm_profile
from agentlib.tools import turn_support
from agentlib.tools.registry import ToolRegistry
import os


def test_verbose_3_emits_full_prompts(monkeypatch):
    from agentlib.llm import calls as calls_mod

    emitted: list[str] = []

    def cap(ev: dict) -> None:
        if (ev.get("type") or "") == "output":
            emitted.append(str(ev.get("text") or ""))

    monkeypatch.setattr(calls_mod, "sink_emit", cap)

    # Minimal deps for message_to_agent_json_text
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    reg = ToolRegistry(default_tools_dir=os.path.join(project_dir, "tools"))
    reg.load_plugin_toolsets(reg.default_tools_dir)
    reg.register_aliases()
    deps = agent_json.AgentJsonDeps(
        all_known_tools=reg.all_known_tools,
        coerce_enabled_tools=reg.coerce_enabled_tools,
        merge_tool_param_aliases=turn_support.merge_tool_param_aliases,
    )

    def msg_to_json(msg, enabled_tools=None):
        return agent_json.message_to_agent_json_text(msg, enabled_tools, deps)

    class _Resp:
        def __init__(self, line: str):
            self._line = line
            self.status_code = 200

        def raise_for_status(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def iter_lines(self, decode_unicode=True):
            yield self._line

    agent_line = json.dumps(
        {"message": {"content": '{"action":"answer","answer":"ok"}'}, "done": True}
    )

    def fake_post(url, json=None, **kwargs):
        assert kwargs.get("stream") is True
        return _Resp(agent_line)

    monkeypatch.setattr(calls_mod.requests, "post", fake_post)

    msgs = [{"role": "user", "content": "hello"}]
    raw = call_ollama_chat(
        msgs,
        primary_profile=default_primary_llm_profile(),
        enabled_tools=None,
        verbose=3,
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
        message_to_agent_json_text=msg_to_json,
        verbose_emit_final_agent_readable=lambda _t: None,
        format_ollama_usage_line=lambda _u: "",
        set_last_ollama_usage=lambda _u: None,
        call_hosted_agent_chat_impl=lambda *a, **k: "{}",
    )

    assert json.loads(raw).get("answer") == "ok"
    joined = "\n".join(emitted)
    assert "LLM request prompts" in joined
    assert '"content": "hello"' in joined

