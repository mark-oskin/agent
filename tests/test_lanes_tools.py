from __future__ import annotations

import json
import time
from pathlib import Path

from agentlib.tools import plugins


def test_plugin_toolset_registers_lanes_tool(tmp_path):
    plugins.load_plugin_toolsets(tools_dir=str(tmp_path), default_tools_dir=str(tmp_path))
    assert "agent_send" not in plugins.PLUGIN_TOOL_HANDLERS

    # Load default tools/ plugins (includes lanes.py).
    project_dir = Path(__file__).resolve().parent.parent
    plugins.load_plugin_toolsets(tools_dir=None, default_tools_dir=str(project_dir / "tools"))
    assert "lanes" in plugins.PLUGIN_TOOLSETS
    assert "agent_send" in plugins.PLUGIN_TOOL_HANDLERS
    assert plugins.PLUGIN_TOOL_TO_TOOLSET.get("agent_send") == "lanes"


def test_agent_send_requires_host_callbacks(monkeypatch):
    from tools import lanes

    lanes.set_lanes_host(enqueue_line=None, delegate_line=None)
    out = lanes.agent_send({"agent": "A", "line": "hi"})
    assert "no multi-agent host" in out.lower()


def test_agent_send_enqueue_nonblocking(monkeypatch):
    from tools import lanes

    lanes.set_lanes_host(
        enqueue_line=lambda agent, line: {"ok": True, "queued": True, "agent": agent, "line": line},
        delegate_line=None,
    )
    out = lanes.agent_send({"agent": "Worker", "line": "/show model"})
    data = json.loads(out)
    assert data["ok"] is True
    assert data["wait"] is False
    assert data["mode"] == "enqueue"
    assert data["result"]["ok"] is True


def test_agent_send_delegate_timeout(monkeypatch):
    from tools import lanes

    def slow_delegate(_agent: str, _line: str) -> dict:
        time.sleep(0.05)
        return {"type": "turn", "answered": True, "answer": "done"}

    lanes.set_lanes_host(enqueue_line=None, delegate_line=slow_delegate)
    out = lanes.agent_send({"agent": "Worker", "line": "hi", "wait": True, "timeout_ms": 1})
    data = json.loads(out)
    assert data["ok"] is False
    assert data["timeout"] is True

