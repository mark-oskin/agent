from __future__ import annotations

import json
import threading
from typing import Any, Callable, Optional

_enqueue_line: Optional[Callable[[str, str], dict]] = None
_delegate_line: Optional[Callable[[str, str], dict]] = None


def set_lanes_host(
    *,
    enqueue_line: Optional[Callable[[str, str], dict]] = None,
    delegate_line: Optional[Callable[[str, str], dict]] = None,
) -> None:
    """
    Install host callbacks (normally from agent_tui).

    This plugin tool is meant to be used from a multi-agent host that can route a
    command line into another AgentSession lane.
    """

    global _enqueue_line, _delegate_line
    _enqueue_line = enqueue_line
    _delegate_line = delegate_line


def _missing_host_msg() -> str:
    return (
        "agent_send tool error: no multi-agent host configured. "
        "Run agent_tui.py or another host that wires enqueue/delegate lane hooks."
    )


def agent_send(params: dict) -> str:
    """
    Send one REPL line to another agent lane (agent_tui host).

    Params:
      - agent (string): lane label
      - line (string): one REPL line (prompt or slash command)
      - wait (bool, default false): wait for the other lane to finish that line
      - timeout_ms (int, optional): max wait for wait=true
    """
    p = params or {}
    agent = str(p.get("agent") or "").strip()
    line = str(p.get("line") or "").strip()
    wait = bool(p.get("wait", False))
    timeout_ms_raw = p.get("timeout_ms", None)
    timeout_ms: Optional[int]
    try:
        timeout_ms = int(timeout_ms_raw) if timeout_ms_raw is not None else None
    except (TypeError, ValueError):
        timeout_ms = None

    if not agent:
        return "agent_send tool error: missing required parameter: agent"
    if not line:
        return "agent_send tool error: missing required parameter: line"

    eq = _enqueue_line
    dl = _delegate_line
    if eq is None and dl is None:
        return _missing_host_msg()

    # Non-blocking: prefer enqueue.
    if not wait:
        if eq is None:
            # If only delegate is available, fall back to synchronous behavior.
            try:
                out = dl(agent, line) if dl is not None else {"ok": False, "error": _missing_host_msg()}
            except Exception as e:
                out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            return json.dumps({"ok": True, "wait": False, "mode": "delegate_fallback", "result": out}, ensure_ascii=False)
        try:
            out = eq(agent, line)
        except Exception as e:
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return json.dumps({"ok": True, "wait": False, "mode": "enqueue", "result": out}, ensure_ascii=False)

    # Blocking: requires delegate (or emulate with delegate fallback).
    if dl is None:
        return "agent_send tool error: wait=true requires a host delegate hook."

    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["result"] = dl(agent, line)
            box["ok"] = True
        except Exception as e:
            box["ok"] = False
            box["error"] = f"{type(e).__name__}: {e}"

    th = threading.Thread(target=worker, name="agent_send_delegate", daemon=True)
    th.start()
    if timeout_ms is not None and timeout_ms > 0:
        th.join(timeout_ms / 1000.0)
        if th.is_alive():
            return json.dumps(
                {
                    "ok": False,
                    "wait": True,
                    "timeout": True,
                    "timeout_ms": timeout_ms,
                    "agent": agent,
                },
                ensure_ascii=False,
            )
    else:
        th.join()
    if not box.get("ok", False):
        return json.dumps({"ok": False, "wait": True, "error": box.get("error", "unknown error")}, ensure_ascii=False)
    return json.dumps({"ok": True, "wait": True, "result": box.get("result")}, ensure_ascii=False)


TOOLSET = {
    "name": "lanes",
    "description": "Cross-lane control for agent_tui (send one REPL line to another lane).",
    "triggers": [],
    "tools": [
        {
            "id": "agent_send",
            "description": "Send one REPL line to another agent lane (agent_tui host).",
            "aliases": ("agent_send", "send_to_agent", "send agent", "lane_send"),
            "params": {
                "agent": "Target lane label (string).",
                "line": "One REPL line to execute on that lane (string).",
                "wait": "Optional: wait for completion (bool, default false).",
                "timeout_ms": "Optional: timeout for wait=true in ms (int).",
            },
            "returns": "JSON string: queued ack for wait=false, or target lane result for wait=true (may timeout).",
            "handler": agent_send,
        }
    ],
}

