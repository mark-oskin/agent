"""Load/save REPL context bundles (JSON message lists)."""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from agentlib.coercion import scalar_to_str

_VALID_CONTEXT_ROLES = frozenset({"user", "assistant", "system", "tool"})


def _coerce_message_content(content: Any) -> str:
    if content is None:
        return ""
    if not isinstance(content, str):
        return str(content)
    return content


def _parse_one_context_message(m: dict, index: int) -> dict:
    role = (m.get("role") or "").strip().lower()
    if role not in _VALID_CONTEXT_ROLES:
        raise ValueError(f"message {index}: invalid role {role!r}")
    content = _coerce_message_content(m.get("content"))
    row: dict = {"role": role, "content": content}
    if role == "assistant" and m.get("tool_calls"):
        row["tool_calls"] = m["tool_calls"]
    if role == "tool":
        tool_name = m.get("tool_name")
        if tool_name is not None and str(tool_name).strip():
            row["tool_name"] = str(tool_name).strip()
        tool_call_id = m.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id.strip():
            row["tool_call_id"] = tool_call_id.strip()
    return row


def parse_context_messages_data(raw) -> list:
    """Normalize JSON (bundle dict or bare list) into Ollama-style message dicts."""
    if isinstance(raw, dict) and isinstance(raw.get("messages"), list):
        msgs = raw["messages"]
    elif isinstance(raw, list):
        msgs = raw
    else:
        raise ValueError('context must be a JSON array of messages or {"messages": [...]}')
    out = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        out.append(_parse_one_context_message(m, i))
    if not out:
        raise ValueError("no valid messages in context file")
    return out


def load_context_messages(path: str, *, scalar_to_str_fn: Callable[..., str] = scalar_to_str) -> list:
    """Load a prior chat from JSON written by --save_context (or a bare list of messages)."""
    p = scalar_to_str_fn(path, "").strip()
    if not p:
        raise ValueError("empty path")
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return parse_context_messages_data(raw)


def save_context_bundle(
    path: str,
    messages: list,
    user_query: str,
    final_answer: Optional[str],
    answered: bool,
    *,
    scalar_to_str_fn: Callable[..., str] = scalar_to_str,
) -> None:
    """Persist full message list plus the new question and final answer (if any)."""
    p = scalar_to_str_fn(path, "").strip()
    if not p:
        raise ValueError("empty save path")
    bundle = {
        "version": 1,
        "user_query": user_query,
        "final_answer": final_answer,
        "answered": answered,
        "messages": messages,
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
