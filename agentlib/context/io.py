"""Load/save REPL context bundles (JSON message lists)."""

from __future__ import annotations

import json
from typing import Callable, Optional

from agentlib.coercion import scalar_to_str


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
        role = (m.get("role") or "").strip()
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"message {i}: invalid role {role!r}")
        content = m.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        out.append({"role": role, "content": content})
    if not out:
        raise ValueError("no valid messages in context file")
    return out


def load_context_messages(path: str, *, scalar_to_str_fn: Callable[..., str] = scalar_to_str) -> list:
    """Load a prior chat from JSON written by --save_context (or a bare list of {role, content})."""
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
