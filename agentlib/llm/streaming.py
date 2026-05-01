"""Merge Ollama /api/chat streaming JSON lines into one assistant message."""

from __future__ import annotations

import json
from typing import Callable, Optional, Tuple

from agentlib.sink import sink_emit


def merge_tool_arguments_delta(old_a, new_a):
    """Combine streamed `arguments` chunks without duplicating full JSON objects."""
    if old_a is None:
        return new_a
    if new_a is None:
        return old_a
    if isinstance(old_a, dict) and isinstance(new_a, dict):
        return {**old_a, **new_a}
    if isinstance(old_a, str) and isinstance(new_a, str):
        o, n = old_a.strip(), new_a.strip()
        if not o:
            return new_a
        if not n:
            return old_a
        merged = old_a + new_a
        for cand in (merged, new_a, old_a):
            try:
                json.loads(cand)
                return cand
            except json.JSONDecodeError:
                continue
        return merged
    if isinstance(new_a, dict):
        return new_a
    return old_a


def merge_partial_tool_calls(prev, new):
    """Merge streaming tool_call fragments (Ollama/OpenAI-style deltas)."""
    if not new:
        return prev or []
    if not prev:
        return new
    by_idx = {}
    for tc in prev:
        i = tc.get("index", 0)
        by_idx[i] = tc
    for tc in new:
        i = tc.get("index", 0)
        if i not in by_idx:
            by_idx[i] = tc
            continue
        old = by_idx[i]
        fn_old = (old.get("function") or {}) if isinstance(old.get("function"), dict) else {}
        fn_new = (tc.get("function") or {}) if isinstance(tc.get("function"), dict) else {}
        name = (fn_new.get("name") or fn_old.get("name") or "").strip()
        merged_args = merge_tool_arguments_delta(fn_old.get("arguments"), fn_new.get("arguments"))
        by_idx[i] = {
            **old,
            **tc,
            "function": {
                **fn_old,
                **fn_new,
                "name": name,
                "arguments": merged_args,
            },
        }
    return [by_idx[k] for k in sorted(by_idx.keys())]


def ollama_usage_from_chat_response(data: dict) -> Optional[dict]:
    """Extract token/duration stats Ollama includes on /api/chat (esp. final stream chunk)."""
    if not isinstance(data, dict):
        return None
    out: dict = {}
    for k in ("prompt_eval_count", "eval_count"):
        v = data.get(k)
        if isinstance(v, int) and v >= 0:
            out[k] = v
    for k in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
        v = data.get(k)
        if isinstance(v, int) and v >= 0:
            out[k] = v
    return out or None


def merge_stream_message_chunks(
    lines_iter,
    *,
    stream_chunks: bool = False,
    agent_stream_thinking_enabled: Callable[[], bool],
    ollama_usage_from_chat_response_fn: Callable[[dict], Optional[dict]] = ollama_usage_from_chat_response,
) -> Tuple[dict, Optional[dict], bool]:
    """Merge streaming /api/chat chunks into one assistant message dict + final usage dict."""
    acc = {"role": "assistant", "content": "", "thinking": ""}
    tool_calls = None
    usage: Optional[dict] = None
    streamed_content = False
    show_thinking = agent_stream_thinking_enabled()
    thinking_started = False
    done_thinking_banner_printed = False
    for line in lines_iter:
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = data.get("message") or {}
        if msg.get("content"):
            chunk = msg["content"]
            if show_thinking and thinking_started and not done_thinking_banner_printed:
                sink_emit({"type": "thinking", "text": "\n\n[Done thinking]\n", "end": "", "partial": True})
                done_thinking_banner_printed = True
            if stream_chunks:
                sink_emit({"type": "output", "text": chunk, "end": "", "partial": True})
                streamed_content = True
            acc["content"] += chunk
        if msg.get("thinking"):
            tchunk = msg["thinking"]
            acc["thinking"] += tchunk
            if show_thinking:
                if not thinking_started:
                    sink_emit({"type": "thinking", "text": "\n[Thinking]\n", "end": "", "partial": True})
                    thinking_started = True
                sink_emit({"type": "thinking", "text": tchunk, "end": "", "partial": True})
        if msg.get("tool_calls"):
            tool_calls = merge_partial_tool_calls(tool_calls, msg["tool_calls"])
        if data.get("done"):
            u = ollama_usage_from_chat_response_fn(data)
            if u:
                usage = u
            break
    if tool_calls is not None:
        acc["tool_calls"] = tool_calls
    return acc, usage, streamed_content
