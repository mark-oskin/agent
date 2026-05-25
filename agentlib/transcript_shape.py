"""Provider-shaped chat transcript helpers (Ollama tool_calls + tool role messages)."""

from __future__ import annotations

from typing import Any, Optional


def first_tool_call_id(tool_calls) -> Optional[str]:
    if not tool_calls:
        return None
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        tid = tc.get("id")
        if isinstance(tid, str) and tid.strip():
            return tid.strip()
    return None


def assistant_transcript_message(
    raw_ollama_msg: Optional[dict],
    *,
    fallback_content: str,
    use_provider_shape: bool,
) -> dict:
    """
    Build an assistant message for session transcript.

    When ``use_provider_shape`` and the model returned native ``tool_calls``,
    preserve them instead of storing normalized agent JSON in ``content``.
    """
    if use_provider_shape and isinstance(raw_ollama_msg, dict):
        tool_calls = raw_ollama_msg.get("tool_calls")
        if tool_calls:
            out: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
            content = raw_ollama_msg.get("content")
            if content is not None and str(content).strip():
                out["content"] = str(content)
            else:
                out["content"] = ""
            return out
    return {"role": "assistant", "content": fallback_content}


def tool_transcript_message(
    tool_name: str,
    result: str,
    *,
    tool_call_id: Optional[str] = None,
) -> dict:
    """Ollama/OpenAI-style tool result row for transcript + API replay."""
    msg: dict[str, Any] = {
        "role": "tool",
        "content": result if isinstance(result, str) else str(result),
        "tool_name": (tool_name or "").strip() or "unknown",
    }
    if tool_call_id:
        msg["tool_call_id"] = tool_call_id
    return msg


def append_turn_messages(
    messages: list,
    *,
    assistant_msg: dict,
    tool_msg: Optional[dict] = None,
    user_followup: Optional[str] = None,
) -> None:
    messages.append(assistant_msg)
    if tool_msg is not None:
        messages.append(tool_msg)
    if user_followup:
        messages.append({"role": "user", "content": user_followup})
