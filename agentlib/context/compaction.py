"""Optional compaction of chat history when approaching context limits."""

from __future__ import annotations

import re
from typing import Any, Callable, Optional, cast

from agentlib.llm.profile import LlmProfile


def approx_message_tokens(messages: list) -> int:
    # Heuristic: ~4 chars/token + small per-message overhead.
    total_chars = 0
    for m in messages:
        if isinstance(m, dict):
            c = m.get("content")
            if isinstance(c, str):
                total_chars += len(c)
    overhead = 8 * max(1, len(messages))
    return overhead + (total_chars // 4)


def context_limit_tokens(
    profile: Optional[LlmProfile],
    *,
    settings_get_int: Callable[[tuple, int], int],
) -> int:
    lim = settings_get_int(("agent", "context_tokens"), 0)
    if lim > 0:
        return lim
    if profile is not None and getattr(profile, "backend", "") == "hosted":
        return settings_get_int(("agent", "hosted_context_tokens"), 131072)
    return settings_get_int(("agent", "ollama_context_tokens"), 131072)


def summarize_conversation_for_context(
    *,
    profile: Optional[LlmProfile],
    user_query: str,
    text: str,
    call_hosted_chat_plain: Callable,
    call_ollama_plaintext: Callable[[list, str], str],
    ollama_model: str,
) -> str:
    prompt = (
        "Summarize the conversation so far to preserve long-term context for the assistant.\n"
        "Keep: user goals, non-negotiable constraints, decisions made, file names/paths, commands run, "
        "errors encountered, and next steps.\n"
        "Omit: chit-chat, repeated content, raw tool output unless it contains critical facts.\n"
        "Output plain text only (NOT JSON), max ~250-500 words.\n\n"
        f"User's current request:\n{user_query}\n\n"
        "Conversation to summarize:\n"
        f"{text}\n"
    )
    msgs = [
        {"role": "system", "content": "You are a summarizer. Output plain text only."},
        {"role": "user", "content": prompt},
    ]
    if profile is not None and profile.backend == "hosted":
        return call_hosted_chat_plain(msgs, profile)
    return call_ollama_plaintext(msgs, ollama_model)


def format_messages_for_summary(messages: list) -> str:
    lines = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        content = m.get("content")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        lines.append(f"{role}:\n{content}\n")
    return "\n".join(lines).strip()


def maybe_compact_context_window(
    messages: list,
    *,
    user_query: str,
    primary_profile: Optional[LlmProfile],
    verbose: int,
    context_cfg: Optional[dict] = None,
    settings_get_bool: Callable[[tuple, bool], bool],
    settings_get_int: Callable[[tuple, int], int],
    call_hosted_chat_plain: Callable,
    call_ollama_plaintext: Callable[[list, str], str],
    ollama_model: str,
    summarize_conversation_fn: Optional[Callable[..., str]] = None,
) -> list:
    cfg = context_cfg if isinstance(context_cfg, dict) else {}
    enabled = bool(cfg.get("enabled", True))
    if settings_get_bool(("agent", "disable_context_manager"), False):
        enabled = False
    if not enabled:
        return messages

    trigger_frac = float(cfg.get("trigger_frac", 0.75))
    target_frac = float(cfg.get("target_frac", 0.55))
    keep_tail = int(cfg.get("keep_tail_messages", 12))

    trigger_frac = max(0.05, min(0.95, trigger_frac))
    target_frac = max(0.05, min(trigger_frac, target_frac))
    keep_tail = max(4, keep_tail)

    limit = int(cfg.get("tokens", 0) or 0)
    if limit <= 0:
        limit = context_limit_tokens(primary_profile, settings_get_int=settings_get_int)
    if limit <= 0:
        return messages
    approx = approx_message_tokens(messages)
    if approx <= int(limit * trigger_frac):
        return messages

    head: list = []
    rest = list(messages)
    if rest and isinstance(rest[0], dict) and rest[0].get("role") == "system":
        head.append(rest[0])
        rest = rest[1:]
    if len(rest) <= keep_tail + 2:
        return messages

    tail = rest[-keep_tail:]
    to_summarize = rest[:-keep_tail]
    text = format_messages_for_summary(to_summarize)
    if not text.strip():
        return messages

    summarizer = summarize_conversation_fn or summarize_conversation_for_context
    summary = summarizer(
        profile=primary_profile,
        user_query=user_query,
        text=text,
        call_hosted_chat_plain=call_hosted_chat_plain,
        call_ollama_plaintext=call_ollama_plaintext,
        ollama_model=ollama_model,
    )
    summary = cast(str, summary).strip()
    if not summary:
        return messages

    summary_msg = {
        "role": "system",
        "content": (
            "Running conversation summary (auto-generated to fit context window):\n"
            f"{summary}"
        ),
    }
    new_messages = [*head, summary_msg, *tail]
    approx2 = approx_message_tokens(new_messages)
    if approx2 > int(limit * target_frac) and len(tail) > 6:
        new_messages = [*head, summary_msg, *tail[-6:]]
    if verbose >= 3:
        from agentlib.sink import sink_emit

        sink_emit(
            {
                "type": "debug",
                "text": (
                    f"[DEBUG] context manager compacted messages: ~{approx} -> "
                    f"~{approx_message_tokens(new_messages)} tokens"
                ),
            }
        )
    return new_messages


_RE_WORDS = re.compile(r"\S+")


def count_words(text: str) -> int:
    return len(_RE_WORDS.findall((text or "").strip()))


def _truncate_to_max_words(text: str, max_words: int) -> str:
    words = _RE_WORDS.findall((text or "").strip())
    if len(words) <= max_words:
        return (text or "").strip()
    return " ".join(words[:max_words]) + "\n…(truncated to word budget)\n"


def _truncate_to_approx_tokens(text: str, max_tokens: int) -> str:
    cap = max(200, int(max_tokens) * 4)
    t = (text or "").strip()
    if len(t) <= cap:
        return t
    return t[: cap - 40].rstrip() + "\n…(truncated to token budget)\n"


def llm_compress_transcript_for_repl(
    *,
    profile: Optional[LlmProfile],
    transcript: str,
    constraint_kind: str,
    constraint_value: int,
    call_hosted_chat_plain: Callable[[list, Any], str],
    call_ollama_plaintext: Callable[[list, str], str],
    ollama_model: str,
) -> str:
    """
    Ask the primary LLM to compress a transcript for continued conversation.

    ``constraint_kind`` is ``"approx_tokens"`` (aim under ~N heuristic tokens) or ``"max_words"``
    (hard cap on words in the model output).
    """
    t = (transcript or "").strip()
    if not t:
        return ""

    if constraint_kind == "max_words":
        n = max(1, int(constraint_value))
        constraint_line = (
            f"Your entire reply must be at most {n} words. "
            "Count words in the final text only; stay under the limit."
        )
        post_trim = lambda s: _truncate_to_max_words(s, n)
    elif constraint_kind == "approx_tokens":
        tok = max(32, int(constraint_value))
        approx_chars = max(200, tok * 4)
        constraint_line = (
            f"Your entire reply must fit within approximately {tok} tokens "
            f"(aim well under ~{approx_chars} characters of body text; be concise)."
        )
        post_trim = lambda s: _truncate_to_approx_tokens(s, tok)
    else:
        raise ValueError(f"unknown constraint_kind: {constraint_kind!r}")

    prompt = (
        "You are compressing a prior chat transcript so the assistant can continue with less context.\n\n"
        "Preserve: user goals, constraints, decisions, file paths, commands run, errors, open questions, "
        "and technical facts the user cares about.\n"
        "Omit: chit-chat, repeated content, and raw tool dumps unless a short conclusion is impossible "
        "without them.\n\n"
        "Write plain text or light markdown as one cohesive summary of what matters for the next turns "
        "(not a screenplay unless necessary).\n\n"
        f"{constraint_line}\n\n"
        "Do not preface with phrases like 'Here is a summary'. Do not comment on these instructions.\n\n"
        "Transcript to compress:\n---\n"
        f"{t}\n---\n"
    )
    msgs = [
        {"role": "system", "content": "You compress conversation transcripts. Output only the compressed text."},
        {"role": "user", "content": prompt},
    ]
    if profile is not None and getattr(profile, "backend", "") == "hosted":
        raw = call_hosted_chat_plain(msgs, profile)
    else:
        raw = call_ollama_plaintext(msgs, ollama_model)
    out = cast(str, raw).strip()
    if not out:
        return ""
    return post_trim(out)
