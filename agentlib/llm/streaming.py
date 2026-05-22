"""Merge Ollama /api/chat and hosted OpenAI-style SSE streams into one assistant message."""

from __future__ import annotations

import codecs
import json
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional, Tuple

from agentlib.agent_json import _extract_json_string_field
from agentlib.llm.gen_rate import GenRateTracker
from agentlib.llm.token_estimate import CharsPerTokenEstimator, estimate_tokens_from_text
from agentlib.sink import sink_emit

_assistant_answer_streamed: ContextVar[bool] = ContextVar("_assistant_answer_streamed", default=False)

_TOOL_ACTION_RE = re.compile(r'"action"\s*:\s*"tool_call"', re.IGNORECASE)


def reset_assistant_answer_streamed() -> None:
    from agentlib.sink import reset_cli_answer_display

    _assistant_answer_streamed.set(False)
    reset_cli_answer_display()


def assistant_answer_was_streamed() -> bool:
    from agentlib.sink import cli_answer_display_nonempty

    return bool(_assistant_answer_streamed.get()) or cli_answer_display_nonempty()


def iter_ollama_ndjson_lines_from_response(response) -> Iterator[str]:
    """
    Yield complete newline-delimited JSON lines from a streamed Ollama ``/api/chat`` body.

    Unlike ``Response.iter_lines``, this reassembles UTF-8 across arbitrary TCP chunk
    boundaries (multi-byte characters and NDJSON records are not split incorrectly)
    and yields the last record even when the server closes without a trailing newline.
    """
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    buf = ""
    for chunk in response.iter_content(chunk_size=65536, decode_unicode=False):
        if not chunk:
            continue
        if not isinstance(chunk, bytes):
            chunk = bytes(chunk)
        buf += decoder.decode(chunk, final=False)
        while True:
            idx = buf.find("\n")
            if idx < 0:
                break
            raw_line = buf[:idx]
            buf = buf[idx + 1 :]
            line = raw_line.rstrip("\r").strip()
            if line:
                yield line
    buf += decoder.decode(b"", final=True)
    tail = buf.rstrip("\r\n").strip()
    if tail:
        yield tail


def iter_openai_sse_data_objects(response) -> Iterator[dict]:
    """Yield parsed JSON objects from an OpenAI-style ``data: {...}`` SSE body."""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    buf = ""
    for chunk in response.iter_content(chunk_size=65536, decode_unicode=False):
        if not chunk:
            continue
        if not isinstance(chunk, bytes):
            chunk = bytes(chunk)
        buf += decoder.decode(chunk, final=False)
        while True:
            idx = buf.find("\n")
            if idx < 0:
                break
            line = buf[:idx].rstrip("\r")
            buf = buf[idx + 1 :]
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                yield {"_sse_done": True}
                return
            if not payload:
                continue
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue
    buf += decoder.decode(b"", final=True)
    for line in buf.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            yield {"_sse_done": True}
            return
        if not payload:
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


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


def merge_stream_content(acc: str, chunk: str) -> str:
    """
    Merge streamed LLM message fragments (JSON bodies, thinking text).

    Conservative rules only — do not drop chunks that happen to appear as
    substrings elsewhere in partial JSON (e.g. ``tool_call``).

    When the model starts a fresh ``{...}`` object instead of extending ``acc``,
    replace the buffer rather than concatenating (avoids glued JSON in Draft extract).
    """
    if not chunk:
        return acc
    if not acc:
        return chunk
    if chunk.startswith(acc):
        return chunk
    if acc.startswith(chunk):
        return acc
    if _stream_content_is_fresh_json_object(acc, chunk):
        return chunk
    return acc + chunk


def _stream_content_is_fresh_json_object(acc: str, chunk: str) -> bool:
    """True when ``chunk`` begins a new top-level JSON object, not a continuation of ``acc``."""
    if not acc or not chunk:
        return False
    if chunk.startswith(acc) or acc.startswith(chunk):
        return False
    return chunk.lstrip().startswith("{")


def merge_visible_answer_text(acc: str, chunk: str) -> str:
    """
    Merge user-visible answer tokens for live TUI/REPL display.

    Handles cumulative chunks, suffix-prefix overlap, and model restarts
    (e.g. acc ``2 + 2 equals 4. `` plus chunk ``+ 2 equals 4.``).
    """
    if not chunk:
        return acc
    if not acc:
        return chunk
    if chunk.startswith(acc):
        return chunk
    if acc.startswith(chunk):
        return acc
    if acc.endswith(chunk):
        return acc
    max_k = min(len(acc), len(chunk))
    for k in range(max_k, 1, -1):
        if acc[-k:] == chunk[:k]:
            return acc + chunk[k:]
    if len(chunk) >= 4 and chunk in acc:
        return acc
    return acc + chunk


def _merge_stream_content(acc: str, chunk: str) -> str:
    return merge_stream_content(acc, chunk)


def _content_indicates_tool_call(content: str, tool_calls) -> bool:
    if tool_calls:
        return True
    head = (content or "")[:4096]
    if _TOOL_ACTION_RE.search(head):
        return True
    if re.search(r'"tool_calls"\s*:', head):
        return True
    return False


@dataclass
class _VisibleStreamState:
    answer_emitted_len: int = 0
    answer_emitted_snapshot: str = ""
    tool_mode: Optional[bool] = None
    tool_progress_emitted: bool = False
    streamed_content: bool = False
    gen_rate: GenRateTracker = field(default_factory=GenRateTracker)
    last_eval_count: int = 0
    has_ollama_eval: bool = False
    cpt_estimator: Optional[CharsPerTokenEstimator] = None

    def reset_answer_stream(self, *, stream_user_visible: bool = False) -> None:
        """Forget Draft deltas after the model restarts its JSON object."""
        self.answer_emitted_len = 0
        self.answer_emitted_snapshot = ""
        self.tool_mode = None
        self.tool_progress_emitted = False
        if stream_user_visible:
            sink_emit({"type": "answer_reset"})


def _emit_gen_rate(state: _VisibleStreamState, *, stream_user_visible: bool) -> None:
    if not stream_user_visible:
        return
    rate = state.gen_rate.sample_interval(min_elapsed=0.2)
    if rate is not None:
        sink_emit({"type": "gen_rate", "tok_per_sec": rate})


def _record_gen_tokens(
    state: _VisibleStreamState, n: int, *, stream_user_visible: bool
) -> None:
    if not stream_user_visible or n <= 0:
        return
    state.gen_rate.add_tokens(n)


def _record_thinking_chunk_tokens(
    tchunk: str, state: _VisibleStreamState, *, stream_user_visible: bool
) -> None:
    """Count streamed reasoning tokens toward gen rate (Ollama eval_count already tracks)."""
    if not tchunk or state.has_ollama_eval:
        return
    _record_gen_tokens(
        state,
        estimate_tokens_from_text(tchunk, estimator=state.cpt_estimator),
        stream_user_visible=stream_user_visible,
    )


def _record_ollama_eval_from_chunk(
    data: dict,
    state: _VisibleStreamState,
    *,
    stream_user_visible: bool,
    ollama_usage_from_chat_response_fn: Callable[[dict], Optional[dict]],
) -> None:
    if not stream_user_visible:
        return
    usage = ollama_usage_from_chat_response_fn(data)
    if not usage or "eval_count" not in usage:
        return
    ec = usage["eval_count"]
    if not isinstance(ec, int) or ec < state.last_eval_count:
        return
    delta = ec - state.last_eval_count
    state.last_eval_count = ec
    state.has_ollama_eval = True
    _record_gen_tokens(state, delta, stream_user_visible=stream_user_visible)


def _emit_visible_answer_delta(
    content: str, state: _VisibleStreamState, *, stream_user_visible: bool
) -> None:
    answer = _extract_json_string_field(
        content, "answer", allow_unterminated=True, use_last=True
    )
    if answer is None:
        return
    if state.answer_emitted_len > 0:
        if len(answer) < state.answer_emitted_len:
            state.reset_answer_stream(stream_user_visible=stream_user_visible)
        elif (
            state.answer_emitted_snapshot
            and not answer.startswith(state.answer_emitted_snapshot)
        ):
            state.reset_answer_stream(stream_user_visible=stream_user_visible)
    if len(answer) <= state.answer_emitted_len:
        return
    delta = answer[state.answer_emitted_len :]
    state.answer_emitted_len = len(answer)
    state.answer_emitted_snapshot = answer
    if not delta:
        return
    sink_emit(
        {
            "type": "answer",
            "text": answer,
            "end": "",
            "partial": True,
            "flush": True,
            "full_snapshot": True,
        }
    )
    _assistant_answer_streamed.set(True)
    state.streamed_content = True
    if not state.has_ollama_eval and not state.tool_mode:
        _record_gen_tokens(
            state,
            estimate_tokens_from_text(delta, estimator=state.cpt_estimator),
            stream_user_visible=stream_user_visible,
        )


def _apply_content_chunk(
    acc_content: str,
    chunk: str,
    *,
    stream_chunks: bool,
    stream_user_visible: bool,
    state: _VisibleStreamState,
    tool_calls,
) -> None:
    if stream_chunks:
        sink_emit({"type": "output", "text": chunk, "end": "", "partial": True})
        state.streamed_content = True
    if not stream_user_visible or not chunk:
        return
    if state.tool_mode is None and _content_indicates_tool_call(acc_content, tool_calls):
        state.tool_mode = True
    if state.tool_mode:
        if not state.tool_progress_emitted:
            sink_emit({"type": "progress", "text": "Preparing tool call…"})
            state.tool_progress_emitted = True
        return
    state.tool_mode = False
    _emit_visible_answer_delta(acc_content, state, stream_user_visible=stream_user_visible)


def merge_stream_message_chunks(
    lines_iter,
    *,
    stream_chunks: bool = False,
    stream_user_visible: bool = False,
    agent_stream_thinking_enabled: Callable[[], bool],
    ollama_usage_from_chat_response_fn: Callable[[dict], Optional[dict]] = ollama_usage_from_chat_response,
    chars_per_token_estimator: Optional[CharsPerTokenEstimator] = None,
) -> Tuple[dict, Optional[dict], bool]:
    """Merge streaming /api/chat chunks into one assistant message dict + final usage dict."""
    acc = {"role": "assistant", "content": "", "thinking": ""}
    tool_calls = None
    usage: Optional[dict] = None
    show_thinking = agent_stream_thinking_enabled()
    thinking_started = False
    done_thinking_banner_printed = False
    saw_done = False
    vis = _VisibleStreamState(cpt_estimator=chars_per_token_estimator)
    for line in lines_iter:
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            preview = line if len(line) <= 240 else line[:240] + "…"
            sink_emit(
                {
                    "type": "warning",
                    "text": f"Ollama stream line ignored (invalid JSON): {e}; line ({len(line)} chars): {preview}",
                }
            )
            continue
        _record_ollama_eval_from_chunk(
            data,
            vis,
            stream_user_visible=stream_user_visible,
            ollama_usage_from_chat_response_fn=ollama_usage_from_chat_response_fn,
        )
        msg = data.get("message") or {}
        if msg.get("content"):
            chunk = msg["content"]
            if show_thinking and thinking_started and not done_thinking_banner_printed:
                sink_emit({"type": "thinking", "text": "\n\n[Done thinking]\n", "end": "", "partial": True})
                done_thinking_banner_printed = True
            prev_content = acc["content"]
            merged_content = _merge_stream_content(prev_content, chunk)
            if _stream_content_is_fresh_json_object(prev_content, chunk) and merged_content == chunk:
                vis.reset_answer_stream(stream_user_visible=stream_user_visible)
            acc["content"] = merged_content
            _apply_content_chunk(
                acc["content"],
                chunk,
                stream_chunks=stream_chunks,
                stream_user_visible=stream_user_visible,
                state=vis,
                tool_calls=tool_calls,
            )
        if msg.get("thinking"):
            tchunk = msg["thinking"]
            acc["thinking"] = _merge_stream_content(acc["thinking"], tchunk)
            if show_thinking:
                if not thinking_started:
                    sink_emit({"type": "thinking", "text": "\n[Thinking]\n", "end": "", "partial": True})
                    thinking_started = True
                sink_emit({"type": "thinking", "text": tchunk, "end": "", "partial": True})
            _record_thinking_chunk_tokens(tchunk, vis, stream_user_visible=stream_user_visible)
        if msg.get("tool_calls"):
            tool_calls = merge_partial_tool_calls(tool_calls, msg["tool_calls"])
            if stream_user_visible and vis.tool_mode is not False:
                vis.tool_mode = True
        if data.get("done"):
            saw_done = True
            u = ollama_usage_from_chat_response_fn(data)
            if u:
                usage = u
            _emit_gen_rate(vis, stream_user_visible=stream_user_visible)
            break
    if not saw_done:
        sink_emit(
            {
                "type": "warning",
                "text": "Ollama stream ended without a terminal chunk with done:true; "
                "the HTTP body was fully read but the protocol sequence may be incomplete.",
            }
        )
    if tool_calls is not None:
        acc["tool_calls"] = tool_calls
    if stream_user_visible and vis.answer_emitted_len > 0:
        sink_emit({"type": "output", "text": "", "end": "\n", "flush": True})
    if chars_per_token_estimator is not None:
        CharsPerTokenEstimator.observe_from_assistant_message(
            chars_per_token_estimator, acc, usage
        )
    return acc, usage, vis.streamed_content


def merge_hosted_stream_chunks(
    sse_iter,
    *,
    stream_chunks: bool = False,
    stream_user_visible: bool = False,
    agent_stream_thinking_enabled: Callable[[], bool],
    chars_per_token_estimator: Optional[CharsPerTokenEstimator] = None,
) -> Tuple[dict, Optional[dict], bool]:
    """Merge OpenAI-style chat completion SSE into one assistant message dict."""
    acc = {"role": "assistant", "content": "", "thinking": ""}
    tool_calls = None
    show_thinking = agent_stream_thinking_enabled()
    thinking_started = False
    done_thinking_banner_printed = False
    vis = _VisibleStreamState(cpt_estimator=chars_per_token_estimator)
    for data in sse_iter:
        if not isinstance(data, dict):
            continue
        if data.get("_sse_done"):
            break
        choices = data.get("choices") or []
        if not choices:
            continue
        choice0 = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice0.get("delta") or {}
        if not isinstance(delta, dict):
            delta = {}
        chunk = delta.get("content") or ""
        if chunk:
            if show_thinking and thinking_started and not done_thinking_banner_printed:
                sink_emit({"type": "thinking", "text": "\n\n[Done thinking]\n", "end": "", "partial": True})
                done_thinking_banner_printed = True
            prev_content = acc["content"]
            merged_content = _merge_stream_content(prev_content, chunk)
            if _stream_content_is_fresh_json_object(prev_content, chunk) and merged_content == chunk:
                vis.reset_answer_stream(stream_user_visible=stream_user_visible)
            acc["content"] = merged_content
            _apply_content_chunk(
                acc["content"],
                chunk,
                stream_chunks=stream_chunks,
                stream_user_visible=stream_user_visible,
                state=vis,
                tool_calls=tool_calls,
            )
        for think_key in ("reasoning_content", "thinking"):
            tchunk = delta.get(think_key) or ""
            if tchunk:
                acc["thinking"] = _merge_stream_content(acc["thinking"], tchunk)
                if show_thinking:
                    if not thinking_started:
                        sink_emit({"type": "thinking", "text": "\n[Thinking]\n", "end": "", "partial": True})
                        thinking_started = True
                    sink_emit({"type": "thinking", "text": tchunk, "end": "", "partial": True})
                _record_thinking_chunk_tokens(tchunk, vis, stream_user_visible=stream_user_visible)
        if delta.get("tool_calls"):
            tool_calls = merge_partial_tool_calls(tool_calls, delta["tool_calls"])
            if stream_user_visible:
                vis.tool_mode = True
        if choice0.get("finish_reason") is not None:
            _emit_gen_rate(vis, stream_user_visible=stream_user_visible)
            break
    if tool_calls is not None:
        acc["tool_calls"] = tool_calls
    if stream_user_visible and vis.answer_emitted_len > 0:
        sink_emit({"type": "output", "text": "", "end": "\n", "flush": True})
    return acc, None, vis.streamed_content
