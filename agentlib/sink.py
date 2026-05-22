"""Central output routing: optional typed emit callback (streaming) or stdout/stderr."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Optional

EmitCallable = Callable[[dict], None]

DRAFT_LABEL = "Draft:"
FINAL_LABEL = "Final:"

_emit_cv: ContextVar[Optional[EmitCallable]] = ContextVar("_agent_emit_sink", default=None)
_cli_answer_stream_buf: ContextVar[str] = ContextVar("_cli_answer_stream_buf", default="")
_cli_draft_header_printed: ContextVar[bool] = ContextVar("_cli_draft_header_printed", default=False)


def reset_cli_answer_display() -> None:
    """Clear CLI draft accumulation (call at start of each agent turn)."""
    _cli_answer_stream_buf.set("")
    _cli_draft_header_printed.set(False)


def cli_answer_display_nonempty() -> bool:
    return bool((_cli_answer_stream_buf.get() or "").strip())


def emit_sink_active() -> bool:
    return _emit_cv.get() is not None


@contextmanager
def emit_sink_scope(emit: Optional[EmitCallable]) -> Iterator[None]:
    """When `emit` is set, nested code should call `sink_emit` instead of `print`, for live streaming."""
    if emit is None:
        yield
        return
    token = _emit_cv.set(emit)
    try:
        yield
    finally:
        _emit_cv.reset(token)


def sink_print_compat(*args, sep: str = " ", end: str = "\n", flush: bool = True, file=None) -> None:
    """Behaves like ``print()`` but routes through ``sink_emit`` when an emit sink is active."""
    typ = "stderr" if file is sys.stderr else "output"
    text = sep.join(str(a) for a in args)
    sink_emit({"type": typ, "text": text, "end": end, "flush": flush})


def print_turn_final_answer(text: str) -> None:
    """Print the authoritative end-of-turn answer (CLI stdout or TUI ``final_answer`` emit)."""
    body = (text or "").strip()
    if not body:
        return
    fn = _emit_cv.get()
    if fn is not None:
        fn({"type": "final_answer", "text": body})
        return
    if _cli_draft_header_printed.get():
        print(file=sys.stdout)
    print(f"{FINAL_LABEL}\n{body}\n", end="", flush=True)


def sink_emit(ev: dict) -> None:
    """Deliver one structured event to the active emit sink, or print as legacy fallback."""
    fn = _emit_cv.get()
    typ = ev.get("type") or "output"
    text = ev.get("text")
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)
    end = ev.get("end", "\n")
    flush = bool(ev.get("flush", True))

    if fn is not None:
        if typ == "final_answer":
            fn({"type": "final_answer", "text": text.strip()})
            return
        payload = {"type": typ, "text": text}
        if end != "\n":
            payload["end"] = end
        if ev.get("partial"):
            payload["partial"] = True
        fn(payload)
        return

    fil = (
        sys.stderr
        if typ in ("progress", "stderr", "warning", "debug", "error")
        else sys.stdout
    )

    if typ == "final_answer":
        print_turn_final_answer(text)
        return

    if typ == "answer" and ev.get("partial"):
        from agentlib.llm.streaming import merge_visible_answer_text

        buf = _cli_answer_stream_buf.get()
        merged = merge_visible_answer_text(buf, text)
        if merged == buf:
            return
        delta = merged[len(buf) :]
        _cli_answer_stream_buf.set(merged)
        if not delta:
            return
        if not _cli_draft_header_printed.get():
            print(f"{DRAFT_LABEL}\n", end="", file=fil, flush=flush)
            _cli_draft_header_printed.set(True)
        print(delta, end=end, file=fil, flush=flush)
        return

    print(text, end=end, file=fil, flush=flush)


def sink_delegate_capture_append(ev: dict, buf: list[str]) -> None:
    """
    Append sink event text so delegated ``execute_line`` can return it in ``output`` (e.g. ``agent_send`` wait=true).

    Skips ``thinking`` / ``answer`` / ``final_answer`` (assistant turns expose ``answer`` on the result dict).
    """
    t = ev.get("type") or "output"
    if t in ("thinking", "answer", "final_answer"):
        return
    text = ev.get("text")
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)
    end = ev.get("end", "\n")
    if not isinstance(end, str):
        end = "\n"
    buf.append(text + end)
