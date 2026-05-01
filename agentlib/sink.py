"""Central output routing: optional typed emit callback (streaming) or stdout/stderr."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Optional

EmitCallable = Callable[[dict], None]

_emit_cv: ContextVar[Optional[EmitCallable]] = ContextVar("_agent_emit_sink", default=None)


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
    print(text, end=end, file=fil, flush=flush)
