"""Optional append-only log for full LLM request payloads (``--debug_log``)."""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Optional

_lock = threading.Lock()
_path: Optional[str] = None


def set_debug_llm_log_path(path: Optional[str]) -> None:
    """Set path for LLM prompt logging, or ``None`` to disable."""
    global _path
    with _lock:
        if path is None or not str(path).strip():
            _path = None
        else:
            _path = os.path.abspath(os.path.expanduser(str(path).strip()))


def get_debug_llm_log_path() -> Optional[str]:
    with _lock:
        return _path


def debug_llm_log_enabled() -> bool:
    return get_debug_llm_log_path() is not None


def append_llm_prompt_log(text: str) -> None:
    """Append one block (caller supplies trailing newlines as needed)."""
    p = get_debug_llm_log_path()
    if not p or not text:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sep = f"\n--- LLM request ({stamp}) ---\n"
    parent = os.path.dirname(p)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except Exception:
            pass
    try:
        with open(p, "a", encoding="utf-8") as f:
            f.write(sep)
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
    except Exception:
        return
