"""Startup probe of local Ollama model names for REPL tab completion."""

from __future__ import annotations

from typing import Any, Callable, Optional


def probe_ollama_model_names(
    base_url: str,
    *,
    http_get: Optional[Callable[..., Any]] = None,
    timeout: float = 4.0,
) -> tuple[str, ...]:
    """
    Best-effort fetch of ``GET /api/tags`` names for tab completion.

    Returns an empty tuple when Ollama is unreachable or ``requests`` is unavailable.
    Intended to run once per process at REPL/TUI startup (short timeout).
    """
    try:
        import requests
    except ImportError:
        return ()
    get = http_get or requests.get
    from agentlib.llm.discovery import fetch_ollama_local_model_names

    try:
        names = fetch_ollama_local_model_names(base_url, http_get=get, timeout=timeout)
        return tuple(names)
    except Exception:
        return ()
