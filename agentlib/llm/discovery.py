"""Discover models from a running Ollama instance."""

from __future__ import annotations

from typing import Any, Callable


def fetch_ollama_local_model_names(
    base_url: str,
    *,
    http_get: Callable[..., Any],
    timeout: float = 60,
) -> list[str]:
    """Return sorted unique model names from GET /api/tags."""
    base = (base_url or "").strip().rstrip("/")
    r = http_get(f"{base}/api/tags", timeout=timeout)
    r.raise_for_status()
    data = r.json() or {}
    names = []
    for m in data.get("models") or []:
        n = (m.get("name") or "").strip()
        if n:
            names.append(n)
    return sorted(set(names))
