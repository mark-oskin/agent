"""Discover models from a running Ollama instance."""

from __future__ import annotations

from typing import Any, Callable


def fetch_ollama_model_show(
    base_url: str,
    model_name: str,
    *,
    http_post: Callable[..., Any],
    timeout: float = 60,
) -> dict[str, Any]:
    """Return the JSON body from ``POST /api/show`` (model details / Modelfile / parameters)."""
    base = (base_url or "").strip().rstrip("/")
    name = (model_name or "").strip()
    if not name:
        raise ValueError("model name is empty")
    r = http_post(
        f"{base}/api/show",
        json={"name": name},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, dict) else {}


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
