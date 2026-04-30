"""Shared coercion helpers for tool params and prefs."""

from __future__ import annotations

import json


def scalar_to_str(value, default=""):
    """Coerce tool parameters to str (models may emit numbers, lists, etc.)."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        parts = [scalar_to_str(x, "") for x in value]
        parts = [p for p in parts if p]
        return " ".join(parts) if parts else default
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def scalar_to_int(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def coerce_verbose_level(v) -> int:
    """0 = off, 1 = log tool invocations, 2 = log tools + stream model JSON (local Ollama)."""
    if isinstance(v, bool):
        return 2 if v else 0
    if v is None:
        return 0
    n = scalar_to_int(v, 0)
    if n < 0:
        return 0
    if n > 2:
        return 2
    return n
