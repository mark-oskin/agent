"""Load prompt template definitions from JSON files on disk."""

from __future__ import annotations

import json
import os
from typing import Callable, Optional


def load_prompt_templates_from_dir(dir_path: str) -> dict:
    """Load ``prompt_templates/*.json`` into a name → template-object map."""
    out: dict = {}
    if not os.path.isdir(dir_path):
        return out
    for fn in sorted(os.listdir(dir_path)):
        if not fn.endswith(".json") or fn.startswith("."):
            continue
        name, _ = os.path.splitext(fn)
        name = (name or "").strip()
        if not name:
            continue
        path = os.path.join(dir_path, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(obj, dict):
            out[name] = obj
    return out


def merge_prompt_templates(
    prefs: Optional[dict],
    *,
    resolved_prompt_templates_dir: Callable[[Optional[dict]], str],
    default_prompt_templates_dir: Callable[[], str],
) -> dict:
    """Load templates from the configured directory, then apply ~/.agent.json object overrides (user wins)."""
    dpath = resolved_prompt_templates_dir(prefs)
    base = load_prompt_templates_from_dir(dpath)
    if not base:
        base = load_prompt_templates_from_dir(default_prompt_templates_dir())
    if not prefs or not isinstance(prefs, dict):
        return base
    raw = prefs.get("prompt_templates")
    if not isinstance(raw, dict):
        return base
    out = dict(base)
    for name, obj in raw.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(obj, dict):
            continue
        out[name.strip()] = dict(obj)
    return out
