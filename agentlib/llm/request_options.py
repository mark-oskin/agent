"""Sanitize profile ``request_options`` for Ollama vs OpenAI-compatible backends."""

from __future__ import annotations

import json
import numbers
from typing import Any, Mapping

# Keys the agent sets on the hosted body; user must not override these.
_HOSTED_BODY_RESERVED = frozenset({"messages", "model", "stream"})

# Typical OpenAI-style keys (server may accept more; unknown keys are forwarded if JSON-serializable).
_HOSTED_ALLOWED = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "stop",
        "n",
        "logit_bias",
        "user",
        "response_format",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning_effort",
        "service_tier",
    }
)

# Common Ollama runner / Modelfile-style option keys (nested under ``options`` on /api/chat).
_OLLAMA_ALLOWED = frozenset(
    {
        "num_ctx",
        "num_predict",
        "num_keep",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "typical_p",
        "tfs_z",
        "repeat_penalty",
        "repeat_last_n",
        "frequency_penalty",
        "presence_penalty",
        "mirostat",
        "mirostat_eta",
        "mirostat_tau",
        "penalize_newline",
        "stop",
        "seed",
    }
)


def normalize_request_options_pref(obj: Any) -> dict[str, Any]:
    """Load ``request_options`` from prefs JSON; shallow copy with string keys only."""
    if not isinstance(obj, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in obj.items():
        ks = str(k).strip()
        if not ks:
            continue
        if _is_json_leaf_value(v):
            out[ks] = v
        elif isinstance(v, list) and all(_is_json_leaf_value(x) for x in v):
            out[ks] = list(v)
        elif isinstance(v, dict) and ks in ("response_format", "logit_bias"):
            out[ks] = dict(v)
    return out


def _is_json_leaf_value(v: Any) -> bool:
    if v is None or isinstance(v, bool):
        return True
    if isinstance(v, str):
        return True
    if isinstance(v, numbers.Integral) and not isinstance(v, bool):
        return True
    if isinstance(v, float):
        return True
    return False


def parse_request_option_scalar_value(s: str) -> Any:
    """Parse a CLI value for `/set primary request_options set ...`."""
    raw = (s or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    if re_int := _try_int(raw):
        return re_int
    if re_float := _try_float(raw):
        return re_float
    return raw


def _try_int(s: str):
    try:
        return int(s, 10)
    except ValueError:
        return None


def _try_float(s: str):
    try:
        x = float(s)
    except ValueError:
        return None
    if x.is_integer() and "." not in s and "e" not in s.lower():
        return int(x)
    return x


def merge_hosted_request_options(
    body: dict[str, Any],
    request_options: Mapping[str, Any] | None,
    *,
    default_temperature: float | None = None,
) -> None:
    """Mutate ``body`` in place: apply sanitized options, then default temperature if absent."""
    opts = dict(request_options or {})
    for k, v in opts.items():
        key = str(k).strip()
        if not key or key in _HOSTED_BODY_RESERVED:
            continue
        if key not in _HOSTED_ALLOWED:
            continue
        if key == "stop":
            if isinstance(v, str) or (
                isinstance(v, list) and all(isinstance(x, str) for x in v)
            ):
                body[key] = v
            continue
        if key in ("response_format", "logit_bias") and isinstance(v, dict):
            body[key] = v
            continue
        if _is_json_leaf_value(v):
            body[key] = v
    if default_temperature is not None and "temperature" not in body:
        body["temperature"] = default_temperature


def merge_ollama_options_payload(payload: dict[str, Any], request_options: Mapping[str, Any] | None) -> None:
    """Mutate ``payload`` in place: merge into ``options`` for Ollama ``/api/chat``."""
    opts = dict(request_options or {})
    if not opts:
        return
    nested: dict[str, Any] = {}
    for k, v in opts.items():
        key = str(k).strip()
        if not key:
            continue
        if key in _OLLAMA_ALLOWED:
            if key == "stop" and isinstance(v, list) and all(isinstance(x, str) for x in v):
                nested[key] = v
            elif _is_json_leaf_value(v) or (key == "stop" and isinstance(v, str)):
                nested[key] = v
    if not nested:
        return
    cur = payload.get("options")
    if isinstance(cur, dict):
        merged = {**cur, **nested}
    else:
        merged = dict(nested)
    payload["options"] = merged
