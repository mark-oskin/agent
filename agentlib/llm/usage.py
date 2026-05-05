"""Format Ollama /api/chat usage stats for logs and REPL /usage."""

from __future__ import annotations

from typing import Optional


def ollama_eval_generation_tok_per_sec(usage: dict) -> Optional[float]:
    """Tokens generated per second during the eval (decode) phase; needs eval_duration from Ollama."""
    n = usage.get("eval_count")
    dt_ns = usage.get("eval_duration")
    if not isinstance(n, int) or n < 0:
        return None
    if not isinstance(dt_ns, int) or dt_ns <= 0:
        return None
    return n / (dt_ns / 1e9)


def format_ollama_usage_line(usage: dict) -> str:
    parts = []
    if "prompt_eval_count" in usage:
        parts.append(f"prompt_eval_count={usage['prompt_eval_count']}")
    if "eval_count" in usage:
        parts.append(f"eval_count={usage['eval_count']}")
    rate = ollama_eval_generation_tok_per_sec(usage)
    if rate is not None:
        parts.append(f"gen_tok/s≈{rate:.1f}")
    for key, label in (
        ("total_duration", "total"),
        ("load_duration", "load"),
        ("prompt_eval_duration", "prompt"),
        ("eval_duration", "gen"),
    ):
        if key in usage:
            parts.append(f"{label}_s={usage[key] / 1e9:.3f}")
    return "[Ollama usage] " + ", ".join(parts) if parts else "[Ollama usage] (no counts in response)"


def format_last_ollama_usage_for_repl(last_usage: Optional[dict]) -> str:
    """Human-readable report for /usage (last local Ollama agent chat only)."""
    if last_usage is None:
        return (
            "No Ollama usage captured yet. Stats come from the local primary model's last "
            "/api/chat response (not hosted APIs). After a turn, try again, or use "
            "/set verbose 2 to print usage after each Ollama call (level 2)."
        )
    return (
        format_ollama_usage_line(last_usage)
        + "\n(Ollama: prompt_eval_count / eval_count — not identical to OpenAI-style prompt/completion tokens; "
        "gen_tok/s uses eval_count ÷ eval_duration when both are present.)"
    )
