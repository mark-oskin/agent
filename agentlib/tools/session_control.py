"""REPL slash commands for the session_command native tool (blocklist + captured output)."""

from __future__ import annotations

import shlex
from typing import Callable, Optional, Tuple

# Commands that must not run from an autonomous tool call.
_BLOCKED_EXACT = frozenset({"/quit", "/exit", "/q"})
_BLOCKED_PREFIXES = (
    "/while",
    "/skill",
    "/use-skill",
    "/use-skills",
    "/reuse-skill",
    "/send",
    "/fork_background",
    "/fork",
    "/call_python",
    "/run_command",
    "/source",
    "/import",
)


def merge_command_transcript_output(structured: str, captured: str) -> str:
    """Combine ``SessionLineResult.output`` with text collected from the sink."""
    s = (structured or "").strip()
    c = (captured or "").strip()
    if not c:
        return s
    if not s:
        return c
    if s == c or s in c:
        return c
    if c in s:
        return s
    return f"{c}\n\n{s}"


def session_command_blocked_reason(line: str) -> Optional[str]:
    """Return an error message if this slash line must not run via ``session_command``."""
    raw = (line or "").strip()
    low = raw.lower()
    if low in _BLOCKED_EXACT:
        return "session_command error: /quit and /exit are not allowed via session_command."
    for prefix in _BLOCKED_PREFIXES:
        if low == prefix or low.startswith(prefix + " ") or low.startswith(prefix + "\t"):
            return (
                f"session_command error: {prefix} is not allowed via session_command "
                "(unbounded side effects or arbitrary code). Run it manually in the REPL."
            )
    if low.startswith("!"):
        return (
            "session_command error: shell escapes (!) are not allowed via session_command. "
            "Use /run_command manually if needed."
        )
    try:
        toks = shlex.split(raw)
    except ValueError:
        return None
    if toks and toks[0].lower() in ("/set", "/settings") and len(toks) >= 2:
        if toks[1].lower().replace("-", "_") == "lock":
            return (
                "session_command error: /set lock is not allowed via session_command "
                "(permanent for the session). The user must run it manually."
            )
    return None


def validate_session_command(line: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Validate one REPL slash line for ``session_command``.

    Returns ``(canonical_line, None)`` or ``(None, error_message)``.
    """
    raw = (line or "").strip()
    if not raw:
        return None, "session_command error: command must be a non-empty string."
    if "\n" in raw or "\r" in raw:
        return None, "session_command error: command must be a single line."
    if not raw.startswith("/"):
        return (
            None,
            "session_command error: must be a slash command (e.g. /show models, /set thinking show).",
        )
    try:
        shlex.split(raw)
    except ValueError as e:
        return None, f"session_command error: {e}"
    blocked = session_command_blocked_reason(raw)
    if blocked:
        return None, blocked
    return raw, None


def execute_session_slash_command(
    line: str,
    *,
    execute_line: Callable[[str], dict],
) -> str:
    """Run one REPL slash command; return captured output for the tool transcript."""
    norm, err = validate_session_command(line)
    if err:
        return err
    assert norm is not None
    try:
        res = execute_line(norm)
    except Exception as e:
        return f"session_command error: {type(e).__name__}: {e}"
    if not isinstance(res, dict):
        return str(res).strip() or f"OK ({norm})"
    if res.get("quit"):
        return "session_command error: command requested quit"
    if res.get("type") == "command":
        out = (res.get("output") or "").strip()
        pre = res.get("prefill_prompt")
        if isinstance(pre, str) and pre.strip():
            note = (
                f"(Also returned prefill_prompt for the host input field, "
                f"{len(pre.strip())} characters — not shown in full.)"
            )
            out = merge_command_transcript_output(out, note)
        return out.strip() or f"OK ({norm})"
    if res.get("type") == "turn":
        ans = res.get("answer")
        if isinstance(ans, str) and ans.strip():
            return ans.strip()
        return (
            f"session_command error: {norm!r} started an agent turn instead of a slash command. "
            "Use a /command line only."
        )
    return f"OK ({norm})"


def normalize_allowlisted_session_command(line: str) -> Optional[str]:
    """Return the canonical slash line if allowed, else None (legacy helper)."""
    norm, err = validate_session_command(line)
    return norm if not err else None


def execute_allowlisted_session_command(
    line: str,
    *,
    execute_line: Callable[[str], dict],
    settings_locked: bool = False,
) -> str:
    del settings_locked  # session enforces lock; sink capture returns its message
    return execute_session_slash_command(line, execute_line=execute_line)
