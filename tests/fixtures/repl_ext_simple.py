"""Minimal REPL extension with one slash command and a post-load line."""

from __future__ import annotations

from agentlib.session import SessionLineResult


def register_repl(session, registry):
    def on_ext_ok(sess, rest: str) -> SessionLineResult:
        return SessionLineResult(output=f"ext_ok:{rest.strip()}")

    registry.register_help("/ext_ok <text> — test fixture extension (prints ext_ok:…)")
    registry.register_command("ext_ok", on_ext_ok)
    return ["/extensions"]
