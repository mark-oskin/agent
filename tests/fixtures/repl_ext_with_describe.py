"""Fixture: documents /load options via ``describe_repl_load_options``."""

from __future__ import annotations

from agentlib.session import SessionLineResult


def describe_repl_load_options() -> str:
    return "TEST_EXT_LOAD_OPTIONS_LINE\n"


def register_repl(session, registry):
    def _z(sess, rest: str) -> SessionLineResult:
        return SessionLineResult(output=f"z:{rest}")

    registry.register_command("ext_with_desc_z", _z)
    return None
