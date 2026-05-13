"""Test fixture: exposes ``registry.load_flags`` via ``/probe_load_flags``."""

from __future__ import annotations

from agentlib.session import SessionLineResult


def register_repl(session, registry):
    session._test_repl_load_flags = frozenset(getattr(registry, "load_flags", frozenset()))

    def _dump(sess, rest: str) -> SessionLineResult:  # noqa: ARG001
        fs = getattr(sess, "_test_repl_load_flags", frozenset())
        return SessionLineResult(output=",".join(sorted(fs)) if fs else "(empty)")

    registry.register_command("probe_load_flags", _dump)
    return None
