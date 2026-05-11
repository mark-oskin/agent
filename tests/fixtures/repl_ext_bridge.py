"""Sample REPL extension: registers ``/pingx`` forwarding to ``/call_python`` on ``repl_ext_ping_runner.py``."""

from __future__ import annotations

from pathlib import Path

from agentlib.repl_extensions import ReplExtensionRegistry
from agentlib.session import SessionLineResult


def register_repl(session, registry):
    runner = Path(__file__).resolve().parent / "repl_ext_ping_runner.py"
    sub = ReplExtensionRegistry(session, str(runner))

    def pingx(sess, rest: str) -> SessionLineResult:
        return sub.invoke_call_python("ping", rest)

    registry.register_command("pingx", pingx)
    return ["/extensions"]
