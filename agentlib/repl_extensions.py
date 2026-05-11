"""REPL extension loading via ``/load`` — user-supplied Python registers slash commands."""

from __future__ import annotations

import shlex
from typing import Any, Callable, List, TYPE_CHECKING

if TYPE_CHECKING:
    from agentlib.session import AgentSession

# Maximum nested ``execute_line`` depth when running post-load lines from ``/load``.
MAX_REPL_POST_LOAD_DEPTH = 32


class ReplExtensionRegistry:
    """
    Passed to ``register_repl(session, registry)`` from a loaded extension module.

    Use :meth:`register_command` to bind ``/name`` handlers. Use :meth:`invoke_call_python`
    to forward to ``/call_python THIS_SCRIPT subcommand …`` with safe quoting.
    """

    def __init__(self, session: "AgentSession", script_path: str) -> None:
        self.session = session
        self.script_path = script_path
        self._keys: List[str] = []

    def command_keys(self) -> tuple[str, ...]:
        return tuple(self._keys)

    def register_command(self, name: str, handler: Callable[..., Any]) -> None:
        """Register ``/foo`` (``name`` may be ``foo`` or ``/foo``). Handler receives ``(session, rest)``."""
        key = (name or "").strip().lower()
        if not key.startswith("/"):
            key = "/" + key.lstrip("/")
        self.session._repl_register_extension_command(key, handler)
        self._keys.append(key)

    def call_python_line(self, subcommand: str, rest: str = "") -> str:
        """Build a single ``/call_python`` line for this extension's script (quoted)."""
        return _format_call_python_invocation(self.script_path, subcommand, rest)

    def invoke_call_python(self, subcommand: str, rest: str = "") -> Any:
        """Run ``/call_python <this script> <subcommand> …`` through normal REPL dispatch."""
        return self.session._execute_command_line(self.call_python_line(subcommand, rest))


def _format_call_python_invocation(script_path: str, subcommand: str, rest: str = "") -> str:
    parts = ["/call_python", shlex.quote(script_path), shlex.quote(subcommand.strip() or "")]
    tail = (rest or "").strip()
    if not tail:
        return " ".join(parts)
    try:
        extra = shlex.split(tail)
    except ValueError:
        parts.append(tail)
        return " ".join(parts)
    parts.extend(shlex.quote(x) for x in extra)
    return " ".join(parts)
