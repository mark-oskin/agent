"""
REPL commands for the global in-memory queues defined in ``tools.queue``.

Load with ``/load extensions/queue_control.py`` (from repo root or an absolute path).
The same store is used by the ``queue`` toolset (``list_add``, ``list_remove``, …).
"""

from __future__ import annotations

import shlex

from agentlib.repl_extensions import ReplExtensionRegistry
from agentlib.session import SessionLineResult

import tools.queue as Q

_USAGE = "/queue show lists — /queue <listname> add | remove | peek | length | clear | save | load"


def _cmd_queue(session, rest: str) -> SessionLineResult:
    r = (rest or "").strip()
    if not r or r.lower() in ("help", "-h", "--help"):
        return SessionLineResult(output=_USAGE)
    try:
        parts = shlex.split(r)
    except ValueError as e:
        return SessionLineResult(output=f"/queue: {e}")
    if not parts:
        return SessionLineResult(output=_USAGE)
    if parts[0].lower() == "show" and len(parts) >= 2 and parts[1].lower() == "lists":
        return SessionLineResult(output=Q.format_show_lists())
    if len(parts) < 2:
        return SessionLineResult(output=_USAGE)

    name, op = parts[0], parts[1].lower()
    if op == "add":
        if len(parts) < 3:
            return SessionLineResult(output="/queue: add requires text after the list name.")
        data = " ".join(parts[2:])
        return SessionLineResult(output=Q.queue_add(name, data))
    if op == "remove":
        if len(parts) != 2:
            return SessionLineResult(
                output="/queue: remove takes no extra arguments (example: `/queue mylist remove`)."
            )
        return SessionLineResult(output=Q.queue_remove(name))
    if op == "peek":
        if len(parts) != 2:
            return SessionLineResult(output="/queue: peek takes no extra arguments.")
        return SessionLineResult(output=Q.queue_peek(name))
    if op == "length":
        if len(parts) != 2:
            return SessionLineResult(output="/queue: length takes no extra arguments.")
        return SessionLineResult(output=f"{name}: {Q.queue_length(name)} item(s)")
    if op == "clear":
        if len(parts) != 2:
            return SessionLineResult(output="/queue: clear takes no extra arguments.")
        return SessionLineResult(output=Q.queue_clear(name))
    if op == "save":
        if len(parts) != 3:
            return SessionLineResult(
                output="/queue: save requires exactly one filename (quote the path if it has spaces)."
            )
        path = session._resolve_session_path(parts[2])
        return SessionLineResult(output=Q.queue_save(name, path))
    if op == "load":
        if len(parts) != 3:
            return SessionLineResult(
                output="/queue: load requires exactly one filename (quote the path if it has spaces)."
            )
        path = session._resolve_session_path(parts[2])
        return SessionLineResult(output=Q.queue_load(name, path))
    return SessionLineResult(output=f"/queue: unknown subcommand {parts[1]!r}. Try `/queue help`.")


def register_repl(_session, registry: ReplExtensionRegistry):
    registry.register_help(_USAGE)
    registry.register_command("queue", _cmd_queue)
