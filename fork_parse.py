"""Parse `/fork` and `/kill` lines for agent_tui (no Textual dependency)."""

from __future__ import annotations

import shlex


def parse_kill_command(line: str) -> str | None:
    """
    Parse ``/kill <name>`` or ``/kill "multi word name"``.

    Returns the trimmed agent name or ``None`` if not a kill line / invalid.
    """
    s = (line or "").strip()
    if not s.startswith("/kill"):
        return None
    rest = s[5:].lstrip()
    if not rest:
        return None
    if rest.startswith('"'):
        end = rest.find('"', 1)
        if end == -1:
            return None
        name = rest[1:end].strip()
        return name if name else None
    return rest.split(None, 1)[0].strip() or None


def parse_fork_command(line: str) -> tuple[str, list[str]] | None:
    """
    Parse ``/fork <name> ["cmd1,cmd2,..."]``.

    Returns ``(agent_name, commands)`` or ``None`` if not a fork line / invalid.
    When an optional trailing segment is quoted (double quotes), split inside it on commas.
    Otherwise a non-empty trailing segment after the name is treated as a single command.
    """
    s = (line or "").strip()
    if not s.startswith("/fork"):
        return None
    rest = s[5:].lstrip()
    if not rest:
        return None

    q = rest.find('"')
    if q == -1:
        try:
            parts = shlex.split(rest)
        except ValueError:
            return None
        if not parts:
            return None
        name = parts[0].strip()
        if not name:
            return None
        tail = " ".join(parts[1:]).strip()
        cmds = [tail] if tail else []
        return name, cmds

    name_candidate = rest[:q].strip()
    try:
        name_parts = shlex.split(name_candidate)
    except ValueError:
        return None
    if len(name_parts) != 1:
        return None
    name = name_parts[0].strip()
    if not name:
        return None
    quoted = rest[q:]
    if len(quoted) < 2 or quoted[-1] != '"':
        return None
    inner = quoted[1:-1]
    cmds = [x.strip() for x in inner.split(",") if x.strip()]
    return name, cmds


def parse_fork_background_command(line: str) -> tuple[str, list[str]] | None:
    """
    Parse ``/fork_background <name> ["cmd1,cmd2,..."]`` — same payload shape as ``/fork``.

    Returns ``None`` if not a ``/fork_background`` line or if the remainder is empty/invalid.
    """
    s = (line or "").strip()
    prefix = "/fork_background"
    if not s.startswith(prefix):
        return None
    tail = s[len(prefix) :].lstrip()
    if not tail:
        return None
    return parse_fork_command("/fork " + tail)


def format_fork_command_line(name: str, commands: list[str] | None = None) -> str:
    """
    Build a ``/fork`` line understood by [parse_fork_command][fork_parse.parse_fork_command].

    ``commands`` become one comma-separated quoted segment when non-empty (same as manual ``/fork``).
    """
    nm = (name or "").strip()
    commands = commands or []
    cmds = [str(c).strip() for c in commands if str(c).strip()]
    head = f"/fork {shlex.quote(nm)}" if nm else "/fork"
    if not cmds:
        return head
    inner = ",".join(cmds)
    return f'{head} "{inner}"'
