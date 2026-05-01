"""Parse `/fork` and `/kill` lines for agent_tui (no Textual dependency)."""

from __future__ import annotations


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
        parts = rest.split(None, 1)
        name = parts[0].strip()
        if not name:
            return None
        tail = parts[1].strip() if len(parts) > 1 else ""
        cmds = [tail] if tail else []
        return name, cmds

    name = rest[:q].strip()
    if not name:
        return None
    quoted = rest[q:]
    if len(quoted) < 2 or quoted[-1] != '"':
        return None
    inner = quoted[1:-1]
    cmds = [x.strip() for x in inner.split(",") if x.strip()]
    return name, cmds
