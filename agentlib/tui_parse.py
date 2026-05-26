"""Parse `/fork`, `/send`, and `/kill` lines for agent_tui (no Textual dependency)."""

from __future__ import annotations

import shlex


def _split_fork_command_list(inner: str) -> list[str] | None:
    """Split text inside outer double-quotes into commands.

    Commas separate commands unless:
    - A segment is wrapped in ASCII single quotes ``' ... '`` (commas inside are literal), or
    - A comma is escaped as ``\\,`` (always a literal comma, never splits).

    Inside single quotes, ``\\\\`` → ``\\``, ``\\'`` → ``'``, ``\\,`` → ``,``.

    Returns ``None`` if a single quote is left unclosed.
    """
    out: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(inner)
    in_sq = False

    def flush() -> None:
        nonlocal buf
        s = "".join(buf).strip()
        if s:
            out.append(s)
        buf = []

    while i < n:
        c = inner[i]
        if in_sq:
            if c == "'":
                in_sq = False
                i += 1
                continue
            if c == "\\" and i + 1 < n:
                nxt = inner[i + 1]
                if nxt == "\\":
                    buf.append("\\")
                    i += 2
                    continue
                if nxt == "'":
                    buf.append("'")
                    i += 2
                    continue
                if nxt == ",":
                    buf.append(",")
                    i += 2
                    continue
            buf.append(c)
            i += 1
            continue
        if c.isspace() and not buf:
            i += 1
            continue
        if c == "'":
            in_sq = True
            i += 1
            continue
        if c == "\\" and i + 1 < n and inner[i + 1] == ",":
            buf.append(",")
            i += 2
            continue
        if c == ",":
            flush()
            i += 1
            continue
        buf.append(c)
        i += 1

    if in_sq:
        return None
    flush()
    return out


def _format_fork_command_segment(cmd: str) -> str:
    """Format one command token for comma-separated fork list inside outer double-quotes."""
    s = cmd
    needs_sq = (
        "," in s
        or "'" in s
        or "\\" in s
        or s.strip() != s
    )
    if not needs_sq:
        return s.strip()
    chunk: list[str] = []
    for c in s:
        if c == "\\":
            chunk.append("\\\\")
        elif c == "'":
            chunk.append("\\'")
        else:
            chunk.append(c)
    return "'" + "".join(chunk) + "'"


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
    When an optional trailing segment is wrapped in double quotes, split the inner text on
    commas except:

    - Text inside single quotes ``'like, this'`` is one fragment (comma is literal).
    - ``\\,`` is always a literal comma (outside or inside single quotes).

    Outside double quotes, a non-empty trailing segment after the name is one command.

    Inner ``\\``, ``\\'``, ``\\,`` work inside ``' ... '``: backslash-quote-comma semantics above.
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
    cmds = _split_fork_command_list(inner)
    if cmds is None:
        return None
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


def parse_agent_dispatch_command(line: str, verb: str) -> tuple[str, list[str]] | None:
    """
    Parse ``/<verb> <name> COMMAND…`` or ``/<verb> <name> "cmd1,cmd2,…"``.

    Used for ``/send`` (async) and ``/turn`` (blocking). The optional double-quoted segment uses
    the same comma / single-quote / ``\\,`` rules as :func:`parse_fork_command`.

    Returns ``(agent_name, commands)`` or ``None`` if invalid. At least one command is required.
    """
    v = (verb or "").strip().lower()
    if not v:
        return None
    prefix = f"/{v}"
    s = (line or "").strip()
    if not s.lower().startswith(prefix):
        return None
    rest = s[len(prefix) :].lstrip()
    if not rest:
        return None

    q = rest.find('"')
    if q == -1:
        try:
            parts = shlex.split(rest)
        except ValueError:
            return None
        if len(parts) < 2:
            return None
        name = parts[0].strip()
        if not name:
            return None
        tail = shlex.join(parts[1:]).strip()
        if not tail:
            return None
        return name, [tail]

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
    cmds = _split_fork_command_list(inner)
    if cmds is None or not cmds:
        return None
    return name, cmds


def parse_send_command(line: str) -> tuple[str, list[str]] | None:
    """Parse ``/send <name> COMMAND…`` (see :func:`parse_agent_dispatch_command`)."""
    return parse_agent_dispatch_command(line, "send")


def parse_turn_command(line: str) -> tuple[str, list[str]] | None:
    """Parse ``/turn <name> COMMAND…`` (see :func:`parse_agent_dispatch_command`)."""
    return parse_agent_dispatch_command(line, "turn")


def format_turn_command_line(name: str, commands: list[str] | None = None) -> str:
    """Build a ``/turn`` line understood by :func:`parse_turn_command`."""
    nm = (name or "").strip()
    commands = commands or []
    cmds = [str(c).strip() for c in commands if str(c).strip()]
    head = f"/turn {shlex.quote(nm)}" if nm else "/turn"
    if not cmds:
        return head
    inner = ",".join(_format_fork_command_segment(c) for c in cmds)
    return f'{head} "{inner}"'


def format_send_command_line(name: str, commands: list[str] | None = None) -> str:
    """
    Build a ``/send`` line understood by :func:`parse_send_command`.

    When non-empty, ``commands`` are joined with commas; any entry that contains a comma,
    quote, backslash, or leading/trailing spaces is wrapped in single quotes with escapes
    so the line round-trips (same as :func:`format_fork_command_line`).
    """
    nm = (name or "").strip()
    commands = commands or []
    cmds = [str(c).strip() for c in commands if str(c).strip()]
    head = f"/send {shlex.quote(nm)}" if nm else "/send"
    if not cmds:
        return head
    inner = ",".join(_format_fork_command_segment(c) for c in cmds)
    return f'{head} "{inner}"'


def format_fork_command_line(name: str, commands: list[str] | None = None) -> str:
    """
    Build a ``/fork`` line understood by :func:`parse_fork_command`.

    When non-empty, ``commands`` are joined with commas; any entry that contains a comma,
    quote, backslash, or leading/trailing spaces is wrapped in single quotes with escapes
    so the line round-trips.
    """
    nm = (name or "").strip()
    commands = commands or []
    cmds = [str(c).strip() for c in commands if str(c).strip()]
    head = f"/fork {shlex.quote(nm)}" if nm else "/fork"
    if not cmds:
        return head
    inner = ",".join(_format_fork_command_segment(c) for c in cmds)
    return f'{head} "{inner}"'
