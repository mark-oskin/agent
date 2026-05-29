"""Tab completion for REPL slash commands (CLI readline + TUI prompt)."""

from __future__ import annotations

from typing import Callable, List, Optional, TYPE_CHECKING

from agentlib.repl.command_registry import (
    ReplCompletionContext,
    complete_command_token,
)

if TYPE_CHECKING:
    from agentlib.session import AgentSession


def repl_token_context(line: str, cursor: int) -> tuple[list[str], str, int, int]:
    """
    Return ``(tokens_before_partial, partial, word_start, token_index)``.

    ``tokens_before_partial`` are fully typed words before the word being completed.
    ``partial`` is the incomplete word under the cursor.
    """
    if cursor < 0:
        cursor = 0
    if cursor > len(line):
        cursor = len(line)
    prefix = line[:cursor]
    word_start = max(prefix.rfind(" "), prefix.rfind("\t")) + 1
    partial = prefix[word_start:cursor]
    before = prefix[:word_start].strip()
    tokens = before.split() if before else []
    return tokens, partial, word_start, len(tokens)


def complete_repl_candidates(
    session: "AgentSession",
    line: str,
    cursor: int,
    *,
    ctx: Optional[ReplCompletionContext] = None,
) -> List[str]:
    """Return completion strings for the word at ``cursor`` (each is a full replacement word)."""
    ctx = ctx or ReplCompletionContext()
    stripped = line.lstrip()
    if not stripped.startswith("/"):
        return []
    tokens, partial, _word_start, _token_index = repl_token_context(line, cursor)
    return complete_command_token(session, tokens, partial, ctx)


def apply_repl_completion(
    line: str,
    cursor: int,
    candidates: List[str],
) -> tuple[str, int]:
    """
    Apply tab completion to ``line`` at ``cursor``.

    Returns ``(new_line, new_cursor)``. If ``candidates`` is empty, returns unchanged.
    """
    if not candidates:
        return line, cursor
    _tokens, partial, word_start, _token_index = repl_token_context(line, cursor)
    if len(candidates) == 1:
        replacement = candidates[0]
    else:
        replacement = _common_prefix(candidates)
        if len(replacement) <= len(partial):
            return line, cursor
    new_line = line[:word_start] + replacement + line[cursor:]
    new_cursor = word_start + len(replacement)
    if len(candidates) == 1:
        if new_cursor >= len(new_line) or new_line[new_cursor] in ("", " "):
            if new_cursor >= len(new_line):
                new_line = new_line + " "
                new_cursor += 1
            elif new_line[new_cursor] != " ":
                new_line = new_line[:new_cursor] + " " + new_line[new_cursor:]
                new_cursor += 1
    return new_line, new_cursor


def _common_prefix(items: List[str]) -> str:
    if not items:
        return ""
    prefix = items[0]
    for item in items[1:]:
        while prefix and not item.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def offset_to_location(text: str, offset: int) -> tuple[int, int]:
    if offset < 0:
        offset = 0
    if offset > len(text):
        offset = len(text)
    before = text[:offset]
    row = before.count("\n")
    last_nl = before.rfind("\n")
    col = offset if last_nl < 0 else offset - last_nl - 1
    return row, col


def text_area_cursor_offset(text_area) -> int:
    """Byte/char offset of the Textual ``TextArea`` cursor."""
    row, col = text_area.cursor_location
    lines = text_area.text.split("\n")
    if row < 0:
        row = 0
    if row >= len(lines):
        return len(text_area.text)
    return sum(len(lines[i]) + 1 for i in range(row)) + min(col, len(lines[row]))


class ReadlineReplCompleter:
    """GNU readline completer callback bound to a session."""

    def __init__(self, session_getter: Callable[[], "AgentSession"]) -> None:
        self._session_getter = session_getter
        self._ctx = ReplCompletionContext()
        self._matches: List[str] = []

    def set_context(self, ctx: ReplCompletionContext) -> None:
        self._ctx = ctx

    def __call__(self, text: str, state: int) -> Optional[str]:
        if state == 0:
            try:
                import readline
            except ImportError:
                self._matches = []
                return None
            line = readline.get_line_buffer()
            end = readline.get_endidx()
            session = self._session_getter()
            self._matches = complete_repl_candidates(
                session, line, end, ctx=self._ctx
            )
            # readline passes only the partial word; filter to those extending ``text``.
            if text:
                self._matches = [m for m in self._matches if m.startswith(text)]
        if state < len(self._matches):
            return self._matches[state]
        return None


def install_readline_completer(
    session_getter: Callable[[], "AgentSession"],
    *,
    ctx: Optional[ReplCompletionContext] = None,
) -> Optional[ReadlineReplCompleter]:
    """Install tab completion on GNU readline (no-op when readline is unavailable)."""
    try:
        import readline
    except ImportError:
        return None
    completer = ReadlineReplCompleter(session_getter)
    if ctx is not None:
        completer.set_context(ctx)
    readline.set_completer_delims(" \t\n\"'")
    readline.set_completer(completer)
    _bind_readline_tab_complete(readline)
    return completer


def _bind_readline_tab_complete(readline) -> None:
    """Bind Tab to complete (libedit on macOS uses a different syntax than GNU readline)."""
    doc = (readline.__doc__ or "").lower()
    if "libedit" in doc:
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
