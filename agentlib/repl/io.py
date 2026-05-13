from __future__ import annotations

from typing import Callable, Optional


REPL_INPUT_MAX_DEFAULT = 131072


def read_repl_lines_until_balanced_triple_double_quotes(
    repl_read_line: Callable[[str], str],
    *,
    first_prompt: str,
    continuation_prompt: str = "... ",
    max_bytes: int,
    repl_commit_history: Optional[Callable[[str], None]] = None,
) -> str:
    """Read lines until non-overlapping ``\"\"\"`` substrings have an even count (zero counts).

    Odd counts continue with ``continuation_prompt`` on the next read. Input is UTF-8 and must
    not exceed ``max_bytes`` total.

    If ``repl_commit_history`` is set, it is called **once** with the final stripped block (so a
    multi-physical-line ``\"\"\" … \"\"\"`` entry becomes a single readline history item).
    """
    chunks: list[str] = []
    while True:
        prompt = first_prompt if not chunks else continuation_prompt
        line = repl_read_line(prompt)
        chunks.append(line)
        buf = "\n".join(chunks)
        raw_len = len(buf.encode("utf-8"))
        if raw_len > max_bytes:
            raise ValueError(
                f"repl input exceeded max_bytes ({max_bytes}); close \"\"\" sooner or raise the "
                "limit in prefs (agent.repl_input_max_bytes)."
            )
        if buf.count('"""') % 2 == 0:
            out = buf.strip()
            if repl_commit_history is not None and out:
                repl_commit_history(out)
            return out


def repl_buffered_line_max_bytes(*, settings_get_int) -> int:
    """
    Effective max bytes for buffered-line REPL input.

    Legacy semantics:
    - default to 128KiB when unset/invalid
    - clamp to a minimum of 4096 bytes
    """
    v = int(settings_get_int(("agent", "repl_input_max_bytes"), 0) or 0)
    if v <= 0:
        v = REPL_INPUT_MAX_DEFAULT
    return max(4096, v)

