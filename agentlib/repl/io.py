from __future__ import annotations


REPL_INPUT_MAX_DEFAULT = 131072


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

