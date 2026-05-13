"""Tests for triple-quote–balanced multiline REPL input (``agentlib.repl.io``)."""

from __future__ import annotations

import pytest

from agentlib.repl.io import _invoke_repl_read_line, read_repl_lines_until_balanced_triple_double_quotes


def test_invoke_repl_read_line_passes_line_index_when_supported():
    seen: list[tuple[str, int]] = []

    def two_arg(p: str, i: int) -> str:
        seen.append((p, i))
        return f"L{i}"

    assert _invoke_repl_read_line(two_arg, ">", 0) == "L0"
    assert _invoke_repl_read_line(two_arg, "... ", 1) == "L1"
    assert seen == [(">", 0), ("... ", 1)]


def test_plain_line_one_read_no_triple_quotes():
    seq = iter(["/help"])
    prompts: list[str] = []

    def rl(p: str) -> str:
        prompts.append(p)
        return next(seq)

    out = read_repl_lines_until_balanced_triple_double_quotes(
        rl, first_prompt=">", continuation_prompt="... ", max_bytes=4096
    )
    assert out == "/help"
    assert prompts == [">"]


def test_multiline_closed_on_third_physical_line():
    seq = iter(['/code """', "line two", '"""'])
    prompts: list[str] = []

    def rl(p: str) -> str:
        prompts.append(p)
        return next(seq)

    out = read_repl_lines_until_balanced_triple_double_quotes(
        rl, first_prompt=">", continuation_prompt="... ", max_bytes=65536
    )
    assert out == '/code """\nline two\n"""'
    assert prompts == [">", "... ", "... "]


def test_repl_commit_history_once_for_multiline_block():
    """One logical block → one ``repl_commit_history`` call with the full stripped string."""
    seq = iter(['a """', "b", '"""'])
    commits: list[str] = []

    def rl(_p: str) -> str:
        return next(seq)

    out = read_repl_lines_until_balanced_triple_double_quotes(
        rl,
        first_prompt=">",
        continuation_prompt="... ",
        max_bytes=65536,
        repl_commit_history=commits.append,
    )
    assert out == 'a """\nb\n"""'
    assert commits == ['a """\nb\n"""']


def test_inner_triple_quotes_on_same_line():
    """Outer pair; inner ``\"\"\"…\"\"\"`` adds two occurrences (even)."""
    seq = iter(['"""', 'x = """inner"""', '"""'])

    def rl(_p: str) -> str:
        return next(seq)

    out = read_repl_lines_until_balanced_triple_double_quotes(
        rl, first_prompt=">", max_bytes=65536
    )
    assert 'inner' in out
    assert out.count('"""') % 2 == 0


def test_exceeds_max_bytes_while_unbalanced_raises():
    lines = ['"""'] + ["pad"] * 200

    def rl(_p: str) -> str:
        return lines.pop(0) if lines else ""

    with pytest.raises(ValueError, match="exceeded max_bytes"):
        read_repl_lines_until_balanced_triple_double_quotes(
            rl, first_prompt=">", max_bytes=80
        )
