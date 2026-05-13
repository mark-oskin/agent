"""Tests for core ``grep`` tool (``agentlib.tools.builtins.grep``)."""

from __future__ import annotations

import os

from agentlib.tools import builtins as tool_builtins


def test_grep_finds_lines_in_file(tmp_path):
    p = tmp_path / "sample.py"
    p.write_text("alpha = 1\nbeta = 2\n", encoding="utf-8")
    out = tool_builtins.grep(r"beta", str(p))
    assert "sample.py:2:beta = 2" in out
    assert "[Grep summary]" in out


def test_grep_respects_glob_in_directory(tmp_path):
    (tmp_path / "a.py").write_text("ZZ_MARKER_A\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("ZZ_MARKER_B\n", encoding="utf-8")
    out = tool_builtins.grep("ZZ_MARKER", str(tmp_path), glob_pattern="*.py")
    assert "ZZ_MARKER_A" in out
    assert "ZZ_MARKER_B" not in out


def test_grep_empty_pattern_error():
    assert tool_builtins.grep("", ".", None).startswith("Grep error:")


def test_grep_invalid_regex():
    out = tool_builtins.grep("[", str(os.path.abspath(".")), None)
    assert "Grep error:" in out and "invalid regex" in out


def test_grep_no_matches_message(tmp_path):
    (tmp_path / "x.py").write_text("only_this\n", encoding="utf-8")
    out = tool_builtins.grep("nope_nope", str(tmp_path))
    assert "0 matches" in out
