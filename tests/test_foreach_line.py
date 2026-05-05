"""Smoke tests for repo-root ``foreach_line.py`` helper."""


def test_foreach_line_no_argv_prints_help_and_ok(capsys):
    import foreach_line as fl

    assert fl.main(["foreach_line.py"]) == 0
    out = capsys.readouterr().out
    assert "/call_python" in out
    assert "foreach_line.py" in out


def test_foreach_line_missing_ai_exits_2():
    import foreach_line as fl

    # ``ai`` is only injected by /call_python, not at import time.
    assert fl.main(["foreach_line.py", "-f", "/no/such/file"]) == 2
