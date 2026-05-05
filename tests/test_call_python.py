"""Tests for ``/call_python`` (embedded session)."""

from agentlib.embedding import build_embedded_session


def test_call_python_help():
    _, sess = build_embedded_session(verbose=0)
    r = sess.execute_line("/call_python help")
    assert r["type"] == "command"
    assert not r.get("quit")


def test_call_python_c_pass():
    _, sess = build_embedded_session(verbose=0)
    r = sess.execute_line('/call_python -c "pass"')
    assert r["type"] == "command"


def test_call_python_print(capsys):
    _, sess = build_embedded_session(verbose=0)
    sess.execute_line(r'/call_python -c "print(123)"')
    assert "123" in capsys.readouterr().out


def test_call_python_c_supports_newline_escapes(capsys):
    _, sess = build_embedded_session(verbose=0)
    sess.execute_line(r'/call_python -c "print(1)\nprint(2)"')
    out = capsys.readouterr().out
    assert "1" in out
    assert "2" in out


def test_call_python_c_supports_literal_multiline_paste(capsys):
    _, sess = build_embedded_session(verbose=0)
    r0 = sess.execute_line('/call_python -c "print(1)')
    assert r0["type"] == "command"
    assert not r0.get("quit")
    r1 = sess.execute_line('print(2)"')
    assert r1["type"] == "command"
    out = capsys.readouterr().out
    assert "1" in out
    assert "2" in out


def test_call_python_ai_noop(capsys):
    _, sess = build_embedded_session(verbose=0)
    sess.execute_line(r'/call_python -c "r = ai(\"\"); print(r.get(\"type\"))"')
    assert "noop" in capsys.readouterr().out


def test_call_python_file(tmp_path, capsys):
    path = tmp_path / "snippet.py"
    path.write_text("print('from_file')\n", encoding="utf-8")
    _, sess = build_embedded_session(verbose=0)
    sess.execute_line(f"/call_python {path}")
    assert "from_file" in capsys.readouterr().out


def test_call_python_delegate_missing(capsys):
    _, sess = build_embedded_session(verbose=0)
    sess.execute_line(r'/call_python -c "ai(\"x\", \"Nobody\")"')
    assert "multi-agent" in capsys.readouterr().out.lower()
