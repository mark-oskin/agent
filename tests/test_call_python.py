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
