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


def test_call_python_file_forwards_argv(tmp_path, capsys):
    script = tmp_path / "argv_echo.py"
    script.write_text("import sys\nprint('ARGV:' + '|'.join(sys.argv))\n", encoding="utf-8")
    _, sess = build_embedded_session(verbose=0)
    sess.execute_line(f"/call_python {script} --one two")
    out = capsys.readouterr().out.replace("\\\\", "/")
    assert "ARGV:" in out
    assert str(script.resolve()) in out
    assert "--one|two" in out


def test_call_python_sys_exit_zero_no_traceback(tmp_path, capsys):
    script = tmp_path / "exit_clean.py"
    script.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    _, sess = build_embedded_session(verbose=0)
    sess.execute_line(f"/call_python {script}")
    cap = capsys.readouterr()
    combined = cap.out + cap.err
    assert "Traceback" not in combined


def test_call_python_resolves_relative_file_flag_to_session_cwd(tmp_path, capsys):
    script = tmp_path / "cat_f.py"
    script.write_text(
        "import sys\n"
        "p = sys.argv[sys.argv.index('-f') + 1]\n"
        "print(open(p, encoding='utf-8').read().strip())\n",
        encoding="utf-8",
    )
    work = tmp_path / "work"
    work.mkdir()
    (work / "classes.txt").write_text("hello-from-classes\n", encoding="utf-8")

    _, sess = build_embedded_session(verbose=0)
    sess.execute_line(f"/cd {work}")
    sess.execute_line(f"/call_python {script} -f classes.txt")
    assert "hello-from-classes" in capsys.readouterr().out


def test_call_python_delegate_missing(capsys):
    _, sess = build_embedded_session(verbose=0)
    sess.execute_line(r'/call_python -c "ai(\"x\", \"Nobody\")"')
    assert "multi-agent" in capsys.readouterr().out.lower()
