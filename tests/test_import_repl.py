"""Tests for ``/import FILE`` on ``AgentSession``."""

import os

from agentlib.embedding import build_embedded_session


def test_non_import_line_with_bad_quoting_is_not_shlex_split(monkeypatch, capsys):
    """Long prompts with unbalanced ``"`` must not hit ``shlex`` or masquerade as ``/import`` errors."""
    calls: list[str] = []

    def stub(user_query: str):
        calls.append(user_query)
        return True, "stubbed"

    _, sess = build_embedded_session(verbose=0)
    monkeypatch.setattr(sess, "_execute_user_request", stub)

    bad = 'Summarize this text: He said "hello'
    r = sess.execute_line(bad)
    assert r["type"] == "turn"
    assert calls == [bad]
    assert "/import:" not in capsys.readouterr().out


def test_import_usage(capsys):
    _, sess = build_embedded_session(verbose=0)
    r = sess.execute_line("/import")
    assert r["type"] == "command"
    assert "usage" in capsys.readouterr().out.lower()


def test_import_not_a_file(capsys):
    _, sess = build_embedded_session(verbose=0)
    r = sess.execute_line("/import definitely-missing-file-xyz.bin")
    assert r["type"] == "command"
    assert "not a file" in capsys.readouterr().out.lower()


def test_import_sends_synthetic_message(monkeypatch, tmp_path):
    fp = tmp_path / "knowledge.txt"
    fp.write_text("hi", encoding="utf-8")

    calls: list[str] = []

    def stub(user_query: str):
        calls.append(user_query)
        return True, "stubbed"

    _, sess = build_embedded_session(verbose=0)
    monkeypatch.setattr(sess, "_execute_user_request", stub)

    r = sess.execute_line(f'/import "{fp}"')
    assert r["type"] == "turn"
    assert calls and "import it into this conversation" in calls[0]
    assert os.path.normpath(os.path.abspath(str(fp))) in calls[0]


def test_import_resolves_relative_to_session_cwd(monkeypatch, tmp_path):
    sub = tmp_path / "nest"
    sub.mkdir()
    fp = sub / "memo.txt"
    fp.write_text("x", encoding="utf-8")

    calls: list[str] = []

    def stub(user_query: str):
        calls.append(user_query)
        return True, "stubbed"

    _, sess = build_embedded_session(verbose=0)
    monkeypatch.setattr(sess, "_execute_user_request", stub)
    sess.execute_line(f"/cd {sub}")
    r = sess.execute_line("/import memo.txt")
    assert r["type"] == "turn"
    assert os.path.normpath(os.path.abspath(str(fp))) in calls[0]


def test_import_with_emit_invokes_stub(monkeypatch, tmp_path):
    fp = tmp_path / "doc.txt"
    fp.write_text("{}", encoding="utf-8")

    calls: list[str] = []

    def stub(user_query: str):
        calls.append(user_query)
        return True, "done"

    _, sess = build_embedded_session(verbose=0)
    monkeypatch.setattr(sess, "_execute_user_request", stub)

    def emit(_ev):
        pass

    r = sess.execute_line(f"/import {fp}", emit=emit)
    assert r["type"] == "turn"
    assert calls
