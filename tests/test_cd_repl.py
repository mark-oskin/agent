"""Tests for ``/cd`` session working directory (shell/tools cwd)."""

from agentlib.embedding import build_embedded_session


def test_cd_updates_session_cwd_and_shell_commands(tmp_path, monkeypatch):
    from agentlib.tools import builtins as tb

    d = tmp_path / "proj"
    d.mkdir()

    calls: list[tuple[str, str]] = []

    def stub(cmd, cwd=None):
        calls.append((cmd, cwd))
        return "STDOUT:\nstub\nSTDERR:\n"

    monkeypatch.setattr(tb, "run_command", stub)

    _, sess = build_embedded_session(verbose=0)
    sess.execute_line(f"/cd {d}")
    sess.execute_line("! echo hi")

    assert sess.session_cwd == str(d.resolve())
    assert calls == [("echo hi", str(d.resolve()))]


def test_embedded_sessions_do_not_share_deps_objects():
    app1, s1 = build_embedded_session(verbose=0)
    _, s2 = build_embedded_session(verbose=0, app=app1)
    assert s1._conversation_turn_deps is not s2._conversation_turn_deps
