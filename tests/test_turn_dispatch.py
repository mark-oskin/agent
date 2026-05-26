"""``/turn`` and ``/send`` dispatch with ``self`` agent name."""

from __future__ import annotations

from agentlib.embedding import build_embedded_session
from agentlib.tui_parse import parse_turn_command


def test_parse_turn_command():
    assert parse_turn_command("/turn self 2+2") == ("self", ["2+2"])
    assert parse_turn_command("/turn Bob /help") == ("Bob", ["/help"])


def test_send_queues_during_agent_turn():
    _, sess = build_embedded_session(verbose=0)
    sess._agent_turn_depth = 1
    r = sess.execute_line("/send self hello")
    assert "Queued" in (r.get("output") or "")
    assert len(sess._after_turn_send_queue) == 1
    assert sess._after_turn_send_queue[0]["cmds"] == ["hello"]


def test_turn_self_plain_text_mocked(monkeypatch):
    _, sess = build_embedded_session(verbose=0)
    monkeypatch.setattr(
        sess, "_execute_user_request_body", lambda q: (True, f"answer:{q}")
    )
    r = sess.execute_line("/turn self What is 2+2?")
    assert r["type"] == "command"
    assert "answer:" in (r.get("output") or "") and "2+2" in (r.get("output") or "")


def test_turn_self_uses_delegate_when_configured():
    calls: list[tuple[str, str]] = []

    def dl(agent: str, cmd: str) -> dict:
        calls.append((agent, cmd))
        return {"type": "turn", "quit": False, "answer": f"delegated:{cmd}"}

    _, sess = build_embedded_session(verbose=0)
    sess.python_delegate_line = dl
    r = sess.execute_line("/turn self ping")
    assert calls == [("self", "ping")]
    assert r["type"] == "command"
    assert "delegated:ping" in (r.get("output") or "")


def test_turn_self_in_process_when_inside_turn(monkeypatch):
    calls: list[tuple[str, str]] = []

    def dl(agent: str, cmd: str) -> dict:
        calls.append((agent, cmd))
        return {"type": "turn", "quit": False, "answer": "delegated"}

    _, sess = build_embedded_session(verbose=0)
    sess.python_delegate_line = dl
    sess._agent_turn_depth = 1
    monkeypatch.setattr(
        sess, "_execute_user_request_body", lambda q: (True, f"answer:{q}")
    )
    r = sess.execute_line("/turn self hi")
    assert calls == []
    assert "answer:hi" in (r.get("output") or "")


def test_nested_turn_disables_session_command_tool(monkeypatch):
    import agentlib.session as session_mod

    _, sess = build_embedded_session(verbose=0)
    seen_tools: list = []

    def fake_run(*_a, enabled_tools=None, **_k):
        seen_tools.append(enabled_tools)
        return True, "ok"

    monkeypatch.setattr(session_mod, "run_agent_conversation_turn", fake_run)
    monkeypatch.setattr(sess, "_prepare_agent_turn_messages", lambda *a, **k: ("sys", "user"))
    monkeypatch.setattr(sess, "_match_skill_for_turn", lambda q: (None, None))
    monkeypatch.setattr(sess, "_route_requires_websearch", lambda *a, **k: None)
    monkeypatch.setattr(sess, "_user_wants_written_deliverable", lambda q: False)
    sess._agent_turn_depth = 1
    sess.execute_line("/turn self nested")
    assert seen_tools
    assert "session_command" not in seen_tools[0]


def test_session_command_turn_self_returns_answer(monkeypatch):
    _, sess = build_embedded_session(verbose=0)
    monkeypatch.setattr(
        sess, "_execute_user_request_body", lambda q: (True, f"reply:{q}")
    )
    out = sess._execute_session_command_for_tool("/turn self hi")
    assert "reply:hi" in out
