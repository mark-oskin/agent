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
