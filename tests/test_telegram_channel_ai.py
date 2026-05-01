"""Tests for ``telegram_channel_ai`` helpers (mocked HTTP)."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import telegram_channel_ai as tc


def test_parse_allowed_chat_ids_empty():
    assert tc._parse_allowed_chat_ids(None) is None
    assert tc._parse_allowed_chat_ids("") is None
    assert tc._parse_allowed_chat_ids("  ") is None


def test_parse_allowed_chat_ids_values():
    assert tc._parse_allowed_chat_ids("-1001,-1002") == {-1001, -1002}


def test_posts_from_update_channel_only():
    u = {"update_id": 1, "channel_post": {"chat": {"id": -99}, "text": "hi"}}
    posts = tc._posts_from_update(u, include_groups=False)
    assert len(posts) == 1
    assert tc._post_text(posts[0]) == "hi"


def test_posts_from_update_include_groups():
    u = {"update_id": 2, "message": {"chat": {"id": -1}, "text": "g"}}
    assert tc._posts_from_update(u, False) == []
    posts = tc._posts_from_update(u, True)
    assert tc._post_text(posts[0]) == "g"


def test_env_bool_default_and_opt_out(monkeypatch):
    monkeypatch.delenv("TELEGRAM_INCLUDE_GROUPS", raising=False)
    assert tc._env_bool("TELEGRAM_INCLUDE_GROUPS", default=True) is True
    monkeypatch.setenv("TELEGRAM_INCLUDE_GROUPS", "0")
    assert tc._env_bool("TELEGRAM_INCLUDE_GROUPS", default=True) is False


def test_posts_from_update_edited_channel():
    u = {"update_id": 3, "edited_channel_post": {"chat": {"id": -3}, "text": "edited"}}
    posts = tc._posts_from_update(u, False)
    assert len(posts) == 1
    assert tc._post_text(posts[0]) == "edited"


def test_strip_bot_username():
    assert tc._strip_bot_username("@MyBot hello", "MyBot") == "hello"
    assert tc._strip_bot_username("plain", "MyBot") == "plain"


def _mock_response(payload):
    r = MagicMock()
    r.status_code = 200
    r.text = ""
    r.json.return_value = payload
    return r


class _FakeAgent:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def execute_line(self, line: str, emit=None):
        self.lines.append(line)
        if emit:
            emit({"type": "output", "text": "printed line", "partial": False})
        return {"type": "turn", "answered": True, "answer": "final answer text"}


def test_run_listener_invokes_execute_line_once():
    agent = _FakeAgent()

    stop = threading.Event()
    calls = [0]

    def side_effect(*_a, **_kw):
        calls[0] += 1
        n = calls[0]
        if n == 1:
            return _mock_response({"ok": True, "result": {"url": ""}})
        if n == 2:
            return _mock_response(
                {
                    "ok": True,
                    "result": [
                        {
                            "update_id": 42,
                            "channel_post": {
                                "chat": {"id": -1005},
                                "text": "question from channel",
                            },
                        }
                    ],
                }
            )
        stop.set()
        return _mock_response({"ok": True, "result": []})

    fake_http = MagicMock()
    fake_http.get.side_effect = side_effect
    fake_http.post.return_value = _mock_response({"ok": True, "result": True})

    with patch.object(tc.requests, "Session", return_value=fake_http):
        tc.run_listener(
            agent_session=agent,
            print_fn=lambda *_a, **_k: None,
            bot_token="dummy",
            allowed_chat_ids={-1005},
            poll_timeout=1,
            stop_event=stop,
            mirror_replies=False,
        )

    assert agent.lines == ["question from channel"]
    assert fake_http.get.call_count == 3
    assert fake_http.post.call_count >= 1


def test_run_listener_filters_chat_id():
    agent = _FakeAgent()

    stop = threading.Event()
    calls = [0]

    def side_effect(*_a, **_kw):
        calls[0] += 1
        n = calls[0]
        if n == 1:
            return _mock_response({"ok": True, "result": {"url": ""}})
        if n == 2:
            return _mock_response(
                {
                    "ok": True,
                    "result": [
                        {
                            "update_id": 7,
                            "channel_post": {
                                "chat": {"id": -9999},
                                "text": "blocked",
                            },
                        }
                    ],
                }
            )
        stop.set()
        return _mock_response({"ok": True, "result": []})

    fake_http = MagicMock()
    fake_http.get.side_effect = side_effect
    fake_http.post.return_value = _mock_response({"ok": True, "result": True})

    with patch.object(tc.requests, "Session", return_value=fake_http):
        tc.run_listener(
            agent_session=agent,
            print_fn=lambda *_a, **_k: None,
            bot_token="dummy",
            allowed_chat_ids={-1005},
            poll_timeout=1,
            stop_event=stop,
            mirror_replies=False,
        )

    assert agent.lines == []
