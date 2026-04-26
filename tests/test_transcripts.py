"""
Multi-step conversation transcripts: mock model emits a sequence of JSON tool/answer
steps; we assert final stdout and key invariants.
"""

from __future__ import annotations

import json

import pytest

from tests.harness import j, run_main

# --- shared stubs ---

WEB_SNIPPET = "[Web results]\nLink: https://src.example/doc\nTitle: Doc\nSnippet: text\n"


def fetch_ok(url: str) -> str:
    return f"Fetched URL: {url}\nFinal URL: {url}\n\nResearch body here."


def test_transcript_write_file_run_command_answer(monkeypatch):
    out = run_main(
        monkeypatch,
        ["Run my script"],
        [
            j(action="tool_call", tool="write_file", parameters={"path": "a.sh", "content": "#!/bin/sh\necho hi"}),
            j(action="tool_call", tool="run_command", parameters={"command": "sh a.sh"}),
            j(action="answer", answer="done"),
        ],
        route_web=None,
        route_after_answer=None,
        stub_write_file=lambda path, content: "ok",
        stub_run_command=lambda cmd: "STDOUT:\nhi\nSTDERR:\n",
    )
    assert out == "done"


def test_transcript_list_directory_read_file_answer(monkeypatch):
    out = run_main(
        monkeypatch,
        ["What files are here?"],
        [
            j(action="tool_call", tool="list_directory", parameters={"path": "."}),
            j(action="tool_call", tool="read_file", parameters={"path": "README"}),
            j(action="answer", answer="readme contents"),
        ],
        stub_list_directory=lambda path: '["README","agent.py"]',
        stub_read_file=lambda path: "README body",
    )
    assert out == "readme contents"


def test_transcript_read_replace_read_answer(monkeypatch):
    state = {"n": 0}

    def rf(path):
        state["n"] += 1
        return "foo OLD bar" if state["n"] == 1 else "foo NEW bar"

    out = run_main(
        monkeypatch,
        ["Fix the file"],
        [
            j(action="tool_call", tool="read_file", parameters={"path": "x.txt"}),
            j(
                action="tool_call",
                tool="replace_text",
                parameters={"path": "x.txt", "pattern": "OLD", "replacement": "NEW"},
            ),
            j(action="tool_call", tool="read_file", parameters={"path": "x.txt"}),
            j(action="answer", answer="updated"),
        ],
        stub_read_file=rf,
        stub_replace_text=lambda path, pattern, replacement, replace_all=True: "ok",
    )
    assert out == "updated"
    assert state["n"] == 2


def test_tool_exception_does_not_crash_agent(monkeypatch):
    """If a tool raises, the agent should not crash; it should report a tool fault and continue."""

    def boom(_cmd):
        raise RuntimeError("boom")

    out = run_main(
        monkeypatch,
        ["Run something"],
        [
            j(action="tool_call", tool="run_command", parameters={"command": "echo hi"}),
            j(action="answer", answer="done"),
        ],
        route_web=None,
        stub_run_command=boom,
    )
    assert out == "done"


@pytest.mark.parametrize(
    "tool,params,stub_kw",
    [
        ("search_web", {"query": "hello"}, {"stub_search_web": lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))}),
        ("fetch_page", {"url": "https://example.com"}, {"stub_fetch_page": lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))}),
        ("run_command", {"command": "echo hi"}, {"stub_run_command": lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))}),
        ("write_file", {"path": "x.txt", "content": "hi"}, {"stub_write_file": lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))}),
        ("read_file", {"path": "x.txt"}, {"stub_read_file": lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))}),
        ("list_directory", {"path": "."}, {"stub_list_directory": lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))}),
        ("download_file", {"url": "https://example.com/a", "path": "a.bin"}, {"stub_download_file": lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))}),
        ("tail_file", {"path": "app.log", "lines": 5}, {"stub_tail_file": lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))}),
        ("replace_text", {"path": "x.txt", "pattern": "a", "replacement": "b", "replace_all": True}, {"stub_replace_text": lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))}),
        ("call_python", {"code": "x=1"}, {"stub_call_python": lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))}),
    ],
)
def test_all_tools_exceptions_are_caught(monkeypatch, tool, params, stub_kw):
    """
    Any tool invocation raising an exception must not crash the agent loop.
    (use_git is implemented directly; we cover it separately below.)
    """
    extra = stub_kw

    out = run_main(
        monkeypatch,
        ["Run tool"],
        [
            j(action="tool_call", tool=tool, parameters=params),
            j(action="answer", answer="ok"),
        ],
        route_web=None,
        **extra,
    )
    assert out == "ok"


def test_use_git_exception_is_caught(monkeypatch):
    import agent as d

    def boom_use_git(_p):
        raise RuntimeError("boom")

    monkeypatch.setattr(d, "use_git", boom_use_git)
    out = run_main(
        monkeypatch,
        ["Run tool"],
        [
            j(action="tool_call", tool="use_git", parameters={"op": "status", "worktree": "."}),
            j(action="answer", answer="ok"),
        ],
        route_web=None,
    )
    assert out == "ok"


def test_transcript_tail_file_answer(monkeypatch):
    out = run_main(
        monkeypatch,
        ["Show log tail"],
        [
            j(action="tool_call", tool="tail_file", parameters={"path": "app.log", "lines": 5}),
            j(action="answer", answer="last lines ok"),
        ],
        stub_tail_file=lambda path, lines=20: "L4\nL5\n",
    )
    assert out == "last lines ok"


def test_transcript_call_python_answer(monkeypatch):
    out = run_main(
        monkeypatch,
        ["Compute with python"],
        [
            j(action="tool_call", tool="call_python", parameters={"code": "x = 40 + 2"}),
            j(action="answer", answer="forty-two"),
        ],
        stub_call_python=lambda code, globals=None: '{"x": 42}',
    )
    assert out == "forty-two"


def test_transcript_download_file_answer(monkeypatch):
    out = run_main(
        monkeypatch,
        ["Download the file"],
        [
            j(
                action="tool_call",
                tool="download_file",
                parameters={"url": "https://x.test/f.bin", "path": "f.bin"},
            ),
            j(action="answer", answer="saved"),
        ],
        stub_download_file=lambda url, path: f"saved to {path}",
    )
    assert out == "saved"


def test_transcript_web_required_initial_answer_nudged_then_search_fetch(monkeypatch):
    """Model tries to answer first; harness blocks until search + fetch."""
    out = run_main(
        monkeypatch,
        ["current facts please"],
        [
            j(action="answer", answer="guess"),
            j(action="tool_call", tool="search_web", parameters={"query": "q"}),
            j(action="tool_call", tool="fetch_page", parameters={"url": "https://u.test/p"}),
            j(action="answer", answer="verified"),
        ],
        route_web="q",
        route_after_answer=None,
        stub_search_web=lambda q: WEB_SNIPPET,
        stub_fetch_page=fetch_ok,
    )
    assert out == "verified"


def test_transcript_wget_blocked_when_web_required(monkeypatch):
    out = run_main(
        monkeypatch,
        ["q"],
        [
            j(action="tool_call", tool="search_web", parameters={"query": "q"}),
            j(
                action="tool_call",
                tool="run_command",
                parameters={"command": "wget -qO- https://example.com"},
            ),
            j(action="tool_call", tool="fetch_page", parameters={"url": "https://example.com"}),
            j(action="answer", answer="ok"),
        ],
        route_web="q",
        stub_search_web=lambda q: WEB_SNIPPET.replace("src.example", "example.com"),
        stub_fetch_page=fetch_ok,
        stub_run_command=lambda cmd: pytest.fail(f"wget should be blocked: {cmd}"),
    )
    assert out == "ok"


def test_transcript_three_bad_json_then_answer(monkeypatch):
    """Objects with null action and no tool hit the malformed-recovery branch."""
    bad = json.dumps({"action": None, "note": "incomplete"})
    out = run_main(
        monkeypatch,
        ["hi"],
        [
            bad,
            bad,
            bad,
            j(action="answer", answer="first"),
            j(action="answer", answer="finally"),
        ],
    )
    assert out == "finally"


def test_transcript_deliverable_plus_web_search_fetch_write_read(monkeypatch):
    doc = ("# Doc\n" + ("line\n" * 120)).rstrip("\n")
    out = run_main(
        monkeypatch,
        ["Write a 2 page document about X. Source from web. Write the document."],
        [
            j(action="tool_call", tool="search_web", parameters={"query": "X"}),
            j(action="tool_call", tool="fetch_page", parameters={"url": "https://src.example/doc"}),
            j(
                action="tool_call",
                tool="write_file",
                parameters={"path": "out.md", "content": doc},
            ),
            j(action="answer", answer="short"),
            j(action="tool_call", tool="read_file", parameters={"path": "out.md"}),
            j(action="answer", answer=doc),
        ],
        route_web="X",
        route_after_answer=None,
        stub_search_web=lambda q: WEB_SNIPPET,
        stub_fetch_page=fetch_ok,
        stub_write_file=lambda path, content: "ok",
        stub_read_file=lambda path: doc,
    )
    assert out == doc


def test_transcript_run_command_after_web_allowed_echo(monkeypatch):
    out = run_main(
        monkeypatch,
        ["pipeline"],
        [
            j(action="tool_call", tool="search_web", parameters={"query": "q"}),
            j(action="tool_call", tool="fetch_page", parameters={"url": "https://u.test"}),
            j(action="tool_call", tool="run_command", parameters={"command": "echo pipeline"}),
            j(action="answer", answer="done"),
        ],
        route_web="q",
        stub_search_web=lambda q: WEB_SNIPPET,
        stub_fetch_page=fetch_ok,
        stub_run_command=lambda cmd: "STDOUT:\npipeline\nSTDERR:\n",
    )
    assert out == "done"


def test_transcript_list_dir_after_fetch_then_answer(monkeypatch):
    out = run_main(
        monkeypatch,
        ["q"],
        [
            j(action="tool_call", tool="search_web", parameters={"query": "q"}),
            j(action="tool_call", tool="fetch_page", parameters={"url": "https://u.test"}),
            j(action="tool_call", tool="list_directory", parameters={"path": "."}),
            j(action="answer", answer="listed"),
        ],
        route_web="q",
        stub_search_web=lambda q: WEB_SNIPPET,
        stub_fetch_page=fetch_ok,
        stub_list_directory=lambda p: '["a"]',
    )
    assert out == "listed"


def test_transcript_read_file_error_then_retry(monkeypatch):
    n = {"c": 0}

    def rf(path):
        n["c"] += 1
        return "Read error: nope" if n["c"] == 1 else "second ok"

    out = run_main(
        monkeypatch,
        ["read x"],
        [
            j(action="tool_call", tool="read_file", parameters={"path": "x"}),
            j(action="tool_call", tool="read_file", parameters={"path": "x"}),
            j(action="answer", answer="got it"),
        ],
        stub_read_file=rf,
    )
    assert out == "got it"
    assert n["c"] == 2


def test_transcript_search_web_default_query_from_user_query(monkeypatch):
    """Empty search params should default to user query."""

    def sw(q):
        assert q.strip() == "my user topic"
        return WEB_SNIPPET

    out = run_main(
        monkeypatch,
        ["my user topic"],
        [
            j(action="tool_call", tool="search_web", parameters={}),
            j(action="answer", answer="s"),
        ],
        route_web=None,
        stub_search_web=sw,
    )
    assert out == "s"


def test_transcript_fetch_page_error_then_answer(monkeypatch):
    out = run_main(
        monkeypatch,
        ["q"],
        [
            j(action="tool_call", tool="search_web", parameters={"query": "q"}),
            j(action="tool_call", tool="fetch_page", parameters={"url": "https://bad"}),
            j(action="tool_call", tool="fetch_page", parameters={"url": "https://good"}),
            j(action="answer", answer="recovered"),
        ],
        route_web="q",
        stub_search_web=lambda q: WEB_SNIPPET,
        stub_fetch_page=lambda url: (
            "Fetch error: bad" if "bad" in url else fetch_ok(url)
        ),
    )
    assert out == "recovered"


def test_transcript_unknown_action_then_valid(monkeypatch):
    out = run_main(
        monkeypatch,
        ["x"],
        [
            json.dumps({"action": None, "foo": 1}),
            j(action="answer", answer="ok"),
            j(action="answer", answer="ok"),
        ],
    )
    assert out == "ok"


def test_transcript_tool_name_as_action_search_web(monkeypatch):
    """Some models emit known tool name as action."""
    out = run_main(
        monkeypatch,
        ["z"],
        [
            j(action="search_web", parameters={"query": "z"}),
            j(action="answer", answer="r"),
        ],
        route_web=None,
        stub_search_web=lambda q: WEB_SNIPPET,
    )
    assert out == "r"
