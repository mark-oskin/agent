"""
Further expansion: unit tests for agent helpers + longer / edge transcripts.
"""

from __future__ import annotations

import importlib
import json

import pytest

from tests.harness import j, run_main

WEB = "[Web results]\nLink: https://a.example/x\nTitle: t\nSnippet: s\n"


def _d():
    return importlib.import_module("agent")


# --- parse / normalize ---


def test_parse_nested_json_in_prose_prefers_tool_call():
    d = _d()
    raw = 'Here: {"action":"tool_call","tool":"read_file","parameters":{"path":"p"}} thanks'
    out = d.parse_agent_json(raw)
    assert out["action"] == "tool_call"
    assert out["tool"] == "read_file"


def test_parse_multiple_brace_objects_prefers_known_tool():
    d = _d()
    raw = (
        '{"noise":1} '
        '{"action":"tool_call","tool":"search_web","parameters":{"query":"q"}}'
    )
    out = d.parse_agent_json(raw)
    assert out["tool"] == "search_web"


def test_parse_fetch_page_as_top_level_action():
    d = _d()
    raw = json.dumps({"action": "fetch_page", "url": "https://u"})
    out = d.parse_agent_json(raw)
    assert out["action"] == "tool_call"
    assert out["tool"] == "fetch_page"
    assert "u" in out["parameters"].get("url", "")


def test_parse_read_file_alias_href_for_fetch_merged_to_url():
    d = _d()
    raw = json.dumps({"action": "tool_call", "tool": "fetch_page", "parameters": {"href": "https://z"}})
    out = d.parse_agent_json(raw)
    assert out["parameters"]["url"] == "https://z"


def test_parse_search_web_alias_q():
    d = _d()
    raw = json.dumps({"action": "tool_call", "tool": "search_web", "parameters": {"q": "abc"}})
    out = d.parse_agent_json(raw)
    assert out["parameters"]["query"] == "abc"


def test_parse_run_command_alias_cmd():
    d = _d()
    raw = json.dumps({"action": "tool_call", "tool": "run_command", "parameters": {"cmd": "echo x"}})
    out = d.parse_agent_json(raw)
    assert "echo x" in out["parameters"].get("command", "")


def test_parse_tool_arguments_single_quoted_dict():
    d = _d()
    out = d._parse_tool_arguments("{'query': 'x'}")
    assert out.get("query") == "x"


def test_scalar_to_str_list_joins():
    d = _d()
    assert d._scalar_to_str(["a", "b"], "") == "a b"


def test_scalar_to_int_bool():
    d = _d()
    assert d._scalar_to_int(True, 0) == 1
    assert d._scalar_to_int(False, 0) == 0


# --- streaming merge ---


def test_merge_stream_message_chunks_accumulates_content():
    d = _d()
    lines = [
        json.dumps({"message": {"content": "he"}, "done": False}),
        json.dumps({"message": {"content": "llo"}, "done": True}),
    ]
    msg, usage, streamed = d._merge_stream_message_chunks(iter(lines))
    assert msg["content"] == "hello"
    assert usage is None
    assert streamed is False


def test_merge_stream_message_chunks_collects_usage_from_final_chunk():
    d = _d()
    lines = [
        json.dumps({"message": {"content": "a"}, "done": False}),
        json.dumps(
            {
                "message": {"content": "b"},
                "done": True,
                "prompt_eval_count": 10,
                "eval_count": 3,
                "total_duration": 2_000_000_000,
            }
        ),
    ]
    msg, usage, streamed = d._merge_stream_message_chunks(iter(lines))
    assert msg["content"] == "ab"
    assert streamed is False
    assert usage is not None
    assert usage["prompt_eval_count"] == 10
    assert usage["eval_count"] == 3
    assert usage["total_duration"] == 2_000_000_000


def test_merge_stream_message_chunks_verbose_streams_content(capsys):
    d = _d()
    lines = [
        json.dumps({"message": {"content": "he"}, "done": False}),
        json.dumps({"message": {"content": "llo"}, "done": True}),
    ]
    msg, usage, streamed = d._merge_stream_message_chunks(iter(lines), stream_chunks=True)
    assert capsys.readouterr().out == "hello"
    assert streamed is True
    assert msg["content"] == "hello"


def test_ollama_generation_tok_per_sec_uses_eval_duration():
    d = _d()
    u = {"eval_count": 100, "eval_duration": 2_000_000_000}
    assert abs(d._ollama_eval_generation_tok_per_sec(u) - 50.0) < 1e-9


def test_format_ollama_usage_line_uses_ollama_field_names_and_gen_rate():
    d = _d()
    line = d._format_ollama_usage_line(
        {"prompt_eval_count": 12, "eval_count": 30, "eval_duration": 1_000_000_000}
    )
    assert "prompt_eval_count=12" in line
    assert "eval_count=30" in line
    assert "gen_tok/s≈30.0" in line


def test_tool_result_indicates_retryable_failure():
    d = _d()
    assert d._tool_result_indicates_retryable_failure("run_command", "Command error: boom")
    assert d._tool_result_indicates_retryable_failure("call_python", "Exec error: oops")
    assert d._tool_result_indicates_retryable_failure(
        "fetch_page", "Fetch error: HTTP 403 for this URL."
    )
    assert d._tool_result_indicates_retryable_failure(
        "search_web", "No results found for this search. Try again…"
    )
    assert not d._tool_result_indicates_retryable_failure(
        "run_command", "STDOUT:\nok\nSTDERR:\n"
    )
    assert not d._tool_result_indicates_retryable_failure("run_command", "")


def test_web_tool_result_followup_hint():
    d = _d()
    h = d._web_tool_result_followup_hint("fetch_page", "Fetch error: timeout")
    assert "search_web" in h and "URL" in h
    h2 = d._web_tool_result_followup_hint("search_web", "No results found for this search")
    assert "rephrase" in h2.lower() or "query" in h2.lower()


def test_parse_tool_recovery_payload():
    d = _d()
    raw = '{"recovery":"retry","parameters":{"command":"echo fixed"},"rationale":"use echo"}'
    p = d._parse_tool_recovery_payload(raw)
    assert p is not None
    params, rat = p
    assert params["command"] == "echo fixed"
    assert "echo" in rat.lower()


def test_confirm_tool_recovery_retry_tty_auto_without_prompt(monkeypatch):
    d = _d()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def _no_input(*_a, **_k):
        raise AssertionError("tool recovery no longer calls input()")

    monkeypatch.setattr("builtins.input", _no_input)
    assert d._confirm_tool_recovery_retry(
        "run_command",
        {"command": "bad"},
        {"command": "good"},
        "fix",
        interactive_tool_recovery=True,
    )


def test_confirm_tool_recovery_retry_non_tty_returns_true_without_env(monkeypatch):
    d = _d()
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.delenv("AGENT_AUTO_CONFIRM_TOOL_RETRY", raising=False)
    assert d._confirm_tool_recovery_retry(
        "run_command",
        {"command": "bad"},
        {"command": "good"},
        "fix",
        interactive_tool_recovery=True,
    )


def test_repl_buffered_line_max_bytes_env(monkeypatch):
    d = _d()
    monkeypatch.setenv("AGENT_REPL_INPUT_MAX_BYTES", "200000")
    try:
        assert d._repl_buffered_line_max_bytes() == 200000
    finally:
        monkeypatch.delenv("AGENT_REPL_INPUT_MAX_BYTES", raising=False)


def test_repl_buffered_line_max_bytes_minimum(monkeypatch):
    d = _d()
    monkeypatch.setenv("AGENT_REPL_INPUT_MAX_BYTES", "100")
    try:
        assert d._repl_buffered_line_max_bytes() == 4096
    finally:
        monkeypatch.delenv("AGENT_REPL_INPUT_MAX_BYTES", raising=False)


def test_tool_recovery_may_run_gated_on_tty_or_env(monkeypatch):
    d = _d()
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.delenv("AGENT_AUTO_CONFIRM_TOOL_RETRY", raising=False)
    assert not d._tool_recovery_may_run(True)
    monkeypatch.setenv("AGENT_AUTO_CONFIRM_TOOL_RETRY", "1")
    try:
        assert d._tool_recovery_may_run(False)
    finally:
        monkeypatch.delenv("AGENT_AUTO_CONFIRM_TOOL_RETRY", raising=False)


def test_merge_partial_tool_calls_merges_arguments_strings():
    d = _d()
    prev = [{"index": 0, "function": {"name": "search_web", "arguments": '{"quer'}}]
    new = [{"index": 0, "function": {"arguments": 'y":"z"}'}}]
    merged = d._merge_partial_tool_calls(prev, new)
    args = merged[0]["function"]["arguments"]
    parsed = d._parse_tool_arguments(args)
    assert parsed.get("query") == "z"


# --- transcripts: dedupe, verbose, errors, replace_all ---


def test_transcript_duplicate_search_web_skips_second_stub_call(monkeypatch):
    calls = {"n": 0}

    def sw(q):
        calls["n"] += 1
        return WEB

    out = run_main(
        monkeypatch,
        ["q"],
        [
            j(action="tool_call", tool="search_web", parameters={"query": "same"}),
            j(action="tool_call", tool="search_web", parameters={"query": "same"}),
            j(action="answer", answer="done"),
        ],
        stub_search_web=sw,
    )
    assert out == "done"
    assert calls["n"] == 1


def test_transcript_duplicate_fetch_page_skips_second_call(monkeypatch):
    calls = {"n": 0}

    def fp(url):
        calls["n"] += 1
        return f"Fetched URL: {url}\nFinal URL: {url}\n\nbody"

    out = run_main(
        monkeypatch,
        ["q"],
        [
            j(action="tool_call", tool="search_web", parameters={"query": "q"}),
            j(action="tool_call", tool="fetch_page", parameters={"url": "https://u"}),
            j(action="tool_call", tool="fetch_page", parameters={"url": "https://u"}),
            j(action="answer", answer="x"),
        ],
        route_web="q",
        stub_search_web=lambda q: WEB,
        stub_fetch_page=fp,
    )
    assert out == "x"
    assert calls["n"] == 1


def test_transcript_verbose_includes_tool_marker_in_stdout(monkeypatch):
    """verbose level 2 (default --verbose): tool lines on stdout."""
    out = run_main(
        monkeypatch,
        ["--verbose", "hi"],
        [
            j(action="tool_call", tool="search_web", parameters={"query": "q"}),
            j(action="answer", answer="ok"),
        ],
        stub_search_web=lambda q: WEB,
    )
    assert "[*] Executing tool: search_web" in out
    assert out.rstrip().endswith("ok")


def test_transcript_verbose_1_shows_tools_not_full_stream_path(monkeypatch):
    """Level 1 logs tool invocations (same tool line as level 2 with stubbed LLM)."""
    out = run_main(
        monkeypatch,
        ["--verbose", "1", "hi"],
        [
            j(action="tool_call", tool="search_web", parameters={"query": "q"}),
            j(action="answer", answer="ok"),
        ],
        stub_search_web=lambda q: WEB,
    )
    assert "[*] Executing tool: search_web" in out
    assert out.rstrip().endswith("ok")


def test_transcript_replace_text_replace_all_false(monkeypatch):
    out = run_main(
        monkeypatch,
        ["patch"],
        [
            j(
                action="tool_call",
                tool="replace_text",
                parameters={
                    "path": "f.txt",
                    "pattern": "a",
                    "replacement": "b",
                    "replace_all": False,
                },
            ),
            j(action="answer", answer="patched"),
        ],
        stub_replace_text=lambda path, pattern, replacement, replace_all=True: (
            f"ok replace_all={replace_all!r}"
        ),
    )
    assert out == "patched"


def test_transcript_call_python_error_then_answer(monkeypatch):
    out = run_main(
        monkeypatch,
        ["py"],
        [
            j(action="tool_call", tool="call_python", parameters={"code": "raise ValueError('nope')"}),
            j(action="answer", answer="handled"),
        ],
        stub_call_python=lambda code, globals=None: "Exec error: nope",
    )
    assert out == "handled"


def test_transcript_list_directory_error_then_answer(monkeypatch):
    out = run_main(
        monkeypatch,
        ["ls"],
        [
            j(action="tool_call", tool="list_directory", parameters={"path": "/nope"}),
            j(action="answer", answer="skip"),
        ],
        stub_list_directory=lambda path: "List dir error: denied",
    )
    assert out == "skip"


def test_transcript_tail_file_error_then_answer(monkeypatch):
    out = run_main(
        monkeypatch,
        ["tail"],
        [
            j(action="tool_call", tool="tail_file", parameters={"path": "/nope"}),
            j(action="answer", answer="done"),
        ],
        stub_tail_file=lambda path, lines=20: "Tail error: x",
    )
    assert out == "done"


def test_transcript_write_file_error_then_answer(monkeypatch):
    out = run_main(
        monkeypatch,
        ["save"],
        [
            j(action="tool_call", tool="write_file", parameters={"path": "x", "content": "y"}),
            j(action="answer", answer="aborted"),
        ],
        stub_write_file=lambda path, content: "Write error: disk full",
    )
    assert out == "aborted"


def test_transcript_download_error_then_answer(monkeypatch):
    out = run_main(
        monkeypatch,
        ["dl"],
        [
            j(
                action="tool_call",
                tool="download_file",
                parameters={"url": "https://x", "path": "out.bin"},
            ),
            j(action="answer", answer="give up"),
        ],
        stub_download_file=lambda url, path: "Download error: net",
    )
    assert out == "give up"


def test_transcript_web_two_answer_nudges_before_tools(monkeypatch):
    out = run_main(
        monkeypatch,
        ["facts"],
        [
            j(action="answer", answer="guess1"),
            j(action="answer", answer="guess2"),
            j(action="tool_call", tool="search_web", parameters={"query": "facts"}),
            j(action="tool_call", tool="fetch_page", parameters={"url": "https://u"}),
            j(action="answer", answer="final"),
        ],
        route_web="facts",
        stub_search_web=lambda q: WEB,
        stub_fetch_page=lambda url: f"Fetched URL: {url}\nFinal URL: {url}\n\nz",
    )
    assert out == "final"


def test_transcript_letter_meta_then_artifact(monkeypatch):
    """First tool-free answer must not be only web/timeliness meta for a letter request."""
    letter = "Dear Mayor,\n\nPlease plow Main Street.\n\nSincerely,\nA resident\n"
    out = run_main(
        monkeypatch,
        ["Write a letter to the mayor asking for snow plowing."],
        [
            j(action="answer", answer="No web search needed; this is timeless policy."),
            j(action="answer", answer=letter),
        ],
        route_web=None,
    )
    assert out.strip() == letter.strip()


def test_transcript_letter_clears_forced_web_route(monkeypatch):
    """Plain letter requests must not get router-mandated search_web (stability over model loops)."""
    out = run_main(
        monkeypatch,
        ["Write a letter to the mayor asking for better sidewalks."],
        [
            j(action="answer", answer="draft"),
            j(action="answer", answer="final letter text"),
        ],
        route_web="should-not-run",
        route_after_answer=None,
    )
    assert out == "final letter text"


def test_transcript_deliverable_no_web_write_read(monkeypatch):
    body = ("# R\n" + ("L\n" * 80)).rstrip("\n")
    out = run_main(
        monkeypatch,
        ["Write a document about Y. Write the document."],
        [
            j(action="tool_call", tool="write_file", parameters={"path": "y.md", "content": body}),
            j(action="tool_call", tool="read_file", parameters={"path": "y.md"}),
            j(action="answer", answer=body),
        ],
        route_web=None,
        stub_write_file=lambda p, c: "ok",
        stub_read_file=lambda p: body,
    )
    assert out == body


def test_transcript_unknown_top_level_action_recovery(monkeypatch):
    out = run_main(
        monkeypatch,
        ["u"],
        [
            json.dumps({"action": "maybe_later", "x": 1}),
            j(action="answer", answer="fixed"),
            j(action="answer", answer="fixed"),
        ],
    )
    assert out == "fixed"


def test_transcript_two_sequential_fetch_different_urls(monkeypatch):
    seen = []

    def fp(url):
        seen.append(url)
        return f"Fetched URL: {url}\nFinal URL: {url}\n\nok"

    out = run_main(
        monkeypatch,
        ["q"],
        [
            j(action="tool_call", tool="search_web", parameters={"query": "q"}),
            j(action="tool_call", tool="fetch_page", parameters={"url": "https://one"}),
            j(action="tool_call", tool="fetch_page", parameters={"url": "https://two"}),
            j(action="answer", answer="both"),
        ],
        route_web="q",
        stub_search_web=lambda q: WEB,
        stub_fetch_page=fp,
    )
    assert out == "both"
    assert seen == ["https://one", "https://two"]


def test_tool_result_user_message_truncates_long_output(monkeypatch):
    d = _d()
    monkeypatch.setenv("OLLAMA_TOOL_OUTPUT_MAX", "50")
    try:
        long_body = "x" * 100
        msg = d._tool_result_user_message("search_web", {"query": "q"}, long_body)
        assert "truncated" in msg
    finally:
        monkeypatch.delenv("OLLAMA_TOOL_OUTPUT_MAX", raising=False)


def test_enrich_present_day_does_not_double_year(monkeypatch):
    d = _d()
    from datetime import date as real_date

    class FakeDate(real_date):
        @classmethod
        def today(cls):
            return real_date(2026, 1, 1)

    monkeypatch.setattr(d.datetime, "date", FakeDate)
    q = d._enrich_search_query_for_present_day("current prices 2026")
    assert q.count("2026") == 1


def test_router_prompt_includes_user_query_snippet():
    d = _d()
    p = d._router_prompt("hello world", "2099-01-01")
    assert "hello world" in p
    assert "2099-01-01" in p


def test_router_prompt_hint_when_transcript_expected():
    d = _d()
    p = d._router_prompt("yesterday?", "2099-01-01", has_prior_transcript=True)
    assert "Earlier messages" in p


def test_route_requires_websearch_passes_transcript_to_llm(monkeypatch):
    d = _d()
    captured: dict = {}

    def cap(msgs, primary_profile=None, enabled_tools=None, verbose=0, **kwargs):
        captured["msgs"] = msgs
        return json.dumps({"action": "web_search", "query": "Seattle Mariners yesterday"})

    monkeypatch.setattr(d, "call_ollama_chat", cap)
    transcript = [
        {"role": "user", "content": "Mariners score today?"},
        {"role": "assistant", "content": '{"action":"answer","answer":"checking"}'},
    ]
    q = d._route_requires_websearch(
        "who won yesterday?",
        "2099-01-02",
        transcript_messages=transcript,
    )
    assert q == "Seattle Mariners yesterday"
    assert len(captured["msgs"]) == 3
    assert captured["msgs"][0]["role"] == "user"
    assert captured["msgs"][-1]["role"] == "user"
    assert "who won yesterday" in captured["msgs"][-1]["content"]


def test_router_transcript_slice_respects_max_messages(monkeypatch):
    d = _d()
    monkeypatch.setenv("AGENT_ROUTER_TRANSCRIPT_MAX_MESSAGES", "2")
    try:
        t = [{"role": "user", "content": str(i)} for i in range(5)]
        s = d._router_transcript_slice(t)
        assert len(s) == 2
        assert s[0]["content"] == "3"
        assert s[1]["content"] == "4"
    finally:
        monkeypatch.delenv("AGENT_ROUTER_TRANSCRIPT_MAX_MESSAGES", raising=False)


def test_deliverable_followup_contains_path():
    d = _d()
    assert "out.md" in d._deliverable_followup_block("out.md")


def test_is_tool_result_weak_for_fetch_error():
    d = _d()
    assert d._is_tool_result_weak_for_dedup("Fetch error: timeout") is True


def test_clean_json_response_strips_prefix():
    d = _d()
    assert d.clean_json_response('noise {"a":1}') == '{"a":1}'


def test_call_python_rejects_shell_like_source():
    d = _d()
    out = d.call_python('@echo off\necho not python')
    assert "Exec error" in out
    assert "write_file" in out or "Python" in out


def test_call_python_rejects_empty():
    d = _d()
    assert "empty" in d.call_python("").lower()


def test_call_python_runs_valid_code():
    d = _d()
    out = d.call_python("x = 40 + 2")
    assert json.loads(out).get("x") == 42


def test_call_python_captures_stdout_and_locals():
    d = _d()
    out = d.call_python('print("hello")\nresult = 99')
    assert "STDOUT:" in out
    assert "hello" in out
    assert "--- locals (JSON) ---" in out
    tail = out.split("--- locals (JSON) ---", 1)[1].strip()
    assert json.loads(tail).get("result") == 99


def test_call_python_locals_non_jsonable_values_use_str():
    d = _d()
    out = d.call_python(
        "class O: pass\nx = O()  # not JSON-serializable by default\nn = 1"
    )
    data = json.loads(out)
    assert data.get("n") == 1
    assert "O object" in str(data.get("x"))


def test_tool_params_fingerprint_search_web_canonical_query():
    d = _d()
    a = d._tool_params_fingerprint(
        "search_web",
        {"query": "  Hello  World ", "engine": "x", "max_results": 5},
    )
    b = d._tool_params_fingerprint("search_web", {"query": "hello world"})
    assert a == b
    c = d._tool_params_fingerprint("search_web", {"query": "hello world", "max_results": 10})
    assert a != c


def test_search_web_effective_max_results_clamped(monkeypatch):
    d = _d()
    monkeypatch.setenv("AGENT_SEARCH_WEB_MAX_RESULTS", "12")
    assert d._search_web_effective_max_results({}) == 12
    assert d._search_web_effective_max_results({"max_results": "7"}) == 7
    assert d._search_web_effective_max_results({"max_results": 99}) == 30
    assert d._search_web_effective_max_results({"max_results": 0}) == 1
    monkeypatch.delenv("AGENT_SEARCH_WEB_MAX_RESULTS", raising=False)
    assert d._search_web_effective_max_results({}) == 5


def test_write_file_rejects_empty_content():
    d = _d()
    out = d.write_file("/tmp/will-not-be-used-empty-letter.txt", "")
    assert out.startswith("Write error:")
    assert "content" in out.lower()


def test_extract_json_object_from_text_none_on_plain():
    d = _d()
    assert d._extract_json_object_from_text("no braces here") is None
