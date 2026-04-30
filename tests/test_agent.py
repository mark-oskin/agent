"""
Unit-style tests for agent.py's agent loop.

Ollama and most network I/O are mocked; tools like search_web/fetch_page are stubbed
so behavior is deterministic.
"""

from __future__ import annotations

import importlib
import io
import json
import os
from contextlib import redirect_stdout

import pytest

from tests.harness import run_main
from agentlib.settings import AgentSettings


# --- parse_agent_json / normalization (1–12) ---


def test_safe_path_under_dir_rejects_traversal():
    d = importlib.import_module("agent")
    assert d._safe_path_under_dir("/a/b/skills", "../../etc/passwd") is None


def test_expand_skill_artifacts_includes_reference(tmp_path):
    d = importlib.import_module("agent")
    refd = tmp_path / "references"
    refd.mkdir(parents=True)
    (refd / "note.txt").write_text("REFBODY", encoding="utf-8")
    meta = {
        "prompt": "base",
        "reference_files": ["references/note.txt"],
        "grounding_commands": ["foo -h"],
    }
    out = d._expand_skill_artifacts(str(tmp_path), meta, "base")
    assert "base" in out
    assert "REFBODY" in out
    assert "foo -h" in out


def test_grounded_cli_skill_loads_bundled_reference():
    d = importlib.import_module("agent")
    skills = os.path.join(os.path.dirname(d.__file__), "skills")
    m = d._load_skills_from_dir(skills)
    assert "grounded_cli" in m
    p = m["grounded_cli"]["prompt"]
    assert "Bundled reference file: references/grounding_cli_template.md" in p
    assert "Unfamiliar CLI" in p or "grounding" in p.lower()


def test_parse_action_null_with_tool():
    d = importlib.import_module("agent")
    raw = json.dumps({"action": None, "tool": "search_web", "parameters": {"query": "x"}})
    out = d.parse_agent_json(raw)
    assert out["action"] == "tool_call"
    assert out["tool"] == "search_web"
    assert out["parameters"]["query"] == "x"


def test_tool_progress_message_includes_useful_params():
    d = importlib.import_module("agent")
    s1 = d._tool_progress_message("search_web", {"query": "hello world"})
    assert "query" in s1 and "hello world" in s1
    s2 = d._tool_progress_message("read_file", {"path": "/tmp/x.txt"})
    assert "path" in s2 and "/tmp/x.txt" in s2


def test_plugin_toolsets_load_and_route(monkeypatch):
    d = importlib.import_module("agent")
    assert "dev" in d._PLUGIN_TOOLSETS
    assert "desktop" in d._PLUGIN_TOOLSETS
    # If both are enabled, routing should pick dev for a pytest-y query.
    enabled_toolsets = {"dev", "desktop"}
    active = d._route_active_toolsets_for_request("please run pytest", enabled_toolsets)
    assert "dev" in active


def test_tools_dir_override_loads_plugins(tmp_path, monkeypatch):
    d = importlib.import_module("agent")
    tdir = tmp_path / "tools"
    tdir.mkdir()
    (tdir / "x.py").write_text(
        "def hi(params):\n"
        "    return 'hi'\n"
        "TOOLSET = {\n"
        "  'name': 'xset',\n"
        "  'description': 'x',\n"
        "  'triggers': ['hi'],\n"
        "  'tools': [{'id':'x_hi','description':'hi','aliases':['hi'], 'handler': hi}],\n"
        "}\n",
        encoding="utf-8",
    )
    d._TOOL_REGISTRY.load_plugin_toolsets(str(tdir))
    d._TOOL_REGISTRY.register_aliases()
    assert "xset" in d._PLUGIN_TOOLSETS
    assert "x_hi" in d._PLUGIN_TOOL_HANDLERS


def test_ollama_request_think_value_gpt_oss_defaults_to_medium(monkeypatch):
    d = importlib.import_module("agent")
    d._SETTINGS_OBJ = AgentSettings.defaults()
    d._settings_set(("ollama", "model"), "gpt-oss:20b")
    d._settings_set(("agent", "thinking"), True)
    d._settings_set(("agent", "thinking_level"), "")
    assert d._ollama_request_think_value() == "medium"


def test_ollama_request_think_false_when_disabled_even_if_level_set(monkeypatch):
    """Stale AGENT_THINKING_LEVEL must not send think= to Ollama when thinking is off."""
    d = importlib.import_module("agent")
    d._SETTINGS_OBJ = AgentSettings.defaults()
    d._settings_set(("agent", "thinking"), False)
    d._settings_set(("agent", "thinking_level"), "high")
    assert d._ollama_request_think_value() is False


def test_stream_thinking_prints_done_thinking_separator(monkeypatch):
    d = importlib.import_module("agent")
    d._SETTINGS_OBJ = AgentSettings.defaults()
    d._settings_set(("agent", "stream_thinking"), True)
    lines = iter(
        [
            json.dumps({"message": {"thinking": "Let's do that."}, "done": False}),
            json.dumps({"message": {"content": "final answer"}, "done": True}),
        ]
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        d._merge_stream_message_chunks(lines, stream_chunks=False)
    out = buf.getvalue()
    assert "[Thinking]" in out
    assert "[Done thinking]" in out


def test_parse_action_string_null_with_tool():
    d = importlib.import_module("agent")
    raw = '{"action": "null", "tool": "read_file", "parameters": {"path": "/tmp/a"}}'
    out = d.parse_agent_json(raw)
    assert out["action"] == "tool_call"
    assert out["tool"] == "read_file"


def test_parse_content_promoted_to_answer():
    d = importlib.import_module("agent")
    raw = json.dumps({"content": "hello", "action": None})
    out = d.parse_agent_json(raw)
    assert out["action"] == "answer"
    assert out["answer"] == "hello"
    assert "content" not in out


def test_parse_tool_top_level_run_command():
    d = importlib.import_module("agent")
    raw = json.dumps({"action": "run_command", "command": "echo hi"})
    out = d.parse_agent_json(raw)
    assert out["action"] == "tool_call"
    assert out["tool"] == "run_command"
    assert "echo hi" in out["parameters"].get("command", "")


def test_parse_use_git_top_level_op_and_worktree():
    d = importlib.import_module("agent")
    raw = json.dumps(
        {
            "action": "use_git",
            "op": "status",
            "worktree": "/tmp",
        }
    )
    out = d.parse_agent_json(raw)
    assert out["tool"] == "use_git"
    assert out["parameters"].get("op") == "status"
    assert out["parameters"].get("worktree") == "/tmp"


def test_parse_agent_json_literal_newlines_inside_answer():
    """Many models emit RFC-invalid JSON with literal control chars inside quoted strings."""
    d = importlib.import_module("agent")
    raw = '{"action":"answer","answer":"Line A\nLine B"}'.replace("\\n", "\n")
    assert "\n" in raw and "\\n" not in raw
    out = d.parse_agent_json(raw)
    assert out["action"] == "answer"
    assert "Line A" in out["answer"] and "Line B" in out["answer"]


def test_parse_agent_json_unicode_quote_delimiters():
    """Unicode smart quotes occasionally wrap JSON keys/strings."""
    d = importlib.import_module("agent")
    raw = "{\u201caction\u201d:\u201canswer\u201d,\u201canswer\u201d:\u201chi\u201d}"
    out = d.parse_agent_json(raw)
    assert out["action"] == "answer"
    assert out["answer"] == "hi"


def test_merge_tool_param_aliases_use_git():
    d = importlib.import_module("agent")
    p = d._merge_tool_param_aliases(
        "use_git", {"operation": "log", "cwd": "/tmp/proj", "m": "hi"}
    )
    assert p.get("op") == "log"
    assert p.get("worktree") == "/tmp/proj"
    assert p.get("message") == "hi"


def test_merge_tool_param_aliases_search():
    d = importlib.import_module("agent")
    p = d._merge_tool_param_aliases("search_web", {"q": "abc"})
    assert p["query"] == "abc"


def test_merge_tool_param_aliases_write_file_body():
    d = importlib.import_module("agent")
    p = d._merge_tool_param_aliases(
        "write_file", {"path": "x.txt", "body": "hello"}
    )
    assert p["path"] == "x.txt"
    assert p["content"] == "hello"


def test_ensure_tool_defaults_search_query_from_user():
    d = importlib.import_module("agent")
    p = d._ensure_tool_defaults("search_web", {}, "hello world")
    assert p["query"] == "hello world"


def test_is_tool_result_weak_requires_url_for_search_sections():
    d = importlib.import_module("agent")
    weak = d._is_tool_result_weak_for_dedup("[Web results]\nTitle: x\nSnippet: y")
    assert weak is True


def test_context_window_manager_summarizes_when_over_budget(monkeypatch):
    d = importlib.import_module("agent")
    d._SETTINGS_OBJ = AgentSettings.defaults()
    d._settings_set(("agent", "context_tokens"), 200)
    d._settings_set(("agent", "context_trigger_frac"), 0.50)
    d._settings_set(("agent", "context_target_frac"), 0.40)
    d._settings_set(("agent", "context_keep_tail_messages"), 4)

    def fake_summary(**kwargs):  # noqa: ARG001
        return "SUMMARY: keep constraints + decisions."

    monkeypatch.setattr(d, "_summarize_conversation_for_context", fake_summary)

    msgs = [{"role": "system", "content": "system rules here"}]
    # Build enough content to exceed the budget.
    for i in range(20):
        msgs.append({"role": "user", "content": f"u{i} " + ("x" * 60)})
        msgs.append({"role": "assistant", "content": f"a{i} " + ("y" * 60)})

    out = d._maybe_compact_context_window(
        msgs, user_query="do thing", primary_profile=None, verbose=0, context_cfg=None
    )
    # Should have inserted a running summary system message and shortened history.
    assert any(
        m.get("role") == "system"
        and isinstance(m.get("content"), str)
        and m["content"].startswith("Running conversation summary")
        for m in out
        if isinstance(m, dict)
    )
    assert len(out) < len(msgs)
    # Tail should remain (by content markers).
    tail_contents = [m.get("content") for m in out[-4:] if isinstance(m, dict)]
    assert any(isinstance(c, str) and "u19" in c for c in tail_contents)


def test_context_manager_prefs_applied_without_env(monkeypatch):
    d = importlib.import_module("agent")
    d._SETTINGS_OBJ = AgentSettings.defaults()

    def fake_summary(**kwargs):  # noqa: ARG001
        return "SUMMARY"

    monkeypatch.setattr(d, "_summarize_conversation_for_context", fake_summary)
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(12):
        msgs.append({"role": "user", "content": "x" * 80})
        msgs.append({"role": "assistant", "content": "y" * 80})
    out = d._maybe_compact_context_window(
        msgs,
        user_query="q",
        primary_profile=None,
        verbose=0,
        context_cfg={"enabled": True, "tokens": 200, "trigger_frac": 0.5, "target_frac": 0.4, "keep_tail_messages": 4},
    )
    assert len(out) < len(msgs)


def test_prompt_templates_resolve_overlay_and_full(tmp_path):
    d = importlib.import_module("agent")
    from agentlib import prompt_templates_io

    templates = prompt_templates_io.load_prompt_templates_from_dir(d._default_prompt_templates_dir())
    # overlay template yields default base + overlay.
    from agentlib import prompts as agent_prompts

    coding = agent_prompts.resolve_prompt_template_text("coding", templates)
    assert isinstance(coding, str) and len(coding) > 1000
    # full template overrides base.
    templates2 = dict(templates)
    templates2["xfull"] = {"kind": "full", "text": "FULL PROMPT"}
    out = agent_prompts.resolve_prompt_template_text("xfull", templates2)
    assert out == "FULL PROMPT"

    # path template loads from file.
    p = tmp_path / "p.txt"
    p.write_text("FILE PROMPT", encoding="utf-8")
    templates2["xfile"] = {"kind": "full", "path": str(p)}
    out2 = agent_prompts.resolve_prompt_template_text("xfile", templates2)
    assert out2 == "FILE PROMPT"
    strong = d._is_tool_result_weak_for_dedup("[Web results]\nLink: https://a.test\nTitle: t")
    assert strong is False


def test_enrich_search_query_adds_year_for_current(monkeypatch):
    d = importlib.import_module("agent")
    from datetime import date as real_date

    class FakeDate(real_date):
        @classmethod
        def today(cls):
            return real_date(2026, 4, 17)

    monkeypatch.setattr(d.datetime, "date", FakeDate)
    q = d._enrich_search_query_for_present_day("current widget specs")
    assert "2026" in q


# --- progress heartbeats (stderr) ---


def test_agent_progress_prints_to_stderr_when_enabled(monkeypatch, capsys):
    d = importlib.import_module("agent")
    d._SETTINGS_OBJ = AgentSettings.defaults()
    d._settings_set(("agent", "quiet"), False)
    d._settings_set(("agent", "progress"), True)
    d._agent_progress("hello")
    err = capsys.readouterr().err
    assert "→" in err
    assert "hello" in err


def test_agent_progress_silent_when_quiet(monkeypatch, capsys):
    d = importlib.import_module("agent")
    d._SETTINGS_OBJ = AgentSettings.defaults()
    d._settings_set(("agent", "quiet"), True)
    d._agent_progress("hello")
    assert capsys.readouterr().err == ""


def test_agent_progress_silent_when_progress_off(monkeypatch, capsys):
    d = importlib.import_module("agent")
    d._SETTINGS_OBJ = AgentSettings.defaults()
    d._settings_set(("agent", "quiet"), False)
    d._settings_set(("agent", "progress"), False)
    d._agent_progress("hello")
    assert capsys.readouterr().err == ""


def test_apply_cli_primary_model_ollama_sets_setting(monkeypatch):
    d = importlib.import_module("agent")
    d._SETTINGS_OBJ = AgentSettings.defaults()
    d._settings_set(("ollama", "model"), "before")
    p = d.default_primary_llm_profile()
    p2 = d._apply_cli_primary_model("after", p)
    assert p2 is p
    assert d._settings_get_str(("ollama", "model"), "") == "after"


def test_apply_cli_primary_model_hosted_replaces_model_only(monkeypatch):
    d = importlib.import_module("agent")
    d._SETTINGS_OBJ = AgentSettings.defaults()
    d._settings_set(("ollama", "model"), "ollama-unchanged")
    h = d.LlmProfile(
        backend="hosted",
        base_url="https://api.example.com/v1",
        model="old-hosted",
        api_key="sk-test",
    )
    h2 = d._apply_cli_primary_model("new-hosted", h)
    assert h2.model == "new-hosted"
    assert d._settings_get_str(("ollama", "model"), "") == "ollama-unchanged"


# --- main() loop / gates (13–30) ---


def test_main_simple_answer_no_web(monkeypatch):
    out = run_main(
        monkeypatch,
        ["hello"],
        [
            json.dumps({"action": "answer", "answer": "first"}),
            json.dumps({"action": "answer", "answer": "final"}),
        ],
        route_web=None,
        route_after_answer=None,
    )
    assert out == "final"


def test_no_web_skips_fetch_page_requirement(monkeypatch):
    out = run_main(
        monkeypatch,
        ["What is a mutex?"],
        [
            json.dumps({"action": "answer", "answer": "first"}),
            json.dumps({"action": "answer", "answer": "final"}),
        ],
        route_web=None,
        route_after_answer=None,
    )
    assert out == "final"


def test_web_required_allows_answer_after_strong_search(monkeypatch):
    """Web gate requires URL-backed search; fetch_page is not mandatory (avoids step-limit deadlocks)."""
    out = run_main(
        monkeypatch,
        ["q"],
        [
            json.dumps({"action": "tool_call", "tool": "search_web", "parameters": {"query": "q"}}),
            json.dumps({"action": "answer", "answer": "done"}),
        ],
        route_web="q",
        route_after_answer=None,
        stub_search_web=lambda q: "[Web results]\nLink: https://u.test\nTitle: t\nSnippet: s\n",
    )
    assert out == "done"


def test_web_required_blocks_curl_run_command(monkeypatch):
    responses = [
        json.dumps({"action": "tool_call", "tool": "search_web", "parameters": {"query": "q"}}),
        json.dumps(
            {
                "action": "tool_call",
                "tool": "run_command",
                "parameters": {"command": "curl -s https://example.com"},
            }
        ),
        json.dumps({"action": "tool_call", "tool": "fetch_page", "parameters": {"url": "https://example.com"}}),
        json.dumps({"action": "answer", "answer": "ok"}),
    ]
    out = run_main(
        monkeypatch,
        ["q"],
        responses,
        route_web="q",
        route_after_answer=None,
        stub_search_web=lambda q: "[Web results]\nLink: https://example.com\nTitle: t\nSnippet: s\n",
        stub_fetch_page=lambda url: f"Fetched URL: {url}\nFinal URL: {url}\n\nx",
        stub_run_command=lambda cmd: pytest.fail(f"run_command should not run curl: {cmd}"),
    )
    assert out == "ok"


def test_web_weak_search_triggers_retry_then_answer(monkeypatch):
    responses = [
        json.dumps({"action": "tool_call", "tool": "search_web", "parameters": {"query": "q"}}),
        json.dumps({"action": "tool_call", "tool": "search_web", "parameters": {"query": "q2"}}),
        json.dumps({"action": "tool_call", "tool": "fetch_page", "parameters": {"url": "https://u.test"}}),
        json.dumps({"action": "answer", "answer": "ok"}),
    ]
    calls = {"n": 0}

    def sw(q):
        calls["n"] += 1
        if calls["n"] == 1:
            return "[Web results]\nno urls here"
        return "[Web results]\nLink: https://u.test\nTitle: t\nSnippet: s\n"

    out = run_main(
        monkeypatch,
        ["q"],
        responses,
        route_web="q",
        route_after_answer=None,
        stub_search_web=sw,
        stub_fetch_page=lambda url: f"Fetched URL: {url}\nFinal URL: {url}\n\nz",
    )
    assert out == "ok"


def test_web_required_allows_answer_after_fetch_page_even_if_search_was_weak(monkeypatch):
    """
    When web verification is required, a successful fetch_page should count as verification too.
    This avoids step-limit loops when search_web is thin/blocked but fetch_page succeeds.
    """
    responses = [
        json.dumps({"action": "tool_call", "tool": "search_web", "parameters": {"query": "q"}}),
        json.dumps({"action": "tool_call", "tool": "fetch_page", "parameters": {"url": "https://u.test"}}),
        json.dumps({"action": "answer", "answer": "ok"}),
    ]
    out = run_main(
        monkeypatch,
        ["q"],
        responses,
        route_web="q",
        route_after_answer=None,
        stub_search_web=lambda q: "[Web results]\nno urls here",
        stub_fetch_page=lambda url: f"Fetched URL: {url}\nFinal URL: {url}\n\nbody",
    )
    assert out == "ok"


def test_malformed_action_recovered(monkeypatch):
    out = run_main(
        monkeypatch,
        ["q"],
        [
            "not json",
            json.dumps({"action": "answer", "answer": "fixed"}),
        ],
        route_web=None,
        route_after_answer=None,
    )
    assert out == "fixed"


def test_deliverable_blocks_answer_until_read_file(monkeypatch):
    # Avoid trailing-newline ambiguity between JSON text and Python triple-quoted strings.
    doc = ("# Title\n" + ("paragraph\n" * 200)).rstrip("\n")
    responses = [
        json.dumps(
            {
                "action": "tool_call",
                "tool": "write_file",
                "parameters": {"path": "out.md", "content": doc},
            }
        ),
        json.dumps({"action": "answer", "answer": "synopsis only"}),
        json.dumps({"action": "tool_call", "tool": "read_file", "parameters": {"path": "out.md"}}),
        json.dumps({"action": "answer", "answer": doc}),
    ]

    def wf(path, content):
        assert path == "out.md"
        return "File out.md written successfully."

    def rf(path):
        assert path == "out.md"
        return doc

    out = run_main(
        monkeypatch,
        ["Write a two page document about X. Write the document."],
        responses,
        route_web=None,
        route_after_answer=None,
        stub_write_file=wf,
        stub_read_file=rf,
    )
    assert out == doc


def test_deliverable_short_answer_after_read_file_rejected(monkeypatch):
    doc = ("LINE\n" * 400).rstrip("\n")
    responses = [
        json.dumps(
            {
                "action": "tool_call",
                "tool": "write_file",
                "parameters": {"path": "out.md", "content": doc},
            }
        ),
        json.dumps({"action": "tool_call", "tool": "read_file", "parameters": {"path": "out.md"}}),
        json.dumps({"action": "answer", "answer": "too short"}),
        json.dumps({"action": "answer", "answer": doc}),
    ]

    def wf(path, content):
        return "ok"

    def rf(path):
        return doc

    out = run_main(
        monkeypatch,
        ["Write a document about X. Write the document."],
        responses,
        route_web=None,
        route_after_answer=None,
        stub_write_file=wf,
        stub_read_file=rf,
    )
    assert out == doc


def test_router_after_answer_forces_search(monkeypatch):
    out = run_main(
        monkeypatch,
        ["plain"],
        [
            json.dumps({"action": "answer", "answer": "guess"}),
            json.dumps({"action": "tool_call", "tool": "search_web", "parameters": {"query": "q"}}),
            json.dumps({"action": "answer", "answer": "final"}),
        ],
        route_web=None,
        route_after_answer="verify-q",
        stub_search_web=lambda q: "[Web results]\nLink: https://a.test\nTitle: t\nSnippet: s\n",
    )
    assert out == "final"


def test_unknown_tool_returns_error_string_in_conversation(monkeypatch):
    """Unknown tool name should produce a tool result the model can see."""
    responses = [
        json.dumps({"action": "tool_call", "tool": "nope", "parameters": {}}),
        json.dumps({"action": "answer", "answer": "recovered"}),
    ]
    out = run_main(monkeypatch, ["q"], responses)
    assert out == "recovered"


def test_duplicate_tool_call_skipped(monkeypatch):
    responses = [
        json.dumps({"action": "tool_call", "tool": "search_web", "parameters": {"query": "same"}}),
        json.dumps({"action": "tool_call", "tool": "search_web", "parameters": {"query": "same"}}),
        json.dumps({"action": "answer", "answer": "done"}),
    ]
    calls = {"c": 0}

    def sw(q):
        calls["c"] += 1
        return "[Web results]\nLink: https://x.test\nTitle: t\nSnippet: s\n"

    out = run_main(monkeypatch, ["q"], responses, route_web=None, stub_search_web=sw)
    assert out == "done"
    assert calls["c"] == 1


def test_duplicate_weak_search_web_skipped(monkeypatch):
    """Weak URL-less results must still dedupe identical search_web parameters."""
    responses = [
        json.dumps({"action": "tool_call", "tool": "search_web", "parameters": {"query": "same"}}),
        json.dumps({"action": "tool_call", "tool": "search_web", "parameters": {"query": "same"}}),
        json.dumps({"action": "answer", "answer": "done"}),
    ]
    calls = {"c": 0}

    def sw(q):
        calls["c"] += 1
        return "[Web results]\nTitle: t\nSnippet: no link here\n"

    out = run_main(monkeypatch, ["q"], responses, route_web=None, stub_search_web=sw)
    assert out == "done"
    assert calls["c"] == 1


def test_write_file_error_does_not_set_deliverable_path(monkeypatch):
    responses = [
        json.dumps({"action": "answer", "answer": "x"}),
        json.dumps(
            {
                "action": "tool_call",
                "tool": "write_file",
                "parameters": {"path": "bad", "content": "c"},
            }
        ),
        json.dumps({"action": "answer", "answer": "final"}),
    ]

    def wf(path, content):
        return "Write error: boom"

    out = run_main(
        monkeypatch,
        ["Write a document. Write the document."],
        responses,
        stub_write_file=wf,
    )
    assert out == "final"


def test_fetch_page_prefix_contains_urls(monkeypatch):
    d = importlib.import_module("agent")

    class Resp:
        status_code = 200
        url = "https://example.com/there"
        text = "<html><body>Hi</body></html>"

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):  # noqa: ARG001
        assert "example.com" in url
        return Resp()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(d.requests, "get", fake_get)
    try:
        from agentlib.tools import builtins as tool_builtins

        out = tool_builtins.fetch_page("https://example.com/here")
        assert "Fetched URL:" in out
        assert "Final URL:" in out
        assert "https://example.com/there" in out
    finally:
        monkeypatch.undo()


def test_user_wants_written_deliverable_true():
    d = importlib.import_module("agent")
    assert d._user_wants_written_deliverable("Give me a 2 page document about foo. Write the document.") is True


def test_user_wants_written_deliverable_letter():
    d = importlib.import_module("agent")
    assert d._user_wants_written_deliverable("Write a letter to the mayor about parking.") is True


def test_deliverable_skip_mandatory_web_letter_without_sources():
    d = importlib.import_module("agent")
    assert d._deliverable_skip_mandatory_web("Write a letter to the president.")


def test_deliverable_skip_mandatory_web_false_when_sources_asked():
    d = importlib.import_module("agent")
    assert not d._deliverable_skip_mandatory_web(
        "Write a 2 page document about X. Source from web. Write the document."
    )


def test_is_self_capability_question():
    d = importlib.import_module("agent")
    h = d._is_self_capability_question
    assert h("What kind of model are you?")
    assert h("What kinds of outputs can you produce and inputs can you take?")
    assert not h("Who is the president of France?")


def test_self_capability_followup_lists_tools():
    d = importlib.import_module("agent")
    msg = d._self_capability_followup("What can you do?", "meta only")
    assert "search_web" in msg and "call_python" in msg
    assert "directly" in msg.lower()


def test_deliverable_first_answer_followup_demands_artifact():
    d = importlib.import_module("agent")
    msg = d._deliverable_first_answer_followup(
        "Write a letter to the president.",
        "No web search is needed because this is timeless.",
    )
    assert "full text" in msg.lower()
    assert "web search" in msg.lower()


def test_user_wants_written_deliverable_false():
    d = importlib.import_module("agent")
    assert d._user_wants_written_deliverable("What is 2+2?") is False


def test_answer_missing_written_body_threshold():
    d = importlib.import_module("agent")
    body = "A" * 1000
    assert d._answer_missing_written_body(body[:100], len(body)) is True
    assert d._answer_missing_written_body(body, len(body)) is False


def test_tool_result_user_message_includes_deliverable_reminder():
    d = importlib.import_module("agent")
    msg = d._tool_result_user_message("fetch_page", {"url": "u"}, "out", deliverable_reminder="REM")
    assert "REM" in msg


def test_parse_prose_json_embedded():
    d = importlib.import_module("agent")
    raw = 'prefix {"action":"answer","answer":"ok"} suffix'
    out = d.parse_agent_json(raw)
    assert out["action"] == "answer"


def test_step_limit_message(monkeypatch):
    """If model never answers, main prints step limit message."""
    responses = [json.dumps({"action": "tool_call", "tool": "search_web", "parameters": {"query": "q"}})] * 40
    out = run_main(
        monkeypatch,
        ["q"],
        responses,
        route_web=None,
        stub_search_web=lambda q: "[Web results]\nLink: https://a.test\nTitle: t\nSnippet: s\n",
    )
    assert "step limit" in out.lower()


def test_web_required_step_limit_when_never_strong_search(monkeypatch):
    responses = [json.dumps({"action": "tool_call", "tool": "search_web", "parameters": {"query": "q"}})] * 40

    def sw(q):
        return "No results found."

    out = run_main(
        monkeypatch,
        ["q"],
        responses,
        route_web="q",
        stub_search_web=sw,
    )
    assert "unable to verify" in out.lower()


def test_normalize_tool_name_alias_toolName():
    d = importlib.import_module("agent")
    raw = json.dumps({"action": "tool_call", "toolName": "list_directory", "parameters": {"path": "."}})
    out = d.parse_agent_json(raw)
    assert out["tool"] == "list_directory"


def test_main_error_action_prints(monkeypatch):
    out = run_main(
        monkeypatch,
        ["q"],
        [json.dumps({"action": "error", "error": "boom"})],
    )
    assert "boom" in out
