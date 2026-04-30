"""
Additional tests to bring the suite to 100 cases: native tool_calls, enrich rules,
real filesystem round-trips, usage edge cases, and more transcripts.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import shlex
import sys
from contextlib import redirect_stdout

import pytest

from tests.harness import j, run_main
from agentlib.settings import AgentSettings

WEB = "[Web results]\nLink: https://a.example/x\nTitle: t\nSnippet: s\n"


def _d():
    return importlib.import_module("agent")


# --- native Ollama tool_calls → agent JSON ---


def test_message_to_agent_json_text_prefers_tool_calls_over_noise():
    d = _d()
    msg = {
        "content": 'Say {"action":"answer","answer":"wrong"} here',
        "tool_calls": [
            {
                "index": 0,
                "function": {
                    "name": "search_web",
                    "arguments": '{"query": "from native"}',
                },
            }
        ],
    }
    raw = d._message_to_agent_json_text(msg)
    out = d.parse_agent_json(raw)
    assert out["tool"] == "search_web"
    assert out["parameters"]["query"] == "from native"


def test_tool_calls_functions_prefix_maps_to_tool():
    d = _d()
    msg = {
        "tool_calls": [
            {
                "function": {
                    "name": "functions.read_file",
                    "arguments": json.dumps({"path": "/tmp/x"}),
                }
            }
        ]
    }
    raw = d._message_to_agent_json_text(msg)
    out = d.parse_agent_json(raw)
    assert out["tool"] == "read_file"


def test_tool_call_to_agent_dict_tool_dot_prefix():
    d = _d()
    out = d._tool_call_to_agent_dict("tool.list_directory", '{"path": "."}')
    assert out["tool"] == "list_directory"


def test_tool_call_to_agent_dict_filename_becomes_path():
    d = _d()
    out = d._tool_call_to_agent_dict(
        "write_file", json.dumps({"filename": "out.txt", "content": "z"})
    )
    assert out["parameters"]["path"] == "out.txt"
    assert "filename" not in out["parameters"]


# --- parse / merge edge cases ---


def test_parse_markdown_fence_json_variant():
    d = _d()
    raw = "```JSON\n" + json.dumps({"action": "answer", "answer": "x"}) + "\n```"
    out = d.parse_agent_json(raw)
    assert out["answer"] == "x"


def test_parse_action_false_normalized_like_missing():
    d = _d()
    raw = json.dumps({"action": False, "tool": "read_file", "parameters": {"path": "p"}})
    out = d.parse_agent_json(raw)
    assert out["action"] == "tool_call"


def test_iter_balanced_brace_objects_respects_string_escape():
    d = _d()
    text = r'pre {"k": "a\"b"} post'
    spans = list(d._iter_balanced_brace_objects(text))
    assert len(spans) == 1
    obj = d._try_json_loads_object(spans[0])
    assert obj["k"] == 'a"b'


def test_first_url_in_text_extracts():
    d = _d()
    s = "See https://example.com/path) for more"
    assert d._first_url_in_text(s).startswith("https://example.com/path")


# --- enrich (present-day bias) ---


def test_enrich_who_was_not_appended(monkeypatch):
    d = _d()
    q = "Who was the first president of the USA?"
    out = d._enrich_search_query_for_present_day(q)
    assert out == q


def test_enrich_who_is_president_appends_current_year(monkeypatch):
    d = _d()
    from datetime import date as real_date

    class FakeDate(real_date):
        @classmethod
        def today(cls):
            return real_date(2026, 6, 1)

    monkeypatch.setattr(d.datetime, "date", FakeDate)
    out = d._enrich_search_query_for_present_day("Who is the president of France?")
    assert "current" in out.lower()
    assert "2026" in out


def test_enrich_respects_ollama_search_enrich_off(monkeypatch):
    d = _d()
    d._SETTINGS_OBJ = AgentSettings.defaults()
    d._settings_set(("ollama", "search_enrich"), False)
    q = "Who is the president of France?"
    assert d._enrich_search_query_for_present_day(q) == q


# --- CLI usage ---


def test_main_interactive_mode_exits_on_eof(monkeypatch):
    d = _d()
    monkeypatch.setattr(sys, "argv", ["agent.py"])

    def _eof(_=""):
        raise EOFError()

    monkeypatch.setattr("builtins.input", _eof)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d.main()
    out = buf.getvalue()
    assert "Interactive mode" in out


def test_interactive_repl_settings_load_save(tmp_path, monkeypatch):
    d = _d()
    d._SETTINGS_OBJ = AgentSettings.defaults()
    ctx_file = tmp_path / "ctx.json"
    ctx_file.write_text(
        json.dumps([{"role": "user", "content": "hi from file"}]),
        encoding="utf-8",
    )
    out_file = tmp_path / "out.json"
    lines = [
        "/settings model repl-test-model",
        "/settings enable second_opinion",
        "/settings disable second_opinion",
        "/settings enable second-opinion",
        f"/load_context {ctx_file}",
        f"/save_context {out_file}",
        "/quit",
    ]
    it = iter(lines)

    def fake_input(_=""):
        return next(it)

    monkeypatch.setattr("builtins.input", fake_input)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d._interactive_repl(
            verbose=0,
            second_opinion_enabled=False,
            cloud_ai_enabled=False,
            save_context_path=None,
        )
    out = buf.getvalue()
    assert d._settings_get_str(("ollama", "model"), "") == "repl-test-model"
    assert "second_opinion enabled" in out
    assert "second_opinion disabled" in out
    assert "Loaded 1 message" in out
    assert "Wrote current session" in out
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert isinstance(data["messages"], list)


def test_interactive_settings_verbose_toggle(monkeypatch):
    d = _d()
    lines = [
        "/settings verbose on",
        "/settings verbose off",
        "/settings enable verbose",
        "/settings disable verbose",
        "/quit",
    ]
    it = iter(lines)

    def fake_input(_=""):
        return next(it)

    monkeypatch.setattr("builtins.input", fake_input)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d._interactive_repl(
            verbose=0,
            second_opinion_enabled=False,
            cloud_ai_enabled=False,
            save_context_path=None,
        )
    out = buf.getvalue()
    assert "verbose level 2" in out
    assert "verbose level 0" in out


def test_agent_prefs_roundtrip(tmp_path, monkeypatch):
    d = _d()
    pref_path = tmp_path / ".agent.json"
    d._set_agent_prefs_path_override(str(pref_path))
    d._SETTINGS_OBJ = AgentSettings.defaults()
    payload = d._build_agent_prefs_payload(
        primary_profile=d.default_primary_llm_profile(),
        second_opinion_on=True,
        cloud_ai_enabled=True,
        enabled_tools=set(d._CORE_TOOLS) - {"run_command"},
        reviewer_hosted_profile=None,
        reviewer_ollama_model="mymodel:latest",
        session_save_path="/tmp/x.json",
        system_prompt_override=None,
        system_prompt_path_override=None,
        context_manager={"enabled": False, "tokens": 1234, "trigger_frac": 0.7, "target_frac": 0.5, "keep_tail_messages": 9},
        verbose_level=1,
    )
    d._write_agent_prefs_file(payload)
    raw = d._load_agent_prefs()
    assert raw is not None
    st = d._session_defaults_from_prefs(raw)
    assert st["second_opinion_enabled"] is True
    assert st["cloud_ai_enabled"] is True
    assert "run_command" not in st["enabled_tools"]
    assert st["reviewer_ollama_model"] == "mymodel:latest"
    assert st["save_context_path"] == "/tmp/x.json"
    assert st["verbose"] == 1
    assert isinstance(st.get("context_manager"), dict)
    assert st["context_manager"]["enabled"] is False
    assert st["context_manager"]["tokens"] == 1234
    data = json.loads(pref_path.read_text(encoding="utf-8"))
    assert data["version"] == 4
    assert data.get("system_prompt") is None
    assert data.get("system_prompt_path") is None
    assert data.get("verbose") == 1
    d._set_agent_prefs_path_override(None)


def test_prefs_ollama_openai_agent_blobs_apply_to_settings(monkeypatch):
    d = _d()
    d._SETTINGS_OBJ = AgentSettings.defaults()
    prefs = {
        "version": 4,
        "ollama": {"HOST": "http://o.test:11434"},
        "openai": {"BASE_URL": "https://api.x/v1"},
        "agent": {"PROGRESS": "0"},
    }
    d._session_defaults_from_prefs(prefs)
    assert d._settings_get_str(("ollama", "host"), "") == "http://o.test:11434"
    assert d._settings_get_str(("openai", "base_url"), "") == "https://api.x/v1"
    assert d._settings_get_bool(("agent", "progress"), True) is False


def test_prefs_stored_env_overrides_existing_settings(monkeypatch):
    d = _d()
    d._SETTINGS_OBJ = AgentSettings.defaults()
    d._settings_set(("ollama", "host"), "http://from-shell:1")
    prefs = {"version": 4, "ollama": {"HOST": "http://from-file:2"}}
    d._session_defaults_from_prefs(prefs)
    # Prefs apply (no shell env precedence anymore).
    assert d._settings_get_str(("ollama", "host"), "") == "http://from-file:2"


def test_cli_config_overrides_default_agent_json_path(tmp_path, monkeypatch):
    d = _d()
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"version": 3, "second_opinion_enabled": True}), encoding="utf-8")
    d._set_agent_prefs_path_override(None)
    # Apply via argv parsing helper (same logic as main()).
    rem = d._parse_and_apply_cli_config_flag(["--config", str(cfg), "hello"])
    assert rem == ["hello"]
    assert d._agent_prefs_path() == str(cfg)
    loaded = d._load_agent_prefs()
    assert isinstance(loaded, dict)
    assert loaded.get("second_opinion_enabled") is True


def test_prefs_system_prompt_inline_roundtrip(tmp_path, monkeypatch):
    d = _d()
    pref_path = tmp_path / "prefs.json"
    d._set_agent_prefs_path_override(str(pref_path))
    payload = d._build_agent_prefs_payload(
        primary_profile=d.default_primary_llm_profile(),
        second_opinion_on=False,
        cloud_ai_enabled=False,
        enabled_tools=set(d._CORE_TOOLS),
        reviewer_hosted_profile=None,
        reviewer_ollama_model=None,
        session_save_path=None,
        system_prompt_override="You are a penguin. JSON only.",
        system_prompt_path_override=None,
    )
    d._write_agent_prefs_file(payload)
    st = d._session_defaults_from_prefs(d._load_agent_prefs())
    assert st["system_prompt"] == "You are a penguin. JSON only."
    assert st["system_prompt_path"] is None
    d._set_agent_prefs_path_override(None)


def test_prefs_system_prompt_path_roundtrip(tmp_path, monkeypatch):
    d = _d()
    pf = tmp_path / "mysys.txt"
    pf.write_text("Loaded from path.\n", encoding="utf-8")
    pref_path = tmp_path / "prefs.json"
    d._set_agent_prefs_path_override(str(pref_path))
    payload = d._build_agent_prefs_payload(
        primary_profile=d.default_primary_llm_profile(),
        second_opinion_on=False,
        cloud_ai_enabled=False,
        enabled_tools=set(d._CORE_TOOLS),
        reviewer_hosted_profile=None,
        reviewer_ollama_model=None,
        session_save_path=None,
        system_prompt_override=None,
        system_prompt_path_override=str(pf),
    )
    d._write_agent_prefs_file(payload)
    st = d._session_defaults_from_prefs(d._load_agent_prefs())
    assert st["system_prompt_path"] == str(pf)
    assert st["system_prompt"] == "Loaded from path.\n"
    d._set_agent_prefs_path_override(None)


def test_interactive_settings_ollama_show_renders(tmp_path, monkeypatch):
    d = _d()
    d._set_agent_prefs_path_override(str(tmp_path / "x.json"))
    lines = ["/settings ollama show", "/quit"]
    it = iter(lines)

    def fake_input(_=""):
        return next(it)

    monkeypatch.setattr("builtins.input", fake_input)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d._interactive_repl(
            verbose=0,
            second_opinion_enabled=False,
            cloud_ai_enabled=False,
            save_context_path=None,
            prefs_loaded=False,
        )
    out = buf.getvalue()
    assert '"host"' in out and '"model"' in out
    d._set_agent_prefs_path_override(None)


def test_interactive_settings_thinking_and_stream_thinking(tmp_path, monkeypatch):
    d = _d()
    d._SETTINGS_OBJ = AgentSettings.defaults()
    d._set_agent_prefs_path_override(str(tmp_path / "x.json"))
    lines = [
        "/settings thinking show",
        "/settings thinking level high",
        "/settings thinking show",
        "/settings disable stream_thinking",
        "/settings thinking off",
        "/settings thinking show",
        "/quit",
    ]
    it = iter(lines)

    def fake_input(_=""):
        return next(it)

    monkeypatch.setattr("builtins.input", fake_input)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d._interactive_repl(
            verbose=0,
            second_opinion_enabled=False,
            cloud_ai_enabled=False,
            save_context_path=None,
            prefs_loaded=False,
        )
    out = buf.getvalue()
    assert "thinking:" in out
    assert d._settings_get_str(("agent", "thinking_level"), "") in ("high", "")
    assert isinstance(d._settings_get_bool(("agent", "stream_thinking"), False), bool)
    d._set_agent_prefs_path_override(None)


def test_interactive_settings_tools_lists_toolsets_and_describe(tmp_path, monkeypatch):
    d = _d()
    d._set_agent_prefs_path_override(str(tmp_path / "x.json"))
    lines = [
        "/settings tools",
        "/settings tools describe run_pytest",
        "/settings tools describe dev",
        "/quit",
    ]
    it = iter(lines)

    def fake_input(_=""):
        return next(it)

    monkeypatch.setattr("builtins.input", fake_input)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d._interactive_repl(
            verbose=0,
            second_opinion_enabled=False,
            cloud_ai_enabled=False,
            save_context_path=None,
            prefs_loaded=False,
        )
    out = buf.getvalue()
    assert "Toolsets (plugins)" in out
    assert "Tool: run_pytest" in out
    assert "Toolset: dev" in out
    d._set_agent_prefs_path_override(None)


def test_interactive_settings_save_command(tmp_path, monkeypatch):
    d = _d()
    pref_path = tmp_path / "saved.json"
    d._set_agent_prefs_path_override(str(pref_path))
    lines = ["/settings save", "/quit"]
    it = iter(lines)

    def fake_input(_=""):
        return next(it)

    monkeypatch.setattr("builtins.input", fake_input)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d._interactive_repl(
            verbose=0,
            second_opinion_enabled=True,
            cloud_ai_enabled=False,
            save_context_path=None,
            prefs_loaded=False,
        )
    assert "Saved settings" in buf.getvalue()
    data = json.loads(pref_path.read_text(encoding="utf-8"))
    assert data["version"] == 4
    assert data["second_opinion_enabled"] is True
    d._set_agent_prefs_path_override(None)


def test_ctrl_c_cancels_request_but_keeps_repl_running(tmp_path, monkeypatch):
    d = _d()
    d._set_agent_prefs_path_override(str(tmp_path / "x.json"))

    def boom(*args, **kwargs):  # noqa: ARG001
        raise KeyboardInterrupt()

    monkeypatch.setattr(d, "call_ollama_chat", boom)
    lines = ["hello", "/quit"]
    it = iter(lines)

    def fake_input(_=""):
        return next(it)

    monkeypatch.setattr("builtins.input", fake_input)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d._interactive_repl(
            verbose=0,
            second_opinion_enabled=False,
            cloud_ai_enabled=False,
            save_context_path=None,
            prefs_loaded=False,
        )
    out = buf.getvalue()
    assert "[Cancelled]" in out
    d._set_agent_prefs_path_override(None)


def test_cli_disable_enable_tool_flags(monkeypatch):
    from tests.harness import j, run_main

    out = run_main(
        monkeypatch,
        ["-disable_tool", "search_web", "-enable_tool", "search_web", "hello"],
        [
            j(action="answer", answer="draft"),
            j(action="answer", answer="hi"),
        ],
        route_web=None,
    )
    assert out == "hi"


def test_normalize_tool_user_aliases():
    d = _d()
    assert d._normalize_tool_name("web search") == "search_web"
    assert d._normalize_tool_name("shell") == "run_command"
    assert d._normalize_tool_name("Search-Web") == "search_web"
    assert d._normalize_tool_name("nope_tool_xyz") is None


def test_interactive_settings_tools_command(monkeypatch):
    d = _d()
    lines = ["/settings tools", "/quit"]
    it = iter(lines)

    def fake_input(_=""):
        return next(it)

    monkeypatch.setattr("builtins.input", fake_input)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d._interactive_repl(
            verbose=0,
            second_opinion_enabled=False,
            cloud_ai_enabled=False,
            save_context_path=None,
        )
    out = buf.getvalue()
    assert "search_web" in out
    assert "run_command" in out
    assert "[on]" in out


def test_cli_list_tools(monkeypatch):
    out = run_main(monkeypatch, ["--list-tools"], [])
    assert "search_web" in out
    assert "shell" in out.lower() or "run_command" in out


def test_cli_enable_tool_accepts_user_alias(monkeypatch):
    out = run_main(
        monkeypatch,
        ["-disable_tool", "search_web", "-enable_tool", "web", "hello"],
        [
            j(action="answer", answer="draft"),
            j(action="answer", answer="hi"),
        ],
        route_web=None,
    )
    assert out == "hi"


def test_cli_unknown_tool_prints_hint(monkeypatch):
    out = run_main(monkeypatch, ["-enable_tool", "notatool"], [])
    assert "Unknown tool" in out
    assert "list-tools" in out or "/settings tools" in out


def test_interactive_phrase_disable_enable_web_search(monkeypatch):
    d = _d()
    lines = [
        "/settings disable web search",
        "/settings enable web search",
        "/quit",
    ]
    it = iter(lines)

    def fake_input(_=""):
        return next(it)

    monkeypatch.setattr("builtins.input", fake_input)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d._interactive_repl(
            verbose=0,
            second_opinion_enabled=False,
            cloud_ai_enabled=False,
            save_context_path=None,
        )
    out = buf.getvalue()
    assert "Tool disabled: search_web" in out
    assert "Tool enabled: search_web" in out


def test_interactive_settings_tools(monkeypatch):
    d = _d()
    lines = [
        "/settings disable search_web",
        "/settings enable search_web",
        "/quit",
    ]
    it = iter(lines)

    def fake_input(_=""):
        return next(it)

    monkeypatch.setattr("builtins.input", fake_input)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d._interactive_repl(
            verbose=0,
            second_opinion_enabled=False,
            cloud_ai_enabled=False,
            save_context_path=None,
        )
    out = buf.getvalue()
    assert "Tool disabled: search_web" in out
    assert "Tool enabled: search_web" in out


def test_route_requires_websearch_skips_when_search_web_disabled(monkeypatch):
    d = _d()
    et = frozenset(t for t in d._CORE_TOOLS if t != "search_web")
    called = []

    def no_chat(*a, **k):
        called.append(1)
        return "{}"

    monkeypatch.setattr(d, "call_ollama_chat", no_chat)
    assert d._route_requires_websearch("today's news", "2026-01-01", enabled_tools=et) is None
    assert called == []


def test_interactive_settings_llm_profiles(monkeypatch):
    d = _d()
    lines = [
        "/settings primary llm hosted https://api.example/v1 fake-model sk-test",
        "/settings second_opinion llm ollama tinyllama:latest",
        "/settings second_opinion llm hosted https://review.example/v1 rev-model",
        "/settings primary llm ollama",
        "/quit",
    ]
    it = iter(lines)

    def fake_input(_=""):
        return next(it)

    monkeypatch.setattr("builtins.input", fake_input)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d._interactive_repl(
            verbose=0,
            second_opinion_enabled=False,
            cloud_ai_enabled=False,
            save_context_path=None,
        )
    out = buf.getvalue()
    assert "https://api.example/v1" in out
    assert "https://review.example/v1" in out
    assert "tinyllama:latest" in out
    assert "Primary LLM: local Ollama." in out


def test_interactive_use_skill_unknown(monkeypatch):
    d = _d()
    lines = [
        "/skill not_a_skill hello",
        "/quit",
    ]
    it = iter(lines)

    def fake_input(_=""):
        return next(it)

    monkeypatch.setattr("builtins.input", fake_input)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d._interactive_repl(
            verbose=0,
            second_opinion_enabled=False,
            cloud_ai_enabled=False,
            save_context_path=None,
        )
    out = buf.getvalue().lower()
    assert "unknown skill" in out


def test_interactive_skill_list(monkeypatch):
    d = _d()
    lines = [
        "/skill list",
        "/quit",
    ]
    it = iter(lines)

    def fake_input(_=""):
        return next(it)

    monkeypatch.setattr("builtins.input", fake_input)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d._interactive_repl(
            verbose=0,
            second_opinion_enabled=False,
            cloud_ai_enabled=False,
            save_context_path=None,
        )
    out = buf.getvalue()
    assert ("Skills:" in out) or ("(no skills loaded)" in out)


def test_interactive_skill_help_and_prompt_template_help(monkeypatch):
    d = _d()
    lines = [
        "/skill help",
        "/settings prompt_template help",
        "/quit",
    ]
    it = iter(lines)

    def fake_input(_=""):
        return next(it)

    monkeypatch.setattr("builtins.input", fake_input)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d._interactive_repl(
            verbose=0,
            second_opinion_enabled=False,
            cloud_ai_enabled=False,
            save_context_path=None,
        )
    out = buf.getvalue().lower()
    assert "/skill" in out and "prompt_template" in out


def test_interactive_help_is_top_level(monkeypatch):
    d = _d()
    lines = [
        "/help",
        "/quit",
    ]
    it = iter(lines)

    def fake_input(_=""):
        return next(it)

    monkeypatch.setattr("builtins.input", fake_input)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d._interactive_repl(
            verbose=0,
            second_opinion_enabled=False,
            cloud_ai_enabled=False,
            save_context_path=None,
        )
    out = buf.getvalue()
    assert "/skill ..." in out
    assert "/settings ..." in out
    assert "try /skill help" in out.lower()
    assert "try /settings help" in out.lower()


def test_parse_while_repl_tokens_and_judge_bit():
    d = _d()
    toks = shlex.split('/while --max 3 "pytest ok" do "fix code"')
    m, c, body = d._parse_while_repl_tokens(toks)
    assert m == 3 and c == "pytest ok" and body == ["fix code"]
    toks2 = shlex.split("/while 'x' do 'y'")
    assert d._parse_while_repl_tokens(toks2) == (50, "x", ["y"])
    toks3 = shlex.split('/while "c" do "p1", "p2"')
    assert d._parse_while_repl_tokens(toks3)[2] == ["p1", "p2"]
    toks4 = shlex.split('/while "c" do "a" , "b" , "c"')
    assert d._parse_while_repl_tokens(toks4)[2] == ["a", "b", "c"]
    with pytest.raises(ValueError):
        d._parse_while_repl_tokens(["/while"])
    with pytest.raises(ValueError):
        d._parse_while_repl_tokens(shlex.split("/while --max 0 'a' do 'b'"))
    assert d._parse_while_judge_bit("1") == 1
    assert d._parse_while_judge_bit("0") == 0
    assert d._parse_while_judge_bit("noise 1 trailing") == 1


def test_interactive_show_model_and_reviewer(monkeypatch):
    d = _d()
    d._SETTINGS_OBJ = AgentSettings.defaults()
    d._settings_set(("ollama", "model"), "custom-llm:latest")
    lines = ["/show model", "/show reviewer", "/quit"]
    it = iter(lines)

    def fake_input(_=""):
        return next(it)

    monkeypatch.setattr("builtins.input", fake_input)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d._interactive_repl(
            verbose=0,
            second_opinion_enabled=False,
            cloud_ai_enabled=False,
            save_context_path=None,
        )
    out = buf.getvalue()
    assert "Primary LLM: ollama (" in out and "custom-llm:latest" in out
    assert "Second-opinion reviewer:" in out


# --- transcripts ---


def test_transcript_web_three_early_answers_before_tools(monkeypatch):
    out = run_main(
        monkeypatch,
        ["need web"],
        [
            j(action="answer", answer="g1"),
            j(action="answer", answer="g2"),
            j(action="answer", answer="g3"),
            j(action="tool_call", tool="search_web", parameters={"query": "need web"}),
            j(action="tool_call", tool="fetch_page", parameters={"url": "https://u.test"}),
            j(action="answer", answer="done"),
        ],
        route_web="need web",
        stub_search_web=lambda q: WEB,
        stub_fetch_page=lambda url: f"Fetched URL: {url}\nFinal URL: {url}\n\nz",
    )
    assert out == "done"


def test_transcript_download_success(monkeypatch):
    out = run_main(
        monkeypatch,
        ["get it"],
        [
            j(
                action="tool_call",
                tool="download_file",
                parameters={"url": "https://files.example/a.tgz", "path": "a.tgz"},
            ),
            j(action="answer", answer="ok"),
        ],
        stub_download_file=lambda url, path: f"Downloaded {url} -> {path}",
    )
    assert out == "ok"


def test_transcript_call_python_with_globals_parameter(monkeypatch):
    out = run_main(
        monkeypatch,
        ["run"],
        [
            j(
                action="tool_call",
                tool="call_python",
                parameters={"code": "y = x + 1", "globals": {"x": 41}},
            ),
            j(action="answer", answer="done"),
        ],
        stub_call_python=lambda code, globals=None: '{"y": 42}',
    )
    assert out == "done"


def test_transcript_uppercase_CURL_run_command_runs_under_web_required(monkeypatch):
    """Current implementation only blocks lowercase curl/wget (word-boundary match)."""
    seen = []

    def rc(cmd):
        seen.append(cmd)
        return "STDOUT:\nran\nSTDERR:\n"

    out = run_main(
        monkeypatch,
        ["q"],
        [
            j(action="tool_call", tool="search_web", parameters={"query": "q"}),
            j(
                action="tool_call",
                tool="run_command",
                parameters={"command": "CURL -s https://example.com"},
            ),
            j(action="tool_call", tool="fetch_page", parameters={"url": "https://example.com"}),
            j(action="answer", answer="x"),
        ],
        route_web="q",
        stub_search_web=lambda q: WEB.replace("a.example", "example"),
        stub_fetch_page=lambda url: f"Fetched URL: {url}\nFinal URL: {url}\n\nb",
        stub_run_command=rc,
    )
    assert out == "x"
    assert seen and "CURL" in seen[0]


def test_transcript_write_replace_read_answer(monkeypatch):
    body = "ONE\nTWO\nTHREE\n"

    def rf(path):
        return "ONE\nFIXED\nTHREE\n"

    out = run_main(
        monkeypatch,
        ["patch file"],
        [
            j(action="tool_call", tool="write_file", parameters={"path": "w.txt", "content": body}),
            j(
                action="tool_call",
                tool="replace_text",
                parameters={"path": "w.txt", "pattern": "TWO", "replacement": "FIXED"},
            ),
            j(action="tool_call", tool="read_file", parameters={"path": "w.txt"}),
            j(action="answer", answer="ONE\nFIXED\nTHREE\n"),
        ],
        stub_write_file=lambda p, c: "ok",
        stub_replace_text=lambda path, pattern, replacement, replace_all=True: "ok",
        stub_read_file=rf,
    )
    assert "FIXED" in out


def test_transcript_router_after_answer_forces_search_then_fetch(monkeypatch):
    out = run_main(
        monkeypatch,
        ["verify me"],
        [
            j(action="answer", answer="draft"),
            j(action="tool_call", tool="search_web", parameters={"query": "verify me"}),
            j(action="tool_call", tool="fetch_page", parameters={"url": "https://u"}),
            j(action="answer", answer="final"),
        ],
        route_web=None,
        route_after_answer="must search",
        stub_search_web=lambda q: WEB,
        stub_fetch_page=lambda url: f"Fetched URL: {url}\nFinal URL: {url}\n\nbody",
    )
    assert out == "final"


# --- real filesystem (no mocks) ---


def test_write_file_and_read_file_roundtrip(tmp_path, monkeypatch):
    from agentlib.tools import builtins as tool_builtins

    p = tmp_path / "round.txt"
    monkeypatch.chdir(tmp_path)
    tool_builtins.write_file(str(p), "alpha\nbeta\n")
    assert tool_builtins.read_file(str(p)) == "alpha\nbeta\n"


def test_replace_text_on_real_file(tmp_path, monkeypatch):
    from agentlib.tools import builtins as tool_builtins

    monkeypatch.chdir(tmp_path)
    p = tmp_path / "r.txt"
    p.write_text("aa OLD bb OLD cc", encoding="utf-8")
    tool_builtins.replace_text(str(p), "OLD", "NEW", replace_all=True)
    assert p.read_text(encoding="utf-8") == "aa NEW bb NEW cc"


def test_list_directory_real(tmp_path, monkeypatch):
    from agentlib.tools import builtins as tool_builtins

    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.txt").write_text("y", encoding="utf-8")
    raw = tool_builtins.list_directory(str(tmp_path))
    names = json.loads(raw)
    assert "a.txt" in names and "b.txt" in names


# --- weak / dedupe helpers ---


def test_is_tool_result_weak_instant_answer_without_url():
    d = _d()
    r = "[DuckDuckGo instant answer]\nSome text without link"
    assert d._is_tool_result_weak_for_dedup(r) is True

