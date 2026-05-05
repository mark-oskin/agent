"""
Additional tests to bring the suite to 100 cases: native tool_calls, enrich rules,
real filesystem round-trips, usage edge cases, and more transcripts.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import sys
from contextlib import redirect_stdout

import pytest

from agentlib import agent_json
from agentlib.agent_json import AgentJsonDeps
from agentlib.repl.while_cmd import parse_while_judge_bit, parse_while_repl_tokens
from agentlib.tools import turn_support
from agentlib.tools.registry import ToolRegistry
from agentlib.tools.websearch import enrich_search_query_for_present_day, first_url_in_text
from tests.harness import build_test_app, build_test_session, j, run_main, run_session_lines
from agentlib.settings import AgentSettings

WEB = "[Web results]\nLink: https://a.example/x\nTitle: t\nSnippet: s\n"


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGISTRY = ToolRegistry(default_tools_dir=os.path.join(PROJECT_DIR, "tools"))
REGISTRY.load_plugin_toolsets(REGISTRY.default_tools_dir)
REGISTRY.register_aliases()
DEPS = AgentJsonDeps(
    all_known_tools=REGISTRY.all_known_tools,
    coerce_enabled_tools=REGISTRY.coerce_enabled_tools,
    merge_tool_param_aliases=turn_support.merge_tool_param_aliases,
)


# --- native Ollama tool_calls → agent JSON ---


def test_message_to_agent_json_text_prefers_tool_calls_over_noise():
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
    raw = agent_json.message_to_agent_json_text(msg, None, DEPS)
    out = agent_json.parse_agent_json(raw, DEPS)
    assert out["tool"] == "search_web"
    assert out["parameters"]["query"] == "from native"


def test_tool_calls_functions_prefix_maps_to_tool():
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
    raw = agent_json.message_to_agent_json_text(msg, None, DEPS)
    out = agent_json.parse_agent_json(raw, DEPS)
    assert out["tool"] == "read_file"


def test_tool_call_to_agent_dict_tool_dot_prefix():
    out = agent_json.tool_call_to_agent_dict("tool.list_directory", '{"path": "."}')
    assert out["tool"] == "list_directory"


def test_tool_call_to_agent_dict_filename_becomes_path():
    out = agent_json.tool_call_to_agent_dict(
        "write_file", json.dumps({"filename": "out.txt", "content": "z"})
    )
    assert out["parameters"]["path"] == "out.txt"
    assert "filename" not in out["parameters"]


# --- parse / merge edge cases ---


def test_parse_markdown_fence_json_variant():
    raw = "```JSON\n" + json.dumps({"action": "answer", "answer": "x"}) + "\n```"
    out = agent_json.parse_agent_json(raw, DEPS)
    assert out["answer"] == "x"


def test_parse_action_false_normalized_like_missing():
    raw = json.dumps({"action": False, "tool": "read_file", "parameters": {"path": "p"}})
    out = agent_json.parse_agent_json(raw, DEPS)
    assert out["action"] == "tool_call"


def test_iter_balanced_brace_objects_respects_string_escape():
    text = r'pre {"k": "a\"b"} post'
    spans = list(agent_json.iter_balanced_brace_objects(text))
    assert len(spans) == 1
    obj = agent_json.try_json_loads_object(spans[0])
    assert obj["k"] == 'a"b'


def test_first_url_in_text_extracts():
    s = "See https://example.com/path) for more"
    assert first_url_in_text(s).startswith("https://example.com/path")


# --- enrich (present-day bias) ---


def test_enrich_who_was_not_appended(monkeypatch):
    q = "Who was the first president of the USA?"
    assert enrich_search_query_for_present_day(q, settings=AgentSettings.defaults()) == q


def test_enrich_who_is_president_appends_current_year(monkeypatch):
    from datetime import date as real_date

    class FakeDate(real_date):
        @classmethod
        def today(cls):
            return real_date(2026, 6, 1)

    import agentlib.tools.websearch as websearch

    monkeypatch.setattr(websearch.datetime, "date", FakeDate)
    out = enrich_search_query_for_present_day("Who is the president of France?", settings=AgentSettings.defaults())
    assert "current" in out.lower()
    assert "2026" in out


def test_enrich_respects_ollama_search_enrich_off(monkeypatch):
    from agentlib.tools.websearch import enrich_search_query_for_present_day

    settings = AgentSettings.defaults()
    settings.set(("ollama", "search_enrich"), False)
    q = "Who is the president of France?"
    assert enrich_search_query_for_present_day(q, settings=settings) == q


# --- CLI usage ---


def test_main_interactive_mode_exits_on_eof(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["agent.py"])

    def _eof(_=""):
        raise EOFError()

    monkeypatch.setattr("builtins.input", _eof)
    buf = io.StringIO()
    with redirect_stdout(buf):
        import agentlib.app as app_mod

        app_mod.main()
    out = buf.getvalue()
    assert "Interactive mode" in out


def test_interactive_repl_settings_load_save(tmp_path, monkeypatch):
    ctx_file = tmp_path / "ctx.json"
    ctx_file.write_text(
        json.dumps([{"role": "user", "content": "hi from file"}]),
        encoding="utf-8",
    )
    out_file = tmp_path / "out.json"
    lines = [
        "/set model repl-test-model",
        "/set enable second_opinion",
        "/set disable second_opinion",
        "/set enable second-opinion",
        f"/load_context {ctx_file}",
        f"/save_context {out_file}",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert _app.settings.get_str(("ollama", "model"), "") == "repl-test-model"
    assert "second_opinion enabled" in out
    assert "second_opinion disabled" in out
    assert "Loaded 1 message" in out
    assert "Wrote current session" in out
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert isinstance(data["messages"], list)


def test_interactive_settings_verbose_toggle(monkeypatch):
    lines = [
        "/set verbose on",
        "/set verbose off",
        "/set enable verbose",
        "/set disable verbose",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "verbose level 2" in out
    assert "verbose level 0" in out


def test_agent_prefs_roundtrip(tmp_path, monkeypatch):
    from agentlib import prefs
    from agentlib.prefs import bootstrap as prefs_bootstrap
    from agentlib.llm.profile import default_primary_llm_profile

    pref_path = tmp_path / ".agent.json"
    prefs.set_agent_prefs_path_override(str(pref_path))
    app = build_test_app(monkeypatch)
    app.settings = AgentSettings.defaults()
    payload = prefs_bootstrap.build_agent_prefs_payload(
        settings=app.settings,
        core_tools=app.registry.core_tools,
        plugin_toolsets=app.registry.plugin_toolsets,
        primary_profile=default_primary_llm_profile(),
        second_opinion_on=True,
        cloud_ai_enabled=True,
        enabled_tools=set(app.registry.core_tools) - {"run_command"},
        reviewer_hosted_profile=None,
        reviewer_ollama_model="mymodel:latest",
        session_save_path="/tmp/x.json",
        system_prompt_override=None,
        system_prompt_path_override=None,
        context_manager={"enabled": False, "tokens": 1234, "trigger_frac": 0.7, "target_frac": 0.5, "keep_tail_messages": 9},
        verbose_level=1,
    )
    prefs.write_agent_prefs_file(payload)
    raw = prefs.load_agent_prefs()
    assert raw is not None
    st = prefs_bootstrap.session_defaults_from_prefs(
        raw,
        migrate_prefs=lambda p: prefs.apply_prefs_to_settings(app.settings, p),
        settings=app.settings,
        core_tools=app.registry.core_tools,
        plugin_toolsets=app.registry.plugin_toolsets,
        normalize_tool_name=app.registry.normalize_tool_name,
        merge_prompt_templates=lambda p: p,
        load_skills_from_dir=lambda p: {},
        resolved_prompt_templates_dir=lambda _p=None: "",
        resolved_skills_dir=lambda _p=None: "",
        resolved_tools_dir=lambda _p=None: "",
        default_prompt_templates_dir=lambda: "",
        default_skills_dir=lambda: "",
        load_plugin_toolsets=lambda tools_dir=None: None,
        register_tool_aliases=lambda: None,
    )
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
    prefs.set_agent_prefs_path_override(None)


def test_prefs_ollama_openai_agent_blobs_apply_to_settings(monkeypatch):
    from agentlib.prefs import bootstrap as prefs_bootstrap

    app = build_test_app(monkeypatch)
    app.settings = AgentSettings.defaults()
    prefs = {
        "version": 4,
        "ollama": {"HOST": "http://o.test:11434"},
        "openai": {"BASE_URL": "https://api.x/v1"},
        "agent": {"PROGRESS": "0"},
    }
    _ = prefs_bootstrap.session_defaults_from_prefs(
        prefs,
        migrate_prefs=lambda p: __import__("agentlib.prefs").prefs.apply_prefs_to_settings(app.settings, p),
        settings=app.settings,
        core_tools=app.registry.core_tools,
        plugin_toolsets=app.registry.plugin_toolsets,
        normalize_tool_name=app.registry.normalize_tool_name,
        merge_prompt_templates=lambda p: p,
        load_skills_from_dir=lambda p: {},
        resolved_prompt_templates_dir=lambda _p=None: "",
        resolved_skills_dir=lambda _p=None: "",
        resolved_tools_dir=lambda _p=None: "",
        default_prompt_templates_dir=lambda: "",
        default_skills_dir=lambda: "",
        load_plugin_toolsets=lambda tools_dir=None: None,
        register_tool_aliases=lambda: None,
    )
    assert app.settings.get_str(("ollama", "host"), "") == "http://o.test:11434"
    assert app.settings.get_str(("openai", "base_url"), "") == "https://api.x/v1"
    assert app.settings.get_bool(("agent", "progress"), True) is False


def test_prefs_stored_env_overrides_existing_settings(monkeypatch):
    from agentlib.prefs import bootstrap as prefs_bootstrap
    from agentlib import prefs as prefs_mod

    app = build_test_app(monkeypatch)
    app.settings = AgentSettings.defaults()
    app.settings.set(("ollama", "host"), "http://from-shell:1")
    prefs = {"version": 4, "ollama": {"HOST": "http://from-file:2"}}
    _ = prefs_bootstrap.session_defaults_from_prefs(
        prefs,
        migrate_prefs=lambda p: prefs_mod.apply_prefs_to_settings(app.settings, p),
        settings=app.settings,
        core_tools=app.registry.core_tools,
        plugin_toolsets=app.registry.plugin_toolsets,
        normalize_tool_name=app.registry.normalize_tool_name,
        merge_prompt_templates=lambda p: p,
        load_skills_from_dir=lambda p: {},
        resolved_prompt_templates_dir=lambda _p=None: "",
        resolved_skills_dir=lambda _p=None: "",
        resolved_tools_dir=lambda _p=None: "",
        default_prompt_templates_dir=lambda: "",
        default_skills_dir=lambda: "",
        load_plugin_toolsets=lambda tools_dir=None: None,
        register_tool_aliases=lambda: None,
    )
    # Prefs apply (no shell env precedence anymore).
    assert app.settings.get_str(("ollama", "host"), "") == "http://from-file:2"


def test_cli_config_overrides_default_agent_json_path(tmp_path, monkeypatch):
    from agentlib import prefs
    from agentlib.cli import parse_and_apply_cli_config_flag

    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"version": 3, "second_opinion_enabled": True}), encoding="utf-8")
    prefs.set_agent_prefs_path_override(None)
    # Apply via argv parsing helper (same logic as main()).
    rem = parse_and_apply_cli_config_flag(["--config", str(cfg), "hello"])
    assert rem == ["hello"]
    assert prefs.agent_prefs_path() == str(cfg)
    loaded = prefs.load_agent_prefs()
    assert isinstance(loaded, dict)
    assert loaded.get("second_opinion_enabled") is True


def test_prefs_system_prompt_inline_roundtrip(tmp_path, monkeypatch):
    from agentlib import prefs
    from agentlib.prefs import bootstrap as prefs_bootstrap
    from agentlib.llm.profile import default_primary_llm_profile

    pref_path = tmp_path / "prefs.json"
    prefs.set_agent_prefs_path_override(str(pref_path))
    app = build_test_app(monkeypatch)
    payload = prefs_bootstrap.build_agent_prefs_payload(
        settings=app.settings,
        core_tools=app.registry.core_tools,
        plugin_toolsets=app.registry.plugin_toolsets,
        primary_profile=default_primary_llm_profile(),
        second_opinion_on=False,
        cloud_ai_enabled=False,
        enabled_tools=set(app.registry.core_tools),
        reviewer_hosted_profile=None,
        reviewer_ollama_model=None,
        session_save_path=None,
        system_prompt_override="You are a penguin. JSON only.",
        system_prompt_path_override=None,
    )
    prefs.write_agent_prefs_file(payload)
    st = prefs_bootstrap.session_defaults_from_prefs(
        prefs.load_agent_prefs(),
        migrate_prefs=lambda p: prefs.apply_prefs_to_settings(app.settings, p),
        settings=app.settings,
        core_tools=app.registry.core_tools,
        plugin_toolsets=app.registry.plugin_toolsets,
        normalize_tool_name=app.registry.normalize_tool_name,
        merge_prompt_templates=lambda p: p,
        load_skills_from_dir=lambda p: {},
        resolved_prompt_templates_dir=lambda _p=None: "",
        resolved_skills_dir=lambda _p=None: "",
        resolved_tools_dir=lambda _p=None: "",
        default_prompt_templates_dir=lambda: "",
        default_skills_dir=lambda: "",
        load_plugin_toolsets=lambda tools_dir=None: None,
        register_tool_aliases=lambda: None,
    )
    assert st["system_prompt"] == "You are a penguin. JSON only."
    assert st["system_prompt_path"] is None
    prefs.set_agent_prefs_path_override(None)


def test_prefs_system_prompt_path_roundtrip(tmp_path, monkeypatch):
    from agentlib import prefs
    from agentlib.prefs import bootstrap as prefs_bootstrap
    from agentlib.llm.profile import default_primary_llm_profile

    pf = tmp_path / "mysys.txt"
    pf.write_text("Loaded from path.\n", encoding="utf-8")
    pref_path = tmp_path / "prefs.json"
    prefs.set_agent_prefs_path_override(str(pref_path))
    app = build_test_app(monkeypatch)
    payload = prefs_bootstrap.build_agent_prefs_payload(
        settings=app.settings,
        core_tools=app.registry.core_tools,
        plugin_toolsets=app.registry.plugin_toolsets,
        primary_profile=default_primary_llm_profile(),
        second_opinion_on=False,
        cloud_ai_enabled=False,
        enabled_tools=set(app.registry.core_tools),
        reviewer_hosted_profile=None,
        reviewer_ollama_model=None,
        session_save_path=None,
        system_prompt_override=None,
        system_prompt_path_override=str(pf),
    )
    prefs.write_agent_prefs_file(payload)
    st = prefs_bootstrap.session_defaults_from_prefs(
        prefs.load_agent_prefs(),
        migrate_prefs=lambda p: prefs.apply_prefs_to_settings(app.settings, p),
        settings=app.settings,
        core_tools=app.registry.core_tools,
        plugin_toolsets=app.registry.plugin_toolsets,
        normalize_tool_name=app.registry.normalize_tool_name,
        merge_prompt_templates=lambda p: p,
        load_skills_from_dir=lambda p: {},
        resolved_prompt_templates_dir=lambda _p=None: "",
        resolved_skills_dir=lambda _p=None: "",
        resolved_tools_dir=lambda _p=None: "",
        default_prompt_templates_dir=lambda: "",
        default_skills_dir=lambda: "",
        load_plugin_toolsets=lambda tools_dir=None: None,
        register_tool_aliases=lambda: None,
    )
    assert st["system_prompt_path"] == str(pf)
    assert st["system_prompt"] == "Loaded from path.\n"
    prefs.set_agent_prefs_path_override(None)


def test_interactive_settings_ollama_show_renders(tmp_path, monkeypatch):
    lines = ["/set ollama show", "/quit"]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0, prefs_path=str(tmp_path / "x.json"))
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert '"host"' in out and '"model"' in out


def test_interactive_settings_thinking_and_stream_thinking(tmp_path, monkeypatch):
    lines = [
        "/set thinking show",
        "/set thinking level high",
        "/set thinking show",
        "/set disable stream_thinking",
        "/set thinking off",
        "/set thinking show",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0, prefs_path=str(tmp_path / "x.json"))
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "thinking:" in out
    assert session.settings.get_str(("agent", "thinking_level"), "") in ("high", "")
    assert isinstance(session.settings.get_bool(("agent", "stream_thinking"), False), bool)


def test_interactive_settings_tools_lists_toolsets_and_describe(tmp_path, monkeypatch):
    lines = [
        "/set tools",
        "/set tools describe run_pytest",
        "/set tools describe dev",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0, prefs_path=str(tmp_path / "x.json"))
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "Toolsets (plugins)" in out
    assert "Tool: run_pytest" in out
    assert "Toolset: dev" in out


def test_interactive_settings_save_command(tmp_path, monkeypatch):
    from agentlib import prefs

    pref_path = tmp_path / "saved.json"
    lines = ["/set save", "/quit"]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(
            monkeypatch,
            verbose=0,
            second_opinion_enabled=True,
            prefs_path=str(pref_path),
            write_prefs=True,
        )
        run_session_lines(session, lines)
    assert "Saved settings" in buf.getvalue()
    data = json.loads(pref_path.read_text(encoding="utf-8"))
    assert data["version"] == 4
    assert data["second_opinion_enabled"] is True
    prefs.set_agent_prefs_path_override(None)


def test_ctrl_c_cancels_request_but_keeps_repl_running(tmp_path, monkeypatch):
    from agentlib.repl.loop import run_interactive_repl_loop

    def boom(*args, **kwargs):  # noqa: ARG001
        raise KeyboardInterrupt()

    _app, session = build_test_session(monkeypatch, verbose=0, prefs_path=str(tmp_path / "x.json"))
    monkeypatch.setattr(_app, "call_ollama_chat", boom)
    # Rebuild deps cache since we patched the method after session creation.
    session._conversation_turn_deps = _app.conversation_turn_deps()

    lines = iter(["hello", "/quit"])

    def repl_read_line(_prompt: str) -> str:
        return next(lines)

    buf = io.StringIO()
    with redirect_stdout(buf):
        run_interactive_repl_loop(
            session,
            install_readline=lambda: None,
            repl_read_line=repl_read_line,
            flush_repl_history=lambda: None,
            agent_progress=lambda _m: None,
        )
    out = buf.getvalue()
    assert "[Cancelled]" in out


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
    assert REGISTRY.normalize_tool_name("web search") == "search_web"
    assert REGISTRY.normalize_tool_name("shell") == "run_command"
    assert REGISTRY.normalize_tool_name("Search-Web") == "search_web"
    assert REGISTRY.normalize_tool_name("nope_tool_xyz") is None


def test_interactive_settings_tools_command(monkeypatch):
    lines = ["/set tools", "/quit"]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
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
    assert "list-tools" in out or "/set tools" in out


def test_interactive_phrase_disable_enable_web_search(monkeypatch):
    lines = [
        "/set disable web search",
        "/set enable web search",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "Tool disabled: search_web" in out
    assert "Tool enabled: search_web" in out


def test_interactive_settings_tools(monkeypatch):
    lines = [
        "/set disable search_web",
        "/set enable search_web",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "Tool disabled: search_web" in out
    assert "Tool enabled: search_web" in out


def test_route_requires_websearch_skips_when_search_web_disabled(monkeypatch):
    from agentlib import routing

    et = frozenset({"fetch_page", "run_command"})  # any set without search_web
    called = []

    def no_chat(*a, **k):
        called.append(1)
        return "{}"

    # If search_web isn't enabled, routing must return None without calling the model.
    assert (
        routing.route_requires_websearch(
            "today's news",
            "2026-01-01",
            primary_profile=None,
            enabled_tools=et,
            transcript_messages=None,
            coerce_enabled_tools=lambda x: set(x or set()),
            call_ollama_chat=no_chat,
            parse_agent_json=lambda _s: {},
            scalar_to_str=lambda x, default="": str(x) if x is not None else default,
            router_transcript_max_messages=80,
        )
        is None
    )
    assert called == []


def test_interactive_settings_llm_profiles(monkeypatch):
    lines = [
        "/set primary llm hosted https://api.example/v1 fake-model sk-test",
        "/set second_opinion llm ollama tinyllama:latest",
        "/set second_opinion llm hosted https://review.example/v1 rev-model",
        "/set primary llm ollama",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "https://api.example/v1" in out
    assert "https://review.example/v1" in out
    assert "tinyllama:latest" in out
    assert "Primary LLM: local Ollama." in out


def test_interactive_use_skill_unknown(monkeypatch):
    lines = [
        "/skill not_a_skill hello",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue().lower()
    assert "unknown skill" in out


def test_interactive_skill_list(monkeypatch):
    lines = [
        "/skill list",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert ("Skills:" in out) or ("(no skills loaded)" in out)


def test_interactive_skill_help_and_prompt_template_help(monkeypatch):
    lines = [
        "/skill help",
        "/set prompt_template help",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue().lower()
    assert "/skill" in out and "prompt_template" in out


def test_interactive_help_is_top_level(monkeypatch):
    lines = [
        "/help",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "/skill ..." in out
    assert "/set ..." in out
    assert "try /skill help" in out.lower()
    assert "try /set help" in out.lower()


def test_parse_while_repl_tokens_and_judge_bit():
    toks = shlex.split('/while --max 3 "pytest ok" do "fix code"')
    m, c, body = parse_while_repl_tokens(toks)
    assert m == 3 and c == "pytest ok" and body == ["fix code"]
    toks2 = shlex.split("/while 'x' do 'y'")
    assert parse_while_repl_tokens(toks2) == (50, "x", ["y"])
    toks3 = shlex.split('/while "c" do "p1", "p2"')
    assert parse_while_repl_tokens(toks3)[2] == ["p1", "p2"]
    toks4 = shlex.split('/while "c" do "a" , "b" , "c"')
    assert parse_while_repl_tokens(toks4)[2] == ["a", "b", "c"]
    with pytest.raises(ValueError):
        parse_while_repl_tokens(["/while"])
    with pytest.raises(ValueError):
        parse_while_repl_tokens(shlex.split("/while --max 0 'a' do 'b'"))
    assert parse_while_judge_bit("1") == 1
    assert parse_while_judge_bit("0") == 0
    assert parse_while_judge_bit("noise 1 trailing") == 1


def test_interactive_show_model_and_reviewer(monkeypatch):
    lines = ["/show model", "/show reviewer", "/quit"]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        session.settings.set(("ollama", "model"), "custom-llm:latest")
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "Primary LLM: ollama (" in out and "custom-llm:latest" in out
    assert "Second-opinion reviewer:" in out


def test_interactive_show_models_lists_local_ollama_models(monkeypatch):
    lines = ["/show models", "/quit"]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        session._fetch_ollama_local_model_names = lambda: ["qwen3.6:latest", "laguna-xs.2:latest"]
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "qwen3.6:latest" in out
    assert "laguna-xs.2:latest" in out


def test_interactive_source_reads_and_executes_lines(tmp_path, monkeypatch):
    script = tmp_path / "script.txt"
    script.write_text(
        "\n".join(
            [
                "/set model sourced-llm:latest",
                "/show model",
                "/quit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [f"/source {script}", "should not run", "/quit"]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "Primary LLM:" in out and "sourced-llm:latest" in out


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
    r = "[DuckDuckGo instant answer]\nSome text without link"
    assert turn_support.is_tool_result_weak_for_dedup(r) is True

