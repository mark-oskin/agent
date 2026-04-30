"""
Shared test harness: mock Ollama + stub tools for agent.main().
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from typing import Any, Callable, List, Optional

import pytest


def build_test_app(monkeypatch: pytest.MonkeyPatch):
    """
    Create a fresh app instance for tests.

    Tests should not depend on `agent.py` internals; `agent.py` is a shim.
    """
    from agentlib.app import default_app
    from agentlib import AgentSettings

    app = default_app()
    app.settings = AgentSettings.defaults()
    # Keep tests deterministic: ignore the developer's ~/.agent.json entirely.
    monkeypatch.setattr(app, "load_prefs", lambda: None)
    return app


def build_test_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    verbose: int = 0,
    second_opinion_enabled: bool = False,
    cloud_ai_enabled: bool = False,
    save_context_path: Optional[str] = None,
    enabled_tools: Optional[set[str]] = None,
    prefs_path: Optional[str] = None,
    write_prefs: bool = False,
):
    """
    Create a fresh AgentSession for interactive-command tests.

    This is intentionally lower-level than `agentlib.app._interactive_repl`: tests should exercise
    `AgentSession.execute_line` directly and keep only minimal coverage for the REPL loop itself.
    """
    from agentlib.session import AgentSession
    from agentlib.llm.profile import LlmProfile, default_primary_llm_profile
    from agentlib import prompt_templates_io, prompts
    from agentlib.context.io import load_context_messages, save_context_bundle
    from agentlib.skills.loader import load_skills_from_dir
    from agentlib import prefs as prefs_mod
    from agentlib.prefs import bootstrap as prefs_bootstrap
    from agentlib.repl.while_cmd import call_while_condition_judge, parse_while_repl_tokens

    app = build_test_app(monkeypatch)
    if prefs_path is not None:
        prefs_mod.set_agent_prefs_path_override(str(prefs_path))


    templates = prompt_templates_io.load_prompt_templates_from_dir(app.default_prompt_templates_dir())
    skills_map = load_skills_from_dir(app.default_skills_dir())
    primary_profile = default_primary_llm_profile()
    enabled_tools_use = enabled_tools if enabled_tools is not None else set(app.registry.core_tools)

    def describe_llm_profile_short(p: LlmProfile) -> str:
        if p.backend != "hosted":
            return "ollama"
        key = "set" if (p.api_key or "").strip() else "missing"
        return f"hosted {p.model!r} @ {p.base_url!r} (api_key: {key})"

    def format_session_primary_llm_line(p: LlmProfile) -> str:
        if p.backend == "hosted":
            return describe_llm_profile_short(p)
        return f"ollama ({app.ollama_model()!r})"

    def format_session_reviewer_line(hosted: Optional[LlmProfile], ollama_model: Optional[str]) -> str:
        if hosted is not None and hosted.backend == "hosted":
            return describe_llm_profile_short(hosted)
        return f"ollama ({(ollama_model or app.ollama_second_opinion_model())!r})"

    def hosted_review_ready(cloud: bool, reviewer) -> bool:
        if cloud and app.settings.get_str(("openai", "api_key"), "").strip():
            return True
        if (
            reviewer is not None
            and getattr(reviewer, "backend", "") == "hosted"
            and (getattr(reviewer, "api_key", "") or "").strip()
        ):
            return True
        return False

    def interactive_turn_user_message(
        user_query: str,
        today_str: str,
        second_opinion: bool,
        cloud: bool,
        *,
        primary_profile=None,
        reviewer_ollama_model=None,
        reviewer_hosted_profile=None,
        enabled_tools=None,
        system_instruction_override: Optional[str] = None,
        skill_suffix: Optional[str] = None,
    ) -> str:
        return prompts.interactive_turn_user_message(
            user_query=user_query,
            today_str=today_str,
            second_opinion=second_opinion,
            cloud=cloud,
            primary_profile=primary_profile or default_primary_llm_profile(),
            reviewer_ollama_model=reviewer_ollama_model,
            reviewer_hosted_profile=reviewer_hosted_profile,
            enabled_tools=enabled_tools,
            system_instruction_override=system_instruction_override,
            skill_suffix=skill_suffix,
            ollama_model=app.ollama_model(),
            hosted_review_ready=hosted_review_ready,
            tool_policy_runner_text=app.registry.tool_policy_runner_text,
        )

    def call_while_judge(condition: str, messages: list, *, primary_profile, verbose: int) -> int:
        return call_while_condition_judge(
            condition,
            messages,
            primary_profile=primary_profile,
            verbose=verbose,
            default_primary_llm_profile=default_primary_llm_profile,
            call_hosted_chat_plain=app.call_hosted_chat_plain,
            call_ollama_plaintext=app.call_ollama_plaintext,
            ollama_model=app.ollama_model(),
        )

    session = AgentSession(
        settings=app.settings,
        verbose=verbose,
        second_opinion_enabled=second_opinion_enabled,
        cloud_ai_enabled=cloud_ai_enabled,
        save_context_path=save_context_path,
        enabled_tools=frozenset(enabled_tools_use),
        enabled_toolsets=frozenset(),
        primary_profile=primary_profile,
        reviewer_hosted_profile=None,
        reviewer_ollama_model=None,
        skills_map=skills_map,
        prompt_templates=templates,
        prompt_template_default="coding",
        prompt_templates_dir=app.default_prompt_templates_dir(),
        skills_dir=app.default_skills_dir(),
        tools_dir=app.default_tools_dir(),
        context_cfg={},
        system_prompt_override=None,
        system_prompt_path=None,
        session_prompt_template=None,
        agent_progress=app.agent_progress,
        fetch_ollama_local_model_names=lambda: [],
        format_last_ollama_usage_for_repl=lambda: "",
        format_session_primary_llm_line=format_session_primary_llm_line,
        format_session_reviewer_line=format_session_reviewer_line,
        print_skill_usage_verbose=app.print_skill_usage_verbose,
        match_skill_detail=lambda u, sm: (None, None),
        ml_select_skill_id=lambda *_a, **_k: (None, "disabled"),
        skill_plan_steps=lambda **_k: (None, "disabled"),
        effective_enabled_tools_for_skill=app.effective_enabled_tools_for_skill,
        effective_enabled_tools_for_turn=app.registry.effective_enabled_tools_for_turn,
        route_requires_websearch=app.route_requires_websearch,
        deliverable_skip_mandatory_web=lambda _q: False,
        user_wants_written_deliverable=lambda _q: False,
        interactive_turn_user_message=interactive_turn_user_message,
        conversation_turn_deps=app.conversation_turn_deps(),
        save_context_bundle=save_context_bundle,
        load_context_messages=load_context_messages,
        registry=app.registry,
        build_agent_prefs_payload=lambda **kwargs: prefs_bootstrap.build_agent_prefs_payload(
            settings=app.settings,
            core_tools=app.registry.core_tools,
            plugin_toolsets=app.registry.plugin_toolsets,
            **kwargs,
        ),
        write_agent_prefs_file=(prefs_mod.write_agent_prefs_file if write_prefs else (lambda _payload: None)),
        agent_prefs_path=(prefs_mod.agent_prefs_path if write_prefs else (lambda: "")),
        settings_group_keys_lines=app.settings.group_keys_lines,
        settings_group_show=app.settings.group_show,
        settings_group_set=app.settings.group_set,
        settings_group_unset=app.settings.group_unset,
        settings_get=app.settings.get,
        settings_set=app.settings.set,
        LlmProfile_cls=LlmProfile,
        default_primary_llm_profile=default_primary_llm_profile,
        describe_llm_profile_short=describe_llm_profile_short,
        ollama_second_opinion_model=app.ollama_second_opinion_model,
        ollama_request_think_value=app.ollama_request_think_value,
        agent_thinking_level=lambda: app.settings.get_str(("agent", "thinking_level"), ""),
        agent_thinking_enabled_default_false=lambda: app.settings.get_bool(("agent", "thinking"), False),
        agent_stream_thinking_enabled=lambda: app.settings.get_bool(("agent", "stream_thinking"), False),
        verbose_ack_message=app.verbose_ack_message,
        parse_while_repl_tokens=parse_while_repl_tokens,
        call_while_condition_judge=call_while_judge,
    )
    return app, session


def run_session_lines(session, lines: list[str]) -> None:
    """
    Drive `AgentSession.execute_line` like the REPL loop would:
    print any `SessionLineResult.output` and stop on quit.
    """
    for line in lines:
        res = session.execute_line(line)
        if res.output:
            print(res.output)
        if res.quit:
            break


def build_agent_json_deps(app) -> object:
    """Create `AgentJsonDeps` wired to an app's registry."""
    from agentlib.agent_json import AgentJsonDeps
    from agentlib.tools import turn_support

    return AgentJsonDeps(
        all_known_tools=app.registry.all_known_tools,
        coerce_enabled_tools=app.registry.coerce_enabled_tools,
        merge_tool_param_aliases=turn_support.merge_tool_param_aliases,
    )


def run_main(
    monkeypatch: pytest.MonkeyPatch,
    argv: List[str],
    responses: List[str],
    *,
    route_web: Optional[str] = None,
    route_after_answer: Optional[str] = None,
    stub_search_web: Optional[Callable[..., str]] = None,
    stub_fetch_page: Optional[Callable[..., str]] = None,
    stub_write_file: Optional[Callable[..., str]] = None,
    stub_read_file: Optional[Callable[..., str]] = None,
    stub_run_command: Optional[Callable[..., str]] = None,
    stub_list_directory: Optional[Callable[..., str]] = None,
    stub_tail_file: Optional[Callable[..., str]] = None,
    stub_replace_text: Optional[Callable[..., str]] = None,
    stub_download_file: Optional[Callable[..., str]] = None,
    stub_call_python: Optional[Callable[..., str]] = None,
) -> str:
    """Run app.run(argv) capturing stdout."""
    call_i = {"i": 0}

    def fake_call_ollama_chat(
        messages, primary_profile=None, enabled_tools=None, verbose=0, **kwargs
    ):  # noqa: ARG001
        idx = call_i["i"]
        call_i["i"] += 1
        if idx >= len(responses):
            return json.dumps({"action": "error", "error": f"no ollama response #{idx}"})
        return responses[idx]

    def fake_route_requires_websearch(
        user_query,
        today_str,
        primary_profile=None,
        enabled_tools=None,
        transcript_messages=None,
        **kwargs,
    ):  # noqa: ARG001
        return route_web

    def fake_route_after_answer(
        user_query,
        today_str,
        proposed_answer,
        primary_profile=None,
        enabled_tools=None,
        transcript_messages=None,
        **kwargs,
    ):  # noqa: ARG001
        return route_after_answer

    app = build_test_app(monkeypatch)
    from agentlib.tools import builtins as tool_builtins
    monkeypatch.setattr(app, "call_ollama_chat", fake_call_ollama_chat)
    monkeypatch.setattr(app, "route_requires_websearch", fake_route_requires_websearch)
    monkeypatch.setattr(app, "route_requires_websearch_after_answer", fake_route_after_answer)
    if stub_search_web is not None:
        monkeypatch.setattr(
            tool_builtins,
            "search_web",
            lambda query, params=None, settings=None: stub_search_web(query),
        )
    if stub_fetch_page is not None:
        monkeypatch.setattr(tool_builtins, "fetch_page", lambda url: stub_fetch_page(url))
    if stub_write_file is not None:
        monkeypatch.setattr(tool_builtins, "write_file", lambda path, content: stub_write_file(path, content))
    if stub_read_file is not None:
        monkeypatch.setattr(tool_builtins, "read_file", lambda path: stub_read_file(path))
    if stub_run_command is not None:
        monkeypatch.setattr(tool_builtins, "run_command", lambda command: stub_run_command(command))
    if stub_list_directory is not None:
        monkeypatch.setattr(tool_builtins, "list_directory", lambda path: stub_list_directory(path))
    if stub_tail_file is not None:
        monkeypatch.setattr(tool_builtins, "tail_file", lambda path, lines=20: stub_tail_file(path, lines=lines))
    if stub_replace_text is not None:
        monkeypatch.setattr(
            tool_builtins,
            "replace_text",
            lambda path, pattern, replacement, replace_all=True: stub_replace_text(
                path, pattern, replacement, replace_all=replace_all
            ),
        )
    if stub_download_file is not None:
        monkeypatch.setattr(tool_builtins, "download_file", lambda url, path: stub_download_file(url, path))
    if stub_call_python is not None:
        monkeypatch.setattr(tool_builtins, "call_python", lambda code, globals=None: stub_call_python(code, globals=globals))

    from agentlib.app import main as app_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        app_main(argv, app=app)
    return buf.getvalue().strip()


def j(**kwargs: Any) -> str:
    """Shorthand for model JSON lines."""
    return json.dumps(kwargs, separators=(",", ":"))
