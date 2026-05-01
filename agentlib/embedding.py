from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agentlib.session import AgentSession


def fork_embedded_session(parent_session: "AgentSession", *, app):
    """
    Clone ``parent_session`` message history and session-local settings into a new
    ``AgentSession`` that shares ``app`` (registry, prefs-backed wiring).

    The child's ``session_save_path`` is cleared so forks do not overwrite the parent's
    context save file.
    """
    _, session = build_embedded_session(
        verbose=int(parent_session.verbose),
        primary_profile=parent_session.primary_profile,
        app=app,
    )
    session.messages = copy.deepcopy(parent_session.messages)
    session.enabled_tools = set(parent_session.enabled_tools)
    session.enabled_toolsets = set(parent_session.enabled_toolsets)
    session.second_opinion_on = bool(parent_session.second_opinion_on)
    session.cloud_ai_enabled = bool(parent_session.cloud_ai_enabled)
    session.session_save_path = None
    session.session_system_prompt = parent_session.session_system_prompt
    session.session_system_prompt_path = parent_session.session_system_prompt_path
    session.session_prompt_template = parent_session.session_prompt_template
    ctx = parent_session.context_cfg
    session.context_cfg = copy.deepcopy(ctx) if isinstance(ctx, dict) else {}
    session.template_default = parent_session.template_default
    pt = parent_session.prompt_templates
    session.prompt_templates = copy.deepcopy(pt) if isinstance(pt, dict) else {}
    session.reviewer_hosted_profile = parent_session.reviewer_hosted_profile
    session.reviewer_ollama_model = parent_session.reviewer_ollama_model
    sm = parent_session.skills_map
    session.skills_map = copy.deepcopy(sm) if isinstance(sm, dict) else sm
    session.last_reuse_skill_id = parent_session.last_reuse_skill_id
    session.python_fork_agent = getattr(parent_session, "python_fork_agent", None)
    session.python_delegate_line = getattr(parent_session, "python_delegate_line", None)
    return session


def build_embedded_session(
    *,
    verbose: int = 0,
    primary_profile=None,
    app=None,
    python_fork_agent=None,
    python_delegate_line=None,
):
    """
    Create an `AgentSession` suitable for embedding in other Python programs.

    This is the same wiring the CLI uses for interactive mode, but without the
    stdin loop. Callers can drive the session via:

      session.execute_line("...")              # normal
      session.execute_line("...", emit=emit)   # typed events stream as they occur

    `/` commands (e.g. `/settings ...`) work the same way in embedded mode.

    Optional ``python_fork_agent`` / ``python_delegate_line`` hooks enable ``/call_python``
    helpers ``fork_agent()`` and ``ai(..., agent_name)`` in multi-agent hosts.

    Parameters
    ----------
    primary_profile:
        Override the prefs primary LLM profile for this session only. For local Ollama,
        use ``LlmProfile(backend=\"ollama\", model=\"name:tag\")`` to pin a model.
    app:
        Reuse an existing ``AgentApp`` (same registry/settings); omit on first session.
    """
    import requests

    from agentlib.app import default_app
    from agentlib.context.io import load_context_messages, save_context_bundle
    from agentlib.llm.discovery import fetch_ollama_local_model_names as fetch_ollama_local_model_names_impl
    from agentlib.deliverables import deliverable_skip_mandatory_web, user_wants_written_deliverable
    from agentlib.llm.profile import (
        LlmProfile,
        default_primary_llm_profile,
        effective_ollama_model_from_profile,
    )
    from agentlib.prefs import bootstrap as prefs_bootstrap
    from agentlib import prefs as prefs_mod
    from agentlib.repl.while_cmd import call_while_condition_judge, parse_while_repl_tokens
    from agentlib.session import AgentSession
    from agentlib.skills.loader import load_skills_from_dir
    from agentlib.skills.selection import match_skill_detail

    reuse_app = app is not None
    if not reuse_app:
        app = default_app()
    raw_prefs = app.load_prefs()
    st = app.session_defaults_from_prefs(raw_prefs)

    # Reload plugin toolsets after prefs are applied so tools_dir override works.
    if not reuse_app:
        try:
            app.registry.load_plugin_toolsets(app.resolved_tools_dir(raw_prefs))
            app.registry.register_aliases()
        except Exception:
            pass

    enabled_tools = set(st.get("enabled_tools") or set(app.registry.core_tools))
    enabled_toolsets = set(st.get("enabled_toolsets") or set())
    primary_profile = primary_profile if primary_profile is not None else (
        st.get("primary_profile") or default_primary_llm_profile()
    )

    def describe_llm_profile_short(p: LlmProfile) -> str:
        if p.backend != "hosted":
            return "ollama"
        key = "set" if (p.api_key or "").strip() else "missing"
        return f"hosted {p.model!r} @ {p.base_url!r} (api_key: {key})"

    def format_session_primary_llm_line(p: LlmProfile) -> str:
        if p.backend == "hosted":
            return describe_llm_profile_short(p)
        om = effective_ollama_model_from_profile(p, app.ollama_model())
        return f"ollama ({om!r})"

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
        from agentlib import prompts

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
            ollama_model=effective_ollama_model_from_profile(
                primary_profile or default_primary_llm_profile(), app.ollama_model()
            ),
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
            ollama_model=effective_ollama_model_from_profile(primary_profile, app.ollama_model()),
        )

    skills_map = st.get("skills")
    if not isinstance(skills_map, dict) or not skills_map:
        skills_map = load_skills_from_dir(app.resolved_skills_dir(raw_prefs))

    templates = st.get("prompt_templates")
    if not isinstance(templates, dict) or not templates:
        from agentlib import prompt_templates_io

        templates = prompt_templates_io.load_prompt_templates_from_dir(
            app.resolved_prompt_templates_dir(raw_prefs)
        )

    prompt_template_default = (st.get("prompt_template_default") or "").strip() or "coding"

    session = AgentSession(
        settings=app.settings,
        verbose=int(verbose),
        second_opinion_enabled=bool(st.get("second_opinion_enabled", False)),
        cloud_ai_enabled=bool(st.get("cloud_ai_enabled", False)),
        save_context_path=st.get("save_context_path"),
        enabled_tools=frozenset(enabled_tools),
        enabled_toolsets=frozenset(enabled_toolsets),
        primary_profile=primary_profile,
        reviewer_hosted_profile=st.get("reviewer_hosted_profile"),
        reviewer_ollama_model=st.get("reviewer_ollama_model"),
        skills_map=skills_map,
        prompt_templates=templates,
        prompt_template_default=prompt_template_default,
        prompt_templates_dir=app.resolved_prompt_templates_dir(raw_prefs),
        skills_dir=app.resolved_skills_dir(raw_prefs),
        tools_dir=app.resolved_tools_dir(raw_prefs),
        context_cfg=st.get("context_manager") if isinstance(st.get("context_manager"), dict) else {},
        system_prompt_override=st.get("system_prompt"),
        system_prompt_path=st.get("system_prompt_path"),
        session_prompt_template=None,
        agent_progress=app.agent_progress,
        fetch_ollama_local_model_names=lambda: fetch_ollama_local_model_names_impl(
            app.ollama_base_url(), http_get=requests.get, timeout=60
        ),
        format_last_ollama_usage_for_repl=lambda: "",
        format_session_primary_llm_line=format_session_primary_llm_line,
        format_session_reviewer_line=format_session_reviewer_line,
        print_skill_usage_verbose=app.print_skill_usage_verbose,
        match_skill_detail=lambda u, sm: match_skill_detail(u, sm),
        ml_select_skill_id=lambda user_request, sm, **kw: app.ml_select_skill_id(
            user_request, sm, **kw
        ),
        skill_plan_steps=lambda **kw: app.skill_plan_steps(**kw),
        effective_enabled_tools_for_skill=app.effective_enabled_tools_for_skill,
        effective_enabled_tools_for_turn=app.registry.effective_enabled_tools_for_turn,
        route_requires_websearch=app.route_requires_websearch,
        deliverable_skip_mandatory_web=deliverable_skip_mandatory_web,
        user_wants_written_deliverable=user_wants_written_deliverable,
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
        write_agent_prefs_file=prefs_mod.write_agent_prefs_file,
        agent_prefs_path=prefs_mod.agent_prefs_path,
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
        python_fork_agent=python_fork_agent,
        python_delegate_line=python_delegate_line,
    )
    return app, session

