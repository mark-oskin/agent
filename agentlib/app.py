from __future__ import annotations

"""
Application composition root.

`agent.py` should be a tiny shim. This module owns process-global wiring:
- settings + prefs + CLI overrides
- tool registry (core + plugins)
- ConversationTurnDeps construction/caching
- session construction + REPL/one-shot execution
"""

import datetime
import json
import os
import sys
from dataclasses import dataclass, replace
from typing import AbstractSet, Callable, Optional

import requests

from agentlib import AgentSettings, agent_json, prefs, prompt_templates_io, prompts, routing, routing_followups
from agentlib.coercion import coerce_verbose_level, scalar_to_int, scalar_to_str
from agentlib.context.compaction import maybe_compact_context_window, summarize_conversation_for_context
from agentlib.context.io import load_context_messages, save_context_bundle
from agentlib.deliverables import (
    answer_missing_written_body,
    deliverable_first_answer_followup,
    deliverable_followup_block,
    deliverable_skip_mandatory_web,
    user_wants_written_deliverable,
)
from agentlib.llm import streaming, usage as llm_usage
from agentlib.llm.discovery import fetch_ollama_local_model_names as fetch_ollama_local_model_names_impl
from agentlib.llm.calls import (
    call_hosted_agent_chat,
    call_hosted_chat_plain,
    call_llm_json_content as call_llm_json_content_impl,
    call_ollama_chat as call_ollama_chat_impl,
    call_ollama_plaintext as call_ollama_plaintext_impl,
)
from agentlib.llm.profile import LlmProfile, default_primary_llm_profile
from agentlib.llm.second_opinion import (
    second_opinion_result_user_message,
    second_opinion_reviewer_messages,
)
from agentlib.prefs import bootstrap as prefs_bootstrap
from agentlib.repl.loop import run_interactive_repl_loop
from agentlib.repl.while_cmd import call_while_condition_judge, parse_while_repl_tokens, parse_while_judge_bit
from agentlib.runtime import ConversationTurnDeps, run_agent_conversation_turn
from agentlib.skills.loader import load_skills_from_dir
from agentlib.skills.planner import skill_plan_steps as skill_plan_steps_impl
from agentlib.skills.selection import match_skill_detail, ml_select_skill_id as ml_select_skill_id_impl
from agentlib.tools import builtins as tool_builtins
from agentlib.tools import turn_support
from agentlib.tools.registry import ToolRegistry
from agentlib.tools.websearch import search_backend_banner_line, search_web_effective_max_results


@dataclass
class AgentApp:
    settings: AgentSettings
    registry: ToolRegistry
    project_dir: str

    _cached_turn_deps: Optional[ConversationTurnDeps] = None
    _repl_readline_installed: bool = False

    # --- directories / defaults ---

    def default_prompt_templates_dir(self) -> str:
        return os.path.join(self.project_dir, "prompt_templates")

    def default_skills_dir(self) -> str:
        return os.path.join(self.project_dir, "skills")

    def default_tools_dir(self) -> str:
        return os.path.join(self.project_dir, "tools")

    def resolved_prompt_templates_dir(self, prefs_obj: Optional[dict] = None) -> str:
        if prefs_obj and isinstance(prefs_obj, dict) and (prefs_obj.get("prompt_templates_dir") or "").strip():
            return os.path.abspath(os.path.expanduser(str(prefs_obj["prompt_templates_dir"]).strip()))
        p = self.settings.get_str(("agent", "prompt_templates_dir"), "")
        if p.strip():
            return os.path.abspath(os.path.expanduser(p.strip()))
        return self.default_prompt_templates_dir()

    def resolved_skills_dir(self, prefs_obj: Optional[dict] = None) -> str:
        if prefs_obj and isinstance(prefs_obj, dict) and (prefs_obj.get("skills_dir") or "").strip():
            return os.path.abspath(os.path.expanduser(str(prefs_obj["skills_dir"]).strip()))
        p = self.settings.get_str(("agent", "skills_dir"), "")
        if p.strip():
            return os.path.abspath(os.path.expanduser(p.strip()))
        return self.default_skills_dir()

    def resolved_tools_dir(self, prefs_obj: Optional[dict] = None) -> str:
        if prefs_obj and isinstance(prefs_obj, dict) and (prefs_obj.get("tools_dir") or "").strip():
            return os.path.abspath(os.path.expanduser(str(prefs_obj["tools_dir"]).strip()))
        p = self.settings.get_str(("agent", "tools_dir"), "")
        if p.strip():
            return os.path.abspath(os.path.expanduser(p.strip()))
        return self.default_tools_dir()

    # --- settings helpers ---

    def settings_get_str(self, path: tuple[str, str], default: str = "") -> str:
        return self.settings.get_str(path, default=default)

    def settings_get_bool(self, path: tuple[str, str], default: bool = False) -> bool:
        return self.settings.get_bool(path, default=default)

    def settings_get_int(self, path: tuple[str, str], default: int = 0) -> int:
        return self.settings.get_int(path, default=default)

    def settings_set(self, path: tuple[str, str], value) -> None:
        self.settings.set(path, value)

    # --- LLM profile + request helpers ---

    def ollama_base_url(self) -> str:
        return self.settings_get_str(("ollama", "host"), "http://localhost:11434").rstrip("/")

    def ollama_model(self) -> str:
        return self.settings_get_str(("ollama", "model"), "gemma4:e4b")

    def ollama_second_opinion_model(self) -> str:
        return self.settings_get_str(("ollama", "second_opinion_model"), "llama3.2:latest").strip()

    def openai_api_key(self) -> str:
        return self.settings_get_str(("openai", "api_key"), "")

    def openai_base_url(self) -> str:
        return self.settings_get_str(("openai", "base_url"), "https://api.openai.com/v1").rstrip("/")

    def openai_cloud_model(self) -> str:
        return self.settings_get_str(
            ("openai", "cloud_model"),
            self.settings_get_str(("openai", "model"), "gpt-4o-mini"),
        ).strip()

    def ollama_request_think_value(self) -> object:
        """
        Compute Ollama request `think` value with legacy semantics:
        - If thinking is off, return False even if a stale level is set.
        - If a level is set and thinking is on, return the level string.
        - Special-case: for gpt-oss:* models, thinking=True defaults to "medium".
        """
        thinking_on = bool(self.settings_get_bool(("agent", "thinking"), False))
        if not thinking_on:
            return False
        lvl = self.settings_get_str(("agent", "thinking_level"), "").strip().lower()
        if lvl:
            return lvl
        m = self.settings_get_str(("ollama", "model"), "")
        if m.strip().lower().startswith("gpt-oss:"):
            return "medium"
        return True

    def apply_cli_primary_model(self, name: str, profile: LlmProfile) -> LlmProfile:
        s = (name or "").strip()
        if not s:
            return profile
        if profile.backend == "hosted":
            return replace(profile, model=s)
        self.settings_set(("ollama", "model"), s)
        return profile

    # --- prefs ---

    def load_prefs(self) -> Optional[dict]:
        return prefs.load_agent_prefs()

    def apply_prefs(self, prefs_obj: Optional[dict]) -> None:
        if isinstance(prefs_obj, dict):
            prefs.apply_prefs_to_settings(self.settings, prefs_obj)

    def session_defaults_from_prefs(self, prefs_obj: Optional[dict]) -> dict:
        return prefs_bootstrap.session_defaults_from_prefs(
            prefs_obj,
            migrate_prefs=lambda p: self.apply_prefs(p),
            settings=self.settings,
            core_tools=self.registry.core_tools,
            plugin_toolsets=self.registry.plugin_toolsets,
            normalize_tool_name=self.registry.normalize_tool_name,
            merge_prompt_templates=lambda p: prompt_templates_io.merge_prompt_templates(
                p,
                resolved_prompt_templates_dir=self.resolved_prompt_templates_dir,
                default_prompt_templates_dir=self.default_prompt_templates_dir,
            ),
            load_skills_from_dir=lambda p: load_skills_from_dir(p),
            resolved_prompt_templates_dir=self.resolved_prompt_templates_dir,
            resolved_skills_dir=self.resolved_skills_dir,
            resolved_tools_dir=self.resolved_tools_dir,
            default_prompt_templates_dir=self.default_prompt_templates_dir,
            default_skills_dir=self.default_skills_dir,
            load_plugin_toolsets=lambda tools_dir=None: self.registry.load_plugin_toolsets(tools_dir),
            register_tool_aliases=self.registry.register_aliases,
        )

    # --- JSON parsing helpers for skills ---

    @staticmethod
    def parse_json_with_skill_id(raw: str) -> dict:
        o = agent_json.try_json_loads_object(raw)
        if isinstance(o, dict) and "skill_id" in o:
            return o
        for span in agent_json.iter_balanced_brace_objects(raw or ""):
            o2 = agent_json.try_json_loads_object(span)
            if isinstance(o2, dict) and "skill_id" in o2:
                return o2
        return {}

    @staticmethod
    def parse_workflow_plan_dict(raw: str) -> Optional[dict]:
        def is_plan(o) -> bool:
            if not isinstance(o, dict):
                return False
            st = o.get("steps")
            if not isinstance(st, list) or len(st) < 1:
                return False
            for item in st:
                if not isinstance(item, dict):
                    return False
                if not (item.get("title") or "").strip():
                    return False
            return True

        o = agent_json.try_json_loads_object(raw)
        if is_plan(o):
            return o
        for span in agent_json.iter_balanced_brace_objects(raw or ""):
            o2 = agent_json.try_json_loads_object(span)
            if is_plan(o2):
                return o2
        return None

    # --- progress ---

    def agent_progress_enabled(self) -> bool:
        if self.settings_get_bool(("agent", "quiet"), False):
            return False
        return self.settings_get_bool(("agent", "progress"), True)

    def agent_progress(self, msg: str) -> None:
        if not self.agent_progress_enabled() or not (msg or "").strip():
            return
        print(f"→ {msg.strip()}", file=sys.stderr, flush=True)

    # --- LLM calls (overridable in tests by monkeypatching these methods) ---

    def merge_stream_message_chunks(self, lines_iter, *, stream_chunks: bool = False):
        return streaming.merge_stream_message_chunks(
            lines_iter,
            stream_chunks=stream_chunks,
            agent_stream_thinking_enabled=lambda: bool(
                self.settings_get_bool(("agent", "stream_thinking"), False)
            ),
            ollama_usage_from_chat_response_fn=streaming.ollama_usage_from_chat_response,
        )

    def call_ollama_chat(
        self, messages: list, primary_profile=None, enabled_tools=None, *, verbose: int = 0
    ) -> str:
        prof = primary_profile or default_primary_llm_profile()
        return call_ollama_chat_impl(
            messages,
            primary_profile=prof,
            enabled_tools=enabled_tools,
            verbose=verbose,
            ollama_base_url=self.ollama_base_url(),
            ollama_model=self.ollama_model(),
            ollama_think_value=self.ollama_request_think_value(),
            ollama_debug=bool(self.settings_get_bool(("ollama", "debug"), False)),
            merge_stream_message_chunks=self.merge_stream_message_chunks,
            ollama_usage_from_chat_response=streaming.ollama_usage_from_chat_response,
            message_to_agent_json_text=lambda msg, enabled_tools=None: agent_json.message_to_agent_json_text(
                msg, enabled_tools, self.agent_json_deps()
            ),
            verbose_emit_final_agent_readable=lambda _txt: None,
            format_ollama_usage_line=lambda u: "",
            set_last_ollama_usage=lambda _u: None,
            call_hosted_agent_chat_impl=call_hosted_agent_chat,
        )

    def call_ollama_plaintext(self, messages: list, model: str) -> str:
        return call_ollama_plaintext_impl(
            base_url=self.ollama_base_url(),
            messages=messages,
            model=model,
            think_value=self.ollama_request_think_value(),
            merge_stream_message_chunks=self.merge_stream_message_chunks,
        )

    def call_hosted_chat_plain(self, messages: list, profile: LlmProfile) -> str:
        return call_hosted_chat_plain(
            messages,
            base_url=profile.base_url,
            model=profile.model,
            api_key=profile.api_key,
        )

    def call_openai_chat_plain(self, messages: list) -> str:
        prof = LlmProfile(
            backend="hosted",
            base_url=self.openai_base_url(),
            model=self.openai_cloud_model(),
            api_key=self.openai_api_key(),
        )
        return self.call_hosted_chat_plain(messages, prof)

    def call_llm_json_content(self, messages: list, primary_profile=None, *, verbose: int = 0) -> str:
        prof = primary_profile or default_primary_llm_profile()
        return call_llm_json_content_impl(
            messages,
            primary_profile=prof,
            verbose=verbose,
            ollama_base_url=self.ollama_base_url(),
            ollama_model=self.ollama_model(),
            merge_stream_message_chunks=self.merge_stream_message_chunks,
            ollama_usage_from_chat_response=streaming.ollama_usage_from_chat_response,
            set_last_ollama_usage=lambda _u: None,
        )

    # --- tool/router helpers (overridable in tests) ---

    def route_requires_websearch(
        self,
        user_query: str,
        today_str: str,
        primary_profile,
        enabled_tools: Optional[AbstractSet[str]],
        transcript_messages: Optional[list],
    ) -> Optional[str]:
        return routing.route_requires_websearch(
            user_query,
            today_str,
            primary_profile,
            enabled_tools,
            transcript_messages,
            coerce_enabled_tools=self.registry.coerce_enabled_tools,
            call_ollama_chat=lambda msgs, prof, tools: self.call_ollama_chat(
                msgs, prof, tools, verbose=0
            ),
            parse_agent_json=lambda txt: agent_json.parse_agent_json(txt, self.agent_json_deps()),
            scalar_to_str=scalar_to_str,
            router_transcript_max_messages=self.settings_get_int(("agent", "router_transcript_max_messages"), 80),
        )

    def route_requires_websearch_after_answer(
        self,
        user_query: str,
        today_str: str,
        proposed_answer: str,
        primary_profile,
        enabled_tools: Optional[AbstractSet[str]],
        transcript_messages: Optional[list],
    ) -> Optional[str]:
        return routing.route_requires_websearch_after_answer(
            user_query,
            today_str,
            proposed_answer,
            primary_profile,
            enabled_tools,
            transcript_messages,
            coerce_enabled_tools=self.registry.coerce_enabled_tools,
            call_ollama_chat=lambda msgs, prof, tools: self.call_ollama_chat(
                msgs, prof, tools, verbose=0
            ),
            parse_agent_json=lambda txt: agent_json.parse_agent_json(txt, self.agent_json_deps()),
            scalar_to_str=scalar_to_str,
            router_transcript_max_messages=self.settings_get_int(("agent", "router_transcript_max_messages"), 80),
        )

    # --- agent_json deps ---

    def agent_json_deps(self) -> agent_json.AgentJsonDeps:
        return agent_json.AgentJsonDeps(
            all_known_tools=self.registry.all_known_tools,
            coerce_enabled_tools=self.registry.coerce_enabled_tools,
            merge_tool_param_aliases=turn_support.merge_tool_param_aliases,
        )

    # --- skill wrappers ---

    def print_skill_usage_verbose(
        self,
        verbose: int,
        *,
        source: str,
        skill_id: Optional[str],
        base_tools: AbstractSet[str],
        effective_tools: AbstractSet[str],
        detail: Optional[str] = None,
    ) -> None:
        if verbose < 1:
            return
        sk = skill_id if skill_id else "(none)"
        et = sorted(effective_tools)
        bt = sorted(base_tools)
        if set(bt) != set(et):
            print(f"[*] [skills:{source}] id={sk!r} tools={et} (narrowed from session {bt})")
        else:
            print(f"[*] [skills:{source}] id={sk!r} tools={et}")
        if detail:
            print(f"[*] [skills:{source}] {detail}")

    def effective_enabled_tools_for_skill(
        self, base_enabled: AbstractSet[str], skills_map: dict, skill_id: Optional[str]
    ) -> AbstractSet[str]:
        if not skill_id or not isinstance(skills_map, dict):
            return base_enabled
        rec = skills_map.get(skill_id)
        if not isinstance(rec, dict):
            return base_enabled
        raw = rec.get("tools")
        if not isinstance(raw, list) or not raw:
            return base_enabled
        wanted = {
            str(t).strip()
            for t in raw
            if isinstance(t, str) and t.strip() in self.registry.all_known_tools()
        }
        if not wanted:
            return base_enabled
        narrowed = wanted & set(base_enabled)
        return frozenset(narrowed) if narrowed else base_enabled

    def ml_select_skill_id(self, user_request: str, skills_map: dict, *, primary_profile, verbose: int):
        return ml_select_skill_id_impl(
            user_request,
            skills_map,
            primary_profile=primary_profile,
            verbose=verbose,
            call_llm_json_content=lambda msgs, prof, verbose=0: self.call_llm_json_content(
                msgs, prof, verbose=verbose
            ),
            agent_progress=self.agent_progress,
            try_json_loads_object=agent_json.try_json_loads_object,
            parse_json_with_skill_id=self.parse_json_with_skill_id,
        )

    def skill_plan_steps(
        self,
        *,
        user_request: str,
        today_str: str,
        skill_id: str,
        skills_map: dict,
        primary_profile,
        verbose: int,
        enabled_tools: Optional[AbstractSet[str]] = None,
        _enabled_tools: Optional[AbstractSet[str]] = None,
        system_prompt_override: Optional[str] = None,
        _system_prompt_override: Optional[str] = None,
    ):
        et = enabled_tools if enabled_tools is not None else _enabled_tools
        spo = system_prompt_override if system_prompt_override is not None else _system_prompt_override
        return skill_plan_steps_impl(
            user_request=user_request,
            today_str=today_str,
            skill_id=skill_id,
            skills_map=skills_map,
            primary_profile=primary_profile,
            enabled_tools=et if et is not None else frozenset(),
            verbose=verbose,
            system_prompt_override=spo,
            agent_progress=self.agent_progress,
            call_llm_json_content=lambda msgs, prof, verbose=0: self.call_llm_json_content(
                msgs, prof, verbose=verbose
            ),
            try_json_loads_object=agent_json.try_json_loads_object,
            parse_workflow_plan_dict=self.parse_workflow_plan_dict,
            scalar_to_int=scalar_to_int,
            scalar_to_str=scalar_to_str,
        )

    # --- ConversationTurnDeps ---

    def conversation_turn_deps(self) -> ConversationTurnDeps:
        if self._cached_turn_deps is not None:
            return self._cached_turn_deps

        def _maybe_compact(messages: list, *, user_query: str, primary_profile, verbose: int, context_cfg=None) -> list:
            return maybe_compact_context_window(
                messages,
                user_query=user_query,
                primary_profile=primary_profile,
                verbose=verbose,
                context_cfg=context_cfg,
                settings_get_bool=self.settings_get_bool,
                settings_get_int=self.settings_get_int,
                call_hosted_chat_plain=self.call_hosted_chat_plain,
                call_ollama_plaintext=lambda msgs, model: self.call_ollama_plaintext(msgs, model),
                ollama_model=self.ollama_model(),
                summarize_conversation_fn=lambda **kwargs: summarize_conversation_for_context(
                    **kwargs,
                    call_hosted_chat_plain=self.call_hosted_chat_plain,
                    call_ollama_plaintext=lambda msgs, model: self.call_ollama_plaintext(msgs, model),
                    ollama_model=self.ollama_model(),
                ),
            )

        deps = ConversationTurnDeps(
            coerce_enabled_tools=self.registry.coerce_enabled_tools,
            maybe_compact_context_window=_maybe_compact,
            call_ollama_chat=lambda msgs, prof, tools, verbose=0: self.call_ollama_chat(
                msgs, prof, tools, verbose=verbose
            ),
            parse_agent_json=lambda t: agent_json.parse_agent_json(t, self.agent_json_deps()),
            deliverable_followup_block=lambda p: deliverable_followup_block(p, scalar_to_str),
            answer_missing_written_body=answer_missing_written_body,
            scalar_to_str=lambda x, default="": scalar_to_str(x, default),
            hosted_review_ready=lambda cloud, reviewer: bool(cloud and self.openai_api_key())
            or (
                reviewer is not None
                and getattr(reviewer, "backend", "") == "hosted"
                and (getattr(reviewer, "api_key", "") or "").strip()
            ),
            second_opinion_reviewer_messages=second_opinion_reviewer_messages,
            second_opinion_result_user_message=second_opinion_result_user_message,
            call_ollama_plaintext=lambda msgs, model: self.call_ollama_plaintext(msgs, model),
            call_hosted_chat_plain=self.call_hosted_chat_plain,
            call_openai_chat_plain=self.call_openai_chat_plain,
            ollama_second_opinion_model=self.ollama_second_opinion_model,
            route_requires_websearch_after_answer=(
                lambda uq, today, ans, prof, tools, transcript_messages=None, **_kw: self.route_requires_websearch_after_answer(
                    uq, today, ans, prof, tools, transcript_messages
                )
            ),
            deliverable_skip_mandatory_web=deliverable_skip_mandatory_web,
            deliverable_first_answer_followup=deliverable_first_answer_followup,
            is_self_capability_question=routing_followups.is_self_capability_question,
            self_capability_followup=routing_followups.self_capability_followup,
            tool_need_review_followup=routing_followups.tool_need_review_followup,
            extract_json_object_from_text=lambda t: agent_json.extract_json_object_from_text(
                t, self.agent_json_deps()
            ),
            all_known_tools=self.registry.all_known_tools,
            merge_tool_param_aliases=turn_support.merge_tool_param_aliases,
            ensure_tool_defaults=turn_support.ensure_tool_defaults,
            tool_params_fingerprint=lambda tool, params: turn_support.tool_params_fingerprint(
                tool,
                params,
                search_web_effective_max_results=lambda p: search_web_effective_max_results(
                    p, settings=self.settings
                ),
            ),
            search_backend_banner_line=lambda: search_backend_banner_line(self.settings),
            search_web=lambda query, params=None: tool_builtins.search_web(
                query, params=params, settings=self.settings
            ),
            fetch_page=tool_builtins.fetch_page,
            run_command=tool_builtins.run_command,
            use_git=tool_builtins.use_git,
            write_file=tool_builtins.write_file,
            list_directory=tool_builtins.list_directory,
            read_file=tool_builtins.read_file,
            download_file=tool_builtins.download_file,
            tail_file=tool_builtins.tail_file,
            replace_text=tool_builtins.replace_text,
            call_python=tool_builtins.call_python,
            plugin_tool_handlers=self.registry.plugin_tool_handlers,
            tool_fault_result=lambda tool, exc: f"Tool fault: {tool} raised {type(exc).__name__}: {exc}",
            tool_recovery_may_run=lambda interactive: (bool(interactive) and sys.stdin.isatty())
            or bool(self.settings_get_bool(("agent", "auto_confirm_tool_retry"), False)),
            tool_recovery_tools=turn_support.TOOL_RECOVERY_TOOLS,
            tool_result_indicates_retryable_failure=turn_support.tool_result_indicates_retryable_failure,
            suggest_tool_recovery_params=lambda tool, params, result, user_query, primary_profile, et, verbose: turn_support.suggest_tool_recovery_params(
                tool,
                params,
                result,
                user_query,
                primary_profile,
                et,
                verbose,
                call_ollama_chat=lambda msgs, prof, tools, verbose=0: self.call_ollama_chat(
                    msgs, prof, tools, verbose=verbose
                ),
                merge_aliases=turn_support.merge_tool_param_aliases,
                ensure_defaults=turn_support.ensure_tool_defaults,
            ),
            confirm_tool_recovery_retry=lambda tool, old_params, new_params, rationale, interactive_tool_recovery=False, **_kw: turn_support.confirm_tool_recovery_retry(
                tool,
                old_params,
                new_params,
                rationale,
                interactive_tool_recovery=bool(interactive_tool_recovery),
                stdin_isatty=sys.stdin.isatty(),
            ),
            agent_progress=self.agent_progress,
            tool_progress_message=lambda tool, params: _tool_progress_message(tool, params),
            is_tool_result_weak_for_dedup=turn_support.is_tool_result_weak_for_dedup,
            tool_result_user_message=lambda tool, params, result, deliverable_reminder="": turn_support.tool_result_user_message(
                tool,
                params,
                result,
                deliverable_reminder=deliverable_reminder,
                tool_output_max=self.settings_get_int(("ollama", "tool_output_max"), 14000),
            ),
        )
        self._cached_turn_deps = deps
        return deps

    # --- CLI helpers ---

    @staticmethod
    def strip_leading_dashes_flag(a: str) -> str:
        x = (a or "").lower().replace("_", "-")
        while x.startswith("-"):
            x = x[1:]
        return x

    @staticmethod
    def print_cli_help() -> None:
        print(
            "Usage:\n"
            "  agent [options] [question...]\n"
            "  With no question and no action flags, start the interactive REPL.\n"
            "  With a question, run a single non-interactive turn.\n"
        )

    @staticmethod
    def verbose_ack_message(level: int) -> str:
        labels = ("off", "tool invocations", "tool invocations and streamed model JSON")
        return f"verbose level {level} ({labels[level]}) for this session."

    # --- REPL I/O ---

    def repl_history_path(self) -> str:
        override = self.settings_get_str(("agent", "repl_history"), "")
        if override:
            return os.path.expanduser(override)
        return os.path.join(os.path.expanduser("~"), ".agent_repl_history")

    def repl_input_max_bytes(self) -> int:
        v = self.settings_get_int(("agent", "repl_input_max_bytes"), 0)
        return v if v > 0 else 131072

    def flush_repl_readline_history(self) -> None:
        if not self._repl_readline_installed:
            return
        path = self.repl_history_path()
        try:
            import readline  # type: ignore[import-not-found]

            readline.write_history_file(path)
        except Exception:
            pass

    def interactive_repl_install_readline(self) -> None:
        if self._repl_readline_installed:
            return
        path = self.repl_history_path()
        try:
            import readline  # type: ignore[import-not-found]

            try:
                readline.read_history_file(path)
            except FileNotFoundError:
                pass
            self._repl_readline_installed = True
        except Exception:
            self._repl_readline_installed = False

    def repl_read_line(self, prompt: str) -> str:
        if not sys.stdin.isatty():
            return input(prompt)
        if not self.settings_get_bool(("agent", "repl_buffered_line"), False):
            return input(prompt)
        maxb = self.repl_input_max_bytes()
        print(prompt, end="", flush=True)
        try:
            raw = sys.stdin.buffer.readline(maxb + 1)
        except (OSError, ValueError):
            return ""
        if len(raw) > maxb:
            print(f"\n[Input truncated to {maxb} bytes]", file=sys.stderr)
            raw = raw[:maxb]
        text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if self._repl_readline_installed and text.strip():
            try:
                import readline  # type: ignore[import-not-found]

                readline.add_history(text)
            except Exception:
                pass
        return text

    # --- run modes ---

    def run_interactive(
        self,
        *,
        st: dict,
        verbose: int,
        second_opinion: bool,
        cloud_ai: bool,
        save_context_path: Optional[str],
        enabled_tools: AbstractSet[str],
        prompt_template_selected: Optional[str],
    ) -> None:
        _interactive_repl(
            app=self,
            verbose=verbose,
            second_opinion_enabled=second_opinion,
            cloud_ai_enabled=cloud_ai,
            save_context_path=save_context_path,
            enabled_tools=enabled_tools,
            enabled_toolsets=st.get("enabled_toolsets"),
            primary_profile=st.get("primary_profile"),
            reviewer_hosted_profile=st.get("reviewer_hosted_profile"),
            reviewer_ollama_model=st.get("reviewer_ollama_model"),
            prefs_loaded=False,
            system_prompt_override=st.get("system_prompt"),
            system_prompt_path=st.get("system_prompt_path"),
            prompt_templates=st.get("prompt_templates"),
            prompt_template_default=st.get("prompt_template_default"),
            prompt_templates_dir=st.get("prompt_templates_dir"),
            skills_dir=st.get("skills_dir"),
            tools_dir=st.get("tools_dir"),
            skills_map=st.get("skills"),
            context_cfg=st.get("context_manager"),
            prompt_template_selected=prompt_template_selected,
        )

    def run(self, argv: Optional[list[str]] = None) -> None:
        from agentlib.cli import parse_and_apply_cli_config_flag, parse_main_argv

        argv0 = list(sys.argv[1:] if argv is None else argv)
        argv1 = parse_and_apply_cli_config_flag(argv0)
        raw_prefs = self.load_prefs()
        st = self.session_defaults_from_prefs(raw_prefs)

        try:
            self.registry.load_plugin_toolsets(self.resolved_tools_dir(raw_prefs))
            self.registry.register_aliases()
        except Exception:
            pass

        verbose0 = coerce_verbose_level(st.get("verbose", 0))
        second_opinion0 = bool(st["second_opinion_enabled"])
        cloud0 = bool(st["cloud_ai_enabled"])
        enabled_tools0 = set(st["enabled_tools"])
        save_context_path0: Optional[str] = st["save_context_path"]
        primary_profile0 = st["primary_profile"]
        reviewer_hosted_profile: Optional[LlmProfile] = st["reviewer_hosted_profile"]
        reviewer_ollama_model: Optional[str] = st["reviewer_ollama_model"]

        parsed = parse_main_argv(
            argv1,
            verbose=verbose0,
            second_opinion_enabled=second_opinion0,
            cloud_ai_enabled=cloud0,
            save_context_path=save_context_path0,
            enabled_tools=enabled_tools0,
            primary_profile=primary_profile0,
            reviewer_hosted_profile=reviewer_hosted_profile,
            reviewer_ollama_model=reviewer_ollama_model,
            strip_leading_dashes_flag=self.strip_leading_dashes_flag,
            print_cli_help=self.print_cli_help,
            apply_cli_primary_model=self.apply_cli_primary_model,
            normalize_tool_name=self.registry.normalize_tool_name,
            format_unknown_tool_hint=self.registry.format_unknown_tool_hint,
            format_settings_tools_list=self.registry.format_settings_tools_list,
        )
        if parsed.help_requested:
            return

        verbose = parsed.verbose
        second_opinion_enabled = parsed.second_opinion_enabled
        cloud_ai_enabled = parsed.cloud_ai_enabled
        prompt_template_selected = parsed.prompt_template_selected
        load_context_path = parsed.load_context_path
        save_context_path = parsed.save_context_path
        enabled_tools = set(parsed.enabled_tools)
        query_parts = list(parsed.query_parts)

        if not sys.stdout.isatty() and not parsed.verbose_flag_set:
            verbose = 0
        if not sys.stdout.isatty() and not parsed.second_opinion_flag_set:
            second_opinion_enabled = False
        if not sys.stdout.isatty() and not parsed.cloud_ai_flag_set:
            cloud_ai_enabled = False

        if not query_parts:
            if load_context_path:
                print("Error: --load_context requires a follow-up question on the command line.")
                return
            self.run_interactive(
                st=st,
                verbose=verbose,
                second_opinion=second_opinion_enabled,
                cloud_ai=cloud_ai_enabled,
                save_context_path=save_context_path,
                enabled_tools=frozenset(enabled_tools),
                prompt_template_selected=prompt_template_selected,
            )
            return

        # One-shot: keep legacy behavior for now by calling the shared runtime directly.
        user_query = " ".join(query_parts)
        today_str = datetime.date.today().strftime("%Y-%m-%d (%A)")
        messages = [{"role": "user", "content": user_query}]
        answered, final_answer = run_agent_conversation_turn(
            messages,
            user_query,
            today_str,
            self.conversation_turn_deps(),
            web_required=False,
            deliverable_wanted=user_wants_written_deliverable(user_query),
            verbose=verbose,
            second_opinion_enabled=second_opinion_enabled,
            cloud_ai_enabled=cloud_ai_enabled,
            primary_profile=parsed.primary_profile,
            reviewer_hosted_profile=reviewer_hosted_profile,
            reviewer_ollama_model=reviewer_ollama_model,
            enabled_tools=frozenset(enabled_tools),
            interactive_tool_recovery=bool(sys.stdin.isatty() and sys.stdout.isatty()),
            context_cfg=st.get("context_manager"),
            print_answer=False,
        )
        _ = answered
        if final_answer:
            print(final_answer)
        if save_context_path:
            try:
                save_context_bundle(save_context_path, messages, user_query, final_answer, answered)
            except OSError:
                pass


def default_app() -> AgentApp:
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    settings = AgentSettings.defaults()
    registry = ToolRegistry(default_tools_dir=os.path.join(project_dir, "tools"))
    registry.load_plugin_toolsets(registry.default_tools_dir)
    registry.register_aliases()
    return AgentApp(settings=settings, registry=registry, project_dir=project_dir)


def _runner_instruction_bits(
    app: "AgentApp",
    second_opinion_enabled: bool,
    cloud_ai_enabled: bool,
    *,
    primary_profile,
    reviewer_ollama_model: Optional[str],
    reviewer_hosted_profile: Optional[LlmProfile],
    enabled_tools: AbstractSet[str],
    tool_policy_runner_text: Callable[[Optional[AbstractSet[str]]], str],
) -> str:
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

    return prompts.runner_instruction_bits(
        second_opinion=second_opinion_enabled,
        cloud=cloud_ai_enabled,
        primary_profile=primary_profile,
        reviewer_hosted_profile=reviewer_hosted_profile,
        reviewer_ollama_model=reviewer_ollama_model,
        enabled_tools=enabled_tools,
        ollama_model=app.ollama_model(),
        hosted_review_ready=hosted_review_ready,
        tool_policy_runner_text=tool_policy_runner_text,
    )


def main(argv: Optional[list[str]] = None, *, app: Optional["AgentApp"] = None) -> None:
    """
    Stable CLI entrypoint used by `agent.py` and tests.

    The goal is that callers/tests never need `agent.py` internals.
    """
    from agentlib.cli import parse_and_apply_cli_config_flag, parse_main_argv

    app0 = app or _APP
    argv0 = list(sys.argv[1:] if argv is None else argv)
    argv1 = parse_and_apply_cli_config_flag(argv0)
    raw_prefs = app0.load_prefs()
    st = app0.session_defaults_from_prefs(raw_prefs)

    # Reload plugin toolsets after prefs are applied so tools_dir override works.
    try:
        app0.registry.load_plugin_toolsets(app0.resolved_tools_dir(raw_prefs))
        app0.registry.register_aliases()
    except Exception:
        pass

    verbose0 = coerce_verbose_level(st.get("verbose", 0))
    second_opinion0 = bool(st["second_opinion_enabled"])
    cloud0 = bool(st["cloud_ai_enabled"])
    enabled_tools0 = set(st["enabled_tools"])
    save_context_path0: Optional[str] = st["save_context_path"]
    primary_profile0 = st["primary_profile"]
    reviewer_hosted_profile: Optional[LlmProfile] = st["reviewer_hosted_profile"]
    reviewer_ollama_model: Optional[str] = st["reviewer_ollama_model"]

    parsed = parse_main_argv(
        argv1,
        verbose=verbose0,
        second_opinion_enabled=second_opinion0,
        cloud_ai_enabled=cloud0,
        save_context_path=save_context_path0,
        enabled_tools=enabled_tools0,
        primary_profile=primary_profile0,
        reviewer_hosted_profile=reviewer_hosted_profile,
        reviewer_ollama_model=reviewer_ollama_model,
        strip_leading_dashes_flag=app0.strip_leading_dashes_flag,
        print_cli_help=app0.print_cli_help,
        apply_cli_primary_model=app0.apply_cli_primary_model,
        normalize_tool_name=app0.registry.normalize_tool_name,
        format_unknown_tool_hint=app0.registry.format_unknown_tool_hint,
        format_settings_tools_list=app0.registry.format_settings_tools_list,
    )
    if parsed.help_requested:
        return

    verbose = parsed.verbose
    verbose_flag_set = parsed.verbose_flag_set
    second_opinion_enabled = parsed.second_opinion_enabled
    second_opinion_flag_set = parsed.second_opinion_flag_set
    cloud_ai_enabled = parsed.cloud_ai_enabled
    cloud_ai_flag_set = parsed.cloud_ai_flag_set
    load_context_path = parsed.load_context_path
    save_context_path = parsed.save_context_path
    enabled_tools = set(parsed.enabled_tools)
    primary_profile = parsed.primary_profile
    prompt_template_selected = parsed.prompt_template_selected
    query_parts = list(parsed.query_parts)

    # One-shot scripting mode: when stdout is redirected, default to quiet unless explicitly overridden.
    if not sys.stdout.isatty() and not verbose_flag_set:
        verbose = 0
    if not sys.stdout.isatty() and not second_opinion_flag_set:
        second_opinion_enabled = False
    if not sys.stdout.isatty() and not cloud_ai_flag_set:
        cloud_ai_enabled = False

    if not query_parts:
        if load_context_path:
            print("Error: --load_context requires a follow-up question on the command line.")
            return
        _interactive_repl(
            app=app0,
            verbose=verbose,
            second_opinion_enabled=second_opinion_enabled,
            cloud_ai_enabled=cloud_ai_enabled,
            save_context_path=save_context_path,
            enabled_tools=frozenset(enabled_tools),
            enabled_toolsets=st.get("enabled_toolsets"),
            primary_profile=primary_profile,
            reviewer_hosted_profile=reviewer_hosted_profile,
            reviewer_ollama_model=reviewer_ollama_model,
            prefs_loaded=raw_prefs is not None,
            system_prompt_override=st.get("system_prompt"),
            system_prompt_path=st.get("system_prompt_path"),
            prompt_templates=st.get("prompt_templates"),
            prompt_template_default=st.get("prompt_template_default"),
            prompt_templates_dir=st.get("prompt_templates_dir"),
            skills_dir=st.get("skills_dir"),
            tools_dir=st.get("tools_dir"),
            skills_map=st.get("skills"),
            context_cfg=st.get("context_manager"),
            prompt_template_selected=prompt_template_selected,
        )
        return

    # One-shot turn
    user_query = " ".join(query_parts)
    today_str = datetime.date.today().strftime("%Y-%m-%d (%A)")
    deliverable_wanted = user_wants_written_deliverable(user_query)
    sys_prompt_override = st.get("system_prompt")

    prompt_templates = (
        st.get("prompt_templates")
        if isinstance(st.get("prompt_templates"), dict)
        else prompt_templates_io.load_prompt_templates_from_dir(app0.default_prompt_templates_dir())
    )
    prompt_template_default = (st.get("prompt_template_default") or "coding").strip()
    if sys_prompt_override is None:
        chosen = (prompt_template_selected or prompt_template_default or "").strip()
        if chosen:
            resolved = prompts.resolve_prompt_template_text(chosen, prompt_templates)
            if resolved:
                sys_prompt_override = resolved
            else:
                print(
                    f"Error: unknown or invalid prompt template {chosen!r}.",
                    file=sys.stderr,
                )
                return

    si0 = prompts.effective_system_instruction_text(sys_prompt_override)
    first_user = (
        f"{si0}\n\n"
        f"Today's date (system clock): {today_str}\n\n"
        f"User request: {user_query}\n\n"
        "Respond with JSON only. No other text."
    )
    ri = _runner_instruction_bits(
        app0,
        second_opinion_enabled,
        cloud_ai_enabled,
        primary_profile=primary_profile,
        reviewer_ollama_model=reviewer_ollama_model,
        reviewer_hosted_profile=reviewer_hosted_profile,
        enabled_tools=frozenset(enabled_tools),
        tool_policy_runner_text=app0.registry.tool_policy_runner_text,
    )
    if ri:
        first_user += "\n\n" + ri

    if load_context_path:
        try:
            messages = load_context_messages(load_context_path)
        except Exception as e:
            print(f"Error loading context: {e}")
            return
        cont = (
            f"Today's date (system clock): {today_str}\n\n"
            f"New user request:\n{user_query}\n\n"
            "Continue the conversation. Respond with JSON only. No other text."
        )
        if ri:
            cont += "\n\n" + ri
        messages.append({"role": "user", "content": cont})
    else:
        messages = [{"role": "user", "content": first_user}]

    router_query = app0.route_requires_websearch(
        user_query, today_str, primary_profile, frozenset(enabled_tools), transcript_messages=messages
    )
    if deliverable_skip_mandatory_web(user_query):
        router_query = None
    web_required = bool(router_query)
    if router_query and "search_web" in enabled_tools:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Before answering, you MUST call the tool search_web.\n"
                    "Respond with JSON only in tool_call form.\n"
                    f'Suggested query: "{router_query}"'
                ),
            }
        )

    answered, final_answer = run_agent_conversation_turn(
        messages,
        user_query,
        today_str,
        app0.conversation_turn_deps(),
        web_required=web_required,
        deliverable_wanted=deliverable_wanted,
        verbose=verbose,
        second_opinion_enabled=second_opinion_enabled,
        cloud_ai_enabled=cloud_ai_enabled,
        primary_profile=primary_profile,
        reviewer_hosted_profile=reviewer_hosted_profile,
        reviewer_ollama_model=reviewer_ollama_model,
        enabled_tools=frozenset(enabled_tools),
        interactive_tool_recovery=bool(sys.stdin.isatty() and sys.stdout.isatty()),
        context_cfg=st.get("context_manager"),
        print_answer=False,
    )
    if final_answer:
        print(final_answer)

    if save_context_path:
        try:
            save_context_bundle(save_context_path, messages, user_query, final_answer, answered)
        except OSError as e:
            print(f"Warning: could not save context: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Back-compat test surface (was historically in agent.py)
# ---------------------------------------------------------------------------
#
# The project is in the middle of migrating callers/tests off of `agent.py` internals.
# Keep this lightweight, stable module surface so tests can import `agentlib.app`
# and avoid touching `agent.py`.
#

_APP = default_app()

# Common “defaults” paths (assets live at repo root, not inside the package dir).
_PROJECT_DIR = _APP.project_dir


def _interactive_repl(
    *,
    app: Optional["AgentApp"] = None,
    verbose: int,
    second_opinion_enabled: bool,
    cloud_ai_enabled: bool,
    save_context_path: Optional[str],
    enabled_tools: Optional[AbstractSet[str]] = None,
    enabled_toolsets: Optional[AbstractSet[str]] = None,
    primary_profile: Optional[LlmProfile] = None,
    reviewer_hosted_profile: Optional[LlmProfile] = None,
    reviewer_ollama_model: Optional[str] = None,
    prefs_loaded: bool = False,
    system_prompt_override: Optional[str] = None,
    system_prompt_path: Optional[str] = None,
    prompt_templates: Optional[dict] = None,
    prompt_template_default: Optional[str] = None,
    prompt_templates_dir: Optional[str] = None,
    skills_dir: Optional[str] = None,
    tools_dir: Optional[str] = None,
    skills_map: Optional[dict] = None,
    context_cfg: Optional[dict] = None,
    prompt_template_selected: Optional[str] = None,
) -> None:
    """Multi-turn stdin loop when no query is given on the command line (legacy surface)."""
    _ = prefs_loaded
    a = app or _APP
    second_opinion_on = second_opinion_enabled
    session_save_path = save_context_path
    ptd0 = (prompt_templates_dir or "").strip()
    session_pt_dir = os.path.abspath(os.path.expanduser(ptd0) if ptd0 else a.default_prompt_templates_dir())
    skd0 = (skills_dir or "").strip()
    session_skills_dir = os.path.abspath(os.path.expanduser(skd0) if skd0 else a.default_skills_dir())
    tld0 = (tools_dir or "").strip()
    session_tools_dir = os.path.abspath(os.path.expanduser(tld0) if tld0 else a.default_tools_dir())
    skills_m = skills_map if isinstance(skills_map, dict) else {}
    templates = (
        prompt_templates
        if isinstance(prompt_templates, dict)
        else prompt_templates_io.load_prompt_templates_from_dir(a.default_prompt_templates_dir())
    )
    template_default = (prompt_template_default or "").strip() or "coding"
    session_prompt_template: Optional[str] = None
    session_system_prompt = system_prompt_override
    session_system_prompt_path = (
        os.path.abspath(os.path.expanduser(system_prompt_path)) if (system_prompt_path or "").strip() else None
    )
    if session_system_prompt is None and not session_system_prompt_path:
        resolved = prompts.resolve_prompt_template_text(template_default, templates)
        if resolved:
            session_system_prompt = resolved
            session_prompt_template = template_default
    if prompt_template_selected:
        session_prompt_template = prompt_template_selected or session_prompt_template

    context_cfg_use = context_cfg if isinstance(context_cfg, dict) else {}
    primary_profile_use = primary_profile or default_primary_llm_profile()
    enabled_tools_use = set(enabled_tools) if enabled_tools is not None else set(a.registry.core_tools)
    enabled_toolsets_use = set(enabled_toolsets) if enabled_toolsets is not None else set()

    from agentlib.session import AgentSession

    def describe_llm_profile_short(p) -> str:
        if p.backend != "hosted":
            return "ollama"
        key = "set" if (p.api_key or "").strip() else "missing"
        return f"hosted {p.model!r} @ {p.base_url!r} (api_key: {key})"

    def format_session_primary_llm_line(p) -> str:
        if p.backend == "hosted":
            return describe_llm_profile_short(p)
        return f"ollama ({a.ollama_model()!r})"

    def format_session_reviewer_line(hosted, ollama_model) -> str:
        if hosted is not None and hosted.backend == "hosted":
            return describe_llm_profile_short(hosted)
        return f"ollama ({(ollama_model or a.ollama_second_opinion_model())!r})"

    def fetch_ollama_local_model_names() -> list[str]:
        return fetch_ollama_local_model_names_impl(a.ollama_base_url(), http_get=requests.get, timeout=60)

    def format_last_ollama_usage_for_repl() -> str:
        return llm_usage.format_last_ollama_usage_for_repl(None)

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
        def hosted_review_ready(cloud_b: bool, reviewer) -> bool:
            if cloud_b and a.settings.get_str(("openai", "api_key"), "").strip():
                return True
            if (
                reviewer is not None
                and getattr(reviewer, "backend", "") == "hosted"
                and (getattr(reviewer, "api_key", "") or "").strip()
            ):
                return True
            return False

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
            ollama_model=a.ollama_model(),
            hosted_review_ready=hosted_review_ready,
            tool_policy_runner_text=a.registry.tool_policy_runner_text,
        )

    def call_while_judge(condition: str, messages: list, *, primary_profile, verbose: int) -> int:
        return call_while_condition_judge(
            condition,
            messages,
            primary_profile=primary_profile,
            verbose=verbose,
            default_primary_llm_profile=default_primary_llm_profile,
            call_hosted_chat_plain=a.call_hosted_chat_plain,
            call_ollama_plaintext=a.call_ollama_plaintext,
            ollama_model=a.ollama_model(),
            scalar_to_str_fn=scalar_to_str,
        )

    def route_req(uq, today, prof, tools, transcript_messages=None, **_k):
        return a.route_requires_websearch(uq, today, prof, tools, transcript_messages)

    session = AgentSession(
        settings=a.settings,
        verbose=verbose,
        second_opinion_enabled=second_opinion_on,
        cloud_ai_enabled=cloud_ai_enabled,
        save_context_path=session_save_path,
        enabled_tools=frozenset(enabled_tools_use),
        enabled_toolsets=frozenset(enabled_toolsets_use),
        primary_profile=primary_profile_use,
        reviewer_hosted_profile=reviewer_hosted_profile,
        reviewer_ollama_model=reviewer_ollama_model,
        skills_map=skills_m,
        prompt_templates=templates,
        prompt_template_default=template_default,
        prompt_templates_dir=session_pt_dir,
        skills_dir=session_skills_dir,
        tools_dir=session_tools_dir,
        context_cfg=context_cfg_use,
        system_prompt_override=session_system_prompt,
        system_prompt_path=session_system_prompt_path,
        session_prompt_template=session_prompt_template,
        agent_progress=a.agent_progress,
        fetch_ollama_local_model_names=fetch_ollama_local_model_names,
        format_last_ollama_usage_for_repl=format_last_ollama_usage_for_repl,
        format_session_primary_llm_line=format_session_primary_llm_line,
        format_session_reviewer_line=format_session_reviewer_line,
        print_skill_usage_verbose=a.print_skill_usage_verbose,
        match_skill_detail=match_skill_detail,
        ml_select_skill_id=lambda user_request, skills_map, **kw: a.ml_select_skill_id(
            user_request, skills_map, **kw
        ),
        skill_plan_steps=lambda **kw: a.skill_plan_steps(**kw),
        effective_enabled_tools_for_skill=a.effective_enabled_tools_for_skill,
        effective_enabled_tools_for_turn=a.registry.effective_enabled_tools_for_turn,
        route_requires_websearch=route_req,
        deliverable_skip_mandatory_web=deliverable_skip_mandatory_web,
        user_wants_written_deliverable=user_wants_written_deliverable,
        interactive_turn_user_message=interactive_turn_user_message,
        conversation_turn_deps=(_conversation_turn_deps() if app is None or app is _APP else a.conversation_turn_deps()),
        save_context_bundle=save_context_bundle,
        load_context_messages=load_context_messages,
        registry=a.registry,
        build_agent_prefs_payload=lambda **kwargs: prefs_bootstrap.build_agent_prefs_payload(
            settings=a.settings,
            core_tools=a.registry.core_tools,
            plugin_toolsets=a.registry.plugin_toolsets,
            **kwargs,
        ),
        write_agent_prefs_file=prefs.write_agent_prefs_file,
        agent_prefs_path=prefs.agent_prefs_path,
        settings_group_keys_lines=a.settings.group_keys_lines,
        settings_group_show=a.settings.group_show,
        settings_group_set=a.settings.group_set,
        settings_group_unset=a.settings.group_unset,
        settings_get=a.settings.get,
        settings_set=a.settings.set,
        LlmProfile_cls=LlmProfile,
        default_primary_llm_profile=default_primary_llm_profile,
        describe_llm_profile_short=describe_llm_profile_short,
        ollama_second_opinion_model=a.ollama_second_opinion_model,
        ollama_request_think_value=a.ollama_request_think_value,
        agent_thinking_level=lambda: a.settings_get_str(("agent", "thinking_level"), ""),
        agent_thinking_enabled_default_false=lambda: a.settings_get_bool(("agent", "thinking"), False),
        agent_stream_thinking_enabled=lambda: a.settings_get_bool(("agent", "stream_thinking"), False),
        verbose_ack_message=a.verbose_ack_message,
        parse_while_repl_tokens=parse_while_repl_tokens,
        call_while_condition_judge=call_while_judge,
    )

    run_interactive_repl_loop(
        session,
        install_readline=a.interactive_repl_install_readline,
        repl_read_line=a.repl_read_line,
        flush_repl_history=a.flush_repl_readline_history,
        agent_progress=a.agent_progress,
    )


def _default_prompt_templates_dir() -> str:
    return _APP.default_prompt_templates_dir()


def _default_skills_dir() -> str:
    return _APP.default_skills_dir()


def _default_tools_dir() -> str:
    return _APP.default_tools_dir()


def _enrich_search_query_for_present_day(query: str) -> str:
    return _enrich_search_query_for_present_day_impl(query, settings=_APP.settings)


def _settings_get(path: tuple[str, str]):
    return _APP.settings.get(path)


def _settings_get_str(path: tuple[str, str], default: str = "") -> str:
    return _APP.settings.get_str(path, default=default)


def _settings_get_bool(path: tuple[str, str], default: bool = False) -> bool:
    return _APP.settings.get_bool(path, default=default)


def _settings_get_int(path: tuple[str, str], default: int = 0) -> int:
    return _APP.settings.get_int(path, default=default)


def _settings_set(path: tuple[str, str], value) -> None:
    _APP.settings.set(path, value)


_TOOL_REGISTRY = _APP.registry
_PLUGIN_TOOLSETS = _APP.registry.plugin_toolsets
_PLUGIN_TOOL_HANDLERS = _APP.registry.plugin_tool_handlers

_normalize_tool_name = _APP.registry.normalize_tool_name
_route_active_toolsets_for_request = _APP.registry.route_active_toolsets_for_request

_iter_balanced_brace_objects = agent_json.iter_balanced_brace_objects
_try_json_loads_object = agent_json.try_json_loads_object
from agentlib.tools.websearch import enrich_search_query_for_present_day as _enrich_search_query_for_present_day_impl
from agentlib.tools.websearch import first_url_in_text as _first_url_in_text

from agentlib.skills.loader import (
    expand_skill_artifacts as _expand_skill_artifacts,
    load_skills_from_dir as _load_skills_from_dir,
    safe_path_under_dir as _safe_path_under_dir,
)


def parse_agent_json(raw: str) -> dict:
    return agent_json.parse_agent_json(raw, _APP.agent_json_deps())


def call_ollama_chat(*args, **kwargs):
    return _APP.call_ollama_chat(*args, **kwargs)


def _message_to_agent_json_text(msg: dict, enabled_tools: Optional[AbstractSet[str]] = None) -> str:
    return agent_json.message_to_agent_json_text(msg, enabled_tools, _APP.agent_json_deps())


def _tool_call_to_agent_dict(function_name: str, arguments):
    return agent_json.tool_call_to_agent_dict(function_name, arguments)


def _parse_tool_arguments(arguments):
    return agent_json.parse_tool_arguments(arguments)


def _parse_context_messages_data(obj):
    from agentlib.context.io import parse_context_messages_data

    return parse_context_messages_data(obj)


def _load_context_messages(path: str) -> list:
    return load_context_messages(path)


def _save_context_bundle(path: str, messages: list, user_query: str, final_answer: str, answered: bool) -> None:
    return save_context_bundle(path, messages, user_query, final_answer, answered)


def _ollama_request_think_value() -> object:
    return _APP.ollama_request_think_value()


def _apply_cli_primary_model(name: str, profile: LlmProfile) -> LlmProfile:
    return _APP.apply_cli_primary_model(name, profile)


def _agent_progress(msg: str) -> None:
    return _APP.agent_progress(msg)


def _merge_stream_message_chunks(lines_iter, *, stream_chunks: bool = False):
    return _APP.merge_stream_message_chunks(lines_iter, stream_chunks=stream_chunks)


_merge_tool_param_aliases = turn_support.merge_tool_param_aliases
_ensure_tool_defaults = turn_support.ensure_tool_defaults
_is_tool_result_weak_for_dedup = turn_support.is_tool_result_weak_for_dedup
def _tool_result_user_message(tool: str, params: dict, result: object, *, deliverable_reminder: str = "") -> str:
    return turn_support.tool_result_user_message(
        tool,
        params,
        result,
        deliverable_reminder=deliverable_reminder,
        tool_output_max=_APP.settings.get_int(("ollama", "tool_output_max"), 14000),
    )
_tool_result_indicates_retryable_failure = turn_support.tool_result_indicates_retryable_failure
_web_tool_result_followup_hint = turn_support.web_tool_result_followup_hint
_parse_tool_recovery_payload = turn_support.parse_tool_recovery_payload


def _tool_progress_message(tool: str, params: dict) -> str:
    t = (tool or "").strip()
    if t == "search_web":
        q = (params or {}).get("query")
        return f"Tool: search_web query={q!r}"
    if t == "fetch_page":
        u = (params or {}).get("url")
        return f"Tool: fetch_page url={u!r}"
    if t == "run_command":
        c = (params or {}).get("command")
        return f"Tool: run_command command={c!r}"
    if t == "use_git":
        op = (params or {}).get("op")
        return f"Tool: use_git op={op!r}"
    if t in ("read_file", "write_file", "list_directory", "tail_file", "replace_text", "download_file"):
        p = (params or {}).get("path")
        return f"Tool: {t} path={p!r}"
    if t == "call_python":
        return "Tool: call_python"
    return f"Tool: {t}"


_CACHED_CONVERSATION_TURN_DEPS: Optional[ConversationTurnDeps] = None


def _conversation_turn_deps() -> ConversationTurnDeps:
    """Lazily-built deps for `agentlib.runtime.run_agent_conversation_turn` (legacy module wiring)."""
    global _CACHED_CONVERSATION_TURN_DEPS
    if _CACHED_CONVERSATION_TURN_DEPS is None:
        _CACHED_CONVERSATION_TURN_DEPS = ConversationTurnDeps(
            coerce_enabled_tools=_TOOL_REGISTRY.coerce_enabled_tools,
            maybe_compact_context_window=_maybe_compact_context_window,
            call_ollama_chat=call_ollama_chat,
            parse_agent_json=parse_agent_json,
            deliverable_followup_block=lambda p: deliverable_followup_block(p, scalar_to_str),
            answer_missing_written_body=answer_missing_written_body,
            scalar_to_str=scalar_to_str,
            hosted_review_ready=lambda cloud, reviewer: bool(cloud and _APP.settings.get_str(("openai", "api_key"), "").strip())
            or (
                reviewer is not None
                and getattr(reviewer, "backend", "") == "hosted"
                and (getattr(reviewer, "api_key", "") or "").strip()
            ),
            second_opinion_reviewer_messages=second_opinion_reviewer_messages,
            second_opinion_result_user_message=second_opinion_result_user_message,
            call_ollama_plaintext=_APP.call_ollama_plaintext,
            call_hosted_chat_plain=_APP.call_hosted_chat_plain,
            call_openai_chat_plain=_APP.call_openai_chat_plain,
            ollama_second_opinion_model=_APP.ollama_second_opinion_model,
            route_requires_websearch_after_answer=_route_requires_websearch_after_answer,
            deliverable_skip_mandatory_web=deliverable_skip_mandatory_web,
            deliverable_first_answer_followup=deliverable_first_answer_followup,
            is_self_capability_question=routing_followups.is_self_capability_question,
            self_capability_followup=routing_followups.self_capability_followup,
            tool_need_review_followup=routing_followups.tool_need_review_followup,
            extract_json_object_from_text=_extract_json_object_from_text,
            all_known_tools=_TOOL_REGISTRY.all_known_tools,
            merge_tool_param_aliases=turn_support.merge_tool_param_aliases,
            ensure_tool_defaults=turn_support.ensure_tool_defaults,
            tool_params_fingerprint=_tool_params_fingerprint,
            search_backend_banner_line=lambda: search_backend_banner_line(_APP.settings),
            search_web=lambda query, params=None: tool_builtins.search_web(query, params=params, settings=_APP.settings),
            fetch_page=tool_builtins.fetch_page,
            run_command=tool_builtins.run_command,
            use_git=tool_builtins.use_git,
            write_file=tool_builtins.write_file,
            list_directory=tool_builtins.list_directory,
            read_file=tool_builtins.read_file,
            download_file=tool_builtins.download_file,
            tail_file=tool_builtins.tail_file,
            replace_text=tool_builtins.replace_text,
            call_python=tool_builtins.call_python,
            plugin_tool_handlers=_PLUGIN_TOOL_HANDLERS,
            tool_fault_result=lambda tool, exc: f"Tool fault: {tool} raised {type(exc).__name__}: {exc}",
            tool_recovery_may_run=_tool_recovery_may_run,
            tool_recovery_tools=turn_support.TOOL_RECOVERY_TOOLS,
            tool_result_indicates_retryable_failure=turn_support.tool_result_indicates_retryable_failure,
            suggest_tool_recovery_params=lambda tool, params, result, user_query, primary_profile, et, verbose: turn_support.suggest_tool_recovery_params(
                tool,
                params,
                result,
                user_query,
                primary_profile,
                et,
                verbose,
                call_ollama_chat=call_ollama_chat,
                merge_aliases=turn_support.merge_tool_param_aliases,
                ensure_defaults=turn_support.ensure_tool_defaults,
            ),
            confirm_tool_recovery_retry=_confirm_tool_recovery_retry,
            agent_progress=_agent_progress,
            tool_progress_message=_tool_progress_message,
            is_tool_result_weak_for_dedup=turn_support.is_tool_result_weak_for_dedup,
            tool_result_user_message=_tool_result_user_message,
        )
    return _CACHED_CONVERSATION_TURN_DEPS


def _maybe_compact_context_window(messages: list, *, user_query: str, primary_profile, verbose: int, context_cfg=None) -> list:
    # Provide a patchable summarize hook for tests.
    return maybe_compact_context_window(
        messages,
        user_query=user_query,
        primary_profile=primary_profile,
        verbose=verbose,
        context_cfg=context_cfg,
        settings_get_bool=_APP.settings.get_bool,
        settings_get_int=_APP.settings.get_int,
        call_hosted_chat_plain=_APP.call_hosted_chat_plain,
        call_ollama_plaintext=_APP.call_ollama_plaintext,
        ollama_model=_APP.ollama_model(),
        summarize_conversation_fn=_summarize_conversation_for_context,
    )


def _summarize_conversation_for_context(*, profile, user_query: str, text: str, **_kw) -> str:
    return summarize_conversation_for_context(
        profile=profile,
        user_query=user_query,
        text=text,
        call_hosted_chat_plain=_APP.call_hosted_chat_plain,
        call_ollama_plaintext=_APP.call_ollama_plaintext,
        ollama_model=_APP.ollama_model(),
    )


def _route_requires_websearch(user_query: str, today_str: str, primary_profile=None, enabled_tools=None, transcript_messages=None, **_kw):
    prof = primary_profile or default_primary_llm_profile()
    return routing.route_requires_websearch(
        user_query,
        today_str,
        prof,
        enabled_tools,
        transcript_messages,
        coerce_enabled_tools=_TOOL_REGISTRY.coerce_enabled_tools,
        call_ollama_chat=call_ollama_chat,
        parse_agent_json=parse_agent_json,
        scalar_to_str=scalar_to_str,
        router_transcript_max_messages=_APP.settings.get_int(("agent", "router_transcript_max_messages"), 80),
    )


def _route_requires_websearch_after_answer(
    user_query: str,
    today_str: str,
    proposed_answer: str,
    primary_profile=None,
    enabled_tools=None,
    transcript_messages=None,
    **_kw,
):
    prof = primary_profile or default_primary_llm_profile()
    return routing.route_requires_websearch_after_answer(
        user_query,
        today_str,
        proposed_answer,
        prof,
        enabled_tools,
        transcript_messages,
        coerce_enabled_tools=_TOOL_REGISTRY.coerce_enabled_tools,
        call_ollama_chat=call_ollama_chat,
        parse_agent_json=parse_agent_json,
        scalar_to_str=scalar_to_str,
        router_transcript_max_messages=_APP.settings.get_int(("agent", "router_transcript_max_messages"), 80),
    )


_user_wants_written_deliverable = user_wants_written_deliverable
_deliverable_skip_mandatory_web = deliverable_skip_mandatory_web
_deliverable_first_answer_followup = deliverable_first_answer_followup
_answer_missing_written_body = answer_missing_written_body
_is_self_capability_question = routing_followups.is_self_capability_question
_self_capability_followup = routing_followups.self_capability_followup


def _parse_and_apply_cli_config_flag(argv: list[str]) -> list[str]:
    from agentlib.cli import parse_and_apply_cli_config_flag

    return parse_and_apply_cli_config_flag(argv)


_set_agent_prefs_path_override = prefs.set_agent_prefs_path_override
_agent_prefs_path = prefs.agent_prefs_path
_load_agent_prefs = prefs.load_agent_prefs
_write_agent_prefs_file = prefs.write_agent_prefs_file


def _session_defaults_from_prefs(prefs_obj: Optional[dict]) -> dict:
    return _APP.session_defaults_from_prefs(prefs_obj)


def _build_agent_prefs_payload(**kwargs) -> dict:
    return prefs_bootstrap.build_agent_prefs_payload(
        settings=_APP.settings,
        core_tools=_APP.registry.core_tools,
        plugin_toolsets=_APP.registry.plugin_toolsets,
        **kwargs,
    )


# coercion helpers (tests expect these names)
_scalar_to_str = scalar_to_str
_scalar_to_int = scalar_to_int


# Ollama usage helpers (tests expect these names)
from agentlib.llm.usage import (
    format_ollama_usage_line as _format_ollama_usage_line,
    ollama_eval_generation_tok_per_sec as _ollama_eval_generation_tok_per_sec,
)


def _confirm_tool_recovery_retry(
    tool: str,
    old_params: dict,
    new_params: dict,
    rationale: str,
    *,
    interactive_tool_recovery: bool,
) -> bool:
    return turn_support.confirm_tool_recovery_retry(
        tool,
        old_params,
        new_params,
        rationale,
        interactive_tool_recovery=interactive_tool_recovery,
        stdin_isatty=sys.stdin.isatty(),
    )


_merge_partial_tool_calls = streaming.merge_partial_tool_calls


def _router_prompt(user_query: str, today_str: str, *, has_prior_transcript: bool = False) -> str:
    return routing.router_prompt(user_query, today_str, has_prior_transcript=has_prior_transcript)


def _router_transcript_slice(transcript_messages: Optional[list]) -> list:
    return routing.router_transcript_slice(
        transcript_messages,
        router_transcript_max_messages=_APP.settings.get_int(("agent", "router_transcript_max_messages"), 80),
    )


def _repl_buffered_line_max_bytes() -> int:
    v = _APP.settings.get_int(("agent", "repl_input_max_bytes"), 0)
    if v <= 0:
        v = 131072
    return max(4096, int(v))


def _tool_recovery_may_run(interactive_tool_recovery: bool) -> bool:
    return (interactive_tool_recovery and sys.stdin.isatty()) or _APP.settings.get_bool(
        ("agent", "auto_confirm_tool_retry"), False
    )


_TOOL_RECOVERY_TOOLS = turn_support.TOOL_RECOVERY_TOOLS

_parse_while_judge_bit = parse_while_judge_bit

_parse_while_repl_tokens = parse_while_repl_tokens

clean_json_response = agent_json.clean_json_response


def _extract_json_object_from_text(text: str):
    return agent_json.extract_json_object_from_text(text, _APP.agent_json_deps())


def _tool_params_fingerprint(tool: str, params: object) -> str:
    return turn_support.tool_params_fingerprint(
        tool,
        params,
        scalar_to_str_fn=scalar_to_str,
        search_web_effective_max_results=lambda p: search_web_effective_max_results(p, settings=_APP.settings),
    )


def _search_web_effective_max_results(params: object) -> int:
    return search_web_effective_max_results(params, settings=_APP.settings)


def _deliverable_followup_block(path: str) -> str:
    return deliverable_followup_block(path, scalar_to_str)


def _settings_group_keys_lines(group: str) -> str:
    return _APP.settings.group_keys_lines(group)


def _settings_group_show(group: str) -> str:
    return _APP.settings.group_show(group)


def _settings_group_set(group: str, raw_key: str, raw_value: str) -> str:
    return _APP.settings.group_set(group, raw_key, raw_value)


def _settings_group_unset(group: str, raw_key: str) -> str:
    return _APP.settings.group_unset(group, raw_key)

