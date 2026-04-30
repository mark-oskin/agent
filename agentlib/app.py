from __future__ import annotations

"""
Application composition root.

`agent.py` should be a thin entrypoint. This module owns process-global wiring:
- settings + prefs
- tool registry (core + plugins)
- conversation turn deps wiring
"""

from dataclasses import dataclass
from typing import AbstractSet, Optional

import requests

from agentlib import AgentSettings, prefs, prompts, prompt_templates_io, routing_followups
from agentlib.agent_json import AgentJsonDeps
from agentlib.context.compaction import maybe_compact_context_window, summarize_conversation_for_context
from agentlib.context.io import load_context_messages, save_context_bundle
from agentlib.deliverables import (
    answer_missing_written_body,
    deliverable_first_answer_followup,
    deliverable_skip_mandatory_web,
    user_wants_written_deliverable,
)
from agentlib.llm import streaming, usage as llm_usage
from agentlib.llm.calls import (
    call_hosted_agent_chat,
    call_hosted_chat_plain,
    call_llm_json_content,
    call_ollama_chat,
    call_ollama_plaintext,
)
from agentlib.llm.discovery import fetch_ollama_local_model_names
from agentlib.llm.profile import LlmProfile, default_primary_llm_profile
from agentlib.llm.second_opinion import (
    second_opinion_result_user_message,
    second_opinion_reviewer_messages,
)
from agentlib.prefs import bootstrap as prefs_bootstrap
from agentlib.repl.loop import run_interactive_repl_loop
from agentlib.repl.while_cmd import call_while_condition_judge, parse_while_repl_tokens
from agentlib.runtime import ConversationTurnDeps
from agentlib.tools import builtins as tool_builtins
from agentlib.tools import turn_support
from agentlib.tools.registry import ToolRegistry
from agentlib.tools.websearch import search_backend_banner_line


@dataclass
class AgentApp:
    settings: AgentSettings
    registry: ToolRegistry

    # --- settings helpers (composition root) ---

    def settings_get_str(self, path: tuple[str, str], default: str = "") -> str:
        return self.settings.get_str(path, default=default)

    def settings_get_bool(self, path: tuple[str, str], default: bool = False) -> bool:
        return self.settings.get_bool(path, default=default)

    def settings_get_int(self, path: tuple[str, str], default: int = 0) -> int:
        return self.settings.get_int(path, default=default)

    def settings_set(self, path: tuple[str, str], value) -> None:
        self.settings.set(path, value)

    def ollama_base_url(self) -> str:
        return self.settings_get_str(("ollama", "host"), "http://localhost:11434").rstrip("/")

    def ollama_model(self) -> str:
        return self.settings_get_str(("ollama", "model"), "gemma4:e4b")

    def ollama_think_value(self) -> object:
        # Keep old semantics: prefer agent.thinking_level over boolean.
        lvl = self.settings_get_str(("agent", "thinking_level"), "").strip().lower()
        if lvl:
            return lvl
        return bool(self.settings_get_bool(("agent", "thinking"), False))

    def ollama_debug(self) -> bool:
        return bool(self.settings_get_bool(("ollama", "debug"), False))

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
                resolved_prompt_templates_dir=lambda _prefs: self.settings_get_str(
                    ("agent", "prompt_templates_dir"), ""
                )
                or "",
                default_prompt_templates_dir=lambda: "",
            ),
            load_skills_from_dir=lambda _p: {},
            resolved_prompt_templates_dir=lambda _prefs: "",
            resolved_skills_dir=lambda _prefs: "",
            resolved_tools_dir=lambda _prefs: "",
            default_prompt_templates_dir=lambda: "",
            default_skills_dir=lambda: "",
            load_plugin_toolsets=lambda _p: None,
            register_tool_aliases=lambda: None,
        )

    # --- agent_json deps ---

    def agent_json_deps(self) -> AgentJsonDeps:
        return AgentJsonDeps(
            all_known_tools=self.registry.all_known_tools,
            coerce_enabled_tools=self.registry.coerce_enabled_tools,
            merge_tool_param_aliases=lambda tool, params: turn_support.merge_tool_param_aliases(
                tool, params
            ),
        )

    # --- conversation deps ---

    def conversation_turn_deps(self) -> ConversationTurnDeps:
        def _merge_stream_message_chunks(lines_iter, *, stream_chunks: bool = False):
            return streaming.merge_stream_message_chunks(
                lines_iter,
                stream_chunks=stream_chunks,
                agent_stream_thinking_enabled=lambda: bool(
                    self.settings_get_bool(("agent", "stream_thinking"), False)
                ),
                ollama_usage_from_chat_response_fn=streaming.ollama_usage_from_chat_response,
            )

        last_usage: dict | None = None

        def _set_usage(u: Optional[dict]) -> None:
            nonlocal last_usage
            last_usage = u

        def _call_ollama_chat(messages: list, primary_profile=None, enabled_tools=None, *, verbose: int = 0) -> str:
            prof = primary_profile or default_primary_llm_profile()
            return call_ollama_chat(
                messages,
                primary_profile=prof,
                enabled_tools=enabled_tools,
                verbose=verbose,
                ollama_base_url=self.ollama_base_url(),
                ollama_model=self.ollama_model(),
                ollama_think_value=self.ollama_think_value(),
                ollama_debug=self.ollama_debug(),
                merge_stream_message_chunks=_merge_stream_message_chunks,
                ollama_usage_from_chat_response=streaming.ollama_usage_from_chat_response,
                message_to_agent_json_text=lambda msg, enabled_tools=None: __import__(
                    "agentlib.agent_json", fromlist=["message_to_agent_json_text"]
                ).message_to_agent_json_text(msg, enabled_tools, self.agent_json_deps()),
                verbose_emit_final_agent_readable=lambda txt: None,
                format_ollama_usage_line=llm_usage.format_ollama_usage_line,
                set_last_ollama_usage=_set_usage,
                call_hosted_agent_chat_impl=lambda *a, **k: call_hosted_agent_chat(*a, **k),
            )

        def _call_ollama_plaintext(messages: list, model: str) -> str:
            return call_ollama_plaintext(
                base_url=self.ollama_base_url(),
                messages=messages,
                model=model,
                think_value=self.ollama_think_value(),
                merge_stream_message_chunks=_merge_stream_message_chunks,
            )

        def _call_hosted_chat_plain(messages: list, profile: LlmProfile) -> str:
            return call_hosted_chat_plain(
                messages,
                base_url=profile.base_url,
                model=profile.model,
                api_key=profile.api_key,
            )

        def _call_openai_chat_plain(messages: list) -> str:
            prof = LlmProfile(
                backend="hosted",
                base_url=self.openai_base_url(),
                model=self.openai_cloud_model(),
                api_key=self.openai_api_key(),
            )
            return _call_hosted_chat_plain(messages, prof)

        def _maybe_compact(messages: list, *, user_query: str, primary_profile, verbose: int, context_cfg=None) -> list:
            return maybe_compact_context_window(
                messages,
                user_query=user_query,
                primary_profile=primary_profile,
                verbose=verbose,
                context_cfg=context_cfg,
                settings_get_bool=self.settings_get_bool,
                settings_get_int=self.settings_get_int,
                call_hosted_chat_plain=_call_hosted_chat_plain,
                call_ollama_plaintext=_call_ollama_plaintext,
                ollama_model=self.ollama_model(),
                summarize_conversation_fn=lambda **kwargs: summarize_conversation_for_context(
                    **kwargs,
                    call_hosted_chat_plain=_call_hosted_chat_plain,
                    call_ollama_plaintext=_call_ollama_plaintext,
                    ollama_model=self.ollama_model(),
                ),
            )

        return ConversationTurnDeps(
            coerce_enabled_tools=self.registry.coerce_enabled_tools,
            maybe_compact_context_window=_maybe_compact,
            call_ollama_chat=_call_ollama_chat,
            parse_agent_json=lambda t: __import__(
                "agentlib.agent_json", fromlist=["parse_agent_json"]
            ).parse_agent_json(t, self.agent_json_deps()),
            deliverable_followup_block=lambda p: __import__(
                "agentlib.deliverables", fromlist=["deliverable_followup_block"]
            ).deliverable_followup_block(p, str),
            answer_missing_written_body=answer_missing_written_body,
            scalar_to_str=str,
            hosted_review_ready=lambda cloud, reviewer: bool(cloud and self.openai_api_key())
            or (
                reviewer is not None
                and getattr(reviewer, "backend", "") == "hosted"
                and (getattr(reviewer, "api_key", "") or "").strip()
            ),
            second_opinion_reviewer_messages=second_opinion_reviewer_messages,
            second_opinion_result_user_message=second_opinion_result_user_message,
            call_ollama_plaintext=_call_ollama_plaintext,
            call_hosted_chat_plain=_call_hosted_chat_plain,
            call_openai_chat_plain=_call_openai_chat_plain,
            ollama_second_opinion_model=self.ollama_second_opinion_model,
            route_requires_websearch_after_answer=lambda *a, **k: None,
            deliverable_skip_mandatory_web=deliverable_skip_mandatory_web,
            deliverable_first_answer_followup=deliverable_first_answer_followup,
            is_self_capability_question=routing_followups.is_self_capability_question,
            self_capability_followup=routing_followups.self_capability_followup,
            tool_need_review_followup=routing_followups.tool_need_review_followup,
            extract_json_object_from_text=lambda t: __import__(
                "agentlib.agent_json", fromlist=["extract_json_object_from_text"]
            ).extract_json_object_from_text(t, self.agent_json_deps()),
            all_known_tools=self.registry.all_known_tools,
            merge_tool_param_aliases=lambda tool, params: turn_support.merge_tool_param_aliases(tool, params),
            ensure_tool_defaults=lambda tool, params, uq: turn_support.ensure_tool_defaults(tool, params, uq),
            tool_params_fingerprint=lambda tool, params: turn_support.tool_params_fingerprint(
                tool,
                params,
                search_web_effective_max_results=lambda p: __import__(
                    "agentlib.tools.websearch", fromlist=["search_web_effective_max_results"]
                ).search_web_effective_max_results(p, settings=self.settings),
            ),
            search_backend_banner_line=lambda: search_backend_banner_line(self.settings),
            search_web=lambda query, params=None: tool_builtins.search_web(query, params=params, settings=self.settings),
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
            tool_recovery_may_run=lambda interactive: bool(interactive),
            tool_recovery_tools=turn_support.TOOL_RECOVERY_TOOLS,
            tool_result_indicates_retryable_failure=turn_support.tool_result_indicates_retryable_failure,
            suggest_tool_recovery_params=lambda *a, **k: None,
            confirm_tool_recovery_retry=lambda *a, **k: True,
            agent_progress=lambda msg: None,
            tool_progress_message=lambda tool, params: f"Tool: {tool} {params}",
            is_tool_result_weak_for_dedup=turn_support.is_tool_result_weak_for_dedup,
            tool_result_user_message=lambda tool, params, result, deliverable_reminder="": turn_support.tool_result_user_message(
                tool,
                params,
                result,
                deliverable_reminder=deliverable_reminder,
                tool_output_max=self.settings_get_int(("ollama", "tool_output_max"), 14000),
            ),
        )


def default_app(*, agent_module_dir: str) -> AgentApp:
    settings = AgentSettings.defaults()
    registry = ToolRegistry(default_tools_dir=f"{agent_module_dir}/tools")
    # Load bundled plugin toolsets on startup.
    registry.load_plugin_toolsets(registry.default_tools_dir)
    registry.register_aliases()
    return AgentApp(settings=settings, registry=registry)

