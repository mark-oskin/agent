#!/usr/bin/env -S uv run python

import sys
import json
import contextlib
import io
import subprocess
import re
import os
import shlex
import difflib
import time
import tempfile
import datetime
import warnings
from dataclasses import replace
from typing import AbstractSet, Callable, Optional, Tuple

import importlib
import importlib.util

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+.*",
)

import requests

#
# Settings
# --------
# This codebase intentionally does NOT read configuration from environment variables.
# All settings come from ~/.agent.json (prefs) and/or CLI flags (process-local).
#

from agentlib import AgentSettings
from agentlib import prefs as _prefs
from agentlib.agent_json import AgentJsonDeps
from agentlib import agent_json
from agentlib import deliverables as _deliverables
from agentlib import routing as _routing
from agentlib.runtime import ConversationTurnDeps, run_agent_conversation_turn
from agentlib.coercion import coerce_verbose_level as _coerce_verbose_level
from agentlib.coercion import scalar_to_int as _scalar_to_int
from agentlib.coercion import scalar_to_str as _scalar_to_str
from agentlib.context.compaction import maybe_compact_context_window as _compact_context_window_impl
from agentlib.context.compaction import summarize_conversation_for_context as _summarize_conversation_impl
from agentlib.llm import streaming as llm_streaming
from agentlib.llm import usage as llm_usage
from agentlib.llm.discovery import fetch_ollama_local_model_names as _fetch_ollama_local_model_names_impl
from agentlib.llm.second_opinion import (
    second_opinion_result_user_message as _second_opinion_result_user_message,
    second_opinion_reviewer_messages as _second_opinion_reviewer_messages,
)
from agentlib import prompts as agent_prompts
from agentlib import prompt_templates_io
from agentlib.repl import run_interactive_repl_loop
from agentlib.context import io as context_io
from agentlib.repl import while_cmd as repl_while_cmd
from agentlib import routing_followups
from agentlib.skills.loader import expand_skill_artifacts as _expand_skill_artifacts
from agentlib.skills.loader import load_skills_from_dir as _load_skills_from_dir
from agentlib.skills.loader import safe_path_under_dir as _safe_path_under_dir
from agentlib.tools import builtins as tool_builtins
from agentlib.tools.registry import ToolRegistry
from agentlib.llm.profile import LlmProfile, default_primary_llm_profile
from agentlib.llm.profile import llm_profile_from_pref as _llm_profile_from_pref
from agentlib.llm.profile import llm_profile_to_pref as _llm_profile_to_pref
from agentlib.prefs import bootstrap as prefs_bootstrap
from agentlib.tools import turn_support
from agentlib.tools.websearch import first_url_in_text as _first_url_in_text
from agentlib.tools.websearch import search_backend_banner_line as _websearch_backend_banner_line

# Module-global settings object (defaults until main() applies prefs/CLI).
_SETTINGS_OBJ: AgentSettings = AgentSettings.defaults()


def _settings_get(path: Tuple[str, str]):
    return _SETTINGS_OBJ.get(path)


def _settings_get_str(path: Tuple[str, str], default: str = "") -> str:
    return _SETTINGS_OBJ.get_str(path, default=default)


def _settings_get_bool(path: Tuple[str, str], default: bool = False) -> bool:
    return _SETTINGS_OBJ.get_bool(path, default=default)


def _settings_get_int(path: Tuple[str, str], default: int = 0) -> int:
    return _SETTINGS_OBJ.get_int(path, default=default)


def _settings_get_float(path: Tuple[str, str], default: float = 0.0) -> float:
    return _SETTINGS_OBJ.get_float(path, default=default)


def _settings_set(path: Tuple[str, str], value) -> None:
    _SETTINGS_OBJ.set(path, value)


def _ollama_base_url():
    return _settings_get_str(("ollama", "host"), "http://localhost:11434").rstrip("/")


def _ollama_model():
    return _settings_get_str(("ollama", "model"), "gemma4:e4b")


def _apply_cli_primary_model(name: str, profile: LlmProfile) -> LlmProfile:
    """
    --model override for this process: for local Ollama, set the session/prefs model; for hosted
    primary, set LlmProfile.model.
    """
    s = (name or "").strip()
    if not s:
        return profile
    if profile.backend == "hosted":
        return replace(profile, model=s)
    _settings_set(("ollama", "model"), s)
    return profile


def _hosted_review_ready(
    cloud_ai_enabled: bool, reviewer_hosted_profile: Optional[LlmProfile]
) -> bool:
    if cloud_ai_enabled and _settings_get_str(("openai", "api_key"), ""):
        return True
    if (
        reviewer_hosted_profile is not None
        and reviewer_hosted_profile.backend == "hosted"
        and (reviewer_hosted_profile.api_key or "").strip()
    ):
        return True
    return False


def _describe_llm_profile_short(p: LlmProfile) -> str:
    if p.backend != "hosted":
        return "ollama"
    key = "set" if (p.api_key or "").strip() else "missing"
    return f"hosted {p.model!r} @ {p.base_url!r} (api_key: {key})"


def _format_session_primary_llm_line(p: LlmProfile) -> str:
    """One-line description of the primary LLM (REPL banner and /show model)."""
    if p.backend == "hosted":
        return _describe_llm_profile_short(p)
    return f"ollama ({_ollama_model()!r})"


def _format_session_reviewer_line(
    hosted: Optional[LlmProfile], ollama_model: Optional[str]
) -> str:
    """One-line description of the second-opinion reviewer (banner and /show reviewer)."""
    if hosted is not None and hosted.backend == "hosted":
        return _describe_llm_profile_short(hosted)
    return f"ollama ({(ollama_model or _ollama_second_opinion_model())!r})"


def _strip_leading_dashes_flag(a: str) -> str:
    x = (a or "").lower().replace("_", "-")
    while x.startswith("-"):
        x = x[1:]
    return x


def _print_cli_help() -> None:
    """Print usage for non-interactive `python agent.py` invocation."""
    print(
        "Usage:\n"
        "  agent [options] [question...]\n"
        "  With no question and no action flags, start the interactive REPL.\n"
        "  With a question, run a single non-interactive turn (stdin need not be a TTY).\n"
        "\n"
        "Options:\n"
        "  -h, -?, --help        Show this help and exit\n"
        "  --config <file>       Use this config file instead of ~/.agent.json\n"
        "  --list-tools          List each tool, id, and on/off, then exit\n"
        "  -enable-tool <id>     Enable a tool for this session (repeatable)\n"
        "  -disable-tool <id>   Disable a tool (repeatable)\n"
        "  -verbose [0|1|2]     Verbose: 0 default, 1+ traces tools; optional level\n"
        "  --second_opinion, --second-opinion   Use second-opinion path when the model requests it (see prefs)\n"
        "  --cloud_ai, --cloud-ai   Allow hosted/second-opinion backends (see prefs)\n"
        "  --load_context, --load-context <file>  Load session JSON, then require a question on the command line\n"
        "  --save_context, --save-context <file>  After the run, write context bundle to this file\n"
        "  --prompt_template, --prompt-template <name>  Use this prompt template for this run\n"
        "  --model, -model <name>  Primary model: Ollama tag (sets OLLAMA_MODEL) or hosted model id when using hosted primary\n"
        "                          Also accepted: --model=<name> (same run only; does not update ~/.agent.json)\n"
        "\n"
        "Config:   persist Ollama / OpenAI / process options in ~/.agent.json via  /settings ollama|openai|agent  in the REPL "
        "(exporting variables still overrides the file for a one-off run).\n"
    )


def _verbose_ack_message(level: int) -> str:
    labels = (
        "off",
        "tool invocations",
        "tool invocations and streamed model JSON",
    )
    return f"verbose level {level} ({labels[level]}) for this session."


def _agent_progress_enabled() -> bool:
    """
    Short progress lines to stderr for long multi-step or tool-heavy work at verbose=0.
    Disable with AGENT_PROGRESS=0 or AGENT_QUIET=1.
    """
    if _settings_get_bool(("agent", "quiet"), False):
        return False
    return _settings_get_bool(("agent", "progress"), True)


def _agent_progress(msg: str) -> None:
    """One-line heartbeat; does not use verbose (see _agent_progress_enabled)."""
    if not _agent_progress_enabled() or not (msg or "").strip():
        return
    print(f"→ {msg.strip()}", file=sys.stderr, flush=True)


def _progress_clip(s: object, max_len: int = 120) -> str:
    t = _scalar_to_str(s, "").replace("\n", " ").strip()
    if len(t) > max_len:
        return t[: max_len - 1] + "…"
    return t


def _tool_progress_message(tool: str, params: dict) -> str:
    """Compact, useful progress line for verbose=0 heartbeats."""
    t = (tool or "").strip()
    p = params if isinstance(params, dict) else {}
    if t == "search_web":
        return (
            f"Tool: search_web {_search_backend_banner_line()} "
            f"query={_progress_clip(p.get('query'))!r}"
        )
    if t == "fetch_page":
        return f"Tool: fetch_page url={_progress_clip(p.get('url'))!r}"
    if t == "read_file":
        return f"Tool: read_file path={_progress_clip(p.get('path'))!r}"
    if t == "list_directory":
        return f"Tool: list_directory path={_progress_clip(p.get('path'))!r}"
    if t == "tail_file":
        return f"Tool: tail_file path={_progress_clip(p.get('path'))!r} lines={_progress_clip(p.get('lines', 20))}"
    if t == "run_command":
        return f"Tool: run_command command={_progress_clip(p.get('command'))!r}"
    if t == "write_file":
        return f"Tool: write_file path={_progress_clip(p.get('path'))!r}"
    if t == "replace_text":
        return (
            f"Tool: replace_text path={_progress_clip(p.get('path'))!r} "
            f"pattern={_progress_clip(p.get('pattern'))!r}"
        )
    if t == "download_file":
        return (
            f"Tool: download_file url={_progress_clip(p.get('url'))!r} "
            f"path={_progress_clip(p.get('path'))!r}"
        )
    if t == "use_git":
        op = _progress_clip(p.get("op") or p.get("operation"))
        return f"Tool: use_git op={op!r}"
    if t == "call_python":
        return "Tool: call_python"
    return f"Tool: {t}"


_AGENT_PREFS_VERSION = _prefs.AGENT_PREFS_VERSION


def _agent_prefs_path() -> str:
    return _prefs.agent_prefs_path()


def _set_agent_prefs_path_override(path: Optional[str]) -> None:
    _prefs.set_agent_prefs_path_override(path)


def _parse_and_apply_cli_config_flag(argv: list[str]) -> list[str]:
    from agentlib.cli import parse_and_apply_cli_config_flag

    return parse_and_apply_cli_config_flag(argv)


def _agent_module_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _default_prompt_templates_dir() -> str:
    return os.path.join(_agent_module_dir(), "prompt_templates")


def _default_skills_dir() -> str:
    return os.path.join(_agent_module_dir(), "skills")


def _default_tools_dir() -> str:
    return os.path.join(_agent_module_dir(), "tools")


# Tool registry (core + plugin tools). Loaded at import time.
# If tools_dir is overridden in prefs, main() will reload.
_TOOL_REGISTRY = ToolRegistry(default_tools_dir=_default_tools_dir())
_TOOL_REGISTRY.load_plugin_toolsets(_default_tools_dir())
_TOOL_REGISTRY.register_aliases()

# Tool registry helpers (core tools + plugins + aliases).
_all_known_tools = _TOOL_REGISTRY.all_known_tools
_coerce_enabled_tools = _TOOL_REGISTRY.coerce_enabled_tools
_normalize_tool_name = _TOOL_REGISTRY.normalize_tool_name
_format_unknown_tool_hint = _TOOL_REGISTRY.format_unknown_tool_hint
_format_settings_tools_list = _TOOL_REGISTRY.format_settings_tools_list
_describe_tool_call_contract = _TOOL_REGISTRY.describe_tool_call_contract
_tool_policy_runner_text = _TOOL_REGISTRY.tool_policy_runner_text
_plugin_tools_for_toolset = _TOOL_REGISTRY.plugin_tools_for_toolset
_route_active_toolsets_for_request = _TOOL_REGISTRY.route_active_toolsets_for_request
_effective_enabled_tools_for_turn = _TOOL_REGISTRY.effective_enabled_tools_for_turn

_PLUGIN_TOOL_HANDLERS = _TOOL_REGISTRY.plugin_tool_handlers
_PLUGIN_TOOLSETS = _TOOL_REGISTRY.plugin_toolsets


def _resolved_prompt_templates_dir(prefs: Optional[dict] = None) -> str:
    if prefs and isinstance(prefs, dict) and (prefs.get("prompt_templates_dir") or "").strip():
        return os.path.abspath(os.path.expanduser(str(prefs["prompt_templates_dir"]).strip()))
    return _default_prompt_templates_dir()


def _resolved_skills_dir(prefs: Optional[dict] = None) -> str:
    if prefs and isinstance(prefs, dict) and (prefs.get("skills_dir") or "").strip():
        return os.path.abspath(os.path.expanduser(str(prefs["skills_dir"]).strip()))
    return _default_skills_dir()


def _resolved_tools_dir(prefs: Optional[dict] = None) -> str:
    if prefs and isinstance(prefs, dict) and (prefs.get("tools_dir") or "").strip():
        return os.path.abspath(os.path.expanduser(str(prefs["tools_dir"]).strip()))
    return _default_tools_dir()


def _format_skills_for_selector(skills_map: dict) -> str:
    from agentlib.skills.selection import format_skills_for_selector

    return format_skills_for_selector(skills_map)


def _ml_select_skill_id(
    user_request: str,
    skills_map: dict,
    *,
    primary_profile: Optional[LlmProfile],
    verbose: int,
) -> Tuple[Optional[str], str]:
    from agentlib.skills.selection import ml_select_skill_id

    return ml_select_skill_id(
        user_request,
        skills_map,
        primary_profile=primary_profile,
        verbose=verbose,
        call_llm_json_content=call_llm_json_content,
        agent_progress=_agent_progress,
        try_json_loads_object=_try_json_loads_object,
        parse_json_with_skill_id=_parse_json_with_skill_id,
    )


def _skill_plan_steps(
    *,
    user_request: str,
    today_str: str,
    skill_id: str,
    skills_map: dict,
    primary_profile: Optional[LlmProfile],
    _enabled_tools: AbstractSet[str],
    verbose: int,
    _system_prompt_override: Optional[str],
) -> Tuple[Optional[list], str]:
    from agentlib.skills.planner import skill_plan_steps

    return skill_plan_steps(
        user_request=user_request,
        today_str=today_str,
        skill_id=skill_id,
        skills_map=skills_map,
        primary_profile=primary_profile,
        enabled_tools=_enabled_tools,
        verbose=verbose,
        system_prompt_override=_system_prompt_override,
        agent_progress=_agent_progress,
        call_llm_json_content=call_llm_json_content,
        try_json_loads_object=_try_json_loads_object,
        parse_workflow_plan_dict=_parse_workflow_plan_dict,
        scalar_to_int=_scalar_to_int,
        scalar_to_str=_scalar_to_str,
    )


def _match_skill_detail(
    user_text: str, skills: Optional[dict]
) -> Tuple[Optional[str], Optional[str]]:
    from agentlib.skills.selection import match_skill_detail

    return match_skill_detail(user_text, skills)


def _match_skill_id(user_text: str, skills: Optional[dict]) -> Optional[str]:
    from agentlib.skills.selection import match_skill_id

    return match_skill_id(user_text, skills)


def _print_skill_usage_verbose(
    verbose: int,
    *,
    source: str,
    skill_id: Optional[str],
    base_tools: AbstractSet[str],
    effective_tools: AbstractSet[str],
    detail: Optional[str] = None,
) -> None:
    """Log skill id and effective tool set when verbose >= 1."""
    if verbose < 1:
        return
    sk = skill_id if skill_id else "(none)"
    et = sorted(effective_tools)
    bt = sorted(base_tools)
    if set(bt) != set(et):
        print(
            f"[*] [skills:{source}] id={sk!r} tools={et} (narrowed from session {bt})"
        )
    else:
        print(f"[*] [skills:{source}] id={sk!r} tools={et}")
    if detail:
        print(f"[*] [skills:{source}] {detail}")


def _effective_enabled_tools_for_skill(
    base_enabled: AbstractSet[str], skills_map: dict, skill_id: Optional[str]
) -> AbstractSet[str]:
    """If the skill fixed a tool list, narrow to the intersection with session-enabled tools."""
    if not skill_id or not isinstance(skills_map, dict):
        return base_enabled
    rec = skills_map.get(skill_id)
    if not isinstance(rec, dict):
        return base_enabled
    raw = rec.get("tools")
    if not isinstance(raw, list) or not raw:
        return base_enabled
    wanted = {str(t).strip() for t in raw if isinstance(t, str) and t.strip() in _all_known_tools()}
    if not wanted:
        return base_enabled
    narrowed = wanted & set(base_enabled)
    return frozenset(narrowed) if narrowed else base_enabled


_REPL_READLINE_INSTALLED = False


def _repl_history_path() -> str:
    override = _settings_get_str(("agent", "repl_history"), "")
    if override:
        return os.path.expanduser(override)
    return os.path.join(os.path.expanduser("~"), ".agent_repl_history")


def _agent_stream_thinking_enabled() -> bool:
    """
    Whether to stream `message.thinking` chunks to the user during Ollama streaming.
    Controlled via prefs/CLI (agent.stream_thinking).
    """
    return _settings_get_bool(("agent", "stream_thinking"), False)


def _agent_thinking_level() -> Optional[str]:
    v = _settings_get_str(("agent", "thinking_level"), "").strip().lower()
    if v in ("low", "medium", "high"):
        return v
    return None


def _agent_thinking_enabled_default_false() -> bool:
    """
    Default behavior: thinking OFF unless explicitly enabled.
    """
    return _settings_get_bool(("agent", "thinking"), False)


def _ollama_request_think_value() -> object:
    """
    Value for Ollama's request-level `think` field.
    When thinking is off, always return False so non–thinking models never receive
    a string or true (a stale AGENT_THINKING_LEVEL alone must not enable think).
    When on: use AGENT_THINKING_LEVEL if set, else bool or gpt-oss default level.
    """
    if not _agent_thinking_enabled_default_false():
        return False
    lvl = _agent_thinking_level()
    if lvl:
        return lvl
    # gpt-oss models ignore boolean think; they require a level string.
    try:
        mod = (_ollama_model() or "").strip().lower()
    except Exception:
        mod = ""
    if mod.startswith("gpt-oss"):
        return "medium"
    return True


def _flush_repl_readline_history() -> None:
    if not _REPL_READLINE_INSTALLED:
        return
    try:
        import readline  # type: ignore[import-not-found]

        readline.write_history_file(_repl_history_path())
    except (ImportError, OSError):
        pass


def _interactive_repl_install_readline() -> bool:
    """Hook GNU readline into input() when running in a real terminal (history + line edit)."""
    global _REPL_READLINE_INSTALLED
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False
    try:
        import readline  # type: ignore[import-not-found]
    except ImportError:
        return False
    path = _repl_history_path()
    if not _REPL_READLINE_INSTALLED:
        try:
            readline.read_history_file(path)
        except OSError:
            pass
        try:
            readline.set_history_length(2000)
        except AttributeError:
            pass
        import atexit

        atexit.register(_flush_repl_readline_history)
        _REPL_READLINE_INSTALLED = True
    return True


_REPL_INPUT_MAX_DEFAULT = 131072


def _repl_buffered_line_max_bytes() -> int:
    v = _settings_get_int(("agent", "repl_input_max_bytes"), 0)
    if v <= 0:
        v = _REPL_INPUT_MAX_DEFAULT
    return max(4096, int(v))


def _repl_read_line(prompt: str) -> str:
    """
    Read one REPL line. Default: input() + readline (history, arrows, usual editing).

    Enable buffered-line mode in prefs (agent.repl_buffered_line=true) on a TTY to read from
    stdin in binary mode up to agent.repl_input_max_bytes (default 128KiB) for large single-line pastes;
    readline.add_history is called afterward so ↑ still recalls those lines (without per-line readline editing on entry).
    is called afterward so ↑ still recalls those lines (without per-line readline editing on entry).
    """
    if not sys.stdin.isatty():
        return input(prompt)
    if not _settings_get_bool(("agent", "repl_buffered_line"), False):
        return input(prompt)
    maxb = _repl_buffered_line_max_bytes()
    print(prompt, end="", flush=True)
    try:
        raw = sys.stdin.buffer.readline(maxb + 1)
    except (OSError, ValueError):
        return ""
    if len(raw) > maxb:
        print(f"\n[Input truncated to {maxb} bytes]", file=sys.stderr)
        raw = raw[:maxb]
    text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
    if _REPL_READLINE_INSTALLED and text.strip():
        try:
            import readline  # type: ignore[import-not-found]

            readline.add_history(text)
        except (ImportError, OSError, AttributeError):
            pass
    return text


def _load_agent_prefs() -> Optional[dict]:
    return _prefs.load_agent_prefs()


def _migrate_settings_groups_from_prefs(prefs: dict) -> None:
    _prefs.apply_prefs_to_settings(_SETTINGS_OBJ, prefs)


def _session_defaults_from_prefs(prefs: Optional[dict]) -> dict:
    return prefs_bootstrap.session_defaults_from_prefs(
        prefs,
        migrate_prefs=_migrate_settings_groups_from_prefs,
        settings=_SETTINGS_OBJ,
        core_tools=_TOOL_REGISTRY.core_tools,
        plugin_toolsets=_PLUGIN_TOOLSETS,
        normalize_tool_name=_normalize_tool_name,
        merge_prompt_templates=lambda p: prompt_templates_io.merge_prompt_templates(
            p,
            resolved_prompt_templates_dir=_resolved_prompt_templates_dir,
            default_prompt_templates_dir=_default_prompt_templates_dir,
        ),
        load_skills_from_dir=_load_skills_from_dir,
        resolved_prompt_templates_dir=_resolved_prompt_templates_dir,
        resolved_skills_dir=_resolved_skills_dir,
        resolved_tools_dir=_resolved_tools_dir,
        default_prompt_templates_dir=_default_prompt_templates_dir,
        default_skills_dir=_default_skills_dir,
        load_plugin_toolsets=lambda tools_dir=None: _TOOL_REGISTRY.load_plugin_toolsets(tools_dir),
        register_tool_aliases=_TOOL_REGISTRY.register_aliases,
    )


def _settings_group_keys_lines(group: str) -> str:
    return _SETTINGS_OBJ.group_keys_lines(group)


def _settings_group_show(group: str) -> str:
    return _SETTINGS_OBJ.group_show(group)


def _settings_group_set(group: str, raw_key: str, raw_value: str) -> str:
    return _SETTINGS_OBJ.group_set(group, raw_key, raw_value)


def _settings_group_unset(group: str, raw_key: str) -> str:
    return _SETTINGS_OBJ.group_unset(group, raw_key)


def _write_agent_prefs_file(payload: dict) -> None:
    _prefs.write_agent_prefs_file(payload)


def _build_agent_prefs_payload(
    *,
    primary_profile: LlmProfile,
    second_opinion_on: bool,
    cloud_ai_enabled: bool,
    enabled_tools: AbstractSet[str],
    enabled_toolsets: Optional[AbstractSet[str]] = None,
    reviewer_hosted_profile: Optional[LlmProfile],
    reviewer_ollama_model: Optional[str],
    session_save_path: Optional[str],
    system_prompt_override: Optional[str] = None,
    system_prompt_path_override: Optional[str] = None,
    prompt_templates: Optional[dict] = None,
    prompt_template_default: Optional[str] = None,
    prompt_templates_dir: Optional[str] = None,
    skills_dir: Optional[str] = None,
    tools_dir: Optional[str] = None,
    context_manager: Optional[dict] = None,
    verbose_level: int = 0,
) -> dict:
    return prefs_bootstrap.build_agent_prefs_payload(
        settings=_SETTINGS_OBJ,
        primary_profile=primary_profile,
        second_opinion_on=second_opinion_on,
        cloud_ai_enabled=cloud_ai_enabled,
        enabled_tools=enabled_tools,
        core_tools=_TOOL_REGISTRY.core_tools,
        plugin_toolsets=_PLUGIN_TOOLSETS,
        reviewer_hosted_profile=reviewer_hosted_profile,
        reviewer_ollama_model=reviewer_ollama_model,
        session_save_path=session_save_path,
        system_prompt_override=system_prompt_override,
        system_prompt_path_override=system_prompt_path_override,
        prompt_templates=prompt_templates,
        prompt_template_default=prompt_template_default,
        prompt_templates_dir=prompt_templates_dir,
        skills_dir=skills_dir,
        tools_dir=tools_dir,
        context_manager=context_manager,
        verbose_level=verbose_level,
        enabled_toolsets=enabled_toolsets,
    )


def _default_search_web_max_results() -> int:
    from agentlib.tools.websearch import default_search_web_max_results

    return default_search_web_max_results(_SETTINGS_OBJ)


def _search_web_max_results_clamped(n: object, *, fallback: int) -> int:
    from agentlib.tools.websearch import search_web_max_results_clamped

    return search_web_max_results_clamped(n, fallback=fallback)


def _search_web_effective_max_results(params: object) -> int:
    from agentlib.tools.websearch import search_web_effective_max_results

    return search_web_effective_max_results(params, settings=_SETTINGS_OBJ)


def _merge_tool_param_aliases(tool: str, params: dict) -> dict:
    """Map alternate parameter names models use into the keys our tools expect."""
    return turn_support.merge_tool_param_aliases(tool, params, scalar_to_str_fn=_scalar_to_str)


_CACHED_AGENT_JSON_DEPS: Optional[AgentJsonDeps] = None


def _agent_json_deps() -> AgentJsonDeps:
    global _CACHED_AGENT_JSON_DEPS
    if _CACHED_AGENT_JSON_DEPS is None:
        _CACHED_AGENT_JSON_DEPS = AgentJsonDeps(
            all_known_tools=_all_known_tools,
            coerce_enabled_tools=_coerce_enabled_tools,
            merge_tool_param_aliases=_merge_tool_param_aliases,
        )
    return _CACHED_AGENT_JSON_DEPS


def parse_agent_json(resp_text):
    return agent_json.parse_agent_json(resp_text, _agent_json_deps())


def _message_to_agent_json_text(msg, enabled_tools=None):
    return agent_json.message_to_agent_json_text(msg, enabled_tools, _agent_json_deps())


def _extract_json_object_from_text(text: str) -> Optional[str]:
    return agent_json.extract_json_object_from_text(text, _agent_json_deps())


_iter_balanced_brace_objects = agent_json.iter_balanced_brace_objects
_try_json_loads_object = agent_json.try_json_loads_object
_tool_call_to_agent_dict = agent_json.tool_call_to_agent_dict
_parse_tool_arguments = agent_json.parse_tool_arguments
clean_json_response = agent_json.clean_json_response


_user_wants_written_deliverable = _deliverables.user_wants_written_deliverable
_deliverable_skip_mandatory_web = _deliverables.deliverable_skip_mandatory_web
_answer_missing_written_body = _deliverables.answer_missing_written_body
_deliverable_first_answer_followup = _deliverables.deliverable_first_answer_followup


def _deliverable_followup_block(path: str) -> str:
    return _deliverables.deliverable_followup_block(path, _scalar_to_str)


def _ensure_tool_defaults(tool: str, params: dict, user_query: str) -> dict:
    """Fill required parameters when the model emits an empty object."""
    return turn_support.ensure_tool_defaults(
        tool, params, user_query, scalar_to_str_fn=_scalar_to_str
    )


def _enrich_search_query_for_present_day(query: str) -> str:
    from agentlib.tools.websearch import enrich_search_query_for_present_day

    return enrich_search_query_for_present_day(query, settings=_SETTINGS_OBJ)


_ollama_usage_from_chat_response = llm_streaming.ollama_usage_from_chat_response
_merge_partial_tool_calls = llm_streaming.merge_partial_tool_calls

_ollama_eval_generation_tok_per_sec = llm_usage.ollama_eval_generation_tok_per_sec
_format_ollama_usage_line = llm_usage.format_ollama_usage_line


_last_ollama_chat_usage: Optional[dict] = None


def _format_last_ollama_usage_for_repl() -> str:
    """Human-readable report for /usage (last local Ollama agent chat only)."""
    return llm_usage.format_last_ollama_usage_for_repl(_last_ollama_chat_usage)


def _merge_stream_message_chunks(lines_iter, *, stream_chunks: bool = False):
    """Merge streaming /api/chat chunks into one assistant message dict + final usage dict."""
    return llm_streaming.merge_stream_message_chunks(
        lines_iter,
        stream_chunks=stream_chunks,
        agent_stream_thinking_enabled=_agent_stream_thinking_enabled,
        ollama_usage_from_chat_response_fn=_ollama_usage_from_chat_response,
    )


def _tool_params_fingerprint(tool: str, params) -> str:
    """Stable key for deduplicating identical tool calls."""
    return turn_support.tool_params_fingerprint(
        tool,
        params,
        scalar_to_str_fn=_scalar_to_str,
        search_web_effective_max_results=_search_web_effective_max_results,
    )


def _web_tool_result_followup_hint(tool: str, result: str) -> str:
    """
    Short, tool-specific nudge when web tools return errors or no usable data,
    so the model can recover without an extra model round in some clients.
    """
    return turn_support.web_tool_result_followup_hint(tool, result)


def _is_tool_result_weak_for_dedup(result: str) -> bool:
    """If true, search_web output does not count toward saw_strong_web_result (no URL-backed sources)."""
    return turn_support.is_tool_result_weak_for_dedup(result)


def _tool_result_user_message(
    tool: str, params: dict, result: str, deliverable_reminder: str = ""
) -> str:
    """User follow-up after a tool run so the model reads output and stops re-querying."""
    return turn_support.tool_result_user_message(
        tool,
        params,
        result,
        deliverable_reminder=deliverable_reminder,
        tool_output_max=_settings_get_int(("ollama", "tool_output_max"), 14000),
        scalar_to_str_fn=_scalar_to_str,
    )


_TOOL_RECOVERY_TOOLS = turn_support.TOOL_RECOVERY_TOOLS


def _tool_result_indicates_retryable_failure(tool: str, result) -> bool:
    """True when output looks like a hard tool failure (not normal STDERR on stderr)."""
    return turn_support.tool_result_indicates_retryable_failure(tool, result)


def _parse_tool_recovery_payload(resp_text: str) -> Optional[Tuple[dict, str]]:
    """Parse a recovery-only JSON object (not normalized as agent tool_call JSON)."""
    return turn_support.parse_tool_recovery_payload(resp_text)


def _suggest_tool_recovery_params(
    tool: str,
    params: dict,
    result: str,
    user_query: str,
    primary_profile: Optional[LlmProfile],
    et: AbstractSet[str],
    verbose: int,
) -> Optional[Tuple[dict, str]]:
    """Ask the primary model for corrected parameters for one retry."""
    return turn_support.suggest_tool_recovery_params(
        tool,
        params,
        result,
        user_query,
        primary_profile,
        et,
        verbose,
        call_ollama_chat=call_ollama_chat,
        merge_aliases=_merge_tool_param_aliases,
        ensure_defaults=_ensure_tool_defaults,
    )


def _env_tool_retry_auto_confirm() -> bool:
    return _settings_get_bool(("agent", "auto_confirm_tool_retry"), False)


def _tool_recovery_may_run(interactive_tool_recovery: bool) -> bool:
    """Recovery calls the model; only do that when a human or env can confirm a retry."""
    return (interactive_tool_recovery and sys.stdin.isatty()) or _env_tool_retry_auto_confirm()


def _confirm_tool_recovery_retry(
    tool: str,
    old_params: dict,
    new_params: dict,
    rationale: str,
    *,
    interactive_tool_recovery: bool,
) -> bool:
    """Log model-proposed recovery; always proceed with one automatic retry (no y/N prompt)."""
    return turn_support.confirm_tool_recovery_retry(
        tool,
        old_params,
        new_params,
        rationale,
        interactive_tool_recovery=interactive_tool_recovery,
        stdin_isatty=sys.stdin.isatty(),
    )


def _ollama_second_opinion_model():
    return _settings_get_str(("ollama", "second_opinion_model"), "llama3.2:latest").strip()


def _openai_api_key():
    return _settings_get_str(("openai", "api_key"), "")


def _openai_base_url():
    return _settings_get_str(("openai", "base_url"), "https://api.openai.com/v1").rstrip("/")


def _openai_cloud_model():
    # Reviewer cloud model id.
    return _settings_get_str(("openai", "cloud_model"), _settings_get_str(("openai", "model"), "gpt-4o-mini")).strip()


def call_ollama_plaintext(messages: list, model: str) -> str:
    from agentlib.llm.calls import call_ollama_plaintext as _impl

    return _impl(
        base_url=_ollama_base_url(),
        messages=messages,
        model=model,
        think_value=_ollama_request_think_value(),
        merge_stream_message_chunks=_merge_stream_message_chunks,
    )


def call_hosted_chat_plain(messages: list, profile: LlmProfile) -> str:
    from agentlib.llm.calls import call_hosted_chat_plain as _impl

    return _impl(
        messages,
        base_url=profile.base_url,
        model=profile.model,
        api_key=profile.api_key,
    )


def call_openai_chat_plain(messages: list) -> str:
    """Legacy helper: uses JSON/CLI openai.* settings."""
    prof = LlmProfile(
        backend="hosted",
        base_url=_openai_base_url(),
        model=_openai_cloud_model(),
        api_key=_openai_api_key(),
    )
    return call_hosted_chat_plain(messages, prof)


def call_llm_json_content(
    messages: list,
    primary_profile: Optional[LlmProfile] = None,
    *,
    verbose: int = 0,
) -> str:
    from agentlib.llm.calls import call_llm_json_content as _impl

    prof = primary_profile or default_primary_llm_profile()

    def _set_usage(u: Optional[dict]) -> None:
        global _last_ollama_chat_usage
        _last_ollama_chat_usage = u

    return _impl(
        messages,
        primary_profile=prof,
        verbose=verbose,
        ollama_base_url=_ollama_base_url(),
        ollama_model=_ollama_model(),
        merge_stream_message_chunks=_merge_stream_message_chunks,
        ollama_usage_from_chat_response=_ollama_usage_from_chat_response,
        set_last_ollama_usage=_set_usage,
    )


def _parse_json_with_skill_id(raw: str) -> dict:
    """Parse skill-selector JSON: {\"skill_id\": \"...\", \"rationale\": \"...\" }."""
    o = _try_json_loads_object(raw)
    if isinstance(o, dict) and "skill_id" in o:
        return o
    for span in _iter_balanced_brace_objects(raw or ""):
        o2 = _try_json_loads_object(span)
        if isinstance(o2, dict) and "skill_id" in o2:
            return o2
    return {}


def _parse_workflow_plan_dict(raw: str) -> Optional[dict]:
    """Parse workflow plan JSON: must contain non-empty 'steps' list of objects with title."""
    o = _try_json_loads_object(raw)
    if _is_workflow_plan_obj(o):
        return o
    for span in _iter_balanced_brace_objects(raw or ""):
        o2 = _try_json_loads_object(span)
        if _is_workflow_plan_obj(o2):
            return o2
    return None


def _is_workflow_plan_obj(o) -> bool:
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


def _summarize_conversation_for_context(
    *,
    profile: Optional[LlmProfile],
    user_query: str,
    text: str,
    **_: object,
) -> str:
    return _summarize_conversation_impl(
        profile=profile,
        user_query=user_query,
        text=text,
        call_hosted_chat_plain=call_hosted_chat_plain,
        call_ollama_plaintext=call_ollama_plaintext,
        ollama_model=_ollama_model(),
    )


def _maybe_compact_context_window(
    messages: list,
    *,
    user_query: str,
    primary_profile: Optional[LlmProfile],
    verbose: int,
    context_cfg: Optional[dict] = None,
) -> list:
    return _compact_context_window_impl(
        messages,
        user_query=user_query,
        primary_profile=primary_profile,
        verbose=verbose,
        context_cfg=context_cfg,
        settings_get_bool=_settings_get_bool,
        settings_get_int=_settings_get_int,
        call_hosted_chat_plain=call_hosted_chat_plain,
        call_ollama_plaintext=call_ollama_plaintext,
        ollama_model=_ollama_model(),
        summarize_conversation_fn=_summarize_conversation_for_context,
    )


def _verbose_emit_final_agent_readable(agent_json_text: str) -> None:
    """After verbose raw streaming (or one-shot raw text), print parsed agent JSON readably."""
    if not (agent_json_text or "").strip():
        return
    d = parse_agent_json(agent_json_text)
    print("\n--- Final agent message ---", flush=True)
    print(json.dumps(d, indent=2, ensure_ascii=False), flush=True)


def call_hosted_agent_chat(
    messages: list,
    profile: LlmProfile,
    enabled_tools: Optional[AbstractSet[str]] = None,
    *,
    verbose: int = 0,
) -> str:
    from agentlib.llm.calls import call_hosted_agent_chat as _impl

    return _impl(
        messages,
        base_url=profile.base_url,
        model=profile.model,
        api_key=profile.api_key,
        enabled_tools=enabled_tools,
        verbose=verbose,
        message_to_agent_json_text=_message_to_agent_json_text,
        verbose_emit_final_agent_readable=_verbose_emit_final_agent_readable,
    )


def call_ollama_chat(
    messages: list,
    primary_profile: Optional[LlmProfile] = None,
    enabled_tools: Optional[AbstractSet[str]] = None,
    *,
    verbose: int = 0,
) -> str:
    from agentlib.llm.calls import call_ollama_chat as _impl

    prof = primary_profile or default_primary_llm_profile()

    def _set_usage(u: Optional[dict]) -> None:
        global _last_ollama_chat_usage
        _last_ollama_chat_usage = u

    return _impl(
        messages,
        primary_profile=prof,
        enabled_tools=enabled_tools,
        verbose=verbose,
        ollama_base_url=_ollama_base_url(),
        ollama_model=_ollama_model(),
        ollama_think_value=_ollama_request_think_value(),
        ollama_debug=_settings_get_bool(("ollama", "debug"), False),
        merge_stream_message_chunks=_merge_stream_message_chunks,
        ollama_usage_from_chat_response=_ollama_usage_from_chat_response,
        message_to_agent_json_text=_message_to_agent_json_text,
        verbose_emit_final_agent_readable=_verbose_emit_final_agent_readable,
        format_ollama_usage_line=_format_ollama_usage_line,
        set_last_ollama_usage=_set_usage,
        call_hosted_agent_chat_impl=call_hosted_agent_chat,
    )


def _search_backend_banner_line() -> str:
    """Same banner prefix as prepended to search_web tool output."""
    return _websearch_backend_banner_line(_SETTINGS_OBJ)

def _tool_fault_result(tool: str, exc: BaseException) -> str:
    """Convert a tool crash/exception into a stable string the model can reason about."""
    t = (tool or "").strip() or "(unknown tool)"
    en = type(exc).__name__
    msg = str(exc).strip()
    if not msg:
        msg = repr(exc)
    return f"Tool fault: {t} raised {en}: {msg}"


# System instructions exposing all tool actions
from agentlib.prompts import SYSTEM_INSTRUCTIONS

ROUTER_INSTRUCTIONS = _routing.ROUTER_INSTRUCTIONS


def _router_transcript_slice(transcript_messages: Optional[list]) -> list:
    return _routing.router_transcript_slice(
        transcript_messages,
        router_transcript_max_messages=_settings_get_int(
            ("agent", "router_transcript_max_messages"), 80
        ),
    )


def _router_llm_messages(transcript_slice: list, tail_user_content: str) -> list:
    return _routing.router_llm_messages(transcript_slice, tail_user_content)


def _router_prompt(
    user_query: str, today_str: str, *, has_prior_transcript: bool = False
) -> str:
    return _routing.router_prompt(
        user_query, today_str, has_prior_transcript=has_prior_transcript
    )


_tool_need_review_followup = routing_followups.tool_need_review_followup
_is_self_capability_question = routing_followups.is_self_capability_question
_self_capability_followup = routing_followups.self_capability_followup


def _route_requires_websearch(
    user_query: str,
    today_str: str,
    primary_profile: Optional[LlmProfile] = None,
    enabled_tools: Optional[AbstractSet[str]] = None,
    transcript_messages: Optional[list] = None,
) -> Optional[str]:
    return _routing.route_requires_websearch(
        user_query,
        today_str,
        primary_profile,
        enabled_tools,
        transcript_messages,
        coerce_enabled_tools=_coerce_enabled_tools,
        call_ollama_chat=call_ollama_chat,
        parse_agent_json=parse_agent_json,
        scalar_to_str=_scalar_to_str,
        router_transcript_max_messages=_settings_get_int(
            ("agent", "router_transcript_max_messages"), 80
        ),
    )


def _route_requires_websearch_after_answer(
    user_query: str,
    today_str: str,
    proposed_answer: str,
    primary_profile: Optional[LlmProfile] = None,
    enabled_tools: Optional[AbstractSet[str]] = None,
    transcript_messages: Optional[list] = None,
) -> Optional[str]:
    return _routing.route_requires_websearch_after_answer(
        user_query,
        today_str,
        proposed_answer,
        primary_profile,
        enabled_tools,
        transcript_messages,
        coerce_enabled_tools=_coerce_enabled_tools,
        call_ollama_chat=call_ollama_chat,
        parse_agent_json=parse_agent_json,
        scalar_to_str=_scalar_to_str,
        router_transcript_max_messages=_settings_get_int(
            ("agent", "router_transcript_max_messages"), 80
        ),
    )


def _parse_context_messages_data(raw) -> list:
    """Normalize JSON (bundle dict or bare list) into Ollama-style message dicts."""
    return context_io.parse_context_messages_data(raw)


def _load_context_messages(path: str) -> list:
    """Load a prior chat from JSON written by --save_context (or a bare list of {role, content})."""
    return context_io.load_context_messages(path, scalar_to_str_fn=_scalar_to_str)


def _save_context_bundle(path: str, messages: list, user_query: str, final_answer: Optional[str], answered: bool):
    """Persist full message list plus the new question and final answer (if any)."""
    context_io.save_context_bundle(
        path,
        messages,
        user_query,
        final_answer,
        answered,
        scalar_to_str_fn=_scalar_to_str,
    )


def _fetch_ollama_local_model_names():
    """Return sorted unique model names from GET /api/tags (local Ollama)."""
    return _fetch_ollama_local_model_names_impl(
        _ollama_base_url(), http_get=requests.get, timeout=60
    )


def _runner_instruction_bits(
    second_opinion: bool,
    cloud: bool,
    *,
    primary_profile: Optional[LlmProfile] = None,
    reviewer_ollama_model: Optional[str] = None,
    reviewer_hosted_profile: Optional[LlmProfile] = None,
    enabled_tools: Optional[AbstractSet[str]] = None,
) -> str:
    from agentlib.prompts import runner_instruction_bits

    pp = primary_profile or default_primary_llm_profile()
    return runner_instruction_bits(
        second_opinion=second_opinion,
        cloud=cloud,
        primary_profile=pp,
        reviewer_hosted_profile=reviewer_hosted_profile,
        reviewer_ollama_model=reviewer_ollama_model,
        enabled_tools=enabled_tools,
        ollama_model=_ollama_model(),
        hosted_review_ready=_hosted_review_ready,
        tool_policy_runner_text=_tool_policy_runner_text,
    )


def _interactive_turn_user_message(
    user_query: str,
    today_str: str,
    second_opinion: bool,
    cloud: bool,
    *,
    primary_profile: Optional[LlmProfile] = None,
    reviewer_ollama_model: Optional[str] = None,
    reviewer_hosted_profile: Optional[LlmProfile] = None,
    enabled_tools: Optional[AbstractSet[str]] = None,
    system_instruction_override: Optional[str] = None,
    skill_suffix: Optional[str] = None,
) -> str:
    from agentlib.prompts import interactive_turn_user_message

    return interactive_turn_user_message(
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
        ollama_model=_ollama_model(),
        hosted_review_ready=_hosted_review_ready,
        tool_policy_runner_text=_tool_policy_runner_text,
    )


def _while_conversation_excerpt_for_judge(messages: list, max_chars: int = 12000) -> str:
    """Compact transcript text for condition judging (truncate per message)."""
    return repl_while_cmd.while_conversation_excerpt_for_judge(
        messages, max_chars, scalar_to_str_fn=_scalar_to_str
    )


_parse_while_judge_bit = repl_while_cmd.parse_while_judge_bit
_post_do_tokens_to_body_prompts = repl_while_cmd.post_do_tokens_to_body_prompts
_parse_while_repl_tokens = repl_while_cmd.parse_while_repl_tokens


def _call_while_condition_judge(
    condition: str,
    messages: list,
    *,
    primary_profile: Optional[LlmProfile],
    verbose: int,
) -> int:
    return repl_while_cmd.call_while_condition_judge(
        condition,
        messages,
        primary_profile=primary_profile,
        verbose=verbose,
        default_primary_llm_profile=default_primary_llm_profile,
        call_hosted_chat_plain=call_hosted_chat_plain,
        call_ollama_plaintext=call_ollama_plaintext,
        ollama_model=_ollama_model(),
        scalar_to_str_fn=_scalar_to_str,
    )


_SKILL_HELP_TEXT = (
    "/skill — run a skill\n\n"
    "A skill is a task-specific mode: it adds an “Active skill” prompt suffix and may narrow tools "
    "or run a multi-step workflow.\n\n"
    "Subcommands:\n"
    "  /skill list\n"
    "  /skill <id> <request>     Run a specific skill id (must exist in skills_dir)\n"
    "  /skill auto <request>     Ask the model to pick a skill, then run it\n"
    "  /skill reuse <request>    Reuse the last skill id\n\n"
    "See also: /settings prompt_template help\n"
)


_PROMPT_TEMPLATE_HELP_TEXT = (
    "/settings prompt_template — manage prompt templates\n\n"
    "A prompt_template is the session/run’s base system prompt (default behavior and tone for every turn).\n"
    "A skill (/skill ...) is a task-specific overlay that can also narrow tools and run workflows.\n\n"
    "Commands:\n"
    "  /settings prompt_template list\n"
    "  /settings prompt_template show\n"
    "  /settings prompt_template use <name>\n"
    "  /settings prompt_template default <name>\n"
    "  /settings prompt_template set <name> <text>\n"
    "  /settings prompt_template delete <name>\n\n"
    "See also: /skill help\n"
)


_SETTINGS_HELP_TEXT = (
    "/settings — session configuration\n\n"
    "Top-level:\n"
    "  /settings help\n"
    "  /settings save\n\n"
    "LLM selection:\n"
    "  /settings model <ollama-model-name>\n"
    "  /settings primary llm ollama|hosted ...\n"
    "  /settings second_opinion llm ollama|hosted ...\n\n"
    "Tools:\n"
    "  /settings tools\n"
    "  /settings enable <tool or phrase>\n"
    "  /settings disable <tool or phrase>\n\n"
    "Prompts:\n"
    "  /settings prompt_template help\n"
    "  /settings system_prompt show|reset|file|save|...\n\n"
    "Other:\n"
    "  /settings thinking show|on|off|level ...\n"
    "  /settings context show|on|off|...\n"
    "  /settings verbose 0|1|2|on|off\n\n"
    "JSON-backed settings groups (stored in ~/.agent.json via /settings ollama|openai|agent ...):\n"
    "  /settings ollama show|keys|set|unset\n"
    "  /settings openai show|keys|set|unset\n"
    "  /settings agent show|keys|set|unset\n\n"
    "Tip: use /skill help for skills.\n"
)
def _interactive_repl(
    *,
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
):
    """Multi-turn stdin loop when no query is given on the command line."""
    second_opinion_on = second_opinion_enabled
    session_save_path = save_context_path
    ptd0 = (prompt_templates_dir or "").strip()
    session_pt_dir = os.path.abspath(
        os.path.expanduser(ptd0) if ptd0 else _default_prompt_templates_dir()
    )
    skd0 = (skills_dir or "").strip()
    session_skills_dir = os.path.abspath(
        os.path.expanduser(skd0) if skd0 else _default_skills_dir()
    )
    tld0 = (tools_dir or "").strip()
    session_tools_dir = os.path.abspath(
        os.path.expanduser(tld0) if tld0 else _default_tools_dir()
    )
    skills_m = skills_map if isinstance(skills_map, dict) else {}
    templates = (
        prompt_templates
        if isinstance(prompt_templates, dict)
        else prompt_templates_io.load_prompt_templates_from_dir(
            _default_prompt_templates_dir()
        )
    )
    template_default = (prompt_template_default or "").strip() or "coding"
    session_prompt_template: Optional[str] = None
    session_system_prompt = system_prompt_override
    session_system_prompt_path = (
        os.path.abspath(os.path.expanduser(system_prompt_path))
        if (system_prompt_path or "").strip()
        else None
    )
    if session_system_prompt is None and not session_system_prompt_path:
        resolved = agent_prompts.resolve_prompt_template_text(template_default, templates)
        if resolved:
            session_system_prompt = resolved
            session_prompt_template = template_default

    # Context window manager session config (prefs defaults; env still overrides at runtime).
    context_cfg = context_cfg if isinstance(context_cfg, dict) else {}
    primary_profile = primary_profile or default_primary_llm_profile()
    enabled_tools = (
        set(enabled_tools) if enabled_tools is not None else set(_TOOL_REGISTRY.core_tools)
    )
    enabled_toolsets = set(enabled_toolsets) if enabled_toolsets is not None else set()

    from agentlib.session import AgentSession

    session = AgentSession(
        settings=_SETTINGS_OBJ,
        verbose=verbose,
        second_opinion_enabled=second_opinion_on,
        cloud_ai_enabled=cloud_ai_enabled,
        save_context_path=session_save_path,
        enabled_tools=frozenset(enabled_tools),
        enabled_toolsets=frozenset(enabled_toolsets),
        primary_profile=primary_profile,
        reviewer_hosted_profile=reviewer_hosted_profile,
        reviewer_ollama_model=reviewer_ollama_model,
        skills_map=skills_m,
        prompt_templates=templates,
        prompt_template_default=template_default,
        prompt_templates_dir=session_pt_dir,
        skills_dir=session_skills_dir,
        tools_dir=session_tools_dir,
        context_cfg=context_cfg,
        system_prompt_override=session_system_prompt,
        system_prompt_path=session_system_prompt_path,
        session_prompt_template=session_prompt_template,
        agent_progress=_agent_progress,
        fetch_ollama_local_model_names=_fetch_ollama_local_model_names,
        format_last_ollama_usage_for_repl=_format_last_ollama_usage_for_repl,
        format_session_primary_llm_line=_format_session_primary_llm_line,
        format_session_reviewer_line=_format_session_reviewer_line,
        print_skill_usage_verbose=_print_skill_usage_verbose,
        match_skill_detail=_match_skill_detail,
        ml_select_skill_id=_ml_select_skill_id,
        skill_plan_steps=_skill_plan_steps,
        effective_enabled_tools_for_skill=_effective_enabled_tools_for_skill,
        effective_enabled_tools_for_turn=_effective_enabled_tools_for_turn,
        route_requires_websearch=_route_requires_websearch,
        deliverable_skip_mandatory_web=_deliverable_skip_mandatory_web,
        user_wants_written_deliverable=_user_wants_written_deliverable,
        interactive_turn_user_message=_interactive_turn_user_message,
        conversation_turn_deps=_conversation_turn_deps(),
        save_context_bundle=_save_context_bundle,
        load_context_messages=_load_context_messages,
        registry=_TOOL_REGISTRY,
        build_agent_prefs_payload=_build_agent_prefs_payload,
        write_agent_prefs_file=_write_agent_prefs_file,
        agent_prefs_path=_agent_prefs_path,
        settings_group_keys_lines=_settings_group_keys_lines,
        settings_group_show=_settings_group_show,
        settings_group_set=_settings_group_set,
        settings_group_unset=_settings_group_unset,
        settings_get=_settings_get,
        settings_set=_settings_set,
        LlmProfile_cls=LlmProfile,
        default_primary_llm_profile=default_primary_llm_profile,
        describe_llm_profile_short=_describe_llm_profile_short,
        ollama_second_opinion_model=_ollama_second_opinion_model,
        ollama_request_think_value=_ollama_request_think_value,
        agent_thinking_level=_agent_thinking_level,
        agent_thinking_enabled_default_false=_agent_thinking_enabled_default_false,
        agent_stream_thinking_enabled=_agent_stream_thinking_enabled,
        verbose_ack_message=_verbose_ack_message,
        parse_while_repl_tokens=_parse_while_repl_tokens,
        call_while_condition_judge=_call_while_condition_judge,
    )

    run_interactive_repl_loop(
        session,
        install_readline=_interactive_repl_install_readline,
        repl_read_line=_repl_read_line,
        flush_repl_history=_flush_repl_readline_history,
        agent_progress=_agent_progress,
    )


_CACHED_CONVERSATION_TURN_DEPS: Optional[ConversationTurnDeps] = None


def _conversation_turn_deps() -> ConversationTurnDeps:
    """Lazily-built deps for `agentlib.runtime.run_agent_conversation_turn` (module wiring)."""
    global _CACHED_CONVERSATION_TURN_DEPS
    if _CACHED_CONVERSATION_TURN_DEPS is None:
        _CACHED_CONVERSATION_TURN_DEPS = ConversationTurnDeps(
            coerce_enabled_tools=_coerce_enabled_tools,
            maybe_compact_context_window=_maybe_compact_context_window,
            call_ollama_chat=call_ollama_chat,
            parse_agent_json=parse_agent_json,
            deliverable_followup_block=_deliverable_followup_block,
            answer_missing_written_body=_answer_missing_written_body,
            scalar_to_str=_scalar_to_str,
            hosted_review_ready=_hosted_review_ready,
            second_opinion_reviewer_messages=_second_opinion_reviewer_messages,
            second_opinion_result_user_message=_second_opinion_result_user_message,
            call_ollama_plaintext=call_ollama_plaintext,
            call_hosted_chat_plain=call_hosted_chat_plain,
            call_openai_chat_plain=call_openai_chat_plain,
            ollama_second_opinion_model=_ollama_second_opinion_model,
            route_requires_websearch_after_answer=_route_requires_websearch_after_answer,
            deliverable_skip_mandatory_web=_deliverable_skip_mandatory_web,
            deliverable_first_answer_followup=_deliverable_first_answer_followup,
            is_self_capability_question=_is_self_capability_question,
            self_capability_followup=_self_capability_followup,
            tool_need_review_followup=_tool_need_review_followup,
            extract_json_object_from_text=_extract_json_object_from_text,
            all_known_tools=_all_known_tools,
            merge_tool_param_aliases=_merge_tool_param_aliases,
            ensure_tool_defaults=_ensure_tool_defaults,
            tool_params_fingerprint=_tool_params_fingerprint,
            search_backend_banner_line=_search_backend_banner_line,
            search_web=lambda query, params=None: tool_builtins.search_web(
                query, params=params, settings=_SETTINGS_OBJ
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
            plugin_tool_handlers=_PLUGIN_TOOL_HANDLERS,
            tool_fault_result=_tool_fault_result,
            tool_recovery_may_run=_tool_recovery_may_run,
            tool_recovery_tools=_TOOL_RECOVERY_TOOLS,
            tool_result_indicates_retryable_failure=_tool_result_indicates_retryable_failure,
            suggest_tool_recovery_params=_suggest_tool_recovery_params,
            confirm_tool_recovery_retry=_confirm_tool_recovery_retry,
            agent_progress=_agent_progress,
            tool_progress_message=_tool_progress_message,
            is_tool_result_weak_for_dedup=_is_tool_result_weak_for_dedup,
            tool_result_user_message=_tool_result_user_message,
        )
    return _CACHED_CONVERSATION_TURN_DEPS


def main():
    argv = _parse_and_apply_cli_config_flag(list(sys.argv[1:]))
    raw_prefs = _load_agent_prefs()
    st = _session_defaults_from_prefs(raw_prefs)
    # Reload plugin toolsets after prefs/env are applied so AGENT_TOOLS_DIR / tools_dir override works.
    try:
        _TOOL_REGISTRY.load_plugin_toolsets(_resolved_tools_dir(raw_prefs))
        _TOOL_REGISTRY.register_aliases()
    except Exception:
        pass
    from agentlib.cli import parse_main_argv

    verbose0 = _coerce_verbose_level(st.get("verbose", 0))
    second_opinion0 = bool(st["second_opinion_enabled"])
    cloud0 = bool(st["cloud_ai_enabled"])
    enabled_tools0 = set(st["enabled_tools"])
    save_context_path0: Optional[str] = st["save_context_path"]
    primary_profile0 = st["primary_profile"]
    reviewer_hosted_profile: Optional[LlmProfile] = st["reviewer_hosted_profile"]
    reviewer_ollama_model: Optional[str] = st["reviewer_ollama_model"]

    parsed = parse_main_argv(
        argv,
        verbose=verbose0,
        second_opinion_enabled=second_opinion0,
        cloud_ai_enabled=cloud0,
        save_context_path=save_context_path0,
        enabled_tools=enabled_tools0,
        primary_profile=primary_profile0,
        reviewer_hosted_profile=reviewer_hosted_profile,
        reviewer_ollama_model=reviewer_ollama_model,
        strip_leading_dashes_flag=_strip_leading_dashes_flag,
        print_cli_help=_print_cli_help,
        apply_cli_primary_model=_apply_cli_primary_model,
        normalize_tool_name=_normalize_tool_name,
        format_unknown_tool_hint=_format_unknown_tool_hint,
        format_settings_tools_list=_format_settings_tools_list,
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

    prompt_templates = st.get("prompt_templates") if isinstance(st.get("prompt_templates"), dict) else (
        prompt_templates_io.load_prompt_templates_from_dir(_default_prompt_templates_dir())
    )
    prompt_template_default = (st.get("prompt_template_default") or "coding").strip()

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
        )
        return
    user_query = " ".join(query_parts)
    interactive_tool_recovery = sys.stdin.isatty() and sys.stdout.isatty()
    skills_map_cli = st.get("skills") or {}
    if not isinstance(skills_map_cli, dict):
        skills_map_cli = {}
    skill_id_cli, tr_cli = _match_skill_detail(user_query, skills_map_cli)
    et_oneshot0 = _effective_enabled_tools_for_skill(
        frozenset(enabled_tools), skills_map_cli, skill_id_cli
    )
    et_oneshot = _effective_enabled_tools_for_turn(
        base_enabled_tools=et_oneshot0,
        enabled_toolsets=st.get("enabled_toolsets") or set(),
        user_query=user_query,
    )
    if verbose >= 1:
        dcli = (
            f"trigger match: longest substring {tr_cli!r} (skill {skill_id_cli!r})"
            if skill_id_cli and tr_cli
            else "trigger match: no skill (no trigger substring matched)"
        )
        _print_skill_usage_verbose(
            verbose,
            source="cli",
            skill_id=skill_id_cli,
            base_tools=enabled_tools,
            effective_tools=et_oneshot,
            detail=dcli,
        )
    today = datetime.date.today()
    today_str = today.strftime("%Y-%m-%d (%A)")
    deliverable_wanted = _user_wants_written_deliverable(user_query)
    sys_prompt_override = st.get("system_prompt")
    if sys_prompt_override is None:
        chosen = (prompt_template_selected or prompt_template_default or "").strip()
        if chosen:
            resolved = agent_prompts.resolve_prompt_template_text(chosen, prompt_templates)
            if resolved:
                sys_prompt_override = resolved
            else:
                print(
                    f"Error: unknown or invalid prompt template {chosen!r}. "
                    "Use /settings prompt_template list (interactive) or define it in ~/.agent.json.",
                    file=sys.stderr,
                )
                return
    si0 = agent_prompts.effective_system_instruction_text(sys_prompt_override)
    if skill_id_cli:
        rec0 = skills_map_cli.get(skill_id_cli) or {}
        psk0 = str(rec0.get("prompt") or "").strip()
        if psk0:
            si0 = (
                si0
                + "\n\n--- Active skill: "
                + str(skill_id_cli)
                + " ---\n"
                + psk0
            )
    first_user = (
        f"{si0}\n\n"
        f"Today's date (system clock): {today_str}\n\n"
        f"User request: {user_query}\n\n"
        "Respond with JSON only. No other text."
    )
    ri = _runner_instruction_bits(
        second_opinion_enabled,
        cloud_ai_enabled,
        primary_profile=primary_profile,
        reviewer_ollama_model=reviewer_ollama_model,
        reviewer_hosted_profile=reviewer_hosted_profile,
        enabled_tools=et_oneshot,
    )
    if ri:
        first_user += "\n\n" + ri
    if load_context_path:
        try:
            messages = _load_context_messages(load_context_path)
        except OSError as e:
            print(f"Error loading context file: {e}")
            return
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error loading context: {e}")
            return
        cont = (
            f"Today's date (system clock): {today_str}\n\n"
            f"New user request:\n{user_query}\n\n"
            "Continue the conversation. Respond with JSON only. No other text."
        )
        if skill_id_cli:
            rec1 = skills_map_cli.get(skill_id_cli) or {}
            psk1 = str(rec1.get("prompt") or "").strip()
            if psk1:
                cont += (
                    "\n\n--- Active skill: "
                    + str(skill_id_cli)
                    + " ---\n"
                    + psk1
                )
        ri2 = _runner_instruction_bits(
            second_opinion_enabled,
            cloud_ai_enabled,
            primary_profile=primary_profile,
            reviewer_ollama_model=reviewer_ollama_model,
            reviewer_hosted_profile=reviewer_hosted_profile,
            enabled_tools=et_oneshot,
        )
        if ri2:
            cont += "\n\n" + ri2
        messages.append({"role": "user", "content": cont})
    else:
        messages = [{"role": "user", "content": first_user}]
    router_query = _route_requires_websearch(
        user_query,
        today_str,
        primary_profile,
        et_oneshot,
        transcript_messages=messages,
    )
    if _deliverable_skip_mandatory_web(user_query):
        router_query = None
    web_required = bool(router_query)
    if router_query and "search_web" in et_oneshot:
        # Force the first step to be a web search, but let the agent choose the exact tool_call JSON.
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
        _conversation_turn_deps(),
        web_required=web_required,
        deliverable_wanted=deliverable_wanted,
        verbose=verbose,
        second_opinion_enabled=second_opinion_enabled,
        cloud_ai_enabled=cloud_ai_enabled,
        primary_profile=primary_profile,
        reviewer_hosted_profile=reviewer_hosted_profile,
        reviewer_ollama_model=reviewer_ollama_model,
        enabled_tools=et_oneshot,
        interactive_tool_recovery=interactive_tool_recovery,
        context_cfg=st.get("context_manager"),
    )

    if save_context_path:
        try:
            _save_context_bundle(save_context_path, messages, user_query, final_answer, answered)
        except OSError as e:
            print(f"Warning: could not save context: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
