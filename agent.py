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
import html as html_module
from dataclasses import dataclass, replace
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


@dataclass
class LlmProfile:
    """Primary or reviewer endpoint: local Ollama or an OpenAI-compatible HTTPS API."""

    backend: str  # "ollama" | "hosted"
    base_url: str = ""
    model: str = ""
    api_key: str = ""


def default_primary_llm_profile() -> LlmProfile:
    return LlmProfile(backend="ollama")


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


def _read_api_key(env_name: str) -> str:
    # Environment variables are intentionally not used for settings; keep this helper name for
    # legacy call sites but interpret the argument as the key itself.
    return (env_name or "").strip()


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


# Core tools we implement in this script (Ollama may emit other native tool names — ignore those).
_CORE_TOOLS = frozenset(
    {
        "search_web",
        "fetch_page",
        "run_command",
        "use_git",
        "write_file",
        "read_file",
        "list_directory",
        "download_file",
        "tail_file",
        "replace_text",
        "call_python",
    }
)


def _all_known_tools() -> frozenset[str]:
    """Core tools + currently loaded plugin tools."""
    return frozenset(set(_CORE_TOOLS) | set(_PLUGIN_TOOL_HANDLERS.keys()))

# (internal_id, short description, user-facing aliases — spaces and hyphens ok)
_TOOL_ENTRIES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    (
        "search_web",
        "Web search (DuckDuckGo)",
        (
            "web",
            "web search",
            "internet search",
            "duckduckgo",
            "ddg",
            "lookup",
        ),
    ),
    (
        "fetch_page",
        "Fetch a URL / read HTML",
        ("fetch", "fetch page", "read page", "http get", "url fetch", "get url"),
    ),
    (
        "run_command",
        "Run a shell command",
        ("shell", "terminal", "bash", "sh", "command line", "cmd", "subprocess"),
    ),
    (
        "use_git",
        "Git operations (vetted; prefer over raw shell for git)",
        ("git", "commit", "push", "pull", "diff", "rebase", "merge", "clone"),
    ),
    (
        "write_file",
        "Write or overwrite a file",
        ("write", "save file", "create file", "file write"),
    ),
    ("read_file", "Read a file", ("read", "cat", "open file", "slurp")),
    (
        "list_directory",
        "List directory contents",
        ("ls", "dir", "list dir", "folder", "directory"),
    ),
    ("download_file", "Download a file from URL", ("download", "wget", "curl download")),
    ("tail_file", "Read end of a file (tail)", ("tail", "log tail")),
    (
        "replace_text",
        "Search-and-replace in a file",
        ("replace", "search replace", "sed"),
    ),
    ("call_python", "Run Python code in-process", ("python", "py", "eval", "code eval")),
)


from agentlib.tools import plugins as _tool_plugins
from agentlib.tools import routing as _tool_routing

# Plugin toolsets (loaded from tools/ directory).
# Toolsets are off by default and can be enabled by the user.
_PLUGIN_TOOLSETS = _tool_plugins.PLUGIN_TOOLSETS
_PLUGIN_TOOL_HANDLERS = _tool_plugins.PLUGIN_TOOL_HANDLERS
_PLUGIN_TOOL_TO_TOOLSET = _tool_plugins.PLUGIN_TOOL_TO_TOOLSET
_PLUGIN_TOOLSET_TRIGGERS = _tool_plugins.PLUGIN_TOOLSET_TRIGGERS


def _load_plugin_toolsets(tools_dir: Optional[str] = None) -> None:
    _tool_plugins.load_plugin_toolsets(
        tools_dir=tools_dir,
        default_tools_dir=_default_tools_dir(),
    )


def _plugin_tool_entries() -> Tuple[Tuple[str, str, Tuple[str, ...]], ...]:
    return _tool_plugins.plugin_tool_entries()

_TOOL_ALIASES = _tool_routing.TOOL_ALIASES


def _canonicalize_user_tool_phrase(phrase: str) -> str:
    return _tool_routing.canonicalize_user_tool_phrase(phrase)


def _register_tool_aliases() -> None:
    _tool_routing.register_tool_aliases()





def _coerce_enabled_tools(ets: Optional[AbstractSet[str]]):
    """None means all tools enabled (default)."""
    if ets is None:
        return _all_known_tools()
    return frozenset(ets)


def _resolve_tool_token(phrase: str) -> Optional[str]:
    return _tool_routing.resolve_tool_token(phrase)


def _normalize_tool_name(token: str) -> Optional[str]:
    """Map user text or internal id to canonical tool name."""
    return _tool_routing.normalize_tool_name(token)


def _plugin_tools_for_toolset(toolset: str) -> set[str]:
    return _tool_plugins.plugin_tools_for_toolset(toolset)


def _route_active_toolsets_for_request(user_query: str, enabled_toolsets: AbstractSet[str]) -> set[str]:
    return _tool_routing.route_active_toolsets_for_request(user_query, enabled_toolsets)


def _effective_enabled_tools_for_turn(
    *,
    base_enabled_tools: AbstractSet[str],
    enabled_toolsets: AbstractSet[str],
    user_query: str,
) -> frozenset[str]:
    return _tool_routing.effective_enabled_tools_for_turn(
        base_enabled_tools=base_enabled_tools,
        enabled_toolsets=enabled_toolsets,
        user_query=user_query,
    )


def _all_tool_name_suggestion_pool() -> list[str]:
    pool = set(_all_known_tools())
    pool.update(_TOOL_ALIASES.keys())
    pool.add("second_opinion")
    return sorted(pool)


def _format_unknown_tool_hint(phrase: str) -> str:
    return _tool_routing.format_unknown_tool_hint(phrase)


def _format_settings_tools_list(enabled_tools: AbstractSet[str]) -> str:
    return _tool_routing.format_settings_tools_list(enabled_tools)


def _describe_tool_call_contract(tool_id: str) -> str:
    return _tool_routing.describe_tool_call_contract(tool_id)


def _tool_policy_runner_text(ets: Optional[AbstractSet[str]]) -> str:
    return _tool_routing.tool_policy_runner_text(ets)


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


def _scalar_to_str(value, default=""):
    """Coerce tool parameters to str (models may emit numbers, lists, etc.)."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        parts = [_scalar_to_str(x, "") for x in value]
        parts = [p for p in parts if p]
        return " ".join(parts) if parts else default
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _scalar_to_int(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _coerce_verbose_level(v) -> int:
    """0 = off, 1 = log tool invocations, 2 = log tools + stream model JSON (local Ollama)."""
    if isinstance(v, bool):
        return 2 if v else 0
    if v is None:
        return 0
    n = _scalar_to_int(v, 0)
    if n < 0:
        return 0
    if n > 2:
        return 2
    return n


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


# Load bundled plugin toolsets (default tools_dir) at import time.
# If tools_dir is overridden in prefs/env, main() / _session_defaults_from_prefs will reload.
_load_plugin_toolsets(_default_tools_dir())
_register_tool_aliases()


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


def _load_prompt_templates_from_dir(dir_path: str) -> dict:
    out: dict = {}
    if not os.path.isdir(dir_path):
        return out
    for fn in sorted(os.listdir(dir_path)):
        if not fn.endswith(".json") or fn.startswith("."):
            continue
        name, _ = os.path.splitext(fn)
        name = (name or "").strip()
        if not name:
            continue
        path = os.path.join(dir_path, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(obj, dict):
            out[name] = obj
    return out


def _safe_path_under_dir(base_dir: str, relpath: str) -> Optional[str]:
    from agentlib.skills.loader import safe_path_under_dir

    return safe_path_under_dir(base_dir, relpath)


def _expand_skill_artifacts(skills_dir: str, meta: dict, base_prompt: str) -> str:
    from agentlib.skills.loader import expand_skill_artifacts

    return expand_skill_artifacts(skills_dir, meta, base_prompt)


def _load_skills_from_dir(dir_path: str) -> dict:
    from agentlib.skills.loader import load_skills_from_dir

    return load_skills_from_dir(dir_path)


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


def _settings_group_keys_lines(group: str) -> str:
    return _SETTINGS_OBJ.group_keys_lines(group)


def _settings_group_show(group: str) -> str:
    return _SETTINGS_OBJ.group_show(group)


def _settings_group_set(group: str, raw_key: str, raw_value: str) -> str:
    return _SETTINGS_OBJ.group_set(group, raw_key, raw_value)


def _settings_group_unset(group: str, raw_key: str) -> str:
    return _SETTINGS_OBJ.group_unset(group, raw_key)


def _llm_profile_to_pref(profile: LlmProfile) -> dict:
    if profile.backend != "hosted":
        return {"backend": "ollama"}
    d: dict = {
        "backend": "hosted",
        "base_url": profile.base_url,
        "model": profile.model,
    }
    if (profile.api_key or "").strip():
        d["api_key"] = profile.api_key.strip()
    return d


def _llm_profile_from_pref(obj: object) -> Optional[LlmProfile]:
    if not isinstance(obj, dict):
        return None
    bk = _scalar_to_str(obj.get("backend"), "").strip().lower()
    if bk == "ollama":
        return LlmProfile(backend="ollama")
    if bk != "hosted":
        return None
    bu = _scalar_to_str(obj.get("base_url"), "").strip().rstrip("/")
    mod = _scalar_to_str(obj.get("model"), "").strip()
    if not bu.startswith(("http://", "https://")) or not mod:
        return None
    prof = LlmProfile(backend="hosted", base_url=bu, model=mod)
    ak = obj.get("api_key")
    if isinstance(ak, str) and ak.strip():
        prof.api_key = ak.strip()
    return prof


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
    payload: dict = {
        "version": _AGENT_PREFS_VERSION,
        "second_opinion_enabled": bool(second_opinion_on),
        "cloud_ai_enabled": bool(cloud_ai_enabled),
        "verbose": _coerce_verbose_level(verbose_level),
        "primary_llm": _llm_profile_to_pref(primary_profile),
        "enabled_tools": sorted(enabled_tools)
        if len(enabled_tools) < len(_CORE_TOOLS)
        else None,
    }
    # Persist JSON-backed settings groups (lowercase keys).
    payload.update(_SETTINGS_OBJ.as_groups_dict())
    if reviewer_hosted_profile is not None and reviewer_hosted_profile.backend == "hosted":
        payload["second_opinion_reviewer"] = _llm_profile_to_pref(reviewer_hosted_profile)
    else:
        rev: dict = {"backend": "ollama"}
        if reviewer_ollama_model and str(reviewer_ollama_model).strip():
            rev["ollama_model"] = str(reviewer_ollama_model).strip()
        payload["second_opinion_reviewer"] = rev
    if session_save_path and str(session_save_path).strip():
        payload["save_context_path"] = str(session_save_path).strip()
    payload["system_prompt"] = None
    payload["system_prompt_path"] = None
    spp = (system_prompt_path_override or "").strip()
    if spp:
        payload["system_prompt_path"] = os.path.abspath(os.path.expanduser(spp))
    elif system_prompt_override is not None and str(system_prompt_override).strip():
        payload["system_prompt"] = str(system_prompt_override)
    if prompt_templates is not None:
        payload["prompt_templates"] = prompt_templates
    if prompt_template_default is not None:
        payload["prompt_template_default"] = str(prompt_template_default).strip() or None
    ptd = (prompt_templates_dir or "").strip()
    if ptd:
        payload["prompt_templates_dir"] = os.path.abspath(os.path.expanduser(ptd))
    skd = (skills_dir or "").strip()
    if skd:
        payload["skills_dir"] = os.path.abspath(os.path.expanduser(skd))
    tld = (tools_dir or "").strip()
    if tld:
        payload["tools_dir"] = os.path.abspath(os.path.expanduser(tld))
    if context_manager is not None:
        payload["context_manager"] = context_manager
    ets = {str(x).strip().lower() for x in (enabled_toolsets or set()) if str(x).strip()}
    ets = {x for x in ets if x in _PLUGIN_TOOLSETS}
    payload["enabled_toolsets"] = sorted(ets) if ets else None
    return payload


def _session_defaults_from_prefs(prefs: Optional[dict]) -> dict:
    if isinstance(prefs, dict):
        _migrate_settings_groups_from_prefs(prefs)
    # Ensure plugin toolsets are loaded from the resolved tools_dir for this prefs object.
    try:
        _load_plugin_toolsets(_resolved_tools_dir(prefs))
        _register_tool_aliases()
    except Exception:
        pass
    _pt = _default_prompt_templates_dir()
    _sk = _default_skills_dir()
    out = {
        "enabled_tools": set(_CORE_TOOLS),
        "enabled_toolsets": set(),
        "second_opinion_enabled": False,
        "cloud_ai_enabled": False,
        "verbose": 0,
        "primary_profile": default_primary_llm_profile(),
        "reviewer_hosted_profile": None,
        "reviewer_ollama_model": None,
        "save_context_path": None,
        "system_prompt": None,
        "system_prompt_path": None,
        "prompt_templates_dir": _pt,
        "skills_dir": _sk,
        "tools_dir": _resolved_tools_dir(prefs),
        "prompt_templates": _merge_prompt_templates(None),
        "skills": _load_skills_from_dir(_resolved_skills_dir(None)),
        "prompt_template_default": "coding",
        "context_manager": {
            "enabled": True,
            "tokens": 0,
            "trigger_frac": 0.75,
            "target_frac": 0.55,
            "keep_tail_messages": 12,
        },
    }
    if not prefs or not isinstance(prefs, dict):
        return out
    ver = _scalar_to_int(prefs.get("version"), _AGENT_PREFS_VERSION)
    if ver > _AGENT_PREFS_VERSION:
        return out
    if isinstance(prefs.get("second_opinion_enabled"), bool):
        out["second_opinion_enabled"] = prefs["second_opinion_enabled"]
    if isinstance(prefs.get("cloud_ai_enabled"), bool):
        out["cloud_ai_enabled"] = prefs["cloud_ai_enabled"]
    if "verbose" in prefs:
        out["verbose"] = _coerce_verbose_level(prefs.get("verbose"))
    pl = prefs.get("primary_llm")
    if isinstance(pl, dict):
        pp = _llm_profile_from_pref(pl)
        if pp:
            out["primary_profile"] = pp
    raw_et = prefs.get("enabled_tools")
    if isinstance(raw_et, list):
        et = set()
        for t in raw_et:
            tn = _normalize_tool_name(str(t))
            if tn:
                et.add(tn)
        if et:
            out["enabled_tools"] = et
    raw_ts = prefs.get("enabled_toolsets")
    if isinstance(raw_ts, list):
        ts: set[str] = set()
        for one in raw_ts:
            nm = str(one).strip().lower()
            if nm and nm in _PLUGIN_TOOLSETS:
                ts.add(nm)
        out["enabled_toolsets"] = ts
    rev = prefs.get("second_opinion_reviewer")
    if isinstance(rev, dict):
        rb = _scalar_to_str(rev.get("backend"), "").strip().lower()
        if rb == "hosted":
            hp = _llm_profile_from_pref(rev)
            if hp and hp.backend == "hosted":
                out["reviewer_hosted_profile"] = hp
                out["reviewer_ollama_model"] = None
        elif rb == "ollama":
            out["reviewer_hosted_profile"] = None
            rom = rev.get("ollama_model")
            if isinstance(rom, str) and rom.strip():
                out["reviewer_ollama_model"] = rom.strip()
            else:
                out["reviewer_ollama_model"] = None
    scp = prefs.get("save_context_path")
    if isinstance(scp, str) and scp.strip():
        out["save_context_path"] = scp.strip()
    spp = prefs.get("system_prompt_path")
    if isinstance(spp, str) and spp.strip():
        path = os.path.expanduser(spp.strip())
        out["system_prompt_path"] = path
        try:
            with open(path, "r", encoding="utf-8") as f:
                out["system_prompt"] = f.read()
        except OSError:
            out["system_prompt"] = None
            out["system_prompt_path"] = None
    elif isinstance(prefs.get("system_prompt"), str):
        sp = prefs["system_prompt"]
        if sp.strip():
            out["system_prompt"] = sp
    out["prompt_templates_dir"] = _resolved_prompt_templates_dir(prefs)
    out["skills_dir"] = _resolved_skills_dir(prefs)
    out["prompt_templates"] = _merge_prompt_templates(prefs)
    out["skills"] = _load_skills_from_dir(out["skills_dir"])
    out["tools_dir"] = _resolved_tools_dir(prefs)
    ptd = prefs.get("prompt_template_default")
    if isinstance(ptd, str) and ptd.strip():
        out["prompt_template_default"] = ptd.strip()
    cm = prefs.get("context_manager")
    if isinstance(cm, dict):
        merged = dict(out["context_manager"])
        for k in ("enabled", "tokens", "trigger_frac", "target_frac", "keep_tail_messages"):
            if k in cm:
                merged[k] = cm[k]
        out["context_manager"] = merged
    return out


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
    p = dict(params) if isinstance(params, dict) else {}
    if tool == "search_web":
        if not _scalar_to_str(p.get("query"), "").strip():
            for alt in ("q", "search", "keywords", "keyword", "text"):
                t = _scalar_to_str(p.get(alt), "").strip()
                if t:
                    p["query"] = t
                    p.pop(alt, None)
                    break
    elif tool == "fetch_page":
        if not _scalar_to_str(p.get("url"), "").strip():
            for alt in ("href", "link", "uri"):
                t = _scalar_to_str(p.get(alt), "").strip()
                if t:
                    p["url"] = t
                    p.pop(alt, None)
                    break
    elif tool == "run_command":
        if not _scalar_to_str(p.get("command"), "").strip():
            for alt in ("cmd", "shell", "line"):
                t = _scalar_to_str(p.get(alt), "").strip()
                if t:
                    p["command"] = t
                    p.pop(alt, None)
                    break
    elif tool == "use_git":
        if not _scalar_to_str(p.get("op"), "").strip():
            for alt in ("operation", "git_op", "subcommand", "sub_cmd"):
                t = _scalar_to_str(p.get(alt), "").strip()
                if t:
                    p["op"] = t
                    p.pop(alt, None)
                    break
        if not _scalar_to_str(p.get("worktree"), "").strip():
            for alt in ("cwd", "repo", "work_tree", "path_root"):
                t = _scalar_to_str(p.get(alt), "").strip()
                if t:
                    p["worktree"] = t
                    p.pop(alt, None)
                    break
        if not _scalar_to_str(p.get("message"), "").strip():
            for alt in ("m", "msg", "commit_message"):
                t = _scalar_to_str(p.get(alt), "").strip()
                if t:
                    p["message"] = t
                    p.pop(alt, None)
                    break
        if p.get("paths") is None:
            for alt in ("files", "file", "file_paths"):
                if p.get(alt) is not None:
                    p["paths"] = p.get(alt)
                    p.pop(alt, None)
                    break
    elif tool == "write_file":
        if not _scalar_to_str(p.get("content"), "").strip():
            for alt in ("body", "text", "contents", "data"):
                t = _scalar_to_str(p.get(alt), "").strip()
                if t:
                    p["content"] = t
                    p.pop(alt, None)
                    break
    return p


def _ensure_tool_defaults(tool: str, params: dict, user_query: str) -> dict:
    """Fill required parameters when the model emits an empty object."""
    p = dict(params) if isinstance(params, dict) else {}
    uq = (user_query or "").strip()
    if tool == "search_web":
        if not _scalar_to_str(p.get("query"), "").strip():
            p["query"] = uq if uq else "web search"
    return p


def _enrich_search_query_for_present_day(query: str) -> str:
    from agentlib.tools.websearch import enrich_search_query_for_present_day

    return enrich_search_query_for_present_day(query, settings=_SETTINGS_OBJ)


def _merge_tool_arguments_delta(old_a, new_a):
    """Combine streamed `arguments` chunks without duplicating full JSON objects."""
    if old_a is None:
        return new_a
    if new_a is None:
        return old_a
    if isinstance(old_a, dict) and isinstance(new_a, dict):
        return {**old_a, **new_a}
    if isinstance(old_a, str) and isinstance(new_a, str):
        o, n = old_a.strip(), new_a.strip()
        if not o:
            return new_a
        if not n:
            return old_a
        merged = old_a + new_a
        for cand in (merged, new_a, old_a):
            try:
                json.loads(cand)
                return cand
            except json.JSONDecodeError:
                continue
        return merged
    if isinstance(new_a, dict):
        return new_a
    return old_a


def _merge_partial_tool_calls(prev, new):
    """Merge streaming tool_call fragments (Ollama/OpenAI-style deltas)."""
    if not new:
        return prev or []
    if not prev:
        return new
    by_idx = {}
    for tc in prev:
        i = tc.get("index", 0)
        by_idx[i] = tc
    for tc in new:
        i = tc.get("index", 0)
        if i not in by_idx:
            by_idx[i] = tc
            continue
        old = by_idx[i]
        fn_old = (old.get("function") or {}) if isinstance(old.get("function"), dict) else {}
        fn_new = (tc.get("function") or {}) if isinstance(tc.get("function"), dict) else {}
        name = (fn_new.get("name") or fn_old.get("name") or "").strip()
        merged_args = _merge_tool_arguments_delta(fn_old.get("arguments"), fn_new.get("arguments"))
        by_idx[i] = {
            **old,
            **tc,
            "function": {
                **fn_old,
                **fn_new,
                "name": name,
                "arguments": merged_args,
            },
        }
    return [by_idx[k] for k in sorted(by_idx.keys())]


def _ollama_usage_from_chat_response(data: dict) -> Optional[dict]:
    """Extract token/duration stats Ollama includes on /api/chat (esp. final stream chunk)."""
    if not isinstance(data, dict):
        return None
    out: dict = {}
    for k in ("prompt_eval_count", "eval_count"):
        v = data.get(k)
        if isinstance(v, int) and v >= 0:
            out[k] = v
    for k in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
        v = data.get(k)
        if isinstance(v, int) and v >= 0:
            out[k] = v
    return out or None


def _ollama_eval_generation_tok_per_sec(usage: dict) -> Optional[float]:
    """Tokens generated per second during the eval (decode) phase; needs eval_duration from Ollama."""
    n = usage.get("eval_count")
    dt_ns = usage.get("eval_duration")
    if not isinstance(n, int) or n < 0:
        return None
    if not isinstance(dt_ns, int) or dt_ns <= 0:
        return None
    return n / (dt_ns / 1e9)


def _format_ollama_usage_line(usage: dict) -> str:
    parts = []
    # Ollama names these fields; they are not the same contract as OpenAI "prompt_tokens"/"completion_tokens".
    if "prompt_eval_count" in usage:
        parts.append(f"prompt_eval_count={usage['prompt_eval_count']}")
    if "eval_count" in usage:
        parts.append(f"eval_count={usage['eval_count']}")
    rate = _ollama_eval_generation_tok_per_sec(usage)
    if rate is not None:
        parts.append(f"gen_tok/s≈{rate:.1f}")
    for key, label in (
        ("total_duration", "total"),
        ("load_duration", "load"),
        ("prompt_eval_duration", "prompt"),
        ("eval_duration", "gen"),
    ):
        if key in usage:
            parts.append(f"{label}_s={usage[key] / 1e9:.3f}")
    return "[Ollama usage] " + ", ".join(parts) if parts else "[Ollama usage] (no counts in response)"


_last_ollama_chat_usage: Optional[dict] = None


def _format_last_ollama_usage_for_repl() -> str:
    """Human-readable report for /usage (last local Ollama agent chat only)."""
    if _last_ollama_chat_usage is None:
        return (
            "No Ollama usage captured yet. Stats come from the local primary model’s last "
            "/api/chat response (not hosted APIs). After a turn, try again, or use "
            "/settings verbose 2 to print usage after each Ollama call (level 2)."
        )
    return (
        _format_ollama_usage_line(_last_ollama_chat_usage)
        + "\n(Ollama: prompt_eval_count / eval_count — not identical to OpenAI-style prompt/completion tokens; "
        "gen_tok/s uses eval_count ÷ eval_duration when both are present.)"
    )


def _merge_stream_message_chunks(lines_iter, *, stream_chunks: bool = False):
    """Merge streaming /api/chat chunks into one assistant message dict + final usage dict.

    When ``stream_chunks`` is True, prints each ``message.content`` delta as it arrives (the raw JSON
    string fragments Ollama streams for ``format: json``). Returns whether any content was printed.
    """
    acc = {"role": "assistant", "content": "", "thinking": ""}
    tool_calls = None
    usage: Optional[dict] = None
    streamed_content = False
    show_thinking = _agent_stream_thinking_enabled()
    thinking_started = False
    done_thinking_banner_printed = False
    for line in lines_iter:
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = data.get("message") or {}
        if msg.get("content"):
            chunk = msg["content"]
            if show_thinking and thinking_started and not done_thinking_banner_printed:
                # Make the transition visually obvious: thinking ended and content began.
                print("\n\n[Done thinking]\n", end="", flush=True)
                done_thinking_banner_printed = True
            if stream_chunks:
                print(chunk, end="", flush=True)
                streamed_content = True
            acc["content"] += chunk
        if msg.get("thinking"):
            tchunk = msg["thinking"]
            acc["thinking"] += tchunk
            if show_thinking:
                if not thinking_started:
                    print("\n[Thinking]\n", end="", flush=True)
                    thinking_started = True
                print(tchunk, end="", flush=True)
        if msg.get("tool_calls"):
            tool_calls = _merge_partial_tool_calls(tool_calls, msg["tool_calls"])
        if data.get("done"):
            u = _ollama_usage_from_chat_response(data)
            if u:
                usage = u
            break
    if tool_calls is not None:
        acc["tool_calls"] = tool_calls
    return acc, usage, streamed_content


def _parse_tool_arguments(arguments):
    """Normalize tool `arguments` from Ollama (str, dict, or malformed JSON)."""
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        s = arguments.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
        # Trailing commas / truncated fragments
        for fix in (s, re.sub(r",\s*}", "}", s), re.sub(r",\s*]", "]", s)):
            try:
                parsed = json.loads(fix)
                return dict(parsed) if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                continue
        # Single quotes used like Python dict (best-effort)
        if "'" in s and '"' not in s:
            try:
                parsed = json.loads(s.replace("'", '"'))
                return dict(parsed) if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


def _tool_call_to_agent_dict(function_name: str, arguments):
    """Map Ollama native tool_calls to our agent JSON shape."""
    raw = _parse_tool_arguments(arguments)
    if not raw:
        raw = {}
    if raw.get("action") == "tool_call" and raw.get("tool"):
        return {
            "action": "tool_call",
            "tool": raw["tool"],
            "parameters": raw.get("parameters") or {},
        }
    name = (function_name or "").strip()
    if name == "tool_call" and raw.get("tool"):
        return {
            "action": "tool_call",
            "tool": raw["tool"],
            "parameters": raw.get("parameters") or {},
        }
    params = dict(raw)
    if name.startswith("tool."):
        tool = name.split(".", 1)[1]
    elif name.startswith("functions.") or name.startswith("function."):
        tool = name.split(".", 1)[1]
    else:
        tool = name or "unknown"
    if "filename" in params and "path" not in params:
        params["path"] = params.pop("filename")
    return {"action": "tool_call", "tool": tool, "parameters": params}


def _tool_calls_to_agent_json_text(
    tool_calls, enabled_tools: Optional[AbstractSet[str]] = None
) -> Optional[str]:
    """Pick the first native tool_call that maps to a known, session-enabled tool."""
    et = _coerce_enabled_tools(enabled_tools)
    if not tool_calls:
        return None
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        args = fn.get("arguments")
        mapped = _tool_call_to_agent_dict(name, args)
        t = mapped.get("tool") if mapped else None
        if mapped and t in _all_known_tools() and t in et:
            return json.dumps(mapped)
    return None


def _iter_balanced_brace_objects(text: str):
    """Yield `{...}` spans with brace depth matching outside of JSON strings (double-quoted)."""
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, c in enumerate(text):
        if escape:
            escape = False
            continue
        if in_str:
            if c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : i + 1]
                    start = -1


def _normalize_unicode_json_quotes(s: str) -> str:
    """Map common Unicode “smart” double quotes to ASCII so delimiters parse as JSON."""
    if not s:
        return s
    trans = str.maketrans(
        {
            "\u201c": '"',
            "\u201d": '"',
            "\u201e": '"',
            "\u2033": '"',
            "\uff02": '"',
        }
    )
    return s.translate(trans)


def _escape_controls_inside_json_strings(s: str) -> str:
    """
    RFC 8259 forbids literal ASCII control characters inside JSON strings.
    Many LLMs emit literal newlines/tabs inside \"answer\" values anyway — escape them so json.loads succeeds.
    Tracks double-quoted regions using the same rules as _iter_balanced_brace_objects.
    """
    if not s:
        return s
    out: list[str] = []
    in_str = False
    escape = False
    for c in s:
        if escape:
            escape = False
            out.append(c)
            continue
        if in_str:
            if c == "\\":
                escape = True
                out.append(c)
                continue
            if c == '"':
                in_str = False
                out.append(c)
                continue
            o = ord(c)
            if o < 32:
                if c == "\n":
                    out.append("\\n")
                elif c == "\r":
                    out.append("\\r")
                elif c == "\t":
                    out.append("\\t")
                elif c == "\f":
                    out.append("\\f")
                elif c == "\b":
                    out.append("\\b")
                else:
                    out.append("\\u%04x" % o)
                continue
            out.append(c)
            continue
        if c == '"':
            in_str = True
        out.append(c)
    return "".join(out)


def _fallback_extract_answer_field(raw: str) -> Optional[str]:
    """
    Last resort when JSON is too broken to parse: scan for \"answer\"\\s*:\\s*\" and read a
    double-quoted string with standard JSON-style escapes (handles embedded quotes and newlines badly; good enough).
    """
    text = _normalize_unicode_json_quotes(raw.strip())
    # Avoid scraping unrelated JSON fragments that mention "answer" as a word or nested keys.
    if not re.search(r'"action"\s*:\s*"answer"', text):
        return None
    m = re.search(r'"answer"\s*:\s*"', text)
    if not m:
        return None
    i = m.end()
    buf: list[str] = []
    escape = False
    while i < len(text):
        c = text[i]
        if escape:
            escape = False
            # Map common JSON escapes back to literal text for the answer we return.
            if c == "n":
                buf.append("\n")
            elif c == "r":
                buf.append("\r")
            elif c == "t":
                buf.append("\t")
            elif c == '"':
                buf.append('"')
            elif c == "\\":
                buf.append("\\")
            elif c == "/":
                buf.append("/")
            elif c == "f":
                buf.append("\f")
            elif c == "b":
                buf.append("\b")
            elif c == "u" and i + 4 < len(text):
                hex_part = text[i + 1 : i + 5]
                if len(hex_part) == 4 and all(ch in "0123456789abcdefABCDEF" for ch in hex_part):
                    buf.append(chr(int(hex_part, 16)))
                    i += 4
                else:
                    buf.append("\\")
                    buf.append(c)
            else:
                buf.append("\\")
                buf.append(c)
            i += 1
            continue
        if c == "\\":
            escape = True
            i += 1
            continue
        if c == '"':
            return "".join(buf)
        buf.append(c)
        i += 1
    return None


def _try_json_loads_object(s: str):
    """Parse a JSON object string; apply light repairs for common model mistakes."""
    if not s or not s.strip():
        return None
    s = s.strip()

    variants = [
        s,
        _normalize_unicode_json_quotes(s),
        _escape_controls_inside_json_strings(s),
        _escape_controls_inside_json_strings(_normalize_unicode_json_quotes(s)),
    ]
    seen = set()
    uniq_variants = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            uniq_variants.append(v)

    for cand in uniq_variants:
        fixes = (
            cand,
            re.sub(r",\s*}", "}", cand),
            re.sub(r",\s*]", "]", cand),
        )
        for fix in fixes:
            try:
                v = json.loads(fix)
                if isinstance(v, dict):
                    return v
            except json.JSONDecodeError:
                continue

    extracted = _fallback_extract_answer_field(_normalize_unicode_json_quotes(s))
    if extracted is not None and extracted.strip():
        return {"action": "answer", "answer": extracted}

    return None


def _best_agent_dict_from_text(text: str) -> Optional[dict]:
    """
    Find the best JSON object in mixed prose: prefer dicts with 'action',
    then dicts whose tool name is known.
    """
    if not text or not text.strip():
        return None
    candidates = []
    for span in _iter_balanced_brace_objects(text):
        parsed = _try_json_loads_object(span)
        if isinstance(parsed, dict):
            candidates.append(parsed)
    if not candidates:
        return None

    def score(d: dict) -> tuple:
        action = d.get("action")
        tool = d.get("tool") or (action if action in _all_known_tools() else None)
        has_action = 1 if action else 0
        known_tool = 1 if tool in _all_known_tools() else 0
        tool_call_shape = 1 if action == "tool_call" and tool in _all_known_tools() else 0
        return (tool_call_shape, known_tool, has_action)

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def _extract_json_object_from_text(text: str) -> Optional[str]:
    """If the model buried JSON in prose/thinking, pull out the best object span."""
    best = _best_agent_dict_from_text(text)
    if not best:
        return None
    try:
        return json.dumps(best)
    except (TypeError, ValueError):
        return None


def _message_to_agent_json_text(
    msg: dict, enabled_tools: Optional[AbstractSet[str]] = None
) -> str:
    """Build a JSON string for parse_agent_json from an Ollama chat message."""
    # Prefer structured native tool_calls when present (works even if content is noise).
    from_tools = _tool_calls_to_agent_json_text(
        msg.get("tool_calls"), enabled_tools
    )
    if from_tools:
        return from_tools

    text = (msg.get("content") or "").strip()
    if text:
        extracted = _extract_json_object_from_text(text)
        if extracted:
            return extracted
        best = _best_agent_dict_from_text(text)
        if best:
            try:
                return json.dumps(best)
            except (TypeError, ValueError):
                pass
        return text

    thinking = (msg.get("thinking") or "").strip()
    if thinking:
        extracted = _extract_json_object_from_text(thinking)
        if extracted:
            return extracted
        best = _best_agent_dict_from_text(thinking)
        if best:
            try:
                return json.dumps(best)
            except (TypeError, ValueError):
                pass
        return thinking
    return ""


def _tool_params_fingerprint(tool: str, params) -> str:
    """Stable key for deduplicating identical tool calls."""
    if not isinstance(params, dict):
        params = {}
    # Models often add extra keys (max_results, engine, …) that bypassed dedupe; only the query matters.
    if tool == "search_web":
        q = _scalar_to_str(params.get("query"), "").strip()
        qn = re.sub(r"\s+", " ", q).lower().strip(" \t.?!")
        mrx = _search_web_effective_max_results(params)
        return f"{tool}\0{json.dumps({'query': qn, 'max_results': mrx}, sort_keys=True, ensure_ascii=False)}"
    return f"{tool}\0{json.dumps(params, sort_keys=True, ensure_ascii=False)}"


def _web_tool_result_followup_hint(tool: str, result: str) -> str:
    """
    Short, tool-specific nudge when web tools return errors or no usable data,
    so the model can recover without an extra model round in some clients.
    """
    r = (result or "").strip()
    if not r:
        return ""
    t = (tool or "").strip().lower()
    if t == "fetch_page":
        if "Fetch error:" in r[:400] or r.startswith("Fetch error:"):
            return (
                "The page fetch did not return usable content. If HTTP 4xx/5xx: try another "
                "URL (official docs, archive, or a different path). Use search_web to find a "
                "credible page if you do not have a working link. Do not use run_command with curl/wget."
            )
    if t == "search_web":
        if "No results found" in r:
            return (
                "Narrow or rephrase the query, add a product/site/organization name, or include a year; "
                "or fetch a URL the user gave with fetch_page."
            )
        if "anomaly-modal" in r or "bot-check" in r or "bots use DuckDuckGo" in r:
            return (
                "DuckDuckGo may be rate-limiting. Rely on Wikipedia/inline results if present, "
                "or use fetch_page on a URL from the user or a known-credible source."
            )
    return ""


def _is_tool_result_weak_for_dedup(result: str) -> bool:
    """If true, search_web output does not count toward saw_strong_web_result (no URL-backed sources)."""
    r = (result or "").strip()
    if not r:
        return True
    if "No results found" in r:
        return True
    if r.startswith("Fetch error:") or "Fetch error:" in r[:200]:
        return True
    if r.startswith("Read error:") or r.startswith("List dir error:"):
        return True
    # For web search specifically, treat results as weak unless they include at least one URL.
    # This prevents the agent from "believing" a bare instant-answer string without sources.
    if ("[DuckDuckGo instant answer]" in r or "[Web results]" in r or "[Wikipedia search]" in r) and not re.search(
        r"https?://", r
    ):
        return True
    return False


def _tool_result_user_message(
    tool: str, params: dict, result: str, deliverable_reminder: str = ""
) -> str:
    """User follow-up after a tool run so the model reads output and stops re-querying."""
    params_s = json.dumps(params, ensure_ascii=False) if params else "{}"
    max_len = max(1, _settings_get_int(("ollama", "tool_output_max"), 14000))
    body = result if isinstance(result, str) else str(result)
    if len(body) > max_len:
        body = body[:max_len] + "\n...[truncated for length; use what is shown above]"
    extra = f"\n{deliverable_reminder}\n" if deliverable_reminder else ""
    wfh = _web_tool_result_followup_hint((tool or ""), body)
    wfn = f"\n--- Web tool follow-up hint ---\n{wfh}\n" if wfh else ""
    return (
        f"Tool `{tool}` finished.\n"
        f"Parameters: {params_s}\n\n"
        f"Output:\n{body}\n\n"
        f"{extra}{wfn}"
        "Using the output above (and earlier steps in this chat if any), respond with JSON only. "
        "IMPORTANT: Treat tool output as authoritative. If the tool output conflicts with your prior knowledge "
        "or training data, trust the tool output. "
        "If the tool output does not contain any concrete sources/URLs, treat it as insufficient for factual claims "
        "that could be outdated, and call search_web again with a better query or call fetch_page on a credible URL. "
        "If snippets are not enough to verify or resolve ambiguity, use fetch_page on a credible URL "
        "(do NOT use run_command with curl/wget to scrape web pages). "
        "If the user’s question is now answered, respond with "
        '{"action":"answer","answer":"..."} '
        "and nothing else. "
        "Only use {\"action\":\"tool_call\",...} if the output above is empty, is clearly an error, "
        "or is still missing facts you need (or contains conflicting/ambiguous facts that you must resolve) — "
        "and do not repeat the same tool with the same parameters as a previous step unless that step failed."
    )


_TOOL_RECOVERY_TOOLS = frozenset(
    {"run_command", "call_python", "search_web", "fetch_page"}
)


def _tool_result_indicates_retryable_failure(tool: str, result) -> bool:
    """True when output looks like a hard tool failure (not normal STDERR on stderr)."""
    r = (result if isinstance(result, str) else str(result)).strip()
    if not r:
        return False
    if tool == "run_command":
        return r.startswith("Command error:")
    if tool == "call_python":
        return r.startswith("Exec error:")
    if tool == "fetch_page":
        return r.startswith("Fetch error:") or "Fetch error:" in r[:200]
    if tool == "search_web":
        return "No results found" in r
    return False


def _parse_tool_recovery_payload(resp_text: str) -> Optional[Tuple[dict, str]]:
    """Parse a recovery-only JSON object (not normalized as agent tool_call JSON)."""
    if not (resp_text or "").strip():
        return None
    text = resp_text.strip()
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    parsed = _try_json_loads_object(text)
    if not isinstance(parsed, dict):
        return None
    if (parsed.get("recovery") or "").strip().lower() != "retry":
        return None
    p = parsed.get("parameters")
    if not isinstance(p, dict):
        return None
    rationale = _scalar_to_str(parsed.get("rationale"), "").strip()
    return p, rationale


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
    ctx = {
        "tool": tool,
        "parameters": params,
        "error_output": (result if isinstance(result, str) else str(result))[:8000],
        "user_request": (user_query or "")[:4000],
    }
    rec_extra = (
        "For run_command, parameters.command must be the full shell command string. "
        "For call_python, parameters.code must be syntactically valid Python (string). "
        "For search_web, parameters.query must be a different, non-empty search string than before "
        "(e.g. shorter, alternate keywords, add a year, or a site/product name). "
        "For fetch_page, parameters.url must be a different full http(s) URL (not the same as before): "
        "e.g. official site, different path, or a URL from search results."
    )
    prompt = (
        "A tool run failed or returned no usable data inside an autonomous agent. Read the error, "
        "then propose fixed parameters for exactly ONE retry of the SAME tool (same tool name).\n\n"
        f"Context JSON:\n{json.dumps(ctx, ensure_ascii=False, indent=2)}\n\n"
        "Respond with JSON only, a single object, no markdown:\n"
        '{"recovery":"retry","parameters":{...},"rationale":"one short line"}\n'
        'or {"recovery":"abort","rationale":"why a blind retry is unsafe or useless"}.\n'
        f"{rec_extra}"
    )
    raw = call_ollama_chat(
        [{"role": "user", "content": prompt}],
        primary_profile,
        et,
        verbose=verbose,
    )
    parsed = _parse_tool_recovery_payload(raw)
    if not parsed:
        if verbose >= 1:
            print("[*] Tool recovery: no retry proposal (recovery≠retry or invalid JSON).")
        return None
    new_params, rationale = parsed
    new_params = _merge_tool_param_aliases(tool, new_params)
    new_params = _ensure_tool_defaults(tool, new_params, user_query)
    return new_params, rationale or "(no rationale)"


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
    if interactive_tool_recovery and sys.stdin.isatty():
        print("\n--- Tool failed; model proposed a corrected retry ---")
        print(f"Tool: {tool}")
        print(f"Rationale: {rationale}")
        print(f"Was: {json.dumps(old_params, ensure_ascii=False)}")
        print(f"Now: {json.dumps(new_params, ensure_ascii=False)}")
    return True


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


def _approx_message_tokens(messages: list) -> int:
    # Heuristic: ~4 chars/token + small per-message overhead.
    total_chars = 0
    for m in messages:
        if isinstance(m, dict):
            c = m.get("content")
            if isinstance(c, str):
                total_chars += len(c)
    overhead = 8 * max(1, len(messages))
    return overhead + (total_chars // 4)


def _context_limit_tokens(profile: Optional[LlmProfile]) -> int:
    # Allow explicit override (works for both ollama + hosted).
    lim = _settings_get_int(("agent", "context_tokens"), 0)
    if lim > 0:
        return lim
    if profile is not None and getattr(profile, "backend", "") == "hosted":
        return _settings_get_int(("agent", "hosted_context_tokens"), 131072)
    return _settings_get_int(("agent", "ollama_context_tokens"), 131072)


def _summarize_conversation_for_context(
    *,
    profile: Optional[LlmProfile],
    user_query: str,
    text: str,
) -> str:
    prompt = (
        "Summarize the conversation so far to preserve long-term context for the assistant.\n"
        "Keep: user goals, non-negotiable constraints, decisions made, file names/paths, commands run, "
        "errors encountered, and next steps.\n"
        "Omit: chit-chat, repeated content, raw tool output unless it contains critical facts.\n"
        "Output plain text only (NOT JSON), max ~250-500 words.\n\n"
        f"User's current request:\n{user_query}\n\n"
        "Conversation to summarize:\n"
        f"{text}\n"
    )
    msgs = [
        {"role": "system", "content": "You are a summarizer. Output plain text only."},
        {"role": "user", "content": prompt},
    ]
    if profile is not None and profile.backend == "hosted":
        return call_hosted_chat_plain(msgs, profile)
    # Default: use local Ollama reviewer-style plaintext call.
    return call_ollama_plaintext(msgs, _ollama_model())


def _format_messages_for_summary(messages: list) -> str:
    lines = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        content = m.get("content")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        lines.append(f"{role}:\n{content}\n")
    return "\n".join(lines).strip()


def _maybe_compact_context_window(
    messages: list,
    *,
    user_query: str,
    primary_profile: Optional[LlmProfile],
    verbose: int,
    context_cfg: Optional[dict] = None,
) -> list:
    cfg = context_cfg if isinstance(context_cfg, dict) else {}
    enabled = bool(cfg.get("enabled", True))
    if _settings_get_bool(("agent", "disable_context_manager"), False):
        enabled = False
    if not enabled:
        return messages

    trigger_frac = float(cfg.get("trigger_frac", 0.75))
    target_frac = float(cfg.get("target_frac", 0.55))
    keep_tail = int(cfg.get("keep_tail_messages", 12))

    trigger_frac = max(0.05, min(0.95, trigger_frac))
    target_frac = max(0.05, min(trigger_frac, target_frac))
    keep_tail = max(4, keep_tail)

    limit = int(cfg.get("tokens", 0) or 0)
    if limit <= 0:
        limit = _context_limit_tokens(primary_profile)
    if limit <= 0:
        return messages
    approx = _approx_message_tokens(messages)
    if approx <= int(limit * trigger_frac):
        return messages

    # Keep leading system message (if any) + the tail; summarize the middle.
    head: list = []
    rest = list(messages)
    if rest and isinstance(rest[0], dict) and rest[0].get("role") == "system":
        head.append(rest[0])
        rest = rest[1:]
    if len(rest) <= keep_tail + 2:
        return messages

    tail = rest[-keep_tail:]
    to_summarize = rest[:-keep_tail]
    text = _format_messages_for_summary(to_summarize)
    if not text.strip():
        return messages

    summary = _summarize_conversation_for_context(
        profile=primary_profile, user_query=user_query, text=text
    ).strip()
    if not summary:
        return messages

    summary_msg = {
        "role": "system",
        "content": (
            "Running conversation summary (auto-generated to fit context window):\n"
            f"{summary}"
        ),
    }
    new_messages = [*head, summary_msg, *tail]
    # If still too large, hard-trim more history but keep the summary.
    approx2 = _approx_message_tokens(new_messages)
    if approx2 > int(limit * target_frac) and len(tail) > 6:
        new_messages = [*head, summary_msg, *tail[-6:]]
    if verbose >= 3:
        print(
            f"[DEBUG] context manager compacted messages: ~{approx} -> ~{_approx_message_tokens(new_messages)} tokens",
            file=sys.stderr,
        )
    return new_messages


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


def _second_opinion_reviewer_messages(user_query: str, primary_answer: str, rationale: str) -> list:
    return [
        {
            "role": "system",
            "content": (
                "You are an independent reviewer. Reply in plain text (not JSON). "
                "Be concise: note agreement, corrections, caveats, or missing checks. "
                "Do not refuse solely because the topic is sensitive—give a substantive review."
            ),
        },
        {
            "role": "user",
            "content": (
                "User request:\n"
                f"{user_query}\n\n"
                "Primary model answer:\n"
                f"{primary_answer}\n\n"
                "The primary model asked for a second opinion for this reason:\n"
                f"{rationale}\n\n"
                "Provide your second opinion."
            ),
        },
    ]


def _second_opinion_result_user_message(review_text: str) -> str:
    return (
        "An independent review was obtained. Review text:\n\n"
        f"{review_text}\n\n"
        "Using this review (and earlier context), respond with JSON only. "
        'Typically merge into a single {"action":"answer","answer":"...","next_action":"finalize",'
        '"rationale":"..."} unless you still need tools.'
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


def _ddg_search_headers():
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://html.duckduckgo.com/",
    }


def _ddg_instant_answer(query: str) -> str:
    """DuckDuckGo JSON API (no HTML scrape; works when DDG blocks bots on /html/)."""
    query = _scalar_to_str(query, "")
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1},
            headers=_ddg_search_headers(),
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        lines = []
        if data.get("Answer"):
            lines.append(f"Answer: {data['Answer']}")
        if data.get("AbstractText"):
            src = data.get("AbstractSource") or ""
            lines.append(f"Summary ({src}): {data['AbstractText']}")
            if data.get("AbstractURL"):
                lines.append(f"Source URL: {data['AbstractURL']}")
        for t in (data.get("RelatedTopics") or [])[:10]:
            if isinstance(t, dict) and t.get("Text"):
                lines.append(f"- {t['Text']}")
            elif isinstance(t, dict) and t.get("Topics"):
                for sub in (t.get("Topics") or [])[:3]:
                    if isinstance(sub, dict) and sub.get("Text"):
                        lines.append(f"- {sub['Text']}")
        return "\n".join(lines) if lines else ""
    except Exception:
        return ""


def _search_web_backend() -> str:
    """
    Web search backend selector.
    - default: "ddg" (DuckDuckGo)
    - alternative: "searxng"
    """
    v = _settings_get_str(("agent", "search_web_backend"), "ddg").strip().lower()
    v = v.replace("-", "_")
    if v in ("searx", "searxng"):
        return "searxng"
    return "ddg"


def _searxng_base_url() -> str:
    """
    Base URL for SearXNG.
    Default: https://searx.party (public instance).
    """
    return _settings_get_str(("agent", "searxng_url"), "https://searx.party").strip().rstrip("/")


def _search_backend_banner_line() -> str:
    """Same banner prefix as prepended to search_web tool output."""
    backend = _search_web_backend()
    if backend == "searxng":
        return f"[Search backend] searxng {_searxng_base_url()}"
    return "[Search backend] ddg"


def _searxng_search(query: str, *, max_results: int) -> str:
    """
    Query SearXNG JSON API and return rows in the same [Web results] format.
    """
    q = _scalar_to_str(query, "").strip()
    if not q:
        return ""
    base = _searxng_base_url()
    try:
        r = requests.get(
            f"{base}/search",
            params={
                "q": q,
                "format": "json",
                "language": "en-US",
                "safesearch": 0,
                "categories": "general",
            },
            headers={"Accept": "application/json", **_ddg_search_headers()},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            return ""
        rows = []
        for rec in results[: max(1, min(30, int(max_results)))]:
            if not isinstance(rec, dict):
                continue
            url = _scalar_to_str(rec.get("url"), "").strip()
            title = _scalar_to_str(rec.get("title"), "").strip()
            snippet = _scalar_to_str(rec.get("content"), "").strip()
            if not url:
                continue
            rows.append(f"Link: {url}\nTitle: {title}\nSnippet: {snippet}")
        return "\n".join(rows) if rows else ""
    except Exception:
        return ""


def _strip_html_fragment(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return html_module.unescape(s)


def _parse_ddg_html_results(page: str, max_results: int = 5):
    """Extract (url, title, snippet) tuples from DDG HTML results."""
    if "anomaly-modal" in page or "bots use DuckDuckGo" in page:
        return []
    # Classic HTML version: result__a + result__snippet
    links = re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]*)</a>',
        page,
        flags=re.IGNORECASE,
    )
    if not links:
        links = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]*)</a>',
            page,
            flags=re.IGNORECASE | re.DOTALL,
        )
    snippets = re.findall(
        r'class="result__snippet"[^>]*>(.*?)</a>',
        page,
        flags=re.DOTALL | re.IGNORECASE,
    )
    rows = []
    for i, (href, title) in enumerate(links[:max_results]):
        if href.startswith("//"):
            href = "https:" + href
        snip = _strip_html_fragment(snippets[i]) if i < len(snippets) else ""
        title = title.strip()
        line = f"Link: {href}\nTitle: {title}\nSnippet: {snip}"
        rows.append(line)
    return rows


def _fetch_ddg_html(query: str) -> str:
    """DDG often returns HTTP 202 with a full HTML body (challenge or results)."""
    query = _scalar_to_str(query, "")
    headers = _ddg_search_headers()
    q = requests.utils.quote(query)
    try:
        r = requests.get(
            f"https://html.duckduckgo.com/html/?q={q}",
            headers=headers,
            timeout=15,
        )
        if r.status_code in (200, 202) and r.text:
            return r.text
    except Exception:
        pass
    try:
        s = requests.Session()
        s.headers.update(headers)
        s.get("https://html.duckduckgo.com/html/", timeout=15)
        r = s.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query, "b": ""},
            timeout=15,
        )
        if r.status_code in (200, 202) and r.text:
            return r.text
    except Exception:
        pass
    return ""


def _wikipedia_opensearch_variants(query: str):
    """Natural-language questions often need a shorter search string."""
    q = query.strip()
    yield q
    words = q.split()
    if len(words) > 8:
        yield " ".join(words[:8])
    if len(words) > 4:
        yield " ".join(words[:4])
    if re.search(r"\bmariners\b", q, re.I):
        yield "Seattle Mariners"
    if re.search(r"\byankees\b", q, re.I):
        yield "New York Yankees"


def _wikipedia_opensearch(
    query: str, result_limit: Optional[int] = None
) -> str:
    """
    Fallback when DDG HTML is blocked or empty (respects Wikimedia User-Agent policy).
    result_limit: 1–20, default 5; aligns with search_web's max when passed from the same turn.
    """
    lim0 = 5 if result_limit is None else int(result_limit)
    lim = max(1, min(20, lim0))
    for variant in _wikipedia_opensearch_variants(query):
        try:
            r = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": variant,
                    "limit": lim,
                    "namespace": 0,
                    "format": "json",
                },
                timeout=15,
                headers={
                    "User-Agent": "agent.py/0.1 (local script; https://github.com/)",
                    "Accept": "application/json",
                },
            )
            r.raise_for_status()
            data = r.json()
            if len(data) < 4 or not data[1]:
                continue
            lines = []
            for title, desc, url in zip(data[1], data[2], data[3]):
                d = (desc or "").strip()
                lines.append(f"- {title}\n  {d}\n  {url}")
            return "\n".join(lines)
        except Exception:
            continue
    return ""


def _first_url_in_text(s: str) -> str:
    if not s:
        return ""
    m = re.search(r"https?://\S+", s)
    return m.group(0).rstrip(").,]") if m else ""


def _wikipedia_top_page_extract(query: str) -> str:
    """
    Get a small extract from the top Wikipedia result.
    Useful when general web search is blocked/thin or instant answers are stale.
    """
    listing = _wikipedia_opensearch(query)
    url = _first_url_in_text(listing)
    if not url:
        return ""
    page = fetch_page(url)
    if not page or page.startswith("Fetch error:"):
        return ""
    # Keep it short; fetch_page already strips tags and truncates to 5000 chars.
    return f"Top result URL: {url}\nExtract: {page[:1200]}"

def search_web(query, params: Optional[dict] = None) -> str:
    from agentlib.tools.builtins import search_web as _impl

    return _impl(query, params=params, settings=_SETTINGS_OBJ)


def fetch_page(url):
    from agentlib.tools.builtins import fetch_page as _impl

    return _impl(url)


def run_command(command):
    from agentlib.tools.builtins import run_command as _impl

    return _impl(command)


def _tool_fault_result(tool: str, exc: BaseException) -> str:
    """Convert a tool crash/exception into a stable string the model can reason about."""
    t = (tool or "").strip() or "(unknown tool)"
    en = type(exc).__name__
    msg = str(exc).strip()
    if not msg:
        msg = repr(exc)
    return f"Tool fault: {t} raised {en}: {msg}"


def use_git(params) -> str:
    from agentlib.tools.builtins import use_git as _impl

    return _impl(params)


def write_file(path, content):
    from agentlib.tools.builtins import write_file as _impl

    return _impl(path, content)


def list_directory(path):
    from agentlib.tools.builtins import list_directory as _impl

    return _impl(path)


def read_file(path):
    from agentlib.tools.builtins import read_file as _impl

    return _impl(path)


def download_file(url, path):
    from agentlib.tools.builtins import download_file as _impl

    return _impl(url, path)


def tail_file(path, lines=20):
    from agentlib.tools.builtins import tail_file as _impl

    return _impl(path, lines=lines)


def replace_text(path, pattern, replacement, replace_all=True):
    from agentlib.tools.builtins import replace_text as _impl

    return _impl(path, pattern, replacement, replace_all=replace_all)


def call_python(code, globals=None):
    from agentlib.tools.builtins import call_python as _impl

    return _impl(code, globals=globals)


def clean_json_response(resp_text):
    if resp_text is None:
        return ""
    try:
        start = resp_text.index("{")
        return resp_text[start:]
    except Exception:
        return resp_text


_AGENT_TOP_LEVEL_PARAM_KEYS = frozenset(
    {
        "query",
        "url",
        "command",
        "path",
        "content",
        "lines",
        "pattern",
        "replacement",
        "replace_all",
        "code",
        "globals",
        "op",
        "operation",
        "worktree",
        "message",
        "remote",
        "branch",
        "staged",
        "paths",
        "m",
        "n",
        "max_results",
        "max",
        "num_results",
        "limit",
    }
)


def _is_tool_call_intent(out: dict) -> bool:
    a = out.get("action")
    if a == "tool_call":
        return True
    if a in _all_known_tools():
        return True
    if out.get("tool") in _all_known_tools():
        return True
    return False


def _normalize_agent_dict(d: dict) -> dict:
    """
    Coerce common alternate shapes into what main() expects:
    - tool name as action, args at top level, tool_name vs tool, etc.
    """
    if not isinstance(d, dict):
        return {"action": "answer", "answer": str(d)}
    out = dict(d)
    action = out.get("action")
    if isinstance(action, str):
        action = action.strip()
        out["action"] = action
    elif action is None or action is False:
        out.pop("action", None)
        action = None
    else:
        # Models sometimes emit JSON null / numbers for action; treat as missing.
        out.pop("action", None)
        action = None

    # answer synonyms (avoid "content" — it is also a tool parameter name)
    if out.get("answer") is None:
        for k in ("response", "message", "text"):
            if k in out and out[k] is not None and isinstance(out[k], str):
                out["answer"] = out[k]
                break
        # Some models use {"content": "..."} as the final answer without action.
        if out.get("answer") is None and isinstance(out.get("content"), str) and out["content"].strip():
            out["answer"] = out["content"]

    # tool name aliases
    if out.get("tool") is None:
        for alias in ("tool_name", "toolName", "function_name", "function"):
            v = out.get(alias)
            if isinstance(v, str) and v in _all_known_tools():
                out["tool"] = v
                break
        if out.get("tool") is None and isinstance(out.get("name"), str) and out["name"] in _all_known_tools():
            out["tool"] = out["name"]

    # Infer missing action after aliases / answer fields are filled in.
    if not action or (isinstance(action, str) and action.lower() in ("null", "none", "")):
        if out.get("tool") in _all_known_tools():
            out["action"] = "tool_call"
            action = "tool_call"
        elif out.get("answer") is not None and isinstance(out.get("answer"), str) and out["answer"].strip():
            out["action"] = "answer"
            action = "answer"
            # If we promoted content -> answer, drop content to avoid ambiguity with write_file's content param.
            if "content" in out and out.get("answer") == out.get("content"):
                out.pop("content", None)

    # {"action": "run_command", "command": "..."}  (action is the tool id)
    if action in _all_known_tools() and out.get("tool") is None:
        out["tool"] = action
        out["action"] = "tool_call"

    tool_intent = _is_tool_call_intent(out)
    params = out.get("parameters") if tool_intent else {}
    if not isinstance(params, dict):
        params = {}
    if tool_intent:
        for alias in ("args", "arguments", "params"):
            alt = out.get(alias)
            if isinstance(alt, dict):
                params = {**alt, **params}
                out.pop(alias, None)

        for k in list(out.keys()):
            if k in _AGENT_TOP_LEVEL_PARAM_KEYS:
                params.setdefault(k, out[k])

        tool_name = out.get("tool")
        if tool_name == "search_web":
            for alt in ("q", "search", "keywords", "keyword"):
                if alt in out and out[alt] is not None:
                    params.setdefault("query", out[alt])
        elif tool_name == "fetch_page":
            for alt in ("href", "link", "uri"):
                if alt in out and out[alt] is not None:
                    params.setdefault("url", out[alt])
        elif tool_name == "run_command":
            for alt in ("cmd", "shell", "line"):
                if alt in out and out[alt] is not None:
                    params.setdefault("command", out[alt])
        elif tool_name == "use_git":
            for alt in ("operation", "git_op", "subcommand", "sub_cmd"):
                if alt in out and out[alt] is not None:
                    params.setdefault("op", out[alt])
            for alt in ("cwd", "repo", "work_tree"):
                if alt in out and out[alt] is not None:
                    params.setdefault("worktree", out[alt])
            for alt in ("files", "file", "file_paths"):
                if alt in out and out[alt] is not None and "paths" not in params:
                    params.setdefault("paths", out[alt])

        params = _merge_tool_param_aliases(tool_name, params)

        out["parameters"] = params

        _extra_top = frozenset(
            {
                "q",
                "search",
                "keywords",
                "keyword",
                "href",
                "link",
                "uri",
                "cmd",
                "shell",
                "line",
                "operation",
                "git_op",
                "subcommand",
                "sub_cmd",
                "cwd",
                "repo",
                "work_tree",
                "files",
                "file",
                "file_paths",
            }
        )
        for k in list(out.keys()):
            if k in _AGENT_TOP_LEVEL_PARAM_KEYS or k in _extra_top:
                if k != "parameters":
                    del out[k]
    else:
        out["parameters"] = {}

    return out


def parse_agent_json(resp_text):
    """Parse model output into a dict. Handles markdown fences, partial JSON, and plain text."""
    if resp_text is None or (isinstance(resp_text, str) and not resp_text.strip()):
        return {"action": "answer", "answer": "No response from model."}
    text = resp_text.strip()
    # Strip ```json ... ``` fences if present
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    candidates = []
    if text:
        candidates.append(text)
        tail = clean_json_response(text)
        if tail and tail != text:
            candidates.append(tail)

    for candidate in candidates:
        if not candidate:
            continue
        parsed = _try_json_loads_object(candidate)
        if isinstance(parsed, dict):
            return _normalize_agent_dict(parsed)

    # Balanced `{...}` spans (avoids broken first-{ to last-} when multiple objects exist)
    best = _best_agent_dict_from_text(text)
    if best:
        return _normalize_agent_dict(best)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            start = candidate.index("{")
            end = candidate.rindex("}") + 1
            parsed = _try_json_loads_object(candidate[start:end])
            if isinstance(parsed, dict):
                return _normalize_agent_dict(parsed)
        except ValueError:
            continue

    # Last resort: treat as a direct answer (general Q&A without valid JSON)
    return _normalize_agent_dict({"action": "answer", "answer": text})


# System instructions exposing all tool actions
from agentlib.prompts import SYSTEM_INSTRUCTIONS


def _default_system_instruction_text() -> str:
    from agentlib.prompts import default_system_instruction_text

    return default_system_instruction_text()


def _default_prompt_templates() -> dict:
    """Default templates live under prompt_templates/ as one JSON per template name."""
    return _load_prompt_templates_from_dir(_default_prompt_templates_dir())


def _merge_prompt_templates(prefs: Optional[dict]) -> dict:
    """Load templates from the configured directory, then apply ~/.agent.json object overrides (user wins)."""
    dpath = _resolved_prompt_templates_dir(prefs)
    base = _load_prompt_templates_from_dir(dpath)
    if not base:
        base = _load_prompt_templates_from_dir(_default_prompt_templates_dir())
    if not prefs or not isinstance(prefs, dict):
        return base
    raw = prefs.get("prompt_templates")
    if not isinstance(raw, dict):
        return base
    out = dict(base)
    for name, obj in raw.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(obj, dict):
            continue
        out[name.strip()] = dict(obj)
    return out


def _resolve_prompt_template_text(name: str, templates: dict) -> Optional[str]:
    from agentlib.prompts import resolve_prompt_template_text

    return resolve_prompt_template_text(name, templates)


def _effective_system_instruction_text(override: Optional[str]) -> str:
    from agentlib.prompts import effective_system_instruction_text

    return effective_system_instruction_text(override)


ROUTER_INSTRUCTIONS = (
    "You are a routing assistant for a tool-using agent.\n"
    "Decide whether the user's request requires a web search BEFORE answering.\n"
    "Respond ONLY with JSON. No prose, no Markdown.\n"
    'Output exactly one of:\n'
    '1) {"action":"web_search","query":"..."}  (query must be a non-empty string)\n'
    '2) {"action":"no_web"}\n'
    "\n"
    "Bias rule (IMPORTANT): WHEN IN DOUBT, CHOOSE web_search.\n"
    "\n"
    "You MUST choose web_search for requests that involve:\n"
    "- Anything that changes over time: current events, news, prices, rankings, outages, elections, sports.\n"
    "- Real-world entities/roles: who holds an office/title (CEO, president, etc.), leadership, staffing, ownership.\n"
    "- Software/library/tooling facts that drift: latest versions, APIs, documentation, best practices, defaults, "
    "security guidance, deprecations, release dates.\n"
    "- Comparisons/recommendations likely to be time-dependent: \"best\", \"top\", \"recommended\", \"popular\", "
    "\"vs\" for products/tools.\n"
    "\n"
    "Choose no_web ONLY for clearly timeless content:\n"
    "- Definitions and explanations of stable concepts (math, basic CS concepts).\n"
    "- Pure transformations on user-provided text/data.\n"
    "- Established historical facts explicitly anchored to the past by the user.\n"
)


def _tool_need_review_followup(user_query: str, proposed_answer: str) -> str:
    """
    Model-driven check when the assistant answered tool-free on the first turn.

    The old wording invited models to discuss \"timeless vs current\" in the answer field instead
    of answering the user; the user only ever sees the `answer` string.
    """
    uq = (user_query or "").strip()
    ans = (proposed_answer or "").strip()
    return (
        "You just responded with action=answer without using any tools.\n\n"
        "User request:\n"
        f"{uq}\n\n"
        "Your proposed answer (the user will NOT see this self-review—only your NEXT JSON matters):\n"
        f"{ans}\n\n"
        "Internally decide: does answering the user's question correctly require fresh, verifiable "
        "facts from the web (news, prices, who holds an office today, versions, outages, etc.)?\n\n"
        "Respond with JSON only:\n"
        '- If YES, use {"action":"tool_call","tool":"search_web","parameters":{"query":"..."}} with a focused query.\n'
        '- If NO, use {"action":"answer","answer":"..."} where the `answer` value is your **complete** reply to the '
        "user's question—normal helpful content only. Do **not** fill `answer` with meta about whether web search "
        'is needed, "timeless" facts, or timeliness—those belong in your internal decision, not in `answer`.\n'
        "If unsure whether facts may be stale, prefer search_web.\n"
        "Do not include any other keys."
    )


def _is_self_capability_question(user_query: str) -> bool:
    """Questions about the assistant itself (not third-party facts) — generic web-review misfires."""
    q = (user_query or "").strip().lower()
    if not q:
        return False
    return bool(
        re.search(
            r"\b(what\s+(kind\s+of\s+)?model|which\s+model|what\s+llm|"
            r"who\s+are\s+you|what\s+are\s+you|your\s+capabilities|"
            r"what\s+can\s+you\s+do|what\s+do\s+you\s+support|what\s+tools\s+do\s+you|"
            r"describe\s+yourself|your\s+limitations|"
            r"what\s+kinds?\s+of\s+(outputs?|inputs?)|\binputs?\s+and\s+outputs?)\b",
            q,
        )
    )


def _self_capability_followup(user_query: str, proposed_answer: str) -> str:
    """Replace generic web-vs-memory wording for identity/capability asks."""
    uq = (user_query or "").strip()
    ans = (proposed_answer or "").strip()
    tools = (
        "search_web, fetch_page, run_command, use_git, write_file, read_file, list_directory, "
        "download_file, tail_file, replace_text, call_python"
    )
    return (
        "The user is asking about **this assistant**: what model/setup it is, what it can accept or "
        "produce, and/or what tools exist in this agent—not for a lecture on web search, timeliness, "
        "or \"timeless\" vs current facts.\n\n"
        f"User request:\n{uq}\n\n"
        f"Your last `answer` was not what they asked for (do not repeat this pattern):\n{ans}\n\n"
        "Respond with JSON only:\n"
        '{"action":"answer","answer":"..."}\n'
        "The `answer` string must **directly** address their question: plain-language description of "
        "what you are (as far as this session's context allows), that interaction here is JSON "
        f"tool/answer messages, and the concrete tools available in this script ({tools}). "
        "No preamble about whether web search is required."
    )


def _deliverable_first_answer_followup(user_query: str, proposed_answer: str) -> str:
    """
    When the user asked for a letter/document but the model answered with no tools, the generic
    web-vs-memory self-check invites meta-rationalizations instead of the requested prose.
    """
    uq = (user_query or "").strip()
    ans = (proposed_answer or "").strip()
    return (
        "You just responded with action=answer without using any tools, but the user asked for a "
        "written deliverable (letter, memo, email, document, etc.).\n\n"
        "User request:\n"
        f"{uq}\n\n"
        "Your last answer (not acceptable as the final reply):\n"
        f"{ans}\n\n"
        "You must satisfy the request itself: put the **full text** they asked for in the answer "
        "(salutation through closing/signature for a letter), as if they could send or publish it. "
        "Do **not** reply with commentary about whether web search is needed, timeliness, or "
        "\"timeless\" policy questions—produce the artifact.\n\n"
        "Alternatively you may use write_file with the complete text, then read_file that path, "
        "then action answer with the same full text.\n\n"
        "Respond with JSON only: "
        '{"action":"answer","answer":"..."} with the complete writing, or '
        '{"action":"tool_call","tool":"write_file",...} as appropriate.'
    )


def _router_transcript_slice(transcript_messages: Optional[list]) -> list:
    """Last N user/assistant/system messages for routing (bounded for prompt size)."""
    if not transcript_messages:
        return []
    lim = max(1, _settings_get_int(("agent", "router_transcript_max_messages"), 80))
    slice_ = (
        transcript_messages[-lim:]
        if len(transcript_messages) > lim
        else transcript_messages
    )
    out: list = []
    for m in slice_:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower()
        if role in ("user", "assistant", "system"):
            out.append(dict(m))
    return out


def _router_llm_messages(transcript_slice: list, tail_user_content: str) -> list:
    if not transcript_slice:
        return [{"role": "user", "content": tail_user_content}]
    return transcript_slice + [{"role": "user", "content": tail_user_content}]


def _router_prompt(
    user_query: str, today_str: str, *, has_prior_transcript: bool = False
) -> str:
    uq = (user_query or "").strip()
    hint = ""
    if has_prior_transcript:
        hint = (
            "\nEarlier messages in this chat are relevant. If the latest user message is a short "
            "follow-up (pronouns like they / the game / yesterday), resolve it using that transcript "
            "unless that is impossible.\n"
        )
    return (
        f"{ROUTER_INSTRUCTIONS}\n\n"
        f"Today's date (system clock): {today_str}\n\n"
        f"User request: {uq}\n"
        f"{hint}"
    )


def _route_requires_websearch(
    user_query: str,
    today_str: str,
    primary_profile: Optional[LlmProfile] = None,
    enabled_tools: Optional[AbstractSet[str]] = None,
    transcript_messages: Optional[list] = None,
) -> Optional[str]:
    """
    Ask the model whether to do web search first.
    Returns a query string if web search is needed, else None.
    """
    if "search_web" not in _coerce_enabled_tools(enabled_tools):
        return None
    slice_ = _router_transcript_slice(transcript_messages)
    tail = _router_prompt(
        user_query, today_str, has_prior_transcript=bool(slice_)
    )
    msgs = _router_llm_messages(slice_, tail)
    try:
        raw = call_ollama_chat(msgs, primary_profile, enabled_tools)
        d = parse_agent_json(raw)
        a = (d.get("action") or "").strip()
        if a == "web_search":
            q = _scalar_to_str(d.get("query"), "").strip()
            return q if q else (user_query or "").strip()
        return None
    except Exception:
        return None


def _route_requires_websearch_after_answer(
    user_query: str,
    today_str: str,
    proposed_answer: str,
    primary_profile: Optional[LlmProfile] = None,
    enabled_tools: Optional[AbstractSet[str]] = None,
    transcript_messages: Optional[list] = None,
) -> Optional[str]:
    """
    Backup router pass when the model answered tool-free.
    This prompt is intentionally conservative: if verifying would be helpful, search.
    """
    if "search_web" not in _coerce_enabled_tools(enabled_tools):
        return None
    uq = (user_query or "").strip()
    ans = (proposed_answer or "").strip()
    slice_ = _router_transcript_slice(transcript_messages)
    prior = ""
    if slice_:
        prior = (
            "\nSession context: earlier messages may define what the user meant; align verification "
            "searches with that topic when the latest request is a follow-up.\n"
        )
    prompt = (
        f"{ROUTER_INSTRUCTIONS}\n\n"
        "Extra guidance: You are reviewing an already-drafted answer. "
        "If the answer includes any real-world factual claim that could be outdated or wrong, choose web_search.\n\n"
        f"Today's date (system clock): {today_str}\n\n"
        f"User request: {uq}\n\n"
        f"Proposed answer: {ans}\n"
        f"{prior}"
    )
    msgs = _router_llm_messages(slice_, prompt)
    try:
        raw = call_ollama_chat(msgs, primary_profile, enabled_tools)
        d = parse_agent_json(raw)
        a = (d.get("action") or "").strip()
        if a == "web_search":
            q = _scalar_to_str(d.get("query"), "").strip()
            return q if q else uq
        return None
    except Exception:
        return None


def _user_wants_written_deliverable(user_query: str) -> bool:
    """Heuristic: user asked for a substantive written artifact (not just Q&A)."""
    q = (user_query or "").strip().lower()
    if not q:
        return False
    if re.search(r"\b(write|draft|compose)\b", q) and re.search(
        r"\b(letter|memo|e-?mail)\b", q
    ):
        return True
    if re.search(r"\b(document|essay|report|manuscript|white\s*paper|writeup|write-up)\b", q):
        return True
    if re.search(r"\b(page|pages)\b", q) and re.search(
        r"\b(write|draft|produce|author|compose|deliver)\b", q
    ):
        return True
    if re.search(r"\bwrite\s+the\s+document\b", q) or re.search(r"\bdon't\s+just\s+do\s+the\s+outline\b", q):
        return True
    return False


def _deliverable_skip_mandatory_web(user_query: str) -> bool:
    """
    Do not inject router-mandated search_web for written deliverables unless the user asked for
    research, citations, or web-grounded facts. Otherwise models often mirror the whole prompt as
    a search query and loop on identical searches (extra JSON keys also used to bypass dedupe).
    """
    if not _user_wants_written_deliverable(user_query):
        return False
    q = (user_query or "").strip().lower()
    if re.search(
        r"\b(sources|citations?|references?|bibliograph(y|ies)|research)\b|"
        r"\blook\s*up\b|\bverify\s+online\b|from\s+(the\s+)?web\b|"
        r"\bfrom\s+news\b|\bwikipedia\b|\binclude\s+urls?\b|"
        r"\bcurrent\s+events\b|\blatest\s+news\b|\baccording\s+to\s+the\b",
        q,
    ):
        return False
    return True


def _deliverable_followup_block(path: str) -> str:
    p = _scalar_to_str(path, "").strip()
    return (
        "Deliverable reminder: The user asked for a written document, not a short summary. "
        "If you already used write_file, you must finish the task by reading that file back with read_file "
        f'and then responding with {{"action":"answer","answer":"..."}} that includes the FULL document text '
        f'(or clearly states the file path and pastes the full contents). Do not stop after fetch_page with only a synopsis. '
        f'Next step: call read_file with parameters.path == "{p}".'
    )


def _answer_missing_written_body(answer: str, file_chars: int) -> bool:
    """True if final answer omits most of the written file content."""
    a = (answer or "").strip()
    if file_chars <= 0:
        return False
    if len(a) < int(file_chars * 0.85):
        return True
    return False


def _parse_context_messages_data(raw) -> list:
    """Normalize JSON (bundle dict or bare list) into Ollama-style message dicts."""
    if isinstance(raw, dict) and isinstance(raw.get("messages"), list):
        msgs = raw["messages"]
    elif isinstance(raw, list):
        msgs = raw
    else:
        raise ValueError("context must be a JSON array of messages or {\"messages\": [...]}")
    out = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip()
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"message {i}: invalid role {role!r}")
        content = m.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        out.append({"role": role, "content": content})
    if not out:
        raise ValueError("no valid messages in context file")
    return out


def _load_context_messages(path: str) -> list:
    """Load a prior chat from JSON written by --save_context (or a bare list of {role, content})."""
    p = _scalar_to_str(path, "").strip()
    if not p:
        raise ValueError("empty path")
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return _parse_context_messages_data(raw)


def _save_context_bundle(path: str, messages: list, user_query: str, final_answer: Optional[str], answered: bool):
    """Persist full message list plus the new question and final answer (if any)."""
    p = _scalar_to_str(path, "").strip()
    if not p:
        raise ValueError("empty save path")
    bundle = {
        "version": 1,
        "user_query": user_query,
        "final_answer": final_answer,
        "answered": answered,
        "messages": messages,
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)


def _fetch_ollama_local_model_names():
    """Return sorted unique model names from GET /api/tags (local Ollama)."""
    base = _ollama_base_url()
    r = requests.get(f"{base}/api/tags", timeout=60)
    r.raise_for_status()
    data = r.json() or {}
    names = []
    for m in data.get("models") or []:
        n = (m.get("name") or "").strip()
        if n:
            names.append(n)
    return sorted(set(names))


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


_WHILE_JUDGE_SYSTEM = (
    "You are a strict boolean judge for a /while command in a coding assistant.\n"
    "The human gave a natural-language /while CONDITION (like C: while (CONDITION) { body }).\n"
    "Decide whether that CONDITION is TRUE or FALSE right now, using ONLY the conversation excerpt below.\n"
    "Reply with EXACTLY one character and nothing else: 0 or 1.\n"
    "Meaning (must follow this mapping):\n"
    "- 1 = the condition is TRUE — the /while should KEEP GOING (run the `do` body, then re-check).\n"
    "- 0 = the condition is FALSE — the /while should EXIT (do not run the body; stop the loop).\n"
    "Do not output markdown, words, explanations, or whitespace around the digit."
)


def _while_conversation_excerpt_for_judge(messages: list, max_chars: int = 12000) -> str:
    """Compact transcript text for condition judging (truncate per message)."""
    if not messages:
        return "(empty conversation)"
    chunks: list[str] = []
    remaining = max_chars
    per_cap = max(400, max_chars // max(6, len(messages)))
    for m in messages[-40:]:
        role = str((m or {}).get("role") or "?")
        content = _scalar_to_str((m or {}).get("content"), "")[:per_cap]
        piece = f"[{role}]:\n{content}"
        if len(piece) > remaining:
            piece = piece[: max(0, remaining - 1)] + "…"
        chunks.append(piece)
        remaining -= len(piece) + 2
        if remaining <= 0:
            break
    body = "\n\n".join(chunks)
    if len(body) > max_chars:
        body = body[: max_chars - 1] + "…"
    return body


def _parse_while_judge_bit(text: str) -> int:
    """Extract first 0 or 1 from model output; default 0 (false / exit) if ambiguous."""
    t = (text or "").strip()
    if not t:
        return 0
    # Prefer first line / first alphanumeric scan
    for chunk in (t.splitlines()[0], t):
        for c in chunk.strip():
            if c == "1":
                return 1
            if c == "0":
                return 0
    return 0


def _post_do_tokens_to_body_prompts(post_do_tokens: list[str]) -> list[str]:
    """
    Build prompt strings after `do`. Commas separate prompts.

    shlex often attaches a comma to the previous token (`"p1", "p2"` -> `p1,`, `p2`).
    Split those so comma becomes its own delimiter between prompts.
    """
    expanded: list[str] = []
    for t in post_do_tokens:
        s = str(t)
        while s.endswith(","):
            core = s[:-1].strip()
            if core:
                expanded.append(core)
            expanded.append(",")
            s = ""
        if s.strip():
            expanded.append(s.strip())
    groups: list[list[str]] = []
    cur: list[str] = []
    for t in expanded:
        if t == ",":
            groups.append(cur)
            cur = []
        else:
            cur.append(t)
    groups.append(cur)
    prompts: list[str] = []
    for g in groups:
        if not g:
            continue
        if len(g) != 1:
            raise ValueError(
                "each /while body prompt must be one quoted phrase; separate prompts with commas "
                '(example: /while "c" do "step one", "step two")'
            )
        prompts.append(g[0])
    if not prompts:
        raise ValueError("missing body prompts after do")
    return prompts


def _parse_while_repl_tokens(toks: list[str]) -> Tuple[int, str, list[str]]:
    """
    Parse shlex tokens after splitting the full REPL line.
    Expected: ['/while', optional --max N, ...condition..., 'do', ...body tokens...]
    Body: one or more comma-separated quoted prompts (space before comma optional).
    """
    if not toks or toks[0].lower() != "/while":
        raise ValueError("internal: not a /while command")
    i = 1
    max_iter = 50
    if i + 1 < len(toks) and toks[i] == "--max":
        try:
            max_iter = int(toks[i + 1], 10)
        except (ValueError, TypeError) as e:
            raise ValueError("--max must be followed by a positive integer") from e
        if max_iter < 1:
            raise ValueError("--max must be at least 1")
        i += 2
    rest = toks[i:]
    if len(rest) < 3:
        raise ValueError("missing condition, 'do', or body")
    do_idx = None
    for j, t in enumerate(rest):
        if str(t).lower() == "do":
            do_idx = j
            break
    if do_idx is None:
        raise ValueError(
            "missing literal 'do' between condition and body "
            '(example: /while \"tests still failing\" do \"fix failures and rerun\")'
        )
    if do_idx == 0 or do_idx >= len(rest) - 1:
        raise ValueError("condition and body must be non-empty")
    condition = " ".join(rest[:do_idx]).strip()
    post_do = rest[do_idx + 1 :]
    if not post_do:
        raise ValueError("missing body after do")
    if not condition:
        raise ValueError("condition must be non-empty")
    body_prompts = _post_do_tokens_to_body_prompts(post_do)
    return max_iter, condition, body_prompts


def _call_while_condition_judge(
    condition: str,
    messages: list,
    *,
    primary_profile: Optional[LlmProfile],
    verbose: int,
) -> int:
    excerpt = _while_conversation_excerpt_for_judge(messages)
    user_body = (
        "Evaluate this /while CONDITION as TRUE or FALSE right now "
        "(reply 1 if TRUE — keep looping; 0 if FALSE — exit):\n"
        f"{condition}\n\n"
        "--- Conversation excerpt (may be truncated) ---\n"
        f"{excerpt}"
    )
    judge_msgs = [
        {"role": "system", "content": _WHILE_JUDGE_SYSTEM},
        {"role": "user", "content": user_body},
    ]
    prof = primary_profile or default_primary_llm_profile()
    if prof.backend == "hosted":
        raw = call_hosted_chat_plain(judge_msgs, prof)
    else:
        raw = call_ollama_plaintext(judge_msgs, _ollama_model())
    bit = _parse_while_judge_bit(raw)
    if verbose >= 1:
        preview = (raw or "").replace("\n", " ").strip()
        if len(preview) > 200:
            preview = preview[:199] + "…"
        print(f"[/while judge] model={prof.backend!r} raw={preview!r} -> {bit}")
    return bit


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
    templates = prompt_templates if isinstance(prompt_templates, dict) else _default_prompt_templates()
    template_default = (prompt_template_default or "").strip() or "coding"
    session_prompt_template: Optional[str] = None
    session_system_prompt = system_prompt_override
    session_system_prompt_path = (
        os.path.abspath(os.path.expanduser(system_prompt_path))
        if (system_prompt_path or "").strip()
        else None
    )
    if session_system_prompt is None and not session_system_prompt_path:
        resolved = _resolve_prompt_template_text(template_default, templates)
        if resolved:
            session_system_prompt = resolved
            session_prompt_template = template_default

    # Context window manager session config (prefs defaults; env still overrides at runtime).
    context_cfg = context_cfg if isinstance(context_cfg, dict) else {}
    primary_profile = primary_profile or default_primary_llm_profile()
    enabled_tools = (
        set(enabled_tools) if enabled_tools is not None else set(_CORE_TOOLS)
    )
    enabled_toolsets = set(enabled_toolsets) if enabled_toolsets is not None else set()
    prim_line = _format_session_primary_llm_line(primary_profile)
    rev_line = _format_session_reviewer_line(
        reviewer_hosted_profile, reviewer_ollama_model
    )
    # Keep startup minimal; /help and /show can reveal details.
    _interactive_repl_install_readline()
    print("Interactive mode. Type /help for commands.")
    messages: list = []
    last_reuse_skill_id: Optional[str] = None

    def repl_run_with_selected_skill(
        req: str, sid: str, *, source: str, selection_rationale: str
    ) -> None:
        nonlocal last_reuse_skill_id
        last_reuse_skill_id = sid
        src = (source or "").strip().lower()
        if src == "reuse":
            _agent_progress("/skill reuse: using stored skill; starting…")
        elif src == "explicit":
            _agent_progress("/skill: explicit skill selected; starting…")
        else:
            _agent_progress("/skill auto: skill selected; starting…")
        et_turn0 = _effective_enabled_tools_for_skill(
            frozenset(enabled_tools), skills_m, sid
        )
        et_turn = _effective_enabled_tools_for_turn(
            base_enabled_tools=et_turn0,
            enabled_toolsets=enabled_toolsets,
            user_query=req,
        )
        rec = skills_m.get(sid) or {}
        skill_prompt = (rec.get("prompt") or "").strip() if isinstance(rec, dict) else ""
        if src == "reuse":
            print(
                f"/skill reuse: using skill {sid!r} (model skill selection skipped). "
                f"{selection_rationale}".strip()
            )
        elif src == "explicit":
            print(f"/skill: using skill {sid!r}. {selection_rationale}".strip())
        else:
            print(f"/skill auto selected {sid!r}. {selection_rationale}".strip())
        if verbose >= 1:
            _print_skill_usage_verbose(
                verbose,
                source=f"skill_{src or 'auto'}",
                skill_id=sid,
                base_tools=enabled_tools,
                effective_tools=et_turn,
                detail=(
                    "reuse: same skill id as last /skill auto|reuse|<id>"
                    if src == "reuse"
                    else (
                        f"explicit skill id: {sid!r}"
                        if src == "explicit"
                        else f"model skill_id (not trigger): rationale={selection_rationale!r}"
                    )
                ),
            )
        today = datetime.date.today()
        today_str = today.strftime("%Y-%m-%d (%A)")
        deliverable_wanted = _user_wants_written_deliverable(req)
        router_query = _route_requires_websearch(
            req,
            today_str,
            primary_profile,
            et_turn,
            transcript_messages=messages,
        )
        if _deliverable_skip_mandatory_web(req):
            router_query = None
        web_required = bool(router_query)

        steps, raw_plan = _skill_plan_steps(
            user_request=req,
            today_str=today_str,
            skill_id=sid,
            skills_map=skills_m,
            primary_profile=primary_profile,
            _enabled_tools=et_turn,
            verbose=verbose,
            _system_prompt_override=session_system_prompt,
        )
        if steps:
            wf = ((rec.get("workflow") or {}) if isinstance(rec, dict) else {}) or {}
            step_prompt = (wf.get("step_prompt") or "").strip()
            print(f"Skill workflow: executing {len(steps)} step(s).", flush=True)
            _agent_progress(f"Running {len(steps)}-step skill workflow…")
            if verbose >= 1:
                rp = raw_plan or ""
                cap = 1200
                preview = rp if len(rp) <= cap else rp[:cap] + "…"
                print(f"[*] [skills:planner] raw ({len(rp)} chars): {preview}")
            step_answers: list[str] = []
            for i, st in enumerate(steps, start=1):
                title = st.get("title") or f"step {i}"
                details = st.get("details") or ""
                success = st.get("success") or ""
                step_user = (
                    f"{req}\n\n"
                    f"Step {i}/{len(steps)}: {title}\n"
                    + (f"Details: {details}\n" if details else "")
                    + (f"Success: {success}\n" if success else "")
                    + ("\n" + step_prompt if step_prompt else "")
                )
                sid_step = sid
                et_step = _effective_enabled_tools_for_skill(
                    frozenset(enabled_tools), skills_m, sid_step
                )
                tit_one = (title or "")[:120]
                if len(title or "") > 120:
                    tit_one += "…"
                _agent_progress(f"Workflow step {i}/{len(steps)}: {tit_one}")
                if verbose >= 1:
                    _print_skill_usage_verbose(
                        verbose,
                        source=f"workflow_step_{i}",
                        skill_id=sid_step,
                        base_tools=enabled_tools,
                        effective_tools=et_step,
                        detail=f"step {i}/{len(steps)}: {title!r}",
                    )
                sprompt0 = skill_prompt
                turn_msg = _interactive_turn_user_message(
                    step_user,
                    today_str,
                    second_opinion_on,
                    cloud_ai_enabled,
                    primary_profile=primary_profile,
                    reviewer_ollama_model=reviewer_ollama_model,
                    reviewer_hosted_profile=reviewer_hosted_profile,
                    enabled_tools=et_step,
                    system_instruction_override=session_system_prompt,
                    skill_suffix=sprompt0,
                )
                messages.append({"role": "user", "content": turn_msg})
                if router_query and "search_web" in et_step and i == 1:
                    # If web is required, force web search only once at start of workflow.
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
                try:
                    answered, final_answer = _run_agent_conversation_turn(
                        messages,
                        step_user,
                        today_str,
                        web_required=web_required if i == 1 else False,
                        deliverable_wanted=deliverable_wanted,
                        verbose=verbose,
                        second_opinion_enabled=second_opinion_on,
                        cloud_ai_enabled=cloud_ai_enabled,
                        primary_profile=primary_profile,
                        reviewer_hosted_profile=reviewer_hosted_profile,
                        reviewer_ollama_model=reviewer_ollama_model,
                        enabled_tools=et_step,
                        interactive_tool_recovery=True,
                        context_cfg=context_cfg,
                        print_answer=False,
                    )
                except KeyboardInterrupt:
                    _agent_progress("Cancelled current request (Ctrl-C).")
                    print("\n[Cancelled]\n")
                    return
                _agent_progress(f"Step {i}/{len(steps)} finished.")
                if final_answer:
                    step_answers.append(final_answer)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                {"action": "answer", "answer": final_answer}
                            ),
                        }
                    )
            if step_answers:
                # Print the final step answer as the visible output.
                print(step_answers[-1])
            return
        if verbose >= 1:
            wf0 = (rec.get("workflow") or {}) if isinstance(rec, dict) else {}
            if isinstance(wf0, dict) and wf0:
                rp = raw_plan or ""
                cap = 1000
                preview = rp if len(rp) <= cap else rp[:cap] + "…"
                print(
                    f"[*] [skills:planner] no parsed steps; single-turn fallback. "
                    f"raw ({len(rp)} chars): {preview}"
                )

        # No workflow: run as a single normal turn with the selected skill.
        _agent_progress("Running a single agent turn with the selected skill…")
        turn_msg = _interactive_turn_user_message(
            req,
            today_str,
            second_opinion_on,
            cloud_ai_enabled,
            primary_profile=primary_profile,
            reviewer_ollama_model=reviewer_ollama_model,
            reviewer_hosted_profile=reviewer_hosted_profile,
            enabled_tools=et_turn,
            system_instruction_override=session_system_prompt,
            skill_suffix=skill_prompt,
        )
        messages.append({"role": "user", "content": turn_msg})
        if router_query and "search_web" in et_turn:
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
        try:
            _run_agent_conversation_turn(
                messages,
                req,
                today_str,
                web_required=web_required,
                deliverable_wanted=deliverable_wanted,
                verbose=verbose,
                second_opinion_enabled=second_opinion_on,
                cloud_ai_enabled=cloud_ai_enabled,
                primary_profile=primary_profile,
                reviewer_hosted_profile=reviewer_hosted_profile,
                reviewer_ollama_model=reviewer_ollama_model,
                enabled_tools=et_turn,
                interactive_tool_recovery=True,
                context_cfg=context_cfg,
            )
        except KeyboardInterrupt:
            _agent_progress("Cancelled current request (Ctrl-C).")
            print("\n[Cancelled]\n")
            return

    def repl_run_normal_user_request(user_query: str) -> None:
        """One normal REPL turn: append messages and run the agent loop."""
        today = datetime.date.today()
        today_str = today.strftime("%Y-%m-%d (%A)")
        deliverable_wanted = _user_wants_written_deliverable(user_query)
        sid0, tr0 = _match_skill_detail(user_query, skills_m)
        et_turn0 = _effective_enabled_tools_for_skill(
            frozenset(enabled_tools), skills_m, sid0
        )
        et_turn = _effective_enabled_tools_for_turn(
            base_enabled_tools=et_turn0,
            enabled_toolsets=enabled_toolsets,
            user_query=user_query,
        )
        if verbose >= 1:
            d0 = (
                f"trigger match: longest substring {tr0!r} (skill {sid0!r})"
                if sid0 and tr0
                else "trigger match: no skill (no trigger substring matched)"
            )
            _print_skill_usage_verbose(
                verbose,
                source="repl",
                skill_id=sid0,
                base_tools=enabled_tools,
                effective_tools=et_turn,
                detail=d0,
            )
        sprompt0 = (skills_m.get(sid0) or {}).get("prompt") if sid0 else None
        router_query = _route_requires_websearch(
            user_query,
            today_str,
            primary_profile,
            et_turn,
            transcript_messages=messages,
        )
        if _deliverable_skip_mandatory_web(user_query):
            router_query = None
        web_required = bool(router_query)
        turn_msg = _interactive_turn_user_message(
            user_query,
            today_str,
            second_opinion_on,
            cloud_ai_enabled,
            primary_profile=primary_profile,
            reviewer_ollama_model=reviewer_ollama_model,
            reviewer_hosted_profile=reviewer_hosted_profile,
            enabled_tools=et_turn,
            system_instruction_override=session_system_prompt,
            skill_suffix=sprompt0,
        )
        messages.append({"role": "user", "content": turn_msg})
        if router_query and "search_web" in et_turn:
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
        answered, final_answer = _run_agent_conversation_turn(
            messages,
            user_query,
            today_str,
            web_required=web_required,
            deliverable_wanted=deliverable_wanted,
            verbose=verbose,
            second_opinion_enabled=second_opinion_on,
            cloud_ai_enabled=cloud_ai_enabled,
            primary_profile=primary_profile,
            reviewer_hosted_profile=reviewer_hosted_profile,
            reviewer_ollama_model=reviewer_ollama_model,
            enabled_tools=et_turn,
            interactive_tool_recovery=True,
            context_cfg=context_cfg,
        )
        if session_save_path:
            try:
                _save_context_bundle(
                    session_save_path, messages, user_query, final_answer, answered
                )
            except OSError as e:
                print(f"Warning: could not save context: {e}", file=sys.stderr)

    while True:
        try:
            line = _repl_read_line("> ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            # Ctrl-C at the prompt: cancel the current line, keep REPL alive.
            print("\n[Cancelled]\n")
            continue
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low in ("/quit", "/exit", "/q"):
            break
        if low == "/clear":
            messages.clear()
            last_reuse_skill_id = None
            print("Context cleared (including stored skill for /skill reuse).")
            continue
        if low == "/models":
            try:
                names = _fetch_ollama_local_model_names()
                if names:
                    print("\n".join(names))
                else:
                    print("(no models returned)")
            except Exception as e:
                print(f"/models error: {e}")
            continue
        if low in ("/usage", "/tokens"):
            print(_format_last_ollama_usage_for_repl())
            continue
        if s.startswith("/show"):
            try:
                toks = shlex.split(s)
            except ValueError as e:
                print(f"/show: {e}")
                continue
            if len(toks) < 2 or toks[1].lower() in ("help", "-h", "--help"):
                print(
                    "Usage:\n"
                    "  /show model      Primary LLM in use (Ollama or hosted)\n"
                    "  /show reviewer   Second-opinion reviewer model\n"
                    "\n"
                    "Settings that already have a show line: /settings tools, /settings context show, "
                    "/settings thinking show, /settings system_prompt show, /settings prompt_template show, "
                    "/settings ollama|openai|agent show"
                )
                continue
            sub = toks[1].lower().replace("-", "_")
            if sub in ("model", "primary", "llm"):
                print(f"Primary LLM: {_format_session_primary_llm_line(primary_profile)}")
                continue
            if sub in ("reviewer", "second_opinion", "2nd"):
                print(
                    f"Second-opinion reviewer: {_format_session_reviewer_line(reviewer_hosted_profile, reviewer_ollama_model)}"
                )
                continue
            print("Unknown /show topic. Try: /show model   or   /show reviewer")
            continue
        if s.startswith("/while"):
            try:
                wtoks = shlex.split(s)
            except ValueError as e:
                print(f"/while: {e}")
                continue
            if len(wtoks) == 1 or (
                len(wtoks) == 2 and wtoks[1].lower() in ("help", "-h", "--help")
            ):
                print(
                    "Usage:\n"
                    "  /while [--max N] <condition> do <action>\n"
                    "  <condition> and <action> are shlex-quoted; use double or single quotes (use the other kind for quotes inside).\n"
                    "  Like C while (CONDITION) { … }: the judge returns whether CONDITION is TRUE or FALSE right now.\n"
                    "    1 = TRUE — stay in the loop (run <action>, then re-check).\n"
                    "    0 = FALSE — exit the loop (do not run <action>).\n"
                    "  Default --max is 50 iterations (each iteration: one judge + at most one body).\n"
                    "\n"
                    "  Body: one or more comma-separated quoted prompts (shlex; add a space before each comma if your\n"
                    "  shell glues commas, e.g.  \"a\" , \"b\"  or  \"a\", \"b\"  both work).\n"
                    "  Examples:\n"
                    '    /while "pytest is still failing" do "fix from output and run pytest"\n'
                    "    /while 'work remains' do 'step A', 'step B', 'step C'\n"
                    "    /while --max 10 'server not yet returning 200' do 'patch and curl until OK'\n"
                )
                continue
            try:
                max_while, while_cond, body_prompts = _parse_while_repl_tokens(wtoks)
            except ValueError as e:
                print(f"/while: {e}")
                continue
            try:
                abort_while = False
                for wit in range(1, max_while + 1):
                    try:
                        bit = _call_while_condition_judge(
                            while_cond,
                            messages,
                            primary_profile=primary_profile,
                            verbose=verbose,
                        )
                    except KeyboardInterrupt:
                        _agent_progress("Cancelled /while (condition check).")
                        print("\n[Cancelled]\n")
                        break
                    if bit == 0:
                        print(
                            f"/while: condition false (judge returned 0). "
                            f"Exiting after check {wit}/{max_while}."
                        )
                        break
                    n_steps = len(body_prompts)
                    for si, bp in enumerate(body_prompts, start=1):
                        uq = (
                            f"[ /while iteration {wit}/{max_while} "
                            f"step {si}/{n_steps} ]\n"
                            f"{bp}"
                        )
                        _agent_progress(
                            f"/while: iteration {wit}/{max_while} step {si}/{n_steps}"
                        )
                        try:
                            repl_run_normal_user_request(uq)
                        except KeyboardInterrupt:
                            _agent_progress("Cancelled /while (body).")
                            print("\n[Cancelled]\n")
                            abort_while = True
                            break
                    if abort_while:
                        break
                else:
                    print(
                        f"/while: reached --max {max_while} without judge returning 0 (exit)."
                    )
            except Exception as e:
                print(f"/while error: {e}")
            continue
        if low.startswith("/skill"):
            try:
                toks = shlex.split(s)
            except ValueError as e:
                print(f"/skill: {e}")
                continue
            if len(toks) < 2 or toks[1].lower() in ("help", "-h", "--help"):
                print(_SKILL_HELP_TEXT)
                continue
            sub = toks[1].strip()
            if sub.lower() in ("help", "-h", "--help", "explain"):
                print(_SKILL_HELP_TEXT)
                continue
            if sub.lower() in ("list", "ls"):
                if not skills_m:
                    print("(no skills loaded)")
                else:
                    print("Skills:")
                    for sid in sorted(skills_m.keys()):
                        rec = skills_m.get(sid) or {}
                        desc = (rec.get("description") or "").strip() if isinstance(rec, dict) else ""
                        print(f"- {sid}" + (f": {desc}" if desc else ""))
                continue
            if sub.lower() == "auto":
                req = " ".join(toks[2:]).strip()
                if not req:
                    print("Usage: /skill auto <request>")
                    continue
                sid, why = _ml_select_skill_id(
                    req, skills_m, primary_profile=primary_profile, verbose=verbose
                )
                if not sid:
                    print(f"/skill auto: no skill selected. {why}".strip())
                    continue
                repl_run_with_selected_skill(
                    req, sid, source="auto", selection_rationale=why
                )
                continue
            if sub.lower() == "reuse":
                req = " ".join(toks[2:]).strip()
                if not req:
                    print("Usage: /skill reuse <request>")
                    continue
                if not last_reuse_skill_id:
                    print(
                        "/skill reuse: no stored skill. Run /skill auto <request> or /skill <id> <request> first."
                    )
                    continue
                sid2 = last_reuse_skill_id
                if sid2 not in skills_m:
                    print(
                        f"/skill reuse: stored skill {sid2!r} is not in the current skill set. "
                        "Run /skill auto again (check skills_dir / /settings save)."
                    )
                    last_reuse_skill_id = None
                    continue
                repl_run_with_selected_skill(
                    req,
                    sid2,
                    source="reuse",
                    selection_rationale="Follow-up; model skill selector skipped; same id as last skill run.",
                )
                continue
            # Explicit skill id
            sid = sub
            req = " ".join(toks[2:]).strip()
            if not sid or not req:
                print("Usage: /skill <skill> <request>")
                continue
            if sid not in skills_m:
                print(
                    f"/skill: unknown skill {sid!r}. "
                    "Run /settings save if you changed skills_dir, or check your skills directory."
                )
                continue
            repl_run_with_selected_skill(
                req,
                sid,
                source="explicit",
                selection_rationale="Explicit skill id; model skill selector skipped.",
            )
            continue

        # Back-compat aliases (not documented):
        if low.startswith("/use-skills"):
            try:
                toks = shlex.split(s)
            except ValueError as e:
                print(f"/use-skills: {e}")
                continue
            if len(toks) < 2:
                print("Usage: /use-skills <user request>")
                continue
            req = " ".join(toks[1:]).strip()
            if not req:
                print("Usage: /use-skills <user request>")
                continue
            sid, why = _ml_select_skill_id(
                req, skills_m, primary_profile=primary_profile, verbose=verbose
            )
            if not sid:
                print(f"/use-skills: no skill selected. {why}".strip())
                continue
            repl_run_with_selected_skill(req, sid, source="auto", selection_rationale=why)
            continue
        if low.startswith("/use-skill"):
            try:
                toks = shlex.split(s)
            except ValueError as e:
                print(f"/use-skill: {e}")
                continue
            if len(toks) < 3:
                print("Usage: /use-skill <skill> <user request>")
                continue
            sid = toks[1].strip()
            req = " ".join(toks[2:]).strip()
            if not sid or not req:
                print("Usage: /use-skill <skill> <user request>")
                continue
            if sid not in skills_m:
                print(
                    f"/use-skill: unknown skill {sid!r}. "
                    "Run /settings save if you changed skills_dir, or check your skills directory."
                )
                continue
            repl_run_with_selected_skill(
                req,
                sid,
                source="explicit",
                selection_rationale="Explicit skill id; model skill selector skipped.",
            )
            continue
        if low.startswith("/reuse-skill"):
            try:
                toks = shlex.split(s)
            except ValueError as e:
                print(f"/reuse-skill: {e}")
                continue
            if len(toks) < 2:
                print("Usage: /reuse-skill <follow-up request (same skill as last /use-skills or /reuse-skill)>")
                continue
            req = " ".join(toks[1:]).strip()
            if not req:
                print("Usage: /reuse-skill <follow-up request>")
                continue
            if not last_reuse_skill_id:
                print(
                    "/reuse-skill: no stored skill. Run /use-skills <request> first, "
                    "or use a normal line for trigger-based skills."
                )
                continue
            sid2 = last_reuse_skill_id
            if sid2 not in skills_m:
                print(
                    f"/reuse-skill: stored skill {sid2!r} is not in the current skill set. "
                    "Run /use-skills again (check skills_dir / /settings save)."
                )
                last_reuse_skill_id = None
                continue
            repl_run_with_selected_skill(
                req,
                sid2,
                source="reuse",
                selection_rationale="Follow-up; model skill selector skipped; same id as last skill run.",
            )
            continue
        if low.startswith("/settings"):
            try:
                toks = shlex.split(s)
            except ValueError as e:
                print(f"/settings: {e}")
                continue
            if len(toks) < 2:
                print(_SETTINGS_HELP_TEXT)
                continue
            key = toks[1].lower().replace("-", "_")
            if key in ("help", "-h", "--help"):
                print(_SETTINGS_HELP_TEXT)
                continue
            if key in ("ollama", "openai", "agent"):
                if len(toks) < 3:
                    print(
                        f"Usage: /settings {key} show | keys | set <name> <value> | unset <name>\n"
                        "  Keys are lowercase (e.g. host, model, api_key). "
                        "After changing, use /settings save."
                    )
                    print(_settings_group_keys_lines(key))
                    continue
                sub = toks[2].lower()
                if sub in ("show", "list"):
                    try:
                        print(_settings_group_show(key))
                    except (ValueError, OSError) as e:
                        print(f"/settings {key} show: {e}")
                    continue
                if sub in ("keys", "key", "help"):
                    try:
                        print(_settings_group_keys_lines(key))
                    except (ValueError, OSError) as e:
                        print(f"/settings {key} keys: {e}")
                    continue
                if sub == "set":
                    if len(toks) < 4:
                        print(f"Usage: /settings {key} set <name> <value (optional, quote spaces with shlex)>")
                        continue
                    raw_k = toks[3]
                    value = " ".join(toks[4:]) if len(toks) > 4 else ""
                    try:
                        msg = _settings_group_set(key, raw_k, value)
                    except ValueError as e:
                        print(f"/settings {key} set: {e}")
                        continue
                    print(msg)
                    continue
                if sub in ("unset", "delete", "clear"):
                    if len(toks) < 4:
                        print(f"Usage: /settings {key} unset <name>")
                        continue
                    try:
                        msg = _settings_group_unset(key, toks[3])
                    except ValueError as e:
                        print(f"/settings {key} unset: {e}")
                        continue
                    print(msg)
                    continue
                print(f"Unknown /settings {key} subcommand. Try: /settings {key} show | set | unset | keys")
                continue
            if key == "verbose":
                if len(toks) != 3:
                    print("Usage: /settings verbose 0|1|2|on|off")
                    continue
                tok = toks[2].strip().lower()
                if tok == "on":
                    verbose = 2
                elif tok == "off":
                    verbose = 0
                elif tok in ("0", "1", "2"):
                    verbose = int(tok)
                else:
                    print("Usage: /settings verbose 0|1|2|on|off")
                    continue
                print(_verbose_ack_message(verbose))
                continue
            if key == "tools":
                if len(toks) == 2 or (len(toks) >= 3 and toks[2].lower() in ("list", "ls", "show")):
                    print(_format_settings_tools_list(enabled_tools))
                    if _PLUGIN_TOOLSETS:
                        lines = ["\nToolsets (plugins):"]
                        for nm in sorted(_PLUGIN_TOOLSETS.keys()):
                            on = "on" if nm in enabled_toolsets else "off"
                            desc = str((_PLUGIN_TOOLSETS.get(nm) or {}).get("description") or "").strip()
                            lines.append(f"  [{on}] {nm}" + (f" — {desc}" if desc else ""))
                            # Show tools inside each toolset with effective availability.
                            tnames = sorted(_plugin_tools_for_toolset(nm))
                            for tid in tnames:
                                td_on = (nm in enabled_toolsets) and (tid in enabled_tools)
                                reason = ""
                                if nm not in enabled_toolsets:
                                    reason = " (toolset off)"
                                elif tid not in enabled_tools:
                                    reason = " (tool disabled)"
                                lines.append(f"       - {'on' if td_on else 'off'} {tid}{reason}")
                        lines.append("Enable a toolset:  /settings tools enable <toolset>")
                        lines.append("Disable a toolset: /settings tools disable <toolset>")
                        lines.append("Reload plugins:    /settings tools reload")
                        lines.append("Describe a tool:   /settings tools describe <tool-id>")
                        print("\n".join(lines))
                    continue
                if len(toks) >= 4 and toks[2].lower() in ("enable", "on"):
                    nm = toks[3].strip().lower()
                    if nm in _PLUGIN_TOOLSETS:
                        enabled_toolsets.add(nm)
                        # Enabling a toolset also enables its tools unless they were explicitly disabled.
                        for tid in _plugin_tools_for_toolset(nm):
                            enabled_tools.add(tid)
                        print(f"Toolset enabled: {nm!r} (tools may be routed per request). Use /settings save to persist.")
                    else:
                        print(f"Unknown toolset {nm!r}. Try: /settings tools")
                    continue
                if len(toks) >= 4 and toks[2].lower() in ("disable", "off"):
                    nm = toks[3].strip().lower()
                    if nm in _PLUGIN_TOOLSETS:
                        enabled_toolsets.discard(nm)
                        # Also disable its tools explicitly if they were ever enabled individually.
                        for tid in _plugin_tools_for_toolset(nm):
                            enabled_tools.discard(tid)
                        print(f"Toolset disabled: {nm!r}. Use /settings save to persist.")
                    else:
                        print(f"Unknown toolset {nm!r}. Try: /settings tools")
                    continue
                if len(toks) >= 3 and toks[2].lower() in ("reload", "refresh"):
                    _load_plugin_toolsets(session_tools_dir)
                    _register_tool_aliases()
                    print(f"Reloaded plugin toolsets from {session_tools_dir!r}.")
                    continue
                if len(toks) >= 4 and toks[2].lower() in ("describe", "desc", "help"):
                    tid = toks[3].strip()
                    if not tid:
                        print("Usage: /settings tools describe <tool-id>")
                        continue
                    # Allow toolset name too: describe the toolset and list its tools.
                    nm = tid.strip().lower()
                    if nm in _PLUGIN_TOOLSETS:
                        rec = _PLUGIN_TOOLSETS.get(nm) or {}
                        desc = str(rec.get("description") or "").strip()
                        print(f"Toolset: {nm}\nDescription: {desc if desc else '(none)'}")
                        print("Tools:")
                        for one in sorted(_plugin_tools_for_toolset(nm)):
                            print("  - " + one)
                        continue
                    print(_describe_tool_call_contract(tid))
                    continue
                print("Usage: /settings tools [list] | enable <toolset> | disable <toolset>")
                continue
            if key == "system_prompt":
                if len(toks) < 3:
                    print(
                        "Usage:\n"
                        "  /settings system_prompt show\n"
                        "  /settings system_prompt reset\n"
                        "  /settings system_prompt file <path>     Load UTF-8 file (session; /settings save stores path)\n"
                        "  /settings system_prompt save <path>     Write current effective prompt to a file\n"
                        "  /settings system_prompt <text>          One-line prompt (quote spaces with shlex)\n"
                    )
                    continue
                sub = toks[2].lower()
                if sub == "show":
                    body = _effective_system_instruction_text(session_system_prompt)
                    print(f"Effective system prompt ({len(body)} chars):\n{body}")
                    if session_system_prompt_path:
                        print(f"(File-backed: {session_system_prompt_path!r})")
                    elif session_system_prompt is not None:
                        print("(Session inline override.)")
                    else:
                        print("(Built-in default.)")
                    continue
                if sub in ("reset", "default"):
                    session_system_prompt = None
                    session_system_prompt_path = None
                    print("System prompt reset to built-in default for this session.")
                    continue
                if sub == "file":
                    if len(toks) < 4:
                        print("Usage: /settings system_prompt file <path>")
                        continue
                    path = os.path.expanduser(" ".join(toks[3:]).strip())
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            body = f.read()
                    except OSError as e:
                        print(f"/settings system_prompt file: {e}")
                        continue
                    if not body.strip():
                        print("File is empty.")
                        continue
                    session_system_prompt = body
                    session_system_prompt_path = os.path.abspath(path)
                    print(
                        f"System prompt loaded from {path!r} ({len(body)} chars). "
                        "/settings save will store this path in ~/.agent.json."
                    )
                    continue
                if sub == "save":
                    if len(toks) < 4:
                        print("Usage: /settings system_prompt save <path>")
                        continue
                    path = os.path.expanduser(" ".join(toks[3:]).strip())
                    body = _effective_system_instruction_text(session_system_prompt)
                    try:
                        parent = os.path.dirname(path)
                        if parent and not os.path.isdir(parent):
                            os.makedirs(parent, exist_ok=True)
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(body)
                    except OSError as e:
                        print(f"/settings system_prompt save: {e}")
                        continue
                    print(f"Wrote system prompt ({len(body)} chars) to {path!r}.")
                    continue
                phrase = " ".join(toks[2:])
                if not phrase.strip():
                    print("Usage: /settings system_prompt <non-empty one-line text>")
                    continue
                session_system_prompt = phrase
                session_system_prompt_path = None
                print(
                    f"System prompt set inline ({len(phrase)} chars). "
                    "/settings save will store the text in ~/.agent.json."
                )
                continue
            if key in ("prompt_template", "prompt_templates", "prompt"):
                if len(toks) < 3:
                    print(
                        "Usage:\n"
                        "  /settings prompt_template list\n"
                        "  /settings prompt_template show\n"
                        "  /settings prompt_template use <name>\n"
                        "  /settings prompt_template default <name>\n"
                        "  /settings prompt_template set <name> <text>\n"
                        "  /settings prompt_template delete <name>\n"
                    )
                    continue
                sub = toks[2].lower()
                if sub in ("help", "-h", "--help", "explain"):
                    print(_PROMPT_TEMPLATE_HELP_TEXT)
                    continue
                if sub == "list":
                    names = sorted(templates.keys())
                    if not names:
                        print("(no prompt templates)")
                        continue
                    for nm in names:
                        obj = templates.get(nm) or {}
                        desc = ""
                        if isinstance(obj, dict):
                            desc = str(obj.get("description") or "").strip()
                        mark = ""
                        if session_prompt_template == nm:
                            mark = " *active*"
                        elif template_default == nm:
                            mark = " (default)"
                        line = f"- {nm}{mark}"
                        if desc:
                            line += f": {desc}"
                        print(line)
                    continue
                if sub == "show":
                    active = session_prompt_template or template_default
                    body = _resolve_prompt_template_text(active, templates) or ""
                    print(f"Active template: {active!r}\nPrompt ({len(body)} chars):\n{body}")
                    continue
                if sub in ("use", "select"):
                    if len(toks) < 4:
                        print("Usage: /settings prompt_template use <name>")
                        continue
                    nm = toks[3].strip()
                    if nm not in templates:
                        print(f"Unknown template {nm!r}. Try: /settings prompt_template list")
                        continue
                    resolved = _resolve_prompt_template_text(nm, templates)
                    if not resolved:
                        print(f"Template {nm!r} has no usable text/path.")
                        continue
                    session_system_prompt = resolved
                    session_system_prompt_path = None
                    session_prompt_template = nm
                    print(f"Using prompt template {nm!r} for this session.")
                    continue
                if sub == "default":
                    if len(toks) < 4:
                        print("Usage: /settings prompt_template default <name>")
                        continue
                    nm = toks[3].strip()
                    if nm not in templates:
                        print(f"Unknown template {nm!r}. Try: /settings prompt_template list")
                        continue
                    template_default = nm
                    print(
                        f"Default prompt template set to {nm!r} (use /settings save to persist)."
                    )
                    continue
                if sub == "set":
                    if len(toks) < 5:
                        print("Usage: /settings prompt_template set <name> <text>")
                        continue
                    nm = toks[3].strip()
                    text = " ".join(toks[4:]).strip()
                    if not nm:
                        print("Template name must be non-empty.")
                        continue
                    if not text:
                        print("Template text must be non-empty.")
                        continue
                    templates[nm] = {
                        "kind": "overlay",
                        "description": (templates.get(nm) or {}).get("description", "")
                        if isinstance(templates.get(nm), dict)
                        else "",
                        "text": text,
                    }
                    print(
                        f"Template {nm!r} set/updated (overlay). Use /settings save to persist."
                    )
                    continue
                if sub in ("delete", "del", "rm", "remove"):
                    if len(toks) < 4:
                        print("Usage: /settings prompt_template delete <name>")
                        continue
                    nm = toks[3].strip()
                    on_disk = os.path.join(session_pt_dir, f"{nm}.json")
                    if os.path.isfile(on_disk):
                        print(
                            "Refusing to delete a template that exists as a file on disk in "
                            f"the configured prompt_templates_dir ({session_pt_dir!r}). "
                            "You can override it in ~/.agent.json with a same-named entry."
                        )
                        continue
                    if nm not in templates:
                        print(f"Unknown template {nm!r}.")
                        continue
                    templates.pop(nm, None)
                    if session_prompt_template == nm:
                        session_prompt_template = None
                    print(f"Deleted template {nm!r}. Use /settings save to persist.")
                    continue
                print("Unknown subcommand. Try: /settings prompt_template list")
                continue
            if key in ("context", "context_manager", "context_window"):
                if len(toks) < 3:
                    print(
                        "Usage:\n"
                        "  /settings context show\n"
                        "  /settings context on|off\n"
                        "  /settings context tokens <n>\n"
                        "  /settings context trigger <0..1>\n"
                        "  /settings context target <0..1>\n"
                        "  /settings context keep_tail <n>\n"
                    )
                    continue
                sub = toks[2].lower()
                if sub == "show":
                    print(
                        "Context manager (prefs; env vars may override):\n"
                        f"  enabled: {bool(context_cfg.get('enabled', True))}\n"
                        f"  tokens: {context_cfg.get('tokens', 0)}  (0 = auto per backend)\n"
                        f"  trigger_frac: {context_cfg.get('trigger_frac', 0.75)}\n"
                        f"  target_frac: {context_cfg.get('target_frac', 0.55)}\n"
                        f"  keep_tail_messages: {context_cfg.get('keep_tail_messages', 12)}\n"
                    )
                    continue
                if sub in ("on", "enable", "enabled", "true"):
                    context_cfg["enabled"] = True
                    print("Context manager enabled for this session. Use /settings save to persist.")
                    continue
                if sub in ("off", "disable", "disabled", "false"):
                    context_cfg["enabled"] = False
                    print("Context manager disabled for this session. Use /settings save to persist.")
                    continue
                if sub == "tokens":
                    if len(toks) < 4:
                        print("Usage: /settings context tokens <n>")
                        continue
                    try:
                        n = int(toks[3], 10)
                    except ValueError:
                        print("tokens must be an integer.")
                        continue
                    if n < 0:
                        n = 0
                    context_cfg["tokens"] = n
                    print(f"context tokens set to {n} (0 = auto). Use /settings save to persist.")
                    continue
                if sub == "trigger":
                    if len(toks) < 4:
                        print("Usage: /settings context trigger <0..1>")
                        continue
                    try:
                        x = float(toks[3])
                    except ValueError:
                        print("trigger must be a number.")
                        continue
                    context_cfg["trigger_frac"] = max(0.05, min(0.95, x))
                    print(
                        f"trigger_frac set to {context_cfg['trigger_frac']}. Use /settings save to persist."
                    )
                    continue
                if sub == "target":
                    if len(toks) < 4:
                        print("Usage: /settings context target <0..1>")
                        continue
                    try:
                        x = float(toks[3])
                    except ValueError:
                        print("target must be a number.")
                        continue
                    cur_tr = float(context_cfg.get("trigger_frac", 0.75))
                    context_cfg["target_frac"] = max(0.05, min(cur_tr, x))
                    print(
                        f"target_frac set to {context_cfg['target_frac']}. Use /settings save to persist."
                    )
                    continue
                if sub in ("keep_tail", "keep", "tail"):
                    if len(toks) < 4:
                        print("Usage: /settings context keep_tail <n>")
                        continue
                    try:
                        n = int(toks[3], 10)
                    except ValueError:
                        print("keep_tail must be an integer.")
                        continue
                    context_cfg["keep_tail_messages"] = max(4, n)
                    print(
                        f"keep_tail_messages set to {context_cfg['keep_tail_messages']}. Use /settings save to persist."
                    )
                    continue
                print("Unknown subcommand. Try: /settings context show")
                continue
            if key == "save":
                if len(toks) != 2:
                    print("Usage: /settings save")
                    continue
                try:
                    payload = _build_agent_prefs_payload(
                        primary_profile=primary_profile,
                        second_opinion_on=second_opinion_on,
                        cloud_ai_enabled=cloud_ai_enabled,
                        enabled_tools=enabled_tools,
                        enabled_toolsets=enabled_toolsets,
                        reviewer_hosted_profile=reviewer_hosted_profile,
                        reviewer_ollama_model=reviewer_ollama_model,
                        session_save_path=session_save_path,
                        system_prompt_override=session_system_prompt,
                        system_prompt_path_override=session_system_prompt_path,
                        prompt_templates=templates,
                        prompt_template_default=template_default,
                        prompt_templates_dir=session_pt_dir,
                        skills_dir=session_skills_dir,
                        tools_dir=session_tools_dir,
                        context_manager=context_cfg,
                        verbose_level=verbose,
                    )
                    _write_agent_prefs_file(payload)
                except OSError as e:
                    print(f"/settings save error: {e}")
                    continue
                print(f"Saved settings to {_agent_prefs_path()!r}.")
                continue
            if key == "model":
                if len(toks) < 3:
                    print("Usage: /settings model <ollama-model-name>")
                    continue
                name = toks[2].strip()
                if not name:
                    print("Usage: /settings model <ollama-model-name>")
                    continue
                _settings_set(("ollama", "model"), name)
                print(f"ollama.model set to {name!r}. Use /settings save to persist.")
                continue
            if key == "enable":
                if len(toks) < 3:
                    print(
                        "Usage: /settings enable second_opinion|<tool or phrase>\n"
                        "  Examples: /settings enable web search   /settings enable shell   /settings enable stream_thinking\n"
                        "  See: /settings tools"
                    )
                    continue
                phrase = " ".join(toks[2:])
                feat = _canonicalize_user_tool_phrase(phrase)
                if feat == "second_opinion":
                    second_opinion_on = True
                    print("second_opinion enabled for this session.")
                    continue
                if feat in ("stream_thinking", "streamthinking", "stream_think", "thinking_stream", "showthinking", "show_thinking"):
                    _settings_set(("agent", "stream_thinking"), True)
                    print("stream_thinking enabled for this session (streams model thinking when available). Use /settings save to persist.")
                    continue
                if feat == "verbose":
                    verbose = 2
                    print(_verbose_ack_message(verbose))
                    continue
                tn = _normalize_tool_name(phrase)
                if tn:
                    enabled_tools.add(tn)
                    print(f"Tool enabled: {tn}")
                    continue
                print(_format_unknown_tool_hint(phrase))
                continue
            if key == "disable":
                if len(toks) < 3:
                    print(
                        "Usage: /settings disable second_opinion|<tool or phrase>\n"
                        "  Examples: /settings disable web search   /settings disable shell   /settings disable stream_thinking\n"
                        "  See: /settings tools"
                    )
                    continue
                phrase = " ".join(toks[2:])
                feat = _canonicalize_user_tool_phrase(phrase)
                if feat == "second_opinion":
                    second_opinion_on = False
                    print("second_opinion disabled for this session.")
                    continue
                if feat in ("stream_thinking", "streamthinking", "stream_think", "thinking_stream", "showthinking", "show_thinking"):
                    _settings_set(("agent", "stream_thinking"), False)
                    print("stream_thinking disabled for this session. Use /settings save to persist.")
                    continue
                if feat == "verbose":
                    verbose = 0
                    print(_verbose_ack_message(verbose))
                    continue
                tn = _normalize_tool_name(phrase)
                if tn:
                    enabled_tools.discard(tn)
                    print(f"Tool disabled: {tn}")
                    continue
                print(_format_unknown_tool_hint(phrase))
                continue
            if key == "thinking":
                if len(toks) < 3:
                    print(
                        "Usage:\n"
                        "  /settings thinking show\n"
                        "  /settings thinking on|off\n"
                        "  /settings thinking level low|medium|high\n"
                        "Notes:\n"
                        "  - This controls the Ollama request `think` field (bool or level string).\n"
                        "  - Some models ignore booleans and require levels; others support both.\n"
                        "  - thinking on/level also enables stream_thinking automatically (use /settings disable stream_thinking to hide).\n"
                        "  - Use /settings save to persist.\n"
                    )
                    continue
                sub = toks[2].lower()
                if sub == "show":
                    think_v = _ollama_request_think_value()
                    lvl = _agent_thinking_level()
                    on = _agent_thinking_enabled_default_false()
                    st = "on" if on else "off"
                    print(
                        f"thinking: {st}; level: {lvl or '(none)'}; ollama think value: {think_v!r}; stream_thinking: {_agent_stream_thinking_enabled()}"
                    )
                    continue
                if sub in ("on", "enable", "enabled", "true"):
                    _settings_set(("agent", "thinking"), True)
                    _settings_set(("agent", "stream_thinking"), True)
                    print(
                        "thinking enabled for this session (and stream_thinking enabled). "
                        "Use /settings save to persist."
                    )
                    continue
                if sub in ("off", "disable", "disabled", "false"):
                    _settings_set(("agent", "thinking"), False)
                    _settings_set(("agent", "thinking_level"), "")
                    _settings_set(("agent", "stream_thinking"), False)
                    print(
                        "thinking disabled for this session (and stream_thinking disabled). "
                        "Use /settings save to persist."
                    )
                    continue
                if sub == "level":
                    if len(toks) < 4:
                        print("Usage: /settings thinking level low|medium|high")
                        continue
                    lvl = toks[3].strip().lower()
                    if lvl not in ("low", "medium", "high"):
                        print("thinking level must be one of: low, medium, high")
                        continue
                    _settings_set(("agent", "thinking_level"), lvl)
                    _settings_set(("agent", "thinking"), True)
                    _settings_set(("agent", "stream_thinking"), True)
                    print(
                        f"thinking level set to {lvl!r} for this session (and stream_thinking enabled). "
                        "Use /settings save to persist."
                    )
                    continue
                print("Unknown /settings thinking subcommand. Try: /settings thinking show | on | off | level …")
                continue
            if key == "primary" and len(toks) >= 4 and toks[2].lower() == "llm":
                sub = toks[3].lower()
                if sub == "ollama":
                    primary_profile = default_primary_llm_profile()
                    print("Primary LLM: local Ollama.")
                elif sub == "hosted":
                    if len(toks) < 6:
                        print(
                            "Usage: /settings primary llm hosted <base_url> <model> [api_key]"
                        )
                        continue
                    bu, mod = toks[4], toks[5]
                    if not bu.startswith(("http://", "https://")):
                        print("base_url must start with http:// or https://")
                        continue
                    keyval = toks[6] if len(toks) > 6 else ""
                    primary_profile = LlmProfile(
                        backend="hosted",
                        base_url=bu,
                        model=mod,
                        api_key=keyval,
                    )
                    if not (keyval or "").strip():
                        print(
                            "Note: api_key is not set; hosted primary calls will fail until it is."
                        )
                    print(
                        "Primary LLM: hosted OpenAI-compatible API "
                        f"({_describe_llm_profile_short(primary_profile)})."
                    )
                else:
                    print("Usage: /settings primary llm ollama|hosted …")
                continue
            if (
                toks[1].replace("-", "_").lower() == "second_opinion"
                and len(toks) >= 4
                and toks[2].lower() == "llm"
            ):
                sub = toks[3].lower()
                if sub == "ollama":
                    reviewer_hosted_profile = None
                    reviewer_ollama_model = toks[4] if len(toks) > 4 else None
                    om = reviewer_ollama_model or _ollama_second_opinion_model()
                    print(
                        f"Second-opinion reviewer: local Ollama, model {om!r}."
                    )
                elif sub == "hosted":
                    if len(toks) < 6:
                        print(
                            "Usage: /settings second_opinion llm hosted <base_url> <model> [api_key]"
                        )
                        continue
                    bu, mod = toks[4], toks[5]
                    if not bu.startswith(("http://", "https://")):
                        print("base_url must start with http:// or https://")
                        continue
                    keyval = toks[6] if len(toks) > 6 else ""
                    reviewer_hosted_profile = LlmProfile(
                        backend="hosted",
                        base_url=bu,
                        model=mod,
                        api_key=keyval,
                    )
                    reviewer_ollama_model = None
                    if not (keyval or "").strip():
                        print(
                            "Note: api_key is not set; hosted second opinion will fail until it is."
                        )
                    print(
                        "Second-opinion reviewer: hosted "
                        f"({_describe_llm_profile_short(reviewer_hosted_profile)})."
                    )
                else:
                    print("Usage: /settings second_opinion llm ollama|hosted …")
                continue
            print("Unknown /settings subcommand. Try /help.")
            continue
        if low.startswith("/load_context"):
            rest = s.split(None, 1)
            if len(rest) < 2:
                print("Usage: /load_context <file>")
                continue
            path = rest[1].strip()
            if not path:
                print("Usage: /load_context <file>")
                continue
            try:
                loaded = _load_context_messages(path)
            except (OSError, ValueError, json.JSONDecodeError) as e:
                print(f"/load_context error: {e}")
                continue
            messages[:] = loaded
            print(f"Loaded {len(loaded)} message(s) from {path!r}.")
            continue
        if low.startswith("/save_context"):
            rest = s.split(None, 1)
            if len(rest) < 2:
                print("Usage: /save_context <file>")
                continue
            path = rest[1].strip()
            if not path:
                print("Usage: /save_context <file>")
                continue
            try:
                _save_context_bundle(path, messages, "", None, False)
            except OSError as e:
                print(f"/save_context error: {e}")
                continue
            session_save_path = path
            print(
                f"Wrote current session to {path!r}; further turns auto-save there."
            )
            continue
        if low in ("/help", "/?"):
            print(
                "Commands:\n"
                "  /quit                    Exit\n"
                "  /clear                   Clear in-memory conversation\n"
                "  /help                    Help\n"
                "  /models                  List local Ollama models\n"
                "  /usage                   Last local Ollama usage\n"
                "  /show ...                Show current state (try /show help)\n"
                "  /skill ...               Skills (try /skill help)\n"
                "  /while ...               Loops (try /while help)\n"
                "  /settings ...            Configuration (try /settings help)\n"
                "  /load_context <file>     Replace session messages from JSON\n"
                "  /save_context <file>     Write session JSON; set auto-save path\n"
            )
            continue
        if s.startswith("/"):
            print(f"Unknown command {s.split()[0]!r}. Try /help.")
            continue

        try:
            repl_run_normal_user_request(s)
        except KeyboardInterrupt:
            _agent_progress("Cancelled current request (Ctrl-C).")
            print("\n[Cancelled]\n")
            continue

    _flush_repl_readline_history()


def _run_agent_conversation_turn(
    messages: list,
    user_query: str,
    today_str: str,
    *,
    web_required: bool,
    deliverable_wanted: bool,
    verbose: int,
    second_opinion_enabled: bool,
    cloud_ai_enabled: bool,
    primary_profile: Optional[LlmProfile] = None,
    reviewer_hosted_profile: Optional[LlmProfile] = None,
    reviewer_ollama_model: Optional[str] = None,
    enabled_tools: Optional[AbstractSet[str]] = None,
    interactive_tool_recovery: bool = False,
    context_cfg: Optional[dict] = None,
    print_answer: bool = True,
) -> Tuple[bool, Optional[str]]:
    et = _coerce_enabled_tools(enabled_tools)
    if web_required and "search_web" not in et:
        web_required = False
    seen_tool_fingerprints: set = set()
    reviewed_tool_need = False
    saw_strong_web_result = False
    answered = False
    tool_executed = False
    second_opinion_rounds = 0
    final_answer: Optional[str] = None
    deliverable_path: Optional[str] = None
    deliverable_read_ok = False
    deliverable_file_chars = 0
    for _ in range(30):
        messages = _maybe_compact_context_window(
            messages,
            user_query=user_query,
            primary_profile=primary_profile,
            verbose=verbose,
            context_cfg=context_cfg,
        )
        response_text = call_ollama_chat(
            messages, primary_profile, et, verbose=verbose
        )
        response_data = parse_agent_json(response_text)
        action = response_data.get("action")
        if action == "answer":
            # Robustness: if the model attempted JSON-only but truncated/malformed its JSON,
            # do not treat that as an answer. Ask for a clean retry.
            rt = (response_text or "").strip()
            if rt.startswith("{") and ("\"action\"" in rt or "'action'" in rt) and not rt.rstrip().endswith("}"):
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your last response looked like a JSON object but it was truncated/malformed "
                            "(missing closing braces/quotes). Respond again with a SINGLE valid JSON object "
                            "and no other text."
                        ),
                    }
                )
                continue
            if web_required and not saw_strong_web_result:
                # If routing determined web is required, do not allow a final answer
                # until we've observed at least one non-weak web result in this session.
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You must not answer from memory for this request because web verification is required. "
                            "No usable web results have been obtained yet (or they were empty/blocked). "
                            "Call search_web again with a different, more effective query, or fetch_page on a credible source URL "
                            "from any results you do have. Respond with JSON tool_call only."
                        ),
                    }
                )
                continue
            if (
                deliverable_wanted
                and deliverable_path
                and not deliverable_read_ok
            ):
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": _deliverable_followup_block(deliverable_path),
                    }
                )
                continue
            if deliverable_wanted and deliverable_read_ok and _answer_missing_written_body(
                response_data.get("answer") or "", deliverable_file_chars
            ):
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your answer is too short to be the requested multi-page document. "
                            "Use read_file to load the written file, then respond with action answer whose "
                            "answer field contains the FULL document text (the user asked for the document itself). "
                            "If the file is still too short, expand it with write_file and read_file again."
                        ),
                    }
                )
                continue
            na = (response_data.get("next_action") or "finalize").strip().lower()
            if na == "second_opinion":
                rationale = _scalar_to_str(response_data.get("rationale"), "").strip()
                primary = response_data.get("answer") or ""
                if not rationale:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "second_opinion requires a non-empty string field \"rationale\" explaining why "
                                "you want a review. Respond with JSON only."
                            ),
                        }
                    )
                    continue
                hosted_ready = _hosted_review_ready(
                    cloud_ai_enabled, reviewer_hosted_profile
                )
                if not second_opinion_enabled and not hosted_ready:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Second opinion is not available in this session. Respond with JSON only using "
                                '{"action":"answer","answer":"...","next_action":"finalize","rationale":"..."}.'
                            ),
                        }
                    )
                    continue
                # Session configuration picks the reviewer; model-supplied second_opinion_backend is ignored.
                backend = (
                    "ollama"
                    if second_opinion_enabled
                    else ("openai" if hosted_ready else "")
                )
                if second_opinion_rounds >= 3:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Second opinion limit reached for this session. Respond with JSON only using "
                                '{"action":"answer","answer":"...","next_action":"finalize","rationale":"..."}.'
                            ),
                        }
                    )
                    continue
                reviewer_msgs = _second_opinion_reviewer_messages(user_query, primary, rationale)
                if backend == "ollama":
                    rm = (reviewer_ollama_model or "").strip() or _ollama_second_opinion_model()
                    review = call_ollama_plaintext(reviewer_msgs, rm)
                else:
                    if (
                        reviewer_hosted_profile is not None
                        and reviewer_hosted_profile.backend == "hosted"
                        and (reviewer_hosted_profile.api_key or "").strip()
                    ):
                        review = call_hosted_chat_plain(
                            reviewer_msgs, reviewer_hosted_profile
                        )
                    else:
                        review = call_openai_chat_plain(reviewer_msgs)
                second_opinion_rounds += 1
                tool_executed = True
                reviewed_tool_need = True
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": _second_opinion_result_user_message(review),
                    }
                )
                continue
            if not reviewed_tool_need and not tool_executed:
                reviewed_tool_need = True
                messages.append({"role": "assistant", "content": response_text})
                proposed = response_data.get("answer") or ""
                router_q2 = _route_requires_websearch_after_answer(
                    user_query,
                    today_str,
                    proposed,
                    primary_profile,
                    et,
                    transcript_messages=messages,
                )
                if _deliverable_skip_mandatory_web(user_query):
                    router_q2 = None
                if router_q2:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Before finalizing, you MUST call the tool search_web to verify.\n"
                                "Respond with JSON only in tool_call form.\n"
                                f'Suggested query: "{router_q2}"'
                            ),
                        }
                    )
                    continue
                if deliverable_wanted:
                    follow = _deliverable_first_answer_followup(user_query, proposed)
                elif _is_self_capability_question(user_query):
                    follow = _self_capability_followup(user_query, proposed)
                else:
                    follow = _tool_need_review_followup(user_query, proposed)
                messages.append({"role": "user", "content": follow})
                continue
            messages.append({"role": "assistant", "content": response_text})
            ans_out = response_data.get("answer")
            if ans_out is None or (isinstance(ans_out, str) and not ans_out.strip()):
                # Models sometimes include the final JSON inside a longer string; if parsing
                # picked an incomplete object (e.g. {"action":"answer"}), recover.
                extracted = _extract_json_object_from_text(response_text)
                if extracted:
                    try:
                        recovered = parse_agent_json(extracted)
                    except Exception:
                        recovered = None
                    if isinstance(recovered, dict) and recovered.get("action") == "answer":
                        ra = recovered.get("answer")
                        if isinstance(ra, str) and ra.strip():
                            response_data = recovered
                            ans_out = ra
                if ans_out is None or (isinstance(ans_out, str) and not ans_out.strip()):
                    # Treat as invalid agent JSON rather than printing "None" or echoing JSON back.
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                'Your JSON had action "answer" but was missing a non-empty string field "answer". '
                                "Respond again with a SINGLE valid JSON object in this exact shape:\n"
                                '{"action":"answer","answer":"..."}\n'
                                "No other keys, and no other text."
                            ),
                        }
                    )
                    continue
            if print_answer:
                print(ans_out if ans_out is not None else "")
            final_answer = ans_out if isinstance(ans_out, str) else str(ans_out)
            answered = True
            break
        elif action == "error":
            messages.append({"role": "assistant", "content": response_text})
            err = response_data.get("error")
            print(f"Agent error: {err}")
            final_answer = str(err) if err is not None else None
            answered = True
            break
        elif action == "tool_call" or action in _all_known_tools():
            tool = response_data.get("tool")
            if tool == None:
                tool = action
            params = response_data.get("parameters", {})
            if not isinstance(params, dict):
                params = {}
            params = _merge_tool_param_aliases(tool, params)
            params = _ensure_tool_defaults(tool, params, user_query)
            fp = _tool_params_fingerprint(tool, params)
            orig_fp = fp
            # Re-reading the same path can return different content after write_file/replace_text;
            # do not dedupe read_file/tail_file.
            dedupe_ok = tool not in ("read_file", "tail_file")
            skipped_duplicate = bool(dedupe_ok and fp in seen_tool_fingerprints)
            policy_blocked = False
            if verbose >= 1:
                if skipped_duplicate:
                    print(f"[*] Skipping duplicate tool: {tool} (same logical parameters as earlier)")
                else:
                    if tool == "search_web":
                        print(
                            f"[*] Executing tool: {tool} ({_search_backend_banner_line()}) with {params}"
                        )
                    else:
                        print(f"[*] Executing tool: {tool} with {params}")
            if skipped_duplicate:
                result = (
                    "[Duplicate call skipped: this tool was already run with the same parameters "
                    "in this session. Use the earlier tool output in this conversation to answer.]"
                )
            else:
                tool_executed = True
                if verbose < 1:
                    _agent_progress(_tool_progress_message(tool, params))
                result = ""
                if tool in _all_known_tools() and tool not in et:
                    policy_blocked = True
                    result = (
                        f"Tool error: {tool} is disabled for this run (tool policy). "
                        "Pick a different allowed tool or respond with action answer."
                    )
                else:
                    try:
                        if tool == "search_web":
                            result = search_web(params.get("query"), params=params)
                        elif tool == "fetch_page":
                            result = fetch_page(params.get("url"))
                        elif tool == "run_command":
                            cmd = _scalar_to_str(params.get("command"), "")
                            if web_required and re.search(r"\b(curl|wget)\b", cmd):
                                result = (
                                    "Command error: blocked. When web verification is required, do not use run_command "
                                    "with curl/wget to fetch web content. Use fetch_page instead."
                                )
                            else:
                                result = run_command(cmd)
                        elif tool == "use_git":
                            result = use_git(params)
                        elif tool == "write_file":
                            result = write_file(params.get("path"), params.get("content"))
                        elif tool == "list_directory":
                            result = list_directory(params.get("path"))
                        elif tool == "read_file":
                            result = read_file(params.get("path"))
                        elif tool == "download_file":
                            result = download_file(params.get("url"), params.get("path"))
                        elif tool == "tail_file":
                            result = tail_file(params.get("path"), params.get("lines", 20))
                        elif tool == "replace_text":
                            result = replace_text(
                                params.get("path"),
                                params.get("pattern"),
                                params.get("replacement"),
                                params.get("replace_all", True),
                            )
                        elif tool == "call_python":
                            result = call_python(params.get("code"), params.get("globals"))
                        elif tool in _PLUGIN_TOOL_HANDLERS:
                            result = _PLUGIN_TOOL_HANDLERS[tool](params)
                        else:
                            result = f"Unknown tool: {tool}"
                    except KeyboardInterrupt:
                        raise
                    except BaseException as e:
                        result = _tool_fault_result(str(tool), e)
            if (
                _tool_recovery_may_run(interactive_tool_recovery)
                and not skipped_duplicate
                and not policy_blocked
                and tool in _TOOL_RECOVERY_TOOLS
                and _tool_result_indicates_retryable_failure(tool, result)
            ):
                old_params = dict(params)
                sug = _suggest_tool_recovery_params(
                    tool,
                    old_params,
                    result,
                    user_query,
                    primary_profile,
                    et,
                    verbose,
                )
                if sug is not None:
                    new_params, rationale = sug
                    new_fp = _tool_params_fingerprint(tool, new_params)
                    if new_fp == orig_fp:
                        if verbose >= 1:
                            print("[*] Tool recovery: proposed parameters unchanged; skip retry.")
                    elif dedupe_ok and new_fp in seen_tool_fingerprints:
                        if verbose >= 1:
                            print(
                                "[*] Tool recovery: proposed parameters match an earlier "
                                "tool call; skip retry."
                            )
                    elif _confirm_tool_recovery_retry(
                        tool,
                        old_params,
                        new_params,
                        rationale,
                        interactive_tool_recovery=interactive_tool_recovery,
                    ):
                        params = new_params
                        fp = new_fp
                        if verbose >= 1:
                            print(f"[*] Re-running {tool} after confirmed recovery.")
                        else:
                            _agent_progress(f"Tool: {tool} (retry)")
                        tool_executed = True
                        if tool == "run_command":
                            cmd = _scalar_to_str(params.get("command"), "")
                            if web_required and re.search(r"\b(curl|wget)\b", cmd):
                                result = (
                                    "Command error: blocked. When web verification is required, do not use run_command "
                                    "with curl/wget to fetch web content. Use fetch_page instead."
                                )
                            else:
                                result = run_command(cmd)
                        elif tool == "call_python":
                            result = call_python(params.get("code"), params.get("globals"))
                        elif tool == "search_web":
                            result = search_web(params.get("query"), params=params)
                        elif tool == "fetch_page":
                            result = fetch_page(params.get("url"))
                        note = "[After one user-confirmed corrected retry]\n"
                        if isinstance(result, str) and not result.startswith(
                            "[After one user-confirmed corrected retry]"
                        ):
                            result = note + result
                    elif verbose >= 1:
                        print("[*] Tool recovery: retry not confirmed.")
            if tool == "write_file" and deliverable_wanted and not policy_blocked:
                wp = _scalar_to_str(params.get("path"), "").strip()
                if wp and (not str(result).startswith("Write error:")):
                    deliverable_path = wp
                    deliverable_read_ok = False
                    deliverable_file_chars = 0
            if tool == "read_file" and deliverable_wanted and deliverable_path and not policy_blocked:
                rp = _scalar_to_str(params.get("path"), "").strip()
                if rp == deliverable_path and (not str(result).startswith("Read error:")):
                    deliverable_read_ok = True
                    deliverable_file_chars = len(str(result))
            if (
                tool == "search_web"
                and not skipped_duplicate
                and not policy_blocked
                and not _is_tool_result_weak_for_dedup(result)
            ):
                saw_strong_web_result = True
            # Record fingerprints so identical parameters cannot loop forever; keep both the
            # first attempt and a corrected retry when they differ.
            if dedupe_ok and not skipped_duplicate and not policy_blocked:
                if orig_fp not in seen_tool_fingerprints:
                    seen_tool_fingerprints.add(orig_fp)
                if fp != orig_fp and fp not in seen_tool_fingerprints:
                    seen_tool_fingerprints.add(fp)
            deliverable_reminder = ""
            if deliverable_wanted and deliverable_path and not deliverable_read_ok:
                deliverable_reminder = (
                    f"Goal reminder (user request): {user_query}\n"
                    + _deliverable_followup_block(deliverable_path)
                )
            elif deliverable_wanted and not deliverable_path:
                deliverable_reminder = (
                    f"Goal reminder (user request): {user_query}\n"
                    "If you will satisfy this with write_file, plan to read_file that same path before answering."
                )
            messages.append({"role": "assistant", "content": response_text})
            messages.append(
                {
                    "role": "user",
                    "content": _tool_result_user_message(
                        tool, params, result, deliverable_reminder=deliverable_reminder
                    ),
                }
            )
        else:
            # Malformed model JSON (e.g. action=null) — recover instead of exiting.
            messages.append({"role": "assistant", "content": response_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your last message was not valid agent JSON. "
                        "Respond with JSON only and include a non-null string action. "
                        'Use {"action":"tool_call","tool":<one of the allowed tools>,'
                        '"parameters":{...}} or {"action":"answer","answer":"..."}.'
                    ),
                }
            )
            continue
    if not answered:
        if web_required and not saw_strong_web_result:
            print(
                "Unable to verify with web: no strong search result (URL-backed) was obtained in this turn. "
                "Refusing to answer from memory alone. "
                "Try again with a more specific query, fetch_page on a URL the user provided, "
                "or check network / site blocking."
            )
        else:
            print("Unable to complete the request within the step limit.")
    return answered, final_answer


def main():
    argv = _parse_and_apply_cli_config_flag(list(sys.argv[1:]))
    raw_prefs = _load_agent_prefs()
    st = _session_defaults_from_prefs(raw_prefs)
    # Reload plugin toolsets after prefs/env are applied so AGENT_TOOLS_DIR / tools_dir override works.
    try:
        _load_plugin_toolsets(_resolved_tools_dir(raw_prefs))
        _register_tool_aliases()
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

    prompt_templates = (
        st.get("prompt_templates")
        if isinstance(st.get("prompt_templates"), dict)
        else _default_prompt_templates()
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
            resolved = _resolve_prompt_template_text(chosen, prompt_templates)
            if resolved:
                sys_prompt_override = resolved
            else:
                print(
                    f"Error: unknown or invalid prompt template {chosen!r}. "
                    "Use /settings prompt_template list (interactive) or define it in ~/.agent.json.",
                    file=sys.stderr,
                )
                return
    si0 = _effective_system_instruction_text(sys_prompt_override)
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
    answered, final_answer = _run_agent_conversation_turn(
        messages,
        user_query,
        today_str,
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
