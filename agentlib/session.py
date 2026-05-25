from __future__ import annotations

import copy
import datetime
import hashlib
import importlib.util
import json
import os
import re
import shlex
import sys
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import AbstractSet, Callable, Iterable, Optional

import requests

from agentlib.clipboard_io import ClipboardError, clipboard_read_text, clipboard_write_text
from agentlib.coercion import coerce_verbose_level
from agentlib.sink import emit_sink_scope, sink_emit, sink_print_compat
from agentlib.tools.registry import ToolRegistry
from agentlib.tools.routing import preferred_web_search_tool
from agentlib.tools.turn_support import resolve_path_under_session
from agentlib import prompts as agent_prompts
from agentlib.tui_parse import parse_send_command
from agentlib.tools import builtins as tool_builtins
from agentlib.context.compaction import (
    approx_message_tokens,
    format_messages_for_summary,
    llm_compress_transcript_for_repl,
)
from agentlib.llm.discovery import fetch_ollama_model_show
from agentlib.llm.calls import DEFAULT_TOOL_CALL_MODE
from agentlib.llm.profile import effective_ollama_model_from_profile, preserved_request_options
from agentlib.llm.request_options import (
    normalize_request_options_pref,
    parse_request_option_scalar_value,
)

from .repl_extensions import MAX_REPL_POST_LOAD_DEPTH, ReplExtensionRegistry
from .runtime import ConversationTurnDeps, run_agent_conversation_turn
from .settings import AgentSettings

_EXTENSION_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# Normalized names after ``--foo-bar`` → ``foo_bar`` (see :func:`_normalize_repl_load_flags`).
_REPL_LOAD_KNOWN_FLAGS = frozenset({"single_lane"})


def _split_repl_load_path_and_flags(parts: list[str]) -> tuple[Optional[str], list[str], list[str]]:
    """Split shlex parts (``['/load', …]``) into path tokens and raw ``--`` option tokens."""
    if len(parts) < 2:
        return ("no file path", [], [])
    rest = parts[1:]
    split_at = len(rest)
    for i, t in enumerate(rest):
        if t.startswith("--"):
            split_at = i
            break
    path_tokens = rest[:split_at]
    raw_flags = rest[split_at:]
    while raw_flags and raw_flags[0] == "--":
        raw_flags = raw_flags[1:]
    if not path_tokens:
        return (
            "no file path before options (put the ``.py`` path before any ``--`` flags, e.g. "
            "``/load ./my_ext.py --some_option``)",
            [],
            raw_flags,
        )
    return (None, path_tokens, raw_flags)


def _normalize_repl_load_flags(raw_flags: list[str]) -> tuple[frozenset, list[str]]:
    """Return ``(known_flags, warning_messages)`` for ``/load`` tail options."""
    normalized: set[str] = set()
    warnings: list[str] = []
    for t in raw_flags:
        if not t.startswith("--"):
            warnings.append(f"expected option starting with '--', ignoring {t!r}")
            continue
        name = t[2:].strip().lower().replace("-", "_")
        if not name:
            warnings.append("empty '--' option ignored")
            continue
        if name not in _REPL_LOAD_KNOWN_FLAGS:
            warnings.append(f"unknown /load option {t!r} (ignored)")
            continue
        normalized.add(name)
    return (frozenset(normalized), warnings)


@dataclass(frozen=True)
class _LoadedReplExtension:
    module_name: str
    path: str
    command_keys: tuple[str, ...]
    help_chunks: tuple[str, ...] = ()
    load_flags: frozenset = field(default_factory=frozenset)

# Sentinel: `_instruction_for_import` failed (usage printed); caller returns an empty command result.
_IMPORT_INSTRUCTION_ERROR = object()


@dataclass
class SessionLineResult:
    output: str = ""
    quit: bool = False
    # When set, hosts (e.g. agent_tui) place this text into the user's input field instead of
    # interpreting it as a new REPL line. Ignored unless the runner supports prefilling prompts.
    prefill_prompt: Optional[str] = None


class AgentSession:
    """Owns interactive session state; parses and executes REPL lines."""

    def __init__(
        self,
        *,
        settings: AgentSettings,
        verbose: int,
        second_opinion_enabled: bool,
        cloud_ai_enabled: bool,
        save_context_path: Optional[str],
        enabled_tools: AbstractSet[str],
        enabled_toolsets: AbstractSet[str],
        primary_profile,
        reviewer_hosted_profile,
        reviewer_ollama_model: Optional[str],
        skills_map: dict,
        prompt_templates: dict,
        prompt_template_default: str,
        prompt_templates_dir: str,
        skills_dir: str,
        tools_dir: str,
        context_cfg: Optional[dict],
        system_prompt_override: Optional[str],
        system_prompt_path: Optional[str],
        session_prompt_template: Optional[str],
        # injected helpers / callbacks (from agent.py)
        agent_progress: Callable[[str], None],
        fetch_ollama_local_model_names: Callable[[], list[str]],
        format_last_ollama_usage_for_repl: Callable[[], str],
        format_session_primary_llm_line: Callable[[object], str],
        format_session_reviewer_line: Callable[[object, Optional[str]], str],
        print_skill_usage_verbose: Callable[..., None],
        match_skill_detail: Callable[[str, dict], tuple[Optional[str], Optional[str]]],
        ml_select_skill_id: Callable[..., tuple[Optional[str], str]],
        skill_plan_steps: Callable[..., tuple[Optional[list], str]],
        effective_enabled_tools_for_skill: Callable[[AbstractSet[str], dict, Optional[str]], AbstractSet[str]],
        effective_enabled_tools_for_turn: Callable[..., AbstractSet[str]],
        route_requires_websearch: Callable[..., Optional[str]],
        deliverable_skip_mandatory_web: Callable[[str], bool],
        user_wants_written_deliverable: Callable[[str], bool],
        prepare_agent_turn_messages: Callable[..., tuple],
        conversation_turn_deps: ConversationTurnDeps,
        save_context_bundle: Callable[..., None],
        load_context_messages: Callable[[str], list],
        registry: ToolRegistry,
        # prefs / persistence
        build_agent_prefs_payload: Callable[..., dict],
        write_agent_prefs_file: Callable[[dict], None],
        agent_prefs_path: Callable[[], str],
        settings_group_keys_lines: Callable[[str], str],
        settings_group_show: Callable[[str], str],
        settings_group_set: Callable[[str, str, str], str],
        settings_group_unset: Callable[[str, str], str],
        settings_get: Callable[[tuple, object], object],
        settings_set: Callable[[tuple, object], None],
        # llm profile helpers for /settings primary/second_opinion llm ...
        LlmProfile_cls,
        default_primary_llm_profile: Callable[[], object],
        describe_llm_profile_short: Callable[[object], str],
        ollama_second_opinion_model: Callable[[], str],
        # thinking helpers
        ollama_request_think_value: Callable[[], object],
        agent_thinking_level: Callable[[], str],
        agent_thinking_enabled_default_false: Callable[[], bool],
        agent_stream_thinking_enabled: Callable[[], bool],
        verbose_ack_message: Callable[[int], str],
        parse_while_repl_tokens: Callable[[list[str]], tuple[int, str, list[str]]],
        call_while_condition_judge: Callable[..., int],
        python_fork_agent: Optional[Callable[..., dict]] = None,
        python_fork_background_agent: Optional[Callable[..., dict]] = None,
        python_delegate_line: Optional[Callable[..., dict]] = None,
        python_host_command: Optional[Callable[[dict], dict]] = None,
        python_enqueue_line: Optional[Callable[[str, str], dict]] = None,
        host_app: Optional[object] = None,
    ):
        self.settings = settings
        self.verbose = int(verbose)
        self.second_opinion_on = bool(second_opinion_enabled)
        self.cloud_ai_enabled = bool(cloud_ai_enabled)
        self.session_save_path = save_context_path
        self.enabled_tools = set(enabled_tools)
        self.enabled_toolsets = set(enabled_toolsets)
        # When False and agent.mcp_enabled, discovered mcp_* tools are merged into enabled_tools automatically.
        self.mcp_tools_opt_out = False
        # When True, /set and /mcp cannot change prefs, session tool allowlists, or MCP wiring for this session.
        self.settings_locked = False
        self.primary_profile = primary_profile
        self.reviewer_hosted_profile = reviewer_hosted_profile
        self.reviewer_ollama_model = reviewer_ollama_model
        self.skills_map = skills_map if isinstance(skills_map, dict) else {}
        self.prompt_templates = prompt_templates if isinstance(prompt_templates, dict) else {}
        self.template_default = (prompt_template_default or "").strip() or "coding"
        self.prompt_templates_dir = prompt_templates_dir
        self.skills_dir = skills_dir
        self.tools_dir = tools_dir
        self.context_cfg = context_cfg if isinstance(context_cfg, dict) else {}
        self.session_system_prompt = system_prompt_override
        self.session_system_prompt_path = system_prompt_path
        self.session_prompt_template = session_prompt_template
        # Persist system_prompt only when the user explicitly overrides/pins/loads it.
        self._system_prompt_explicit = bool(
            (isinstance(self.session_system_prompt_path, str) and self.session_system_prompt_path.strip())
            or (self.session_system_prompt is not None and str(self.session_system_prompt).strip())
        )
        # If the prompt is coming from a selected/default prompt template, treat it as non-explicit.
        if self.session_prompt_template and not (self.session_system_prompt_path or "").strip():
            self._system_prompt_explicit = False

        # Persist prompt templates/default only when explicitly changed in-session.
        self._prompt_templates_explicit = False
        self._prompt_template_default_explicit = False

        self.messages: list = []
        self.last_reuse_skill_id: Optional[str] = None
        # Last normal (non-slash) user line and last structured model answer for /last_* .
        self.repl_last_user_query: Optional[str] = None
        self.repl_last_assistant_answer: Optional[str] = None

        # injected helpers / callbacks
        self._agent_progress = agent_progress
        self._fetch_ollama_local_model_names = fetch_ollama_local_model_names
        self._format_last_ollama_usage_for_repl = format_last_ollama_usage_for_repl
        self._format_session_primary_llm_line = format_session_primary_llm_line
        self._format_session_reviewer_line = format_session_reviewer_line
        self._print_skill_usage_verbose = print_skill_usage_verbose
        self._match_skill_detail = match_skill_detail
        self._ml_select_skill_id = ml_select_skill_id
        self._skill_plan_steps = skill_plan_steps
        self._effective_enabled_tools_for_skill = effective_enabled_tools_for_skill
        self._effective_enabled_tools_for_turn = effective_enabled_tools_for_turn
        self._route_requires_websearch = route_requires_websearch
        self._deliverable_skip_mandatory_web = deliverable_skip_mandatory_web
        self._user_wants_written_deliverable = user_wants_written_deliverable
        self._prepare_agent_turn_messages = prepare_agent_turn_messages
        # Shallow copy only: ``call_mcp_tool`` (and other callables) close over ``AgentApp``; after MCP sync the
        # app may hold live ``StdioMcpSession`` objects with ``threading.Lock``, which ``deepcopy`` cannot handle.
        # Per-session isolation is handled via ``replace(session_cwd=…)`` and ``_rebind_session_fs_tools``.
        self._conversation_turn_deps = copy.copy(conversation_turn_deps)
        self.session_cwd = os.path.abspath(os.getcwd())
        self._conversation_turn_deps = replace(self._conversation_turn_deps, session_cwd=self.session_cwd)
        self._rebind_session_fs_tools()
        self._save_context_bundle = save_context_bundle
        self._load_context_messages = load_context_messages
        self._registry = registry

        self._build_agent_prefs_payload = build_agent_prefs_payload
        self._write_agent_prefs_file = write_agent_prefs_file
        self._agent_prefs_path = agent_prefs_path
        self._settings_group_keys_lines = settings_group_keys_lines
        self._settings_group_show = settings_group_show
        self._settings_group_set = settings_group_set
        self._settings_group_unset = settings_group_unset
        self._settings_get = settings_get
        self._settings_set = settings_set

        self._LlmProfile = LlmProfile_cls
        self._default_primary_llm_profile = default_primary_llm_profile
        self._describe_llm_profile_short = describe_llm_profile_short
        self._ollama_second_opinion_model = ollama_second_opinion_model

        self._ollama_request_think_value = ollama_request_think_value
        self._agent_thinking_level = agent_thinking_level
        self._agent_thinking_enabled_default_false = agent_thinking_enabled_default_false
        self._agent_stream_thinking_enabled = agent_stream_thinking_enabled
        self._verbose_ack_message = verbose_ack_message
        self._parse_while_repl_tokens = parse_while_repl_tokens
        self._call_while_condition_judge = call_while_condition_judge
        # Multi-line /call_python support (buffer until closing quote).
        self._call_python_pending: Optional[str] = None
        # REPL extensions from ``/load`` (see ``agentlib.repl_extensions``).
        self._repl_extension_commands: dict[str, Callable[..., SessionLineResult]] = {}
        self._repl_extensions_loaded: list[_LoadedReplExtension] = []
        self._repl_post_load_depth = 0
        # ``/load … --single_lane`` for ``extensions/code.py``: run the /code pipeline on this session only.
        self.repl_code_extension_single_lane = False

        self.python_fork_agent = python_fork_agent
        self.python_fork_background_agent = python_fork_background_agent
        self.python_delegate_line = python_delegate_line
        self.python_host_command = python_host_command
        self.python_enqueue_line = python_enqueue_line
        self._host_app = host_app

    def _chars_per_token_estimator(self):
        app = getattr(self, "_host_app", None)
        if app is not None:
            return app._chars_per_token_estimator
        from agentlib.llm.token_estimate import get_default_chars_per_token_estimator

        return get_default_chars_per_token_estimator()

    def _resolve_session_path(self, raw: object) -> str:
        """Resolve a user/tool path against this session's cwd when relative."""
        return resolve_path_under_session(raw, self.session_cwd, self._conversation_turn_deps.scalar_to_str)

    def _session_run_command(self, cmd: object) -> str:
        """Shell runner used by REPL ``/run_command`` / ``!`` (session cwd-aware)."""
        return tool_builtins.run_command(cmd, cwd=self.session_cwd)

    def _session_write_file(self, path: object, content: object) -> str:
        return tool_builtins.write_file(self._resolve_session_path(path), content)

    def _session_read_file(self, path: object) -> str:
        return tool_builtins.read_file(self._resolve_session_path(path))

    def _session_grep(
        self,
        pattern: object,
        path: object = ".",
        glob_pattern: object = None,
        max_matches: object = 200,
        max_files: object = 8000,
        ignore_case: object = False,
    ) -> str:
        raw_path = self._conversation_turn_deps.scalar_to_str(path, ".") or "."
        resolved = self._resolve_session_path(raw_path)
        return tool_builtins.grep(
            pattern,
            resolved,
            glob_pattern,
            max_matches,
            max_files,
            ignore_case,
        )

    def _session_list_directory(self, path: object) -> str:
        return tool_builtins.list_directory(self._resolve_session_path(path))

    def _session_download_file(self, url: object, path: object) -> str:
        return tool_builtins.download_file(url, self._resolve_session_path(path))

    def _session_tail_file(self, path: object, lines: object = 20) -> str:
        return tool_builtins.tail_file(self._resolve_session_path(path), lines)

    def _session_replace_text(self, path: object, pattern: object, replacement: object, replace_all: object = True) -> str:
        return tool_builtins.replace_text(
            self._resolve_session_path(path),
            pattern,
            replacement,
            replace_all,
        )

    def _rebind_session_fs_tools(self) -> None:
        """Rebuild ConversationTurnDeps tool bindings so filesystem tools honor session cwd."""
        self._conversation_turn_deps = replace(
            self._conversation_turn_deps,
            session_cwd=self.session_cwd,
            write_file=self._session_write_file,
            read_file=self._session_read_file,
            grep=self._session_grep,
            list_directory=self._session_list_directory,
            download_file=self._session_download_file,
            tail_file=self._session_tail_file,
            replace_text=self._session_replace_text,
        )

    def _cmd_cd(self, s: str) -> SessionLineResult:
        """Change this session's working directory for ``run_command`` / ``!`` / tool runs."""
        try:
            toks = shlex.split(s)
        except ValueError as e:
            sink_print_compat(f"/cd: {e}")
            return SessionLineResult()
        if len(toks) < 2:
            sink_print_compat("Usage: /cd <dir>")
            return SessionLineResult()
        raw = shlex.join(toks[1:]).strip()
        if not raw:
            sink_print_compat("Usage: /cd <dir>")
            return SessionLineResult()
        # Relative segments are resolved against this session's cwd, not the process cwd.
        base = (self.session_cwd or os.path.abspath(os.getcwd())).strip()
        target = resolve_path_under_session(raw, base, self._conversation_turn_deps.scalar_to_str)
        if not os.path.isdir(target):
            sink_print_compat(f"/cd: not a directory: {target!r}")
            return SessionLineResult()
        self.session_cwd = target
        self._rebind_session_fs_tools()
        sink_print_compat(f"Working directory: {self.session_cwd}")
        return SessionLineResult()

    def _mcp_servers_normalized(self) -> list:
        raw = self._settings_get(("agent", "mcp_servers"))
        return raw if isinstance(raw, list) else []

    @staticmethod
    def _mcp_entry_name_key(rec: object) -> str:
        if not isinstance(rec, dict):
            return ""
        return str(rec.get("name") or "").strip().lower().replace("-", "_")

    def _mcp_session_enabled_tool_ids(self) -> set[str]:
        from agentlib.tools import mcp_registry

        return {t for t in self.enabled_tools if mcp_registry.is_mcp_tool(t)}

    def _match_skill_for_turn(self, user_query: str) -> tuple[Optional[str], Optional[str]]:
        """Trigger-based skill match for normal turns; off unless ``agent.skill_auto_match_triggers``."""
        if not self.settings.get_bool(("agent", "skill_auto_match_triggers"), False):
            return None, None
        return self._match_skill_detail(user_query, self.skills_map)

    _SETTINGS_LOCK_MSG = (
        "Settings and MCP configuration are locked for this session "
        "(lock is permanent; start a new session to change prefs)."
    )

    def _reject_if_settings_locked(self) -> Optional[SessionLineResult]:
        if self.settings_locked:
            sink_print_compat(self._SETTINGS_LOCK_MSG)
            return SessionLineResult()
        return None

    @staticmethod
    def _mcp_subcommand_read_only(sub: str) -> bool:
        return sub in ("help", "-h", "--help", "list", "status")

    @classmethod
    def _set_subcommand_read_only(cls, key: str, toks: list[str]) -> bool:
        key = key.lower().replace("-", "_")
        n = len(toks)

        def sub(i: int, default: str = "") -> str:
            return toks[i].lower() if n > i else default

        if key in ("help",):
            return True
        if key in ("ollama", "openai", "agent"):
            if n < 3:
                return True
            return sub(2) in ("show", "list", "keys", "key", "help")
        if key == "tools":
            if n < 3:
                return True
            s2 = sub(2)
            if s2 in ("list", "ls", "show"):
                return True
            if s2 in ("describe", "desc", "help") and n >= 4:
                return True
            return False
        if key == "system_prompt":
            if n < 3:
                return True
            return sub(2) == "show"
        if key in ("prompt_template", "prompt_templates", "prompt"):
            if n < 3:
                return True
            s2 = sub(2)
            if s2 in ("help", "-h", "--help", "explain"):
                return True
            return s2 in ("list", "show")
        if key in ("context", "context_manager", "context_window"):
            if n < 3:
                return True
            return sub(2) == "show"
        if key == "thinking":
            if n < 3:
                return True
            return sub(2) == "show"
        if key == "primary":
            if n < 4:
                return True
            if sub(2) == "request_options":
                if n < 4:
                    return True
                return sub(3) in ("show", "help", "-h", "--help")
            return False
        if key == "extensions":
            if n < 3:
                return True
            a = sub(2)
            if a in ("help", "-h", "--help", "show", "list"):
                return True
            if n >= 4 and sub(3) in ("show", "list"):
                return True
            return False
        return False

    def seed_mcp_tools_if_connected(self) -> int:
        """Merge discovered MCP tool ids into ``enabled_tools`` when global MCP is on and session not opted out."""
        from agentlib.tools import mcp_registry

        if self.settings_locked:
            return 0
        if self.mcp_tools_opt_out:
            return 0
        if not self.settings.get_bool(("agent", "mcp_enabled"), False):
            return 0
        ids = mcp_registry.all_ids()
        if not ids:
            return 0
        before = len(self._mcp_session_enabled_tool_ids())
        self.enabled_tools.update(ids)
        return len(self._mcp_session_enabled_tool_ids()) - before

    def mcp_session_enable_tools(self) -> int:
        """Re-enable MCP tools for this session (all currently discovered ids)."""
        self.mcp_tools_opt_out = False
        return self.seed_mcp_tools_if_connected()

    def mcp_session_disable_tools(self) -> int:
        """Opt out of MCP tools for this session and remove mcp_* from ``enabled_tools``."""
        from agentlib.tools import mcp_registry

        self.mcp_tools_opt_out = True
        removed = {t for t in self.enabled_tools if mcp_registry.is_mcp_tool(t)}
        self.enabled_tools -= removed
        return len(removed)

    def _mcp_refresh_connections(
        self, *, connected_msg: str = "[mcp] Connections refreshed.", announce: bool = True
    ) -> None:
        host = getattr(self, "_host_app", None)
        if host is None:
            sink_print_compat("[mcp] Host app not linked — restart the REPL to reconnect MCP.")
            return
        try:
            if self.settings.get_bool(("agent", "mcp_enabled"), False):
                self.seed_mcp_tools_if_connected()
                host.schedule_mcp_resync()
                if announce:
                    sink_print_compat(
                        "[mcp] Reconnect scheduled in the background (per-server handshake can take a few seconds). "
                        "Run `/mcp status` when you want to see tools and errors."
                    )
                return
            connected = host.sync_mcp()
            if connected:
                sink_print_compat(connected_msg)
            else:
                sink_print_compat(
                    "[mcp] MCP is disabled (agent.mcp_enabled=false); servers were not started. "
                    "Run /mcp enable to connect and discover tools."
                )
        except Exception as e:
            sink_print_compat(f"[mcp] refresh failed: {type(e).__name__}: {e}")

    def _cmd_mcp(self, s: str) -> SessionLineResult:
        """`/mcp` — configure Model Context Protocol servers (prefs-backed)."""
        from agentlib.mcp.repl_cmd import MCP_REPL_HELP, parse_mcp_add_http_tokens, parse_mcp_add_stdio_tokens
        from agentlib.tools import mcp_registry

        try:
            toks = shlex.split((s or "").strip())
        except ValueError as e:
            sink_print_compat(f"/mcp: {e}")
            return SessionLineResult()
        if len(toks) < 2:
            sink_print_compat(MCP_REPL_HELP)
            return SessionLineResult()
        sub = toks[1].lower()
        if sub in ("help", "-h", "--help"):
            sink_print_compat(MCP_REPL_HELP)
            return SessionLineResult()

        if not self._mcp_subcommand_read_only(sub):
            blocked = self._reject_if_settings_locked()
            if blocked is not None:
                return blocked

        if sub == "list":
            enabled = self.settings.get_bool(("agent", "mcp_enabled"), False)
            servers = self._mcp_servers_normalized()
            sink_print_compat(
                f"agent.mcp_enabled = {enabled}\nagent.mcp_servers =\n"
                + json.dumps(servers, indent=2, ensure_ascii=False)
            )
            return SessionLineResult()

        if sub == "status":
            enabled = self.settings.get_bool(("agent", "mcp_enabled"), False)
            servers = self._mcp_servers_normalized()
            sink_print_compat(
                f"agent.mcp_enabled = {enabled}\nagent.mcp_servers =\n"
                + json.dumps(servers, indent=2, ensure_ascii=False)
            )
            errs = mcp_registry.last_connect_errors()
            if errs:
                sink_print_compat("Last sync errors:\n" + "\n".join(f"  - {x}" for x in errs))
            else:
                sink_print_compat("Last sync errors: (none recorded)")
            n_tools = len(mcp_registry.all_ids())
            sess_on = len(self._mcp_session_enabled_tool_ids())
            sink_print_compat(f"Discovered MCP tools (this process): {n_tools}")
            sink_print_compat(
                f"MCP tools enabled in this session: {sess_on}"
                + (f" of {n_tools}" if n_tools else " (none discovered yet)")
            )
            if not enabled and servers:
                sink_print_compat(
                    "(With MCP disabled, tool discovery is skipped — run /mcp enable, then /mcp status again.)"
                )
            elif enabled and n_tools and not self.mcp_tools_opt_out and not self.settings_locked:
                added = self.seed_mcp_tools_if_connected()
                if added:
                    sess_on = len(self._mcp_session_enabled_tool_ids())
                    sink_print_compat(
                        f"(Merged {added} newly discovered MCP tool(s) into this session; "
                        f"now {sess_on} of {n_tools} enabled.)"
                    )
            elif enabled and n_tools and self.mcp_tools_opt_out and sess_on == 0:
                sink_print_compat("(This session opted out of MCP tools — use /mcp session on to re-enable.)")
            return SessionLineResult()

        if sub in ("session", "tools"):
            if len(toks) < 3:
                sink_print_compat("Usage: /mcp session on | off")
                return SessionLineResult()
            act = toks[2].lower()
            if act in ("on", "enable", "true", "yes"):
                added = self.mcp_session_enable_tools()
                n = len(mcp_registry.all_ids())
                sink_print_compat(
                    f"MCP tools enabled for this session (+{added} id(s); {len(self._mcp_session_enabled_tool_ids())} of {n} discovered)."
                )
                return SessionLineResult()
            if act in ("off", "disable", "false", "no"):
                removed = self.mcp_session_disable_tools()
                sink_print_compat(f"MCP tools disabled for this session (removed {removed} id(s) from enabled_tools).")
                return SessionLineResult()
            sink_print_compat("Usage: /mcp session on | off")
            return SessionLineResult()

        if sub == "reload":
            self._mcp_refresh_connections()
            return SessionLineResult()

        if sub == "enable":
            self._settings_set(("agent", "mcp_enabled"), True)
            sink_print_compat(
                "agent.mcp_enabled = true (use /set save to persist). "
                "Shared MCP servers will connect; this session enables discovered MCP tools by default "
                "(use /mcp session off to opt out)."
            )
            added = self.seed_mcp_tools_if_connected()
            if added:
                sink_print_compat(f"Enabled {added} MCP tool(s) in this session (already discovered).")
            self._mcp_refresh_connections()
            return SessionLineResult()

        if sub == "disable":
            self._settings_set(("agent", "mcp_enabled"), False)
            sink_print_compat(
                "agent.mcp_enabled = false (use /set save to persist). "
                "Shared MCP connections cleared; per-session enabled_tools entries for mcp_* are unchanged."
            )
            self._mcp_refresh_connections()
            return SessionLineResult()

        if sub == "remove":
            if len(toks) < 3:
                sink_print_compat("Usage: /mcp remove NAME")
                return SessionLineResult()
            target = self._mcp_entry_name_key({"name": toks[2]})
            if not target:
                sink_print_compat("/mcp remove: invalid NAME")
                return SessionLineResult()
            servers = self._mcp_servers_normalized()
            new = [x for x in servers if self._mcp_entry_name_key(x) != target]
            if len(new) == len(servers):
                sink_print_compat(f"/mcp remove: no server named {toks[2]!r}")
                return SessionLineResult()
            self._settings_set(("agent", "mcp_servers"), new)
            sink_print_compat(f"Removed MCP server {toks[2]!r}. Use /set save to persist.")
            self._mcp_refresh_connections()
            return SessionLineResult()

        if sub == "add":
            if len(toks) < 4:
                sink_print_compat("Usage: /mcp add stdio … | /mcp add http …\nTry /mcp help")
                return SessionLineResult()
            kind = toks[2].lower()
            if kind == "stdio":
                spec, err = parse_mcp_add_stdio_tokens(toks)
            elif kind == "http":
                spec, err = parse_mcp_add_http_tokens(toks)
            else:
                sink_print_compat("/mcp add: transport must be stdio or http (see /mcp help)")
                return SessionLineResult()
            if spec is None:
                sink_print_compat(err)
                return SessionLineResult()
            name_key = spec["name"]
            servers = self._mcp_servers_normalized()
            new = [x for x in servers if self._mcp_entry_name_key(x) != name_key]
            new.append(spec)
            self._settings_set(("agent", "mcp_servers"), new)
            sink_print_compat(
                f"MCP server {spec['name']!r} configured ({spec.get('transport')} upsert). "
                "Use /set save to persist."
            )
            self._mcp_refresh_connections()
            return SessionLineResult()

        sink_print_compat(f"/mcp: unknown subcommand {toks[1]!r}. Try /mcp help")
        return SessionLineResult()

    def _cmd_compact(self, s: str) -> SessionLineResult:
        """``/compact`` — LLM-compress transcript, then replace ``messages`` (same scratch reset as ``/clear``)."""
        try:
            parts = shlex.split(s.strip())
        except ValueError as e:
            sink_print_compat(f"/compact: {e}")
            return SessionLineResult()
        if len(parts) > 2:
            sink_print_compat("Usage: /compact [N% | WORDS]   (default: 10% of current size; try /compact help)")
            return SessionLineResult()
        arg = parts[1] if len(parts) > 1 else ""
        if arg.lower() in ("help", "-h", "--help"):
            sink_print_compat(
                "/compact — ask the primary LLM to compress chat history, then replace messages "
                "(same reset as /clear for skill reuse and /last snapshots).\n"
                "  /compact        target ~10% of current estimated token size\n"
                "  /compact 25%    target ~25% of current estimated token size\n"
                "  /compact 400    compressed summary at most 400 words\n"
            )
            return SessionLineResult()

        text = format_messages_for_summary(self.messages)
        if not text.strip():
            sink_print_compat("/compact: nothing to compress (no user/assistant text in history).")
            return SessionLineResult()

        est = self._chars_per_token_estimator()
        approx0 = approx_message_tokens(self.messages, estimator=est)
        if approx0 < 1:
            approx0 = 1

        kind: str
        val: int
        if not arg:
            target_toks = max(1, min(approx0, int(approx0 * 0.10)))
            kind, val = "approx_tokens", target_toks
        elif arg.endswith("%"):
            try:
                pct = float(arg[:-1].strip())
            except ValueError:
                sink_print_compat("/compact: invalid percent; use e.g. 25%")
                return SessionLineResult()
            if not (0.0 < pct <= 100.0):
                sink_print_compat("/compact: percent must be greater than 0 and at most 100.")
                return SessionLineResult()
            target_toks = max(1, min(approx0, int(approx0 * pct / 100.0)))
            kind, val = "approx_tokens", target_toks
        elif arg.isdigit():
            w = int(arg, 10)
            if w < 1:
                sink_print_compat("/compact: word count must be at least 1.")
                return SessionLineResult()
            kind, val = "max_words", w
        else:
            sink_print_compat("Usage: /compact [N% | WORDS]   (try /compact help)")
            return SessionLineResult()

        default_om = self.settings.get_str(("ollama", "model"), "qwen3.6:latest")
        om = effective_ollama_model_from_profile(self.primary_profile, default_om)

        self._agent_progress("Compressing conversation with the primary LLM…")
        try:
            compressed = llm_compress_transcript_for_repl(
                profile=self.primary_profile,
                transcript=text,
                constraint_kind=kind,
                constraint_value=val,
                call_hosted_chat_plain=self._conversation_turn_deps.call_hosted_chat_plain,
                call_ollama_plaintext=self._conversation_turn_deps.call_ollama_plaintext,
                ollama_model=om,
                chars_per_token_estimator=est,
            )
        except Exception as e:
            sink_print_compat(f"/compact failed: {type(e).__name__}: {e}")
            return SessionLineResult()

        if not (compressed or "").strip():
            sink_print_compat("/compact: model returned empty text; history unchanged.")
            return SessionLineResult()

        self.messages.clear()
        self.last_reuse_skill_id = None
        self.repl_last_user_query = None
        self.repl_last_assistant_answer = None

        body = (
            "The following is a compressed transcript of the prior session "
            "(produced by /compact). Continue from this context.\n\n"
            f"{compressed.strip()}"
        )
        self.messages.append({"role": "system", "content": body})

        approx1 = approx_message_tokens(self.messages, estimator=est)
        if kind == "max_words":
            detail = f"max {val} words"
        else:
            detail = f"~{val} target tokens"
        sink_print_compat(
            f"/compact: replaced history (~{approx0} → ~{approx1} est. tokens; {detail})."
        )
        return SessionLineResult()

    def _resolve_call_python_argv_paths(self, argv: list[str]) -> list[str]:
        """Resolve relative paths after ``-f`` / ``--file`` / ``<`` against ``session_cwd``."""
        base = (self.session_cwd or "").strip()
        if not base:
            return argv
        root = os.path.abspath(os.path.expanduser(base))

        def fix(raw: str) -> str:
            if not isinstance(raw, str) or not raw.strip():
                return raw
            exp = os.path.expanduser(raw.strip())
            if os.path.isabs(exp):
                return os.path.normpath(exp)
            return os.path.normpath(os.path.join(root, exp))

        out = list(argv)
        i = 1
        while i < len(out):
            if i + 1 < len(out) and out[i] in ("-f", "--file"):
                out[i + 1] = fix(out[i + 1])
                i += 2
                continue
            if out[i] == "<" and i + 1 < len(out):
                out[i + 1] = fix(out[i + 1])
                i += 2
                continue
            i += 1
        return out

    def _agent_loop_budget(self) -> tuple[int, int, int, int]:
        s = self.settings
        return (
            max(1, s.get_int(("agent", "max_agent_steps"), 30)),
            max(1, s.get_int(("agent", "max_agent_steps_web"), 15)),
            max(1, s.get_int(("agent", "max_tool_calls_web"), 15)),
            max(1, s.get_int(("agent", "max_fetch_page_web"), 15)),
        )

    def host_ctl(self, op: str, arg: Optional[str] = None) -> dict:
        """
        Multi-agent host RPC (optional). Used by ``/list``, ``/switch``, ``/last answer|question``,
        legacy ``/last_*``, and ``/call_python`` helpers when a host (e.g. ``agent_tui``) wires
        ``python_host_command``.
        """
        h = self.python_host_command
        want_local_last = (
            h is None
            and op in ("last_answer", "last_question")
            and (arg is None or not str(arg).strip())
        )
        if want_local_last:
            if op == "last_answer":
                v = self.repl_last_assistant_answer
                hint = "(no last assistant answer yet)"
            else:
                v = self.repl_last_user_query
                hint = "(no last user question yet)"
            text = v.strip() if isinstance(v, str) and v.strip() else hint
            return {"ok": True, "text": text}
        if h is None:
            return {
                "ok": False,
                "error": "Multi-agent host not available (e.g. run agent_tui.py for /list and /switch).",
            }
        try:
            return h({"op": op, "arg": arg, "session": self})
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _sink_host_ctl_result(self, r: dict) -> SessionLineResult:
        if r.get("ok"):
            sink_print_compat(r.get("text") or "")
        else:
            sink_print_compat(r.get("error") or "failed")
        return SessionLineResult()

    def _cmd_send_to_agent(self, s: str) -> SessionLineResult:
        """Forward one or more lines to another agent (async enqueue when ``python_enqueue_line`` is set)."""
        parsed = parse_send_command(s)
        if parsed is None:
            sink_print_compat('Usage: /send AGENT COMMAND…  or  /send AGENT "cmd1,cmd2,…"')
            return SessionLineResult()
        agent_name, cmds = parsed
        eq = self.python_enqueue_line
        if eq is not None:
            for cmd in cmds:
                try:
                    r = eq(agent_name, cmd)
                except BaseException as e:
                    sink_print_compat(f"/send: {type(e).__name__}: {e}")
                    return SessionLineResult()
                if isinstance(r, dict) and r.get("ok"):
                    lab = str(r.get("label") or agent_name)
                    if r.get("queued"):
                        sink_print_compat(f"[queued → {lab}] {cmd}")
                    else:
                        sink_print_compat(f"[started → {lab}] {cmd}")
                elif isinstance(r, dict):
                    sink_print_compat(str(r.get("error") or "/send failed"))
                    return SessionLineResult()
                else:
                    sink_print_compat("[send] scheduled.")
            return SessionLineResult()
        dl = self.python_delegate_line
        if dl is None:
            sink_print_compat(
                "/send requires a multi-agent host (e.g. agent_tui.py); enqueue/delegate hook not configured."
            )
            return SessionLineResult()
        for cmd in cmds:
            try:
                res = dl(agent_name, cmd)
            except BaseException as e:
                sink_print_compat(f"/send: {type(e).__name__}: {e}")
                return SessionLineResult()
            if isinstance(res, dict):
                if res.get("type") == "command":
                    out = res.get("output") or ""
                    if isinstance(out, str) and out.strip():
                        sink_print_compat(out.strip())
                    else:
                        sink_print_compat(f"[sent → {agent_name}] command finished.")
                elif res.get("type") == "turn":
                    if res.get("answered"):
                        ans = res.get("answer")
                        n = len(ans) if isinstance(ans, str) else 0
                        sink_print_compat(f"[sent → {agent_name}] turn answered ({n} chars).")
                    else:
                        sink_print_compat(f"[sent → {agent_name}] turn finished.")
                else:
                    sink_print_compat(f"[sent → {agent_name}] done.")
            else:
                sink_print_compat(f"[sent → {agent_name}] done.")
        return SessionLineResult()

    def execute_line(self, line: str, *, emit: Optional[Callable[[dict], None]] = None) -> dict:
        """
        Execute one REPL line.

        - When `emit` is provided, an emit sink is installed for this line so thinking, progress,
          tool output, and REPL command text stream incrementally via typed emit events (same schema as before).
        - Returns a structured dict for embedding callers (CLI can ignore most fields).
        """
        raw_line = (line or "").rstrip("\n")
        s0 = raw_line.strip()
        if not s0:
            return {"type": "noop", "quit": False}

        # Multi-line /call_python -c: allow pasting code with literal newlines by buffering
        # until shlex sees a closing quote.
        if self._call_python_pending is not None:
            combined = self._call_python_pending + "\n" + raw_line
            kind, payload = self._split_call_python_rest(combined.strip())
            if kind == "error" and isinstance(payload, str) and "No closing quotation" in payload:
                self._call_python_pending = combined
                return {"type": "command", "quit": False, "output": ""}
            self._call_python_pending = None
            s0 = combined.strip()
        else:
            # Start buffering if the user opened a quote but didn't close it yet.
            if s0.lower().startswith("/call_python"):
                kind, payload = self._split_call_python_rest(s0)
                if kind == "error" and isinstance(payload, str) and "No closing quotation" in payload:
                    self._call_python_pending = raw_line
                    return {"type": "command", "quit": False, "output": ""}

        imp = self._instruction_for_import(s0)
        if imp is _IMPORT_INSTRUCTION_ERROR:
            return {"type": "command", "quit": False, "output": ""}
        if isinstance(imp, str):
            if emit is None:
                answered, final_answer = self._execute_user_request(imp)
                self.repl_last_user_query = imp
                self.repl_last_assistant_answer = (
                    final_answer.strip()
                    if isinstance(final_answer, str) and final_answer.strip()
                    else None
                )
                from agentlib.llm import streaming as llm_streaming

                return {
                    "type": "turn",
                    "quit": False,
                    "answered": bool(answered),
                    "answer": final_answer,
                    "answer_streamed": llm_streaming.assistant_answer_was_streamed(),
                }
            with emit_sink_scope(emit):
                answered, final_answer = self._execute_user_request(imp)
                self.repl_last_user_query = imp
                self.repl_last_assistant_answer = (
                    final_answer.strip()
                    if isinstance(final_answer, str) and final_answer.strip()
                    else None
                )
                from agentlib.llm import streaming as llm_streaming

                return {
                    "type": "turn",
                    "quit": False,
                    "answered": bool(answered),
                    "answer": final_answer,
                    "answer_streamed": llm_streaming.assistant_answer_was_streamed(),
                }

        if emit is None:
            # Preserve legacy behavior (prints inside handlers).
            if s0.startswith("/"):
                res = self._execute_command_line(s0)
                d: dict = {"type": "command", "quit": bool(res.quit), "output": res.output}
                if res.prefill_prompt is not None:
                    d["prefill_prompt"] = res.prefill_prompt
                return d
            if s0.startswith("!"):
                res = self._cmd_run_shell_bang(s0)
                return {"type": "command", "quit": bool(res.quit), "output": res.output}
            answered, final_answer = self._execute_user_request(s0)
            self.repl_last_user_query = s0
            self.repl_last_assistant_answer = (
                final_answer.strip()
                if isinstance(final_answer, str) and final_answer.strip()
                else None
            )
            from agentlib.llm import streaming as llm_streaming

            return {
                "type": "turn",
                "quit": False,
                "answered": bool(answered),
                "answer": final_answer,
                "answer_streamed": llm_streaming.assistant_answer_was_streamed(),
            }

        with emit_sink_scope(emit):
            if s0.startswith("/"):
                res = self._execute_command_line(s0)
                payload = {"type": "command", "quit": bool(res.quit), "output": res.output}
                if res.prefill_prompt is not None:
                    payload["prefill_prompt"] = res.prefill_prompt
            elif s0.startswith("!"):
                res = self._cmd_run_shell_bang(s0)
                payload = {"type": "command", "quit": bool(res.quit), "output": res.output}
            else:
                answered, final_answer = self._execute_user_request(s0)
                self.repl_last_user_query = s0
                self.repl_last_assistant_answer = (
                    final_answer.strip()
                    if isinstance(final_answer, str) and final_answer.strip()
                    else None
                )
                from agentlib.llm import streaming as llm_streaming

                payload = {
                    "type": "turn",
                    "quit": False,
                    "answered": bool(answered),
                    "answer": final_answer,
                    "answer_streamed": llm_streaming.assistant_answer_was_streamed(),
                }
            return payload

    def _today_str(self) -> str:
        return datetime.date.today().strftime("%Y-%m-%d (%A)")

    def _instruction_for_import(self, s: str) -> object:
        """If line is `/import FILE`, return the synthetic user message; else ``None``. On parse/validation errors, return ``_IMPORT_INSTRUCTION_ERROR`` after printing."""
        st = (s or "").strip()
        # Only shell-split strings that are actually `/import` commands. Running
        # ``shlex.split`` on arbitrary long prompts (e.g. from ``ai(...)``) breaks on
        # unbalanced quotes and was mis-reported as ``/import: No closing quotation``.
        parts = st.split(None, 1)
        first = parts[0].lower() if parts else ""
        if first != "/import":
            return None
        try:
            toks = shlex.split(st)
        except ValueError as e:
            sink_print_compat(f"/import: {e}")
            return _IMPORT_INSTRUCTION_ERROR
        if not toks or toks[0].lower() != "/import":
            return None
        if len(toks) < 2:
            sink_print_compat("Usage: /import <file>")
            return _IMPORT_INSTRUCTION_ERROR
        raw = os.path.expanduser(shlex.join(toks[1:]).strip())
        if not raw:
            sink_print_compat("Usage: /import <file>")
            return _IMPORT_INSTRUCTION_ERROR
        path = self._resolve_session_path(raw)
        if not os.path.isfile(path):
            sink_print_compat(f"/import: not a file: {path!r}")
            return _IMPORT_INSTRUCTION_ERROR
        display = os.path.normpath(os.path.abspath(path))
        return (
            f"The file {display} contains some knowledge that you have learned in the past "
            "on how to do things.  You should read it and import it into this conversation we are having."
        )

    def _execute_user_request(self, user_query: str) -> tuple[bool, Optional[str]]:
        """One normal REPL turn: append messages and run the agent loop."""
        from agentlib.llm import streaming as llm_streaming

        llm_streaming.reset_assistant_answer_streamed()
        today_str = self._today_str()
        deliverable_wanted = self._user_wants_written_deliverable(user_query)
        sid0, tr0 = self._match_skill_for_turn(user_query)
        et_turn0 = self._effective_enabled_tools_for_skill(
            frozenset(self.enabled_tools), self.skills_map, sid0
        )
        et_turn = self._effective_enabled_tools_for_turn(
            base_enabled_tools=et_turn0,
            enabled_toolsets=self.enabled_toolsets,
            user_query=user_query,
        )
        if self.verbose >= 1:
            d0 = (
                f"trigger match: longest substring {tr0!r} (skill {sid0!r})"
                if sid0 and tr0
                else "trigger match: no skill (no trigger substring matched)"
            )
            self._print_skill_usage_verbose(
                self.verbose,
                source="repl",
                skill_id=sid0,
                base_tools=self.enabled_tools,
                effective_tools=et_turn,
                detail=d0,
            )
        sprompt0 = (self.skills_map.get(sid0) or {}).get("prompt") if sid0 else None
        router_query = self._route_requires_websearch(
            user_query,
            today_str,
            self.primary_profile,
            et_turn,
            transcript_messages=self.messages,
        )
        if self._deliverable_skip_mandatory_web(user_query):
            router_query = None
        web_required = bool(router_query)
        agent_system_message, turn_user = self._prepare_agent_turn_messages(
            user_query,
            today_str,
            self.second_opinion_on,
            self.cloud_ai_enabled,
            primary_profile=self.primary_profile,
            reviewer_ollama_model=self.reviewer_ollama_model,
            reviewer_hosted_profile=self.reviewer_hosted_profile,
            enabled_tools=et_turn,
            system_instruction_override=self.session_system_prompt,
            skill_suffix=sprompt0,
        )
        self.messages.append({"role": "user", "content": turn_user})
        _mw = preferred_web_search_tool(et_turn)
        if router_query and _mw:
            from agentlib.llm.tool_schemas import web_search_required_user_content

            self.messages.append(
                {
                    "role": "user",
                    "content": web_search_required_user_content(
                        _mw,
                        router_query,
                        tool_call_mode=self.settings.get_str(
                            ("agent", "tool_call_mode"), DEFAULT_TOOL_CALL_MODE
                        ),
                        primary_profile=self.primary_profile,
                    ),
                }
            )
        ms, msw, mtcw, mfpw = self._agent_loop_budget()
        answered, final_answer = run_agent_conversation_turn(
            self.messages,
            user_query,
            today_str,
            self._conversation_turn_deps,
            web_required=web_required,
            deliverable_wanted=deliverable_wanted,
            verbose=self.verbose,
            second_opinion_enabled=self.second_opinion_on,
            cloud_ai_enabled=self.cloud_ai_enabled,
            primary_profile=self.primary_profile,
            reviewer_hosted_profile=self.reviewer_hosted_profile,
            reviewer_ollama_model=self.reviewer_ollama_model,
            enabled_tools=et_turn,
            interactive_tool_recovery=True,
            context_cfg=self.context_cfg,
            print_answer=False,
            max_agent_steps=ms,
            max_agent_steps_web=msw,
            max_tool_calls_web=mtcw,
            max_fetch_page_web=mfpw,
            agent_system_message=agent_system_message,
        )
        if self.session_save_path:
            try:
                self._save_context_bundle(
                    self.session_save_path,
                    self.messages,
                    user_query,
                    final_answer,
                    answered,
                )
            except OSError as e:
                sink_emit({"type": "warning", "text": f"Warning: could not save context: {e}"})
        return bool(answered), final_answer

    def _run_with_selected_skill(
        self, req: str, sid: str, *, source: str, selection_rationale: str
    ) -> None:
        self.last_reuse_skill_id = sid
        src = (source or "").strip().lower()
        if src == "reuse":
            self._agent_progress("/skill reuse: using stored skill; starting…")
        elif src == "explicit":
            self._agent_progress("/skill: explicit skill selected; starting…")
        else:
            self._agent_progress("/skill auto: skill selected; starting…")
        et_turn0 = self._effective_enabled_tools_for_skill(
            frozenset(self.enabled_tools), self.skills_map, sid
        )
        et_turn = self._effective_enabled_tools_for_turn(
            base_enabled_tools=et_turn0,
            enabled_toolsets=self.enabled_toolsets,
            user_query=req,
        )
        rec = self.skills_map.get(sid) or {}
        skill_prompt = (rec.get("prompt") or "").strip() if isinstance(rec, dict) else ""
        if src == "reuse":
            sink_print_compat(
                f"/skill reuse: using skill {sid!r} (model skill selection skipped). "
                f"{selection_rationale}".strip()
            )
        elif src == "explicit":
            sink_print_compat(f"/skill: using skill {sid!r}. {selection_rationale}".strip())
        else:
            sink_print_compat(f"/skill auto selected {sid!r}. {selection_rationale}".strip())
        if self.verbose >= 1:
            self._print_skill_usage_verbose(
                self.verbose,
                source=f"skill_{src or 'auto'}",
                skill_id=sid,
                base_tools=self.enabled_tools,
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
        today_str = self._today_str()
        deliverable_wanted = self._user_wants_written_deliverable(req)
        router_query = self._route_requires_websearch(
            req,
            today_str,
            self.primary_profile,
            et_turn,
            transcript_messages=self.messages,
        )
        if self._deliverable_skip_mandatory_web(req):
            router_query = None
        web_required = bool(router_query)
        steps, raw_plan = self._skill_plan_steps(
            user_request=req,
            today_str=today_str,
            skill_id=sid,
            skills_map=self.skills_map,
            primary_profile=self.primary_profile,
            _enabled_tools=et_turn,
            verbose=self.verbose,
            _system_prompt_override=self.session_system_prompt,
        )
        if steps:
            wf = ((rec.get("workflow") or {}) if isinstance(rec, dict) else {}) or {}
            step_prompt = (wf.get("step_prompt") or "").strip()
            sink_print_compat(f"Skill workflow: executing {len(steps)} step(s).", flush=True)
            self._agent_progress(f"Running {len(steps)}-step skill workflow…")
            if self.verbose >= 1:
                rp = raw_plan or ""
                cap = 1200
                preview = rp if len(rp) <= cap else rp[:cap] + "…"
                sink_print_compat(f"[*] [skills:planner] raw ({len(rp)} chars): {preview}")
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
                et_step = self._effective_enabled_tools_for_skill(
                    frozenset(self.enabled_tools), self.skills_map, sid
                )
                tit_one = (title or "")[:120]
                if len(title or "") > 120:
                    tit_one += "…"
                self._agent_progress(f"Workflow step {i}/{len(steps)}: {tit_one}")
                sprompt0 = skill_prompt
                step_system, step_user_msg = self._prepare_agent_turn_messages(
                    step_user,
                    today_str,
                    self.second_opinion_on,
                    self.cloud_ai_enabled,
                    primary_profile=self.primary_profile,
                    reviewer_ollama_model=self.reviewer_ollama_model,
                    reviewer_hosted_profile=self.reviewer_hosted_profile,
                    enabled_tools=et_step,
                    system_instruction_override=self.session_system_prompt,
                    skill_suffix=sprompt0,
                )
                self.messages.append({"role": "user", "content": step_user_msg})
                _mw_step = preferred_web_search_tool(et_step)
                if router_query and _mw_step and i == 1:
                    from agentlib.llm.tool_schemas import web_search_required_user_content

                    self.messages.append(
                        {
                            "role": "user",
                            "content": web_search_required_user_content(
                                _mw_step,
                                router_query,
                                tool_call_mode=self.settings.get_str(
                                    ("agent", "tool_call_mode"), DEFAULT_TOOL_CALL_MODE
                                ),
                                primary_profile=self.primary_profile,
                            ),
                        }
                    )
                ms, msw, mtcw, mfpw = self._agent_loop_budget()
                answered, final_answer = run_agent_conversation_turn(
                    self.messages,
                    step_user,
                    today_str,
                    self._conversation_turn_deps,
                    web_required=web_required if i == 1 else False,
                    deliverable_wanted=deliverable_wanted,
                    verbose=self.verbose,
                    second_opinion_enabled=self.second_opinion_on,
                    cloud_ai_enabled=self.cloud_ai_enabled,
                    primary_profile=self.primary_profile,
                    reviewer_hosted_profile=self.reviewer_hosted_profile,
                    reviewer_ollama_model=self.reviewer_ollama_model,
                    enabled_tools=et_step,
                    interactive_tool_recovery=True,
                    context_cfg=self.context_cfg,
                    print_answer=False,
                    max_agent_steps=ms,
                    max_agent_steps_web=msw,
                    max_tool_calls_web=mtcw,
                    max_fetch_page_web=mfpw,
                    agent_system_message=step_system,
                )
                self._agent_progress(f"Step {i}/{len(steps)} finished.")
                if final_answer:
                    step_answers.append(final_answer)
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": json.dumps({"action": "answer", "answer": final_answer}),
                        }
                    )
            if step_answers:
                sink_print_compat(step_answers[-1])
            return

        self._agent_progress("Running a single agent turn with the selected skill…")
        skill_system, skill_user = self._prepare_agent_turn_messages(
            req,
            today_str,
            self.second_opinion_on,
            self.cloud_ai_enabled,
            primary_profile=self.primary_profile,
            reviewer_ollama_model=self.reviewer_ollama_model,
            reviewer_hosted_profile=self.reviewer_hosted_profile,
            enabled_tools=et_turn,
            system_instruction_override=self.session_system_prompt,
            skill_suffix=skill_prompt,
        )
        self.messages.append({"role": "user", "content": skill_user})
        _mw2 = preferred_web_search_tool(et_turn)
        if router_query and _mw2:
            from agentlib.llm.tool_schemas import web_search_required_user_content

            self.messages.append(
                {
                    "role": "user",
                    "content": web_search_required_user_content(
                        _mw2,
                        router_query,
                        tool_call_mode=self.settings.get_str(
                            ("agent", "tool_call_mode"), DEFAULT_TOOL_CALL_MODE
                        ),
                        primary_profile=self.primary_profile,
                    ),
                }
            )
        ms, msw, mtcw, mfpw = self._agent_loop_budget()
        run_agent_conversation_turn(
            self.messages,
            req,
            today_str,
            self._conversation_turn_deps,
            web_required=web_required,
            deliverable_wanted=deliverable_wanted,
            verbose=self.verbose,
            second_opinion_enabled=self.second_opinion_on,
            cloud_ai_enabled=self.cloud_ai_enabled,
            primary_profile=self.primary_profile,
            reviewer_hosted_profile=self.reviewer_hosted_profile,
            reviewer_ollama_model=self.reviewer_ollama_model,
            enabled_tools=et_turn,
            interactive_tool_recovery=True,
            context_cfg=self.context_cfg,
            max_agent_steps=ms,
            max_agent_steps_web=msw,
            max_tool_calls_web=mtcw,
            max_fetch_page_web=mfpw,
            agent_system_message=skill_system,
        )

    def _repl_register_extension_command(self, key: str, handler: Callable[..., SessionLineResult]) -> None:
        self._repl_extension_commands[key.lower()] = handler

    def _repl_extension_help_text(self) -> str:
        """Extra ``/help`` lines from loaded REPL extensions (see :meth:`ReplExtensionRegistry.register_help`)."""
        chunks: list[str] = []
        for rec in self._repl_extensions_loaded:
            for c in rec.help_chunks:
                if isinstance(c, str) and c.strip():
                    chunks.append(c.strip())
        if not chunks:
            return ""
        lines = ["  Loaded extensions (/load):"]
        for chunk in chunks:
            for raw in chunk.splitlines():
                ln = raw.strip()
                if ln:
                    lines.append("  " + ln)
        return "\n".join(lines) + "\n"

    def _repl_sync_code_extension_lane_mode(self) -> None:
        """Set ``repl_code_extension_single_lane`` from the most recently loaded extension that registers ``/code``."""
        self.repl_code_extension_single_lane = False
        for rec in reversed(self._repl_extensions_loaded):
            if "/code" in rec.command_keys:
                self.repl_code_extension_single_lane = "single_lane" in rec.load_flags
                return

    def _repl_unwind_extension_commands(self, keys: tuple[str, ...]) -> None:
        for k in keys:
            self._repl_extension_commands.pop(k, None)

    def _repl_unload_all_extensions(self) -> None:
        seen: set[str] = set()
        for rec in reversed(self._repl_extensions_loaded):
            if rec.module_name not in seen:
                sys.modules.pop(rec.module_name, None)
                seen.add(rec.module_name)
        self._repl_extensions_loaded.clear()
        self._repl_extension_commands.clear()
        self.repl_code_extension_single_lane = False

    def _repl_drop_loaded_extension_path(self, resolved_path: Path) -> int:
        """Unload extension records that match ``resolved_path`` (same file re-``/load``ed).

        Returns how many prior records were removed.
        """
        key = str(resolved_path)
        n_before = len(self._repl_extensions_loaded)
        kept: list[_LoadedReplExtension] = []
        for rec in self._repl_extensions_loaded:
            if rec.path == key:
                self._repl_unwind_extension_commands(rec.command_keys)
            else:
                kept.append(rec)
        self._repl_extensions_loaded = kept
        self._repl_sync_code_extension_lane_mode()
        return n_before - len(kept)

    def _repl_run_post_load_lines(self, lines: list[str]) -> None:
        self._repl_post_load_depth += 1
        try:
            if self._repl_post_load_depth > MAX_REPL_POST_LOAD_DEPTH:
                sink_print_compat(
                    f"/load: post-load nesting exceeded ({MAX_REPL_POST_LOAD_DEPTH}); skipping further post-load lines."
                )
                return
            for raw in lines:
                ln = (raw or "").strip()
                if not ln:
                    continue
                self.execute_line(ln)
        finally:
            self._repl_post_load_depth -= 1

    def _try_dispatch_repl_extension(self, s: str) -> Optional[SessionLineResult]:
        """Dispatch ``/extname …`` to a loaded extension handler.

        Split only the leading command token; do **not** ``shlex.split`` the full line — user prose
        often contains apostrophes (e.g. ``doesn't``) which breaks shell parsing and drops dispatch.
        """
        if not self._repl_extension_commands:
            return None
        stripped = (s or "").strip()
        if not stripped.startswith("/"):
            return None
        i = 1
        n = len(stripped)
        while i < n and not stripped[i].isspace():
            i += 1
        key = stripped[:i].lower()
        handler = self._repl_extension_commands.get(key)
        if handler is None:
            return None
        rest = stripped[i:].lstrip()
        try:
            return handler(self, rest)
        except BaseException:
            sink_print_compat(traceback.format_exc())
            return SessionLineResult()

    def _repl_load_describe_extension_options(self, path: Path) -> SessionLineResult:
        """Import ``path`` ephemerally and print ``describe_repl_load_options()`` if present."""
        path = path.resolve()
        if not path.is_file():
            sink_print_compat(f"/load: not a file: {path}")
            return SessionLineResult()
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
        module_name = f"repl_ext_doc_{digest}"
        sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            sink_print_compat(f"/load: could not load spec for {path!r}")
            return SessionLineResult()
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(module_name, None)
            sink_print_compat(traceback.format_exc())
            return SessionLineResult()
        fn = getattr(mod, "describe_repl_load_options", None)
        if callable(fn):
            try:
                text = fn()
            except BaseException:
                sys.modules.pop(module_name, None)
                sink_print_compat(traceback.format_exc())
                return SessionLineResult()
            sys.modules.pop(module_name, None)
            if text is not None:
                sink_print_compat(str(text).rstrip())
                return SessionLineResult()
        sys.modules.pop(module_name, None)
        sink_print_compat(
            f"{path.name}: no ``describe_repl_load_options()`` hook (or it returned ``None``).\n"
            "Extension authors can define ``def describe_repl_load_options() -> str:`` to document "
            "``/load`` flags for this file. See also the module docstring and project README."
        )
        return SessionLineResult()

    def _cmd_repl_load(self, s: str) -> SessionLineResult:
        try:
            parts = shlex.split(s.strip())
        except ValueError as e:
            sink_print_compat(f"/load: {e}")
            return SessionLineResult()
        if len(parts) < 2:
            sink_print_compat(
                "/load FILE.py [ … --OPTIONS ]\n\n"
                "Load a Python file that defines ``register_repl(session, registry)``.\n"
                "Tokens after the first ``--`` following the path are treated as ``/load`` options; "
                "normalized names appear on ``registry.load_flags`` (e.g. ``--foo-bar`` → ``foo_bar``).\n"
                "The registry can ``register_command('name', handler)`` for ``/name`` lines and "
                "``register_help('…')`` to append lines to ``/help``.\n"
                "Return ``None``, a single str, or a list of str lines to run via ``execute_line`` after loading.\n\n"
                "Documentation for a specific file (does not register the extension):\n"
                "  /load FILE.py --help     or  /load FILE.py -h\n"
                "  /load info FILE.py\n\n"
                "If the module defines ``describe_repl_load_options() -> str``, that text is printed; "
                "otherwise a short default message is shown.\n\n"
                "Use ``/unload`` to drop all loaded extensions."
            )
            return SessionLineResult()
        if parts[1].lower() == "info":
            if len(parts) < 3:
                sink_print_compat("Usage: /load info FILE.py")
                return SessionLineResult()
            path_tokens = parts[2:]
            path_raw = shlex.join(path_tokens) if len(path_tokens) > 1 else path_tokens[0]
            path = Path(self._resolve_session_path(path_raw)).resolve()
            return self._repl_load_describe_extension_options(path)
        rest = parts[1:]
        help_idx: Optional[int] = None
        for i, t in enumerate(rest):
            if t in ("--help", "-h"):
                help_idx = i
                break
        if help_idx is not None:
            prefix = rest[:help_idx]
            trailing = rest[help_idx + 1 :]
            if trailing:
                sink_print_compat("/load: tokens after --help are ignored for documentation.")
            fake_parts = ["/load", *prefix]
            err, path_tokens, _raw_before_help = _split_repl_load_path_and_flags(fake_parts)
            if err:
                sink_print_compat(f"/load: {err}")
                return SessionLineResult()
            path_raw = shlex.join(path_tokens) if len(path_tokens) > 1 else path_tokens[0]
            path = Path(self._resolve_session_path(path_raw)).resolve()
            return self._repl_load_describe_extension_options(path)
        err, path_tokens, raw_flags = _split_repl_load_path_and_flags(parts)
        if err:
            sink_print_compat(f"/load: {err}")
            return SessionLineResult()
        load_flags, flag_warn = _normalize_repl_load_flags(raw_flags)
        for w in flag_warn:
            sink_print_compat(f"/load: {w}")
        path_raw = shlex.join(path_tokens) if len(path_tokens) > 1 else path_tokens[0]
        path = Path(self._resolve_session_path(path_raw)).resolve()
        if not path.is_file():
            sink_print_compat(f"/load: not a file: {path}")
            return SessionLineResult()
        if self._repl_drop_loaded_extension_path(path):
            sink_print_compat("/load: replacing earlier registration of the same file.")
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
        module_name = f"repl_ext_{digest}"
        if module_name in sys.modules:
            del sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            sink_print_compat(f"/load: could not load spec for {path!r}")
            return SessionLineResult()
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(module_name, None)
            sink_print_compat(traceback.format_exc())
            return SessionLineResult()
        register_fn = getattr(mod, "register_repl", None)
        if not callable(register_fn):
            sys.modules.pop(module_name, None)
            sink_print_compat(
                f"/load: module {path.name!r} has no callable ``register_repl(session, registry)``."
            )
            return SessionLineResult()
        reg = ReplExtensionRegistry(session=self, script_path=str(path), load_flags=load_flags)
        try:
            post = register_fn(self, reg)
        except BaseException:
            self._repl_unwind_extension_commands(tuple(dict.fromkeys(reg.command_keys())))
            self._repl_sync_code_extension_lane_mode()
            sys.modules.pop(module_name, None)
            sink_print_compat(traceback.format_exc())
            return SessionLineResult()
        if post is None:
            post_lines: list[str] = []
        elif isinstance(post, str):
            post_lines = [post] if post.strip() else []
        elif isinstance(post, (list, tuple)):
            post_lines = [str(x) for x in post if str(x).strip()]
        else:
            self._repl_unwind_extension_commands(tuple(dict.fromkeys(reg.command_keys())))
            self._repl_sync_code_extension_lane_mode()
            sys.modules.pop(module_name, None)
            sink_print_compat("/load: register_repl must return None, str, or list[str].")
            return SessionLineResult()
        keys = tuple(dict.fromkeys(reg.command_keys()))
        help_chunks = reg.help_chunks()
        self._repl_extensions_loaded.append(
            _LoadedReplExtension(module_name, str(path), keys, help_chunks=help_chunks, load_flags=load_flags)
        )
        self._repl_sync_code_extension_lane_mode()
        sink_print_compat(f"/load: registered extension from {str(path)!r} ({len(keys)} command(s)).")
        if post_lines:
            self._repl_run_post_load_lines(post_lines)
        return SessionLineResult()

    def _cmd_repl_unload(self, s: str) -> SessionLineResult:
        parts = shlex.split(s.strip())
        if len(parts) > 1:
            sink_print_compat("Usage: /unload")
            return SessionLineResult()
        if not self._repl_extensions_loaded:
            sink_print_compat("(No REPL extensions loaded.)")
            return SessionLineResult()
        self._repl_unload_all_extensions()
        sink_print_compat("Unloaded all REPL extensions.")
        return SessionLineResult()

    def _cmd_repl_extensions(self, s: str) -> SessionLineResult:
        if not self._repl_extensions_loaded:
            sink_print_compat("(No REPL extensions loaded.)")
            return SessionLineResult()
        for rec in self._repl_extensions_loaded:
            keys = ", ".join(rec.command_keys) if rec.command_keys else "(no slash commands)"
            extra = ""
            if rec.load_flags:
                extra = f"  [{', '.join(sorted(rec.load_flags))}]"
            sink_print_compat(f"  {rec.path}  →  {keys}{extra}")
        return SessionLineResult()

    def _execute_command_line(self, s: str) -> SessionLineResult:
        low = s.lower()
        cmd = (low.split(None, 1)[0] if low.strip() else "")
        if low in ("/quit", "/exit", "/q"):
            return SessionLineResult(quit=True)
        if low == "/clear":
            self.messages.clear()
            self.last_reuse_skill_id = None
            self.repl_last_user_query = None
            self.repl_last_assistant_answer = None
            sink_print_compat("Context cleared (including stored skill for /skill reuse).")
            return SessionLineResult()
        if cmd == "/compact":
            return self._cmd_compact(s)
        if low in ("/usage", "/tokens"):
            text = self._format_last_ollama_usage_for_repl()
            sink_print_compat(text.strip() if text.strip() else "No data available.")
            return SessionLineResult()
        if s.startswith("/show"):
            return self._cmd_show(s)
        if s.startswith("/while"):
            return self._cmd_while(s)
        if low.startswith("/skill"):
            return self._cmd_skill(s)
        if low.startswith("/use-skills") or low.startswith("/use-skill") or low.startswith("/reuse-skill"):
            return self._cmd_skill_backcompat(s)
        if cmd in ("/set", "/settings"):
            return self._cmd_settings(s)
        if low.startswith("/source"):
            return self._cmd_source(s)
        if low.startswith("/context"):
            return self._cmd_context(s)
        if low.startswith("/load_context"):
            return self._cmd_load_context(s)
        if low.startswith("/save_context"):
            return self._cmd_save_context(s)
        if cmd == "/unload":
            return self._cmd_repl_unload(s)
        if cmd == "/load":
            return self._cmd_repl_load(s)
        if cmd == "/extensions":
            return self._cmd_repl_extensions(s)
        if cmd in ("/cd", "/chdir"):
            return self._cmd_cd(s)
        if cmd == "/mcp" or low.startswith("/mcp "):
            return self._cmd_mcp(s)
        if low.startswith("/run_command"):
            return self._cmd_run_command(s)
        if low.startswith("/call_python"):
            return self._cmd_call_python(s)
        if low == "/list":
            return self._sink_host_ctl_result(self.host_ctl("list_agents"))
        if low.startswith("/switch"):
            try:
                parts = shlex.split(s)
            except ValueError as e:
                sink_print_compat(f"/switch: {e}")
                return SessionLineResult()
            if len(parts) < 2:
                sink_print_compat("Usage: /switch AGENT_LABEL")
                return SessionLineResult()
            return self._sink_host_ctl_result(self.host_ctl("switch", parts[1]))
        if low.startswith("/last_answer"):
            try:
                parts = shlex.split(s)
            except ValueError as e:
                sink_print_compat(f"/last_answer: {e}")
                return SessionLineResult()
            arg = parts[1] if len(parts) > 1 else None
            return self._sink_host_ctl_result(self.host_ctl("last_answer", arg))
        if low.startswith("/last_question"):
            try:
                parts = shlex.split(s)
            except ValueError as e:
                sink_print_compat(f"/last_question: {e}")
                return SessionLineResult()
            arg = parts[1] if len(parts) > 1 else None
            return self._sink_host_ctl_result(self.host_ctl("last_question", arg))
        if low == "/last" or low.startswith("/last "):
            return self._cmd_last(s)
        if low.startswith("/clipboard"):
            return self._cmd_clipboard(s)
        if low.startswith("/fork_background"):
            return self._cmd_fork_background(s)
        if low.startswith("/fork"):
            return self._cmd_fork(s)
        if s.startswith("/send"):
            return self._cmd_send_to_agent(s)
        if low in ("/help", "/?"):
            ma = ""
            if self.python_fork_agent is not None or self.python_fork_background_agent is not None:
                ma = (
                    '  /fork NAME ["cmds"]…  ·  /fork_background NAME ["cmds"]…\n'
                )
            tui_kill = ""
            if (
                self.python_fork_agent is not None
                or self.python_fork_background_agent is not None
                or self.python_host_command is not None
                or self.python_enqueue_line is not None
            ):
                tui_kill = "  /kill NAME\n"
            host_extras = ""
            if self.python_host_command is not None:
                host_extras = "  /list\n  /switch NAME\n"
            snap_extras = (
                "  /last answer|question [NAME]   (aliases: /last_answer, /last_question)\n"
                "  /clipboard copy|copy all|paste\n"
            )
            delegate_extras = ""
            if self.python_enqueue_line is not None or self.python_delegate_line is not None:
                delegate_extras = (
                    "  /send NAME CMD…  ·  "
                    '/send NAME "cmd1,cmd2,…"\n'
                )
            ext_help = self._repl_extension_help_text()
            sink_print_compat(
                "  /quit · /exit\n"
                "  /clear\n"
                "  /compact [N% | WORDS]   (default 10%; LLM compresses history, replaces messages)\n"
                "  /help · /?\n"
                "  /usage · /tokens\n"
                "  /cd DIR\n"
                "  /source FILE\n"
                "  /import FILE\n"
                "  /context load|save|start_log FILE   (aliases: /load_context, /save_context)\n"
                "  /load FILE.py  ·  /unload  ·  /extensions   (REPL extension modules)\n"
                "  /mcp …          (Model Context Protocol servers — try /mcp help)\n"
                "  /call_python …\n"
                "  /run_command …\n"
                "  ! CMD\n"
                + ma
                + tui_kill
                + host_extras
                + snap_extras
                + delegate_extras
                + ext_help
            )
            return SessionLineResult()
        ext_res = self._try_dispatch_repl_extension(s)
        if ext_res is not None:
            return ext_res
        sink_print_compat(f"Unknown command {s.split()[0]!r}. Try /help.")
        return SessionLineResult()

    def _cmd_fork(self, s: str) -> SessionLineResult:
        """`/fork` — same as TUI entry handler; required for agent_send / delegate paths."""
        from agentlib.tui_parse import parse_fork_command

        hook = self.python_fork_agent
        if hook is None:
            sink_print_compat("/fork requires a multi-agent host (e.g. agent_tui.py); fork hook not configured.")
            return SessionLineResult()
        parsed = parse_fork_command((s or "").strip())
        if parsed is None:
            sink_print_compat('Usage: /fork NAME  or  /fork NAME "cmd1,cmd2,…"')
            return SessionLineResult()
        name, cmds = parsed
        try:
            r = hook(name, cmds)
        except Exception as e:
            sink_print_compat(f"/fork: {type(e).__name__}: {e}")
            return SessionLineResult()
        out = ""
        if isinstance(r, dict):
            try:
                out = json.dumps(r, ensure_ascii=False)
            except (TypeError, ValueError):
                out = str(r)
        elif r is not None:
            out = str(r)
        return SessionLineResult(output=out)

    def _cmd_fork_background(self, s: str) -> SessionLineResult:
        """`/fork_background` — same as TUI entry handler; required for agent_send / delegate paths."""
        from agentlib.tui_parse import parse_fork_background_command

        hook = self.python_fork_background_agent
        if hook is None:
            sink_print_compat(
                "/fork_background requires a multi-agent host (e.g. agent_tui.py); fork hook not configured."
            )
            return SessionLineResult()
        parsed = parse_fork_background_command((s or "").strip())
        if parsed is None:
            sink_print_compat('Usage: /fork_background NAME  or  /fork_background NAME "cmd1,cmd2,…"')
            return SessionLineResult()
        name, cmds = parsed
        try:
            r = hook(name, cmds)
        except Exception as e:
            sink_print_compat(f"/fork_background: {type(e).__name__}: {e}")
            return SessionLineResult()
        out = ""
        if isinstance(r, dict):
            try:
                out = json.dumps(r, ensure_ascii=False)
            except (TypeError, ValueError):
                out = str(r)
        elif r is not None:
            out = str(r)
        return SessionLineResult(output=out)

    def _cmd_clipboard(self, s: str) -> SessionLineResult:
        """`/clipboard copy`, `/clipboard copy all`, `/clipboard paste` (paste returns `prefill_prompt`; host injects into input)."""

        def usage() -> None:
            sink_print_compat("/clipboard copy | copy all | paste")

        try:
            parts = shlex.split((s or "").strip())
        except ValueError as e:
            sink_print_compat(f"/clipboard: {e}")
            return SessionLineResult()

        if len(parts) < 2:
            usage()
            return SessionLineResult()

        sub = parts[1].lower()
        if sub == "copy":
            if len(parts) == 3 and parts[2].lower() == "all":
                try:
                    snap = json.dumps(self.messages, indent=2, ensure_ascii=False, default=str)
                except (TypeError, ValueError) as e:
                    sink_print_compat(f"/clipboard copy all: serialize error: {e}")
                    return SessionLineResult()
                try:
                    clipboard_write_text(snap)
                except ClipboardError as e:
                    sink_print_compat(f"/clipboard: {e}")
                    return SessionLineResult()
                sink_print_compat(f"Copied session to clipboard ({len(snap)} characters).")
                return SessionLineResult()
            if len(parts) != 2:
                usage()
                return SessionLineResult()
            ans = self.repl_last_assistant_answer
            if not isinstance(ans, str) or not ans.strip():
                sink_print_compat("(Nothing to copy: no last assistant answer in this lane yet.)")
                return SessionLineResult()
            try:
                clipboard_write_text(ans)
            except ClipboardError as e:
                sink_print_compat(f"/clipboard: {e}")
                return SessionLineResult()
            sink_print_compat(f"Copied last answer to clipboard ({len(ans)} characters).")
            return SessionLineResult()

        if sub == "paste":
            if len(parts) != 2:
                usage()
                return SessionLineResult()
            try:
                clip = clipboard_read_text()
            except ClipboardError as e:
                sink_print_compat(f"/clipboard: {e}")
                return SessionLineResult()
            line = (clip or "").replace("\x00", "").rstrip("\r\n")
            if not line.strip():
                sink_print_compat("(Clipboard is empty.)")
                return SessionLineResult()
            return SessionLineResult(
                output=f"Pasted from clipboard ({len(line)} characters) — edit below, press Enter when ready.",
                prefill_prompt=line,
            )

        usage()
        return SessionLineResult()

    def _cmd_context(self, s: str) -> SessionLineResult:
        try:
            parts = shlex.split(s.strip())
        except ValueError as e:
            sink_print_compat(f"/context: {e}")
            return SessionLineResult()
        if len(parts) < 3:
            sink_print_compat("/context load|save|start_log FILE")
            return SessionLineResult()
        sub = parts[1].lower()
        path = " ".join(parts[2:]) if len(parts) > 3 else parts[2]
        if sub == "load":
            return self._context_load_path(path)
        if sub == "save":
            return self._context_snapshot_save_path(path)
        if sub == "start_log":
            return self._context_start_log_path(path)
        sink_print_compat("/context load|save|start_log FILE")
        return SessionLineResult()

    def _cmd_primary_request_options(self, toks: list[str]) -> SessionLineResult:
        """`/set primary request_options …` — sampling / generation knobs persisted under ``primary_llm.request_options``."""
        pp = self.primary_profile

        def usage() -> None:
            sink_print_compat(
                "Usage:\n"
                "  /set primary request_options show\n"
                "  /set primary request_options clear\n"
                "  /set primary request_options set <name> <value>\n"
                "  /set primary request_options unset <name>\n"
                '  /set primary request_options merge \'{"temperature": 0.7}\'\n'
                "  /set primary request_options replace '{…}'    (whole map)\n\n"
                "Values for ``set``: numbers, true/false, JSON literals, or plain text.\n"
                "Ollama: merged into chat ``options``. Hosted: merged into chat/completions body.\n"
                "Use /set save to persist to ~/.agent.json."
            )

        sub = "show"
        if len(toks) >= 4:
            sub = toks[3].lower()

        ro = pp.request_options

        def ack() -> None:
            sink_print_compat("Primary request_options updated for this session. Use /set save to persist.")

        if sub in ("help", "-h", "--help"):
            usage()
            return SessionLineResult()

        if sub == "show":
            blob = normalize_request_options_pref(ro)
            if not blob:
                sink_print_compat("(Primary request_options is empty.)")
            else:
                sink_print_compat(json.dumps(blob, indent=2, sort_keys=True, ensure_ascii=False))
            return SessionLineResult()

        if sub == "clear":
            pp.request_options.clear()
            ack()
            return SessionLineResult()

        if sub == "unset":
            if len(toks) < 5:
                sink_print_compat("Usage: /set primary request_options unset <name>")
                return SessionLineResult()
            name = toks[4].strip()
            ro.pop(name, None)
            ro.pop(str(name).replace("-", "_"), None)
            ack()
            return SessionLineResult()

        if sub == "set":
            if len(toks) < 6:
                sink_print_compat("Usage: /set primary request_options set <name> <value>")
                return SessionLineResult()
            nk = toks[4].strip()
            tail = " ".join(toks[5:]).strip()
            try:
                ro[nk] = parse_request_option_scalar_value(tail)
            except Exception as e:
                sink_print_compat(f"/set primary request_options set: bad value ({e}).")
                return SessionLineResult()
            ack()
            return SessionLineResult()

        if sub in ("merge", "json-merge"):
            if len(toks) < 5:
                sink_print_compat('Usage: /set primary request_options merge \'{"temperature": 0.5}\'')
                return SessionLineResult()
            blob_text = " ".join(toks[4:]).strip()
            try:
                loaded = json.loads(blob_text)
            except json.JSONDecodeError as e:
                sink_print_compat(f"/set primary request_options merge: invalid JSON ({e}).")
                return SessionLineResult()
            if not isinstance(loaded, dict):
                sink_print_compat("/set primary request_options merge: JSON must be an object at the root.")
                return SessionLineResult()
            norm = normalize_request_options_pref(loaded)
            pp.request_options = {**pp.request_options, **norm}
            ack()
            return SessionLineResult()

        if sub in ("replace", "json-set"):
            if len(toks) < 5:
                sink_print_compat('Usage: /set primary request_options replace \'{"temperature": 0.2}\'')
                return SessionLineResult()
            blob_text = " ".join(toks[4:]).strip()
            try:
                loaded = json.loads(blob_text)
            except json.JSONDecodeError as e:
                sink_print_compat(f"/set primary request_options replace: invalid JSON ({e}).")
                return SessionLineResult()
            if not isinstance(loaded, dict):
                sink_print_compat("/set primary request_options replace: JSON must be an object at the root.")
                return SessionLineResult()
            pp.request_options = normalize_request_options_pref(loaded)
            ack()
            return SessionLineResult()

        usage()
        return SessionLineResult()

    def _cmd_last(self, s: str) -> SessionLineResult:
        try:
            parts = shlex.split(s.strip())
        except ValueError as e:
            sink_print_compat(f"/last: {e}")
            return SessionLineResult()
        if len(parts) < 2:
            sink_print_compat("/last answer|question [NAME]")
            return SessionLineResult()
        sub = parts[1].lower()
        arg = parts[2] if len(parts) > 2 else None
        if len(parts) > 3:
            arg = " ".join(parts[2:])
        if sub == "answer":
            return self._sink_host_ctl_result(self.host_ctl("last_answer", arg))
        if sub == "question":
            return self._sink_host_ctl_result(self.host_ctl("last_question", arg))
        sink_print_compat("/last answer|question [NAME]")
        return SessionLineResult()

    def _cmd_source(self, s: str) -> SessionLineResult:
        """
        Read a file of commands/prompts and execute them line-by-line.

        Similar to bash `source`: each non-empty line is processed as if typed into the REPL.
        """
        try:
            toks = shlex.split(s)
        except ValueError as e:
            sink_print_compat(f"/source: {e}")
            return SessionLineResult()
        if len(toks) < 2:
            sink_print_compat("Usage: /source <file>")
            return SessionLineResult()
        path = os.path.expanduser(" ".join(toks[1:]).strip())
        if not path:
            sink_print_compat("Usage: /source <file>")
            return SessionLineResult()
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except OSError as e:
            sink_print_compat(f"/source error: {e}")
            return SessionLineResult()

        executed = 0
        for raw in lines:
            line = (raw or "").rstrip("\n")
            if not line.strip():
                continue
            executed += 1
            res = self.execute_line(line)
            if bool((res or {}).get("quit", False)):
                return SessionLineResult(quit=True)
        sink_print_compat(f"Sourced {executed} line(s) from {path!r}.")
        return SessionLineResult()

    def _repl_shell_run(self, cmd: str) -> SessionLineResult:
        cmd = (cmd or "").strip()
        if not cmd:
            sink_print_compat("/run_command: missing command.")
            return SessionLineResult()

        sink_print_compat(self._session_run_command(cmd))
        return SessionLineResult()

    def _cmd_run_shell_bang(self, s: str) -> SessionLineResult:
        """``! COMMAND`` → same shell execution as ``/run_command COMMAND``."""
        t = (s or "").strip()
        if not t.startswith("!"):
            sink_print_compat("Internal error: expected line starting with '!'.")
            return SessionLineResult()
        cmd = t[1:].lstrip()
        if not cmd:
            sink_print_compat(
                "Usage: ! <shell command>\n"
                "(same as /run_command). Try /run_command help."
            )
            return SessionLineResult()
        return self._repl_shell_run(cmd)

    def _cmd_run_command(self, s: str) -> SessionLineResult:
        """Run a local shell command (``run_command`` tool backend; shell=True)."""
        t = (s or "").strip()
        low = t.lower()
        prefix = "/run_command"
        if not low.startswith(prefix):
            sink_print_compat("/run_command: invalid invocation.")
            return SessionLineResult()
        rest = t[len(prefix) :].lstrip()
        if not rest or rest.lower() in ("help", "-h", "--help", "-?"):
            sink_print_compat(
                "/run_command — run a shell command on this machine (subprocess, shell=True)\n\n"
                "Usage:\n"
                "  /run_command help\n"
                "  /run_command COMMAND       Everything after the command name is passed to your shell\n\n"
                "Shorthand:\n"
                "  ! COMMAND                  Same as /run_command COMMAND\n\n"
                "Working directory:\n"
                "  Commands run in this session's cwd (see /cd).\n\n"
                "Uses the same backend as the agent run_command tool; local/trusted use only."
            )
            return SessionLineResult()
        return self._repl_shell_run(rest)

    def _split_call_python_rest(self, s: str) -> tuple[str, object]:
        """Return (kind, payload): help | error | code | file.

        For ``file``, payload is ``list[str]``: ``[PATH, ARG, ...]`` from ``shlex.split`` (script path
        plus tokens forwarded as ``sys.argv`` for the executed script).
        """
        prefix = "/call_python"
        t = (s or "").strip()
        low = t.lower()
        if not low.startswith(prefix):
            return ("bad", None)
        rest = t[len(prefix) :].strip()
        if not rest or rest.lower() in ("help", "-h", "--help", "-?"):
            return ("help", None)
        try:
            parts = shlex.split(rest)
        except ValueError as e:
            return ("error", str(e))
        if not parts:
            return ("help", None)
        if parts[0] == "-c":
            if len(parts) < 2:
                return ("error", "/call_python -c requires Python source")
            # Allow multi-line code to be passed as a single CLI argument by embedding
            # `\n` sequences inside quotes: /call_python -c "line1\nline2".
            # Decode common escapes so users don't need literal newlines (which the REPL
            # would interpret as separate commands).
            raw = " ".join(parts[1:])
            try:
                decoded = bytes(raw, "utf-8").decode("unicode_escape")
            except Exception:
                decoded = raw
            return ("code", decoded)
        return ("file", parts)

    def _cmd_call_python(self, s: str) -> SessionLineResult:
        """
        Execute Python in this interpreter (full ``__builtins__`` — trusted users only).

        Injected globals: ``ai``, ``fork_agent``, ``list_agents``, ``switch_agent``, ``last_answer``,
        ``last_question``, ``session`` (this AgentSession),
        ``print`` (routes through emit/sink like other REPL output).

        ``ai(line)`` runs ``execute_line(line)`` on this session (LLM turns and ``/`` commands).
        ``ai(line, agent_name)`` forwards to ``python_delegate_line`` when configured (multi-agent UIs).

        ``fork_agent(name[, commands])`` calls ``python_fork_agent`` when configured.

        ``send(agent_name, cmd)`` forwards one or more commands (``cmd`` string, or iterable of strings)
        to another lane asynchronously when ``python_enqueue_line`` is wired; otherwise falls back to
        synchronous ``python_delegate_line``.

        ``list_agents()``, ``switch_agent(name)``, ``last_answer(name=None)``, ``last_question(name=None)``
        call ``session.host_ctl(...)`` when ``python_host_command`` is wired (e.g. ``agent_tui``).
        """
        kind, payload = self._split_call_python_rest(s)
        if kind == "help":
            sink_print_compat(
                "/call_python — run Python in the agent process\n\n"
                "Usage:\n"
                "  /call_python help\n"
                "  /call_python -c CODE          Python source (quote for spaces; supports \\n escapes)\n"
                "  /call_python PATH.py [ARG ...]  UTF-8 script; remaining tokens become sys.argv (after script path)\n"
                "                                  Relative paths after -f, --file, or < use this session's cwd (see /cd)\n\n"
                "Multi-line:\n"
                "  You can paste multi-line Python by opening a quote after -c and closing it on a later line.\n\n"
                "Globals:\n"
                "  ai(cmd)                       Same as typing ``cmd`` here (LLM or ``/command``).\n"
                "  ai(cmd, agent_name)           Target another agent when multi-agent hooks exist.\n"
                "  fork_agent(name [, cmds])     Fork a lane when ``python_fork_agent`` is wired.\n"
                "  send(name, cmd | cmds)       Forward one or several cmds to another agent (async when supported).\n"
                "  list_agents()                 Snapshots lanes when ``python_host_command`` is wired.\n"
                "  switch_agent(name)            Focus lane by label (host).\n"
                "  last_answer([name]) · last_question([name])   Same as ``/last answer|question`` (host).\n"
                "  session.host_ctl(op, arg)     Low-level host RPC (same ops as slash commands).\n"
                "  session                       This AgentSession.\n"
                "  print(...)                    Routed like REPL output (emit when streaming).\n"
            )
            return SessionLineResult()
        if kind == "bad":
            sink_print_compat("/call_python: invalid invocation.")
            return SessionLineResult()
        if kind == "error":
            sink_print_compat(f"/call_python: {payload}")
            return SessionLineResult()

        session = self

        def ai(cmd: str, agent_name: Optional[str] = None) -> dict:
            line = (cmd or "").strip()
            if not line:
                return {"type": "noop", "quit": False}
            sub = (agent_name or "").strip()
            if sub:
                dl = session.python_delegate_line
                if dl is None:
                    sink_print_compat(
                        "ai(..., agent_name) requires a multi-agent host "
                        "(e.g. agent_tui.py); delegate hook not configured."
                    )
                    return {"type": "command", "quit": False, "output": "delegate unavailable"}
                return dl(sub, line)
            return session.execute_line(line)

        def fork_agent(name: str, commands: Optional[Iterable[str]] = None) -> dict:
            hook = session.python_fork_agent
            cmds = None if commands is None else list(commands)
            if hook is None:
                sink_print_compat(
                    "fork_agent() requires a multi-agent host (e.g. agent_tui.py); hook not configured."
                )
                return {"type": "fork", "ok": False, "error": "fork unavailable"}
            return hook(str(name).strip(), cmds)

        def list_agents() -> dict:
            return session.host_ctl("list_agents")

        def switch_agent(name: str) -> dict:
            return session.host_ctl("switch", str(name).strip())

        def last_answer(agent_name: Optional[str] = None) -> dict:
            a = (agent_name or "").strip()
            return session.host_ctl("last_answer", a if a else None)

        def last_question(agent_name: Optional[str] = None) -> dict:
            a = (agent_name or "").strip()
            return session.host_ctl("last_question", a if a else None)

        def send(agent_name: str, cmd) -> dict:
            nm = str(agent_name or "").strip()
            if not nm:
                sink_print_compat("send() requires a non-empty agent name.")
                return {"type": "command", "quit": False, "output": "bad send"}
            cmds: list[str]
            if isinstance(cmd, str):
                sline = cmd.strip()
                cmds = [sline] if sline else []
            else:
                try:
                    cmds = [str(x).strip() for x in cmd if str(x).strip()]
                except TypeError:
                    cmds = []
            if not cmds:
                sink_print_compat("send() requires a non-empty command.")
                return {"type": "command", "quit": False, "output": "bad send"}
            eq = session.python_enqueue_line
            if eq is not None:
                try:
                    last_r = None
                    for line in cmds:
                        last_r = eq(nm, line)
                    return last_r if last_r is not None else {"ok": False, "error": "enqueue returned no result"}
                except BaseException as e:
                    return {"ok": False, "error": f"{type(e).__name__}: {e}"}
            dl = session.python_delegate_line
            if dl is None:
                sink_print_compat(
                    "send() requires a multi-agent host (e.g. agent_tui.py); enqueue/delegate hook not configured."
                )
                return {"type": "command", "quit": False, "output": "delegate unavailable"}
            try:
                last = None
                for line in cmds:
                    last = dl(nm, line)
                return last if last is not None else {"type": "command", "quit": False}
            except BaseException as e:
                sink_print_compat(f"send(): {type(e).__name__}: {e}")
                return {"type": "command", "quit": False, "output": "send failed"}

        g = {
            "__builtins__": __builtins__,
            "__name__": "__call_python__",
            "ai": ai,
            "fork_agent": fork_agent,
            "send": send,
            "list_agents": list_agents,
            "switch_agent": switch_agent,
            "last_answer": last_answer,
            "last_question": last_question,
            "session": session,
            "print": sink_print_compat,
        }

        try:
            if kind == "code":
                assert payload is not None
                filename = "<call_python -c>"
                src = payload
            else:
                assert isinstance(payload, list) and payload
                parts = payload
                path = Path(parts[0]).expanduser()
                if not path.is_file():
                    sink_print_compat(f"/call_python: not a file: {path}")
                    return SessionLineResult()
                filename = str(path.resolve())
                src = path.read_text(encoding="utf-8")
                argv_for_script = self._resolve_call_python_argv_paths([filename] + [str(x) for x in parts[1:]])
            code = compile(src, filename, "exec")
            # Use the same mapping for globals and locals so imports and top-level defs
            # live in ``g``. With ``exec(code, g, {})``, CPython treats the code like a class
            # body and binds module-level imports into the empty locals dict, so functions
            # (whose __globals__ is ``g``) see NameError for stdlib/third-party names.
            saved_argv = sys.argv
            if kind == "file":
                sys.argv = argv_for_script
            try:
                exec(code, g, g)
            finally:
                if kind == "file":
                    sys.argv = saved_argv
        except SystemExit as e:
            # Scripts often end with ``raise SystemExit(main())``; treat clean exits as success.
            code = e.code
            if code is None or code == 0 or code is False:
                return SessionLineResult()
            sink_print_compat(f"/call_python: script exited with status {code!r}")
            return SessionLineResult()
        except BaseException:
            sink_print_compat(traceback.format_exc())
            return SessionLineResult()

        return SessionLineResult()

    def _cmd_show(self, s: str) -> SessionLineResult:
        try:
            toks = shlex.split(s)
        except ValueError as e:
            sink_print_compat(f"/show: {e}")
            return SessionLineResult()
        if len(toks) < 2 or toks[1].lower() in ("help", "-h", "--help"):
            sink_print_compat(
                "Usage:\n"
                "  /show model                 Primary LLM in use (Ollama or hosted)\n"
                "  /show model <name> info     Ollama model details (POST /api/show; local Ollama only)\n"
                "  /show models                Local Ollama models available on this machine\n"
                "  /show reviewer              Second-opinion reviewer model\n"
                "\n"
                "Settings that already have a show line: /set tools, /set context show, "
                "/set thinking show, /set system_prompt show, /set prompt_template show, "
                "/set ollama|openai|agent show"
            )
            return SessionLineResult()
        if (
            len(toks) >= 4
            and toks[1].lower() == "model"
            and toks[-1].lower() == "info"
        ):
            model_name = " ".join(toks[2:-1]).strip()
            if not model_name:
                sink_print_compat("Usage: /show model <model-name> info")
                return SessionLineResult()
            if getattr(self.primary_profile, "backend", "") != "ollama":
                sink_print_compat(
                    "/show model <name> info only works with local Ollama "
                    "(primary LLM is not ollama; use /set primary llm ollama …)."
                )
                return SessionLineResult()
            base = (self.settings.get_str(("ollama", "host"), "") or "").strip().rstrip("/")
            if not base:
                base = "http://localhost:11434"
            try:
                data = fetch_ollama_model_show(
                    base, model_name, http_post=requests.post, timeout=60
                )
                sink_print_compat(json.dumps(data, ensure_ascii=False, indent=2))
            except Exception as e:
                sink_print_compat(f"/show model … info error: {e}")
            return SessionLineResult()
        sub = toks[1].lower().replace("-", "_")
        if sub in ("models", "local_models"):
            try:
                names = self._fetch_ollama_local_model_names()
                sink_print_compat("\n".join(names) if names else "(no models returned)")
            except Exception as e:
                sink_print_compat(f"/show models error: {e}")
            return SessionLineResult()
        if sub in ("model", "primary", "llm"):
            sink_print_compat(f"Primary LLM: {self._format_session_primary_llm_line(self.primary_profile)}")
            return SessionLineResult()
        if sub in ("reviewer", "second_opinion", "2nd"):
            sink_print_compat(
                "Second-opinion reviewer: "
                + self._format_session_reviewer_line(self.reviewer_hosted_profile, self.reviewer_ollama_model)
            )
            return SessionLineResult()
        sink_print_compat(
            "Unknown /show topic. Try: /show model, /show model <name> info, /show models, or /show reviewer"
        )
        return SessionLineResult()

    def _cmd_while(self, s: str) -> SessionLineResult:
        try:
            wtoks = shlex.split(s)
        except ValueError as e:
            sink_print_compat(f"/while: {e}")
            return SessionLineResult()
        if len(wtoks) == 1 or (len(wtoks) == 2 and wtoks[1].lower() in ("help", "-h", "--help")):
            sink_print_compat(
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
            return SessionLineResult()
        try:
            max_while, while_cond, body_prompts = self._parse_while_repl_tokens(wtoks)
        except ValueError as e:
            sink_print_compat(f"/while: {e}")
            return SessionLineResult()
        try:
            abort_while = False
            for wit in range(1, max_while + 1):
                try:
                    bit = self._call_while_condition_judge(
                        while_cond,
                        self.messages,
                        primary_profile=self.primary_profile,
                        verbose=self.verbose,
                    )
                except KeyboardInterrupt:
                    self._agent_progress("Cancelled /while (condition check).")
                    sink_print_compat("\n[Cancelled]\n")
                    break
                if bit == 0:
                    sink_print_compat(
                        f"/while: condition false (judge returned 0). Exiting after check {wit}/{max_while}."
                    )
                    break
                n_steps = len(body_prompts)
                for si, bp in enumerate(body_prompts, start=1):
                    uq = f"[ /while iteration {wit}/{max_while} step {si}/{n_steps} ]\n{bp}"
                    self._agent_progress(f"/while: iteration {wit}/{max_while} step {si}/{n_steps}")
                    try:
                        self._execute_user_request(uq)
                    except KeyboardInterrupt:
                        self._agent_progress("Cancelled /while (body).")
                        sink_print_compat("\n[Cancelled]\n")
                        abort_while = True
                        break
                if abort_while:
                    break
            else:
                sink_print_compat(f"/while: reached --max {max_while} without judge returning 0 (exit).")
        except Exception as e:
            sink_print_compat(f"/while error: {e}")
        return SessionLineResult()

    def _cmd_skill(self, s: str) -> SessionLineResult:
        try:
            toks = shlex.split(s)
        except ValueError as e:
            sink_print_compat(f"/skill: {e}")
            return SessionLineResult()
        if len(toks) < 2 or toks[1].lower() in ("help", "-h", "--help"):
            sink_print_compat(
                "Usage:\n"
                "  /skill list\n"
                "  /skill auto <request>\n"
                "  /skill reuse <request>\n"
                "  /skill <skill-id> <request>\n"
            )
            return SessionLineResult()
        sub = toks[1].strip()
        if sub.lower() in ("list", "ls"):
            if not self.skills_map:
                sink_print_compat("(no skills loaded)")
            else:
                sink_print_compat("Skills:")
                for sid in sorted(self.skills_map.keys()):
                    rec = self.skills_map.get(sid) or {}
                    desc = (rec.get("description") or "").strip() if isinstance(rec, dict) else ""
                    sink_print_compat(f"- {sid}" + (f": {desc}" if desc else ""))
            return SessionLineResult()
        if sub.lower() == "auto":
            req = " ".join(toks[2:]).strip()
            if not req:
                sink_print_compat("Usage: /skill auto <request>")
                return SessionLineResult()
            sid, why = self._ml_select_skill_id(
                req, self.skills_map, primary_profile=self.primary_profile, verbose=self.verbose
            )
            if not sid:
                sink_print_compat(f"/skill auto: no skill selected. {why}".strip())
                return SessionLineResult()
            self._run_with_selected_skill(req, sid, source="auto", selection_rationale=why)
            return SessionLineResult()
        if sub.lower() == "reuse":
            req = " ".join(toks[2:]).strip()
            if not req:
                sink_print_compat("Usage: /skill reuse <request>")
                return SessionLineResult()
            if not self.last_reuse_skill_id:
                sink_print_compat("/skill reuse: no stored skill. Run /skill auto <request> or /skill <id> <request> first.")
                return SessionLineResult()
            sid2 = self.last_reuse_skill_id
            if sid2 not in self.skills_map:
                sink_print_compat(
                    f"/skill reuse: stored skill {sid2!r} is not in the current skill set. "
                    "Run /skill auto again (check skills_dir / /set save)."
                )
                self.last_reuse_skill_id = None
                return SessionLineResult()
            self._run_with_selected_skill(
                req,
                sid2,
                source="reuse",
                selection_rationale="Follow-up; model skill selector skipped; same id as last skill run.",
            )
            return SessionLineResult()
        # explicit id
        sid = sub
        req = " ".join(toks[2:]).strip()
        if not sid or not req:
            sink_print_compat("Usage: /skill <skill> <request>")
            return SessionLineResult()
        if sid not in self.skills_map:
            sink_print_compat(
                f"/skill: unknown skill {sid!r}. "
                "Run /set save if you changed skills_dir, or check your skills directory."
            )
            return SessionLineResult()
        self._run_with_selected_skill(req, sid, source="explicit", selection_rationale="Explicit skill id; model skill selector skipped.")
        return SessionLineResult()

    def _cmd_skill_backcompat(self, s: str) -> SessionLineResult:
        low = s.lower()
        if low.startswith("/use-skills"):
            try:
                toks = shlex.split(s)
            except ValueError as e:
                sink_print_compat(f"/use-skills: {e}")
                return SessionLineResult()
            if len(toks) < 2:
                sink_print_compat("Usage: /use-skills <user request>")
                return SessionLineResult()
            req = " ".join(toks[1:]).strip()
            if not req:
                sink_print_compat("Usage: /use-skills <user request>")
                return SessionLineResult()
            sid, why = self._ml_select_skill_id(
                req, self.skills_map, primary_profile=self.primary_profile, verbose=self.verbose
            )
            if not sid:
                sink_print_compat(f"/use-skills: no skill selected. {why}".strip())
                return SessionLineResult()
            self._run_with_selected_skill(req, sid, source="auto", selection_rationale=why)
            return SessionLineResult()
        if low.startswith("/use-skill"):
            try:
                toks = shlex.split(s)
            except ValueError as e:
                sink_print_compat(f"/use-skill: {e}")
                return SessionLineResult()
            if len(toks) < 3:
                sink_print_compat("Usage: /use-skill <skill> <user request>")
                return SessionLineResult()
            sid = toks[1].strip()
            req = " ".join(toks[2:]).strip()
            if not sid or not req:
                sink_print_compat("Usage: /use-skill <skill> <user request>")
                return SessionLineResult()
            if sid not in self.skills_map:
                sink_print_compat(
                    f"/use-skill: unknown skill {sid!r}. "
                    "Run /set save if you changed skills_dir, or check your skills directory."
                )
                return SessionLineResult()
            self._run_with_selected_skill(req, sid, source="explicit", selection_rationale="Explicit skill id; model skill selector skipped.")
            return SessionLineResult()
        if low.startswith("/reuse-skill"):
            try:
                toks = shlex.split(s)
            except ValueError as e:
                sink_print_compat(f"/reuse-skill: {e}")
                return SessionLineResult()
            if len(toks) < 2:
                sink_print_compat("Usage: /reuse-skill <follow-up request (same skill as last /use-skills or /reuse-skill)>")
                return SessionLineResult()
            req = " ".join(toks[1:]).strip()
            if not req:
                sink_print_compat("Usage: /reuse-skill <follow-up request>")
                return SessionLineResult()
            if not self.last_reuse_skill_id:
                sink_print_compat(
                    "/reuse-skill: no stored skill. Run /use-skills <request> first, "
                    "or use a normal line for trigger-based skills."
                )
                return SessionLineResult()
            sid2 = self.last_reuse_skill_id
            if sid2 not in self.skills_map:
                sink_print_compat(
                    f"/reuse-skill: stored skill {sid2!r} is not in the current skill set. "
                    "Run /use-skills again (check skills_dir / /set save)."
                )
                self.last_reuse_skill_id = None
                return SessionLineResult()
            self._run_with_selected_skill(req, sid2, source="reuse", selection_rationale="Follow-up; model skill selector skipped; same id as last skill run.")
            return SessionLineResult()
        sink_print_compat(f"Unknown command {s.split()[0]!r}. Try /help.")
        return SessionLineResult()

    def _context_load_path(self, path: str) -> SessionLineResult:
        path = os.path.expanduser((path or "").strip())
        if not path:
            sink_print_compat("/context load FILE (alias: /load_context FILE)")
            return SessionLineResult()
        try:
            loaded = self._load_context_messages(path)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            sink_print_compat(f"Context load error: {e}")
            return SessionLineResult()
        from agentlib import prompts

        self.messages[:] = prompts.normalize_transcript_messages(loaded)
        sink_print_compat(f"Loaded {len(self.messages)} message(s) from {path!r}.")
        return SessionLineResult()

    def _context_snapshot_save_path(self, path: str) -> SessionLineResult:
        """Write the current transcript once; does not enable per-turn auto-save."""
        path = os.path.expanduser((path or "").strip())
        if not path:
            sink_print_compat("/context save FILE (alias: /save_context FILE)")
            return SessionLineResult()
        try:
            self._save_context_bundle(path, self.messages, "", None, False)
        except OSError as e:
            sink_print_compat(f"Context save error: {e}")
            return SessionLineResult()
        sink_print_compat(f"Wrote context snapshot to {path!r}.")
        return SessionLineResult()

    def _context_start_log_path(self, path: str) -> SessionLineResult:
        """Write the current transcript and append to this file after each normal REPL turn."""
        path = os.path.expanduser((path or "").strip())
        if not path:
            sink_print_compat("/context start_log FILE")
            return SessionLineResult()
        try:
            self._save_context_bundle(path, self.messages, "", None, False)
        except OSError as e:
            sink_print_compat(f"Context start_log error: {e}")
            return SessionLineResult()
        self.session_save_path = path
        sink_print_compat(f"Wrote session to {path!r}; further turns auto-save there.")
        return SessionLineResult()

    def _cmd_load_context(self, s: str) -> SessionLineResult:
        rest = s.split(None, 1)
        if len(rest) < 2:
            sink_print_compat("/context load FILE (alias: /load_context FILE)")
            return SessionLineResult()
        path = rest[1].strip()
        if not path:
            sink_print_compat("/context load FILE (alias: /load_context FILE)")
            return SessionLineResult()
        return self._context_load_path(path)

    def _cmd_save_context(self, s: str) -> SessionLineResult:
        rest = s.split(None, 1)
        if len(rest) < 2:
            sink_print_compat("/save_context FILE  (one snapshot; use /context start_log for auto-save)")
            return SessionLineResult()
        path = rest[1].strip()
        if not path:
            sink_print_compat("/save_context FILE  (one snapshot; use /context start_log for auto-save)")
            return SessionLineResult()
        return self._context_snapshot_save_path(path)

    def _cmd_set_extensions(self, toks: list) -> SessionLineResult:
        """``/set extensions …`` — optional overrides for REPL extensions (persist with ``/set save``)."""
        if len(toks) < 3:
            sink_print_compat(
                "Usage:\n"
                "  /set extensions show\n"
                "  /set extensions <id> show\n"
                "  /set extensions <id> set <key> <value>\n"
                "  /set extensions <id> unset <key>\n"
                "Example: /set extensions code_pipeline set code_test_max 8\n"
                "Valid keys depend on each extension (e.g. ``extensions/code.py`` documents "
                "``code_pipeline`` keys such as design_review_max, code_test_max, inner_round_max, "
                "parse_fail_max, user_ask_max_len).\n"
                "Use /set save to persist."
            )
            return SessionLineResult()
        a = toks[2].lower()
        if a in ("help", "-h", "--help"):
            sink_print_compat(
                "Extension settings live under ~/.agent.json in an ``extensions`` object: each "
                "``<id>`` maps to string/number/bool fields. The shipped ``extensions/code.py`` pipeline "
                "reads ``code_pipeline``.\n"
                "Commands: ``/set extensions show``, ``/set extensions code_pipeline show``, "
                "``… set <key> <value>``, ``… unset <key>``."
            )
            return SessionLineResult()
        if a in ("show", "list"):
            sink_print_compat(self.settings.extensions_show_all())
            return SessionLineResult()
        ext_id = a
        if not _EXTENSION_ID_RE.match(ext_id):
            sink_print_compat(
                f"/set extensions: invalid extension id {ext_id!r} "
                "(start with a letter; then letters, digits, underscore; max 64 chars)."
            )
            return SessionLineResult()
        if len(toks) < 4:
            sink_print_compat("Usage: /set extensions <id> show | set <key> <value> | unset <key>")
            return SessionLineResult()
        op = toks[3].lower()
        if op in ("show", "list"):
            sink_print_compat(self.settings.extensions_show_id(ext_id))
            return SessionLineResult()
        if op == "set":
            if len(toks) < 6:
                sink_print_compat("Usage: /set extensions <id> set <key> <value>")
                return SessionLineResult()
            raw_key = toks[4]
            val = " ".join(toks[5:])
            try:
                sink_print_compat(self.settings.extensions_set_kv(ext_id, raw_key, val))
            except ValueError as e:
                sink_print_compat(f"/set extensions: {e}")
            return SessionLineResult()
        if op in ("unset", "delete", "clear"):
            if len(toks) < 5:
                sink_print_compat("Usage: /set extensions <id> unset <key>")
                return SessionLineResult()
            try:
                sink_print_compat(self.settings.extensions_unset_key(ext_id, toks[4]))
            except ValueError as e:
                sink_print_compat(f"/set extensions: {e}")
            return SessionLineResult()
        sink_print_compat("Unknown /set extensions subcommand. Try: show | set | unset")
        return SessionLineResult()

    def _set_tool_or_toolset(self, phrase: str, *, enable: bool) -> bool:
        """Enable or disable a core/plugin tool id or a plugin toolset by phrase. Returns True if handled."""
        nm = phrase.strip().lower()
        if not nm:
            sink_print_compat("Usage: /set tools <tool or toolset> enable|disable")
            return True
        plugin_toolsets = self._registry.plugin_toolsets
        if nm in plugin_toolsets:
            if enable:
                self.enabled_toolsets.add(nm)
                for tid in self._registry.plugin_tools_for_toolset(nm):
                    self.enabled_tools.add(tid)
                sink_print_compat(
                    f"Toolset enabled: {nm!r} (tools may be routed per request). Use /set save to persist."
                )
            else:
                self.enabled_toolsets.discard(nm)
                for tid in self._registry.plugin_tools_for_toolset(nm):
                    self.enabled_tools.discard(tid)
                sink_print_compat(f"Toolset disabled: {nm!r}. Use /set save to persist.")
            return True
        tn = self._registry.normalize_tool_name(phrase)
        if tn:
            if enable:
                self.enabled_tools.add(tn)
                sink_print_compat(f"Tool enabled: {tn}")
            else:
                self.enabled_tools.discard(tn)
                sink_print_compat(f"Tool disabled: {tn}")
            return True
        sink_print_compat(self._registry.format_unknown_tool_hint(phrase))
        return True

    def _cmd_settings(self, s: str) -> SessionLineResult:
        try:
            toks = shlex.split(s)
        except ValueError as e:
            sink_print_compat(f"/set: {e}")
            return SessionLineResult()
        if len(toks) < 2:
            sink_print_compat("Usage: /set <topic> ...   (try: /set help)")
            return SessionLineResult()
        key = toks[1].lower().replace("-", "_")
        if key in ("help", "-h", "--help"):
            sink_print_compat(
                "Usage:\n"
                "  /set save\n"
                "  /set model <ollama-model>   (local Ollama only; for hosted/Grok use /set primary llm hosted …)\n"
                "  /set enable|disable <feature>   (second_opinion, stream_thinking, verbose, …)\n"
                "  /set tools <tool|toolset> enable|disable\n"
                "  /set tools list|reload|describe …\n"
                "  /set system_prompt ...\n"
                "  /set prompt_template ...\n"
                "  /set context ...\n"
                "  /set thinking ...\n"
                "  /set ollama|openai|agent show|keys|set|unset\n"
                "  /set primary llm ollama|hosted …\n"
                "  /set primary request_options show|set|unset|merge|replace|clear\n"
                "  /set extensions show | /set extensions <id> show | set | unset …\n"
                "  /set lock   (permanently freeze settings and MCP for this session)\n"
            )
            return SessionLineResult()

        if key == "unlock":
            sink_print_compat(
                "There is no /set unlock — settings lock is permanent for this session. "
                "Start a new session to change configuration."
            )
            return SessionLineResult()

        if key == "lock":
            if self.settings_locked:
                sink_print_compat("Settings and MCP are already locked for this session.")
            else:
                self.settings_locked = True
                sink_print_compat(
                    "Settings and MCP configuration locked for this session "
                    "(permanent; use a new session to reconfigure)."
                )
            return SessionLineResult()

        if self.settings_locked and not self._set_subcommand_read_only(key, toks):
            blocked = self._reject_if_settings_locked()
            if blocked is not None:
                return blocked

        if key == "extensions":
            return self._cmd_set_extensions(toks)

        # group-backed settings
        if key in ("ollama", "openai", "agent"):
            if len(toks) < 3:
                sink_print_compat(
                    f"Usage: /set {key} show | keys | set <name> <value> | unset <name>\n"
                    "  Keys are lowercase (e.g. host, model, api_key). After changing, use /set save."
                )
                sink_print_compat(self._settings_group_keys_lines(key))
                return SessionLineResult()
            sub = toks[2].lower()
            if sub in ("show", "list"):
                try:
                    sink_print_compat(self._settings_group_show(key))
                except (ValueError, OSError) as e:
                    sink_print_compat(f"/set {key} show: {e}")
                return SessionLineResult()
            if sub in ("keys", "key", "help"):
                try:
                    sink_print_compat(self._settings_group_keys_lines(key))
                except (ValueError, OSError) as e:
                    sink_print_compat(f"/set {key} keys: {e}")
                return SessionLineResult()
            if sub == "set":
                if len(toks) < 4:
                    sink_print_compat(f"Usage: /set {key} set <name> <value (optional, quote spaces with shlex)>")
                    return SessionLineResult()
                raw_k = toks[3]
                value = " ".join(toks[4:]) if len(toks) > 4 else ""
                try:
                    msg = self._settings_group_set(key, raw_k, value)
                except ValueError as e:
                    sink_print_compat(f"/set {key} set: {e}")
                    return SessionLineResult()
                sink_print_compat(msg)
                if (
                    key == "agent"
                    and getattr(self, "_host_app", None) is not None
                    and raw_k.strip().lower().replace("-", "_") in ("mcp_enabled", "mcp_servers")
                ):
                    try:
                        self._mcp_refresh_connections(connected_msg="[mcp] Reloaded MCP server connections.")
                    except Exception as e:
                        sink_print_compat(f"[mcp] reload failed: {e}")
                return SessionLineResult()
            if sub in ("unset", "delete", "clear"):
                if len(toks) < 4:
                    sink_print_compat(f"Usage: /set {key} unset <name>")
                    return SessionLineResult()
                try:
                    msg = self._settings_group_unset(key, toks[3])
                except ValueError as e:
                    sink_print_compat(f"/set {key} unset: {e}")
                    return SessionLineResult()
                sink_print_compat(msg)
                if (
                    key == "agent"
                    and getattr(self, "_host_app", None) is not None
                    and toks[3].strip().lower().replace("-", "_") in ("mcp_enabled", "mcp_servers")
                ):
                    try:
                        self._mcp_refresh_connections(connected_msg="[mcp] Reloaded MCP server connections.")
                    except Exception as e:
                        sink_print_compat(f"[mcp] reload failed: {e}")
                return SessionLineResult()
            sink_print_compat(f"Unknown /set {key} subcommand. Try: /set {key} show | set | unset | keys")
            return SessionLineResult()

        if key == "verbose":
            if len(toks) != 3:
                sink_print_compat("Usage: /set verbose 0|1|2|3|on|off")
                return SessionLineResult()
            tok = toks[2].strip().lower()
            if tok in ("on", "true", "yes", "y"):
                self.verbose = 2
            elif tok in ("off", "false", "no", "n"):
                self.verbose = 0
            else:
                if not tok.isdigit():
                    sink_print_compat("Usage: /set verbose 0|1|2|3|on|off")
                    return SessionLineResult()
                self.verbose = coerce_verbose_level(int(tok, 10))
            sink_print_compat(self._verbose_ack_message(self.verbose))
            return SessionLineResult()

        if key == "tools":
            if len(toks) == 2 or (len(toks) >= 3 and toks[2].lower() in ("list", "ls", "show")):
                sink_print_compat(self._registry.format_settings_tools_list(self.enabled_tools))
                plugin_toolsets = self._registry.plugin_toolsets
                if plugin_toolsets:
                    lines = ["\nToolsets (plugins):"]
                    for nm in sorted(plugin_toolsets.keys()):
                        on = "on" if nm in self.enabled_toolsets else "off"
                        desc = str((plugin_toolsets.get(nm) or {}).get("description") or "").strip()
                        lines.append(f"  [{on}] {nm}" + (f" — {desc}" if desc else ""))
                        tnames = sorted(self._registry.plugin_tools_for_toolset(nm))
                        for tid in tnames:
                            td_on = (nm in self.enabled_toolsets) and (tid in self.enabled_tools)
                            reason = ""
                            if nm not in self.enabled_toolsets:
                                reason = " (toolset off)"
                            elif tid not in self.enabled_tools:
                                reason = " (tool disabled)"
                            lines.append(f"       - {'on' if td_on else 'off'} {tid}{reason}")
                    lines.append("Enable:  /set tools <tool or toolset> enable   (e.g. /set tools web search enable)")
                    lines.append("Disable: /set tools <tool or toolset> disable")
                    lines.append("Reload plugins:    /set tools reload")
                    lines.append("Describe:          /set tools describe <tool-id>")
                    sink_print_compat("\n".join(lines))
                return SessionLineResult()
            _tool_actions = frozenset({"enable", "disable", "on", "off"})
            if len(toks) >= 4 and toks[2].lower() in _tool_actions:
                phrase = " ".join(toks[3:])
                self._set_tool_or_toolset(phrase, enable=toks[2].lower() in ("enable", "on"))
                return SessionLineResult()
            if len(toks) >= 4 and toks[-1].lower() in _tool_actions:
                phrase = " ".join(toks[2:-1])
                self._set_tool_or_toolset(phrase, enable=toks[-1].lower() in ("enable", "on"))
                return SessionLineResult()
            if len(toks) >= 3 and toks[2].lower() in ("reload", "refresh"):
                self._registry.load_plugin_toolsets(self.tools_dir)
                self._registry.register_aliases()
                if getattr(self, "_host_app", None) is not None:
                    try:
                        self._mcp_refresh_connections(announce=False)
                    except Exception:
                        pass
                sink_print_compat(
                    f"Reloaded plugin toolsets from {self.tools_dir!r}. "
                    "If MCP is enabled, resync runs in the background."
                )
                return SessionLineResult()
            if len(toks) >= 4 and toks[2].lower() in ("describe", "desc", "help"):
                tid = toks[3].strip()
                if not tid:
                    sink_print_compat("Usage: /set tools describe <tool-id>")
                    return SessionLineResult()
                nm = tid.strip().lower()
                plugin_toolsets = self._registry.plugin_toolsets
                if nm in plugin_toolsets:
                    rec = plugin_toolsets.get(nm) or {}
                    desc = str(rec.get("description") or "").strip()
                    sink_print_compat(f"Toolset: {nm}\nDescription: {desc if desc else '(none)'}")
                    sink_print_compat("Tools:")
                    for one in sorted(self._registry.plugin_tools_for_toolset(nm)):
                        sink_print_compat("  - " + one)
                    return SessionLineResult()
                sink_print_compat(self._registry.describe_tool_call_contract(tid))
                return SessionLineResult()
            sink_print_compat(
                "Usage: /set tools [list] | <tool or toolset> enable|disable | reload | describe <id>"
            )
            return SessionLineResult()

        if key == "system_prompt":
            if len(toks) < 3:
                sink_print_compat(
                    "Usage:\n"
                    "  /set system_prompt show\n"
                    "  /set system_prompt reset\n"
                    "  /set system_prompt pin\n"
                    "  /set system_prompt file <path>\n"
                    "  /set system_prompt save <path>\n"
                    "  /set system_prompt <text>\n"
                )
                return SessionLineResult()
            sub = toks[2].lower()
            if sub == "show":
                body = agent_prompts.effective_system_instruction_text_for_tools(
                    self.session_system_prompt,
                    frozenset(self.enabled_tools),
                    tool_call_mode=self.settings.get_str(
                        ("agent", "tool_call_mode"), DEFAULT_TOOL_CALL_MODE
                    ),
                    primary_profile=self.primary_profile,
                )
                sink_print_compat(f"Effective system prompt ({len(body)} chars):\n{body}")
                if self.session_system_prompt_path:
                    sink_print_compat(f"(File-backed: {self.session_system_prompt_path!r})")
                elif self.session_system_prompt is not None:
                    sink_print_compat("(Session inline override.)")
                else:
                    sink_print_compat("(Built-in default.)")
                return SessionLineResult()
            if sub in ("pin", "snapshot"):
                body = agent_prompts.effective_system_instruction_text_for_tools(
                    self.session_system_prompt,
                    frozenset(self.enabled_tools),
                    tool_call_mode=self.settings.get_str(
                        ("agent", "tool_call_mode"), DEFAULT_TOOL_CALL_MODE
                    ),
                    primary_profile=self.primary_profile,
                )
                self.session_system_prompt = body
                self.session_system_prompt_path = None
                self.session_prompt_template = None
                self._system_prompt_explicit = True
                sink_print_compat(
                    f"System prompt pinned for this session ({len(body)} chars). "
                    "Use /set save to persist to ~/.agent.json."
                )
                return SessionLineResult()
            if sub in ("reset", "default"):
                self.session_system_prompt = None
                self.session_system_prompt_path = None
                self._system_prompt_explicit = False
                sink_print_compat("System prompt reset to built-in default for this session.")
                return SessionLineResult()
            if sub == "file":
                if len(toks) < 4:
                    sink_print_compat("Usage: /set system_prompt file <path>")
                    return SessionLineResult()
                path = os.path.expanduser(" ".join(toks[3:]).strip())
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        body = f.read()
                except OSError as e:
                    sink_print_compat(f"/set system_prompt file: {e}")
                    return SessionLineResult()
                if not body.strip():
                    sink_print_compat("File is empty.")
                    return SessionLineResult()
                self.session_system_prompt = body
                self.session_system_prompt_path = os.path.abspath(path)
                self.session_prompt_template = None
                self._system_prompt_explicit = True
                sink_print_compat(
                    f"System prompt loaded from {path!r} ({len(body)} chars). "
                    "/set save will store this path in ~/.agent.json."
                )
                return SessionLineResult()
            if sub == "save":
                if len(toks) < 4:
                    sink_print_compat("Usage: /set system_prompt save <path>")
                    return SessionLineResult()
                path = os.path.expanduser(" ".join(toks[3:]).strip())
                body = agent_prompts.effective_system_instruction_text_for_tools(
                    self.session_system_prompt,
                    frozenset(self.enabled_tools),
                    tool_call_mode=self.settings.get_str(
                        ("agent", "tool_call_mode"), DEFAULT_TOOL_CALL_MODE
                    ),
                    primary_profile=self.primary_profile,
                )
                try:
                    parent = os.path.dirname(path)
                    if parent and not os.path.isdir(parent):
                        os.makedirs(parent, exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(body)
                except OSError as e:
                    sink_print_compat(f"/set system_prompt save: {e}")
                    return SessionLineResult()
                sink_print_compat(f"Wrote system prompt ({len(body)} chars) to {path!r}.")
                return SessionLineResult()
            phrase = " ".join(toks[2:])
            if not phrase.strip():
                sink_print_compat("Usage: /set system_prompt <non-empty one-line text>")
                return SessionLineResult()
            self.session_system_prompt = phrase
            self.session_system_prompt_path = None
            self.session_prompt_template = None
            self._system_prompt_explicit = True
            sink_print_compat(
                f"System prompt set inline ({len(phrase)} chars). "
                "/set save will store the text in ~/.agent.json."
            )
            return SessionLineResult()

        if key in ("prompt_template", "prompt_templates", "prompt"):
            if len(toks) < 3:
                sink_print_compat(
                    "Usage:\n"
                    "  /set prompt_template list\n"
                    "  /set prompt_template show\n"
                    "  /set prompt_template use <name>\n"
                    "  /set prompt_template default <name>\n"
                    "  /set prompt_template set <name> <text>\n"
                    "  /set prompt_template delete <name>\n"
                )
                return SessionLineResult()
            sub = toks[2].lower()
            if sub in ("help", "-h", "--help", "explain"):
                sink_print_compat("Try: /set prompt_template list")
                return SessionLineResult()
            if sub == "list":
                names = sorted(self.prompt_templates.keys())
                if not names:
                    sink_print_compat("(no prompt templates)")
                    return SessionLineResult()
                for nm in names:
                    obj = self.prompt_templates.get(nm) or {}
                    desc = str(obj.get("description") or "").strip() if isinstance(obj, dict) else ""
                    mark = ""
                    if self.session_prompt_template == nm:
                        mark = " *active*"
                    elif self.template_default == nm:
                        mark = " (default)"
                    line = f"- {nm}{mark}"
                    if desc:
                        line += f": {desc}"
                    sink_print_compat(line)
                return SessionLineResult()
            if sub == "show":
                active = self.session_prompt_template or self.template_default
                body = agent_prompts.resolve_prompt_template_text(active, self.prompt_templates) or ""
                sink_print_compat(f"Active template: {active!r}\nPrompt ({len(body)} chars):\n{body}")
                return SessionLineResult()
            if sub in ("use", "select"):
                if len(toks) < 4:
                    sink_print_compat("Usage: /set prompt_template use <name>")
                    return SessionLineResult()
                nm = toks[3].strip()
                if nm not in self.prompt_templates:
                    sink_print_compat(f"Unknown template {nm!r}. Try: /set prompt_template list")
                    return SessionLineResult()
                resolved = agent_prompts.resolve_prompt_template_text(nm, self.prompt_templates)
                if not resolved:
                    sink_print_compat(f"Template {nm!r} has no usable text/path.")
                    return SessionLineResult()
                self.session_system_prompt = resolved
                self.session_system_prompt_path = None
                self.session_prompt_template = nm
                # Selecting a prompt template is not a system_prompt override; it should not be persisted
                # as a system_prompt snapshot unless the user explicitly pins/sets it.
                self._system_prompt_explicit = False
                sink_print_compat(f"Using prompt template {nm!r} for this session.")
                return SessionLineResult()
            if sub == "default":
                if len(toks) < 4:
                    sink_print_compat("Usage: /set prompt_template default <name>")
                    return SessionLineResult()
                nm = toks[3].strip()
                if nm not in self.prompt_templates:
                    sink_print_compat(f"Unknown template {nm!r}. Try: /set prompt_template list")
                    return SessionLineResult()
                self.template_default = nm
                self._prompt_template_default_explicit = True
                sink_print_compat(f"Default prompt template set to {nm!r} (use /set save to persist).")
                return SessionLineResult()
            if sub == "set":
                if len(toks) < 5:
                    sink_print_compat("Usage: /set prompt_template set <name> <text>")
                    return SessionLineResult()
                nm = toks[3].strip()
                text = " ".join(toks[4:]).strip()
                if not nm:
                    sink_print_compat("Template name must be non-empty.")
                    return SessionLineResult()
                if not text:
                    sink_print_compat("Template text must be non-empty.")
                    return SessionLineResult()
                cur = self.prompt_templates.get(nm) or {}
                desc = str(cur.get("description") or "") if isinstance(cur, dict) else ""
                self.prompt_templates[nm] = {"kind": "overlay", "description": desc, "text": text}
                self._prompt_templates_explicit = True
                sink_print_compat(f"Template {nm!r} set/updated (overlay). Use /set save to persist.")
                return SessionLineResult()
            if sub in ("delete", "del", "rm", "remove"):
                if len(toks) < 4:
                    sink_print_compat("Usage: /set prompt_template delete <name>")
                    return SessionLineResult()
                nm = toks[3].strip()
                on_disk = os.path.join(self.prompt_templates_dir, f"{nm}.json")
                if os.path.isfile(on_disk):
                    sink_print_compat(
                        "Refusing to delete a template that exists as a file on disk in "
                        f"the configured prompt_templates_dir ({self.prompt_templates_dir!r}). "
                        "You can override it in ~/.agent.json with a same-named entry."
                    )
                    return SessionLineResult()
                if nm not in self.prompt_templates:
                    sink_print_compat(f"Unknown template {nm!r}.")
                    return SessionLineResult()
                self.prompt_templates.pop(nm, None)
                if self.session_prompt_template == nm:
                    self.session_prompt_template = None
                self._prompt_templates_explicit = True
                sink_print_compat(f"Deleted template {nm!r}. Use /set save to persist.")
                return SessionLineResult()
            sink_print_compat("Unknown subcommand. Try: /set prompt_template list")
            return SessionLineResult()

        if key in ("context", "context_manager", "context_window"):
            if len(toks) < 3:
                sink_print_compat(
                    "Usage:\n"
                    "  /set context show\n"
                    "  /set context on|off\n"
                    "  /set context tokens <n>\n"
                    "  /set context trigger <0..1>\n"
                    "  /set context target <0..1>\n"
                    "  /set context keep_tail <n>\n"
                )
                return SessionLineResult()
            sub = toks[2].lower()
            if sub == "show":
                sink_print_compat(
                    "Context manager (prefs; env vars may override):\n"
                    f"  enabled: {bool(self.context_cfg.get('enabled', True))}\n"
                    f"  tokens: {self.context_cfg.get('tokens', 0)}  (0 = auto per backend)\n"
                    f"  trigger_frac: {self.context_cfg.get('trigger_frac', 0.75)}\n"
                    f"  target_frac: {self.context_cfg.get('target_frac', 0.55)}\n"
                    f"  keep_tail_messages: {self.context_cfg.get('keep_tail_messages', 12)}\n"
                )
                return SessionLineResult()
            if sub in ("on", "enable", "enabled", "true"):
                self.context_cfg["enabled"] = True
                sink_print_compat("Context manager enabled for this session. Use /set save to persist.")
                return SessionLineResult()
            if sub in ("off", "disable", "disabled", "false"):
                self.context_cfg["enabled"] = False
                sink_print_compat("Context manager disabled for this session. Use /set save to persist.")
                return SessionLineResult()
            if sub == "tokens":
                if len(toks) < 4:
                    sink_print_compat("Usage: /set context tokens <n>")
                    return SessionLineResult()
                try:
                    n = int(toks[3], 10)
                except ValueError:
                    sink_print_compat("tokens must be an integer.")
                    return SessionLineResult()
                if n < 0:
                    n = 0
                self.context_cfg["tokens"] = n
                sink_print_compat(f"context tokens set to {n} (0 = auto). Use /set save to persist.")
                return SessionLineResult()
            if sub == "trigger":
                if len(toks) < 4:
                    sink_print_compat("Usage: /set context trigger <0..1>")
                    return SessionLineResult()
                try:
                    x = float(toks[3])
                except ValueError:
                    sink_print_compat("trigger must be a number.")
                    return SessionLineResult()
                self.context_cfg["trigger_frac"] = max(0.05, min(0.95, x))
                sink_print_compat(f"trigger_frac set to {self.context_cfg['trigger_frac']}. Use /set save to persist.")
                return SessionLineResult()
            if sub == "target":
                if len(toks) < 4:
                    sink_print_compat("Usage: /set context target <0..1>")
                    return SessionLineResult()
                try:
                    x = float(toks[3])
                except ValueError:
                    sink_print_compat("target must be a number.")
                    return SessionLineResult()
                cur_tr = float(self.context_cfg.get("trigger_frac", 0.75))
                self.context_cfg["target_frac"] = max(0.05, min(cur_tr, x))
                sink_print_compat(f"target_frac set to {self.context_cfg['target_frac']}. Use /set save to persist.")
                return SessionLineResult()
            if sub in ("keep_tail", "keep", "tail"):
                if len(toks) < 4:
                    sink_print_compat("Usage: /set context keep_tail <n>")
                    return SessionLineResult()
                try:
                    n = int(toks[3], 10)
                except ValueError:
                    sink_print_compat("keep_tail must be an integer.")
                    return SessionLineResult()
                self.context_cfg["keep_tail_messages"] = max(4, n)
                sink_print_compat(
                    f"keep_tail_messages set to {self.context_cfg['keep_tail_messages']}. Use /set save to persist."
                )
                return SessionLineResult()
            sink_print_compat("Unknown subcommand. Try: /set context show")
            return SessionLineResult()

        if key == "save":
            full_snapshot = False
            if len(toks) == 3 and toks[2].strip().lower() == "full":
                full_snapshot = True
            elif any(t.strip().lower() == "--full" for t in toks[2:]):
                full_snapshot = True
            elif len(toks) != 2:
                sink_print_compat("Usage: /set save [full|--full]")
                return SessionLineResult()
            try:
                payload = self._build_agent_prefs_payload(
                    primary_profile=self.primary_profile,
                    second_opinion_on=self.second_opinion_on,
                    cloud_ai_enabled=self.cloud_ai_enabled,
                    enabled_tools=self.enabled_tools,
                    enabled_toolsets=self.enabled_toolsets,
                    reviewer_hosted_profile=self.reviewer_hosted_profile,
                    reviewer_ollama_model=self.reviewer_ollama_model,
                    session_save_path=self.session_save_path,
                    system_prompt_override=(
                        self.session_system_prompt if self._system_prompt_explicit else None
                    ),
                    system_prompt_path_override=(
                        self.session_system_prompt_path if self._system_prompt_explicit else None
                    ),
                    prompt_templates=self.prompt_templates if self._prompt_templates_explicit else None,
                    prompt_template_default=self.template_default if self._prompt_template_default_explicit else None,
                    context_manager=self.context_cfg,
                    verbose_level=self.verbose,
                    full_snapshot=full_snapshot,
                )
                self._write_agent_prefs_file(payload)
            except OSError as e:
                sink_print_compat(f"/set save error: {e}")
                return SessionLineResult()
            sink_print_compat(f"Saved settings to {self._agent_prefs_path()!r}.")
            return SessionLineResult()

        if key == "model":
            if len(toks) < 3:
                sink_print_compat(
                    "Usage: /set model <ollama-model-name>\n"
                    "  Sets the local Ollama tag stored in ollama.model (when primary LLM is Ollama).\n"
                    "  For an OpenAI-compatible hosted API (xAI Grok, etc.), switch primary:\n"
                    "    /set primary llm hosted <base_url> <model> [api_key]\n"
                    "  Example (xAI): /set primary llm hosted https://api.x.ai/v1 grok-2-latest <key>\n"
                    "  (base_url should include /v1 — requests go to {base_url}/chat/completions.)\n"
                    "  To switch back to local Ollama as the primary LLM:\n"
                    "    /set primary llm ollama\n"
                    "  Then pick the local model tag (persist with /set save):\n"
                    "    /set model <ollama-model-name>"
                )
                return SessionLineResult()
            name = toks[2].strip()
            if not name:
                sink_print_compat(
                    "Usage: /set model <ollama-model-name>\n"
                    "  Sets the local Ollama tag stored in ollama.model (when primary LLM is Ollama).\n"
                    "  For an OpenAI-compatible hosted API (xAI Grok, etc.), switch primary:\n"
                    "    /set primary llm hosted <base_url> <model> [api_key]\n"
                    "  Example (xAI): /set primary llm hosted https://api.x.ai/v1 grok-2-latest <key>\n"
                    "  (base_url should include /v1 — requests go to {base_url}/chat/completions.)\n"
                    "  To switch back to local Ollama as the primary LLM:\n"
                    "    /set primary llm ollama\n"
                    "  Then pick the local model tag (persist with /set save):\n"
                    "    /set model <ollama-model-name>"
                )
                return SessionLineResult()
            self._settings_set(("ollama", "model"), name)
            sink_print_compat(f"ollama.model set to {name!r}. Use /set save to persist.")
            return SessionLineResult()

        if key == "enable":
            if len(toks) < 3:
                sink_print_compat(
                    "Usage: /set enable <feature>\n"
                    "  Examples: /set enable second_opinion   /set enable stream_thinking\n"
                    "  Tools: /set tools <tool or phrase> enable   (see /set tools)"
                )
                return SessionLineResult()
            phrase = " ".join(toks[2:])
            if self._registry.normalize_tool_name(phrase):
                self._set_tool_or_toolset(phrase, enable=True)
                return SessionLineResult()
            feat = self._registry.canonicalize_user_tool_phrase(phrase)
            if feat == "second_opinion":
                self.second_opinion_on = True
                sink_print_compat("second_opinion enabled for this session.")
                return SessionLineResult()
            if feat in (
                "stream_assistant",
                "streamassistant",
                "stream_answer",
                "streamanswer",
                "assistant_stream",
            ):
                self._settings_set(("agent", "stream_assistant"), True)
                sink_print_compat(
                    "stream_assistant enabled for this session (streams answer text during generation). Use /set save to persist."
                )
                return SessionLineResult()
            if feat in ("stream_thinking", "streamthinking", "stream_think", "thinking_stream", "showthinking", "show_thinking"):
                self._settings_set(("agent", "stream_thinking"), True)
                sink_print_compat(
                    "stream_thinking enabled for this session (streams model thinking when available). Use /set save to persist."
                )
                return SessionLineResult()
            if feat in (
                "native_tool_calls",
                "native_tools",
                "ollama_native_tools",
                "tool_call_mode_native",
            ):
                self._settings_set(("agent", "tool_call_mode"), "native")
                sink_print_compat(
                    "tool_call_mode set to native for this session (Ollama tools API, Phase 1). Use /set save to persist."
                )
                return SessionLineResult()
            if feat == "verbose":
                self.verbose = 2
                sink_print_compat(self._verbose_ack_message(self.verbose))
                return SessionLineResult()
            sink_print_compat(self._registry.format_unknown_tool_hint(phrase))
            return SessionLineResult()

        if key == "disable":
            if len(toks) < 3:
                sink_print_compat(
                    "Usage: /set disable <feature>\n"
                    "  Examples: /set disable second_opinion   /set disable stream_thinking\n"
                    "  Tools: /set tools <tool or phrase> disable   (see /set tools)"
                )
                return SessionLineResult()
            phrase = " ".join(toks[2:])
            if self._registry.normalize_tool_name(phrase):
                self._set_tool_or_toolset(phrase, enable=False)
                return SessionLineResult()
            feat = self._registry.canonicalize_user_tool_phrase(phrase)
            if feat == "second_opinion":
                self.second_opinion_on = False
                sink_print_compat("second_opinion disabled for this session.")
                return SessionLineResult()
            if feat in (
                "stream_assistant",
                "streamassistant",
                "stream_answer",
                "streamanswer",
                "assistant_stream",
            ):
                self._settings_set(("agent", "stream_assistant"), False)
                sink_print_compat("stream_assistant disabled for this session. Use /set save to persist.")
                return SessionLineResult()
            if feat in ("stream_thinking", "streamthinking", "stream_think", "thinking_stream", "showthinking", "show_thinking"):
                self._settings_set(("agent", "stream_thinking"), False)
                sink_print_compat("stream_thinking disabled for this session. Use /set save to persist.")
                return SessionLineResult()
            if feat == "verbose":
                self.verbose = 0
                sink_print_compat(self._verbose_ack_message(self.verbose))
                return SessionLineResult()
            sink_print_compat(self._registry.format_unknown_tool_hint(phrase))
            return SessionLineResult()

        if key == "tool_call_mode":
            if len(toks) < 3:
                sink_print_compat(
                    "Usage:\n"
                    "  /set tool_call_mode show\n"
                    "  /set tool_call_mode json|native\n"
                    "Notes:\n"
                    "  - native (default): Ollama tools API for Phase 1 core tools; auto-fallback to json if unusable.\n"
                    "  - json: agent JSON in message content (Ollama format: json).\n"
                    "  - Hosted primary LLM still uses JSON mode (Phase 2).\n"
                    "  - Use /set save to persist.\n"
                )
                return SessionLineResult()
            from agentlib.llm.calls import normalize_tool_call_mode
            from agentlib.llm.tool_schemas import NATIVE_PHASE1_TOOL_IDS

            sub = toks[2].lower()
            if sub == "show":
                mode = normalize_tool_call_mode(
                    self.settings.get_str(("agent", "tool_call_mode"), DEFAULT_TOOL_CALL_MODE)
                )
                tools = ", ".join(sorted(NATIVE_PHASE1_TOOL_IDS))
                sink_print_compat(
                    f"tool_call_mode: {mode}; native Phase 1 tools: {tools}\n"
                    "Progress lines tag tool transport: [native] (Ollama tool_calls API) vs [json] (content/thinking JSON)."
                )
                return SessionLineResult()
            if sub in ("json", "legacy", "content"):
                self._settings_set(("agent", "tool_call_mode"), "json")
                sink_print_compat(
                    "tool_call_mode set to json for this session. Use /set save to persist."
                )
                return SessionLineResult()
            if sub in ("native", "tools", "ollama"):
                self._settings_set(("agent", "tool_call_mode"), "native")
                sink_print_compat(
                    "tool_call_mode set to native (Ollama tools API, Phase 1 core tools). "
                    "Use /set save to persist."
                )
                return SessionLineResult()
            sink_print_compat("Usage: /set tool_call_mode json|native   (try /set tool_call_mode help)")
            return SessionLineResult()

        if key == "thinking":
            if len(toks) < 3:
                sink_print_compat(
                    "Usage:\n"
                    "  /set thinking show\n"
                    "  /set thinking on|off\n"
                    "  /set thinking level low|medium|high\n"
                    "Notes:\n"
                    "  - This controls the Ollama request `think` field (bool or level string).\n"
                    "  - Some models ignore booleans and require levels; others support both.\n"
                    "  - thinking on/level also enables stream_thinking automatically (use /set disable stream_thinking to hide).\n"
                    "  - Use /set save to persist.\n"
                )
                return SessionLineResult()
            sub = toks[2].lower()
            if sub == "show":
                think_v = self._ollama_request_think_value()
                lvl = self._agent_thinking_level()
                on = self._agent_thinking_enabled_default_false()
                st = "on" if on else "off"
                sink_print_compat(
                    f"thinking: {st}; level: {lvl or '(none)'}; ollama think value: {think_v!r}; stream_thinking: {self._agent_stream_thinking_enabled()}"
                )
                return SessionLineResult()
            if sub in ("on", "enable", "enabled", "true"):
                self._settings_set(("agent", "thinking"), True)
                self._settings_set(("agent", "stream_thinking"), True)
                sink_print_compat(
                    "thinking enabled for this session (and stream_thinking enabled). Use /set save to persist."
                )
                return SessionLineResult()
            if sub in ("off", "disable", "disabled", "false"):
                self._settings_set(("agent", "thinking"), False)
                self._settings_set(("agent", "thinking_level"), "")
                self._settings_set(("agent", "stream_thinking"), False)
                sink_print_compat(
                    "thinking disabled for this session (and stream_thinking disabled). Use /set save to persist."
                )
                return SessionLineResult()
            if sub == "level":
                if len(toks) < 4:
                    sink_print_compat("Usage: /set thinking level low|medium|high")
                    return SessionLineResult()
                lvl = toks[3].strip().lower()
                if lvl not in ("low", "medium", "high"):
                    sink_print_compat("thinking level must be one of: low, medium, high")
                    return SessionLineResult()
                self._settings_set(("agent", "thinking_level"), lvl)
                self._settings_set(("agent", "thinking"), True)
                self._settings_set(("agent", "stream_thinking"), True)
                sink_print_compat(
                    f"thinking level set to {lvl!r} for this session (and stream_thinking enabled). Use /set save to persist."
                )
                return SessionLineResult()
            sink_print_compat("Unknown /set thinking subcommand. Try: /set thinking show | on | off | level …")
            return SessionLineResult()

        if key == "primary" and len(toks) >= 3 and toks[2].lower() == "request_options":
            return self._cmd_primary_request_options(toks)

        if key == "primary" and len(toks) >= 4 and toks[2].lower() == "llm":
            sub = toks[3].lower()
            if sub == "ollama":
                kept_ro = preserved_request_options(self.primary_profile)
                self.primary_profile = self._default_primary_llm_profile()
                self.primary_profile.request_options = kept_ro
                sink_print_compat("Primary LLM: local Ollama.")
            elif sub == "hosted":
                if len(toks) < 6:
                    sink_print_compat("Usage: /set primary llm hosted <base_url> <model> [api_key]")
                    return SessionLineResult()
                bu, mod = toks[4], toks[5]
                if not bu.startswith(("http://", "https://")):
                    sink_print_compat("base_url must start with http:// or https://")
                    return SessionLineResult()
                keyval = toks[6] if len(toks) > 6 else ""
                kept_ro = preserved_request_options(self.primary_profile)
                self.primary_profile = self._LlmProfile(
                    backend="hosted",
                    base_url=bu,
                    model=mod,
                    api_key=keyval,
                )
                self.primary_profile.request_options = kept_ro
                if not (keyval or "").strip():
                    sink_print_compat("Note: api_key is not set; hosted primary calls will fail until it is.")
                sink_print_compat(
                    "Primary LLM: hosted OpenAI-compatible API "
                    f"({self._describe_llm_profile_short(self.primary_profile)})."
                )
            else:
                sink_print_compat("Usage: /set primary llm ollama|hosted …")
            return SessionLineResult()

        if toks[1].replace("-", "_").lower() == "second_opinion" and len(toks) >= 4 and toks[2].lower() == "llm":
            sub = toks[3].lower()
            if sub == "ollama":
                self.reviewer_hosted_profile = None
                self.reviewer_ollama_model = toks[4] if len(toks) > 4 else None
                om = self.reviewer_ollama_model or self._ollama_second_opinion_model()
                sink_print_compat(f"Second-opinion reviewer: local Ollama, model {om!r}.")
            elif sub == "hosted":
                if len(toks) < 6:
                    sink_print_compat("Usage: /set second_opinion llm hosted <base_url> <model> [api_key]")
                    return SessionLineResult()
                bu, mod = toks[4], toks[5]
                if not bu.startswith(("http://", "https://")):
                    sink_print_compat("base_url must start with http:// or https://")
                    return SessionLineResult()
                keyval = toks[6] if len(toks) > 6 else ""
                self.reviewer_hosted_profile = self._LlmProfile(
                    backend="hosted",
                    base_url=bu,
                    model=mod,
                    api_key=keyval,
                )
                self.reviewer_ollama_model = None
                if not (keyval or "").strip():
                    sink_print_compat("Note: api_key is not set; hosted second opinion will fail until it is.")
                sink_print_compat(
                    "Second-opinion reviewer: hosted "
                    f"({self._describe_llm_profile_short(self.reviewer_hosted_profile)})."
                )
            else:
                sink_print_compat("Usage: /set second_opinion llm ollama|hosted …")
            return SessionLineResult()

        sink_print_compat("Unknown /set subcommand. Try /help.")
        return SessionLineResult()

