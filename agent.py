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
from typing import AbstractSet, Optional, Tuple

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+.*",
)

import requests


def _ollama_base_url():
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _ollama_model():
    return os.environ.get("OLLAMA_MODEL", "gemma4:e4b")


@dataclass
class LlmProfile:
    """Primary or reviewer endpoint: local Ollama or an OpenAI-compatible HTTPS API."""

    backend: str  # "ollama" | "hosted"
    base_url: str = ""
    model: str = ""
    api_key_env: str = "OPENAI_API_KEY"


def default_primary_llm_profile() -> LlmProfile:
    return LlmProfile(backend="ollama")


def _apply_cli_primary_model(name: str, profile: LlmProfile) -> LlmProfile:
    """
    --model override for this process: for local Ollama, set OLLAMA_MODEL; for hosted primary,
    set LlmProfile.model.
    """
    s = (name or "").strip()
    if not s:
        return profile
    if profile.backend == "hosted":
        return replace(profile, model=s)
    os.environ["OLLAMA_MODEL"] = s
    return profile


def _read_api_key(env_name: str) -> str:
    return (os.environ.get(env_name) or "").strip()


def _hosted_review_ready(
    cloud_ai_enabled: bool, reviewer_hosted_profile: Optional[LlmProfile]
) -> bool:
    if cloud_ai_enabled and _openai_api_key():
        return True
    if (
        reviewer_hosted_profile is not None
        and reviewer_hosted_profile.backend == "hosted"
        and _read_api_key(reviewer_hosted_profile.api_key_env)
    ):
        return True
    return False


def _describe_llm_profile_short(p: LlmProfile) -> str:
    if p.backend != "hosted":
        return "ollama"
    return f"hosted {p.model!r} @ {p.base_url!r} (key: {p.api_key_env})"


# Tools we implement in this script (Ollama may emit other native tool names — ignore those).
_KNOWN_TOOLS = frozenset(
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

_TOOL_ALIASES: dict[str, str] = {}


def _canonicalize_user_tool_phrase(phrase: str) -> str:
    s = (phrase or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("-", "_")
    return s


def _register_tool_aliases() -> None:
    _TOOL_ALIASES.clear()
    for internal, _label, aliases in _TOOL_ENTRIES:
        for phrase in (internal, *aliases):
            key = _canonicalize_user_tool_phrase(phrase)
            if key:
                _TOOL_ALIASES[key] = internal


_register_tool_aliases()


def _coerce_enabled_tools(ets: Optional[AbstractSet[str]]):
    """None means all tools enabled (default)."""
    if ets is None:
        return _KNOWN_TOOLS
    return frozenset(ets)


def _resolve_tool_token(phrase: str) -> Optional[str]:
    c = _canonicalize_user_tool_phrase(phrase)
    if not c:
        return None
    if c in _KNOWN_TOOLS:
        return c
    return _TOOL_ALIASES.get(c)


def _normalize_tool_name(token: str) -> Optional[str]:
    """Map user text or internal id to canonical tool name."""
    return _resolve_tool_token(token)


def _all_tool_name_suggestion_pool() -> list[str]:
    pool = set(_KNOWN_TOOLS)
    pool.update(_TOOL_ALIASES.keys())
    pool.add("second_opinion")
    return sorted(pool)


def _format_unknown_tool_hint(phrase: str) -> str:
    c = _canonicalize_user_tool_phrase(phrase)
    lines = [f"Unknown tool {phrase!r}."]
    if c:
        near = difflib.get_close_matches(c, _all_tool_name_suggestion_pool(), n=4, cutoff=0.55)
        if near:
            bits = []
            for m in near:
                if m == "second_opinion":
                    bits.append("second_opinion (feature, not a tool)")
                    continue
                internal = _resolve_tool_token(m) or m
                bits.append(f"{m} → {internal}" if m != internal else internal)
            lines.append("Did you mean: " + ", ".join(bits) + "?")
    lines.append("Run /settings tools (or --list-tools) for every tool and its id.")
    return "\n".join(lines)


def _format_settings_tools_list(enabled_tools: AbstractSet[str]) -> str:
    lines = ["Tools for this session (id in parentheses, use either):"]
    for internal, label, _aliases in _TOOL_ENTRIES:
        on = "on" if internal in enabled_tools else "off"
        lines.append(f"  [{on}] {label}  ({internal})")
    lines.append(
        "You can use plain phrases, e.g. /settings disable web search  "
        "or  -disable_tool shell"
    )
    return "\n".join(lines)


def _tool_policy_runner_text(ets: Optional[AbstractSet[str]]) -> str:
    """Non-empty when some tools are disabled for this session."""
    e = set(_coerce_enabled_tools(ets))
    if e == set(_KNOWN_TOOLS):
        return ""
    disabled = sorted(_KNOWN_TOOLS - e)
    allowed = sorted(e & _KNOWN_TOOLS)
    return (
        "Runner: tool policy — you MUST NOT use tool_call for: "
        + ", ".join(disabled)
        + ". Only these tools may be invoked: "
        + ", ".join(allowed)
        + "."
    )


def _strip_leading_dashes_flag(a: str) -> str:
    x = (a or "").lower().replace("_", "-")
    while x.startswith("-"):
        x = x[1:]
    return x


def _print_cli_help() -> None:
    """Print usage for non-interactive `python agent.py` invocation."""
    print(
        "Usage:\n"
        "  agent.py [options] [question...]\n"
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
        "Config:   persist Ollama / OpenAI / process options in ~/.agent.json via  /settings ollama|openai|agent  in the REPL, "
        "or use /help environment  (exporting variables still overrides the file for a one-off run).\n"
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
    if (os.environ.get("AGENT_QUIET") or "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    p = (os.environ.get("AGENT_PROGRESS") or "1").strip().lower()
    return p not in ("0", "false", "no", "off")


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
        return f"Tool: search_web query={_progress_clip(p.get('query'))!r}"
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


_AGENT_PREFS_VERSION = 3

# Ollama / OpenAI / process settings: stored in ~/.agent.json as
#   "ollama": {"HOST": "…", "MODEL": "…"}  (values use short suffixes or full OLLAMA_* names)
#   "openai": {"API_KEY": "…", "BASE_URL": "…", …}
#   "agent": {"PROGRESS": "1", "QUIET": "0", …}
# On load, a key is written to os.environ only if it is not already set (so exported shell
# variables win for a one-off). Use /settings ollama|openai|agent show|set|unset to change.

_OLLAMA_FILE_ENV_KEYS: Tuple[str, ...] = (
    "OLLAMA_HOST",
    "OLLAMA_MODEL",
    "OLLAMA_SECOND_OPINION_MODEL",
    "OLLAMA_DEBUG",
    "OLLAMA_TOOL_OUTPUT_MAX",
    "OLLAMA_SEARCH_ENRICH",
)
_OPENAI_FILE_ENV_KEYS: Tuple[str, ...] = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_CLOUD_MODEL",
    "OPENAI_MODEL",
)
_AGENT_FILE_ENV_KEYS: Tuple[str, ...] = (
    "AGENT_QUIET",
    "AGENT_PROGRESS",
    "AGENT_PROMPT_TEMPLATES_DIR",
    "AGENT_SKILLS_DIR",
    "AGENT_REPL_HISTORY",
    "AGENT_REPL_INPUT_MAX_BYTES",
    "AGENT_REPL_BUFFERED_LINE",
    "AGENT_THINKING",
    "AGENT_THINKING_LEVEL",
    "AGENT_SHOW_THINKING",
    "AGENT_STREAM_THINKING",
    "AGENT_SEARCH_WEB_MAX_RESULTS",
    "AGENT_AUTO_CONFIRM_TOOL_RETRY",
    "AGENT_CONTEXT_TOKENS",
    "AGENT_HOSTED_CONTEXT_TOKENS",
    "AGENT_OLLAMA_CONTEXT_TOKENS",
    "AGENT_DISABLE_CONTEXT_MANAGER",
    "AGENT_CONTEXT_TRIGGER_FRAC",
    "AGENT_CONTEXT_TARGET_FRAC",
    "AGENT_CONTEXT_KEEP_TAIL_MESSAGES",
    "AGENT_ROUTER_TRANSCRIPT_MAX_MESSAGES",
)


def _file_env_block_prefix(kind: str) -> str:
    k = (kind or "").strip().lower()
    if k == "ollama":
        return "OLLAMA_"
    if k == "openai":
        return "OPENAI_"
    if k == "agent":
        return "AGENT_"
    raise ValueError("kind must be ollama, openai, or agent")


def _normalize_file_env_name(raw: str, kind: str) -> str:
    """Map HOST, ollama_host, or OLLAMA_HOST to OLLAMA_HOST when kind is ollama."""
    pfx = _file_env_block_prefix(kind)
    s = (raw or "").strip()
    if not s:
        raise ValueError("empty key")
    u = s.upper().replace("-", "_")
    if u.startswith(pfx):
        return u
    return pfx + u


def _short_env_key_from_full(full: str, kind: str) -> str:
    pfx = _file_env_block_prefix(kind)
    n = (full or "").strip().upper()
    if n.startswith(pfx):
        return n[len(pfx) :]
    return n


def _iter_stored_file_env_map(prefs: dict, group: str) -> dict:
    out: dict = {}
    try:
        pfx = _file_env_block_prefix(group)
    except ValueError:
        return out
    block = prefs.get(group)
    if not isinstance(block, dict):
        return out
    for sk, sv in block.items():
        if sv is None:
            continue
        t = (str(sk) if sk is not None else "").strip()
        if not t:
            continue
        try:
            full = _normalize_file_env_name(t, group)
        except ValueError:
            continue
        if not str(full).upper().startswith(pfx):
            continue
        svs = str(sv).strip() if not isinstance(sv, bool) else str(int(sv))
        if not svs:
            continue
        out[full] = svs
    return out


def _apply_stored_env_from_prefs(prefs: Optional[dict]) -> None:
    """
    Load ollama / openai / agent blobs and legacy ollama_model fields into the process
    (without overriding keys already in os.environ).
    """
    if not isinstance(prefs, dict):
        return
    for grp in ("ollama", "openai", "agent"):
        m = _iter_stored_file_env_map(prefs, grp)
        for k, v in m.items():
            if k not in os.environ:
                os.environ[k] = v
    if "OLLAMA_MODEL" not in os.environ:
        om = prefs.get("ollama_model")
        if isinstance(om, str) and om.strip():
            os.environ["OLLAMA_MODEL"] = om.strip()
    if "OLLAMA_SECOND_OPINION_MODEL" not in os.environ:
        som = prefs.get("ollama_second_opinion_model")
        if isinstance(som, str) and som.strip():
            os.environ["OLLAMA_SECOND_OPINION_MODEL"] = som.strip()


def _file_env_block_from_process(known_keys: Tuple[str, ...], kind: str) -> dict:
    """Build short-key → value dict for all known keys present in the process environment."""
    out: dict = {}
    for full in known_keys:
        if full in os.environ:
            out[_short_env_key_from_full(full, kind)] = os.environ[full]
    return out


def _file_env_key_help_lines(kind: str) -> str:
    try:
        pfx = _file_env_block_prefix(kind)
    except ValueError:
        return ""
    if kind == "ollama":
        keys = _OLLAMA_FILE_ENV_KEYS
    elif kind == "openai":
        keys = _OPENAI_FILE_ENV_KEYS
    else:
        keys = _AGENT_FILE_ENV_KEYS
    lines = []
    for k in keys:
        short = _short_env_key_from_full(k, kind)
        lines.append(f"    {short:35} ({pfx + short})")
    return "\n".join(lines)


def _file_env_set_process(kind: str, raw_key: str, value: str) -> str:
    full = _normalize_file_env_name(raw_key, kind)
    pfx = _file_env_block_prefix(kind)
    if not full.upper().startswith(pfx):
        raise ValueError(f"key must be for {kind} (prefix {pfx!r})")
    os.environ[full] = value
    return full


def _file_env_unset_process(kind: str, raw_key: str) -> str:
    full = _normalize_file_env_name(raw_key, kind)
    pfx = _file_env_block_prefix(kind)
    if not full.upper().startswith(pfx):
        raise ValueError(f"key must be for {kind} (prefix {pfx!r})")
    if full in os.environ:
        del os.environ[full]
    return full


def _agent_prefs_path() -> str:
    global _AGENT_PREFS_PATH_OVERRIDE
    if isinstance(_AGENT_PREFS_PATH_OVERRIDE, str) and _AGENT_PREFS_PATH_OVERRIDE.strip():
        return _AGENT_PREFS_PATH_OVERRIDE
    return os.path.join(os.path.expanduser("~"), ".agent.json")


_AGENT_PREFS_PATH_OVERRIDE: Optional[str] = None


def _set_agent_prefs_path_override(path: Optional[str]) -> None:
    """Override ~/.agent.json path for this process (used by --config)."""
    global _AGENT_PREFS_PATH_OVERRIDE
    p = (path or "").strip()
    if not p:
        _AGENT_PREFS_PATH_OVERRIDE = None
        return
    _AGENT_PREFS_PATH_OVERRIDE = os.path.abspath(os.path.expanduser(p))


def _parse_and_apply_cli_config_flag(argv: list[str]) -> list[str]:
    """
    Extract --config <file> or --config=<file> from argv, apply override, and return remaining args.
    This must run before loading prefs.
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--config":
            if i + 1 >= len(argv) or not str(argv[i + 1]).strip():
                print("Error: --config requires a file path.", file=sys.stderr)
                sys.exit(2)
            _set_agent_prefs_path_override(argv[i + 1])
            i += 2
            continue
        if isinstance(a, str) and a.startswith("--config="):
            p = a.split("=", 1)[1]
            if not str(p).strip():
                print("Error: --config=<file> requires a non-empty file path.", file=sys.stderr)
                sys.exit(2)
            _set_agent_prefs_path_override(p)
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def _agent_module_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _default_prompt_templates_dir() -> str:
    return os.path.join(_agent_module_dir(), "prompt_templates")


def _default_skills_dir() -> str:
    return os.path.join(_agent_module_dir(), "skills")


def _resolved_prompt_templates_dir(prefs: Optional[dict] = None) -> str:
    p = (os.environ.get("AGENT_PROMPT_TEMPLATES_DIR") or "").strip()
    if p:
        return os.path.abspath(os.path.expanduser(p))
    if prefs and isinstance(prefs, dict) and (prefs.get("prompt_templates_dir") or "").strip():
        return os.path.abspath(os.path.expanduser(str(prefs["prompt_templates_dir"]).strip()))
    return _default_prompt_templates_dir()


def _resolved_skills_dir(prefs: Optional[dict] = None) -> str:
    p = (os.environ.get("AGENT_SKILLS_DIR") or "").strip()
    if p:
        return os.path.abspath(os.path.expanduser(p))
    if prefs and isinstance(prefs, dict) and (prefs.get("skills_dir") or "").strip():
        return os.path.abspath(os.path.expanduser(str(prefs["skills_dir"]).strip()))
    return _default_skills_dir()


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
    """
    Join base_dir with relpath and return the path only if it stays under base_dir
    (prevents path traversal in reference_files).
    """
    base_dir = os.path.abspath(base_dir)
    if not relpath or not isinstance(relpath, str):
        return None
    rp = relpath.strip()
    if not rp or ".." in rp.split(os.sep):
        return None
    cand = os.path.normpath(os.path.join(base_dir, rp))
    if cand != base_dir and not cand.startswith(base_dir + os.sep):
        return None
    return cand


def _expand_skill_artifacts(skills_dir: str, meta: dict, base_prompt: str) -> str:
    """
    Append bundled reference file bodies and optional doc URLs / grounding commands
    to the skill prompt. reference_files are paths **relative to the skills_dir**
    (e.g. "references/helm_cheatsheet.md").
    """
    parts: list = []
    if (base_prompt or "").strip():
        parts.append((base_prompt or "").strip())
    ref_files = meta.get("reference_files")
    if isinstance(ref_files, list) and ref_files:
        for rel in ref_files:
            if not isinstance(rel, str) or not str(rel).strip():
                continue
            abs_p = _safe_path_under_dir(skills_dir, rel.strip())
            if abs_p is None or not os.path.isfile(abs_p):
                parts.append(
                    f"--- Reference file (missing or invalid path under skills dir): {rel} ---\n"
                )
                continue
            try:
                with open(abs_p, "r", encoding="utf-8") as f:
                    body = f.read()
            except OSError as e:
                body = f"(unreadable: {e})"
            parts.append(
                f"--- Bundled reference file: {rel} ---\n" + (body or "").rstrip() + "\n"
            )
    urls = meta.get("doc_urls")
    if isinstance(urls, list) and urls:
        lines = [str(u).strip() for u in urls if isinstance(u, str) and u.strip()]
        if lines:
            parts.append(
                "--- External docs (fetch with fetch_page when online; do not trust memory alone) ---\n"
                + "\n".join(f"- {u}" for u in lines)
                + "\n"
            )
    gcmds = meta.get("grounding_commands")
    if isinstance(gcmds, list) and gcmds:
        lines = [str(c).strip() for c in gcmds if isinstance(c, str) and c.strip()]
        if lines:
            parts.append(
                "--- Suggested grounding commands (run small steps; capture output) ---\n"
                + "\n".join(f"- `{c}`" for c in lines)
                + "\n"
            )
    return "\n\n".join(p for p in parts if p and str(p).strip()).strip()


def _load_skills_from_dir(dir_path: str) -> dict:
    """
    One skill per JSON file: skills/<id>.json
    Optional keys: description, triggers, tools, prompt, workflow,
    reference_files (list of paths under skills_dir), doc_urls, grounding_commands.
    (same structure as the former skill.json + prompt.txt, merged).
    """
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
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        meta = raw
        base_prompt = (meta.get("prompt") or "").strip() if isinstance(meta.get("prompt"), str) else ""
        prompt = _expand_skill_artifacts(dir_path, meta, base_prompt)
        tr = meta.get("triggers")
        if not isinstance(tr, list) or not tr:
            tr = [name]
        triggers = [str(t).strip() for t in tr if str(t).strip()]
        tools = meta.get("tools")
        if tools is not None and not isinstance(tools, list):
            tools = None
        workflow = meta.get("workflow")
        if workflow is not None and not isinstance(workflow, dict):
            workflow = None
        ref_files = meta.get("reference_files")
        if ref_files is not None and not isinstance(ref_files, list):
            ref_files = None
        doc_u = meta.get("doc_urls")
        if doc_u is not None and not isinstance(doc_u, list):
            doc_u = None
        gcmds = meta.get("grounding_commands")
        if gcmds is not None and not isinstance(gcmds, list):
            gcmds = None
        out[name] = {
            "id": name,
            "path": path,
            "description": (meta.get("description") or "").strip()
            if isinstance(meta.get("description"), str)
            else "",
            "triggers": triggers,
            "tools": tools,
            "prompt": prompt,
            "workflow": workflow,
            "reference_files": ref_files,
            "doc_urls": doc_u,
            "grounding_commands": gcmds,
        }
    return out


def _format_skills_for_selector(skills_map: dict) -> str:
    lines = []
    for sid in sorted(skills_map.keys()):
        rec = skills_map.get(sid) or {}
        desc = (rec.get("description") or "").strip() if isinstance(rec, dict) else ""
        w = (rec.get("workflow") or {}) if isinstance(rec, dict) else {}
        multi = bool(isinstance(w, dict) and w)
        has_art = False
        if isinstance(rec, dict):
            has_art = bool(
                (isinstance(rec.get("reference_files"), list) and rec.get("reference_files"))
                or (isinstance(rec.get("doc_urls"), list) and rec.get("doc_urls"))
                or (
                    isinstance(rec.get("grounding_commands"), list)
                    and rec.get("grounding_commands")
                )
            )
        lines.append(
            f"- id: {sid}\n  description: {desc}\n"
            f"  supports_multi_step: {str(multi).lower()}\n"
            f"  has_bundled_grounding: {str(has_art).lower()}"
        )
    return "\n".join(lines)


def _ml_select_skill_id(
    user_request: str,
    skills_map: dict,
    *,
    primary_profile: Optional[LlmProfile],
    verbose: int,
) -> Tuple[Optional[str], str]:
    """
    Ask the current primary model to choose the best skill id for the request.
    Returns (skill_id or None, rationale).
    """
    if not isinstance(skills_map, dict) or not skills_map:
        return None, "No skills loaded."
    req = (user_request or "").strip()
    if not req:
        return None, "Request is empty."
    skill_listing = _format_skills_for_selector(skills_map)
    selector_sys = (
        "You are a skill selector for a coding assistant.\n"
        "Given the user request and the available skills, pick the single best skill id.\n"
        "Respond with JSON only. No Markdown, no code fences.\n"
        'Output schema: {"skill_id":"<id or empty>","rationale":"short"}\n'
        "- Use exactly those two keys. Do not use action, tool, or answer.\n"
        "- Choose exactly one skill_id from the list.\n"
        "- If none are suitable, set skill_id to empty string.\n"
    )
    msgs = [
        {"role": "system", "content": selector_sys},
        {
            "role": "user",
            "content": (
                f"User request:\n{req}\n\n"
                f"Available skills:\n{skill_listing}\n\n"
                "Pick the best skill_id."
            ),
        },
    ]
    _agent_progress("Selecting skill (model)…")
    raw = call_llm_json_content(msgs, primary_profile, verbose=verbose)
    first = _try_json_loads_object(raw)
    if isinstance(first, dict) and first.get("_call_error"):
        return None, str(first.get("_call_error") or "LLM call failed.")
    obj = _parse_json_with_skill_id(raw)
    if not isinstance(obj, dict) or "skill_id" not in obj:
        return None, "Model did not return valid JSON with skill_id."
    sid = (obj.get("skill_id") or "").strip() if isinstance(obj, dict) else ""
    rat = (obj.get("rationale") or "").strip() if isinstance(obj, dict) else ""
    if not sid:
        return None, rat or "Model did not select a skill."
    if sid not in skills_map:
        return None, (rat + " " if rat else "") + f"Model selected unknown skill {sid!r}."
    return sid, rat or f"Selected {sid!r}."


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
    """
    If the selected skill declares a workflow, ask the model for a step plan.
    Returns (steps or None, planner_raw_text_or_error).

    Uses a dedicated planner system prompt and raw JSON content (not call_ollama_chat),
    so the model is not coerced into the main agent {action, answer} format.
    """
    # enabled_tools / system_prompt_override are intentionally unused here: the planner
    # is isolated from the main agent and must not see the long tool/JSON contract.
    rec = skills_map.get(skill_id) or {}
    wf = (rec.get("workflow") or {}) if isinstance(rec, dict) else {}
    if not isinstance(wf, dict) or not wf:
        return None, "Skill has no workflow."
    planner = (wf.get("planner_prompt") or "").strip()
    if not planner:
        return None, "Skill workflow missing planner_prompt."
    max_steps = _scalar_to_int(wf.get("max_steps"), 8)
    if max_steps < 1:
        max_steps = 8
    max_steps = min(max_steps, 20)
    skill_prompt = (rec.get("prompt") or "").strip() if isinstance(rec, dict) else ""
    # Planner only — do not prepend the main agent system instructions; they push action/answer JSON.
    plan_sys = (
        "You are a workflow planner, not the main coding agent.\n"
        "Reply with a single JSON object. No Markdown, no code fences, no action/answer format.\n"
        "Forbidden top-level keys: action, tool, tool_call, parameters, answer, next_action.\n"
        "Required top-level shape:\n"
        '{"questions":[],"steps":[{"title":"string","details":"string","success":"string"}]}\n'
        f"- questions: optional; may be [].\n"
        f"- steps: at least 1, at most {max_steps}. Each title must be non-empty.\n"
    )
    if skill_prompt:
        plan_sys += "\n--- Skill context ---\n" + skill_prompt + "\n"
    plan_sys += (
        "\n--- Planner instructions ---\n"
        + planner
        + f"\n\nContext: today's date (system clock) is {today_str}.\n"
        "If the user is vague, list questions and still provide a best-guess step plan (do not block on clarification)."
    )
    user_body = f"User request:\n{user_request}"
    msgs: list = [
        {"role": "system", "content": plan_sys},
        {"role": "user", "content": user_body},
    ]
    _agent_progress("Planning workflow (model)…")
    raw = call_llm_json_content(msgs, primary_profile, verbose=verbose)
    err0 = _try_json_loads_object(raw)
    if isinstance(err0, dict) and err0.get("_call_error"):
        return None, str(err0.get("_call_error") or "Planner call failed.")
    plan_obj = _parse_workflow_plan_dict(raw)
    if plan_obj is None and (raw or "").strip():
        repair = (
            "Your last reply was not a valid plan JSON (must include a non-empty \"steps\" array "
            "with {title, details, success} objects). Do not use action, answer, or tool. "
            "Output ONE json object only. Previous output:\n"
        )
        cap = 3200
        repair += (raw[:cap] + ("…" if len(raw) > cap else ""))
        msgs2 = list(msgs) + [{"role": "user", "content": repair}]
        _agent_progress("Re-asking model for valid step plan…")
        raw2 = call_llm_json_content(msgs2, primary_profile, verbose=verbose)
        err1 = _try_json_loads_object(raw2)
        if isinstance(err1, dict) and err1.get("_call_error"):
            return None, str(err1.get("_call_error") or "Planner retry failed.")
        plan_obj = _parse_workflow_plan_dict(raw2)
        if plan_obj is not None:
            raw = raw2
    if plan_obj is None:
        return None, raw
    steps = plan_obj.get("steps")
    if not isinstance(steps, list) or not steps:
        return None, raw
    out_steps = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        title = _scalar_to_str(st.get("title"), "").strip()
        details = _scalar_to_str(st.get("details"), "").strip()
        success = _scalar_to_str(st.get("success"), "").strip()
        if not title:
            continue
        out_steps.append({"title": title, "details": details, "success": success})
        if len(out_steps) >= max_steps:
            break
    return out_steps or None, raw


def _match_skill_detail(
    user_text: str, skills: Optional[dict]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Trigger-based skill match: pick the skill whose trigger substring is longest match in user_text.
    Returns (skill_id, winning_trigger) or (None, None).
    """
    if not user_text or not skills:
        return None, None
    low = (user_text or "").lower()
    best_sid: Optional[str] = None
    best_tr: Optional[str] = None
    best_len = 0
    for sid, data in skills.items():
        d = data if isinstance(data, dict) else {}
        for tr in d.get("triggers") or [sid]:
            t = (str(tr) or "").lower().strip()
            if not t:
                continue
            if t in low and len(t) > best_len:
                best_len = len(t)
                best_sid = str(sid)
                best_tr = t
    return best_sid, best_tr


def _match_skill_id(user_text: str, skills: Optional[dict]) -> Optional[str]:
    sid, _ = _match_skill_detail(user_text, skills)
    return sid


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
    wanted = {str(t).strip() for t in raw if isinstance(t, str) and t.strip() in _KNOWN_TOOLS}
    if not wanted:
        return base_enabled
    narrowed = wanted & set(base_enabled)
    return frozenset(narrowed) if narrowed else base_enabled


_REPL_READLINE_INSTALLED = False


def _repl_history_path() -> str:
    override = (os.environ.get("AGENT_REPL_HISTORY") or "").strip()
    if override:
        return os.path.expanduser(override)
    return os.path.join(os.path.expanduser("~"), ".agent_repl_history")


def _agent_stream_thinking_enabled() -> bool:
    """
    Whether to stream `message.thinking` chunks to the user during Ollama streaming.
    Prefer AGENT_STREAM_THINKING; keep AGENT_SHOW_THINKING as a backward-compatible alias.
    """
    raw = (os.environ.get("AGENT_STREAM_THINKING") or "").strip()
    if not raw:
        raw = (os.environ.get("AGENT_SHOW_THINKING") or "").strip()
    v = raw.lower()
    return v in ("1", "true", "yes", "y", "on")


def _agent_thinking_level() -> Optional[str]:
    v = (os.environ.get("AGENT_THINKING_LEVEL") or "").strip().lower()
    if v in ("low", "medium", "high"):
        return v
    return None


def _agent_thinking_enabled_default_false() -> bool:
    """
    Default behavior: thinking OFF unless explicitly enabled.
    Set AGENT_THINKING=1|on|true to enable; 0|off|false to disable.
    """
    raw = (os.environ.get("AGENT_THINKING") or "").strip().lower()
    if not raw:
        return False
    if raw in ("0", "false", "no", "n", "off"):
        return False
    if raw in ("1", "true", "yes", "y", "on"):
        return True
    return False


def _ollama_request_think_value() -> object:
    """
    Value for Ollama's request-level `think` field:
    - If AGENT_THINKING_LEVEL is set: one of "low"|"medium"|"high"
    - Else: boolean based on AGENT_THINKING (default False)
    """
    lvl = _agent_thinking_level()
    if lvl:
        return lvl
    enabled = bool(_agent_thinking_enabled_default_false())
    if not enabled:
        return False
    # gpt-oss models ignore boolean think; they require a level string.
    try:
        mod = (_ollama_model() or "").strip().lower()
    except Exception:
        mod = ""
    if mod.startswith("gpt-oss"):
        return "medium"
    return True


def _file_env_default_value_display(full: str) -> str:
    """
    String shown for a process env name when that variable is not set in os.environ
    (matches the code's implied default for each key).
    """
    if full == "OLLAMA_HOST":
        return "http://localhost:11434"
    if full == "OLLAMA_MODEL":
        return "gemma4:e4b"
    if full == "OLLAMA_SECOND_OPINION_MODEL":
        return "llama3.2:latest"
    if full == "OLLAMA_DEBUG":
        return ""
    if full == "OLLAMA_TOOL_OUTPUT_MAX":
        return "14000"
    if full == "OLLAMA_SEARCH_ENRICH":
        return "1"
    if full == "OPENAI_API_KEY":
        return ""
    if full == "OPENAI_BASE_URL":
        return "https://api.openai.com/v1"
    if full in ("OPENAI_CLOUD_MODEL", "OPENAI_MODEL"):
        return "gpt-4o-mini"
    if full == "AGENT_QUIET":
        return "0"
    if full == "AGENT_PROGRESS":
        return "1"
    if full == "AGENT_PROMPT_TEMPLATES_DIR":
        return _default_prompt_templates_dir()
    if full == "AGENT_SKILLS_DIR":
        return _default_skills_dir()
    if full == "AGENT_REPL_HISTORY":
        return os.path.join(os.path.expanduser("~"), ".agent_repl_history")
    if full == "AGENT_REPL_INPUT_MAX_BYTES":
        return "131072"  # same as _REPL_INPUT_MAX_DEFAULT (defined below)
    if full == "AGENT_REPL_BUFFERED_LINE":
        return "0"
    if full == "AGENT_THINKING":
        return "0"
    if full == "AGENT_THINKING_LEVEL":
        return ""
    if full == "AGENT_SHOW_THINKING":
        return "0"
    if full == "AGENT_STREAM_THINKING":
        return "0"
    if full == "AGENT_SEARCH_WEB_MAX_RESULTS":
        return "5"
    if full == "AGENT_AUTO_CONFIRM_TOOL_RETRY":
        return "0"
    if full == "AGENT_CONTEXT_TOKENS":
        return "0"
    if full in ("AGENT_HOSTED_CONTEXT_TOKENS", "AGENT_OLLAMA_CONTEXT_TOKENS"):
        return "131072"
    if full == "AGENT_DISABLE_CONTEXT_MANAGER":
        return "0"
    if full == "AGENT_CONTEXT_TRIGGER_FRAC":
        return "0.75"
    if full == "AGENT_CONTEXT_TARGET_FRAC":
        return "0.55"
    if full == "AGENT_CONTEXT_KEEP_TAIL_MESSAGES":
        return "12"
    if full == "AGENT_ROUTER_TRANSCRIPT_MAX_MESSAGES":
        return "80"
    return ""


def _file_env_effective_value_line(full: str) -> tuple[str, bool]:
    """
    (display_value, is_builtin_default) — second is True when full is not in os.environ
    and the first string is the built-in default, not a saved value.
    """
    if full in os.environ:
        v = os.environ[full]
    else:
        v = _file_env_default_value_display(full)
    vdisp = (v if v is not None else "").replace("\n", " ")
    if len(vdisp) > 120:
        vdisp = vdisp[:120] + "…"
    is_default = full not in os.environ
    return (vdisp, is_default)


def _format_file_env_group_show(kind: str) -> str:
    try:
        pfx = _file_env_block_prefix(kind)
    except ValueError:
        return ""
    if kind == "ollama":
        keys = _OLLAMA_FILE_ENV_KEYS
    elif kind == "openai":
        keys = _OPENAI_FILE_ENV_KEYS
    else:
        keys = _AGENT_FILE_ENV_KEYS
    lines = [
        f"{kind.upper()} options (in ~/.agent.json; shell env wins if set before start). "
        "If a name is not set, the value shown is the built-in default (see (default))."
    ]
    for k in keys:
        short = _short_env_key_from_full(k, kind)
        vdisp, is_def = _file_env_effective_value_line(k)
        tag = " (default)" if is_def else ""
        lines.append(f"  {short} = {vdisp!r}{tag}  ({k})")
    return "\n".join(lines)


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


def _repl_env_flag_true(name: str, default: str) -> bool:
    v = (os.environ.get(name) or default).strip().lower()
    return v not in ("0", "false", "no", "off", "")


def _repl_buffered_line_max_bytes() -> int:
    return max(4096, _scalar_to_int(os.environ.get("AGENT_REPL_INPUT_MAX_BYTES"), _REPL_INPUT_MAX_DEFAULT))


def _repl_read_line(prompt: str) -> str:
    """
    Read one REPL line. Default: input() + readline (history, arrows, usual editing).

    Set AGENT_REPL_BUFFERED_LINE=1 on a TTY to read from stdin in binary mode up to
    AGENT_REPL_INPUT_MAX_BYTES (default 128KiB) for large single-line pastes; readline.add_history
    is called afterward so ↑ still recalls those lines (without per-line readline editing on entry).
    """
    if not sys.stdin.isatty():
        return input(prompt)
    if not _repl_env_flag_true("AGENT_REPL_BUFFERED_LINE", "0"):
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
    path = _agent_prefs_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _llm_profile_to_pref(profile: LlmProfile) -> dict:
    if profile.backend != "hosted":
        return {"backend": "ollama"}
    d = {
        "backend": "hosted",
        "base_url": profile.base_url,
        "model": profile.model,
        "api_key_env": profile.api_key_env,
    }
    key = _read_api_key(profile.api_key_env)
    if key:
        d["api_key"] = key
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
    keyenv = _scalar_to_str(obj.get("api_key_env"), "OPENAI_API_KEY").strip() or "OPENAI_API_KEY"
    if not bu.startswith(("http://", "https://")) or not mod:
        return None
    prof = LlmProfile(backend="hosted", base_url=bu, model=mod, api_key_env=keyenv)
    ak = obj.get("api_key")
    if isinstance(ak, str) and ak.strip():
        os.environ[keyenv] = ak.strip()
    return prof


def _write_agent_prefs_file(payload: dict) -> None:
    path = _agent_prefs_path()
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    parent = os.path.dirname(path) or os.path.expanduser("~")
    fd, tmp = tempfile.mkstemp(prefix=".agent.", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _build_agent_prefs_payload(
    *,
    primary_profile: LlmProfile,
    second_opinion_on: bool,
    cloud_ai_enabled: bool,
    enabled_tools: AbstractSet[str],
    reviewer_hosted_profile: Optional[LlmProfile],
    reviewer_ollama_model: Optional[str],
    session_save_path: Optional[str],
    system_prompt_override: Optional[str] = None,
    system_prompt_path_override: Optional[str] = None,
    prompt_templates: Optional[dict] = None,
    prompt_template_default: Optional[str] = None,
    prompt_templates_dir: Optional[str] = None,
    skills_dir: Optional[str] = None,
    context_manager: Optional[dict] = None,
    verbose_level: int = 0,
) -> dict:
    om = (os.environ.get("OLLAMA_MODEL") or "").strip() or None
    som = (os.environ.get("OLLAMA_SECOND_OPINION_MODEL") or "").strip() or None
    payload: dict = {
        "version": _AGENT_PREFS_VERSION,
        "ollama_model": om,
        "ollama_second_opinion_model": som,
        "second_opinion_enabled": bool(second_opinion_on),
        "cloud_ai_enabled": bool(cloud_ai_enabled),
        "verbose": _coerce_verbose_level(verbose_level),
        "primary_llm": _llm_profile_to_pref(primary_profile),
        "enabled_tools": sorted(enabled_tools)
        if len(enabled_tools) < len(_KNOWN_TOOLS)
        else None,
    }
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
    if context_manager is not None:
        payload["context_manager"] = context_manager
    oa = _file_env_block_from_process(_OLLAMA_FILE_ENV_KEYS, "ollama")
    oi = _file_env_block_from_process(_OPENAI_FILE_ENV_KEYS, "openai")
    ag = _file_env_block_from_process(_AGENT_FILE_ENV_KEYS, "agent")
    if oa:
        payload["ollama"] = oa
    if oi:
        payload["openai"] = oi
    if ag:
        payload["agent"] = ag
    return payload


def _session_defaults_from_prefs(prefs: Optional[dict]) -> dict:
    if isinstance(prefs, dict):
        _apply_stored_env_from_prefs(prefs)
    _pt = _default_prompt_templates_dir()
    _sk = _default_skills_dir()
    out = {
        "enabled_tools": set(_KNOWN_TOOLS),
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
    """Default max DDG result rows: env AGENT_SEARCH_WEB_MAX_RESULTS, else 5, clamped 1–30."""
    raw = (os.environ.get("AGENT_SEARCH_WEB_MAX_RESULTS") or "5").strip()
    try:
        v = int(float(raw))
    except (TypeError, ValueError):
        v = 5
    return max(1, min(30, v))


def _search_web_max_results_clamped(n: object, *, fallback: int) -> int:
    if n is None or isinstance(n, bool):
        return fallback
    try:
        v = int(float(n))
    except (TypeError, ValueError):
        t = _scalar_to_str(n, "").strip()
        if not t:
            return fallback
        try:
            v = int(float(t))
        except (TypeError, ValueError):
            return fallback
    return max(1, min(30, v))


def _search_web_effective_max_results(params: object) -> int:
    """
    max_results (or max / num_results / n / limit) in tool parameters;
    if absent, use AGENT_SEARCH_WEB_MAX_RESULTS / 5.
    """
    p = params if isinstance(params, dict) else {}
    d = _default_search_web_max_results()
    for k in ("max_results", "max", "num_results", "n", "limit"):
        if p.get(k) is not None:
            return _search_web_max_results_clamped(p.get(k), fallback=d)
    return d


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
    """
    Bias web search toward the present when the query asks who holds a role but omits
    words like 'current' (weak models often mirror the user literally and get stale hits).
    """
    if os.environ.get("OLLAMA_SEARCH_ENRICH", "1").strip() in ("0", "false", "no"):
        return query
    q = (query or "").strip()
    if not q:
        return q
    low = q.lower()
    # Historical / ordinal / past-focused — do not rewrite
    if re.search(
        r"\b(who was|who were|first |second |third |fourth |fifth |"
        r"\d{1,2}(st|nd|rd|th) |original |founding )\b",
        low,
    ):
        return q
    if re.search(r"\b(in 19[0-9]{2}|in 20[01][0-9])\b", low):
        return q
    # Already has a present-day or year hint
    if re.search(r"\b(current|today|now|present|incumbent|latest)\b", low):
        # If the user asked for "current/today/latest" but didn't anchor a year,
        # append the system year to reduce stale hits and improve consistency.
        if not re.search(r"\b20[0-9]{2}\b", low):
            y = datetime.date.today().year
            return f"{q.rstrip('.')} {y}"
        return q
    if re.search(r"\b20[2-9][0-9]\b", low):
        return q
    if re.search(r"\b(who is|who's|who are)\b", low) and re.search(
        r"\b(president|prime minister|governor|mayor|senator|representative|"
        r"ceo|chancellor|speaker|chief justice|king|queen)\b",
        low,
    ):
        y = datetime.date.today().year
        return f"{q.rstrip('.')} current {y}"
    return q


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
        if mapped and t in _KNOWN_TOOLS and t in et:
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


def _try_json_loads_object(s: str):
    """Parse a JSON object string; apply light repairs for common model mistakes."""
    if not s or not s.strip():
        return None
    s = s.strip()
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass
    for fix in (
        s,
        re.sub(r",\s*}", "}", s),
        re.sub(r",\s*]", "]", s),
    ):
        try:
            v = json.loads(fix)
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            continue
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
        tool = d.get("tool") or (action if action in _KNOWN_TOOLS else None)
        has_action = 1 if action else 0
        known_tool = 1 if tool in _KNOWN_TOOLS else 0
        tool_call_shape = 1 if action == "tool_call" and tool in _KNOWN_TOOLS else 0
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
    max_len = int(os.environ.get("OLLAMA_TOOL_OUTPUT_MAX", "14000"))
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
    return (os.environ.get("AGENT_AUTO_CONFIRM_TOOL_RETRY") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )


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
    return os.environ.get("OLLAMA_SECOND_OPINION_MODEL", "llama3.2:latest").strip()


def _openai_api_key():
    return (os.environ.get("OPENAI_API_KEY") or "").strip()


def _openai_base_url():
    return (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")


def _openai_cloud_model():
    return (os.environ.get("OPENAI_CLOUD_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()


def call_ollama_plaintext(messages: list, model: str) -> str:
    """Ollama /api/chat without JSON format — for second-opinion reviewer text."""
    base = _ollama_base_url()
    url = f"{base}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": _ollama_request_think_value(),
    }

    def run_chat(streaming: bool) -> Tuple[str, Optional[dict]]:
        body = {**payload, "stream": streaming}
        if streaming:
            with requests.post(url, json=body, stream=True, timeout=600) as r:
                r.raise_for_status()
                msg, usage, _ = _merge_stream_message_chunks(
                    r.iter_lines(decode_unicode=True), stream_chunks=False
                )
            return (msg.get("content") or "").strip(), usage
        r = requests.post(url, json=body, timeout=600)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message") or {}
        return (msg.get("content") or "").strip(), _ollama_usage_from_chat_response(data)

    text, _usage = run_chat(streaming=True)
    if not text:
        text, _usage = run_chat(streaming=False)
    return text or "(empty reviewer response)"


def call_hosted_chat_plain(messages: list, profile: LlmProfile) -> str:
    """Non-streaming chat.completions for OpenAI-compatible APIs (OpenAI, Grok, Groq, Azure, etc.)."""
    key = _read_api_key(profile.api_key_env)
    if not key:
        return f"Cloud AI error: {profile.api_key_env} is not set."
    base = profile.base_url.rstrip("/")
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": profile.model,
        "messages": messages,
        "stream": False,
        "temperature": 0.3,
    }
    try:
        r = requests.post(url, json=body, headers=headers, timeout=120)
        r.raise_for_status()
        data = r.json()
        choice0 = (data.get("choices") or [{}])[0]
        msg = choice0.get("message") or {}
        return (msg.get("content") or "").strip() or "(empty cloud response)"
    except Exception as e:
        return f"Cloud AI error: {e}"


def call_openai_chat_plain(messages: list) -> str:
    """Legacy env-based OpenAI-compatible call (OPENAI_*)."""
    prof = LlmProfile(
        backend="hosted",
        base_url=_openai_base_url(),
        model=_openai_cloud_model(),
        api_key_env="OPENAI_API_KEY",
    )
    return call_hosted_chat_plain(messages, prof)


def call_llm_json_content(
    messages: list,
    primary_profile: Optional[LlmProfile] = None,
    *,
    verbose: int = 0,
) -> str:
    """
    One-shot model call: return assistant *content* as stored by the model.
    Does NOT run agent post-processing (no _message_to_agent_json_text), so
    the reply can be arbitrary JSON (skill selector, workflow plan, etc.).

    Local Ollama uses /api/chat with format=json. Hosted uses chat.completions
    and returns the message content string.
    """
    global _last_ollama_chat_usage
    prof = primary_profile or default_primary_llm_profile()
    if prof.backend == "hosted":
        _last_ollama_chat_usage = None
        key = _read_api_key(prof.api_key_env)
        if not key:
            return json.dumps({"_call_error": f"{prof.api_key_env} is not set."})
        base = prof.base_url.rstrip("/")
        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": prof.model,
            "messages": messages,
            "stream": False,
            "temperature": 0.2,
        }
        try:
            r = requests.post(url, json=body, headers=headers, timeout=300)
            r.raise_for_status()
            data = r.json()
            choice0 = (data.get("choices") or [{}])[0]
            msg = choice0.get("message") or {}
            text = (msg.get("content") or "").strip()
            if verbose >= 2 and text:
                print(text, flush=True)
            return text or ""
        except Exception as e:
            return json.dumps({"_call_error": f"Hosted JSON call error: {e}"})

    _last_ollama_chat_usage = None
    base = _ollama_base_url()
    url = f"{base}/api/chat"
    payload = {
        "model": _ollama_model(),
        "messages": messages,
        "stream": True,
        "format": "json",
        "think": False,
    }
    try:

        def run_once(streaming: bool) -> Tuple[str, Optional[dict]]:
            body = {**payload, "stream": streaming}
            if streaming:
                with requests.post(url, json=body, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    msg, usage, _ = _merge_stream_message_chunks(
                        r.iter_lines(decode_unicode=True), stream_chunks=verbose >= 2
                    )
            else:
                r = requests.post(url, json=body, timeout=300)
                r.raise_for_status()
                data = r.json()
                msg = data.get("message") or {}
                usage = _ollama_usage_from_chat_response(data)
            return ((msg.get("content") or "").strip(), usage)

        text, usage = run_once(streaming=True)
        if not text:
            text, usage = run_once(streaming=False)
        if usage:
            _last_ollama_chat_usage = usage
        if verbose >= 2 and text:
            print(flush=True)
        return text
    except Exception as e:
        return json.dumps({"_call_error": f"Ollama JSON call error: {e}"})


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


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw, 10)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


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
    lim = _env_int("AGENT_CONTEXT_TOKENS", 0)
    if lim > 0:
        return lim
    if profile is not None and getattr(profile, "backend", "") == "hosted":
        return _env_int("AGENT_HOSTED_CONTEXT_TOKENS", 131072)
    return _env_int("AGENT_OLLAMA_CONTEXT_TOKENS", 131072)


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
    # Env overrides prefs.
    enabled = bool(cfg.get("enabled", True))
    if _env_int("AGENT_DISABLE_CONTEXT_MANAGER", 0) == 1:
        enabled = False
    if not enabled:
        return messages

    trigger_frac = float(cfg.get("trigger_frac", 0.75))
    target_frac = float(cfg.get("target_frac", 0.55))
    keep_tail = int(cfg.get("keep_tail_messages", 12))

    trigger_frac = _env_float("AGENT_CONTEXT_TRIGGER_FRAC", trigger_frac)
    target_frac = _env_float("AGENT_CONTEXT_TARGET_FRAC", target_frac)
    keep_tail = _env_int("AGENT_CONTEXT_KEEP_TAIL_MESSAGES", keep_tail)

    trigger_frac = max(0.05, min(0.95, trigger_frac))
    target_frac = max(0.05, min(trigger_frac, target_frac))
    keep_tail = max(4, keep_tail)

    limit = int(cfg.get("tokens", 0) or 0)
    limit = _env_int("AGENT_CONTEXT_TOKENS", limit)
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
    """Hosted primary agent: same JSON contract as Ollama /api/chat + format json."""
    key = _read_api_key(profile.api_key_env)
    if not key:
        return json.dumps(
            {"action": "error", "error": f"{profile.api_key_env} is not set."}
        )
    base = profile.base_url.rstrip("/")
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": profile.model,
        "messages": messages,
        "stream": False,
        "temperature": 0.3,
    }
    try:
        r = requests.post(url, json=body, headers=headers, timeout=600)
        r.raise_for_status()
        data = r.json()
        choice0 = (data.get("choices") or [{}])[0]
        msg = choice0.get("message") or {}
        text = _message_to_agent_json_text(msg, enabled_tools).strip()
        if not text:
            return json.dumps({"action": "answer", "answer": "No response from model."})
        if verbose >= 2:
            print(text, flush=True)
            _verbose_emit_final_agent_readable(text)
        return text
    except Exception as e:
        print(f"[DEBUG] Hosted chat error: {e}")
        return json.dumps({"action": "error", "error": str(e)})


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
    """
    Agent chat: local Ollama JSON /api/chat, or hosted OpenAI-compatible chat.completions
    when primary_profile.backend == \"hosted\".
    """
    global _last_ollama_chat_usage
    prof = primary_profile or default_primary_llm_profile()
    if prof.backend == "hosted":
        _last_ollama_chat_usage = None
        return call_hosted_agent_chat(messages, prof, enabled_tools, verbose=verbose)

    base = _ollama_base_url()
    url = f"{base}/api/chat"
    payload = {
        "model": _ollama_model(),
        "messages": messages,
        "stream": True,
        "format": "json",
        "think": _ollama_request_think_value(),
    }
    debug = os.environ.get("OLLAMA_DEBUG", "")

    stream_llm = verbose >= 2

    def run_chat(streaming: bool) -> Tuple[str, Optional[dict], bool]:
        body = {**payload, "stream": streaming}
        if streaming:
            with requests.post(url, json=body, stream=True, timeout=600) as r:
                r.raise_for_status()
                msg, usage, streamed = _merge_stream_message_chunks(
                    r.iter_lines(decode_unicode=True), stream_chunks=stream_llm
                )
            if debug:
                print("[DEBUG] Ollama merged message:", msg)
            text = _message_to_agent_json_text(msg, enabled_tools)
            return text, usage, streamed
        r = requests.post(url, json=body, timeout=600)
        r.raise_for_status()
        data = r.json()
        if debug:
            print("[DEBUG] Ollama API response:", data)
        msg = data.get("message") or {}
        text = _message_to_agent_json_text(msg, enabled_tools)
        usage = _ollama_usage_from_chat_response(data)
        if stream_llm and text.strip():
            print(text, flush=True)
            return text, usage, True
        return text, usage, False

    try:
        text, usage, streamed = run_chat(streaming=True)
        text = text.strip()
        if not text:
            text2, usage2, streamed2 = run_chat(streaming=False)
            text = text2.strip()
            if usage2:
                usage = usage2
            streamed = streamed or streamed2
        if usage:
            _last_ollama_chat_usage = usage
        if stream_llm:
            if streamed:
                print(flush=True)
            if text:
                _verbose_emit_final_agent_readable(text)
        if stream_llm and usage:
            print(_format_ollama_usage_line(usage))
        if not text:
            return json.dumps({"action": "answer", "answer": "No response from model."})
        return text
    except Exception as e:
        print(f"[DEBUG] Request error: {e}")
        return json.dumps({"action": "error", "error": str(e)})


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
    """
    DuckDuckGo HTML results + fallbacks.

    Optional tool parameters (in ``params``): max_results (or max, num_results, n, limit) —
    number of DDG result rows to include, 1–30, default from env ``AGENT_SEARCH_WEB_MAX_RESULTS`` or 5.
    """
    query = _scalar_to_str(query, "")
    query = _enrich_search_query_for_present_day(query)
    mr = _search_web_effective_max_results(params or {})
    parts = []
    ia = _ddg_instant_answer(query)
    if ia:
        parts.append("[DuckDuckGo instant answer]\n" + ia)
    page = _fetch_ddg_html(query)
    blocked = "anomaly-modal" in page or "bots use DuckDuckGo" in page
    rows = [] if blocked else _parse_ddg_html_results(page, max_results=mr)
    if rows:
        parts.append("[Web results]\n" + "\n".join(rows))
    elif blocked:
        parts.append(
            "[Note] DuckDuckGo returned a bot-check page instead of HTML results "
            "(common for datacenter IPs). Instant answer and Wikipedia fallback still apply."
        )
    if not rows:
        wiki = _wikipedia_opensearch(query, result_limit=mr)
        if wiki:
            parts.append("[Wikipedia search]\n" + wiki)
        # If we still don't have real snippets, fetch the top Wikipedia page for an extract.
        # This reduces the chance that a stale instant answer is the only signal.
        wiki_extract = _wikipedia_top_page_extract(query)
        if wiki_extract:
            parts.append("[Wikipedia top result extract]\n" + wiki_extract)
    if not parts:
        return (
            "No results found for this search. "
            "Try search_web again with a shorter or alternate query, a product/site/organization name, "
            "or a year (e.g. 2026) for time-sensitive topics. If the user provided a URL, use fetch_page on that URL."
        )
    return "\n\n".join(parts)


def fetch_page(url):
    url = _scalar_to_str(url, "").strip()
    if not url:
        return "Fetch error: empty url string."
    if not re.match(r"^https?://", url, re.IGNORECASE):
        return f"Fetch error: URL must start with http:// or https:// (got {url!r})."
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    last_exc: Optional[BaseException] = None
    for attempt in (0, 1):
        timeout = 12.0 if attempt == 0 else 22.0
        try:
            resp = requests.get(
                url, headers=headers, timeout=timeout, allow_redirects=True
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt == 0:
                time.sleep(0.35)
                continue
            return f"Fetch error: {e}"
        except requests.exceptions.RequestException as e:
            return f"Fetch error: {e}"
        st = int(resp.status_code)
        if st in (429, 500, 502, 503, 504) and attempt == 0:
            time.sleep(0.4)
            continue
        if st >= 400:
            return (
                f"Fetch error: HTTP {st} for this URL. "
                "If access is denied, try a different page (docs, help, or search_web for an official link). "
                "Do not use run_command with curl."
            )
        final_url = resp.url
        text = re.sub(r"<[^>]*>", " ", resp.text)
        text = re.sub(r"\s+", " ", text).strip()
        prefix = f"Fetched URL: {url}\nFinal URL: {final_url}\n\n"
        if attempt == 1:
            prefix = "[After automatic retry] " + prefix
        return prefix + text[:5000]
    if last_exc is not None:
        return f"Fetch error: {last_exc}"
    return f"Fetch error: could not retrieve {url!r} after retry."


def run_command(command):
    command = _scalar_to_str(command, "")
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60
        )
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return f"Command error: {e}"


def _tool_fault_result(tool: str, exc: BaseException) -> str:
    """Convert a tool crash/exception into a stable string the model can reason about."""
    t = (tool or "").strip() or "(unknown tool)"
    en = type(exc).__name__
    msg = str(exc).strip()
    if not msg:
        msg = repr(exc)
    return f"Tool fault: {t} raised {en}: {msg}"


def use_git(params) -> str:
    """Vetted git operations via argument lists (no shell)."""
    p = params if isinstance(params, dict) else {}
    op = _scalar_to_str(p.get("op") or p.get("operation"), "").strip().lower()
    wt = _scalar_to_str(
        p.get("worktree") or p.get("cwd") or p.get("path") or "", ""
    ).strip()
    try:
        cwd0 = os.path.abspath(os.path.expanduser(wt)) if wt else os.getcwd()
    except Exception:
        cwd0 = os.getcwd()
    if not os.path.isdir(cwd0):
        return f"use_git error: worktree is not a directory: {cwd0}"

    def _git_run(args: list, timeout: int = 180) -> str:
        try:
            r = subprocess.run(
                ["git", *args],
                cwd=cwd0,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = (r.stdout or "").rstrip()
            err = (r.stderr or "").rstrip()
            lines = []
            if out:
                lines.append(out)
            if err:
                lines.append("STDERR:\n" + err)
            lines.append(f"(exit {r.returncode})")
            return "\n".join(lines)
        except Exception as e:
            return f"use_git error: {e}"

    check = subprocess.run(
        ["git", "-C", cwd0, "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if check.returncode != 0:
        return (
            f"use_git error: not a git work tree (or git missing): {cwd0}\n"
            f"{(check.stderr or check.stdout or '').strip()}"
        )

    def _repo_path(one: str) -> str:
        s = str(one).strip()
        if not s:
            return s
        p = os.path.expanduser(s)
        if not os.path.isabs(p):
            p = os.path.join(cwd0, p)
        return os.path.normpath(p)

    if not op:
        return "use_git error: parameters.op is required (status, log, diff, add, commit, push, pull, branch)."
    if op in ("status", "st"):
        return _git_run(["status", "-sb"])
    if op == "log":
        n = _scalar_to_int(p.get("n") or p.get("lines"), 20)
        if n is None or n < 1:
            n = 20
        n = min(int(n), 200)
        return _git_run(["log", "--oneline", f"-n{n}"])
    if op == "diff":
        stg = p.get("staged")
        if isinstance(stg, str):
            stg = stg.strip().lower() in ("1", "true", "yes", "on")
        return _git_run(["diff", "--staged"] if stg else ["diff"])
    if op == "add":
        paths = p.get("paths")
        if paths is None and p.get("path"):
            paths = [p.get("path")]
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, list) or not paths:
            return "use_git error: add requires parameters.paths (non-empty list of path strings)."
        args = ["add", "--"]
        for one in paths:
            args.append(_repo_path(str(one)))
        return _git_run(args)
    if op == "commit":
        msg = _scalar_to_str(p.get("message") or p.get("m"), "").strip()
        if not msg:
            return "use_git error: commit requires parameters.message."
        return _git_run(["commit", "-m", msg], timeout=120)
    if op == "push":
        rem = _scalar_to_str(p.get("remote"), "origin").strip() or "origin"
        br = _scalar_to_str(p.get("branch"), "").strip()
        args = ["push", rem]
        if br:
            args.append(br)
        return _git_run(args, timeout=300)
    if op == "pull":
        rem = _scalar_to_str(p.get("remote"), "origin").strip() or "origin"
        br = _scalar_to_str(p.get("branch"), "").strip()
        args = ["pull", rem]
        if br:
            args.append(br)
        return _git_run(args, timeout=300)
    if op in ("branch", "branches"):
        return _git_run(["branch", "-a", "-vv"])
    return f"use_git error: unknown op {op!r} (try status, log, diff, add, commit, push, pull, branch)."


def write_file(path, content):
    path = _scalar_to_str(path, "")
    content = _scalar_to_str(content, "")
    if not path.strip():
        return "Write error: path is empty."
    if not content.strip():
        return (
            "Write error: parameters.content is required (non-empty string) with the full file body. "
            "Do not call write_file with only a path; for a letter or document, put the entire text in content."
        )
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"File {path} written successfully."
    except Exception as e:
        return f"Write error: {e}"


def list_directory(path):
    path = _scalar_to_str(path, "")
    try:
        entries = os.listdir(path)
        return json.dumps(entries)
    except Exception as e:
        return f"List dir error: {e}"


def read_file(path):
    path = _scalar_to_str(path, "")
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Read error: {e}"


def download_file(url, path):
    url = _scalar_to_str(url, "")
    path = _scalar_to_str(path, "")
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
        return f"File downloaded to {path}."
    except Exception as e:
        return f"Download error: {e}"


def tail_file(path, lines=20):
    path = _scalar_to_str(path, "")
    lines = _scalar_to_int(lines, 20)
    if lines < 1:
        lines = 20
    try:
        with open(path, "r") as f:
            content = f.readlines()[-lines:]
        return "".join(content)
    except Exception as e:
        return f"Tail error: {e}"


def replace_text(path, pattern, replacement, replace_all=True):
    path = _scalar_to_str(path, "")
    pattern = _scalar_to_str(pattern, "")
    replacement = _scalar_to_str(replacement, "")
    if replace_all is None:
        replace_all = True
    if isinstance(replace_all, str):
        replace_all = replace_all.strip().lower() in ("1", "true", "yes", "on")
    try:
        with open(path, "r") as f:
            text = f.read()
        flags = 0
        if not replace_all:
            flags = 1
        new_text = re.sub(pattern, replacement, text, flags=flags)
        with open(path, "w") as f:
            f.write(new_text)
        return f"Replaced text in {path}."
    except Exception as e:
        return f"Replace error: {e}"


def call_python(code, globals=None):
    code = _scalar_to_str(code, "")
    if not code.strip():
        return "Exec error: empty code string."
    try:
        compiled = compile(code, "<call_python>", "exec")
    except SyntaxError as e:
        return (
            "Exec error: not valid Python source (call_python only runs Python, not shell scripts or prose). "
            f"{e.msg} at line {e.lineno}. For letters, essays, or files use write_file, "
            'or answer with {"action":"answer","answer":"..."}.'
        )
    except Exception as e:
        return f"Exec error: could not compile code: {e}"
    g = dict(globals) if isinstance(globals, dict) else {}
    local_vars: dict = {}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compiled, g, local_vars)
    except Exception as e:
        out = buf.getvalue()
        prefix = f"STDOUT (partial, before error):\n{out.rstrip()}\n\n" if out.strip() else ""
        return f"{prefix}Exec error: {e}"
    out = buf.getvalue().rstrip()
    # Names bound by exec; omit dunder keys from the summary (e.g. __builtins__ in edge cases).
    to_dump = {
        k: v
        for k, v in local_vars.items()
        if not (isinstance(k, str) and k.startswith("__"))
    }
    try:
        j = json.dumps(to_dump, default=str, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception as e:
        j = f"(error encoding locals as JSON: {e}; keys: {list(to_dump.keys())!r})"
    if not out:
        return j
    return f"STDOUT:\n{out}\n\n--- locals (JSON) ---\n{j}"


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
    if a in _KNOWN_TOOLS:
        return True
    if out.get("tool") in _KNOWN_TOOLS:
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
            if isinstance(v, str) and v in _KNOWN_TOOLS:
                out["tool"] = v
                break
        if out.get("tool") is None and isinstance(out.get("name"), str) and out["name"] in _KNOWN_TOOLS:
            out["tool"] = out["name"]

    # Infer missing action after aliases / answer fields are filled in.
    if not action or (isinstance(action, str) and action.lower() in ("null", "none", "")):
        if out.get("tool") in _KNOWN_TOOLS:
            out["action"] = "tool_call"
            action = "tool_call"
        elif out.get("answer") is not None and isinstance(out.get("answer"), str) and out["answer"].strip():
            out["action"] = "answer"
            action = "answer"
            # If we promoted content -> answer, drop content to avoid ambiguity with write_file's content param.
            if "content" in out and out.get("answer") == out.get("content"):
                out.pop("content", None)

    # {"action": "run_command", "command": "..."}  (action is the tool id)
    if action in _KNOWN_TOOLS and out.get("tool") is None:
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
SYSTEM_INSTRUCTIONS = (
    "You are a universal agent. Output format: reply with a single JSON object only—no text before or after it, "
    "no Markdown code fences, no keys or values that are not valid JSON. The object must include an \"action\" key. "
    "Minimal answer example: {\"action\":\"answer\",\"answer\":\"your user-facing reply as a string\"}. "
    "The \"answer\" string should be the complete helpful response for the user: do not fill it with meta-commentary "
    "about your search process, your uncertainty, or system instructions.\n\n"
    "Decision order (follow roughly in this sequence):\n"
    "(1) If the user needs current or recent real-world facts, or anything that changes over time (news, prices, "
    "versions, outages, sports results, elections, who holds an office or title today, rankings, product availability, "
    "or similar), you MUST call search_web before a final answer and base conclusions on tool output, not memory alone. "
    "When you are unsure whether the web is needed, prefer search_web.\n"
    "(2) If tool output already in the thread fully answers the request, use action answer. Do not repeat the same tool "
    "with the same parameters. Avoid trivially rephrased duplicate searches for the same fact unless the prior result "
    "was empty or clearly an error; if a tool returns an error or useless empty output, try a different query, URL, or tool, "
    "or answer and briefly state the limitation.\n"
    "(3) For timeless material (definitions, math, logic, stable algorithms, widely accepted historical facts) when the "
    "user is not asking for up-to-date real-world data, you may use action answer without tools.\n"
    "(4) After search_web, use fetch_page when snippets are not enough and you need full page text.\n\n"
    "Allowed tool names (exact strings only): search_web, fetch_page, run_command, use_git, write_file, read_file, list_directory, "
    "download_file, tail_file, replace_text, call_python.\n\n"
    "Tool calls use this shape: {\"action\":\"tool_call\",\"tool\":<name>,\"parameters\":{...}} with every required key "
    "for that tool present. Example: {\"action\":\"tool_call\",\"tool\":\"search_web\",\"parameters\":{\"query\":\"search terms\"}}.\n"
    "Required parameters per tool (use JSON strings, numbers, or booleans as noted):\n"
    "1. search_web — parameters.query (non-empty string, the web search terms); optional parameters.max_results "
    "(integer 1–30, how many result rows to parse; default from AGENT_SEARCH_WEB_MAX_RESULTS, else 5).\n"
    "2. fetch_page — parameters.url (string, full http/https URL to fetch).\n"
    "3. run_command — parameters.command (string, shell command to run).\n"
    "4. use_git — parameters.op (string: status|log|diff|add|commit|push|pull|branch), "
    "optional parameters.worktree (repo path), parameters.message (for commit), parameters.remote / parameters.branch (for push/pull), "
    "parameters.staged (boolean, for diff), parameters.paths (array of strings for add).\n"
    "5. write_file — parameters.path (file path string), parameters.content (string to write).\n"
    "6. read_file — parameters.path (file path string).\n"
    "7. list_directory — parameters.path (directory path string).\n"
    "8. download_file — parameters.url (source URL string), parameters.path (destination file path).\n"
    "9. tail_file — parameters.path (file path string); optional: parameters.lines (integer, default 20).\n"
    "10. replace_text — parameters.path, parameters.pattern (regex string), parameters.replacement (string); "
    "optional: parameters.replace_all (boolean, default true).\n"
    "11. call_python — parameters.code (string, syntactically valid Python ONLY). "
    "Tool output includes STDOUT from print() (if any) plus a JSON summary of assigned variables (locals); "
    "use print for human-readable trace. "
    "Never put shell/batch/cmd text, pseudo-code, or natural-language document drafts in code; "
    "those belong in write_file content or in action answer. "
    "Optional: parameters.globals (object, extra globals).\n\n"
    "Finishing: use {\"action\":\"answer\",\"answer\":\"string\","
    "\"next_action\":\"finalize\",\"rationale\":\"short note (e.g. why you are done)\"}. "
    "To request an independent second opinion before finishing, use "
    "{\"action\":\"answer\",\"answer\":\"your best draft so far\",\"next_action\":\"second_opinion\","
    "\"rationale\":\"why you want a review\"}. "
    "The program routes that using session configuration.\n"
    "If you need to write and execute code, first use write_file then use run_command. "
    "If the user asked for a document/report/essay saved to disk, after write_file you should read_file that same path "
    "and put the full document text in the final answer field (unless the user explicitly asked only for a file path). "
    "For letters, memos, or other creative writing, search_web is optional unless the user asked for sources, "
    "citations, or web research—compose with write_file or a direct answer.",
)


def _default_system_instruction_text() -> str:
    return "".join(SYSTEM_INSTRUCTIONS)


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
    """Resolve a template name to an effective system prompt string."""
    key = (name or "").strip()
    if not key:
        return None
    t = templates.get(key)
    if not isinstance(t, dict):
        return None
    kind = str(t.get("kind") or "overlay").strip().lower()
    text = t.get("text")
    path = t.get("path")
    if text is None and isinstance(path, str) and path.strip():
        p = os.path.abspath(os.path.expanduser(path.strip()))
        try:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            text = None
    if not isinstance(text, str) or not text.strip():
        return None
    body = text.strip()
    if kind == "full":
        return body
    # overlay (default)
    return _default_system_instruction_text() + "\n\n" + body


def _effective_system_instruction_text(override: Optional[str]) -> str:
    """Session override replaces the built-in system prompt when non-empty."""
    if override is None:
        return _default_system_instruction_text()
    s = str(override).strip()
    if not s:
        return _default_system_instruction_text()
    return s


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
    lim = max(1, _scalar_to_int(os.environ.get("AGENT_ROUTER_TRANSCRIPT_MAX_MESSAGES"), 80))
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
    """Runner preamble for system instructions (CLI and interactive)."""
    pp = primary_profile or default_primary_llm_profile()
    bits = []
    if pp.backend == "hosted":
        bits.append(
            f"Runner: primary LLM is hosted OpenAI-compatible API: model {pp.model!r}, "
            f"base {pp.base_url!r}, API key from env {pp.api_key_env}."
        )
    else:
        bits.append(f"Runner: primary LLM is local Ollama ({_ollama_model()!r}).")
    if second_opinion or _hosted_review_ready(cloud, reviewer_hosted_profile):
        bits.append(
            "Runner: you may use next_action second_opinion in this session (see system instructions)."
        )
    tp = _tool_policy_runner_text(enabled_tools)
    if tp:
        bits.append(tp)
    return " ".join(bits) if bits else ""


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
    si = _effective_system_instruction_text(system_instruction_override)
    suff = (skill_suffix or "").strip()
    if suff:
        si = si + "\n\n--- Active skill ---\n" + suff
    block = (
        f"{si}\n\n"
        f"Today's date (system clock): {today_str}\n\n"
        f"User request:\n{user_query}\n\n"
        "Respond with JSON only. No other text."
    )
    ri = _runner_instruction_bits(
        second_opinion,
        cloud,
        primary_profile=primary_profile,
        reviewer_ollama_model=reviewer_ollama_model,
        reviewer_hosted_profile=reviewer_hosted_profile,
        enabled_tools=enabled_tools,
    )
    if ri:
        block += "\n\n" + ri
    return block


def _interactive_repl(
    *,
    verbose: int,
    second_opinion_enabled: bool,
    cloud_ai_enabled: bool,
    save_context_path: Optional[str],
    enabled_tools: Optional[AbstractSet[str]] = None,
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
        set(enabled_tools) if enabled_tools is not None else set(_KNOWN_TOOLS)
    )
    prim_line = (
        _describe_llm_profile_short(primary_profile)
        if primary_profile.backend == "hosted"
        else f"ollama ({_ollama_model()!r})"
    )
    if (
        reviewer_hosted_profile is not None
        and reviewer_hosted_profile.backend == "hosted"
    ):
        rev_line = _describe_llm_profile_short(reviewer_hosted_profile)
    else:
        rev_line = f"ollama ({(reviewer_ollama_model or _ollama_second_opinion_model())!r})"
    # Keep startup quiet at verbose=0; the detailed banner is still available at verbose>=1.
    if verbose >= 1:
        print(
            "Interactive mode. Commands: /help  /settings …  /use-skills  /reuse-skill  /quit\n"
            + (
                f"(Loaded defaults from {_agent_prefs_path()})\n"
                if prefs_loaded
                else ""
            )
            + f"Primary LLM: {prim_line}\n"
            f"second_opinion: {'on' if second_opinion_on else 'off'}\n"
            f"Second-opinion default reviewer: {rev_line}\n"
            + (
                "Tools: all enabled."
                if len(enabled_tools) == len(_KNOWN_TOOLS)
                else "Tools disabled: "
                + ", ".join(sorted(_KNOWN_TOOLS - enabled_tools))
            )
            + (
                f"\nauto-save context: {session_save_path!r}"
                if session_save_path
                else "\nauto-save context: (not set)"
            )
            + f"\nverbose: {verbose} (0=off, 1=tool logs, 2=tools + streamed model JSON on local Ollama)"
            + (
                f"\nSystem prompt: custom ({len(_effective_system_instruction_text(session_system_prompt))} chars)"
                + (
                    f" from {session_system_prompt_path!r}"
                    if session_system_prompt_path
                    else (
                        f" (template: {session_prompt_template!r})"
                        if session_prompt_template
                        else " (inline or prefs)"
                    )
                )
                if session_system_prompt is not None
                else "\nSystem prompt: built-in default"
            )
        )
        if _interactive_repl_install_readline():
            print(
                "Line editing: arrow keys, Home/End, delete words, etc.; "
                f"↑/↓ history (file: {_repl_history_path()!r}). "
                "Bindings follow ~/.inputrc if present."
            )
        if sys.stdin.isatty() and _repl_env_flag_true("AGENT_REPL_BUFFERED_LINE", "0"):
            print(
                f"AGENT_REPL_BUFFERED_LINE=1: long single-line pastes via stdin buffer (up to "
                f"{_repl_buffered_line_max_bytes()} bytes; no readline editing while typing). "
                "Override size with AGENT_REPL_INPUT_MAX_BYTES."
            )
    else:
        _interactive_repl_install_readline()
        print("Interactive mode. Type /help for commands.")
    messages: list = []
    last_reuse_skill_id: Optional[str] = None

    def repl_run_with_selected_skill(
        req: str, sid: str, *, from_reuse: bool, selection_rationale: str
    ) -> None:
        nonlocal last_reuse_skill_id
        last_reuse_skill_id = sid
        if from_reuse:
            _agent_progress("/reuse-skill: using stored skill; starting…")
        else:
            _agent_progress("/use-skills: skill selected; starting…")
        et_turn = _effective_enabled_tools_for_skill(
            frozenset(enabled_tools), skills_m, sid
        )
        rec = skills_m.get(sid) or {}
        skill_prompt = (rec.get("prompt") or "").strip() if isinstance(rec, dict) else ""
        if from_reuse:
            print(
                f"/reuse-skill: using skill {sid!r} (model skill selection skipped). "
                f"{selection_rationale}".strip()
            )
        else:
            print(f"/use-skills selected {sid!r}. {selection_rationale}".strip())
        if verbose >= 1:
            _print_skill_usage_verbose(
                verbose,
                source="reuse-skill" if from_reuse else "use-skills",
                skill_id=sid,
                base_tools=enabled_tools,
                effective_tools=et_turn,
                detail=(
                    "reuse: same skill id as last /use-skills or /reuse-skill"
                    if from_reuse
                    else f"model skill_id (not trigger): rationale={selection_rationale!r}"
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
            print("Context cleared (including stored skill for /reuse-skill).")
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
        if low.startswith("/use-skills"):
            # Model picks a skill, then we run the same pipeline as /reuse-skill.
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
            repl_run_with_selected_skill(
                req, sid, from_reuse=False, selection_rationale=why
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
                from_reuse=True,
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
                print(
                    "Usage:\n"
                    "  /settings model <ollama-model-name>\n"
                    "  /settings primary llm ollama\n"
                    "  /settings primary llm hosted <base_url> <model> [api_key_env]\n"
                    "  /settings second_opinion llm ollama [ollama_model]\n"
                    "  /settings second_opinion llm hosted <base_url> <model> [api_key_env]\n"
                    "  /settings enable second_opinion\n"
                    "  /settings disable second_opinion\n"
                    "  /settings verbose 0|1|2|on|off\n"
                    "  /settings enable <tool or phrase>   /settings disable <tool or phrase>\n"
                    "  /settings tools          List tools, ids, and on/off for this session\n"
                    "  /settings system_prompt …  show | reset | file <path> | save <path> | <one-line text>\n"
                    "  /settings prompt_template …  list | use <name> | default <name> | show | set <name> <text> | delete <name>\n"
                    "  /settings context …  show | on|off | tokens <n> | trigger <0..1> | target <0..1> | keep_tail <n>\n"
                    "  /settings thinking …    show | on|off | level low|medium|high\n"
                    "  /settings ollama …       show|keys|set|unset  (OLLAMA_*; saved in ~/.agent.json)\n"
                    "  /settings openai …        show|keys|set|unset  (OPENAI_*)\n"
                    "  /settings agent …         show|keys|set|unset  (AGENT_*; shell env on start overrides the file)\n"
                    "  /settings save            Write current settings to ~/.agent.json"
                )
                continue
            key = toks[1].lower().replace("-", "_")
            if key in ("ollama", "openai", "agent"):
                if len(toks) < 3:
                    print(
                        f"Usage: /settings {key} show | keys | set <name> <value> | unset <name>\n"
                        "  Short names (e.g. HOST) or full env names (e.g. OLLAMA_HOST) are accepted. "
                        "After changing, use /settings save. Shell variables set before launch still override the file for that run."
                    )
                    print(_file_env_key_help_lines(key))
                    continue
                sub = toks[2].lower()
                if sub in ("show", "list"):
                    try:
                        print(_format_file_env_group_show(key))
                    except (ValueError, OSError) as e:
                        print(f"/settings {key} show: {e}")
                    continue
                if sub in ("keys", "key", "help"):
                    try:
                        print(_file_env_key_help_lines(key))
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
                        full = _file_env_set_process(key, raw_k, value)
                    except ValueError as e:
                        print(f"/settings {key} set: {e}")
                        continue
                    print(
                        f"{full} = {value!r} in this process. Use /settings save to write ~/.agent.json."
                    )
                    continue
                if sub in ("unset", "delete", "clear"):
                    if len(toks) < 4:
                        print(f"Usage: /settings {key} unset <name>")
                        continue
                    try:
                        full = _file_env_unset_process(key, toks[3])
                    except ValueError as e:
                        print(f"/settings {key} unset: {e}")
                        continue
                    print(
                        f"{full} removed from the process. Use /settings save; on next start missing keys use built-in or file defaults."
                    )
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
                if len(toks) != 2:
                    print("Usage: /settings tools")
                    continue
                print(_format_settings_tools_list(enabled_tools))
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
                        reviewer_hosted_profile=reviewer_hosted_profile,
                        reviewer_ollama_model=reviewer_ollama_model,
                        session_save_path=session_save_path,
                        system_prompt_override=session_system_prompt,
                        system_prompt_path_override=session_system_prompt_path,
                        prompt_templates=templates,
                        prompt_template_default=template_default,
                        prompt_templates_dir=session_pt_dir,
                        skills_dir=session_skills_dir,
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
                os.environ["OLLAMA_MODEL"] = name
                print(f"OLLAMA_MODEL set to {name!r} (this process only).")
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
                    os.environ["AGENT_STREAM_THINKING"] = "1"
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
                    os.environ["AGENT_STREAM_THINKING"] = "0"
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
                    os.environ["AGENT_THINKING"] = "1"
                    os.environ["AGENT_STREAM_THINKING"] = "1"
                    print(
                        "thinking enabled for this session (and stream_thinking enabled). "
                        "Use /settings save to persist."
                    )
                    continue
                if sub in ("off", "disable", "disabled", "false"):
                    os.environ["AGENT_THINKING"] = "0"
                    os.environ["AGENT_THINKING_LEVEL"] = ""
                    os.environ["AGENT_STREAM_THINKING"] = "0"
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
                    os.environ["AGENT_THINKING_LEVEL"] = lvl
                    os.environ["AGENT_THINKING"] = "1"
                    os.environ["AGENT_STREAM_THINKING"] = "1"
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
                    print("Primary LLM: local Ollama (uses OLLAMA_MODEL from the environment).")
                elif sub == "hosted":
                    if len(toks) < 6:
                        print(
                            "Usage: /settings primary llm hosted <base_url> <model> [api_key_env]"
                        )
                        continue
                    bu, mod = toks[4], toks[5]
                    keyenv = toks[6] if len(toks) > 6 else "OPENAI_API_KEY"
                    if not bu.startswith(("http://", "https://")):
                        print("base_url must start with http:// or https://")
                        continue
                    primary_profile = LlmProfile(
                        backend="hosted",
                        base_url=bu,
                        model=mod,
                        api_key_env=keyenv,
                    )
                    if not _read_api_key(keyenv):
                        print(
                            f"Note: {keyenv} is not set; hosted primary calls will fail until it is."
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
                            "Usage: /settings second_opinion llm hosted <base_url> <model> [api_key_env]"
                        )
                        continue
                    bu, mod = toks[4], toks[5]
                    keyenv = toks[6] if len(toks) > 6 else "OPENAI_API_KEY"
                    if not bu.startswith(("http://", "https://")):
                        print("base_url must start with http:// or https://")
                        continue
                    reviewer_hosted_profile = LlmProfile(
                        backend="hosted",
                        base_url=bu,
                        model=mod,
                        api_key_env=keyenv,
                    )
                    reviewer_ollama_model = None
                    if not _read_api_key(keyenv):
                        print(
                            f"Note: {keyenv} is not set; hosted second opinion will fail until it is."
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
                "  /models                  List local Ollama models (api/tags)\n"
                "  /usage                   Last local Ollama prompt/completion token counts (from /api/chat)\n"
                "  /use-skills <request>    Ask the model to pick a skill, then run it (multi-step if the skill supports it)\n"
                "  /reuse-skill <request>     Follow-up using the same skill as the last /use-skills or /reuse-skill (no re-selection)\n"
                "  After run_command/call_python errors: auto-retry once with model-proposed params in a TTY; "
                "non-TTY uses AGENT_AUTO_CONFIRM_TOOL_RETRY=1 to enable recovery\n"
                "  /settings model <name>   Set OLLAMA_MODEL (local Ollama primary)\n"
                "  /settings primary llm ollama|hosted …   Primary (hosted = OpenAI-compatible /v1 URL + model + optional API key env name)\n"
                "  /settings second_opinion llm ollama|hosted …   Second-opinion reviewer (independent from primary)\n"
                "  /settings enable second_opinion\n"
                "  /settings disable second_opinion\n"
                "  /settings verbose 0|1|2|on|off\n"
                "  /settings enable|disable <tool or phrase>  (e.g. web search, shell)\n"
                "  /settings tools            List tools, internal ids, on/off\n"
                "  /settings system_prompt …  show | reset | file <path> | save <path> | <one-line text>\n"
                "  /settings prompt_template …  list | use <name> | default <name> | show | set <name> <text> | delete <name>\n"
                "  /settings context …       show | on|off | tokens <n> | trigger <0..1> | target <0..1> | keep_tail <n>\n"
                "  /settings thinking …      show | on|off | level low|medium|high\n"
                "  /settings ollama|openai|agent …  show|keys|set|unset  (store OLLAMA/OPENAI/AGENT in ~/.agent.json; save to persist)\n"
                "  /settings save             Persist settings to ~/.agent.json (can include prompt_templates_dir, skills_dir)\n"
                "  /load_context <file>     Replace session messages from JSON\n"
                "  /save_context <file>     Write session JSON; set auto-save path\n"
                "  /help                    Show this help\n"
                "  /help environment         How env vars interact with /settings ollama|openai|agent and ~/.agent.json\n"
            )
            continue
        if low in ("/help environment", "/help env"):
            print(
                "Ollama / OpenAI / agent options are stored in ~/.agent.json (see objects ollama, openai, agent) and edited with:\n"
                "  /settings ollama show|keys|set|unset,  /settings openai …,  /settings agent …,  then  /settings save  to write the file.\n"
                "Each key (short: HOST, or full: OLLAMA_HOST) is applied to the process; save snapshots current environment values for known keys.\n"
                "Precedence: if a full variable (e.g. OLLAMA_HOST) is set in the shell when the program starts, that value is kept and the file is not applied for that name.\n"
                "\n"
                "The same names also work as plain environment variables (e.g. for one-off  OLLAMA_HOST=…  python3 agent.py  ):\n"
                "  OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_SECOND_OPINION_MODEL, OLLAMA_DEBUG, OLLAMA_TOOL_OUTPUT_MAX, OLLAMA_SEARCH_ENRICH\n"
                "  OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_CLOUD_MODEL, OPENAI_MODEL\n"
                "  AGENT_QUIET, AGENT_PROGRESS, AGENT_PROMPT_TEMPLATES_DIR, AGENT_SKILLS_DIR, AGENT_REPL_*,\n"
                "  AGENT_AUTO_CONFIRM_TOOL_RETRY, AGENT_SEARCH_WEB_MAX_RESULTS, AGENT_THINKING, AGENT_THINKING_LEVEL, AGENT_STREAM_THINKING (or legacy AGENT_SHOW_THINKING),\n"
                "  AGENT_CONTEXT_*, AGENT_DISABLE_CONTEXT_MANAGER, AGENT_ROUTER_TRANSCRIPT_MAX_MESSAGES\n"
            )
            continue
        if s.startswith("/"):
            print(f"Unknown command {s.split()[0]!r}. Try /help.")
            continue

        try:
            today = datetime.date.today()
            today_str = today.strftime("%Y-%m-%d (%A)")
            user_query = s
            deliverable_wanted = _user_wants_written_deliverable(user_query)
            sid0, tr0 = _match_skill_detail(user_query, skills_m)
            et_turn = _effective_enabled_tools_for_skill(
                frozenset(enabled_tools), skills_m, sid0
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
                        and _read_api_key(reviewer_hosted_profile.api_key_env)
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
            if print_answer:
                print(ans_out)
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
        elif action == "tool_call" or action in _KNOWN_TOOLS:
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
                if tool in _KNOWN_TOOLS and tool not in et:
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
    verbose = _coerce_verbose_level(st.get("verbose", 0))
    query_parts = []
    second_opinion_enabled = bool(st["second_opinion_enabled"])
    cloud_ai_enabled = bool(st["cloud_ai_enabled"])
    load_context_path: Optional[str] = None
    save_context_path: Optional[str] = st["save_context_path"]
    enabled_tools = set(st["enabled_tools"])
    primary_profile = st["primary_profile"]
    reviewer_hosted_profile: Optional[LlmProfile] = st["reviewer_hosted_profile"]
    reviewer_ollama_model: Optional[str] = st["reviewer_ollama_model"]
    prompt_templates = st.get("prompt_templates") if isinstance(st.get("prompt_templates"), dict) else _default_prompt_templates()
    prompt_template_default = (st.get("prompt_template_default") or "coding").strip()
    prompt_template_selected: Optional[str] = None
    i = 0
    while i < len(argv):
        a = argv[i]
        fa = _strip_leading_dashes_flag(a)
        if (a or "").startswith("-") and fa in ("help", "h", "?"):
            _print_cli_help()
            return
        if (a or "").startswith("-") and (fa == "model" or fa.startswith("model=")):
            mname: str
            if fa == "model":
                if i + 1 >= len(argv):
                    print("Error: --model requires a model name.", file=sys.stderr)
                    return
                mname = argv[i + 1].strip()
                if not mname:
                    print("Error: --model name must be non-empty.", file=sys.stderr)
                    return
                i += 2
            else:
                _eq = a.split("=", 1)
                if len(_eq) < 2 or not _eq[1].strip():
                    print(
                        "Error: --model=<name> requires a non-empty name.",
                        file=sys.stderr,
                    )
                    return
                mname = _eq[1].strip()
                i += 1
            primary_profile = _apply_cli_primary_model(mname, primary_profile)
            continue
        if fa in ("enable-tool",):
            if i + 1 >= len(argv):
                print("Error: -enable_tool requires a tool name.")
                return
            t = _normalize_tool_name(argv[i + 1])
            if not t:
                print("Error: " + _format_unknown_tool_hint(argv[i + 1]))
                return
            enabled_tools.add(t)
            i += 2
            continue
        if fa in ("disable-tool",):
            if i + 1 >= len(argv):
                print("Error: -disable_tool requires a tool name.")
                return
            t = _normalize_tool_name(argv[i + 1])
            if not t:
                print("Error: " + _format_unknown_tool_hint(argv[i + 1]))
                return
            enabled_tools.discard(t)
            i += 2
            continue
        if fa in ("list-tools",):
            print(_format_settings_tools_list(enabled_tools))
            return
        if _strip_leading_dashes_flag(a) == "verbose":
            if i + 1 < len(argv) and argv[i + 1] in ("0", "1", "2"):
                verbose = int(argv[i + 1])
                i += 2
            else:
                verbose = 2
                i += 1
            continue
        elif a in ("--second-opinion", "--second_opinion"):
            second_opinion_enabled = True
            i += 1
        elif a in ("--cloud-ai", "--cloud_ai"):
            cloud_ai_enabled = True
            i += 1
        elif a in ("--load-context", "--load_context"):
            if i + 1 >= len(argv):
                print("Error: --load_context requires a file path.")
                return
            load_context_path = argv[i + 1]
            i += 2
        elif a in ("--save-context", "--save_context"):
            if i + 1 >= len(argv):
                print("Error: --save_context requires a file path.")
                return
            save_context_path = argv[i + 1]
            i += 2
        elif a in ("--prompt-template", "--prompt_template"):
            if i + 1 >= len(argv):
                print("Error: --prompt-template requires a template name.")
                return
            prompt_template_selected = argv[i + 1].strip()
            i += 2
        else:
            query_parts.append(a)
            i += 1
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
    et_oneshot = _effective_enabled_tools_for_skill(
        frozenset(enabled_tools), skills_map_cli, skill_id_cli
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
