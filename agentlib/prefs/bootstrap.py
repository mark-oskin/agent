"""Build and parse ~/.agent.json payloads (session defaults + save bundle)."""

from __future__ import annotations

import os
from typing import AbstractSet, Callable, Optional

from agentlib.coercion import coerce_verbose_level, scalar_to_int
from agentlib.llm.profile import (
    LlmProfile,
    default_primary_llm_profile,
    llm_profile_from_pref,
    llm_profile_to_pref,
)
from agentlib.prefs import AGENT_PREFS_VERSION
from agentlib.settings import AgentSettings


_DEFAULT_CONTEXT_MANAGER: dict = {
    "enabled": True,
    "tokens": 0,
    "trigger_frac": 0.75,
    "target_frac": 0.55,
    "keep_tail_messages": 12,
}


def _normalize_context_manager(cm: Optional[dict]) -> Optional[dict]:
    if cm is None:
        return None
    if not isinstance(cm, dict):
        return None
    merged = dict(_DEFAULT_CONTEXT_MANAGER)
    for k in ("enabled", "tokens", "trigger_frac", "target_frac", "keep_tail_messages"):
        if k in cm:
            merged[k] = cm[k]
    return merged


def build_agent_prefs_payload(
    *,
    settings: AgentSettings,
    primary_profile: LlmProfile,
    second_opinion_on: bool,
    cloud_ai_enabled: bool,
    enabled_tools: AbstractSet[str],
    core_tools: AbstractSet[str],
    plugin_toolsets: dict,
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
    enabled_toolsets: Optional[AbstractSet[str]] = None,
    full_snapshot: bool = False,
) -> dict:
    payload: dict = {"version": AGENT_PREFS_VERSION}

    # Minimal-by-default persistence: omit values that match the built-in defaults.
    if second_opinion_on:
        payload["second_opinion_enabled"] = True
    if cloud_ai_enabled:
        payload["cloud_ai_enabled"] = True
    vv = coerce_verbose_level(verbose_level)
    if vv:
        payload["verbose"] = vv

    default_primary = llm_profile_to_pref(default_primary_llm_profile())
    primary_pref = llm_profile_to_pref(primary_profile)
    if full_snapshot or primary_pref != default_primary:
        payload["primary_llm"] = primary_pref

    if len(enabled_tools) < len(core_tools):
        payload["enabled_tools"] = sorted(enabled_tools)

    payload.update(settings.as_groups_dict() if full_snapshot else settings.as_groups_delta_dict())
    if reviewer_hosted_profile is not None and reviewer_hosted_profile.backend == "hosted":
        payload["second_opinion_reviewer"] = llm_profile_to_pref(reviewer_hosted_profile)
    else:
        if reviewer_ollama_model and str(reviewer_ollama_model).strip():
            payload["second_opinion_reviewer"] = {"backend": "ollama", "ollama_model": str(reviewer_ollama_model).strip()}
    if session_save_path and str(session_save_path).strip():
        payload["save_context_path"] = str(session_save_path).strip()
    spp = (system_prompt_path_override or "").strip()
    if spp:
        payload["system_prompt_path"] = os.path.abspath(os.path.expanduser(spp))
    elif system_prompt_override is not None and str(system_prompt_override).strip():
        payload["system_prompt"] = str(system_prompt_override)
    if prompt_templates is not None:
        payload["prompt_templates"] = prompt_templates
    if prompt_template_default is not None:
        payload["prompt_template_default"] = str(prompt_template_default).strip() or None
    # NOTE: prefer persisting these via settings groups (agent.prompt_templates_dir/skills_dir/tools_dir)
    # rather than top-level snapshot fields which tend to freeze defaults. We still accept these on load
    # for backward compatibility.
    cm_norm = _normalize_context_manager(context_manager)
    if cm_norm is not None and (full_snapshot or cm_norm != _DEFAULT_CONTEXT_MANAGER):
        payload["context_manager"] = cm_norm
    ets = {str(x).strip().lower() for x in (enabled_toolsets or set()) if str(x).strip()}
    ets = {x for x in ets if x in plugin_toolsets}
    if ets:
        payload["enabled_toolsets"] = sorted(ets)
    return payload


def session_defaults_from_prefs(
    prefs: Optional[dict],
    *,
    migrate_prefs: Callable[[dict], None],
    settings: AgentSettings,
    core_tools: AbstractSet[str],
    plugin_toolsets: dict,
    normalize_tool_name: Callable[[str], Optional[str]],
    merge_prompt_templates: Callable[[Optional[dict]], dict],
    load_skills_from_dir: Callable[[str], dict],
    resolved_prompt_templates_dir: Callable[[Optional[dict]], str],
    resolved_skills_dir: Callable[[Optional[dict]], str],
    resolved_tools_dir: Callable[[Optional[dict]], str],
    default_prompt_templates_dir: Callable[[], str],
    default_skills_dir: Callable[[], str],
    load_plugin_toolsets: Callable[..., None],
    register_tool_aliases: Callable[[], None],
) -> dict:
    if isinstance(prefs, dict):
        migrate_prefs(prefs)
    try:
        load_plugin_toolsets(resolved_tools_dir(prefs))
        register_tool_aliases()
    except Exception:
        pass
    _pt = default_prompt_templates_dir()
    _sk = default_skills_dir()
    out = {
        "enabled_tools": set(core_tools),
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
        "tools_dir": resolved_tools_dir(prefs),
        "prompt_templates": merge_prompt_templates(None),
        "skills": load_skills_from_dir(resolved_skills_dir(None)),
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
    ver = scalar_to_int(prefs.get("version"), AGENT_PREFS_VERSION)
    if ver > AGENT_PREFS_VERSION:
        return out
    if isinstance(prefs.get("second_opinion_enabled"), bool):
        out["second_opinion_enabled"] = prefs["second_opinion_enabled"]
    if isinstance(prefs.get("cloud_ai_enabled"), bool):
        out["cloud_ai_enabled"] = prefs["cloud_ai_enabled"]
    if "verbose" in prefs:
        out["verbose"] = coerce_verbose_level(prefs.get("verbose"))
    pl = prefs.get("primary_llm")
    if isinstance(pl, dict):
        pp = llm_profile_from_pref(pl)
        if pp:
            out["primary_profile"] = pp
    raw_et = prefs.get("enabled_tools")
    if isinstance(raw_et, list):
        et = set()
        for t in raw_et:
            tn = normalize_tool_name(str(t))
            if tn:
                et.add(tn)
        if et:
            out["enabled_tools"] = et
    raw_ts = prefs.get("enabled_toolsets")
    if isinstance(raw_ts, list):
        ts: set[str] = set()
        for one in raw_ts:
            nm = str(one).strip().lower()
            if nm and nm in plugin_toolsets:
                ts.add(nm)
        out["enabled_toolsets"] = ts
    rev = prefs.get("second_opinion_reviewer")
    if isinstance(rev, dict):
        rb = str(rev.get("backend") or "").strip().lower()
        if rb == "hosted":
            hp = llm_profile_from_pref(rev)
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
    out["prompt_templates_dir"] = resolved_prompt_templates_dir(prefs)
    out["skills_dir"] = resolved_skills_dir(prefs)
    out["prompt_templates"] = merge_prompt_templates(prefs)
    out["skills"] = load_skills_from_dir(out["skills_dir"])
    out["tools_dir"] = resolved_tools_dir(prefs)
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
