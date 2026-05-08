"""Primary / reviewer LLM endpoint description."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from agentlib.coercion import scalar_to_str
from agentlib.llm.request_options import normalize_request_options_pref


@dataclass
class LlmProfile:
    """Primary or reviewer endpoint: local Ollama or an OpenAI-compatible HTTPS API."""

    backend: str  # "ollama" | "hosted"
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    #: Backend-specific knobs merged into outgoing chat requests (Ollama ``options`` or hosted body).
    request_options: Dict[str, Any] = field(default_factory=dict)


def default_primary_llm_profile() -> LlmProfile:
    return LlmProfile(backend="ollama")


def effective_ollama_model_from_profile(primary_profile: Optional[LlmProfile], default_model: str) -> str:
    """Use ``profile.model`` for local Ollama when set; otherwise ``default_model`` (e.g. from prefs)."""
    p = primary_profile
    if p is not None and getattr(p, "backend", "") == "ollama":
        m = (getattr(p, "model", "") or "").strip()
        if m:
            return m
    return (default_model or "").strip()


def llm_profile_to_pref(profile: LlmProfile) -> dict:
    ro = normalize_request_options_pref(getattr(profile, "request_options", None) or {})
    if profile.backend != "hosted":
        d: dict = {"backend": "ollama"}
        if ro:
            d["request_options"] = ro
        return d
    d = {
        "backend": "hosted",
        "base_url": profile.base_url,
        "model": profile.model,
    }
    if (profile.api_key or "").strip():
        d["api_key"] = profile.api_key.strip()
    if ro:
        d["request_options"] = ro
    return d


def llm_profile_from_pref(obj: object) -> Optional[LlmProfile]:
    if not isinstance(obj, dict):
        return None
    bk = scalar_to_str(obj.get("backend"), "").strip().lower()
    if bk == "ollama":
        prof = LlmProfile(backend="ollama")
        ro = obj.get("request_options")
        prof.request_options = normalize_request_options_pref(ro)
        return prof
    if bk != "hosted":
        return None
    bu = scalar_to_str(obj.get("base_url"), "").strip().rstrip("/")
    mod = scalar_to_str(obj.get("model"), "").strip()
    if not bu.startswith(("http://", "https://")) or not mod:
        return None
    prof = LlmProfile(backend="hosted", base_url=bu, model=mod)
    ak = obj.get("api_key")
    if isinstance(ak, str) and ak.strip():
        prof.api_key = ak.strip()
    ro = obj.get("request_options")
    prof.request_options = normalize_request_options_pref(ro)
    return prof


def preserved_request_options(profile: Optional[LlmProfile]) -> dict[str, Any]:
    """Deep-enough copy of ``request_options`` for carrying across primary/resolver swaps."""
    p = getattr(profile, "request_options", None)
    raw = normalize_request_options_pref(p or {})
    return dict(raw)
