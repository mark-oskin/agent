"""Primary / reviewer LLM endpoint description."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agentlib.coercion import scalar_to_str


@dataclass
class LlmProfile:
    """Primary or reviewer endpoint: local Ollama or an OpenAI-compatible HTTPS API."""

    backend: str  # "ollama" | "hosted"
    base_url: str = ""
    model: str = ""
    api_key: str = ""


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


def llm_profile_from_pref(obj: object) -> Optional[LlmProfile]:
    if not isinstance(obj, dict):
        return None
    bk = scalar_to_str(obj.get("backend"), "").strip().lower()
    if bk == "ollama":
        return LlmProfile(backend="ollama")
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
    return prof
