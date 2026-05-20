import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


DEFAULT_SETTINGS: dict = {
    "ollama": {
        "host": "http://localhost:11434",
        "model": "qwen3.6:latest",
        "second_opinion_model": "llama3.2:latest",
        "debug": False,
        "tool_output_max": 100000,
        "search_enrich": True,
    },
    "openai": {
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "cloud_model": "gpt-4o-mini",
    },
    "agent": {
        "quiet": False,
        "progress": True,
        "prompt_templates_dir": "",
        "skills_dir": "",
        "tools_dir": "",
        "repl_history": "",
        "repl_input_max_bytes": 0,  # 0 means use built-in default
        "repl_buffered_line": False,
        "thinking": False,
        "thinking_level": "",
        "stream_thinking": False,
        # Stream assistant answer text incrementally during LLM generation (REPL/TUI).
        "stream_assistant": True,
        "search_web_max_results": 5,
        "search_web_backend": "ddg",
        "brave_search_api_key": "",
        "default_browser_engine": "chromium",
        "debug_search_web": False,
        "searxng_url": "https://searx.party",
        "auto_confirm_tool_retry": False,
        "context_tokens": 0,
        "hosted_context_tokens": 0,
        "ollama_context_tokens": 0,
        "disable_context_manager": False,
        "context_trigger_frac": 0.75,
        "context_target_frac": 0.55,
        "context_keep_tail_messages": 12,
        "router_transcript_max_messages": 80,
        # Agent loop budgets (per user turn): model iterations + web-verification tool caps.
        "max_agent_steps": 30,
        "max_agent_steps_web": 15,
        "max_tool_calls_web": 15,
        "max_fetch_page_web": 15,
        # When True (default), PDF responses from fetch_page are converted to extracted text for the LLM.
        "fetch_page_pdf_to_text": True,
        # When True, normal REPL turns auto-apply a skill whose trigger substring matches the user message.
        "skill_auto_match_triggers": False,
        # Model Context Protocol: optional external tools via stdio or HTTP JSON-RPC.
        "mcp_enabled": False,
        "mcp_servers": [],
    },
    # Optional overrides for REPL extensions (``/set extensions <id> …``). Each key is an extension id;
    # defaults for an extension belong in that extension's module, not here.
    "extensions": {},
}


def _deepcopy_jsonable(obj: Any) -> Any:
    # Safe, small, dependency-free deep copy for JSON-like shapes.
    return json.loads(json.dumps(obj))


@dataclass
class AgentSettings:
    """
    Process-local settings (defaults -> prefs json -> CLI overrides).

    Storage format is JSON-compatible and intentionally matches ~/.agent.json groups:
      - ollama.*
      - openai.*
      - agent.*
    """

    _data: Dict[str, Dict[str, Any]] = field(default_factory=lambda: _deepcopy_jsonable(DEFAULT_SETTINGS))

    @classmethod
    def defaults(cls) -> "AgentSettings":
        return cls(_data=_deepcopy_jsonable(DEFAULT_SETTINGS))

    def as_groups_dict(self) -> dict:
        """Return a JSON-serializable dict containing only settings groups."""
        out: dict = {}
        for grp in ("ollama", "openai", "agent", "extensions"):
            cur = self._data.get(grp)
            out[grp] = dict(cur) if isinstance(cur, dict) else {}
        return out

    def as_groups_delta_dict(self) -> dict:
        """
        Return only settings that differ from built-in defaults.

        This is used for minimal prefs persistence so new defaults can ship without
        being frozen into ~/.agent.json.
        """
        out: dict = {}
        for grp in ("ollama", "openai", "agent"):
            cur = self._data.get(grp) if isinstance(self._data, dict) else None
            cur = cur if isinstance(cur, dict) else {}
            defs = DEFAULT_SETTINGS.get(grp) if isinstance(DEFAULT_SETTINGS.get(grp), dict) else {}
            delta: dict = {}
            for k, v in cur.items():
                if k in defs and defs.get(k) == v:
                    continue
                delta[k] = v
            if delta:
                out[grp] = delta
        ext_out = self._extensions_delta_vs_defaults()
        if ext_out:
            out["extensions"] = ext_out
        return out

    def _extensions_defaults_root(self) -> dict:
        d = DEFAULT_SETTINGS.get("extensions")
        return d if isinstance(d, dict) else {}

    def _extensions_delta_vs_defaults(self) -> dict:
        root_defs = self._extensions_defaults_root()
        cur_root = self._data.get("extensions")
        if not isinstance(cur_root, dict):
            return {}
        ext_out: dict = {}
        for ext_id, cur_sub in cur_root.items():
            if not isinstance(cur_sub, dict):
                continue
            def_sub = root_defs.get(ext_id)
            if not isinstance(def_sub, dict):
                def_sub = {}
            sub_delta: dict = {}
            for k, v in cur_sub.items():
                if def_sub.get(k) != v:
                    sub_delta[k] = v
            if sub_delta:
                ext_out[str(ext_id)] = sub_delta
        return ext_out

    def extensions_merge_from_prefs(self, blob: dict) -> None:
        """Merge prefs ``extensions`` into ``_data`` (shallow per extension id)."""
        if not isinstance(blob, dict):
            return
        root = self._data.get("extensions")
        if not isinstance(root, dict):
            root = {}
        else:
            root = json.loads(json.dumps(root))
        defs_root = self._extensions_defaults_root()
        for eid, def_sub in defs_root.items():
            if eid not in root and isinstance(def_sub, dict):
                root[eid] = dict(def_sub)
        for eid, sub in blob.items():
            if not isinstance(sub, dict):
                continue
            eid_s = str(eid).strip()
            if not eid_s:
                continue
            base = root.get(eid_s)
            if not isinstance(base, dict):
                d0 = defs_root.get(eid_s)
                base = dict(d0) if isinstance(d0, dict) else {}
            else:
                base = dict(base)
            base.update(sub)
            root[eid_s] = base
        self._data["extensions"] = root

    def extensions_show_all(self) -> str:
        cur = self._data.get("extensions")
        if not isinstance(cur, dict):
            cur = {}
        return json.dumps(cur, indent=2, ensure_ascii=False, sort_keys=True)

    def extensions_show_id(self, ext_id: str) -> str:
        cur = self._data.get("extensions")
        if not isinstance(cur, dict):
            cur = {}
        sub = cur.get(ext_id)
        if not isinstance(sub, dict):
            sub = {}
        return json.dumps(sub, indent=2, ensure_ascii=False, sort_keys=True)

    def _coerce_extension_value(self, raw_value: str, ext_id: str, key: str) -> Any:
        defs_sub = self._extensions_defaults_root().get(ext_id)
        text = (raw_value or "").strip()
        if isinstance(defs_sub, dict) and key in defs_sub:
            dv = defs_sub[key]
            if isinstance(dv, bool):
                return text.lower() in ("1", "true", "yes", "y", "on")
            if isinstance(dv, int) and not isinstance(dv, bool):
                return int(float(text)) if text else 0
            if isinstance(dv, float):
                return float(text) if text else 0.0
            return text
        if text.lower() in ("1", "true", "yes", "y", "on"):
            return True
        if text.lower() in ("0", "false", "no", "n", "off"):
            return False
        try:
            if text and all(c in "-0123456789" for c in text.lstrip("-")):
                return int(text, 10)
        except Exception:
            pass
        try:
            return float(text)
        except Exception:
            return text

    def extensions_set_kv(self, ext_id: str, raw_key: str, raw_value: str) -> str:
        key = (raw_key or "").strip().lower().replace("-", "_")
        if not key:
            raise ValueError("empty key")
        root = self._data.get("extensions")
        if not isinstance(root, dict):
            root = {}
        sub = root.get(ext_id)
        if not isinstance(sub, dict):
            sub = {}
        else:
            sub = dict(sub)
        sub[key] = self._coerce_extension_value(raw_value, ext_id, key)
        root = dict(root)
        root[ext_id] = sub
        self._data["extensions"] = root
        return f"extensions.{ext_id}.{key} set. Use /set save to persist."

    def extensions_unset_key(self, ext_id: str, raw_key: str) -> str:
        key = (raw_key or "").strip().lower().replace("-", "_")
        if not key:
            raise ValueError("empty key")
        root = self._data.get("extensions")
        if not isinstance(root, dict):
            root = {}
        sub = root.get(ext_id)
        if not isinstance(sub, dict):
            sub = {}
        else:
            sub = dict(sub)
        defs_sub = self._extensions_defaults_root().get(ext_id)
        if isinstance(defs_sub, dict) and key in defs_sub:
            sub[key] = defs_sub[key]
        else:
            sub.pop(key, None)
        root = dict(root)
        root[ext_id] = sub
        self._data["extensions"] = root
        return f"extensions.{ext_id}.{key} reset. Use /set save to persist."

    def get(self, path: Tuple[str, str]) -> Any:
        grp, key = path
        g = self._data.get(grp) if isinstance(self._data, dict) else None
        if not isinstance(g, dict):
            return None
        return g.get(key)

    def get_str(self, path: Tuple[str, str], default: str = "") -> str:
        v = self.get(path)
        s = str(v) if v is not None else ""
        s = s.strip()
        return s if s else default

    def get_bool(self, path: Tuple[str, str], default: bool = False) -> bool:
        v = self.get(path)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "y", "on")
        return default

    def get_int(self, path: Tuple[str, str], default: int = 0) -> int:
        v = self.get(path)
        try:
            if v is None:
                return int(default)
            return int(str(v).strip(), 10)
        except Exception:
            return int(default)

    def get_float(self, path: Tuple[str, str], default: float = 0.0) -> float:
        v = self.get(path)
        try:
            if v is None:
                return float(default)
            return float(str(v).strip())
        except Exception:
            return float(default)

    def set(self, path: Tuple[str, str], value: Any) -> None:
        grp, key = path
        if grp not in self._data or not isinstance(self._data.get(grp), dict):
            self._data[grp] = {}
        self._data[grp][key] = value

    def group_keys_lines(self, group: str) -> str:
        grp = (group or "").strip().lower()
        if grp not in ("ollama", "openai", "agent"):
            return ""
        keys = sorted(((DEFAULT_SETTINGS.get(grp) or {}).keys()))
        lines = ["Keys:"]
        for k in keys:
            lines.append(f"  {k}")
        return "\n".join(lines)

    def group_show(self, group: str) -> str:
        grp = (group or "").strip().lower()
        if grp not in ("ollama", "openai", "agent"):
            raise ValueError("group must be ollama, openai, or agent")
        cur = self._data.get(grp) if isinstance(self._data, dict) else None
        if not isinstance(cur, dict):
            cur = {}
        return json.dumps(cur, indent=2, ensure_ascii=False, sort_keys=True)

    def group_set(self, group: str, raw_key: str, raw_value: str) -> str:
        grp = (group or "").strip().lower()
        key = (raw_key or "").strip().lower().replace("-", "_")
        if grp not in ("ollama", "openai", "agent"):
            raise ValueError("group must be ollama, openai, or agent")
        defaults = DEFAULT_SETTINGS.get(grp) or {}
        if key not in defaults:
            raise ValueError(f"unknown key {key!r} for group {grp!r}")
        dv = defaults.get(key)
        text = (raw_value or "").strip()
        if isinstance(dv, bool):
            v: Any = text.lower() in ("1", "true", "yes", "y", "on")
        elif isinstance(dv, int) and not isinstance(dv, bool):
            v = int(float(text)) if text else 0
        elif isinstance(dv, float):
            v = float(text) if text else 0.0
        elif isinstance(dv, list):
            if not text.strip():
                v = []
            else:
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as e:
                    raise ValueError(f"invalid JSON array for {key!r}: {e}") from e
                if not isinstance(parsed, list):
                    raise ValueError(f"{key!r} must be a JSON array")
                v = parsed
        else:
            v = text
        self.set((grp, key), v)
        return f"{grp}.{key} set. Use /set save to persist."

    def group_unset(self, group: str, raw_key: str) -> str:
        grp = (group or "").strip().lower()
        key = (raw_key or "").strip().lower().replace("-", "_")
        if grp not in ("ollama", "openai", "agent"):
            raise ValueError("group must be ollama, openai, or agent")
        defaults = DEFAULT_SETTINGS.get(grp) or {}
        if key not in defaults:
            raise ValueError(f"unknown key {key!r} for group {grp!r}")
        self.set((grp, key), defaults.get(key))
        return f"{grp}.{key} reset to default. Use /set save to persist."

    def apply_prefs_groups_with_legacy_migration(self, prefs: dict) -> None:
        """
        Apply settings from prefs into this settings object, supporting legacy shapes:
        - prefs["ollama"]/["openai"]/["agent"] stored as env-like keys ("HOST", "OLLAMA_HOST", etc)
        - legacy top-level keys like "ollama_model"
        """
        if not isinstance(prefs, dict):
            return

        def apply_group(group: str, mapping: dict) -> None:
            if not isinstance(mapping, dict):
                return
            for k, v in mapping.items():
                if v is None:
                    continue
                key0 = str(k).strip()
                if not key0:
                    continue
                lk = key0.strip().lower().replace("-", "_")
                # Accept both short keys ("host") and env-style keys ("OLLAMA_HOST", "HOST").
                lk = re.sub(r"^(ollama_|openai_|agent_)", "", lk)
                if group == "ollama":
                    aliases = {
                        "host": "host",
                        "model": "model",
                        "second_opinion_model": "second_opinion_model",
                        "second_opinion": "second_opinion_model",
                        "second_opinion_model_tag": "second_opinion_model",
                        "tool_output_max": "tool_output_max",
                        "debug": "debug",
                        "search_enrich": "search_enrich",
                    }
                elif group == "openai":
                    aliases = {
                        "api_key": "api_key",
                        "base_url": "base_url",
                        "model": "model",
                        "cloud_model": "cloud_model",
                    }
                else:
                    aliases = {k: k for k in (DEFAULT_SETTINGS.get("agent") or {}).keys()}
                if lk not in aliases:
                    continue
                self.set((group, aliases[lk]), v)

        for grp in ("ollama", "openai", "agent"):
            apply_group(grp, prefs.get(grp) if isinstance(prefs.get(grp), dict) else {})

        # Legacy top-level keys.
        if isinstance(prefs.get("ollama_model"), str) and prefs["ollama_model"].strip():
            self.set(("ollama", "model"), prefs["ollama_model"].strip())
        if (
            isinstance(prefs.get("ollama_second_opinion_model"), str)
            and prefs["ollama_second_opinion_model"].strip()
        ):
            self.set(("ollama", "second_opinion_model"), prefs["ollama_second_opinion_model"].strip())

        ext_prefs = prefs.get("extensions")
        if isinstance(ext_prefs, dict):
            self.extensions_merge_from_prefs(ext_prefs)

