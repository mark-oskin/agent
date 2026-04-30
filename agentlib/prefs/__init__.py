import json
import os
import tempfile
from typing import Optional

from ..settings import AgentSettings


AGENT_PREFS_VERSION = 4

_AGENT_PREFS_PATH_OVERRIDE: Optional[str] = None


def agent_prefs_path() -> str:
    """Resolved path to prefs JSON (defaults to ~/.agent.json)."""
    global _AGENT_PREFS_PATH_OVERRIDE
    if isinstance(_AGENT_PREFS_PATH_OVERRIDE, str) and _AGENT_PREFS_PATH_OVERRIDE.strip():
        return _AGENT_PREFS_PATH_OVERRIDE
    return os.path.join(os.path.expanduser("~"), ".agent.json")


def set_agent_prefs_path_override(path: Optional[str]) -> None:
    """Override ~/.agent.json path for this process (used by --config)."""
    global _AGENT_PREFS_PATH_OVERRIDE
    p = (path or "").strip()
    if not p:
        _AGENT_PREFS_PATH_OVERRIDE = None
        return
    _AGENT_PREFS_PATH_OVERRIDE = os.path.abspath(os.path.expanduser(p))


def load_agent_prefs() -> Optional[dict]:
    path = agent_prefs_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def apply_prefs_to_settings(settings: AgentSettings, prefs: Optional[dict]) -> None:
    if not isinstance(prefs, dict):
        return
    settings.apply_prefs_groups_with_legacy_migration(prefs)


def write_agent_prefs_file(payload: dict) -> None:
    path = agent_prefs_path()
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
