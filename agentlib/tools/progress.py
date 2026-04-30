from __future__ import annotations

from agentlib.coercion import scalar_to_str
from agentlib.tools.websearch import search_backend_banner_line


def _clip(s: object, max_len: int = 120) -> str:
    t = scalar_to_str(s, "").replace("\n", " ").strip()
    if len(t) > max_len:
        return t[: max_len - 1] + "…"
    return t


def tool_progress_message(tool: str, params: dict, *, search_backend_banner: str = "") -> str:
    """Compact, useful progress line for verbose=0 heartbeats."""
    t = (tool or "").strip()
    p = params if isinstance(params, dict) else {}
    if t == "search_web":
        banner = search_backend_banner.strip()
        banner_bit = f"{banner} " if banner else ""
        return f"Tool: search_web {banner_bit}query={_clip(p.get('query'))!r}"
    if t == "fetch_page":
        return f"Tool: fetch_page url={_clip(p.get('url'))!r}"
    if t == "read_file":
        return f"Tool: read_file path={_clip(p.get('path'))!r}"
    if t == "list_directory":
        return f"Tool: list_directory path={_clip(p.get('path'))!r}"
    if t == "tail_file":
        return f"Tool: tail_file path={_clip(p.get('path'))!r} lines={_clip(p.get('lines', 20))}"
    if t == "run_command":
        return f"Tool: run_command command={_clip(p.get('command'))!r}"
    if t == "write_file":
        return f"Tool: write_file path={_clip(p.get('path'))!r}"
    if t == "replace_text":
        return (
            f"Tool: replace_text path={_clip(p.get('path'))!r} "
            f"pattern={_clip(p.get('pattern'))!r}"
        )
    if t == "download_file":
        return (
            f"Tool: download_file url={_clip(p.get('url'))!r} "
            f"path={_clip(p.get('path'))!r}"
        )
    if t == "use_git":
        op = _clip(p.get("op") or p.get("operation"))
        return f"Tool: use_git op={op!r}"
    if t == "call_python":
        return "Tool: call_python"
    return f"Tool: {t}"


def tool_progress_message_with_settings(tool: str, params: dict, *, settings) -> str:
    return tool_progress_message(
        tool,
        params,
        search_backend_banner=search_backend_banner_line(settings),
    )

