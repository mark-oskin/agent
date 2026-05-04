import shutil
import subprocess
from typing import Optional


def run_applescript(params: dict) -> str:
    """
    Run arbitrary AppleScript via macOS `osascript`.

    This tool has side effects by design: the script can automate apps, modify files, etc.
    """
    p = params or {}
    script = str(p.get("script") or "").strip()
    if not script:
        return "AppleScript tool error: missing required parameter: script"

    timeout_ms: Optional[int]
    try:
        timeout_ms = int(p.get("timeout_ms")) if p.get("timeout_ms") is not None else 20_000
    except (TypeError, ValueError):
        timeout_ms = 20_000
    timeout_s = max(1, timeout_ms) / 1000.0

    exe = shutil.which("osascript")
    if not exe:
        return "AppleScript tool error: osascript not found (this tool requires macOS)."

    try:
        r = subprocess.run(
            [exe, "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return f"AppleScript tool error: timed out after {timeout_ms}ms"
    except Exception as e:
        return f"AppleScript tool error: {type(e).__name__}: {e}"

    out = (r.stdout or "").rstrip()
    err = (r.stderr or "").rstrip()
    lines: list[str] = []
    lines.append("COMMAND: osascript -e <script>")
    if out:
        lines.append("STDOUT:\n" + out)
    if err:
        lines.append("STDERR:\n" + err)
    lines.append(f"(exit {r.returncode})")
    return "\n".join(lines)


TOOLSET = {
    "name": "applescript",
    "description": "Run arbitrary AppleScript (macOS `osascript`). Side effects.",
    "triggers": [
        "applescript",
        "osascript",
        "safari",
        "mail",
        "messages",
        "finder",
        "system events",
    ],
    "tools": [
        {
            "id": "run_applescript",
            "description": "Run arbitrary AppleScript via `osascript -e`.",
            "aliases": ("applescript", "osascript", "run applescript"),
            "params": {
                "script": "AppleScript source code (string)",
                "timeout_ms": "optional: execution timeout in milliseconds (default 20000)",
            },
            "returns": "Command-like output with stdout/stderr and exit code.",
            "handler": run_applescript,
        }
    ],
}

