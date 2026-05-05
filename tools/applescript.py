import shutil
import subprocess
import tempfile
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

    echo_script = bool(p.get("echo_script", False))
    use_temp_file = bool(p.get("use_temp_file", False))

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
        cmd: list[str]
        tmp_path = None
        if use_temp_file:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".applescript", delete=False) as f:
                tmp_path = f.name
                f.write(script)
            cmd = [exe, tmp_path]
        else:
            cmd = [exe, "-e", script]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return f"AppleScript tool error: timed out after {timeout_ms}ms"
    except Exception as e:
        return f"AppleScript tool error: {type(e).__name__}: {e}"
    finally:
        if use_temp_file and tmp_path:
            try:
                import os

                os.unlink(tmp_path)
            except Exception:
                pass

    out = (r.stdout or "").rstrip()
    err = (r.stderr or "").rstrip()
    lines: list[str] = []
    if use_temp_file:
        lines.append("COMMAND: osascript <tempfile.applescript>")
    else:
        lines.append("COMMAND: osascript -e <script>")
    if echo_script:
        lines.append("SCRIPT:\n" + script)
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
                "echo_script": "optional: include SCRIPT: block in tool output (default false)",
                "use_temp_file": "optional: run script via temp .applescript file for better error line/column (default false)",
            },
            "prompt_doc": (
                "run_applescript — parameters.script (AppleScript source code string); "
                "optional parameters.timeout_ms (integer, default 20000), echo_script (bool), use_temp_file (bool). "
                "Date/time rule: for a specific clock time on a calendar day (e.g. today at HH:MM), "
                "do not add hours to `current date`—that offsets from now. "
                "Set hours, minutes, and seconds on the target date explicitly (e.g. assign `current date` to a variable, "
                "then set `hours of`, `minutes of`, `seconds of` on it)."
            ),
            "returns": "Command-like output with stdout/stderr and exit code.",
            "handler": run_applescript,
        }
    ],
}

