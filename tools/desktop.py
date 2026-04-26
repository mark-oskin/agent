import subprocess


def open_url(params: dict) -> str:
    url = str((params or {}).get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return "Desktop tool error: url must start with http:// or https://"
    try:
        r = subprocess.run(["open", url], capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return f"Desktop tool error: open failed (exit {r.returncode}): {(r.stderr or r.stdout or '').strip()}"
        return f"Opened in browser: {url}"
    except Exception as e:
        return f"Desktop tool error: {e}"


TOOLSET = {
    "name": "desktop",
    "description": "Local desktop helpers (macOS). Side effects.",
    "triggers": ["browser", "open url", "open in browser", "desktop"],
    "tools": [
        {
            "id": "open_url",
            "description": "Open a URL in the default browser (macOS `open`).",
            "aliases": ("open url", "open in browser", "browser open"),
            "handler": open_url,
        }
    ],
}

