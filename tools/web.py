"""
Headless browser control (Playwright) for the agent.

Supports two Playwright engines:

* **chromium** — Chrome/Chromium channel (default).
* **webkit** — WebKit (same engine family as Apple Safari). Alias: **safari**.

``safari`` does **not** attach to the Safari.app GUI; use WebKit for automated
layout/JS testing that tracks Safari behaviour. Driving the real Safari browser
would require AppleScript / ``safaridriver`` and is not implemented here.

Enable the toolset in-session::

    /settings tools enable browser
    /settings save

Requires::

    uv sync --extra browser
    playwright install chromium webkit

Pick an engine on each ``browser_navigate`` (``parameters.engine``), or set
``AGENT_BROWSER_ENGINE`` to ``chromium`` or ``webkit`` / ``safari`` for the default
when a call omits ``engine``.

Use ``browser_close`` to tear down; the next navigate recreates the browser.

Security: only ``http://`` and ``https://`` URLs are accepted for navigation.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from typing import Any, Optional, Tuple
from urllib.parse import urlparse

_STATE_LOCK = threading.Lock()
_PW: Any = None
_BROWSER: Any = None
_PAGE: Any = None
_CURRENT_ENGINE: Optional[str] = None  # "chromium" | "webkit"

_PLAYWRIGHT_AVAILABLE = importlib.util.find_spec("playwright") is not None
_PLAYWRIGHT_INSTALL_HINT = (
    "Playwright is not installed. Install it with:\n"
    "  uv sync --extra browser\n"
    "  playwright install chromium webkit"
)


def _no_playwright(tool: str) -> str:
    return f"{tool} error: {_PLAYWRIGHT_INSTALL_HINT}"


def _url_allowed(url: str) -> Optional[str]:
    u = (url or "").strip()
    if not u:
        return "browser error: url is empty"
    try:
        p = urlparse(u)
    except Exception as e:  # pragma: no cover - urlparse rarely fails
        return f"browser error: invalid url ({e})"
    if p.scheme not in ("http", "https"):
        return f"browser error: scheme {p.scheme!r} not allowed (only http/https)"
    if not p.netloc:
        return "browser error: missing host in url"
    return None


def _truncate(s: str, max_chars: int) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + "\n… [truncated] …"


def _normalize_engine(raw: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Return (internal_engine_id, error_message).

    internal_engine_id is ``chromium`` or ``webkit`` (Playwright).
    ``safari`` maps to ``webkit`` — Safari's rendering engine, not Safari.app.
    """
    s = ("" if raw is None else str(raw)).strip().lower()
    if not s:
        s = (os.environ.get("AGENT_BROWSER_ENGINE") or "chromium").strip().lower()
    if s in ("chromium", "chrome", "google-chrome"):
        return "chromium", None
    if s in ("webkit", "safari", "safari-webkit"):
        return "webkit", None
    return "", f"browser error: unknown engine {raw!r} (use chromium, webkit, or safari→webkit)"


def _launch_locked(engine: str) -> Any:
    """Launch Playwright + browser + page (caller must hold ``_STATE_LOCK``)."""
    global _PW, _BROWSER, _PAGE, _CURRENT_ENGINE
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:  # pragma: no cover - env dependent
        raise ImportError(_PLAYWRIGHT_INSTALL_HINT) from e
    sp = sync_playwright().start()
    pw = sp
    if engine == "chromium":
        browser = pw.chromium.launch(headless=True)
    else:
        browser = pw.webkit.launch(headless=True)
    page = browser.new_page()
    _PW, _BROWSER, _PAGE, _CURRENT_ENGINE = sp, browser, page, engine
    return page


def _teardown_locked() -> list[str]:
    """Close page/browser/playwright. Caller must hold ``_STATE_LOCK``."""
    global _PW, _BROWSER, _PAGE, _CURRENT_ENGINE
    msg: list[str] = []
    try:
        if _PAGE is not None:
            _PAGE.close()
    except Exception as e:
        msg.append(f"page close: {e}")
    try:
        if _BROWSER is not None:
            _BROWSER.close()
    except Exception as e:
        msg.append(f"browser close: {e}")
    try:
        if _PW is not None:
            _PW.stop()
    except Exception as e:
        msg.append(f"playwright stop: {e}")
    _PW, _BROWSER, _PAGE, _CURRENT_ENGINE = None, None, None, None
    return msg


def _get_page(params: Optional[dict] = None) -> Any:
    """Return a live Playwright Page, launching or switching engine as needed."""
    if not _PLAYWRIGHT_AVAILABLE:
        raise ImportError(_PLAYWRIGHT_INSTALL_HINT)
    p = params or {}
    with _STATE_LOCK:
        raw: Optional[str] = None
        if p.get("engine") is not None:
            raw = str(p.get("engine"))
        elif p.get("browser") is not None:
            raw = str(p.get("browser"))
        elif _CURRENT_ENGINE is not None:
            raw = _CURRENT_ENGINE
        want, err = _normalize_engine(raw)
        if err:
            raise ValueError(err)
        if _PAGE is not None and _CURRENT_ENGINE == want:
            return _PAGE
        if _PAGE is not None or _PW is not None:
            _teardown_locked()
        return _launch_locked(want)


def _reset_browser(reason: str = "") -> str:
    with _STATE_LOCK:
        msg = _teardown_locked()
    tail = ("; " + "; ".join(msg)) if msg else ""
    return f"Browser session closed. {reason}{tail}".strip()


def _engine_label(engine: Optional[str]) -> str:
    if engine == "webkit":
        return "webkit (Safari-compatible WebKit)"
    return "chromium"


def browser_navigate(params: dict) -> str:
    url = str((params or {}).get("url") or "").strip()
    err = _url_allowed(url)
    if err:
        return err
    try:
        page = _get_page(params or {})
        page.goto(url, wait_until="domcontentloaded", timeout=90_000)
        title = page.title()
        eng = _CURRENT_ENGINE or "chromium"
        return (
            f"Navigated OK.\nEngine: {_engine_label(eng)}\nURL: {page.url}\nTitle: {title!r}"
        )
    except ValueError as e:
        return str(e)
    except ImportError:
        return _no_playwright("browser_navigate")
    except Exception as e:
        return f"browser_navigate error: {type(e).__name__}: {e}"


def browser_click(params: dict) -> str:
    sel = str((params or {}).get("selector") or "").strip()
    if not sel:
        return "browser error: selector is required"
    timeout = int((params or {}).get("timeout_ms") or 30_000)
    try:
        page = _get_page(params or {})
        page.click(sel, timeout=timeout)
        return f"Clicked selector: {sel!r}"
    except ValueError as e:
        return str(e)
    except ImportError:
        return _no_playwright("browser_click")
    except Exception as e:
        return f"browser_click error: {type(e).__name__}: {e}"


def browser_fill(params: dict) -> str:
    sel = str((params or {}).get("selector") or "").strip()
    text = (params or {}).get("text")
    if text is None:
        text = (params or {}).get("value")
    text = "" if text is None else str(text)
    if not sel:
        return "browser error: selector is required"
    timeout = int((params or {}).get("timeout_ms") or 30_000)
    try:
        page = _get_page(params or {})
        page.fill(sel, text, timeout=timeout)
        return f"Filled selector {sel!r} ({len(text)} chars)."
    except ValueError as e:
        return str(e)
    except ImportError:
        return _no_playwright("browser_fill")
    except Exception as e:
        return f"browser_fill error: {type(e).__name__}: {e}"


def browser_type(params: dict) -> str:
    """Type text character-by-character (use ``browser_fill`` for form fields)."""
    sel = str((params or {}).get("selector") or "").strip()
    text = str((params or {}).get("text") or "")
    delay_ms = int((params or {}).get("delay_ms") or 0)
    if not sel:
        return "browser error: selector is required"
    timeout = int((params or {}).get("timeout_ms") or 30_000)
    try:
        page = _get_page(params or {})
        page.type(sel, text, delay=delay_ms, timeout=timeout)
        return f"Typed {len(text)} chars into {sel!r}."
    except ValueError as e:
        return str(e)
    except ImportError:
        return _no_playwright("browser_type")
    except Exception as e:
        return f"browser_type error: {type(e).__name__}: {e}"


def browser_press(params: dict) -> str:
    sel = str((params or {}).get("selector") or "").strip()
    key = str((params or {}).get("key") or "Enter").strip() or "Enter"
    if not sel:
        return "browser error: selector is required"
    timeout = int((params or {}).get("timeout_ms") or 30_000)
    try:
        page = _get_page(params or {})
        page.press(sel, key, timeout=timeout)
        return f"Pressed {key!r} on {sel!r}."
    except ValueError as e:
        return str(e)
    except ImportError:
        return _no_playwright("browser_press")
    except Exception as e:
        return f"browser_press error: {type(e).__name__}: {e}"


def browser_snapshot(params: dict) -> str:
    """Return visible text from the page (or a selector) for the model to read."""
    sel = str((params or {}).get("selector") or "body").strip() or "body"
    max_chars = int((params or {}).get("max_chars") or 24_000)
    timeout = int((params or {}).get("timeout_ms") or 30_000)
    try:
        page = _get_page(params or {})
        loc = page.locator(sel).first
        loc.wait_for(state="attached", timeout=timeout)
        text = loc.inner_text(timeout=timeout)
        title = page.title()
        eng = _CURRENT_ENGINE or "chromium"
        header = (
            f"--- browser snapshot ---\nEngine: {_engine_label(eng)}\n"
            f"URL: {page.url}\nTitle: {title!r}\nSelector: {sel!r}\n--- text ---\n"
        )
        return header + _truncate(text.strip(), max_chars)
    except ValueError as e:
        return str(e)
    except ImportError:
        return _no_playwright("browser_snapshot")
    except Exception as e:
        return f"browser_snapshot error: {type(e).__name__}: {e}"


def browser_wait(params: dict) -> str:
    """Wait for a load state or a selector."""
    p = params or {}
    if p.get("selector"):
        sel = str(p["selector"]).strip()
        timeout = int(p.get("timeout_ms") or 30_000)
        try:
            page = _get_page(p)
            page.locator(sel).first.wait_for(state="visible", timeout=timeout)
            return f"Selector visible: {sel!r}"
        except ValueError as e:
            return str(e)
        except ImportError:
            return _no_playwright("browser_wait")
        except Exception as e:
            return f"browser_wait error: {type(e).__name__}: {e}"
    state = str(p.get("load_state") or "networkidle").strip().lower()
    if state not in ("load", "domcontentloaded", "networkidle"):
        return "browser error: load_state must be load|domcontentloaded|networkidle or pass selector=…"
    timeout = int(p.get("timeout_ms") or 90_000)
    try:
        page = _get_page(p)
        page.wait_for_load_state(state, timeout=timeout)
        return f"Load state reached: {state!r}"
    except ValueError as e:
        return str(e)
    except ImportError:
        return _no_playwright("browser_wait")
    except Exception as e:
        return f"browser_wait error: {type(e).__name__}: {e}"


def browser_close(params: dict) -> str:
    _ = params
    return _reset_browser("Call browser_navigate to open a new session.")


TOOLSET = {
    "name": "browser",
    "description": (
        "Headless browser control via Playwright: Chromium and WebKit (Safari engine)."
        + ("" if _PLAYWRIGHT_AVAILABLE else " (Playwright not installed; enable will fail until installed.)")
    ),
    "triggers": [
        "playwright",
        "headless",
        "browser automation",
        "css selector",
        "open the page",
        "click the button",
        "webkit",
        "safari",
    ],
    "tools": [
        {
            "id": "browser_navigate",
            "description": (
                "Open an http(s) URL (DOMContentLoaded). "
                "Optional engine: chromium (default) | webkit | safari (safari→WebKit, not Safari.app)."
            ),
            "aliases": ("goto", "open url in browser", "navigate browser"),
            "params": {
                "url": "full http(s) URL",
                "engine": "optional: chromium | webkit | safari (aliases: browser)",
            },
            "returns": "engine, final URL, page title",
            "handler": browser_navigate,
        },
        {
            "id": "browser_click",
            "description": "Click an element (CSS selector). Optional engine reuses or switches session.",
            "aliases": ("click element",),
            "params": {"selector": "CSS selector", "timeout_ms": "optional", "engine": "optional chromium|webkit|safari"},
            "returns": "confirmation or error",
            "handler": browser_click,
        },
        {
            "id": "browser_fill",
            "description": "Fill an input or textarea (clears then sets value).",
            "aliases": ("set input", "fill field"),
            "params": {
                "selector": "CSS selector",
                "text": "value (alias: value)",
                "timeout_ms": "optional",
                "engine": "optional",
            },
            "returns": "confirmation or error",
            "handler": browser_fill,
        },
        {
            "id": "browser_type",
            "description": "Type text into a focused element (keystrokes; slower than fill).",
            "aliases": ("type slowly",),
            "params": {"selector": "CSS selector", "text": "string", "delay_ms": "optional", "engine": "optional"},
            "returns": "confirmation or error",
            "handler": browser_type,
        },
        {
            "id": "browser_press",
            "description": "Press a key on an element (default Enter).",
            "aliases": ("press enter",),
            "params": {"selector": "CSS selector", "key": "optional", "engine": "optional"},
            "returns": "confirmation or error",
            "handler": browser_press,
        },
        {
            "id": "browser_snapshot",
            "description": "Read visible text from the page or a CSS selector for the model.",
            "aliases": ("read page", "page text", "get page content"),
            "params": {"selector": "optional", "max_chars": "optional", "engine": "optional"},
            "returns": "engine, URL, title, truncated inner text",
            "handler": browser_snapshot,
        },
        {
            "id": "browser_wait",
            "description": "Wait for load state or for a selector to become visible.",
            "aliases": ("wait for selector",),
            "params": {
                "load_state": "if no selector",
                "selector": "optional",
                "timeout_ms": "optional",
                "engine": "optional",
            },
            "returns": "confirmation or error",
            "handler": browser_wait,
        },
        {
            "id": "browser_close",
            "description": "Close the headless browser session (free RAM). Next navigate recreates it.",
            "aliases": ("close browser",),
            "params": {},
            "returns": "status string",
            "handler": browser_close,
        },
    ],
}
