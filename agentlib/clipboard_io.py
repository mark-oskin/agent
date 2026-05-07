"""Cross-platform text clipboard helpers (clipboard read/write only)."""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Tuple


class ClipboardError(RuntimeError):
    """No clipboard backend succeeded."""


def _run(cmd: Tuple[str, ...], *, input_bytes: bytes | None = None, timeout: float = 30.0) -> subprocess.CompletedProcess:
    kw: dict = {
        "capture_output": True,
        "timeout": timeout,
        "check": False,
    }
    if input_bytes is not None:
        kw["input"] = input_bytes
    return subprocess.run(cmd, **kw)


# --- Darwin ---


def _darwin_clipboard_read() -> str:
    proc = _run(("pbpaste",), timeout=60.0)
    if proc.returncode != 0:
        raise ClipboardError(proc.stderr.decode("utf-8", errors="replace") or f"pbpaste exit {proc.returncode}")
    return proc.stdout.decode("utf-8", errors="replace")


def _darwin_clipboard_write(text: str) -> None:
    proc = subprocess.run(
        ["pbcopy"],
        input=text.encode("utf-8"),
        capture_output=True,
        timeout=60.0,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise ClipboardError(err or f"pbcopy exit {proc.returncode}")


# --- Windows ---


def _windows_clipboard_read() -> str:
    # Get-Clipboard -Raw preserves newlines reasonably on modern PowerShell
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-Command", "(Get-Clipboard -Raw)"],
        capture_output=True,
        text=False,
        timeout=60.0,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise ClipboardError(err or f"Get-Clipboard exit {proc.returncode}")
    return proc.stdout.decode("utf-8", errors="replace").rstrip("\r\n")


def _windows_clipboard_write(text: str) -> None:
    proc = subprocess.run(
        ["clip"],
        input=text.encode("utf-16le"),
        capture_output=True,
        timeout=60.0,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise ClipboardError(err or f"clip exit {proc.returncode}")


# --- Linux (Wayland then X11) ---


def _linux_clipboard_read() -> str:
    if wl := shutil.which("wl-paste"):
        proc = _run(("wl-paste", "-n"))
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")
            raise ClipboardError(err or "wl-paste failed")
        return proc.stdout.decode("utf-8", errors="replace")
    if xclip := shutil.which("xclip"):
        proc = _run((xclip, "-selection", "clipboard", "-o"))
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")
            raise ClipboardError(err or "xclip read failed")
        return proc.stdout.decode("utf-8", errors="replace")
    if xsel := shutil.which("xsel"):
        proc = _run((xsel, "--clipboard", "--output"))
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")
            raise ClipboardError(err or "xsel read failed")
        return proc.stdout.decode("utf-8", errors="replace")
    raise ClipboardError(
        "No clipboard CLI found on Linux (try: wl-paste for Wayland, or xclip / xsel for X11)."
    )


def _linux_clipboard_write(text: str) -> None:
    data = text.encode("utf-8")
    if wl := shutil.which("wl-copy"):
        proc = subprocess.run(
            ["wl-copy"],
            input=data,
            capture_output=True,
            timeout=60.0,
            check=False,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")
            raise ClipboardError(err or "wl-copy failed")
        return
    if xclip := shutil.which("xclip"):
        proc = subprocess.run(
            [xclip, "-selection", "clipboard"],
            input=data,
            capture_output=True,
            timeout=60.0,
            check=False,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")
            raise ClipboardError(err or "xclip write failed")
        return
    if xsel := shutil.which("xsel"):
        proc = subprocess.run(
            [xsel, "--clipboard", "--input"],
            input=data,
            capture_output=True,
            timeout=60.0,
            check=False,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")
            raise ClipboardError(err or "xsel write failed")
        return
    raise ClipboardError(
        "No clipboard CLI found on Linux (try: wl-copy for Wayland, or xclip / xsel for X11)."
    )


def clipboard_read_text() -> str:
    sysname = platform.system().lower()
    if sysname == "darwin":
        return _darwin_clipboard_read()
    if sysname == "windows":
        return _windows_clipboard_read()
    if sysname == "linux":
        return _linux_clipboard_read()
    # BSD / other Unix
    if shutil.which("wl-paste") or shutil.which("xclip") or shutil.which("xsel"):
        return _linux_clipboard_read()
    if shutil.which("pbpaste"):
        return _darwin_clipboard_read()
    raise ClipboardError("Clipboard read is not configured for this platform.")


def clipboard_write_text(text: str) -> None:
    sysname = platform.system().lower()
    if sysname == "darwin":
        _darwin_clipboard_write(text)
        return
    if sysname == "windows":
        _windows_clipboard_write(text)
        return
    if sysname == "linux":
        _linux_clipboard_write(text)
        return
    if shutil.which("wl-copy") or shutil.which("xclip") or shutil.which("xsel"):
        _linux_clipboard_write(text)
        return
    if shutil.which("pbcopy"):
        _darwin_clipboard_write(text)
        return
    raise ClipboardError("Clipboard write is not configured for this platform.")
