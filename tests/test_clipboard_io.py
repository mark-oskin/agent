"""Tests for agentlib.clipboard_io (mocked subprocess; no real clipboard needed)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agentlib import clipboard_io


def test_clipboard_linux_write_prefers_wl_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")

    def which(name: str) -> str | None:
        if name == "wl-copy":
            return "/fake/wl-copy"
        return None

    monkeypatch.setattr("shutil.which", which)
    captured: dict = {}

    def fake_run(cmd, *, capture_output: bool = True, input=None, timeout=None, check=False):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["input"] = input
        mm = MagicMock()
        mm.returncode = 0
        mm.stderr = b""
        return mm

    monkeypatch.setattr("subprocess.run", fake_run)
    clipboard_io.clipboard_write_text("hello ω")
    assert captured["cmd"] == ["wl-copy"]
    assert captured["input"] == b"hello \xcf\x89"
