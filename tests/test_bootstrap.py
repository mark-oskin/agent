"""Entry script bootstrap (_bootstrap.ensure_runtime)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


def test_agent_py_imports_from_parent_cwd():
    """agent.py must resolve deps when cwd is not the project root."""
    env = os.environ.copy()
    env.pop("AGENT_BOOTSTRAP", None)
    proc = subprocess.run(
        [sys.executable, str(PROJECT / "agent.py"), "--help"],
        cwd=str(PROJECT.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "usage:" in proc.stdout.lower()


def test_bootstrap_does_not_import_agentlib():
    """Regression: bootstrap must run before agentlib (session imports requests)."""
    for key in list(sys.modules):
        if key == "agentlib" or key.startswith("agentlib."):
            del sys.modules[key]
    import _bootstrap  # noqa: F401

    assert "agentlib.session" not in sys.modules


def test_bootstrap_adds_project_to_sys_path():
    import _bootstrap

    _bootstrap.ensure_runtime(str(PROJECT / "agent.py"))
    assert str(PROJECT) in sys.path
