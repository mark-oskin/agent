"""
Re-exec repo entry scripts with the project venv when run outside the install cwd.

Standalone module (not under ``agentlib/``) so ``agent.py`` can import it before
``agentlib`` loads — ``agentlib.__init__`` pulls in ``session`` which needs ``requests``.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Optional

_BOOTSTRAP_ENV = "AGENT_BOOTSTRAP"


def _project_dir(script_path: str) -> str:
    return os.path.dirname(os.path.abspath(script_path))


def _marker(project: str, uv_extra: Optional[str]) -> str:
    return f"{project}|{uv_extra or ''}"


def _deps_ok(*, uv_extra: Optional[str]) -> bool:
    try:
        import requests  # noqa: F401
    except ModuleNotFoundError:
        return False
    if uv_extra == "tui":
        try:
            import textual  # noqa: F401
        except ModuleNotFoundError:
            return False
    return True


def _venv_python(project: str) -> Optional[str]:
    for name in ("python3", "python"):
        candidate = os.path.join(project, ".venv", "bin", name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def ensure_runtime(script_path: str, *, uv_extra: Optional[str] = None) -> None:
    """
    Ensure ``script_path`` runs with project dependencies and ``agentlib`` importable.

    When ``requests`` (and optional ``textual``) are missing, re-exec via the project
    ``.venv`` or ``uv run --project <dir>`` so ``agent.py`` / ``agent_tui.py`` work from
    any working directory. Installed console scripts (``uv tool install``) usually skip
    re-exec because dependencies are already present.
    """
    project = _project_dir(script_path)
    if project not in sys.path:
        sys.path.insert(0, project)

    marker = _marker(project, uv_extra)
    if os.environ.get(_BOOTSTRAP_ENV) == marker:
        return
    if _deps_ok(uv_extra=uv_extra):
        return

    env = os.environ.copy()
    env[_BOOTSTRAP_ENV] = marker
    argv_tail = sys.argv[1:]

    vpy = _venv_python(project)
    if vpy is not None:
        os.execvpe(vpy, [vpy, script_path, *argv_tail], env)

    uv = shutil.which("uv")
    if uv:
        cmd = [uv, "run", "--project", project]
        if uv_extra:
            cmd.extend(["--extra", uv_extra])
        cmd.extend(["python", script_path, *argv_tail])
        os.execvpe(uv, cmd, env)

    need = "requests"
    if uv_extra == "tui":
        need = "requests and textual (uv sync --extra tui)"
    print(
        f"Error: {need} not installed.\n"
        f"From any directory, run: uv run --project {project!r} python {os.path.basename(script_path)!r}\n"
        f"Or: cd {project!r} && uv sync"
        + (" --extra tui" if uv_extra == "tui" else ""),
        file=sys.stderr,
    )
    raise SystemExit(1)
