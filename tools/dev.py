import os
import subprocess


def _run(cmd: list[str], timeout: int = 600) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "").rstrip()
        err = (r.stderr or "").rstrip()
        body = []
        body.append("COMMAND: " + " ".join(cmd))
        if out:
            body.append("STDOUT:\n" + out)
        if err:
            body.append("STDERR:\n" + err)
        body.append(f"(exit {r.returncode})")
        return "\n".join(body)
    except Exception as e:
        return f"Dev tool error: {e}"


def run_pytest(params: dict) -> str:
    _ = params  # reserved for future options
    # Prefer uv if present; fall back to python -m pytest.
    if os.path.exists("uv.lock"):
        return _run(["uv", "run", "pytest", "-q"], timeout=900)
    return _run(["python3", "-m", "pytest", "-q"], timeout=900)


TOOLSET = {
    "name": "dev",
    "description": "Developer tools (tests/build helpers).",
    "triggers": ["pytest", "test", "tests", "ci", "build", "compile", "uv run"],
    "tools": [
        {
            "id": "run_pytest",
            "description": "Run the test suite (pytest).",
            "aliases": ("pytest", "run tests", "tests"),
            "handler": run_pytest,
        }
    ],
}

