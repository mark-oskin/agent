from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import time
from typing import Optional

import requests

from agentlib.tools.websearch import readability_excerpt_from_html


def _scalar_to_str(x, default: str) -> str:
    if x is None:
        return default
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return default


def _scalar_to_int(x, default: int) -> int:
    if x is None:
        return int(default)
    if isinstance(x, bool):
        return int(default)
    try:
        if isinstance(x, (int, float)):
            return int(x)
        s = str(x).strip()
        if not s:
            return int(default)
        return int(float(s))
    except Exception:
        return int(default)


# ---- core built-in tools (simple / local) ----


def write_file(path, content):
    path = _scalar_to_str(path, "")
    content = _scalar_to_str(content, "")
    if not path.strip():
        return "Write error: path is empty."
    if not content.strip():
        return (
            "Write error: parameters.content is required (non-empty string) with the full file body. "
            "Do not call write_file with only a path; for a letter or document, put the entire text in content."
        )
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"File {path} written successfully."
    except Exception as e:
        return f"Write error: {e}"


def list_directory(path):
    path = _scalar_to_str(path, "")
    try:
        entries = os.listdir(path)
        return json.dumps(entries)
    except Exception as e:
        return f"List dir error: {e}"


def read_file(path):
    path = _scalar_to_str(path, "")
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Read error: {e}"


def download_file(url, path):
    url = _scalar_to_str(url, "")
    path = _scalar_to_str(path, "")
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
        return f"File downloaded to {path}."
    except Exception as e:
        return f"Download error: {e}"


def tail_file(path, lines=20):
    path = _scalar_to_str(path, "")
    lines = _scalar_to_int(lines, 20)
    if lines < 1:
        lines = 20
    try:
        with open(path, "r") as f:
            content = f.readlines()[-lines:]
        return "".join(content)
    except Exception as e:
        return f"Tail error: {e}"


def replace_text(path, pattern, replacement, replace_all=True):
    path = _scalar_to_str(path, "")
    pattern = _scalar_to_str(pattern, "")
    replacement = _scalar_to_str(replacement, "")
    if replace_all is None:
        replace_all = True
    if isinstance(replace_all, str):
        replace_all = replace_all.strip().lower() in ("1", "true", "yes", "on")
    try:
        with open(path, "r") as f:
            text = f.read()
        count = 0 if bool(replace_all) else 1
        new_text = re.sub(pattern, replacement, text, count=count)
        with open(path, "w") as f:
            f.write(new_text)
        return f"Replaced text in {path}."
    except Exception as e:
        return f"Replace error: {e}"


def call_python(code, globals=None):
    code = _scalar_to_str(code, "")
    if not code.strip():
        return "Exec error: empty code string."
    try:
        compiled = compile(code, "<call_python>", "exec")
    except SyntaxError as e:
        return (
            "Exec error: not valid Python source (call_python only runs Python, not shell scripts or prose). "
            f"{e.msg} at line {e.lineno}. For letters, essays, or files use write_file, "
            'or answer with {"action":"answer","answer":"..."}.'
        )
    except Exception as e:
        return f"Exec error: could not compile code: {e}"
    g = dict(globals) if isinstance(globals, dict) else {}
    local_vars: dict = {}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compiled, g, local_vars)
    except Exception as e:
        out = buf.getvalue()
        prefix = f"STDOUT (partial, before error):\n{out.rstrip()}\n\n" if out.strip() else ""
        return f"{prefix}Exec error: {e}"
    out = buf.getvalue().rstrip()
    to_dump = {
        k: v for k, v in local_vars.items() if not (isinstance(k, str) and k.startswith("__"))
    }
    try:
        j = json.dumps(to_dump, default=str, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception as e:
        j = f"(error encoding locals as JSON: {e}; keys: {list(to_dump.keys())!r})"
    if not out:
        return j
    return f"STDOUT:\n{out}\n\n--- locals (JSON) ---\n{j}"


def run_command(command):
    command = _scalar_to_str(command, "")
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60
        )
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return f"Command error: {e}"


def fetch_page(url):
    url = _scalar_to_str(url, "").strip()
    if not url:
        return "Fetch error: empty url string."
    if not re.match(r"^https?://", url, re.IGNORECASE):
        return f"Fetch error: URL must start with http:// or https:// (got {url!r})."
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    last_exc: Optional[BaseException] = None
    for attempt in (0, 1):
        timeout = 12.0 if attempt == 0 else 22.0
        try:
            resp = requests.get(
                url, headers=headers, timeout=timeout, allow_redirects=True
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt == 0:
                time.sleep(0.35)
                continue
            return f"Fetch error: {e}"
        except requests.exceptions.RequestException as e:
            return f"Fetch error: {e}"
        st = int(resp.status_code)
        if st in (429, 500, 502, 503, 504) and attempt == 0:
            time.sleep(0.4)
            continue
        if st >= 400:
            return (
                f"Fetch error: HTTP {st} for this URL. "
                "If access is denied, try a different page (docs, help, or search_web for an official link). "
                "Do not use run_command with curl."
            )
        final_url = resp.url
        raw_html = resp.text or ""
        prefix = f"Fetched URL: {url}\nFinal URL: {final_url}\n\n"
        if attempt == 1:
            prefix = "[After automatic retry] " + prefix
        title, excerpt = readability_excerpt_from_html(
            raw_html, url=final_url, max_chars=8000
        )
        excerpt = (excerpt or "").strip()
        if excerpt:
            parts = [prefix]
            if (title or "").strip():
                parts.append(f"Title: {(title or '').strip()}\n\n")
            parts.append(excerpt)
            return "".join(parts)
        text = re.sub(r"<[^>]*>", " ", raw_html)
        text = re.sub(r"\s+", " ", text).strip()
        return prefix + text[:5000]
    if last_exc is not None:
        return f"Fetch error: {last_exc}"
    return f"Fetch error: could not retrieve {url!r} after retry."


def use_git(params) -> str:
    """Vetted git operations via argument lists (no shell)."""
    p = params if isinstance(params, dict) else {}
    op = _scalar_to_str(p.get("op") or p.get("operation"), "").strip().lower()
    wt = _scalar_to_str(p.get("worktree") or p.get("cwd") or p.get("path") or "", "").strip()
    try:
        cwd0 = os.path.abspath(os.path.expanduser(wt)) if wt else os.getcwd()
    except Exception:
        cwd0 = os.getcwd()
    if not os.path.isdir(cwd0):
        return f"use_git error: worktree is not a directory: {cwd0}"

    def _git_run(args: list, timeout: int = 180) -> str:
        try:
            r = subprocess.run(
                ["git", *args],
                cwd=cwd0,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = (r.stdout or "").rstrip()
            err = (r.stderr or "").rstrip()
            lines = []
            if out:
                lines.append(out)
            if err:
                lines.append("STDERR:\\n" + err)
            lines.append(f"(exit {r.returncode})")
            return "\\n".join(lines)
        except Exception as e:
            return f"use_git error: {e}"

    check = subprocess.run(
        ["git", "-C", cwd0, "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if check.returncode != 0:
        return (
            f"use_git error: not a git work tree (or git missing): {cwd0}\\n"
            f"{(check.stderr or check.stdout or '').strip()}"
        )

    def _repo_path(one: str) -> str:
        s = str(one).strip()
        if not s:
            return s
        pth = os.path.expanduser(s)
        if not os.path.isabs(pth):
            pth = os.path.join(cwd0, pth)
        return os.path.normpath(pth)

    if not op:
        return "use_git error: parameters.op is required (status, log, diff, add, commit, push, pull, branch)."
    if op in ("status", "st"):
        return _git_run(["status", "-sb"])
    if op == "log":
        n = _scalar_to_int(p.get("n") or p.get("lines"), 20)
        if n < 1:
            n = 20
        n = min(int(n), 200)
        return _git_run(["log", "--oneline", f"-n{n}"])
    if op == "diff":
        stg = p.get("staged")
        if isinstance(stg, str):
            stg = stg.strip().lower() in ("1", "true", "yes", "on")
        return _git_run(["diff", "--staged"] if stg else ["diff"])
    if op == "add":
        paths = p.get("paths")
        if paths is None and p.get("path"):
            paths = [p.get("path")]
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, list) or not paths:
            return "use_git error: add requires parameters.paths (non-empty list of path strings)."
        args = ["add", "--"]
        for one in paths:
            args.append(_repo_path(str(one)))
        return _git_run(args)
    if op == "commit":
        msg = _scalar_to_str(p.get("message") or p.get("m"), "").strip()
        if not msg:
            return "use_git error: commit requires parameters.message."
        return _git_run(["commit", "-m", msg], timeout=120)
    if op == "push":
        rem = _scalar_to_str(p.get("remote"), "origin").strip() or "origin"
        br = _scalar_to_str(p.get("branch"), "").strip()
        args = ["push", rem]
        if br:
            args.append(br)
        return _git_run(args, timeout=300)
    if op == "pull":
        rem = _scalar_to_str(p.get("remote"), "origin").strip() or "origin"
        br = _scalar_to_str(p.get("branch"), "").strip()
        args = ["pull", rem]
        if br:
            args.append(br)
        return _git_run(args, timeout=300)
    if op in ("branch", "branches"):
        return _git_run(["branch", "-a", "-vv"])
    return f"use_git error: unknown op {op!r} (try status, log, diff, add, commit, push, pull, branch)."


def search_web(query, params: Optional[dict] = None, *, settings=None) -> str:
    from . import websearch

    return websearch.search_web(
        str(query or ""),
        params=params,
        settings=settings,
        fetch_page=fetch_page,
    )


def search_web_fetch_top(query, params: Optional[dict] = None, *, settings=None) -> str:
    from . import websearch

    return websearch.search_web_fetch_top(
        str(query or ""),
        params=params,
        settings=settings,
        fetch_page=fetch_page,
    )

