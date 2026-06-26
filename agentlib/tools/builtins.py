from __future__ import annotations

import contextlib
import fnmatch
import io
import json
import os
import re
import subprocess
import time
from typing import Any, Optional, Union

import requests

from agentlib.tools.python_validate import call_python_code_rejected_reason
from agentlib.tools.turn_support import normalize_fetch_urls
from agentlib.tools.websearch import readability_excerpt_from_html

FETCH_PAGE_MAX_URLS_PER_CALL = 25
FETCH_PAGE_BODY_TEXT_MAX = 8000


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


_GREP_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".nox",
        "dist",
        "build",
        ".eggs",
        ".gradle",
        ".idea",
        ".vscode",
    }
)
_GREP_MAX_FILE_BYTES = 2 * 1024 * 1024
_GREP_MAX_OUTPUT_CHARS = 120_000


def _grep_scalar_bool(x, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def _grep_file_matches_glob(rel_path: str, pat: str) -> bool:
    if not pat.strip():
        return True
    rel = rel_path.replace(os.sep, "/")
    bn = os.path.basename(rel)
    return fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(bn, pat)


def grep(pattern, path=".", glob_pattern=None, max_matches=200, max_files=8000, ignore_case=False):
    """
    Search for ``pattern`` (Python ``re`` regex) under ``path`` (file or directory).

    Optional ``glob_pattern`` (e.g. ``*.py``) filters relative paths when searching a directory.
    """
    pattern = _scalar_to_str(pattern, "")
    path = _scalar_to_str(path, ".") or "."
    gp = glob_pattern
    glob_pattern = _scalar_to_str(gp, "").strip() if gp is not None else ""
    glob_pattern = glob_pattern or None
    max_matches = max(1, min(_scalar_to_int(max_matches, 200), 5000))
    max_files = max(1, min(_scalar_to_int(max_files, 8000), 50_000))
    ignore_case = _grep_scalar_bool(ignore_case, False)
    if not pattern.strip():
        return "Grep error: parameters.pattern (regex) is required and must be non-empty."
    flags = re.IGNORECASE if ignore_case else 0
    try:
        cre = re.compile(pattern, flags)
    except re.error as e:
        return f"Grep error: invalid regex: {e}"
    root = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(root):
        return f"Grep error: path does not exist: {root!r}"

    lines_out: list[str] = []
    out_len = 0
    total_matches = 0
    files_scanned = 0
    truncated = False

    def emit(line: str) -> bool:
        nonlocal out_len, truncated
        if truncated:
            return False
        if out_len + len(line) + 1 > _GREP_MAX_OUTPUT_CHARS:
            truncated = True
            return False
        lines_out.append(line)
        out_len += len(line) + 1
        return True

    def scan_one_file(fp: str, rel_disp: str) -> bool:
        """Return False if caller should stop scanning (limits or output cap)."""
        nonlocal files_scanned, total_matches, truncated
        if total_matches >= max_matches or truncated:
            return False
        try:
            sz = os.path.getsize(fp)
        except OSError:
            return True
        if sz > _GREP_MAX_FILE_BYTES:
            return True
        if files_scanned >= max_files:
            return False
        files_scanned += 1
        try:
            with open(fp, "rb") as bf:
                probe = bf.read(8192)
            if b"\x00" in probe:
                return True
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    if total_matches >= max_matches:
                        return False
                    if cre.search(line):
                        piece = line.rstrip("\n\r")
                        if len(piece) > 2000:
                            piece = piece[:2000] + "…"
                        if not emit(f"{rel_disp}:{lineno}:{piece}"):
                            return False
                        total_matches += 1
        except (OSError, UnicodeError):
            return not truncated
        return not truncated

    if os.path.isfile(root):
        rel_disp = os.path.basename(root)
        if glob_pattern and not _grep_file_matches_glob(rel_disp, glob_pattern):
            return (
                "Grep: 0 matches (optional glob_pattern did not match this file; "
                "omit glob_pattern or point path at a directory)."
            )
        scan_one_file(root, rel_disp)
    elif os.path.isdir(root):
        stop_walk = False
        for dirpath, dirnames, filenames in os.walk(root):
            if stop_walk:
                break
            dirnames[:] = [
                d
                for d in dirnames
                if d not in _GREP_SKIP_DIR_NAMES and not str(d).endswith(".egg-info")
            ]
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                rel = os.path.relpath(fp, root).replace(os.sep, "/")
                if glob_pattern and not _grep_file_matches_glob(rel, glob_pattern):
                    continue
                if not scan_one_file(fp, rel):
                    stop_walk = True
                    break
    else:
        return f"Grep error: not a file or directory: {root!r}"

    if not lines_out:
        hint = ""
        if glob_pattern:
            hint = f" (glob_pattern={glob_pattern!r})"
        return f"Grep: 0 matches under {root!r}{hint}."

    footer = []
    footer.append(f"[Grep summary] matches={total_matches} files_opened={files_scanned}")
    if truncated or total_matches >= max_matches or files_scanned >= max_files:
        footer.append(
            "[Grep limits] Output or match/file budget was reached; narrow pattern, path, or glob_pattern "
            "or raise max_matches / max_files."
        )
    body = "\n".join(lines_out)
    return body + "\n" + "\n".join(footer)


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
        new_text, n_subs = re.subn(pattern, replacement, text, count=count)
        if n_subs == 0:
            return (
                f"No matches in {path} for the given regex (file left unchanged). "
                "Patterns are full regular expressions (not plain text): match indentation, "
                "newlines (\\n), and escape metacharacters like . ^ $ * + ? ( ) [ ] {{ }} | \\. "
                "Use read_file on this path first and copy the exact substring to match."
            )
        with open(path, "w") as f:
            f.write(new_text)
        if n_subs == 1:
            return f"Replaced 1 occurrence in {path}."
        return f"Replaced {n_subs} occurrences in {path}."
    except Exception as e:
        return f"Replace error: {e}"


def call_python(code, globals=None):
    code = _scalar_to_str(code, "")
    reject = call_python_code_rejected_reason(code)
    if reject:
        return (
            "Exec error: not valid Python source (call_python only runs Python, not shell scripts or prose). "
            f"{reject}. For letters, essays, or files use write_file, "
            'or answer with {"action":"answer","answer":"..."}.'
        )
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


def run_command(command, cwd: Optional[str] = None):
    command = _scalar_to_str(command, "")
    try:
        cwd0 = None
        if isinstance(cwd, str) and cwd.strip():
            cwd0 = os.path.abspath(os.path.expanduser(cwd.strip()))
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=cwd0,
        )
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return f"Command error: {e}"


def _settings_get_bool(settings: Any, group: str, key: str, default: bool) -> bool:
    if settings is None:
        return default
    try:
        return bool(settings.get_bool((group, key), default=default))  # type: ignore[attr-defined]
    except Exception:
        return default


def _response_looks_like_pdf(resp: requests.Response, body: bytes) -> bool:
    ct = ""
    try:
        hdr = getattr(resp, "headers", None)
        if hdr is not None:
            ct = (hdr.get("Content-Type") or "").lower()
    except Exception:
        ct = ""
    if "application/pdf" in ct:
        return True
    sample = body[:8]
    return sample.startswith(b"%PDF")


def _pdf_bytes_to_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "(PDF text extraction unavailable: pypdf is not installed.)"
    try:
        reader = PdfReader(io.BytesIO(data))
        parts: list[str] = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        return "\n\n".join(parts).strip()
    except Exception as e:
        return f"(Could not parse PDF: {e})"


def _fetch_single_page(url: str, *, settings: Any = None) -> str:
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
        raw_bytes = getattr(resp, "content", b"") or b""
        prefix = f"Fetched URL: {url}\nFinal URL: {final_url}\n\n"
        if attempt == 1:
            prefix = "[After automatic retry] " + prefix

        if _response_looks_like_pdf(resp, raw_bytes):
            if _settings_get_bool(settings, "agent", "fetch_page_pdf_to_text", True):
                pdf_text = _pdf_bytes_to_text(raw_bytes)
                if not pdf_text.strip():
                    pdf_text = (
                        "(No extractable text from this PDF; it may be image-only or encrypted.)"
                    )
                else:
                    if len(pdf_text) > FETCH_PAGE_BODY_TEXT_MAX:
                        pdf_text = pdf_text[:FETCH_PAGE_BODY_TEXT_MAX].rstrip() + "\n\n[Text truncated…]"
                return prefix + "Content-Type: PDF (extracted text)\n\n" + pdf_text
            return (
                prefix
                + "Content-Type: PDF (binary). Text extraction is disabled "
                "(agent.fetch_page_pdf_to_text=false). Enable it to receive extracted text.\n"
            )

        raw_html = resp.text or ""
        title, excerpt = readability_excerpt_from_html(
            raw_html, url=final_url, max_chars=FETCH_PAGE_BODY_TEXT_MAX
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


def fetch_page(url_or_params: Union[str, dict, None] = None, *, settings: Any = None) -> str:
    """
    Fetch one or more http(s) URLs.

    - Pass a string URL (single fetch; used by search_web helpers).
    - Pass a parameters dict with ``url`` and/or ``urls`` (non-empty strings).
    """
    urls: list[str]
    if isinstance(url_or_params, str):
        u = url_or_params.strip()
        urls = [u] if u else []
    elif isinstance(url_or_params, dict):
        urls = normalize_fetch_urls(url_or_params, scalar_to_str_fn=_scalar_to_str)
    elif url_or_params is None:
        urls = []
    else:
        urls = []

    if not urls:
        return (
            "Fetch error: no URL given. "
            "Provide parameters.url (string) or parameters.urls (non-empty array of http/https URLs)."
        )
    if len(urls) > FETCH_PAGE_MAX_URLS_PER_CALL:
        return (
            f"Fetch error: at most {FETCH_PAGE_MAX_URLS_PER_CALL} URLs per fetch_page call "
            f"(got {len(urls)}). Split into multiple tool calls."
        )

    parts_out: list[str] = []
    for i, u in enumerate(urls, start=1):
        block = _fetch_single_page(u, settings=settings)
        if len(urls) > 1:
            header = f"=== Page {i}/{len(urls)} ===\n"
            parts_out.append(header + block)
        else:
            parts_out.append(block)
    return "\n\n---\n\n".join(parts_out)


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

    def _fp(u: str) -> str:
        return fetch_page(u, settings=settings)

    return websearch.search_web(
        str(query or ""),
        params=params,
        settings=settings,
        fetch_page=_fp,
    )


def search_web_fetch_top(query, params: Optional[dict] = None, *, settings=None) -> str:
    from . import websearch

    def _fp(u: str) -> str:
        return fetch_page(u, settings=settings)

    return websearch.search_web_fetch_top(
        str(query or ""),
        params=params,
        settings=settings,
        fetch_page=_fp,
    )

