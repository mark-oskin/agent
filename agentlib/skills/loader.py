import json
import os
from typing import Optional


def safe_path_under_dir(base_dir: str, relpath: str) -> Optional[str]:
    """
    Join base_dir with relpath and return the path only if it stays under base_dir.
    Prevents path traversal in reference_files.
    """
    base_dir = os.path.abspath(base_dir)
    if not relpath or not isinstance(relpath, str):
        return None
    rp = relpath.strip()
    if not rp or ".." in rp.split(os.sep):
        return None
    cand = os.path.normpath(os.path.join(base_dir, rp))
    if cand != base_dir and not cand.startswith(base_dir + os.sep):
        return None
    return cand


def expand_skill_artifacts(skills_dir: str, meta: dict, base_prompt: str) -> str:
    """
    Append bundled reference file bodies and optional doc URLs / grounding commands
    to the skill prompt. reference_files are paths **relative to the skills_dir**
    (e.g. "references/helm_cheatsheet.md").
    """
    parts: list[str] = []
    if (base_prompt or "").strip():
        parts.append((base_prompt or "").strip())
    ref_files = meta.get("reference_files")
    if isinstance(ref_files, list) and ref_files:
        for rel in ref_files:
            if not isinstance(rel, str) or not str(rel).strip():
                continue
            abs_p = safe_path_under_dir(skills_dir, rel.strip())
            if abs_p is None or not os.path.isfile(abs_p):
                parts.append(
                    f"--- Reference file (missing or invalid path under skills dir): {rel} ---\n"
                )
                continue
            try:
                with open(abs_p, "r", encoding="utf-8") as f:
                    body = f.read()
            except OSError as e:
                body = f"(unreadable: {e})"
            parts.append(
                f"--- Bundled reference file: {rel} ---\n" + (body or "").rstrip() + "\n"
            )
    urls = meta.get("doc_urls")
    if isinstance(urls, list) and urls:
        lines = [str(u).strip() for u in urls if isinstance(u, str) and u.strip()]
        if lines:
            parts.append(
                "--- External docs (fetch with fetch_page when online; do not trust memory alone) ---\n"
                + "\n".join(f"- {u}" for u in lines)
                + "\n"
            )
    gcmds = meta.get("grounding_commands")
    if isinstance(gcmds, list) and gcmds:
        lines = [str(c).strip() for c in gcmds if isinstance(c, str) and c.strip()]
        if lines:
            parts.append(
                "--- Suggested grounding commands (run small steps; capture output) ---\n"
                + "\n".join(f"- `{c}`" for c in lines)
                + "\n"
            )
    return "\n\n".join(p for p in parts if p and str(p).strip()).strip()


def load_skills_from_dir(dir_path: str) -> dict:
    """
    One skill per JSON file: skills/<id>.json
    Optional keys: description, triggers, tools, prompt, workflow,
    reference_files (list of paths under skills_dir), doc_urls, grounding_commands.
    """
    out: dict = {}
    if not os.path.isdir(dir_path):
        return out
    for fn in sorted(os.listdir(dir_path)):
        if not fn.endswith(".json") or fn.startswith("."):
            continue
        name, _ = os.path.splitext(fn)
        name = (name or "").strip()
        if not name:
            continue
        path = os.path.join(dir_path, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        meta = raw
        base_prompt = (
            (meta.get("prompt") or "").strip()
            if isinstance(meta.get("prompt"), str)
            else ""
        )
        prompt = expand_skill_artifacts(dir_path, meta, base_prompt)
        tr = meta.get("triggers")
        if not isinstance(tr, list) or not tr:
            tr = [name]
        triggers = [str(t).strip() for t in tr if str(t).strip()]
        tools = meta.get("tools")
        if tools is not None and not isinstance(tools, list):
            tools = None
        workflow = meta.get("workflow")
        if workflow is not None and not isinstance(workflow, dict):
            workflow = None
        ref_files = meta.get("reference_files")
        if ref_files is not None and not isinstance(ref_files, list):
            ref_files = None
        doc_u = meta.get("doc_urls")
        if doc_u is not None and not isinstance(doc_u, list):
            doc_u = None
        gcmds = meta.get("grounding_commands")
        if gcmds is not None and not isinstance(gcmds, list):
            gcmds = None
        out[name] = {
            "id": name,
            "path": path,
            "description": (meta.get("description") or "").strip()
            if isinstance(meta.get("description"), str)
            else "",
            "triggers": triggers,
            "tools": tools,
            "prompt": prompt,
            "workflow": workflow,
            "reference_files": ref_files,
            "doc_urls": doc_u,
            "grounding_commands": gcmds,
        }
    return out

