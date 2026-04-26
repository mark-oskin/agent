# Unfamiliar CLI / program — grounding runbook

Use this pattern when the model (or user) is not reliably trained on a specific tool. **Do not invent flags, config keys, or file paths from memory.**

## 1) Discover from the environment first

1. **Version / build** (if available): e.g. `<program> --version`, `<program> version`, or a subcommand from `--help`.
2. **Help surface**: `<program> --help` and, if applicable, `<program> <subcommand> --help`.
3. **Man page** (Unix): `man <program>` when the program is a standard system utility (only if appropriate for the OS/session).

Capture stdout/stderr; treat that text as **authoritative** over training data.

## 2) Establish scope

- What is the **inputs** (files, env vars, URLs) and the **outputs** (files, exit code, stdout)?
- Is the session **read-only** vs **mutating** (writes DB, deletes files, sends network traffic)? For mutating tasks, prefer a **dry-run** or **preview** flag if the tool supports it.

## 3) Minimal reproduction

- Run the **smallest** command that proves the tool works (e.g. print config, list resources, validate-only).
- Only then expand to the full task.

## 4) Grounding with docs (when online)

- Prefer **official** documentation: project site, versioned docs, or the repository’s README in the release tag the user uses.
- Use `fetch_page` on **exact URLs**; do not guess doc URLs.
- If docs conflict with `--help`, trust **`--help` for that installed version**.

## 5) Safety

- Do not pass **secrets** on the command line where they may be logged; use env vars or files with safe permissions.
- Avoid `eval`, blind `curl | sh`, and `--privileged` without an explicit user request and risk note.
- For destructive operations (delete, drop, overwrite), require an explicit user goal and, when possible, a backup or confirmation step.

## 6) Done when

- You can **re-run** the same command sequence and get a consistent result, or you have left a **script/README** that records the exact commands and version used.
