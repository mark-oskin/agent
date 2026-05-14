---
title: Code pipeline extension (/code)
description: Designer → coder → reviewer → tester pipeline loaded from extensions/code.py; TUI multi-lane vs single-lane.
---

# Code pipeline extension (`/code`)

Load with:

```text
/load extensions/code.py
```

Use an **absolute path** if your session cwd is not the repository root. In **agent_tui**, you can pin the pipeline to the **current lane only** with:

```text
/load extensions/code.py --single_lane
```

That skips post-load **`/fork_background`** helper lanes when the host exposes fork/delegate hooks; the pipeline uses `execute_line` on the current session.

---

## What it does

High-level flow (from the module docstring):

1. **Multi-lane (`agent_tui.py`):** When `python_fork_background_agent` and `python_delegate_line` are both set, the extension can fork background lanes (designer, coder, reviewer, tester) and delegate steps — unless **`--single_lane`** was used on `/load`.

2. **Single-lane (plain `agent` REPL or TUI with `--single_lane`):** The whole pipeline runs on the **current** conversation via `execute_line` (no `/fork` for pipeline steps).

3. **Boot strings** apply only to forked lanes. Per-step prompts include workspace (`session_cwd`), rubric guidance, and a shared `---PIPELINE---` verdict block.

4. **Limits** default inside the module; optional overrides via `/set extensions code_pipeline set <key> <value>` then `/set save` merge on top (prefs id `code_pipeline` under `session.settings.extensions`).

5. **Git snapshot:** Each step can receive an orchestrator snapshot (`git status -sb`, `git diff --stat`, capped `git diff`) when git is available in `session_cwd`.

---

## `/code` usage

Leading flags (any combination, before the description):

| Flag | Effect |
|------|--------|
| `--skip_design` | Use the request as the plan; after a **tester** failure, run **design** to revise. |
| `--skip_review` | Skip the reviewer step. |
| `--skip_test` | Skip the tester step. |

Example:

```text
/code --skip_design --skip_test implement logging in server.py
```

Only **one** pipeline run at a time; a second `/code` while busy returns a short “already running” message.

---

## Load documentation in the REPL

This file implements **`describe_repl_load_options()`** for `/load extensions/code.py --help` (or `/load info extensions/code.py`). Summary of load flags:

- **`--single_lane`** — When the host exposes fork/delegate hooks, run the `/code` pipeline on the current session only (no `/fork_background` helper lanes). Ignored if the host is already single-lane.

---

## See also

- [Extension settings (`/set extensions`)](extension-settings.md) — persist `code_pipeline` and other `extensions.*` keys
- [Settings (`/set`, `/settings`)](settings-repl.md) — broader prefs (`save`, `tools`, …)
- [REPL extensions](repl-extensions.md)
- [README — Multi-agent Textual UI](../README.md) (section **Multi-agent Textual UI**)
