# Contributing

## Setup

Install `uv` and sync dependencies:

```bash
brew install uv
uv sync
```

To install the `agent` CLI on your `PATH` for day-to-day use (isolated env managed by uv):

```bash
uv tool install .
```

See the README **Install** section for reinstall/uninstall notes.

## Tests

```bash
uv run pytest
```

## Style / expectations

- Prefer adding or updating tests for behavior changes.
- Keep the REPL UX simple and discoverable (`/help`, `/settings …`).
- Documentation is part of the change:
  - Update **`README.md`** for user-facing behavior.
  - Update **`docs/*.md`** when REPL or tool behavior changes (see **`docs/index.md`**).
  - Update **`DEVELOPERS.md`** for architecture, embedding, and extension internals.
  - **Website** (nested **`website/`** repo): run **`python3 _sync_docs_from_repo.py`** after editing synced **`docs/`** pages, then **`npm run build`** and **`python3 deploy_ftp.py`** to publish (see **`website/README.md`**).

