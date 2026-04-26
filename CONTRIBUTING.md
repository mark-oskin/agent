# Contributing

## Setup

Install `uv` and sync dependencies:

```bash
brew install uv
uv sync
```

## Tests

```bash
uv run pytest
```

## Style / expectations

- Prefer adding or updating tests for behavior changes.
- Keep the REPL UX simple and discoverable (`/help`, `/settings …`).

