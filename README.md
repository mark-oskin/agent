# agent

Local-first CLI agent that talks to **Ollama** (default) and can optionally use a **hosted OpenAI-compatible** endpoint.

It runs as:
- **Interactive REPL** (default)
- **One-shot CLI** (pass a question on the command line)

## Install

This repo is set up for [`uv`](https://github.com/astral-sh/uv).

```bash
brew install uv
uv sync
```

## Run

Interactive:

```bash
./agent.py
```

One-shot:

```bash
./agent.py "summarize this repo"
```

Help:

```bash
./agent.py --help
```

## Requirements

- **Python**: 3.9+
- **Ollama** running locally (default): set `OLLAMA_HOST` if it is not `http://localhost:11434`

## Configuration

Settings are persisted in **`~/.agent.json`** and edited from the REPL:

- `/settings ollama show|keys|set|unset`
- `/settings openai show|keys|set|unset`
- `/settings agent show|keys|set|unset`
- `/settings save`

To use a different config file:

```bash
./agent.py --config ./my-agent.json
```

### Thinking / streaming thinking (Ollama)

- `/settings thinking on|off`
- `/settings thinking level low|medium|high`
- `/settings enable stream_thinking` / `/settings disable stream_thinking`

Note: for `gpt-oss:*` models, `think` requires a level; if thinking is enabled without a level, the agent defaults it to `"medium"`.

## Running tests

```bash
uv run pytest
```

## License

TBD (add a `LICENSE` file).

