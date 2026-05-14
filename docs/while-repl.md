---
title: /while REPL loop
description: Model-judged condition loop with quoted condition and one or more body prompts; optional --max cap.
---

# `/while` — condition loop

`/while` runs a **C-style loop** driven by the **primary** LLM as a strict boolean judge over the current transcript. It is a REPL-only feature (not a model tool).

---

## Syntax

```text
/while [--max N] <condition> do <body>
```

- **`<condition>`** — Natural-language text (usually **quoted** with `shlex`, so use double or single quotes; nest the other kind for quotes inside).
- **Literal word `do`** — Separates condition from body (required).
- **`<body>`** — One or more **comma-separated** quoted prompts. Each prompt is one **quoted** phrase; commas delimit separate REPL turns. You can write `"a" , "b"` or `"a", "b"` if your terminal glues commas.

Optional **`--max N`** caps **condition checks** (default **50**). Each iteration: one judge call, then (if the judge returns **1**) every body prompt runs in order as its own user turn.

---

## Judge semantics

The judge uses a fixed system prompt (see `WHILE_JUDGE_SYSTEM` in `agentlib/repl/while_cmd.py`). It must reply with **exactly one character**:

| Output | Meaning |
|--------|---------|
| **`1`** | Condition is **TRUE** — **stay** in the loop: run the `do` body (all steps), then **re-check** the condition. |
| **`0`** | Condition is **FALSE** — **exit**: do **not** run the body for this iteration; stop the loop. |

If the model’s reply is ambiguous, the parser takes the first **`0`** or **`1`** it finds on the first line; otherwise it treats the condition as **false** (safe exit).

The judge sees a **truncated excerpt** of recent messages (`while_conversation_excerpt_for_judge`, default ~12k chars).

**Backend:** Uses the same routing as a normal primary call — **hosted** chat or **Ollama** plaintext, depending on `primary_profile` (`call_while_condition_judge` in `while_cmd.py`).

**Verbose:** At `verbose >= 1`, the raw judge reply is echoed as `[/while judge] …`.

---

## Cancellation

**Ctrl-C** during the condition check or during a body step aborts `/while` and prints `[Cancelled]`.

---

## Examples

```text
/while "pytest is still failing" do "fix from output and run pytest"
/while 'work remains' do 'step A', 'step B', 'step C'
/while --max 10 'server not yet returning 200' do 'patch and curl until OK'
```

Bare **`/while`**, **`/while help`**, or **`/while --help`** prints the same usage block as in the REPL.

---

## See also

- [Settings (`/set`, `/settings`)](settings-repl.md)
- [README — Interactive REPL](../README.md#interactive-repl-important-commands)
- Parser: **`agentlib/repl/while_cmd.py`**
- Session driver: **`agentlib/session.py`** (`_cmd_while`)
