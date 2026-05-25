# Native Ollama tools — Phase 1 rollback point

Phase 1 adds optional Ollama native `tools` API support (JSON fallback unchanged by default).

**Baseline commit (before Phase 1):**

```
612274dcd175cba56924e0c27d30bd750b332804
612274d Show Draft from full answer snapshots instead of accumulated deltas.
```

To revert Phase 1 only (after it is committed on top):

```bash
git revert <phase1-commit-sha>   # or reset if not pushed
# full restore to pre-Phase-1 tree:
git checkout 612274dcd175cba56924e0c27d30bd750b332804 -- .
```

Default behavior is `agent.tool_call_mode: native`. Use `/set tool_call_mode json` to revert to JSON-in-content tool calls.
