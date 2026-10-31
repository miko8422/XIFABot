# maid_harness

The harness env below the Maid (`../../docs/ARCHITECTURE.md` §5). It must not import `maid_agent`.

```sh
uv run pytest
```

- `core.py`: `Harness`, the supervisor. One task record and one worker per heavy request, plus confirmation and cancellation.
- `registry.py`: SQLite task registry and event log (the task dataset).
- `inbox.py`: event inbox with dedupe and TTL, delivered to subscribers.
- `models.py`: `TaskRecord`, `TaskEvent`, routes and statuses.
- `workers.py`: worker protocol and the deterministic demo worker. The Codex runner comes in P3.
- `mcp_server.py`: dependency-free MCP stdio server. In P3 it becomes the sub-agent → harness channel.
- `reporters/screen/`: main-display observer (settle, then score the change with SigLIP2, a tile grid, app switches, a per-app baseline and "seen recently"), calibration logger, and local vision. It is still driven by `maid_agent/proactive.py` and becomes a proper reporter in P4.
