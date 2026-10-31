# maid_harness

The harness env below the Maid (`../../docs/ARCHITECTURE.md` §5). It must not import `maid_agent`.

```sh
uv run pytest
```

- `core.py`: `Harness`, the supervisor. One task record and one worker per heavy request, plus confirmation and cancellation. A run ends as result (completed), pause (`needs_input` / `needs_permission`, resumed by `answer()` / `grant()`) or neither (failed). `follow_up()` continues a finished task's session.
- `codex_runner.py`: `CodexRunner.spawn()` starts a detached supervisor per run. `quick` (live web search) and `work` profiles, the filesystem permission profile (reads and writes only in its folder) plus per-task grants, resume by thread id, a stripped environment, workspace pruning.
- `subagent.py`: `python -m maid_harness.subagent run` (supervisor: clean home, Codex, events, heartbeat, timeout, settle) and `… mcp` (the tools over stdio). Runs survive the Maid.
- `subagent_mcp.py`: the sub-agent tools (`report_progress`, `ask_user`, `request_permission`, `submit_result`), bound to one task, writing to SQLite.
- `codex_home.py`: the clean, isolated `CODEX_HOME`, rebuilt from `~/.codex` per run (provider and login only) with a safe fingerprint.
- `capabilities.py`: what the env can do right now. The Maid's 「できること」 and tool list are generated from it.
- `registry.py`: SQLite task registry and event log (the task dataset).
- `inbox.py`: event inbox with dedupe and TTL, delivered to subscribers.
- `models.py`: `TaskRecord`, `TaskEvent`, routes and statuses.
- `workers.py`: worker protocol, the deterministic demo worker, and `FakeSubAgent` (computer-use tasks until P6, or `runner = "fake"`).
- `mcp_server.py`: the older task-level MCP stdio server (`maid-agent harness-mcp`). Sub-agents use `subagent_mcp.py` instead.
- `tests/fake_codex.py`: stands in for `codex` in tests, with the same argv, events and MCP calls.
- `reporters/screen/`: main-display observer (settle, then score the change with SigLIP2, a tile grid, app switches, a per-app baseline and "seen recently"), calibration logger, and local vision. It is still driven by `maid_agent/proactive.py` and becomes a proper reporter in P4.
