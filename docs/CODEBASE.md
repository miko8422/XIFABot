# Codebase guide

How the code is laid out, what each module does, and the idea behind it. Design decisions and the phase plan are in `ARCHITECTURE.md`. This file is the map. State: P3.5 (2026-10-31).

## 1. The three packages and the one rule

```text
              you (voice, web UI)
                      │
┌─────────────────────▼──────────────────────┐
│ maid_agent    who she is and what she does  │  persona, routing via tools, speech arbiter,
│               (depends on both below)       │  conversation log, serve wiring, web UI
└───────┬──────────────────────────┬─────────┘
        │ library                  │ Python API (submit/answer/cancel/…) + events
┌───────▼────────┐      ┌──────────▼──────────────────────────┐
│ maid_voice     │      │ maid_harness                         │
│ low-latency    │      │ tasks, inbox, capabilities,           │
│ voice loop     │      │ sub-agent runner (Codex), reporters   │
└────────────────┘      └──────────────────────────────────────┘
```

**Rule: dependencies point down only.** `maid_voice` knows nothing about tasks, and `maid_harness` knows nothing about the Maid. `maid_agent` plugs into both through narrow seams:
- **`TurnHooks`** in `maid_voice/controller.py`: `plan` / `ack` / `run` / `remember`. The voice loop asks the Maid for extra context and tools, and hands back tool calls and new history.
- **The harness API plus inbox events**: the Maid calls `submit`, `answer`, `grant`, `follow_up` and `cancel`, and listens to events.

This is why each layer can be tested alone (`uv run pytest` runs 78 tests without a mic, models or network) and replaced alone. For example, the Codex runner can be swapped without touching voice code.

## 2. `maid_voice`: the voice loop (about 1,300 lines)

The oldest and most tuned part. It turns speech into her spoken reply at about 0.94 s p50 to first sound.

| Module | What it does | Why it is built this way |
| --- | --- | --- |
| `app.py` | `maid-voice serve` / `bench`. Builds ASR, LLM, TTS and player, runs the websocket web UI, and accepts an *extension* (`maid_agent` plugs in here) | One gateway process; extensions add behavior without forking it |
| `controller.py` | `TurnController`: one turn = ASR → LLM stream → clause segmenter → streaming TTS → playback. Owns generation ids, cancel, speculative turns, `say()` (fixed line), `notify()` (an event put into her own words), `hooks` | Only one generation may own playback; stale audio is dropped by id. Tool calls run only after a turn is *committed*, so a dropped hands-free guess never starts a task |
| `handsfree.py`, `vad.py` | Energy VAD with adaptive noise floor and speculative start at 250 ms of silence, commit at 600 ms | The work starts before you finish, which hides ASR+LLM latency |
| `asr.py` | oMLX Qwen3-ASR client, language pinned to `ja` | Auto-detect misread Japanese as Korean |
| `llm.py` | OpenAI-compatible streaming client (DeepSeek Flash): `context=`, `tools=`, streamed tool calls | The *one* reply call per turn also does routing (no second call) |
| `segmenter.py` | Cuts the token stream into speakable clauses, with an early first clause | TTS can start on the first clause |
| `tts.py` | oMLX Qwen3-TTS streaming, fine-tuned ナツメ voice | — |
| `audio.py` | Capture and playback (`PlaybackQueue` keyed by generation) | Barge-in and stop are instant and never leak old audio |
| `persona.py` | Role card plus expression bank (e5-small retrieval of her real lines) → system prompt | "Sounds like her" comes from her own 3,669 lines, not adjectives |
| `metrics.py` | `TurnTrace`: every timestamp, text, tool call and cost → `turns.jsonl` + `summary.csv` | Every latency claim is measured |
| `config.py`, `config/default.toml` | All settings; secrets only from the repo-root `.env` | `[agent]` in the same file configures the Maid |
| `scripts/`, `sft/` | Probes, benchmarks, voice fine-tuning | Kept for reproducing the voice decisions |

## 3. `maid_harness`: where work and information live (about 1,500 lines)

| Module | What it does | Why |
| --- | --- | --- |
| `core.py` | `Harness`. One task record per task. `settle()` ends a run as **result** → `completed`, **pause** → `needs_input` / `needs_permission`, or **neither** → `failed`. `answer` / `grant` / `follow_up` resume the *same* sub-agent session. In-process and detached workers; `watch()` polls other processes' events, limits concurrency and notices dead supervisors; `recover()` reconciles after a restart | "Exit without submit_result = failed" makes false completion impossible by construction |
| `registry.py` | SQLite tasks plus events: the **task dataset**. Atomic merging updates across processes, event cursors, keyword search | Durable, inspectable, one file, several writers |
| `inbox.py` | Normalized events with dedupe and TTL, delivered to subscribers (the arbiter, the UI) | Delivery, not reasoning; never calls an LLM |
| `models.py` | `TaskRecord` (with `.goal` = latest follow-up), `TaskEvent`, statuses | — |
| `capabilities.py` | What the env can do right now (`live` / `simulated` / `off`, tools, notes) | Her 「できること」 and her tool list are **generated** from this, so she can't claim or deny abilities wrongly |
| `codex_runner.py` | `CodexRunner.spawn()` starts a detached supervisor per run. Profiles `quick` (web search) and `work`. The filesystem permission profile plus per-task grants, the command line (new or `resume` by thread id), the stripped environment, workspace pruning | Codex is the agent loop; we only supervise it |
| `subagent.py` | The two processes a run is made of. `run` (supervisor): builds the clean home, runs Codex, saves events, heartbeats, enforces the timeout, settles the run in SQLite. `mcp`: the sub-agent's tools over stdio | Runs survive the Maid; SQLite is the only channel |
| `subagent_mcp.py` | The four tools (`report_progress`, `ask_user`, `request_permission`, `submit_result`), bound to one task by their arguments | A sub-agent can reach only its own task. Pauses return at once, and the answer comes back as the next message |
| `codex_home.py` | The sub-agents' clean `CODEX_HOME`, rebuilt from `~/.codex` per run: provider and login only, `auth.json` linked, unneeded features off; a safe fingerprint (provider, endpoint host, model, account hash) | Isolation from your real Codex, and cc-switch changes still apply |
| `workers.py` | Worker protocol, `DemoWorker`, `FakeSubAgent` (computer-use until P6) | — |
| `mcp_server.py` | The older task-level MCP stdio server (`maid-agent harness-mcp`) | Not used by sub-agents |
| `reporters/screen/` | Screen observer (settle, then score the change with SigLIP2 + tiles + app baseline), calibration log, local vision (Qwen3.6 on oMLX) | Pixels never leave the Mac. It becomes a proper reporter in P4 |

## 4. `maid_agent`: the Maid (about 1,600 lines)

| Module | What it does | Why |
| --- | --- | --- |
| `agent.py` | `MaidAgent`, the `serve` extension. Wires harness, runner, sub-agent tools, capabilities, turns, arbiter, conversation and screen together; feeds the UI | The only place that knows about everything |
| `turns.py` | `MaidTurns` = the `TurnHooks`. `plan` builds 「できること」 + rules + state block (+ time-gap note) and the tool list; `run` executes tool calls and reports any that changed nothing; `remember` logs the turn and runs the correction pass when she claimed an action without a tool call; keyword backstops for cancel and yes/no; daily token budget | Routing inside the existing reply call (§4.2): no planner, no extra latency |
| `tools.py` | Tool schemas (`delegate`, `look_at_screen`, `answer_task`, `follow_up_task`, `cancel_task`, `confirm_task`, `read_task`, `search_tasks`, `search_conversation`), the Japanese rules, `capability_block`, `state_block` (3-day window, newest details), `usage_since` | — |
| `arbiter.py` | Speech arbiter v1: task news waits for a pause plus a quiet gap, is merged, heartbeats long tasks, mute, requeue once; logs every decision | One place decides when she speaks unprompted |
| `conversation.py` | Conversation log in `runs/maid/memory.sqlite3`, restore on startup, time-gap note | The Maid owns memory; the harness owns the task dataset |
| `proactive.py` | Screen → her: gate → vision → one DeepSeek decision → `say()`. Also `look()` / `facts()` for the reply call | Rebuilt as a reporter in P4 |
| `router.py` | The keyword router: fallback (`routing = "keyword"`), test oracle, and backstop phrases | — |
| `runtime.py`, `demo.py` | The text-only v0.1 Maid layer and `maid-agent demo` | Kept for the demo and tests |
| `persona/` | Role card `natsume.v2.toml` (v1 kept) and the expression bank | Versioned; never rewritten by conversation |
| `static/index.html` | Web UI: turns with latency bars, Tasks panel and capabilities, screen panel | — |
| `scripts/probe_delegate.py` | Routing and latency probe against the real LLM | Gate evidence for prompt changes |

## 5. What happens at runtime

**A turn:** mic → VAD (speculative start) → ASR → `MaidTurns.plan` (abilities + state) → DeepSeek streams text **and** maybe a tool call → clauses → TTS → speaker. The turn is committed, and then the tool call runs (`delegate` → `Harness.submit`) and the history goes to `conversation.remember`.

**A task:** `submit` → `CodexRunner.spawn()` starts a detached supervisor → it runs `codex exec` in `runs/tasks/<id>/` with the clean home → Codex works in a loop (many model calls, shell, search) and calls our MCP tools, which write to SQLite → the supervisor settles the run → `Harness.watch()` picks up `task.completed` → the arbiter waits for a pause → `ctl.notify()` → she says the summary in Japanese. A question or permission request pauses the task; your answer resumes the same Codex thread. If the Maid is restarted meanwhile, the run carries on and she reports it when she is back.

## 6. Where data lives (all under `packages/*/runs/`, git-ignored)

| Path | What |
| --- | --- |
| `maid_voice/runs/<time>-agent/turns.jsonl` | Every turn: texts, timings, tool calls, cost |
| `maid_agent/runs/maid/memory.sqlite3` | Her conversation log (memory bank in P5) |
| `maid_agent/runs/harness/tasks.sqlite3` | Task dataset: every task, every event |
| `maid_agent/runs/tasks/<id>/` | Sub-agent workspace: its files (pruned after 7 days), `events.jsonl` (full Codex stream), `runs.jsonl` (each run's provider/account fingerprint and grants), `stderr.log`, `supervisor.log` |
| `maid_agent/runs/codex_home/` | The sub-agents' clean CODEX_HOME: generated `config.toml`, a link to your `auth.json`, and their Codex sessions (never mixed with yours) |
| `maid_agent/runs/agent/<date>/` | `tool_calls.jsonl`, `corrections.jsonl` (the said-but-didn't pass), `arbiter.jsonl`, `decisions.jsonl` (screen) |
| `maid_agent/runs/screen/` | Screen calibration logs |

## 7. Tests

`uv run pytest` (78 tests, about 16 s): voice controller with fake models, harness and runner (with `maid_harness/tests/fake_codex.py` standing in for Codex, run through the real supervisor and stdio MCP processes: same argv, events and MCP calls, including a Maid dying mid-run), and the Maid per phase (`test_p1.py`, `test_p2.py`, `test_p3.py`, which also covers P3.5). Real-model checks are scripts (`probe_delegate.py`) and the rehearsals recorded in `ARCHITECTURE.md` §9.
