# Maid Agent — Architecture and Plan

Status: draft for discussion. Nothing below the "Decided" list is final until the open questions in §10 are answered.
Date: 2026-10-31
Owner: Alan
Replaces: `maid_agent_MASTER_GOAL.md`, `PRODUCT_MODEL.md`, `maid_agent_REVISED_ARCHITECTURE.md`, `maid_voice_cascade_GOAL.md` (deleted; the original brief is kept in Appendix A).
Still valid as module references: `maid_voice_cascade_ARCHITECTURE.md` (voice runtime internals), `maid_agent_RESPONSE_DESIGN.md` §3–§4 (role card, expression bank, screen scoring).

## 1. What we are building

A personal **Maid Agent**: an always-available companion with the ナツメ persona that talks with the user in Japanese with low latency, reacts instinctively to what the user says and to what changes on the system, and hands anything heavy to sub-agents that work in the background. When a sub-agent finishes, needs an answer, or fails, the Maid tells the user in her voice (later also as a pop-up card).

Two layers, nothing else:

```text
                          user
               voice  (later: pop-up cards, controls)
                          │
┌─────────────────────────▼──────────────────────────────┐
│ MAID AGENT  — fast, light, owns the relationship        │
│  voice runtime (exists) · persona · router/difficulty   │
│  speech arbiter · conversation log · memory bank        │
└──────────┬───────────────────────────────▲─────────────┘
   submit / answer / cancel / query        │ events + task summaries
┌──────────▼───────────────────────────────┴─────────────┐
│ HARNESS ENV  — where information and heavy work live    │
│  event inbox · task registry + dataset · supervisor     │
│  sub-agents: one per task (search, code, computer use)  │
│  reporters: desktop / OS / screen / email / internet    │
└────────────────────────────────────────────────────────┘
```

Botmux and MikuDance are **not part of this project**. MikuDance is the author's own earlier system and is used only as a reference for patterns (one supervised executor per session, narrow digests instead of reading worker state, one-shot completion briefs, durable SQLite registry, memory kept separate from session records). We build our own harness.

## 2. Decided

1. The Maid Agent **is** the low-latency voice system. It will grow features, but it stays light: it chats, reacts, routes, reports. It never does heavy work itself.
2. **No planner in the Maid.** Planning is the sub-agent's job. The Maid only routes and judges difficulty.
3. **One task, one sub-agent.** Sub-agents live in the harness env.
4. Desktop/system information comes from **reporters**: an existing open-source adapter if one fits, otherwise our own, and a reporter sub-agent only when interpretation needs a model.
5. **Computer use is a sub-agent**, not a Maid capability.
6. If a task is unfinished, the Maid says so and gives the latest real state. She never invents progress or claims completion that a sub-agent did not report.
7. The **memory bank belongs to the Maid**. The harness keeps the full **task dataset**. Each finished task produces a short **summary** that feeds both the Maid's spoken report and the memory bank.
8. The event inbox is the loop that wakes the Maid (and delivers messages to sub-agents). It is delivery, not reasoning.
9. Routing is **intent detection through a `delegate` tool inside the existing reply call** (§4.2). (2026-10-31)
10. Sub-agents run on **Codex** (CLI or SDK), chosen because it is open source and can be slimmed down later for faster task processing. Codex is more than this demo needs; the runner interface keeps it replaceable. (2026-10-31) Since P3.5 they run in a **clean, isolated CODEX_HOME** rebuilt from the user's `~/.codex` before every run: provider and login only (so cc-switch changes apply), none of the user's skills, MCP servers, hooks or sessions. Model GPT-5.6-Luna at low effort. (2026-10-31)
11. Sub-agents are **sandboxed by default**. Anything beyond the sandbox goes through the Maid, who asks the user by voice to grant it (§5.5). (2026-10-31) Since P3.5 this covers **reads too**: outside its own folder a sub-agent can read only system basics, and anything else needs a voice grant. The Maid's `.env` secrets are never in a sub-agent's environment. (2026-10-31)
12. ~~Tasks run inside the Maid's process.~~ **Changed 2026-10-31 (P3.5):** sub-agents run detached and **survive the Maid**. Their supervisor writes every result to SQLite and the task folder. On restart the Maid keeps watching the ones still alive, reports the ones that finished while she was down, marks dead ones `interrupted`, and asks pending questions again. Cancel still kills them.
13. The screen observer is **rebuilt as a reporter**. It is part of the agent loop, for context, for more fluent interaction, and for proactive chat while gaming or working (§5.6). (2026-10-31)
14. Desktop activity comes from **ActivityWatch**, not our own adapter (§5.4). (2026-10-31)
15. Conversation persistence moves up to right after the voice-loop connection (§9). (2026-10-31)
16. **Language: speak Japanese or English, hear Japanese, work in English.** The Maid's reply LLM is the only language boundary, so there is no translation layer (§4.5). (2026-10-31)
17. **Gaming on the Mac first.** Any app can be marked as a game to simulate gaming during the demo. A Windows capture service comes after the Maid has its full feature set (§5.6). (2026-10-31)

## 3. Problems in the previous plan, and the fix

| Previous plan (GPT draft, 10-30) | Problem | Fix in this plan |
| --- | --- | --- |
| MikuDance Manager Bot / Botmux as the first "worker adapter" (H3) | Pulls an external Codex/Feishu system into our product | Our own sub-agent runner that launches a headless coding-agent CLI per task (§5.2) |
| Keyword router with mostly Chinese/English word lists | ASR is Japanese-only, so most real requests fall through to `direct`. Word lists don't measure difficulty | The single existing reply LLM call decides the route via a tool call, with no second call. A cheap keyword pre-filter remains as fallback (§4.2) |
| MCP as the Maid↔harness boundary | Both run in one Python process, so MCP there adds nothing | Maid↔harness is an in-process Python API. MCP is used where it helps: **sub-agent → harness** (progress, questions, results) |
| "The Maid may interrupt for normal task info" | Conflicts with the core rule of never talking over the user | Speech arbiter: never over the user, queue until a pause, merge pending updates (§4.3) |
| A separate summary step / summary agent | Extra model call per task | The sub-agent submits its own `user_summary` with the result |
| No step for conversation persistence | Chat doesn't survive a restart | Its own phase, P2, right after the voice loop is connected |
| No process/restart model | Unclear what happens to tasks when the app stops | §6 |
| Routes without difficulty | Doesn't say what kind of sub-agent to start | Difficulty tiers T0–T4 map directly to sub-agent profiles (§4.2) |

## 4. Maid Agent

### 4.1 Responsibilities

- Voice conversation through `maid_voice` (capture, VAD, ASR, LLM, TTS, barge-in), unchanged in behavior.
- Persona: role card and expression bank (`persona/`).
- Route each user turn and judge its difficulty.
- Submit, answer, cancel and query tasks through the harness API.
- Turn inbox events into speech at the right time (speech arbiter).
- Proactively start a conversation when a preset event fires or reported information changes.
- Own the conversation log and the memory bank.

### 4.2 Routing and difficulty

The direct path must stay at today's latency (about 0.94 s p50 to first sound). So routing happens **inside the reply call the Maid already makes**: the LLM gets one tool, `delegate(goal, tier, needs)`. For chitchat it just answers. For heavy work it speaks a short acknowledgement and calls the tool in the same response. The harness starts the task while the acknowledgement plays.

| Tier | Meaning | Who handles it | Example |
| --- | --- | --- | --- |
| T0 direct | Chat, persona, anything answerable from the conversation | Maid | 「今日疲れた」 |
| T1 lookup | Answerable from harness state, no new work | Maid reads registry/reporter snapshot | 「さっきのタスクどうなった？」「今何のアプリ開いてる？」 |
| T2 quick task | Short, read-only, minutes | Sub-agent, `quick` profile | Web search, a coding question |
| T3 work task | Long, may write files or run commands | Sub-agent, `work` profile, task workspace | Change a repo, run a build |
| T4 computer use | Controls GUI or apps | Sub-agent, `computer` profile, **needs confirmation** | 「Steam 開いて」 |
| clarify / refuse | Underspecified, or not allowed | Maid asks or declines | 「あれやって」 |

T1 lookups are fed to the reply call as a small state block (active tasks, latest reporter facts), so asking about a task costs no extra call.

The keyword router in `maid_agent/router.py` stays as a fallback and a test oracle, with Japanese patterns added. If tool calling turns out to add measurable latency or to be unreliable, we fall back to it (gate in P1).

### 4.3 Speech arbiter (when she speaks)

One place decides whether anything from the inbox is spoken. v1 rules, deliberately simple:

1. Never while the user is speaking. Queue until the next pause.
2. Never cut off her own reply for non-urgent news. Speak after it ends.
3. Merge everything pending into one short line (multi-task updates already merge in `maid_agent/runtime.py`).
4. Speak at the next pause: `task.completed`, `task.failed`, `task.needs_input`, confirmation requests.
5. Progress is spoken when asked (T1). Unprompted progress only as a heartbeat for long tasks, at most once per N minutes per task.
6. Reporter-driven proactive remarks: per-source rules plus a global cooldown (the existing 10 min default). The richer MaiBot-style gate from `RESPONSE_DESIGN.md` §5 is a later upgrade, driven by logged decisions.
7. Mute / do-not-disturb silences everything except what the user marks urgent.

Every decision is logged with the rule that fired.

### 4.4 Memory

All owned by the Maid, in one SQLite file:

- **Conversation log**: turns with timestamps. Restored on restart with an elapsed-time note ("brief restart" … "days"), as in `RESPONSE_DESIGN.md` §1.
- **Memory bank**: preferences, instructions, notable facts, each with source and time. User can list, correct and delete entries.
- **Task index**: task ID, user summary, outcome. Pointers into the harness dataset, not copies.

The role card is versioned configuration and is never rewritten by conversation.

### 4.5 Language

| Where | Language | Why |
| --- | --- | --- |
| User → Maid (speech) | Japanese, with English words and short English sentences | ASR pinned to `language = "ja"`. Auto-detect (`""`) turned Japanese into Korean in live use (2026-10-31). The oMLX endpoint takes one language or auto, so "Japanese or English only" would need a detect-then-retry pass |
| Maid → user (speech) | Always Japanese | The persona and the Japanese ナツメ voice. The TTS only ever receives Japanese text |
| Maid → sub-agent (`delegate` arguments) | English | The reply LLM writes the tool arguments in English in the same call where it speaks Japanese, so the translation is free |
| Sub-agent work, task dataset, `user_summary`, MCP traffic | English | Best for Codex, and consistent logs |
| Sub-agent → user (notices) | Japanese | The speech arbiter hands the English summary to the reply LLM as an event turn, and she says it in her own words. These turns are not latency-critical |
| Memory bank | English facts, quoted user wording kept as said | Language-neutral to retrieve. The conversation log keeps the original text |

The Maid's LLM already understands both languages and is already told to answer in Japanese. Translation therefore happens only inside the two LLM calls we make anyway (the `delegate` call, and the in-character notice), and adds no latency or extra layer. We don't need an English voice model.

Risks, checked in P1:
- Auto-detect costs about 20 ms over fixed `ja` (ASR experiment: 268 vs 247 ms median), but in live use it misread Japanese as Korean, so it is off. English sentences came through correctly even with `ja` pinned.
- Japanese ASR mangles English technical words (app names, file names) inside Japanese sentences. Speaking English for work requests avoids this.
- English words in her replies sound wrong in the Japanese voice. The role card's voice rules should require foreign names to be written in katakana.

### 4.6 What she knows she can do (capability list)

An agent is a model plus tools, a loop, an environment, and an accurate picture of what it can do. Without that last part she acts like a chatbot. In the first live P1 test (2026-10-31) she denied abilities she had, because her role card said weather and task state could never be checked.

So her abilities are **data, not prose**:
- Whatever wires a reporter or runner into the harness registers a `Capability` (`maid_harness/capabilities.py`): name, status (`live` / `simulated` / `off`), a one-line English description, the tools it gives her, and a note on its limits.
- Every reply call gets a generated 「できること」 block and exactly the tools of the usable capabilities (`maid_agent/tools.py`). Anything not listed, she says she can't do. Anything `simulated`, she may use but says honestly that the result isn't real.
- The role card says nothing about specific abilities (v2 removed the 「天気…確かめる手段がない」 line). The card is who she is. The capability list is what she is connected to.
- The Maid does not get a harness of her own. She is a client of the one harness, and it tells her what it offers.

Two loop speeds:
- **In-turn lookups** answer in seconds. She speaks a short line, the harness looks, and the result goes back through the speech arbiter as a follow-up line. First one: `look_at_screen` (local vision, ~5 s warm), and the latest screen observation also sits in the state block.
- **Background tasks** are sub-agents (`delegate`), with results reported later.

Chitchat stays a single call.

## 5. Harness env

### 5.1 Event inbox

One inbox with normalized events (`harness/models.py` `TaskEvent`): type, source, task ID, priority, sensitivity, dedupe key, TTL, small payload. It does dedupe, TTL expiry and persistence, then delivers to subscribers:

- **Maid** (speech arbiter): task lifecycle events, `system.changed`, `reporter.warning`.
- **Task channels** (Maid → sub-agent): `task.answer`, `task.cancel`, follow-up instructions for the same task.

The inbox never calls an LLM.

### 5.2 Sub-agent runtime

- The supervisor starts **one sub-agent process per task** from a profile (`quick`, `work`, `computer`). A profile fixes the agent CLI, model, tools, working directory, timeout and permissions.
- First runner: **Codex**, started headless (`codex exec` with its JSON event stream, or the Codex SDK) as a subprocess in a per-task workspace under `runs/tasks/<task_id>/`. Its event stream is saved to the task dataset. Exact flags, session-resume behavior and SDK language support are checked against the installed version at the start of that phase. The runner sits behind a small interface (`start`, `send`, `cancel`, `events`) so a slimmed-down Codex or another agent can replace it.
- The harness injects a small MCP server into each sub-agent with four tools:
  - `report_progress(percent, text)`
  - `ask_user(question)`: the task goes to `needs_input`, and the Maid asks by voice. The answer comes back through `task.answer`.
  - `request_permission(what, why)`: the task goes to `needs_permission` (§5.5).
  - `submit_result(user_summary, details, artifacts)`: the only way a task completes.
- Follow-ups to a running or finished task resume the same agent session rather than starting a new one.
- Cancel kills the process tree. A timeout fails the task with a reason.
- Exit without `submit_result` = failed, never completed.

### 5.3 Task dataset vs summary

| Record | Contains | Who reads it |
| --- | --- | --- |
| Task dataset (harness SQLite + task folder) | Request, tier, profile, lifecycle, every event, full agent output, questions/answers, artifacts, permissions, timing | Debugging, audits, future training/eval |
| `user_summary` (1–3 plain English sentences) | What was done and the outcome | Speech arbiter → Maid speaks it in Japanese (§4.5); memory bank candidate |

Only the summary crosses into the Maid.

### 5.4 Reporters

Preference order, as before: native API/event → narrow polling adapter → reporter sub-agent for interpretation.

| Reporter | Source | Notes |
| --- | --- | --- |
| Desktop activity | **ActivityWatch** (open source): active app and window, AFK state, with a local REST API | First reporter. Cheap, deterministic. Also tells the screen reporter which app is in front |
| Screen | The existing `maid_harness/reporters/screen/observer.py` scorer + vision model, rebuilt as a reporter (§5.6) | Moves out of `proactive.py`, emits `screen.changed` with a description |
| Task-side | Sub-agent lifecycle | Built into the supervisor |
| Later | Email (IMAP IDLE), internet feeds, Steam, notifications | After the event flow is proven |

Reporters emit facts with confidence. They never act, and screen or web text is never treated as an instruction.

### 5.5 Permissions

Sandbox by default, like normal coding-agent use. Escalation always goes through the Maid:

- T2 `quick`: read-only sandbox, network allowed.
- T3 `work`: may write only inside its task workspace.
- When a sub-agent needs more (a real repository, a command outside the sandbox, sending something), it calls `request_permission(what, why)`. The task pauses in `needs_permission`, the Maid asks by voice, and the user's yes/no goes back through the harness. A grant applies to that task only.
- T4 `computer`: always asks before starting, has an action budget, and reports what it changed.
- The Maid never claims an app was opened or a file was changed until `submit_result` says so.

Codex's own sandbox and approval modes are the enforcement layer. The harness maps each profile onto them and turns Codex approval requests into `request_permission`.

### 5.6 Screen reporter and computer use

These are two separate things and should stay separate:

| | Screen reporter (observe → respond) | Computer-use sub-agent (act) |
| --- | --- | --- |
| Purpose | Context for conversation, fluent interaction, proactive chat while gaming or working | Doing something on the computer for the user |
| Runs | Always, in the background, cheap | Only when a T4 task is started |
| Output | Facts: what is on screen, which app, what changed | Actions plus a verified result |
| Risk | Privacy (denylist, local models) | Changing things, so confirmation and an action budget |
| Speed | The Maid reacts within seconds | Slow, many steps |

The screen reporter is built in v1 (P4). Computer use comes last (P6). What carries over is the screen reporter's capture, its latest frame and description, and the frontmost app from ActivityWatch: the computer-use sub-agent can read them through the harness instead of starting blind. Nothing about the screen reporter has to change when computer use arrives.

For gaming and working, the reporter feeds the Maid more than "the screen changed". It also says which app is in front and whether it is a game, full screen, or the user is AFK, so the speech arbiter can use a per-activity rule (for example, short comments on big moments in a game, silence during a call).

Which app counts as what comes from a small config file (`activity_profiles`), for example `game = ["Steam", ...]`, `call = ["zoom.us", ...]`. For the demo, gaming runs on the Mac and any app (Chrome, Claude) can be listed as a game to simulate a play session. Games on the Windows PC need a capture service there that sends frames and activity to the Mac. That waits until the Maid has its full feature set.

## 6. Process model

One process, `maid-agent serve`: voice runtime + Maid core + harness, on asyncio. Each sub-agent run is a **detached** process group: `python -m maid_harness.subagent run` (supervisor) → `codex exec` → `python -m maid_harness.subagent mcp` (its tools). Everything persists to SQLite under `runs/`, which is the only channel between them, so no daemon is needed.

Sub-agents outlive the Maid (P3.5, replacing the first decision that they stop with her). When the Maid restarts, `Harness.recover()` sorts the active tasks: supervisor alive and heartbeating → keep watching; queued → start when there is room; question or permission pending → ask again once; anything else → `interrupted`. `Harness.watch()` then delivers, from a cursor saved in the database, every event written while she was away, so finished results are reported. Cancel kills the whole process group. An interrupted task can be continued on request (same Codex thread).

## 7. Code layout

```text
packages/
  maid_voice/     # unchanged: audio, VAD, ASR, LLM, TTS, turn controller, playback
  maid_agent/     # the Maid: app wiring, persona, router, speech arbiter, memory
  maid_harness/   # inbox, registry/dataset, supervisor, profiles, sub-agent runners,
                  # sub-agent MCP server, reporters (desktop, screen, ...)
```

Done in P0. `maid_harness` holds `core`, `inbox`, `registry`, `models`, `workers`, `mcp_server` and `reporters/screen/`. `maid_agent` holds `router.py` (with `RouteDecision`), `runtime.py` (the Maid-facing `MaidRuntime`), `demo.py`, `proactive.py` and the CLI. The Maid depends on the harness API only, and the harness never imports `maid_agent`.

## 8. Feature status

Updated when a phase lands. ✅ done · 🧪 prototype, to be rebuilt · ⏳ planned.

| Feature | Phase | State |
| --- | --- | --- |
| Voice loop: mic, VAD, ASR, DeepSeek reply, Natsume TTS, barge-in, latency metrics | before P0 | ✅ about 0.94 s p50 to first sound |
| Persona: role card, expression retrieval | before P0 | ✅ |
| Harness core: task records, SQLite registry, event inbox (dedupe, TTL), demo worker, cancel, computer-use confirmation, MCP stdio server | before P0 | ✅ works on its own, not connected to voice |
| Merged multi-task notifications | before P0 | ✅ fixed Japanese templates |
| Capability list → 「できること」 block and tool list, role card v2 | P1 | 🧪 |
| `look_at_screen` (in-turn screen lookup) and the latest screen fact in the state block | P1 | 🧪 |
| Screen observer, calibration logger, proactive screen remarks | before P0 | 🧪 rebuilt as a reporter in P4 |
| Keyword router | before P0 | 🧪 becomes the fallback in P1 |
| Package split (`maid_harness`), docs, `uv sync`, `.env` loading | P0 | ✅ |
| `delegate` tool for intent detection, immediate acknowledgement | P1 | 🧪 built and rehearsed silently, live gate pending |
| Task state for 「どうなった？」 questions (T1 lookups) | P1 | 🧪 |
| Speech arbiter v1: never over the user, merge, heartbeat | P1 | 🧪 |
| Cancel by voice | P1 | 🧪 |
| ASR pinned to Japanese (auto-detect misread Japanese as Korean), English task arguments, in-character notices | P1 | ✅ |
| Conversation log, restored after restart, elapsed-time note | P2 | 🧪 built and rehearsed silently, live gate pending |
| Codex sub-agent runner, `quick`/`work` profiles, sandboxed task workspace | P3 | 🧪 real Codex runs, live voice gate pending |
| Sub-agent tools: `report_progress`, `ask_user`, `request_permission`, `submit_result` | P3 | 🧪 |
| Answering sub-agent questions and permission requests by voice, follow-up resume, timeout, `interrupted` on restart | P3 | 🧪 |
| Clean isolated Codex home, durable detached sub-agents, read sandbox, correction pass, read/search tools, 3-day window, budgets | P3.5 | 🧪 real runs done, live voice gate pending |
| ActivityWatch reporter, `activity_profiles` (game/call/work) | P4 | ⏳ |
| Screen reporter (rebuilt from the observer), per-activity speech rules | P4 | ⏳ |
| Memory bank: from task summaries and 「覚えておいて」, list/correct/delete | P5 | ⏳ |
| Computer-use sub-agent: confirmation, action budget, verified result | P6 | ⏳ |
| Email, internet, Steam reporters; pop-up cards | P7 | ⏳ |
| Windows capture service, cloud deployment | after P7 | ⏳ |

## 9. Plan

Each phase ends with something you can try by voice.

### P0 — Clean foundation (small) — done 2026-10-31
Docs consolidated (this file). Split `maid_harness` out of `maid_agent`. Existing tests stay green.
**Gate:** `uv run pytest` passes, `maid-agent demo` and `maid-agent serve` still work.
**Result:** 28 tests pass. `demo`, `latency` and `harness-mcp` run. `serve` and `screen-log` import cleanly but were not started live, because they need the mic and oMLX.

### P1 — Maid ↔ harness in the live voice loop (fake sub-agent)
`delegate` tool in the reply call, immediate acknowledgement, task submission, T1 state block, speech arbiter v1, cancel by voice, Japanese patterns in the fallback router, ASR switched to auto-detect, English `delegate` arguments, completion notices spoken in character.
**Gate:** say a heavy request in Japanese and another in English → acknowledgement within the direct-path latency (±100 ms) → ask 「どうなった？」 and hear the real state → completion spoken at the next pause, never over you → 「やっぱりやめて」 cancels it. Chitchat latency unchanged.

**Status (2026-10-31): built, live gate not run yet.**
- `maid_agent/tools.py`: the `delegate`, `cancel_task` and `confirm_task` tools, their prompt rules, the T1 state block.
- `maid_agent/turns.py`: the controller hooks. Tool calls run only after the turn is committed, so a dropped hands-free speculation never starts a task. Keyword backstops: a spoken cancel with exactly one active task, and a yes/no to a confirmation asked in the last 2 minutes. `routing = "keyword"` puts `router.py` back in charge.
- `maid_agent/arbiter.py`: speech arbiter v1. Decisions go to `runs/agent/<date>/arbiter.jsonl`, tool calls to `tool_calls.jsonl`.
- `maid_voice`: `stream_response(context=, tools=)` with streamed tool calls, `TurnController.hooks`, `TurnController.notify()` for event turns.
- Harness: `FakeSubAgent` (run time per tier set in `[agent]`), `interrupted` status, `Harness.recover()` on startup.
- Probe (`maid_agent/scripts/probe_delegate.py`, 2 × 8 prompts, real role card): chitchat TTFT p50 513 ms with tools and 713 ms without, so no measurable cost. 15/16 routed right. Once she said 「止める」 without `cancel_task` (now covered by the backstop and a stricter rule), and once she called `delegate` without a word (now a fixed acknowledgement). `max_tokens` raised to 220 so the English `goal` fits.
- Silent rehearsal with real DeepSeek and TTS: delegation in Japanese and English, a real-state answer to 「どうなった？」, completion spoken unprompted, and 「やめて」 cancelling the right one of two tasks all worked.
- Left for the live gate: first-sound latency on the real mic, hands-free. ASR auto-detect was tried live and reverted to `ja` (Japanese came out as Korean).
- First live test (2026-10-31) ran a server started before P1 existed, so it was P0. It also exposed the real gap: she did not know her own abilities. Fixed by the capability list (§4.6), `look_at_screen`, and role card v2. Probe after the fix: 24/24 routed right, including the two weather requests and the screen question from that session, and 「何ができる？」 answered from the list. Chitchat TTFT p50 588 ms with tools vs 620 ms without.

### P2 — Conversation persistence
Conversation log in SQLite, restored on restart with the elapsed-time note.
**Gate:** stop and restart the Maid mid-conversation, and she continues it naturally.

**Status (2026-10-31): built, live gate not run yet.**
- `maid_agent/conversation.py`: every message that enters her history (your turns, her replies, cut-off replies marked （中断）, task notices, screen remarks) goes to `runs/maid/memory.sqlite3`, the Maid's own file, which the memory bank will share in P5. Speculative turns that were dropped are never logged.
- On startup the last `restore_turns` (8) messages go back into her history, and the web UI shows them faded.
- The elapsed-time note goes into the reply call: after a restart until her first reply, or after a pause of more than 30 minutes inside a session. It runs in steps from ほんの少し to N日 and says the conversation is remembered, without inventing what happened in between.
- `TurnController` sends every history change through `hooks.remember()`, so the log and her history can't drift apart.
- Silent rehearsal (real DeepSeek, three sessions on one database, restarts 20 s and 5 h apart): history and note restored every time. Asked 「さっきの話の続き」 / 「結局どっちにしたと思う？」, she picked up the topic in 2 of 4 resumes and hedged in character in the other 2. An early wording of the note made her act as if she had lost her memory, and was rewritten.

### P3 — First real sub-agent (Codex)
Codex runner, `quick` and `work` profiles, sandboxed per-task workspace, sub-agent MCP tools (`report_progress`, `ask_user`, `request_permission`, `submit_result`), `task.answer`, follow-up resume, timeout, restart → `interrupted`.
**Gate:** a real web question and a real sandboxed coding task both run. One asks a question and one asks for a permission, and you answer both by voice. The spoken summary matches what actually happened, and the full run is in the dataset.

**Status (2026-10-31): built, live voice gate not run yet.**
- `maid_harness/codex_runner.py` + `subagent_mcp.py` (see the harness README). Codex 0.165 notes: `resume` has no `-s`/`-C`, so the sandbox is set with `-c sandbox_mode=…` and the cwd is the task folder. MCP tools need `default_tools_approval_mode = "approve"`, because the default `prompt` hangs an unattended `exec` forever. `web_search = "live"` turns on search, and `notify = []` keeps the user's desktop hook out of it.
- The Maid gets `answer_task` and `follow_up_task`. `confirm_task` also answers permission requests, and the yes/no backstop covers them. Task news now includes questions and permission requests.
- Capabilities with `runner = "codex"`: `research` (live, quick tier), `work` (live), `computer_use` (simulated until P6).
- Real runs (Codex, the user's provider and model at effort low/medium):
  - Tokyo weather: 41 s, real answer.
  - `ask_user` → answer → resume on the same thread: 57 s.
  - `work` task blocked outside its folder → `request_permission` → grant → resumed with that folder writable → file written: 62 s.
  - Each run costs about 100–200k input tokens (90% cached) because of the user's full Codex setup. Slimming the sub-agent's Codex config is a later step.
- Silent rehearsal through `MaidAgent` (real DeepSeek + Codex): 「何ができる？」 → the real list; the weather request → real result spoken about 44 s later; 「ついでに明日の大阪も」 → `follow_up_task` on the same session → real forecast.
- A hard kill of the Maid used to orphan the Codex process and lose its result. P3.5 makes that the intended behavior: runs continue and their results are kept.

### P3.5 — Sub-agent hardening — built 2026-10-31
Found in live P3 use, or asked for by Alan. All built. Real-run numbers are from GPT-5.6-Luna low through the user's provider.
1. **Said it, didn't do it.** In live use she said 「探し直してくる」 with no tool call. Fixed by a stricter rule plus a **correction pass**: if a reply matches action words (`turns.py` `CLAIMS`) and called nothing, one extra call (same context, "call the tool now or nothing") runs and executes what it returns. In the rehearsal of the same conversation she called `follow_up_task` herself, so the correction pass did not have to fire.
2. **Results in detail.** The state block shows the newest result's `details`, and the `read_task` tool reads any task's full result. In the rehearsal, 「候補、全部教えて」 got all four products with prices.
3. **No silent no-ops.** `follow_up` works on cancelled tasks. A tool call that changed nothing is fed back to her (`[correction] … did nothing`), so she tells the truth instead of claiming success.
4. **A clean, isolated Codex** (`maid_harness/codex_home.py`). `runs/codex_home/` is rebuilt before every run from `~/.codex`: provider and login settings only, plus `auth.json` linked, never copied. Skills, MCP servers, hooks, notify and the user's sessions are not in it, and 22 Codex features a sub-agent doesn't need are off (the apps connector alone stalled startup about 35 s). Every run records a fingerprint: provider, endpoint host, model, and a hash of the account id. Weather question: 142k → 68k input tokens and 41 → 40 s.
5. **Durable sub-agents** (`maid_harness/subagent.py`). A detached supervisor per run, plus the four tools as a stdio MCP writing to SQLite (`subagent_mcp.py`). The registry does atomic merges across processes, and `Harness.watch()` delivers other processes' events from a saved cursor. Real test: the Maid process exited mid-task, Codex kept running, and a new Maid process found it still running and reported its result.
6. **Sandbox for reads.** A Codex filesystem permission profile: system basics readable, workspace and temp dir writable, everything else blocked. Grants (`read_path`, `write_path`, `network`, `full_access`) are passed per run as one inline TOML table. Real test: reading the repo README was blocked → `request_permission(read_path)` → yes → read. The sub-agent's environment is stripped to PATH/HOME/locale/proxy, so no `.env` keys.
7. **3-day window plus search.** The state block shows active tasks plus the 3 most recent from the last 3 days, and says how many more there are. `search_tasks` and `search_conversation` reach anything older.
8. **Resources.**
   - At most `max_running` (2) sub-agents at once; the rest are queued.
   - Workspace files are pruned after `retention_days` (7), and event logs are kept.
   - Token use is recorded per task and per day, shown in the UI and at startup, with an optional `daily_token_budget`.

Voice stall found in live P3.5 use (fixed): twice in one session the turn hung in "committing" and her next line was "shown but jammed" for about 50 s. Cause, from the oMLX log: Qwen3-ASR fell into a repetition loop on an ordinary clip and generated up to oMLX's default cap of 8,192 tokens (84 s). oMLX runs one request at a time, so ASR timed out and TTS waited behind it. It happened 2 times in 1,589 transcriptions, unrelated to the sub-agents that happened to be running. Fix: `asr.py` sends `max_tokens = 32 + 15 × seconds of audio`. Transcripts are unchanged, and a runaway now stops within a second.

Known limits:
- The model is pinned to `gpt-5.6-luna`. If cc-switch moves to a provider without it, sub-agent runs fail visibly, as `failed` with Codex's error.
- A run resumed after an account switch continues on the new account.

### P4 — Reporters and proactive speech
ActivityWatch reporter, the screen observer rebuilt as a reporter (§5.6), speech arbiter per-activity rules (working, gaming, call, AFK).
**Gate:** a normal working day and a gaming session, with a count of unprompted remarks and your rating of them. No spam, nothing spoken over you.
Known issue to fix here: vision and TTS share oMLX and the GPU, and her voice stutters when they overlap. `look_at_screen` already waits for her acknowledgement to finish (P2), but a proactive look can still overlap with a reply if the user starts talking during it. The reporter should pause or queue vision while she speaks.

Planned (Alan, 2026-10-31), **future work, not yet scheduled**: a CodeIsland-like **"your sessions" reporter**. The Maid knows the Claude Code and Codex sessions *you* are running (not her sub-agents), and speaks up when one finishes or waits for your approval, as an information source and a sense of working together. She has **no power to approve anything**.
- How to get the data: [CodeIsland](https://github.com/wxtsky/CodeIsland) (MIT) installs hooks into `~/.claude/settings.json` and `~/.codex/hooks.json`, and they forward events (session start, tool call, permission request, stop) over a Unix socket. That gives live "needs approval" status. Reading the session files the tools already write (`~/.claude/projects/**`, `~/.codex/sessions/**`) needs no config change but sees approvals late or not at all. Likely answer: our own hook entries next to CodeIsland's, forwarding to the Maid, opt-in, plus the files for history (last 3 days, searchable).
- Isolation: the Maid's sub-agents live in `runs/codex_home/`, a different CODEX_HOME, so they never show up as your sessions and the reporter never reads that home.
- Privacy: session summaries may go to DeepSeek (fine, per Alan), but **`.env` files, `auth.json` and anything that looks like a key are never sent**. A denylist plus secret redaction before any LLM call.

### P5 — Memory bank
Memory bank from task summaries and explicit 「覚えておいて」 requests, list/correct/delete by voice or UI.
**Gate:** preferences are recalled across days, and transient task state is not stored as fact.

### P6 — Computer-use sub-agent
`computer` profile with confirmation, an action budget and a verified result, reading the screen reporter's context.
**Gate:** 「Steam 開いて」 asks first, then does it, then reports only what was verified.

### P7 — More sources and surfaces
Email, internet, Steam reporters. Pop-up cards fed by the same summaries.

### Later — acting on your own sessions
Answering a permission prompt in, or sending a message to, a session *you* started (CodeIsland can do this through its hooks). Only after the read-only reporter has proven itself, always with your confirmation, and never on sessions the Maid didn't start unless you ask by name.

## 10. Open questions for Alan

Answered 2026-10-31: routing, sub-agent tool, permissions, order, restart, screen observer, desktop reporter (see §2, items 9–15).

Also answered 2026-10-31: input language (§4.5) and the gaming machine (§5.6).

No questions are open. New ones go here as they come up.

## Appendix A — Original brief (2026-10-30, sent to Codex)

> OK then, there is a fix on the architecture here. Indeed we need botmux, desktop, steam, user voice, but all of these should be the harness env we need to build. And current architecture should be changed. Here is some of my though that based on the low latency audio system we built. So I think the upper layer should be the real maid agent, which is the low latency system we just build. This low latency system was really simple now, but does not mean in the near future it won't have further features. But the functionality of this maid agent was provide instinct react to user's input, or system's change or reply to the changes that happens on the system which is the harness runtime support involved. And a typical workflow should like this, this upper maid voice agent answers and chat with the user, and actively start a chat or ask user when some preset event was triggered or some information has changed from os, email, internet events, something like that. But, when user asks complex tasks, like go search something, open this app for me, or even ask a coding question, this kind of heavy task was not supported by the maid agent, maid agent only takes in charges of basic chat and interactions, chitchat, but not for heavy task, maid agent should also have these kind of mechanism that could detect whether the user's request was a heavy task or not. Heavy task should be handled by a "sub-agent" on the lower level. Which maid agent can lauch such a agent, all these kind of hard task was taken by the sub-agent on lower layer that lives in the harness env. This harness env should provide the basic information needs of maid agent, but also provides the env for sub-agent to do heavy task, and when there is a task over, sub-agent reports to maid-agent, and maid-agent reminds user with voice, future will support something like a pop-up message card. How this kind of architecture cooperate with the larger system we had before, the Botmux system, it naturally is a codex based framework that has codex session management for feishu bot, and the Botmux system we mentioned here doesn't necessarily goes into our final system. You can take a look at this project of mine, there is a manager bot that manages all codex sessions, and this manager bot was like the maid agent we have here, but maid agent requires low latency interaction, so it does not support that much feature like the manager bot, the manager bot is a codex session here in this project(https://github.com/miko8422/MikuDance.git). We don't actually want everything in botmux, but I think codex based system like this might help us here, we need sub-agent to do heavy task anyway. As the desktop, which is the system information, it should been built(if there does not have any opensource solutions) or supported by a "reporter" sub-agent on the background. And also the "computer use", I think it should also be supported by a sub-agent, so this forms this kind of architecture: maid-agent on top layer, send task to lower sub-agents, get informations from sub-agents, controls sub-agents, and reply to user, maid-agent should keep in light, low-latency and fast interaction, and if the task isn't yet done and maid agent didn't get informations from lower-agents, maid agent should say that task is still going and report current state. As the event inbox, it is the loop we need to build for this maid agent, or directly to the sub-agents, one task one sub-agent. As the memory bank, it is part of the maid agent, it should not be a up/lower relations, we could keep a mechanism that triggers a task summary for maid agent report as well as save to the Tasks dataset, I think for sub agent task record, we should keep a real dataset, and a summary that maid agent uses, and this summary goes to the actual memory bank for Agent memeory. As the planner, I don't think this is maid agent's job, and it should never be, maid agent needs to have this kind of ability to rout and sned tasks to sub-agents, and have a basic judge on how difficult this job might be, and based on the difficulties to create sub-agent sessions. For first version I don't think we need a planner on maid agent, because it might slow the interaction. So how do you think about the new architecture? we already have the voice interaction of maid agent now, what remains are really the big part of this project. And I think you should keep these content into a markdown report for future reference, and we will discuss a new architecture out of this and update on this markdown file, and when everything is settled, we can start our real crafting.
