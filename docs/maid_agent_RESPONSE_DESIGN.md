# Maid Agent — Response System, Role Card and Screen Trigger Design

Status: module reference. The current architecture and plan are in `ARCHITECTURE.md`. Still valid here: §1 (MaiBot ideas), §3 (role card, expression bank), §4 (screen scoring). §2, §5 and §7 (planner flow, gate, R1–R4 build order) are superseded by `ARCHITECTURE.md` §4.3 and §9.
Date: 2026-10-25
Relationship: the voice runtime is `../packages/maid_voice/`, already working.
Reference studied: MaiBot (`Mai-with-u/MaiBot`, main @ 062d45c, 2026-10-24). It is a group-chat bot, so its ideas are adapted for one person and voice, not copied.

## 1. What we take from MaiBot, and what we change

| MaiBot mechanism (where) | What it does | Our adaptation |
| --- | --- | --- |
| **Planner / replyer split** (`prompts/*/maisaka_chat*.prompt`, `maisaka_replyer.prompt`) | A planner LLM with personality + `behavior_style` decides the next action using tools: `reply(reply_reference)`, `wait(seconds)`, `stop()`, memory lookups. A separate replyer turns the *guidance* (not the words) into an in-character message using `reply_style` | Keep for **proactive** turns (screen, events). **Skip the planner when Alan speaks to her**: a second LLM call would add about 0.5 s to the 0.94 s voice path. Direct speech goes straight to the replyer |
| **Persona split into three fields** (`PersonalityConfig`) | `personality` (who she is), `behavior_style` (the planner's rules for when to join in or stay quiet), `reply_style` (how she talks), plus alternative styles chosen at random | The **role card** (§3) keeps this split: identity, behavior, speech. Each LLM role gets only its part |
| **Expression learning** (`learners/expression_learner.py`, `learn_style.prompt`) | Mines chat logs into "when *situation*, say *style*" pairs, retrieves them by vector search, and injects a few per reply | **Mine ナツメ's own 3669 game lines** into an expression bank and retrieve 3–5 per reply with local `embeddinggemma-2`. This is the biggest gain in "sounds like her" |
| **Reply-likelihood gate** (`turn_trigger/reply_likelihood.py`, `dynamic_gate.py`) | A logistic regression over cheap features (mention, question mark, own recent share of messages, time since own last message, …) estimates whether the planner *would* reply. A threshold that adapts to a `talk_value` frequency target decides whether the planner runs at all | Same shape for **proactive triggers**: screen change, event type, novelty, user state, time since she last spoke unprompted, recent dismissals. Weights are hand-set first and refitted from our own decision logs later, as MaiBot does |
| **Idle backoff** (`idle_backoff.py`) | After repeated "nothing to do" decisions, waits exponentially longer before thinking again | The same for screen-triggered planning, so a static screen costs nothing |
| **Proactive turns** (`runtime.enqueue_proactive_task`) | Internal events (plugins, finished sub-agents) wake the planner without any user message | **This is the hook for screen observations**, and later Botmux/Steam |
| **Context restore with elapsed time** (`runtime._build_context_restore_wakeup_text`) | After a restart, the prompt says how long it has been offline, in steps from "brief restart" to "slept for days" | Adopt as is. It meets the master v0.2 gate (a conversation survives a restart) |
| Talk-frequency rules by time of day (`talk_value_rules`) | Quieter at night | Quiet hours and focus modes (§5) |
| Not used | Group logic (@-mentions, multiple chats, quoting), emoji/sticker system, the desktop-*control* sandbox (`src/computer` drives a VM; it does not watch the screen) | — |

## 2. Overall flow

~~~mermaid
flowchart TD
    MIC["Alan speaks or types"] --> FAST["Direct path: replyer only"]
    SCR["Screen observer"] --> OBS["Observation event"]
    EVT["Botmux / Steam / timers (later)"] --> OBS
    OBS --> GATE["Cheap gate: rules + likelihood + frequency + backoff"]
    GATE -->|"below threshold"| LOG["Remember silently / drop, logged with reason"]
    GATE -->|"passes"| PLAN["Planner LLM: speak / remember / look_closer / wait / stop"]
    PLAN -->|"speak(guidance)"| QUEUE["Speech queue: never talks over Alan"]
    QUEUE --> REPLY["Replyer: role card + expressions + context + guidance"]
    FAST --> REPLY
    REPLY --> TTS["maid_voice: fine-tuned ナツメ TTS"]
    CARD["Role card"] --> REPLY
    CARD --> PLAN
    BANK["Expression bank (her real lines)"] --> REPLY
    DB["SQLite: turns, observations, decisions"] <--> PLAN
    DB <--> REPLY
~~~

Two paths, two latency budgets:
- **Direct (Alan → her):** must stay within today's latency (about 0.94 s p50 to first sound). One LLM call, plus expression retrieval (local embedding, about 10 ms).
- **Proactive (world → her):** latency hardly matters (seconds are fine). The cost must, so the gate keeps LLM calls rare.

## 3. Role card (`persona/natsume.v1.toml`)

Versioned and never rewritten by conversation (master §4 D). Each section is shown only to the role that needs it.

| Section | Used by | Content (drafted from her lines; Alan reviews) |
| --- | --- | --- |
| `identity` | planner + replyer | Who she is (ナツメ, adapted from 喫茶ステラ to Alan's home/desk companion), her relationship to Alan, what she knows about her own situation |
| `speech` | replyer | First person **ワタシ** (398× in her lines; 私 only once). She addresses Alan as **キミ** (to confirm). Real tics: **でしょ** (200), **だから** (217), **じゃない** (130), 別に, ふーん, もう. **わよ appears only 12×**, so don't overuse it; today's prompt does |
| `voice_rules` | replyer | Output is spoken: 1–2 short sentences, first sentence shortest, no symbols, emoji, lists or English abbreviations (from today's `llm.py`) |
| `behavior` | planner | When to speak unprompted and when to stay quiet. Prefers silence to filler and comments only on things worth commenting on. Never lectures. Offers rather than does |
| `boundaries` | both | No invented facts (weather, task state) without a source. Uncertain screen readings are phrased as guesses. Never acts on files or devices without asking |
| `examples` | replyer | 5–8 short real lines as tone anchors, chosen from the expression bank |
| `alt_styles` | replyer | Occasional variants, MaiBot style (e.g. extra short, teasing), picked at random with a small probability |

**Expression bank (`persona/natsume_expressions.jsonl`):** built offline once.
1. Group her lines by scene, in order. Only the 3037 cleaned lines (adult scenes and moans are already filtered out).
2. Ask the LLM to extract `{situation, style, example}` from each window.
3. De-duplicate, then embed `situation` with `embeddinggemma-2`.

At reply time, embed the current context and inject the top 3–5 as 話し方の参考. Expected size: a few hundred entries.

## 4. Screen observer (the main proactive trigger), revised 2026-10-26 with あきら

Decisions: **main display only** (the one with the menu bar), **always on**, and the screen is sent to her only when it **changed dramatically**, scored with an image embedding. App event hooks (Codex finishing a session, later mail and other system events via MCP or a runtime) are the precise trigger. The screen is the catch-all for apps without hooks.

Code: `maid_agent/src/maid_agent/screen/` (`observer.py` scorer + loop, `platform_mac.py`, `embed.py`, `logger.py`).

| Stage | What | Cost (M5 Pro, measured) |
| --- | --- | --- |
| Every 2 s, while active | Skip if locked, idle > 5 min, or a denylisted app is in front. Otherwise grab the main display, downscale to 640 px, and compute an 8×6 grid of tile gray levels | 42 ms CPU |
| Settle | A state counts only after 2 frames with no tile changing (ignores scrolling, animation, spinners) | about 0 |
| Score the settled frame against the **reference** (the last frame that fired = what she saw) | `emb_dist` (1 − cosine, SigLIP2 on the GPU): did the meaning change. `tile_max`/`tile_frac`: catches a small popup on an unchanged screen. `app_switch`. `app_z`: change relative to this app's usual change (games/video change all the time). `seen_sim`: matches a recently fired frame in both embedding *and* layout, so switching back is not new | 14 ms GPU |
| Fire (initial hand-set rule) | app switch, or `emb_dist ≥ 0.12`, or a strong but localized tile change; not if seen recently; not if usual for this app | — |
| Then (step R3) | The fired frame becomes a `screen.changed` event in her inbox. Her gate decides; only if it passes does Mage-VL (local) describe it and the planner decide whether to speak | about 1–3 s GPU, rare |

**Calibration first:** `maid-agent screen-log` records every settled state with its scores and a 320 px thumbnail (kept 7 days) and sends nothing. あきら labels "would I want her to see this?", and the thresholds are fitted to those labels (as MaiBot fits its reply gate from planner logs).

Privacy: password managers and similar apps are denylisted *before* capture. Pauses when locked or away. Thumbnails exist only for labeling and are deleted after 7 days. Models are local.

Known gaps:
- Window titles need the Accessibility permission (not granted yet).
- A launchd agent for surviving reboots needs its own Screen Recording permission.
- A blank frame from a missing permission is logged as `blank_capture`.

## 5. Gate: when she speaks unprompted

Applied in order, each step logged with its reason (master §6, "explainable"):

1. **Hard rules:**
   - Never while Alan is speaking or she is already speaking: the message is queued until the next pause.
   - Mute/do-not-disturb wins.
   - Quiet hours.
   - Focus modes: fullscreen game → hold until the next quiet moment or drop. Meeting/call app → silent.
2. **Dedupe and cooldown:**
   - The same `dedupe_key` (e.g. the same error) speaks once.
   - A minimum gap between unprompted remarks (default 10 min).
   - Old observations expire.
3. **Likelihood score** (logistic, MaiBot style). Features:
   - event type and novelty
   - error-like text
   - app switch
   - Alan idle vs typing
   - time since his last words to her
   - time since her last unprompted remark
   - how often he ignored or dismissed her recently
   - time of day
4. **Frequency target:** a `talk_value` knob (unprompted remarks per hour), with the threshold adjusted so the real rate tracks it.
5. **Feedback:** a dismissal ("静かにして", or no reply) lowers the frequency; engagement raises it.
6. **Only then the planner LLM.** Tools:
   - `speak(guidance)`
   - `remember(note)`
   - `look_closer()` (requests a tier-3 analysis or a crop)
   - `wait(seconds)`
   - `stop()`
   - later: `delegate_task` through Manager Bot

## 6. Storage (SQLite, one file)

`turns` (who, text, timestamps, latency trace id), `observations`, `decisions` (gate inputs, score, rule fired, planner reasoning, outcome), `speech_queue`, `expressions`. The `decisions` table doubles as training data for refitting the gate, as MaiBot fits its weights from planner logs.

## 7. Build order

| Step | Delivers | How we check it |
| --- | --- | --- |
| **R1 — Role card + expression bank + replyer refactor** | `persona/` files, mining script, retrieval in the direct path | Blind A/B: old prompt vs role card + expressions on 20 prompts; direct-path latency unchanged (±30 ms) |
| **R2 — Persistence and restart** | SQLite turns, elapsed-time wake-up text, text input | Master v0.2 gate: a conversation survives an app restart |
| **R3 — Event inbox, gate, planner, speech queue** | Normalized events, rules + likelihood + frequency, planner tools, never talking over Alan | Inject test events: no spam, correct deferral, a readable decision log |
| **R4 — Screen observer** | Embedding-scored settle/fire (§4; the calibration logger is already running), Mage-VL observations → R3 | A day of normal use: count of remarks vs target, Alan's precision rating, idle CPU/GPU cost |
| then v0.3 | Botmux / Manager Bot adapter into the same event inbox | Master v0.3 gate |

This moves "desktop perception" from the master doc's v0.6 to right after the conversation core, by Alan's choice (2026-10-25). It needs R3's gate first; without it, a screen watcher would just produce spam.

## 8. Open questions for Alan

1. ~~What should she call you~~: あきら (decided). Role card v1 approved 2026-10-26.
2. ~~Screen watching~~: decided 2026-10-26: main display, always on, embedding-scored. The denylist needs あきら's additions.
3. Unprompted remarks and the whole "when to reply" mechanism: **to be discussed with あきら before R3**.
4. Mage-VL vs Qwen3.6-35B for screenshots: to benchmark in R4 (quality vs GPU contention with TTS).
