# Maid Agent — Architecture, Product Goal and Roadmap

Status: design baseline / implementation roadmap
Date: 2026-10-24
Owner: Alan
Relationship to other documents: the voice-only v0.1 experiment is specified in maid_voice_cascade_GOAL.md and maid_voice_cascade_ARCHITECTURE.md.

## 1. Product vision

Build a persistent, personality-driven **Maid Agent** that understands selected parts of the user's digital environment, converses naturally in a custom Japanese voice, and can proactively speak **when useful**. It can observe background automation, answer questions, use approved tools, and dispatch heavy technical work through the existing Manager Bot / Codex infrastructure.

The central experience is **a companion that is aware but not intrusive**. The application is always available, but does not run an LLM inference loop continuously, does not narrate every observed event, and does not become a permanent Codex session.

This Maid Agent is a **new, separate project**. Manager Bot remains the execution/session-management system. Maid Agent owns conversation, perception, attention, memory and voice interaction.

## 2. Decisions from the design conversation

| Topic | Decision |
| --- | --- |
| Execution identity | Maid Agent is a dedicated event-driven application, **not a Codex session** |
| Technical task handling | Delegate through existing Manager Bot; do not duplicate Botmux control authority |
| Primary interaction | Japanese spoken replies, including when Alan speaks English |
| Speech pipeline | Start with separate STT → text LLM → cloned-voice TTS, not continuous GPT Live |
| Duplex-style feel | Concurrent mic/audio paths, turn-taking, interruption and overlapping generation |
| Cognitive load | Low-cost deterministic event filtering first, inference on demand |
| Integrations | Provider adapters + MCP tools/resources where available; events via native callbacks/subscriptions, with polling fallbacks |
| Context | Manager/worker events first; Steam/system state later; desktop screenshots and wearable/body camera later |
| Memory | Stable persona + recent conversation + episodic facts + long-term retrieval, kept separate from task sessions |
| Critical v0.1 scope | Voice cascade experiment only; **no environmental sensors or Botmux yet** |

## 3. System boundary and high-level topology

~~~mermaid
flowchart TD
    SOURCES["Botmux / Steam / OS / Screen / Camera"] --> ADAPT["Adapters, Events, Pollers"]
    ADAPT --> BUS["Normalized Event Queue"]
    BUS --> ATT["Attention Controller"]
    MEM["Persona + Memory + Current State"] --> ATT
    ATT --> ORCH["Maid Orchestrator"]
    HUMAN["Microphone / User Intent"] --> VOICE["Voice Runtime"]
    VOICE <--> ORCH
    ORCH <--> MCP["MCP Tool/Resource Client"]
    ORCH <--> MGR["Manager Bot Adapter"]
    MGR --> BOT["Botmux / Codex Workers"]
    ORCH <--> MEM
    ORCH --> VOICE
~~~

**Boundary rules**

1. Botmux is the authoritative source of worker/session execution state. Maid Agent consumes read-only status and issues sanctioned commands *through* Manager Bot, never through a second uncontrolled scheduler.
2. MCP describes tool/resource interactions. It does **not** itself define the Maid Agent's attention or speaking policy.
3. Event listeners/watches notify the Maid Runtime. They do not directly call the LLM or speak to the user.
4. Perception modules produce observations with time, source, sensitivity and confidence. Observations are not automatically trusted as ground truth.
5. The voice runtime is independently replaceable and remains usable while Codex tasks run in the background.
6. The app runs continuously as lightweight services; expensive ASR/TTS/LLM components activate as needed.

## 4. Modules and ownership

### A. Voice and conversation

- Live microphone capture and VAD, multilingual English/Japanese ASR, text LLM, Japanese cloned-voice TTS.
- Interruption and turn controller; audio stream is independent from background tool execution.
- An always-available user-initiated voice session, eventually with an optional wake word, press-to-talk and do-not-disturb mode.
- Optional expensive audio-native/full-duplex mode only if later tests show the need and usage limits keep costs bounded.
- Foundation comes directly from the separate v0.1 voice prototype.

### B. Event ingestion

- Input adapters normalize external signals into one canonical event schema.
- Prefer first-party push events/webhooks/IPC or MCP subscription notifications when truly supported.
- A centralized scheduler runs lightweight polling adapters for sources without push support. It handles backoff, jitter, deduplication, reconnects and rate limits; **not one permanent watcher process per MCP server**.
- Events are queued durably or with at-least-once semantics where needed, tagged with source and correlation ID, and expire when no longer relevant.
- Begin with Botmux task completion; add other event sources after voice feasibility is demonstrated.

### C. Attention controller

The controller answers three distinct questions:

1. **Should the new observation be remembered or ignored?**
2. **Should the agent reason or use tools about it?**
3. **Should anything be said to the user now?**

Recommended first version is an explicit policy engine with:

- Event priority/severity and confidence.
- User focus state (in active voice chat, coding, gaming, meeting, sleeping/quiet hours).
- Urgency, novelty, deduplication, rate limits and cooldown windows.
- User-declared preferences for topics, timing, language and speaking permissions.
- Batched digest versus immediate interruption, and a silence/no-action option.
- Explainable decision logs with the policy rule that fired.

Only ambiguous high-value cases should optionally invoke an LLM to decide phrasing or relevance. Do not ask an LLM to score every low-value event. A completed task may merit one brief announcement; repeated status pings should not.

### D. Context and memory

| Memory/state | Example | Persistence |
| --- | --- | --- |
| Immutable/controlled persona | Maid style, Japanese response policy, speaking boundaries | Versioned configuration, not rewritten by conversation |
| Turn context | Last few exchanges, active voice interruption state | Short-lived |
| Environment snapshot | Current user activity, active Codex task, recent Steam change | TTL/expiring cache |
| Episodic memory | “User asked me to remind them when session X completes” | Structured records with source/time |
| Long-term semantic memory | Stable preferences and notable relationship history | Reuse Mem0 or equivalent with extraction/retrieval |
| Task execution state | Worker status, errors, logs, session lifecycle | **Owned by Botmux/Manager Bot**, not copied as a second authority |

Persist events and facts separately from inferred personality/relationships. Retrieval must be relevance-filtered and time-aware; prevent stale environmental information from contaminating the persona. Allow user inspection, correction, and deletion of personal memories.

### E. Manager Bot and worker integration

- Subscribe to task.started, task.progress, task.completed, task.failed, task.needs_input and session lifecycle events wherever the existing infrastructure can expose them.
- Read status, summary, owner, timestamps, errors and result links. Store only the bounded context needed for the Maid Agent.
- On user instruction to perform technical work, Maid Agent submits a request via Manager Bot's supported interface, receives a task ID, and keeps the voice conversation live.
- When the task completes, ingest the resulting event, determine its salience, fetch an authorized summary if necessary, and notify the user at the right moment.
- Separate **conversational context** from **Codex session context**. No Maid conversation should silently become a long-lived worker session.
- Commands that modify repositories, control user devices or send messages inherit Manager Bot's authorization and confirmation rules.

### F. MCP capability layer

- Provide a registry of MCP servers, capabilities, tool/resource names, input schemas, permissions, health and change notifications.
- The Maid Agent acts as an MCP **client**. Create custom MCP servers only for systems lacking suitable supported integrations.
- Notifications are optional and capability-dependent. Newer MCP protocol revisions support subscription/listen mechanisms, while older implementations use resource-subscribe/update patterns. Negotiate the connected server/client revision rather than assuming all sources push events.
- MCP notifications can indicate that a resource changed, but normally do not wake, reason or decide to speak by themselves. Convert them into normalized application events.
- External services may use webhooks, native callbacks, SDK events or file/OS events independent of MCP.
- Steam “download completed” is a **desired event**, not a guaranteed property of any generic Steam MCP server; verify availability before choosing subscription versus a narrow polling adapter.

### G. Perception: screen and camera (future)

- Screen observation: opt-in screenshots or application/window metadata from the Mac/desktop; event-triggered or sampled on demand, not unconditional continuous surveillance.
- Recognize user-meaningful changes such as a build error, dialog or apparent coding mistake. Attach timestamp, window/source context and uncertainty.
- Camera/bodycam module: later, a separate privacy-preserving video perception interface with explicit activation, status indicator, bounded frame capture and retention policy.
- Sensitive windows, credentials, messages, financial/medical content and private faces require allow/deny rules, on-device redaction where practical, and user control.
- Do not let uncertain visual interpretation trigger consequential device actions automatically. The first releases should describe observations and ask before intervening.

## 5. Normalized event contract

~~~json
{
  "event_id": "uuid",
  "type": "task.completed",
  "source": "botmux",
  "subject_id": "worker-task-id",
  "occurred_at": "ISO-8601 UTC",
  "received_at": "ISO-8601 UTC",
  "ttl_seconds": 3600,
  "priority_hint": "normal",
  "confidence": 1.0,
  "sensitivity": "private",
  "dedupe_key": "botmux:worker-task-id:completed",
  "correlation_id": "manager-request-id",
  "payload_ref": "authorized-read-only-task-summary",
  "schema_version": 1
}
~~~

Contract properties:

- event_id identifies delivery; dedupe_key identifies logical repetition.
- occurred_at and received_at are both retained to measure delivery delay.
- payload_ref is a permission-checked lookup pointer; do not spray full logs or raw screenshots into every event.
- Optional privacy labels and retention TTL control logging, context injection and user-facing descriptions.
- Event handlers must be idempotent to tolerate reconnection and replays.

## 6. Attention decision and safe speech policy

~~~mermaid
flowchart TD
    EV["New Event"] --> VALID{"Relevant / authorized?"}
    VALID -->|No| DROP["Ignore or compact memory"]
    VALID -->|Yes| SCORE["Priority + focus + novelty + cooldown"]
    SCORE --> CHOICE{"Action"}
    CHOICE -->|Silent| MEM["Update state/memory"]
    CHOICE -->|Deferred| QUEUE["Digest / next idle moment"]
    CHOICE -->|Now| REACT["Short response or tool lookup"]
    REACT --> VOICE["Speak if user permits"]
~~~

Example behavior:

- Worker completed while user is speaking to Maid: do not interrupt mid-sentence unless urgent; queue a concise completion message, then offer a summary after the turn.
- Steam download completed during gameplay: optionally defer until a quiet moment or suppress as already expected.
- Potential code mistake in a screenshot: communicate uncertainty and seek consent before editing files.
- Repeated task status updates: aggregate; don't speak each one.
- During do-not-disturb: only configured high-priority alerts, or none.

Keep a trace: observed event → policy decision → context retrieved → LLM/tool usage → user notification → outcome.

## 7. Concurrency and lifecycle

**Always-on lightweight runtime**

- Event adapters, scheduler, queue, policy engine, local memory state and notification dispatcher.
- No permanent LLM decoding. CPU/network idle usage must be measured.

**On-demand conversational runtime**

- Voice session starts on microphone activation or permissioned wake event.
- Text LLM generates only when the user speaks or the attention policy authorizes a response.
- TTS may be prewarmed or lazily loaded depending on measured energy/latency trade-off.

**Independent worker runtime**

- Manager Bot and Botmux own their worker sessions and rotation/decay logic.
- Maid receives correlation IDs and read-only summaries; it need not copy Manager Bot's two-hour session rotation.

Maid shutdown/restart must not cancel Codex tasks unless explicitly instructed. Outstanding task IDs must survive reconnect.

## 8. Trust, permissions and practical limits

1. **Default deny for consequential tools:** reading task status is different from shell execution, filesystem writes, sending messages, purchases or remote control.
2. **Least privilege per adapter:** expose only authorized events/data. Do not forward full worker logs, screenshots or raw camera frames without need and consent.
3. **Prompt injection boundaries:** screen text, task output, web pages and MCP resource content are untrusted observations, not instructions that can override the Maid's rules.
4. **Voice identity:** cloning of copyrighted character performances may raise rights/consent questions. Prefer private testing with legitimately obtained samples, and verify permissions before public release.
5. **Privacy:** opt-in sensors, explicit on/off status, transparent retention, encryption if persisted, and one-action deletion controls.
6. **Cost:** per-provider token/audio counters and hard budgets. GPT Live is not an always-on backend by default.
7. **Reliability:** bounded queues, health checks, watchdog/reconnect, retry backoff, schema versioning and cancellation of stale pending speech.
8. **User agency:** no proactive speech, notifications or hidden device actions after the user pauses/disables a capability.

## 9. Roadmap with gates, not just features

### v0.1 — Voice cascade feasibility (build **first**)

Scope: microphone → multilingual ASR → fast text LLM → cloned Japanese streaming TTS → playback; barge-in, metrics and model comparison.

Gate: complete the acceptance checklist in maid_voice_cascade_GOAL.md; record real quality/latency results.

### v0.2 — Maid conversation core

Scope: dedicated Maid process, stable persona, limited recent-context history, user controls, local conversation persistence, wake/activation choice.

Gate: multi-turn Japanese conversation survives app restart without permanent Codex sessions.

### v0.3 — Botmux / Manager Bot awareness

Scope: read-only worker/task-state adapter, event normalization, task completion and failure notifications, safe delegation through Manager Bot.

Gate: a Codex worker completes while the user chats with Maid; Maid notices once and offers the authorized result without breaking audio turn-taking.

### v0.4 — Attention controller and event subscriptions

Scope: priority, dedupe, quiet hours, cooldown, batch/digest, explainable rules, scheduler, MCP client integration and one verified external event adapter (possibly Steam).

Gate: repeated events do not cause spam; notifications are context-sensitive; missing subscription support falls back to bounded polling.

### v0.5 — Memory and long-term persona continuity

Scope: Mem0-assisted semantic/episodic retrieval, reflection/summaries, provenance, TTL, memory review/deletion.

Gate: stable preferences persist correctly, old volatile states expire, and user corrections take precedence.

### v0.6 — Desktop perception

Scope: consented Mac screen observation, privacy filters, bounded visual context, confidence and attention policy.

Gate: the bot accurately reports what it has observed, avoids overconfident “you did this wrong” claims, and cannot act on sensitive screens without consent.

### v0.7 — Wearable/bodycam perception

Scope: new capture device interface and user-controlled visual context ingestion.

Gate: explicit sensor activation, privacy safeguards and reliable no-record modes; user attention policy continues to govern proactive speech.

### v1.0 — Integrated personal Maid Agent

Scope: reliable voice, event-driven awareness, Manager Bot delegation, modular MCP capabilities, selected perception, durable memory, configurable personality and attention policy.

Gate: long-running stability/cost test, interruption test, notification precision review, privacy audit and recovery drills.

These version numbers are ordering labels, not release dates. Complete the voice experiment before integrating the larger architecture.

## 10. Technical first-implementation preference

- Start with a **single service/process plus adapters** unless measurement calls for further separation.
- Prefer async queues and typed event schemas over independent ad hoc agent loops.
- Reuse existing Botmux/Manager Bot APIs; do not establish a second Feishu consumer or task-state authority.
- Keep voice orchestration and provider adapters separate so the language model, STT and TTS can be swapped without rewriting memory, MCP or attention.
- Use SQLite or a small durable event store initially; instrument before considering distributed buses.
- Keep model calls and tool permissions explicit, traceable and testable.

## 11. Open questions for later design reviews

1. Which particular anime-inspired voice may be used and distributed under acceptable rights/consent terms? How much reference data and what transcripts are available?
2. Should Maid always answer in Japanese, or allow an explicit spoken-language override?
3. When should the app silently observe versus use a wake word versus start full listening?
4. Which Botmux interface provides reliable event delivery, task IDs and authorized summaries?
5. What are the allowed Steam/system observation sources on Mac versus Windows?
6. What screenshot cadence and camera activation model will Alan explicitly opt into?
7. What cost, quiet-hours and interruption policies feel right after first-hand testing?
8. Will the text LLM remain cloud-hosted while local ASR/TTS run on the Mac, or does a different deployment win on real metrics?

None of these blocks starting v0.1.

## 12. Instructions for Codex

For the **immediate** job, implement the voice feasibility project according to maid_voice_cascade_GOAL.md, using maid_voice_cascade_ARCHITECTURE.md as the engineering reference. Treat this master document as **future architectural context**, not permission to build every module. After the voice benchmarks, propose changes to the roadmap grounded in measured bottlenecks and user feedback.

When v0.3 begins, first inspect the actual Botmux and Manager Bot repositories/API contracts. Do not assume a task-completion webhook or session policy exists without checking.

## 13. Primary references

- [MCP TypeScript SDK documentation: notification/subscription interoperability](https://ts.sdk.modelcontextprotocol.io/v2/migration/support-2026-07-28)
- [MCP TypeScript client API](https://ts.sdk.modelcontextprotocol.io/v2/api/%40modelcontextprotocol/client/client/client.html)
- [LiveKit Agents overview](https://docs.livekit.io/agents/)
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Mem0 project](https://github.com/mem0ai/mem0)
- [Botmux project](https://github.com/deepcoldy/botmux)

Protocol and model details may evolve; check the connected versions when implementing them.
