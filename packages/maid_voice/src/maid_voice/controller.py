"""Turn controller: commits utterances, owns the active generation id, cancels stale work.

Invariants (ARCHITECTURE §5):
- only one committed response owns playback at a time (PlaybackQueue.active_gen)
- a new turn or an explicit stop cancels the old one *before* the new generation opens
- every downstream packet carries generation_id; stale ones are dropped, never played
- any stage failure ends the turn and returns to 'listening' without raising
- tool calls (side effects) run only once their turn is committed; a dropped speculation runs none
"""

import asyncio
import itertools
import uuid
from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable, Protocol

import numpy as np

from .asr import ASRAdapter
from .config import VoiceProfile
from .llm import ChatTurn, LLMAdapter, ToolCall
from .metrics import MetricsSink, TurnTrace, now_ns
from .segmenter import ClauseSegmenter
from .tts import TTSAdapter

Emit = Callable[[dict], Awaitable[None] | None]


class PlaybackSink:
    """What the controller needs from the player (real Player or a test fake)."""

    def begin_generation(self, gen: int, held: bool = False) -> None: ...
    def release(self, gen: int) -> bool: ...
    def enqueue(self, gen: int, pcm: np.ndarray) -> bool: ...
    def finish_generation(self, gen: int) -> None: ...
    def stop_now(self, gen: int) -> None: ...


@dataclass
class TurnPlan:
    """What an extension adds to one reply call."""
    context: str = ""                                            # appended to the system prompt
    tools: list[dict] | None = None
    calls: list[ToolCall] = field(default_factory=list)          # run after the reply regardless of the LLM
    fallback_calls: list[ToolCall] = field(default_factory=list)  # run only if the LLM called nothing


class TurnHooks(Protocol):
    """Lets an extension (maid_agent) route inside the reply call without maid_voice knowing about tasks."""

    async def plan(self, text: str, kind: str) -> TurnPlan: ...      # kind: "turn" | "notice"
    def ack(self, calls: list[ToolCall]) -> str | None: ...          # line to say if the LLM said nothing
    async def run(self, calls: list[ToolCall], user_text: str) -> None: ...


class TurnController:
    def __init__(self, asr: ASRAdapter, llm: LLMAdapter, tts: TTSAdapter, player: PlaybackSink,
                 voice: VoiceProfile, metrics: MetricsSink | None = None, emit: Emit | None = None,
                 history_turns: int = 4, output_device: str = ""):
        self.asr, self.llm, self.tts, self.player = asr, llm, tts, player
        self.voice = voice
        self.metrics = metrics
        self._emit_cb = emit
        self.history_turns = history_turns
        self.output_device = output_device
        self.session_id = uuid.uuid4().hex[:8]
        self.history: list[ChatTurn] = []
        self.state = "listening"
        self._gen_ids = itertools.count(1)
        self.active_gen = 0
        self._task: asyncio.Task | None = None
        self._trace: TurnTrace | None = None
        self._playback_done: asyncio.Future | None = None
        self._halt_traces: dict[int, TurnTrace] = {}
        self._committed: set[int] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.completed: list[dict] = []
        self.hooks: TurnHooks | None = None
        self._deferred: dict[int, tuple[list[ToolCall], str]] = {}  # gen -> tool calls waiting for commit

    # ---- events -------------------------------------------------------
    async def _emit(self, event: dict) -> None:
        if self._emit_cb:
            res = self._emit_cb(event)
            if asyncio.iscoroutine(res):
                await res

    async def _set_state(self, state: str) -> None:
        self.state = state
        await self._emit({"type": "state", "state": state, "gen": self.active_gen})

    def player_event(self, kind: str, gen: int, t_ns: int) -> None:
        """Called from the audio thread."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._on_player_event, kind, gen, t_ns)

    def _on_player_event(self, kind: str, gen: int, t_ns: int) -> None:
        trace = self._trace
        if kind == "start" and trace and trace.generation_id == gen and trace.t_playback_start_ns is None:
            trace.t_playback_start_ns = t_ns
            asyncio.ensure_future(self._emit({"type": "playback.start", "gen": gen}))
        elif kind == "end" and trace and trace.generation_id == gen:
            trace.t_playback_end_ns = t_ns
            if self._playback_done and not self._playback_done.done():
                self._playback_done.set_result(True)
        elif kind == "halted" and gen in self._halt_traces:
            halted = self._halt_traces.pop(gen)
            halted.t_playback_halted_ns = t_ns
            self._finish(halted)

    # ---- public API ---------------------------------------------------
    def cancel(self, reason: str = "user") -> int | None:
        """Stop current response immediately. Returns the cancelled generation id, if any."""
        if not self._task or self._task.done():
            return None
        gen = self.active_gen
        trace = self._trace
        if trace:
            trace.t_cancel_ns = now_ns()
            trace.cancelled = True
            trace.error = trace.error or f"cancelled:{reason}"
            self._halt_traces[gen] = trace
        self.active_gen = 0
        self._deferred.pop(gen, None)
        self.player.stop_now(gen)
        self._task.cancel()
        self._trace = None
        return gen

    def _is_uncommitted(self) -> bool:
        return bool(self.active_gen) and self.active_gen not in self._committed

    def commit(self, gen: int, t_endpoint_ns: int) -> bool:
        """Confirm a speculative turn: its held audio may now play."""
        if gen != self.active_gen or not self._trace:
            return False
        self._trace.t_endpoint_ns = t_endpoint_ns
        self._committed.add(gen)
        self.player.release(gen)
        if gen in self._deferred:
            asyncio.ensure_future(self._run_calls(*self._deferred.pop(gen)))
        return True

    async def stop(self, reason: str = "user") -> None:
        if self.cancel(reason) is not None:
            await self._set_state("listening")

    def submit(self, pcm: np.ndarray, sample_rate: int, t_speech_end_ns: int | None,
               t_release_ns: int | None, source: str = "mic", t_speech_start_ns: int | None = None,
               held: bool = False, mode: str = "ptt") -> asyncio.Task:
        """Start a response for an utterance.

        held=False: the turn is committed now (push-to-talk release, fixture).
        held=True: speculative. Everything runs, but playback stays gated until commit(gen);
        if the user resumes speaking, the caller cancels it and nothing is ever heard.
        """
        self._loop = asyncio.get_running_loop()
        self.cancel("speculation_dropped" if self._is_uncommitted() else "new_turn")
        gen = next(self._gen_ids)
        self.active_gen = gen
        self.player.begin_generation(gen, held)
        if not held:
            self._committed.add(gen)
        trace = TurnTrace(
            session_id=self.session_id, utterance_id=uuid.uuid4().hex[:8], generation_id=gen,
            source=source, mode=mode, speculative=held, t_speech_start_ns=t_speech_start_ns,
            t_speech_end_ns=t_speech_end_ns or t_release_ns, t_ptt_release_ns=t_release_ns,
            asr_model=self.asr.model, llm_model=self.llm.model, tts_model=self.voice.model or self.tts.model,
            voice_profile=self.voice.name, output_device=self.output_device,
        )
        self._trace = trace
        self._task = asyncio.create_task(self._run_turn(trace, pcm, sample_rate))
        return self._task

    def say(self, text: str, source: str = "proactive") -> asyncio.Task | None:
        """Speak a line she decided on herself (no ASR, no LLM). Refused (None) while a turn is active;
        callers also check that あきら isn't speaking. Esc / barge-in cancel it like any turn."""
        if self.active_gen or self.state != "listening" or not text.strip():
            return None
        self._loop = asyncio.get_running_loop()
        gen = next(self._gen_ids)
        self.active_gen = gen
        self.player.begin_generation(gen, False)
        self._committed.add(gen)
        trace = TurnTrace(
            session_id=self.session_id, utterance_id=uuid.uuid4().hex[:8], generation_id=gen,
            source=source, mode="proactive", t_speech_end_ns=now_ns(), assistant_text=text.strip(),
            llm_model=self.llm.model, tts_model=self.voice.model or self.tts.model,
            voice_profile=self.voice.name, output_device=self.output_device,
        )
        self._trace = trace
        self._task = asyncio.create_task(self._run_say(trace))
        return self._task

    def notify(self, prompt: str, source: str = "notice") -> asyncio.Task | None:
        """Have her say `prompt` (an event, not something あきら said) in her own words through the reply LLM.
        Same refusal rules as say(); the prompt is not added to the history, her line is."""
        if self.active_gen or self.state != "listening" or not prompt.strip():
            return None
        self._loop = asyncio.get_running_loop()
        gen = next(self._gen_ids)
        self.active_gen = gen
        self.player.begin_generation(gen, False)
        self._committed.add(gen)
        trace = TurnTrace(
            session_id=self.session_id, utterance_id=uuid.uuid4().hex[:8], generation_id=gen,
            source=source, mode="notice", t_speech_end_ns=now_ns(), notice=prompt.strip(),
            llm_model=self.llm.model, tts_model=self.voice.model or self.tts.model,
            voice_profile=self.voice.name, output_device=self.output_device,
        )
        self._trace = trace
        self._task = asyncio.create_task(self._run_say(trace, self._notice_stage))
        return self._task

    # ---- turn pipeline ------------------------------------------------
    async def _run_turn(self, trace: TurnTrace, pcm: np.ndarray, sample_rate: int) -> None:
        gen = trace.generation_id
        try:
            await self._set_state("committing")
            trace.mark("asr_request")
            final = await self.asr.transcribe(pcm, sample_rate)
            trace.mark("asr_final")
            trace.user_text, trace.user_language = final.text, final.language
            await self._emit({"type": "asr.final", "gen": gen, "text": final.text, "language": final.language})
            if not final.text:
                trace.error = "empty_transcript"
                self._finish(trace)
                return

            await self._speak(trace, self._llm_stage)
            if gen in self._committed:
                self.history += [ChatTurn("user", trace.user_text), ChatTurn("assistant", trace.assistant_text)]
            self.history = self.history[-2 * self.history_turns:]
            self._finish(trace)
        except asyncio.CancelledError:
            # trace is finished by the 'halted' playback event (see cancel())
            if trace.user_text and gen in self._committed:
                self.history += [ChatTurn("user", trace.user_text),
                                 ChatTurn("assistant", trace.assistant_text + "（中断）")]
            raise
        except Exception as e:  # stage failure: report, recover, keep the app alive
            trace.error = f"{type(e).__name__}: {e}"
            self.player.stop_now(gen)
            await self._emit({"type": "error", "gen": gen, "message": trace.error})
            self._finish(trace)
        finally:
            if self.active_gen == gen:
                self.active_gen = 0
                await self._set_state("listening")

    async def _speak(self, trace: TurnTrace, produce) -> None:
        """Run `produce(trace, clauses)` into streaming TTS and wait until the audio has played."""
        await self._set_state("responding")
        self._playback_done = asyncio.get_running_loop().create_future()
        clauses: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)
        tts_task = asyncio.create_task(self._tts_worker(trace, clauses))
        try:
            await produce(trace, clauses)
            await tts_task
        finally:
            tts_task.cancel()
        if trace.t_tts_first_audio_ns is not None:
            self.player.finish_generation(trace.generation_id)
            await asyncio.wait_for(self._playback_done, timeout=60)

    async def _fixed_text_stage(self, trace: TurnTrace, clauses: asyncio.Queue) -> None:
        seg = ClauseSegmenter()
        for clause in [*seg.push(trace.assistant_text), *seg.flush()]:
            await self._put_clause(trace, clauses, clause)
        await clauses.put(None)

    async def _run_say(self, trace: TurnTrace, stage=None) -> None:
        gen = trace.generation_id
        try:
            await self._speak(trace, stage or self._fixed_text_stage)
            self.history = [*self.history, ChatTurn("assistant", trace.assistant_text)][-2 * self.history_turns:]
            self._finish(trace)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # same recovery as a normal turn
            trace.error = f"{type(e).__name__}: {e}"
            self.player.stop_now(gen)
            await self._emit({"type": "error", "gen": gen, "message": trace.error})
            self._finish(trace)
        finally:
            if self.active_gen == gen:
                self.active_gen = 0
                await self._set_state("listening")

    async def _notice_stage(self, trace: TurnTrace, clauses: asyncio.Queue) -> None:
        await self._llm_stage(trace, clauses, kind="notice")

    async def _llm_stage(self, trace: TurnTrace, clauses: asyncio.Queue, kind: str = "turn") -> None:
        seg = ClauseSegmenter()
        text = trace.notice if kind == "notice" else trace.user_text
        plan = await self.hooks.plan(text, kind) if self.hooks else TurnPlan()
        kwargs = {"context": plan.context, "tools": plan.tools} if self.hooks else {}
        calls: list[ToolCall] = []
        trace.mark("llm_request")
        async for delta in self.llm.stream_response(self.history, text, **kwargs):
            if delta.tool_call:
                calls.append(delta.tool_call)
                continue
            if delta.usage:
                trace.usage = delta.usage
                estimate = getattr(self.llm, "estimate_cost", None)
                trace.est_cost_usd = estimate(delta.usage) if estimate else None
            if not delta.text:
                continue
            if trace.t_llm_first_token_ns is None:
                trace.mark("llm_first_token")
            trace.assistant_text += delta.text
            await self._emit({"type": "llm.delta", "gen": trace.generation_id, "text": delta.text})
            for clause in seg.push(delta.text):
                await self._put_clause(trace, clauses, clause)
        calls = [*plan.calls, *(calls or plan.fallback_calls)]
        if calls and self.hooks:
            trace.tool_calls = [{"name": c.name, "arguments": c.arguments} for c in calls]
            await self._emit({"type": "tool_calls", "gen": trace.generation_id, "calls": trace.tool_calls})
            if not trace.assistant_text.strip() and (line := self.hooks.ack(calls)):
                trace.assistant_text = line  # she called a tool without a word: say a fixed acknowledgement
                for clause in seg.push(line):
                    await self._put_clause(trace, clauses, clause)
            gen = trace.generation_id
            if gen in self._committed:
                asyncio.ensure_future(self._run_calls(calls, trace.user_text))
            else:
                self._deferred[gen] = (calls, trace.user_text)
        for clause in seg.flush():
            await self._put_clause(trace, clauses, clause)
        await clauses.put(None)

    async def _run_calls(self, calls: list[ToolCall], user_text: str) -> None:
        try:
            await self.hooks.run(calls, user_text)
        except Exception as e:  # a failing side effect must not take the voice loop down
            await self._emit({"type": "error", "message": f"tool calls failed: {type(e).__name__}: {e}"})

    async def _put_clause(self, trace: TurnTrace, clauses: asyncio.Queue, clause: str) -> None:
        if trace.t_first_clause_ns is None:
            trace.mark("first_clause")
        trace.clauses.append(clause)
        await self._emit({"type": "clause", "gen": trace.generation_id, "text": clause})
        await clauses.put(clause)

    async def _tts_worker(self, trace: TurnTrace, clauses: asyncio.Queue) -> None:
        gen = trace.generation_id
        while (clause := await clauses.get()) is not None:
            if trace.t_tts_request_ns is None:
                trace.mark("tts_request")
            async for pcm in self.tts.stream_audio(clause, self.voice):
                if trace.t_tts_first_audio_ns is None:
                    trace.mark("tts_first_audio")
                if not self.player.enqueue(gen, pcm):
                    return  # generation no longer active: stop synthesizing

    def _finish(self, trace: TurnTrace) -> None:
        summary = trace.summary()
        record = self.metrics.write(trace) if self.metrics else {**asdict(trace), **summary}
        self.completed.append(record)
        payload = {"type": "turn.summary", "gen": trace.generation_id, "user_text": trace.user_text,
                   "assistant_text": trace.assistant_text, "cancelled": trace.cancelled,
                   "error": trace.error, "mode": trace.mode, **summary}
        asyncio.ensure_future(self._emit(payload))
