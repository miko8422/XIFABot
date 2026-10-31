"""P2: conversation log, restored after a restart with an elapsed-time note."""

import asyncio

import numpy as np
import pytest
from maid_voice.llm import ChatTurn
from maid_voice.metrics import now_ns

from maid_agent.conversation import Conversation, ConversationLog, elapsed_ja
from maid_agent.turns import MaidTurns
from test_p1 import build, say


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_elapsed_steps_from_brief_restart_to_days():
    assert [elapsed_ja(s) for s in (30, 600, 3 * 3600, 10 * 3600, 30 * 3600, 3 * 86400)] == [
        "ほんの少し", "10分くらい", "3時間くらい", "半日くらい", "一日くらい", "3日"]


def test_log_survives_reopening_and_keeps_order(tmp_path):
    db = tmp_path / "memory.sqlite3"
    log = ConversationLog(db)
    log.append("a", [ChatTurn("user", "おはよう"), ChatTurn("assistant", "ん、おはよ。")], "ptt")
    log.append("a", [ChatTurn("assistant", "作業、終わったって。")], "notice")
    log.close()
    again = ConversationLog(db)
    assert [(t.role, t.content, t.mode) for t in again.recent(10)] == [
        ("user", "おはよう", "ptt"), ("assistant", "ん、おはよ。", "ptt"), ("assistant", "作業、終わったって。", "notice")]
    assert [t.content for t in again.recent(2)] == ["ん、おはよ。", "作業、終わったって。"]


def test_note_after_a_restart_until_her_first_reply_then_only_for_long_gaps():
    clock = Clock()
    log = ConversationLog(clock=clock)
    assert Conversation(log, "s1", clock).note() == "", "first ever start: nothing to resume"
    log.append("s1", [ChatTurn("user", "ちょっと休憩する"), ChatTurn("assistant", "ん。")], "ptt")

    clock.t += 3 * 3600
    conv = Conversation(log, "s2", clock)
    note = conv.note()
    assert "止まっていて" in note and "3時間くらい前" in note
    conv.remember([ChatTurn("user", "戻った"), ChatTurn("assistant", "おかえり。")], "ptt")
    assert conv.note() == "", "she has picked the conversation back up"

    clock.t += 45 * 60  # a long pause inside the same session: time note, no restart claim
    note = conv.note()
    assert "45分くらい前" in note and "止まっていて" not in note


def test_a_brief_restart_still_says_she_was_away():
    clock = Clock()
    log = ConversationLog(clock=clock)
    log.append("s1", [ChatTurn("user", "続きね"), ChatTurn("assistant", "うん。")], "ptt")
    clock.t += 20
    note = Conversation(log, "s2", clock).note()
    assert "ほんの少し前" in note and "止まっていて" in note


@pytest.mark.asyncio
async def test_restart_mid_conversation_restores_history_and_the_note_reaches_the_reply_call(tmp_path):
    db = tmp_path / "memory.sqlite3"
    harness, ctl, asr, llm = build()
    ctl.hooks.conversation = Conversation(ConversationLog(db), ctl.session_id)
    llm.text = "カレーでいいんじゃない？"
    await say(ctl, asr, "今日の夕飯、何にしよう")

    # the Maid stops; a new process with a new session starts on the same file
    harness2, ctl2, asr2, llm2 = build()
    conv = Conversation(ConversationLog(db), ctl2.session_id)
    ctl2.hooks.conversation = conv
    ctl2.history = conv.restore(8)
    assert [(t.role, t.content) for t in ctl2.history] == [
        ("user", "今日の夕飯、何にしよう"), ("assistant", "カレーでいいんじゃない？")]
    llm2.text = "さっきのカレーの話？"
    await say(ctl2, asr2, "で、さっきの話だけど")
    assert "前回からの時間" in llm2.requests[-1]["context"] and "止まっていて" in llm2.requests[-1]["context"]
    assert len(conv.log.recent(10)) == 4
    llm2.text = "ん。"
    await say(ctl2, asr2, "うん")
    assert "前回からの時間" not in llm2.requests[-1]["context"]


@pytest.mark.asyncio
async def test_a_dropped_speculation_is_never_logged():
    harness, ctl, asr, llm = build()
    log = ConversationLog()
    ctl.hooks.conversation = Conversation(log, ctl.session_id)
    asr.text = "えっと"
    t = now_ns()
    ctl.submit(np.zeros(1600, dtype="float32"), 16000, t, None, held=True, mode="handsfree")
    await asyncio.sleep(0.05)
    await ctl.stop("speculation_dropped")
    await asyncio.sleep(0.05)
    assert log.recent(10) == [] and ctl.history == []


def test_turn_hooks_without_a_conversation_still_accept_remember():
    harness, ctl, asr, llm = build()
    assert isinstance(ctl.hooks, MaidTurns)
    ctl.hooks.remember([ChatTurn("assistant", "x")], "ptt")  # no log configured: a no-op, not an error
