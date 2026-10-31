"""Conversation log (ARCHITECTURE §4.4, P2): every turn that entered her history, in the Maid's SQLite file.

On startup the last turns are restored into the controller's history, and while the gap matters her
prompt says how long it has been and whether she was restarted, in steps from "brief restart" to
"days" (MaiBot's context-restore wakeup, RESPONSE_DESIGN §1). A long pause inside a running session
gets the same note, without the restart part.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from maid_voice.llm import ChatTurn

GAP_NOTE_S = 30 * 60.0  # inside a session, a pause shorter than this is just the conversation going on


@dataclass(frozen=True)
class LoggedTurn:
    t: float
    session_id: str
    role: str
    content: str
    mode: str


def elapsed_ja(seconds: float) -> str:
    """'brief restart' … 'days', in her words."""
    if seconds < 120:
        return "ほんの少し"
    if seconds < 3600:
        return f"{int(seconds // 60)}分くらい"
    if seconds < 6 * 3600:
        return f"{int(seconds // 3600)}時間くらい"
    if seconds < 86400:
        return "半日くらい"
    days = int(seconds // 86400)
    return "一日くらい" if days < 2 else f"{days}日"


class ConversationLog:
    def __init__(self, path: str | Path = ":memory:", clock=time.time) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS conversation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                t REAL NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                mode TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS conversation_t_idx ON conversation(t);
        """)
        self._db.commit()

    def append(self, session_id: str, turns: list[ChatTurn], mode: str) -> None:
        now = self.clock()
        with self._lock:
            self._db.executemany(
                "INSERT INTO conversation (t, session_id, role, content, mode) VALUES (?, ?, ?, ?, ?)",
                [(now, session_id, t.role, t.content, mode) for t in turns])
            self._db.commit()

    def recent(self, n: int) -> list[LoggedTurn]:
        with self._lock:
            rows = self._db.execute(
                "SELECT t, session_id, role, content, mode FROM conversation ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [LoggedTurn(*r) for r in reversed(rows)]

    def close(self) -> None:
        self._db.close()


class Conversation:
    """What the Maid's turn hooks use: record turns, restore them, and say how long it has been."""

    def __init__(self, log: ConversationLog, session_id: str, clock=time.time) -> None:
        self.log = log
        self.session_id = session_id
        self.clock = clock
        last = log.recent(1)
        self._last_t = last[0].t if last else None
        self._restarted = bool(last) and last[0].session_id != session_id
        self._noted = False  # the restart note is given until her first reply after it

    def restore(self, n: int) -> list[ChatTurn]:
        return [ChatTurn(t.role, t.content) for t in self.log.recent(n)]

    def remember(self, turns: list[ChatTurn], mode: str) -> None:
        self.log.append(self.session_id, turns, mode)
        if any(t.role == "assistant" for t in turns):
            self._noted = True
        self._last_t = self.clock()

    def note(self) -> str:
        """The elapsed-time note for the next reply call, or "" when the conversation is simply going on."""
        if self._last_t is None:
            return ""
        gap = self.clock() - self._last_t
        restart = self._restarted and not self._noted
        if not restart and gap < GAP_NOTE_S:
            return ""
        when = datetime.fromtimestamp(self._last_t).strftime("%m月%d日 %H:%M")
        head = "ワタシはしばらく止まっていて、さっき戻ってきた。" if restart else ""
        return (f"# 前回からの時間\n{head}最後に話したのは{elapsed_ja(gap)}前（{when}）。"
                "その時の会話は、下の履歴のとおり全部覚えている。あきらが「さっきの」と言ったら、履歴の話題のこと。"
                "少しの間なら何もなかったように続け、時間が空いたなら軽く触れてから続ける。"
                "離れていた間にあきらが何をしていたかは知らないので、作らない。")
