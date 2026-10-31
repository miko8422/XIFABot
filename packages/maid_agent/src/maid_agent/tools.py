"""The tools the Maid's reply LLM may call (ARCHITECTURE §4.2), and the prompt text that goes with them.

Routing happens inside the one reply call she already makes: she speaks a short acknowledgement and
calls a tool in the same response. No planner. Which tools she gets, and the 「できること」 block that
tells her what she can do, are both built from the harness capability list (§4.6).
"""

from __future__ import annotations

import time

from maid_harness.capabilities import Capability
from maid_harness.models import TaskRecord

TIERS = ("quick", "work", "computer")

_SCHEMAS = [
    {"type": "function", "function": {
        "name": "look_at_screen",
        "description": (
            "Look at the user's main screen right now (local vision model, a few seconds). Use it when the user "
            "asks what is on the screen or asks you to check something there. You answer after you have seen it."),
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "What to look for, in English."}}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "delegate",
        "description": (
            "Hand heavy work to a background sub-agent: web search, research, coding questions, changing files, "
            "running commands, operating apps. Not for chat or for questions about tasks already listed in the state."),
        "parameters": {"type": "object", "properties": {
            "goal": {"type": "string", "description": (
                "The task in English, self-contained, with every name, path and constraint the user gave. "
                "Write app and file names exactly as the user meant them.")},
            "tier": {"type": "string", "enum": list(TIERS), "description": (
                "quick: read-only, minutes (search, a coding question). work: may write files or run commands. "
                "computer: operates the GUI or apps. Call delegate right away for these too: the harness holds the "
                "task until the user confirms, so asking is not a reason to wait.")},
            "needs": {"type": "array", "items": {"type": "string"},
                      "description": "Capabilities needed, e.g. web, code, files, shell, gui."},
        }, "required": ["goal", "tier"]},
    }},
    {"type": "function", "function": {
        "name": "cancel_task",
        "description": "Stop a task from the state block when the user asks to stop or drop it.",
        "parameters": {"type": "object", "properties": {
            "task_id": {"type": "string"}}, "required": ["task_id"]},
    }},
    {"type": "function", "function": {
        "name": "answer_task",
        "description": "Pass the user's answer to a task that asked a question (あきらの答え待ち in the state).",
        "parameters": {"type": "object", "properties": {
            "task_id": {"type": "string"},
            "answer": {"type": "string", "description": "The user's answer in English, complete and self-contained."}},
            "required": ["task_id", "answer"]},
    }},
    {"type": "function", "function": {
        "name": "follow_up_task",
        "description": ("More instructions for a finished, failed or interrupted task from the state, continuing the same "
                        "sub-agent session: 'also check X', 'do it again with Y', 'continue the interrupted one'."),
        "parameters": {"type": "object", "properties": {
            "task_id": {"type": "string"},
            "message": {"type": "string", "description": "The instruction in English, self-contained."}},
            "required": ["task_id", "message"]},
    }},
    {"type": "function", "function": {
        "name": "confirm_task",
        "description": ("The user's yes or no to a task waiting for confirmation (確認待ち) or for permission (許可待ち)."),
        "parameters": {"type": "object", "properties": {
            "task_id": {"type": "string"}, "approve": {"type": "boolean"}}, "required": ["task_id", "approve"]},
    }},
]
SCHEMAS = {t["function"]["name"]: t for t in _SCHEMAS}
TOOLS = list(SCHEMAS.values())  # every tool, for the probe and tests


def tools_for(caps: list[Capability]) -> list[dict]:
    names = {n for c in caps if c.status != "off" for n in c.tools}
    return [t for name, t in SCHEMAS.items() if name in names]


_STATUS_LABEL = {"live": "使える", "simulated": "練習用"}


def capability_block(caps: list[Capability]) -> str:
    """She can do what is listed here, and nothing else. Generated, never hand-written."""
    lines = []
    for c in caps:
        if c.status == "off":
            continue
        line = f"- {c.name}（{_STATUS_LABEL[c.status]}）: {c.description}"
        if c.note:
            line += f" ({c.note})"
        lines.append(line)
    return ("# できること\n" + "\n".join(lines or ["（今は何もつながっていない）"]) + "\n"
            "ここに書いてあることは、できないと言わない。書いていないことは、できるふりをしない。"
            "「練習用」のものは使ってよいが、結果は本物ではないと正直に言う。"
            "何ができるか聞かれたら、この中から短く答える。")


INSTRUCTIONS = """# 作業の頼まれごと
検索、調べもの、コードの質問、ファイルの変更、コマンドの実行、アプリの操作のような重い作業は、自分ではしない。
そのときは、まず日本語でごく短く引き受けて（例「わかった、調べてくる。」）、同じ返答の中で delegate を呼ぶ。goal は英語で、あきらの言った名前や条件を全部入れる。
雑談や、下の「今の作業」でわかることには delegate を使わない。
画面やアプリを操作する作業も delegate を呼ぶ（tier は computer）。そのときは引き受けるのではなく「始めていい？」と聞く。作業は確認待ちになり、あきらが答えるまで始まらない。
あきらが作業をやめてと言ったら、止めると言うだけでなく必ず cancel_task を呼ぶ。確認待ちや許可待ちの作業にあきらが答えたら confirm_task を呼ぶ。
作業からの質問にあきらが答えたら answer_task を、終わった作業や中断した作業への追加の頼み（続けて、もう一度、ついでに）は新しい delegate ではなく follow_up_task を呼ぶ。
作業が終わったとか、アプリを開いたとか、報告がないことは言わない。進み具合は「今の作業」に書いてあることだけを伝える。
画面のことを聞かれたら look_at_screen を呼んで「ちょっと見る」とだけ言う。見る前に中身を言わない。答えは見たあとに伝える。
英語のアプリ名やファイル名は、カタカナで言う。"""

ACTIVE = ("accepted", "waiting_confirmation", "running", "needs_input", "needs_permission")
_STATUS_JA = {"accepted": "受付済み", "waiting_confirmation": "あきらの確認待ち", "running": "実行中",
              "needs_input": "あきらの答え待ち", "needs_permission": "あきらの許可待ち", "completed": "完了", "failed": "失敗", "cancelled": "中止",
              "interrupted": "再起動で中断"}


def _ago(seconds: float) -> str:
    return f"{int(seconds // 60)}分前" if seconds >= 60 else f"{int(seconds)}秒前"


def state_block(tasks: list[TaskRecord], now: float | None = None, recent: int = 3, facts: list[str] = ()) -> str:
    """The T1 lookup context: active tasks plus the last few finished ones, newest first."""
    now = time.time() if now is None else now
    active = [t for t in tasks if t.status in ACTIVE]
    done = sorted((t for t in tasks if t.status not in ACTIVE), key=lambda t: -t.updated_at)[:recent]
    lines = []
    for t in sorted(active, key=lambda t: -t.created_at) + done:
        line = f"- {t.task_id}: {t.goal} / {_STATUS_JA.get(t.status, t.status)} / 依頼 {_ago(now - t.created_at)}"
        if t.status == "running":
            line += f" / {t.progress}%" + (f" {t.progress_text}" if t.progress_text else "")
        if t.status == "completed" and t.summary:
            line += f" / 結果: {t.summary}"
        if t.status == "failed" and t.error:
            line += f" / 理由: {t.error}"
        if t.status == "needs_input" and t.question:
            line += f" / 質問: {t.question}"
        if t.status == "needs_permission" and (perm := t.metadata.get("permission")):
            line += f" / 頼まれている許可: {perm.get('what')}（理由: {perm.get('why')}）"
        lines.append(line)
    out = "# 今の作業\n" + ("\n".join(lines) if lines else "（なし）")
    if facts:
        out += "\n\n# 最近わかったこと（報告から。推測を含む）\n" + "\n".join(f"- {f}" for f in facts)
    return out
