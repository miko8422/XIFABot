"""Fast, deterministic Maid request triage.

Since P1 this is the fallback and test oracle: the reply LLM routes through the `delegate` tool
(ARCHITECTURE §4.2). `routing = "keyword"` in [agent] puts this router back in charge.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Any

from maid_harness.models import Route


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    confidence: float
    reason: str
    required_capabilities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["required_capabilities"] = list(self.required_capabilities)
        return out


_COMPUTER = (
    "打开", "打开应用", "启动应用", "点击", "鼠标", "键盘", "电脑操作", "屏幕操作",
    "開く", "開いて", "開けて", "起動", "立ち上げて", "クリック", "マウス操作", "キーボード操作", "閉じて", "再生して",
    "open the app", "click", "mouse", "keyboard", "computer use", "open application",
)
_HEAVY = (
    "搜索", "查找", "搜一下", "研究", "代码", "编程", "写一个", "修改", "编辑",
    "运行", "构建", "下载", "比较", "整理", "总结", "监控", "查资料", "任务",
    "検索", "調べ", "探して", "コード", "プログラム", "作って", "書いて", "変更", "編集", "直して", "修正",
    "実行", "走らせ", "ビルド", "テスト", "ダウンロード", "インストール", "比較", "まとめ", "要約", "監視", "リサーチ",
    "search", "research", "find", "coding", "code", "edit", "run", "build", "download",
    "compare", "summarize", "monitor", "look up", "go search",
)
_REPORTER = ("现在几点", "时间", "天气", "前台应用", "电量", "状态", "what time", "weather", "battery",
             "何時", "天気", "バッテリー", "どうなった", "進み具合", "終わった？")
_CANCEL = ("やめて", "止めて", "中止", "キャンセル", "やめといて", "もういい", "取り消して",
           "取消", "停止", "别做了", "cancel", "stop the task", "never mind", "forget it")
_YES = ("いいよ", "お願い", "うん", "はい", "やって", "どうぞ", "いいわ", "yes", "go ahead", "sure", "ok")
_NO = ("だめ", "ダメ", "やめて", "いや", "いらない", "待って", "no", "don't")


def _contains(text: str, words: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(word.casefold() in lowered for word in words)


def route_text(text: str) -> RouteDecision:
    """Return a routing decision without I/O, model calls, or tool execution."""
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return RouteDecision("clarify", 1.0, "empty request")
    if _contains(normalized, _COMPUTER):
        return RouteDecision("computer_use", 0.97, "requires direct computer interaction", ("computer.use",))
    if _contains(normalized, _REPORTER) and not _contains(normalized, _HEAVY):
        return RouteDecision("reporter_query", 0.9, "can be answered by a harness reporter", ("system.read",))
    if _contains(normalized, _HEAVY):
        capabilities = ("web.search",) if _contains(normalized, ("搜索", "查找", "検索", "調べ", "search", "find", "look up")) else ("task.run",)
        return RouteDecision("task", 0.88, "requires work outside the live voice turn", capabilities)
    if len(normalized) < 2 or normalized in {"帮我", "help me", "嗯", "喂"}:
        return RouteDecision("clarify", 0.8, "request has no actionable subject")
    return RouteDecision("direct", 0.96, "ordinary conversation")


def wants_cancel(text: str) -> bool:
    return _contains(text, _CANCEL)


def yes_no(text: str) -> bool | None:
    """A confirmation answer, or None when it is neither (no is checked first: 「やめて」 is a no)."""
    if _contains(text, _NO):
        return False
    return True if _contains(text, _YES) else None


def benchmark(samples: int = 1000) -> dict[str, float | int]:
    cases = ("你好，今天怎么样", "搜索一下本地的报告", "打开 Steam", "现在几点")
    values: list[float] = []
    for i in range(max(1, samples)):
        t0 = time.perf_counter_ns()
        route_text(cases[i % len(cases)])
        values.append((time.perf_counter_ns() - t0) / 1_000_000)
    values.sort()
    pick = lambda p: values[min(len(values) - 1, int(round((p / 100) * (len(values) - 1))))]
    return {"samples": len(values), "p50_ms": round(pick(50), 4), "p95_ms": round(pick(95), 4), "p99_ms": round(pick(99), 4), "max_ms": round(values[-1], 4)}
