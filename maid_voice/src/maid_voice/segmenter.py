"""Turn a stream of LLM text deltas into speakable Japanese clauses.

The first clause is flushed eagerly (also at 、 once it is long enough) to cut time to first
sound; later clauses wait for sentence ends so TTS prosody is less choppy.
"""

SENTENCE_END = "。！？!?…"
SOFT_BREAK = "、，,"


class ClauseSegmenter:
    def __init__(self, first_min_chars: int = 6, max_chars: int = 60):
        self.first_min_chars = first_min_chars
        self.max_chars = max_chars
        self._buf = ""
        self._emitted = 0

    def push(self, text: str) -> list[str]:
        out = []
        for ch in text:
            self._buf += ch
            stripped = self._buf.strip()
            if ch in SENTENCE_END and stripped.strip(SENTENCE_END):
                out.append(self._take())
            elif ch in SOFT_BREAK and self._emitted == 0 and len(stripped) >= self.first_min_chars:
                out.append(self._take())
            elif len(stripped) >= self.max_chars:
                out.append(self._take())
        return out

    def flush(self) -> list[str]:
        return [self._take()] if self._buf.strip() else []

    def _take(self) -> str:
        clause, self._buf = self._buf.strip(), ""
        self._emitted += 1
        return clause
