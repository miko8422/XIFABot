"""What the harness env can do right now, as data (ARCHITECTURE §4.6).

Whatever wires a reporter or a runner registers it here. The Maid builds her 「できること」 prompt block
and her tool list from this, so she never claims something that isn't wired up and never denies something that is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Status = Literal["live", "simulated", "off"]


@dataclass
class Capability:
    name: str
    status: Status
    description: str             # English, one line: what she can do with it
    tools: tuple[str, ...] = ()  # tool names the Maid gets while this is live or simulated
    note: str = ""               # limits worth saying out loud, e.g. "results are not real until P3"


class CapabilityRegistry:
    def __init__(self) -> None:
        self._caps: dict[str, Capability] = {}

    def register(self, cap: Capability) -> Capability:
        self._caps[cap.name] = cap
        return cap

    def set_status(self, name: str, status: Status) -> None:
        if name in self._caps:
            self._caps[name].status = status

    def get(self, name: str) -> Capability | None:
        return self._caps.get(name)

    def all(self) -> list[Capability]:
        return list(self._caps.values())

    def usable(self) -> list[Capability]:
        return [c for c in self._caps.values() if c.status != "off"]
