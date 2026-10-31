"""Small, serializable records shared by the Maid and the harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Route = Literal["direct", "task", "reporter_query", "computer_use", "clarify", "refuse"]
TaskStatus = Literal[
    "accepted",
    "waiting_confirmation",
    "running",
    "needs_input",
    "completed",
    "failed",
    "cancelled",
    "interrupted",  # was running when the Maid stopped (ARCHITECTURE §6)
]


@dataclass(frozen=True)
class TaskEvent:
    event_id: str
    type: str
    source: str
    occurred_at: float
    task_id: str | None = None
    priority: str = "normal"
    confidence: float = 1.0
    sensitivity: str = "private"
    dedupe_key: str | None = None
    ttl_seconds: float = 3600.0
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def expired(self, now: float) -> bool:
        return self.ttl_seconds >= 0 and now > self.occurred_at + self.ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRecord:
    task_id: str
    request: str
    route: Route
    owner_id: str
    worker_kind: str
    status: TaskStatus
    created_at: float
    updated_at: float
    progress: int = 0
    progress_text: str = ""
    summary: str | None = None
    error: str | None = None
    question: str | None = None
    requires_confirmation: bool = False
    confirmed: bool = False
    merged_into: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
