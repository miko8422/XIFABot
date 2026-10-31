"""Event normalization, deduplication, expiry, and in-process delivery."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from .models import TaskEvent
from .registry import TaskRegistry


class EventInbox:
    def __init__(self, registry: TaskRegistry, clock: Callable[[], float] = time.time) -> None:
        self.registry = registry
        self.clock = clock
        self._listeners: list[Callable[[TaskEvent], None]] = []

    def subscribe(self, listener: Callable[[TaskEvent], None]) -> None:
        self._listeners.append(listener)

    def publish(
        self,
        event_type: str,
        source: str,
        payload: dict[str, Any] | None = None,
        *,
        task_id: str | None = None,
        priority: str = "normal",
        confidence: float = 1.0,
        sensitivity: str = "private",
        dedupe_key: str | None = None,
        ttl_seconds: float = 3600.0,
    ) -> TaskEvent:
        event = TaskEvent(
            event_id=str(uuid.uuid4()),
            type=event_type,
            source=source,
            task_id=task_id,
            occurred_at=self.clock(),
            priority=priority,
            confidence=max(0.0, min(1.0, confidence)),
            sensitivity=sensitivity,
            dedupe_key=dedupe_key,
            ttl_seconds=ttl_seconds,
            payload=payload or {},
        )
        stored = self.registry.add_event(event)
        # Duplicate events are not delivered twice, even to in-process listeners.
        if stored.event_id == event.event_id:
            for listener in tuple(self._listeners):
                listener(event)
        return stored

    def list(self, task_id: str | None = None, limit: int = 100) -> list[TaskEvent]:
        now = self.clock()
        return [event for event in self.registry.events(task_id, limit) if not event.expired(now)]
