"""Observer interface for structured run logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypedDict

from aibom_verifier.run_log import emit

_ERROR_EVENTS = {"run_failed", "resolve_failed", "exception"}


class RunObserver(Protocol):
    def on_event(self, event: str, *, logger: str, **fields: object) -> None: ...


class StderrJsonlObserver:
    def on_event(self, event: str, *, logger: str, **fields: object) -> None:
        level = "ERROR" if event in _ERROR_EVENTS else "INFO"
        emit(event, logger=logger, level=level, **fields)


class RecordedEvent(TypedDict):
    event: str
    logger: str
    fields: dict[str, object]


@dataclass
class RecordingObserver:
    events: list[RecordedEvent] = field(default_factory=list)

    def on_event(self, event: str, *, logger: str, **fields: object) -> None:
        self.events.append({"event": event, "logger": logger, "fields": dict(fields)})


def safe_on_event(
    observer: RunObserver | None,
    event: str,
    *,
    logger: str,
    **fields: object,
) -> None:
    """Deliver one event; never let observer/logging failures fail the verify run."""
    if observer is None:
        return
    try:
        observer.on_event(event, logger=logger, **fields)
    except Exception:
        return
