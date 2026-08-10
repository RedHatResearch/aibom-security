"""Structured stderr JSONL helpers for verify runs."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from time import perf_counter_ns

from aibom_verifier import __version__
from aibom_verifier.types import POLICY_VERSION

_RUN_ID: ContextVar[str | None] = ContextVar("aibom_run_id", default=None)
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}
# Process-global threshold: one CLI verify invocation per process.
_configured_level = logging.INFO

_ENVELOPE_KEYS = frozenset(
    {"ts", "level", "logger", "event", "run_id", "policy_version", "tool_version"}
)


def set_run_id(run_id: str) -> None:
    _RUN_ID.set(run_id)


def get_run_id() -> str:
    run_id = _RUN_ID.get()
    if run_id is None:
        raise RuntimeError("run_id is not set")
    return run_id


def validate_log_level(level: str) -> str:
    if level not in _LOG_LEVELS:
        valid_levels = ", ".join(_LOG_LEVELS)
        raise ValueError(f"Invalid AIBOM_LOG_LEVEL {level!r}. Expected one of: {valid_levels}")
    return level


def configure_logging(level: str) -> None:
    global _configured_level
    _configured_level = _LOG_LEVELS[validate_log_level(level)]


def elapsed_ms(start_ns: int) -> int:
    return (perf_counter_ns() - start_ns) // 1_000_000


def emit(event: str, *, logger: str, level: str, **fields: object) -> None:
    """Write one JSONL event to stderr; swallow encode/IO failures."""
    validated_level = validate_log_level(level)
    if _LOG_LEVELS[validated_level] < _configured_level:
        return
    run_id = _RUN_ID.get()
    if run_id is None:
        return

    payload = {key: value for key, value in fields.items() if key not in _ENVELOPE_KEYS}
    payload.update(
        {
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": validated_level,
            "logger": logger,
            "event": event,
            "run_id": run_id,
            "policy_version": POLICY_VERSION,
            "tool_version": __version__,
        }
    )
    try:
        sys.stderr.write(json.dumps(payload) + "\n")
    except (OSError, TypeError, ValueError):
        return
