"""Structured stderr JSONL helpers for verify runs."""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import suppress
from contextvars import ContextVar
from datetime import UTC, datetime
from time import perf_counter_ns
from uuid import uuid4

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


def resolve_run_id(explicit: str | None = None) -> str:
    for candidate in (explicit, os.environ.get("AIBOM_RUN_ID")):
        stripped = (candidate or "").strip()
        if stripped:
            return stripped
    return str(uuid4())


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
    """Write one JSONL event to stderr and optional AIBOM_LOG_FILE; swallow encode/IO failures."""
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
        line = json.dumps(payload) + "\n"
    except (TypeError, ValueError):
        return
    with suppress(OSError):
        sys.stderr.write(line)
    _tee_log_file(line)


def _tee_log_file(line: str) -> None:
    path = os.environ.get("AIBOM_LOG_FILE", "").strip()
    if not path:
        return
    with suppress(OSError), open(path, "a", encoding="utf-8") as handle:
        handle.write(line)
