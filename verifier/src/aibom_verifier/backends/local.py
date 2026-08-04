"""In-process ExecutionBackend (default verify path)."""

from __future__ import annotations

from aibom_verifier.slots.worker import LocalWorker

LocalBackend = LocalWorker

__all__ = ["LocalBackend"]
