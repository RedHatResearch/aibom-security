"""Execution backends package (local / SSH / Compose)."""

from __future__ import annotations

from aibom_verifier.backends.compose_queue import ComposeQueueBackend
from aibom_verifier.backends.local import LocalBackend
from aibom_verifier.backends.ssh_local import SshLocalBackend

__all__ = ["ComposeQueueBackend", "LocalBackend", "SshLocalBackend"]
