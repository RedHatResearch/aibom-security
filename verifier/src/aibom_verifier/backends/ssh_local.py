"""Thin SSH ExecutionBackend: ``ssh`` → ``python -m aibom_verifier.run_test``.

Ignores the in-process ``store``; the remote rebuilds from env and optional
``--store-dir`` / ``--store`` / ``--ignore-cache`` on the remote command
(system ``ssh`` CLI). Proxy store needs ``AIBOM_*`` set in the remote
session (OpenSSH does not forward the local process env by default);
filesystem + shared ``--store-dir`` is the usual localhost SSH path.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Callable

from aibom_verifier.run_test import build_run_test_argv
from aibom_verifier.slots.artifact_store import ArtifactStore
from aibom_verifier.types import TestOutcome, outcome_from_remote_dict

RunFn = Callable[..., subprocess.CompletedProcess[str]]


class SshLocalBackend:
    """Run one node via ``ssh <host>`` invoking ``python -m aibom_verifier.run_test``."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        python: str = "python",
        store_dir: str | None = None,
        store: str | None = None,
        ignore_cache: bool = False,
        ssh_bin: str = "ssh",
        runner: RunFn | None = None,
    ) -> None:
        self._host = host
        self._python = python
        self._store_dir = store_dir
        self._store = store
        self._ignore_cache = ignore_cache
        self._ssh_bin = ssh_bin
        self._runner: RunFn = runner or subprocess.run

    def build_argv(self, node_id: str, inputs: dict) -> list[str]:
        """Build ``ssh`` argv (inputs must already omit non-JSON ``api``)."""
        remote = build_run_test_argv(
            node_id,
            inputs,
            python=self._python,
            store_dir=self._store_dir,
            store=self._store,
            ignore_cache=self._ignore_cache,
        )
        return [self._ssh_bin, self._host, shlex.join(remote)]

    def run(self, node_id: str, inputs: dict, store: ArtifactStore) -> TestOutcome:
        _ = store  # remotes rebuild from env / remote flags
        argv = self.build_argv(node_id, inputs)
        completed = self._runner(
            argv,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise RuntimeError(
                f"ssh run_test failed (exit {completed.returncode})"
                + (f": {stderr}" if stderr else "")
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"ssh run_test returned non-JSON stdout: {exc}") from exc
        return outcome_from_remote_dict(payload, context="ssh run_test stdout")


__all__ = ["SshLocalBackend"]
