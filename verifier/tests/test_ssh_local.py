"""Thin SSH localhost ExecutionBackend (mocked subprocess; no real sshd)."""

from __future__ import annotations

import json
import subprocess

import pytest

from aibom_verifier.backends.ssh_local import SshLocalBackend
from aibom_verifier.slots.artifact_store import InMemoryArtifactStore
from aibom_verifier.types import TestOutcome


def test_ssh_local_argv_starts_with_ssh_localhost_and_run_test():
    backend = SshLocalBackend(
        store_dir="/tmp/cache",
        store="filesystem",
        ignore_cache=True,
    )
    argv = backend.build_argv("block0_shapes", {"target_repo": "org/m", "keep": 1})

    assert argv[0] == "ssh"
    assert argv[1] == "localhost"
    remote = argv[2]
    assert "python -m aibom_verifier.run_test" in remote
    assert "--node-id block0_shapes" in remote
    assert "--store-dir /tmp/cache" in remote
    assert "--store filesystem" in remote
    assert "--ignore-cache" in remote
    assert "org/m" in remote


def test_ssh_local_run_parses_outcome_json():
    outcome = TestOutcome(
        test_id="stub",
        status="pass",
        detail={"ok": True},
    )
    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=json.dumps(outcome.to_dict()),
            stderr="",
        )

    backend = SshLocalBackend(runner=fake_run)
    result = backend.run("stub", {"x": 1}, InMemoryArtifactStore())

    assert result.test_id == "stub"
    assert result.status == "pass"
    assert result.detail == {"ok": True}
    assert captured["argv"][0] == "ssh"
    assert captured["argv"][1] == "localhost"
    assert "python -m aibom_verifier.run_test" in captured["argv"][2]
    assert captured["timeout"] == 600


def test_ssh_local_run_timeout_raises():
    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout", 1))

    backend = SshLocalBackend(runner=fake_run, timeout=1)
    with pytest.raises(RuntimeError, match="timed out"):
        backend.run("stub", {}, InMemoryArtifactStore())


def test_ssh_local_run_nonzero_exit_raises():
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            args=argv,
            returncode=1,
            stdout="",
            stderr='{"ok": false, "error": "run_test_failed"}',
        )

    backend = SshLocalBackend(runner=fake_run)
    with pytest.raises(RuntimeError, match="exit 1"):
        backend.run("stub", {}, InMemoryArtifactStore())


def test_ssh_local_run_non_json_stdout_raises():
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout="not-json",
            stderr="",
        )

    backend = SshLocalBackend(runner=fake_run)
    with pytest.raises(RuntimeError, match="non-JSON"):
        backend.run("stub", {}, InMemoryArtifactStore())


def test_ssh_local_run_missing_test_id_raises():
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=json.dumps({"status": "pass"}),
            stderr="",
        )

    backend = SshLocalBackend(runner=fake_run)
    with pytest.raises(RuntimeError, match="TestOutcome"):
        backend.run("stub", {}, InMemoryArtifactStore())
