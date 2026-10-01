"""aibom-serve CLI: argument parsing, wiring, and the real ``python -m aibom`` seam."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aibom_server import cli

ENV_VARS = ("AIBOM_BIND_HOST", "AIBOM_BIND_PORT", "AIBOM_RUNS_DIR", "AIBOM_API_TOKEN")


def test_parser_defaults_come_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("AIBOM_BIND_HOST", "0.0.0.0")
    monkeypatch.setenv("AIBOM_BIND_PORT", "9999")
    monkeypatch.setenv("AIBOM_RUNS_DIR", "/tmp/x")
    monkeypatch.setenv("AIBOM_API_TOKEN", "tok")

    args = cli.build_parser().parse_args([])

    assert args.host == "0.0.0.0"
    assert args.port == 9999
    assert args.runs_dir == "/tmp/x"
    assert args.token == "tok"


def test_parser_defaults_without_environment(monkeypatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    args = cli.build_parser().parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 8080
    assert args.runs_dir == "runs"
    assert args.token is None


def test_flags_override_environment(monkeypatch) -> None:
    monkeypatch.setenv("AIBOM_BIND_PORT", "9999")
    monkeypatch.setenv("AIBOM_API_TOKEN", "tok")

    args = cli.build_parser().parse_args(["--port", "7777", "--token", "other"])

    assert args.port == 7777
    assert args.token == "other"


def test_empty_token_flag_refuses_to_start(monkeypatch, capsys) -> None:
    # `docker run -e AIBOM_API_TOKEN` with a missing secret exports the empty
    # string; the server must fail fast instead of silently disabling auth.
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--token", ""])

    assert excinfo.value.code == 2
    assert "refusing to start with auth disabled" in capsys.readouterr().err


def test_empty_token_env_refuses_to_start(monkeypatch) -> None:
    monkeypatch.setenv("AIBOM_API_TOKEN", "")

    with pytest.raises(SystemExit) as excinfo:
        cli.main([])

    assert excinfo.value.code == 2


def test_main_wires_runner_app_and_uvicorn(monkeypatch, tmp_path, capsys) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    recorded: dict[str, Any] = {}

    class FakeJobRunner:
        def __init__(
            self,
            runs_dir: str,
            *,
            concurrency: int,
            queue_size: int,
            timeout_seconds: float,
        ) -> None:
            recorded["runner_args"] = {
                "runs_dir": runs_dir,
                "concurrency": concurrency,
                "queue_size": queue_size,
                "timeout_seconds": timeout_seconds,
            }
            self.runs_dir = Path(runs_dir)

    app_sentinel = object()

    def fake_create_app(*, runner: FakeJobRunner, token: str | None) -> object:
        recorded["app_args"] = {"runner": runner, "token": token}
        return app_sentinel

    uvicorn_calls: list[dict[str, Any]] = []

    def fake_run(app: object, *, host: str, port: int) -> None:
        uvicorn_calls.append({"app": app, "host": host, "port": port})

    monkeypatch.setattr(cli, "JobRunner", FakeJobRunner)
    monkeypatch.setattr(cli, "create_app", fake_create_app)
    monkeypatch.setattr(cli, "uvicorn", SimpleNamespace(run=fake_run))

    runs_dir = tmp_path / "runs"
    rc = cli.main(
        [
            "--runs-dir",
            str(runs_dir),
            "--concurrency",
            "3",
            "--queue-size",
            "7",
            "--timeout-secs",
            "99",
        ]
    )

    assert rc == 0
    assert Path(recorded["runner_args"]["runs_dir"]) == runs_dir
    assert recorded["runner_args"]["concurrency"] == 3
    assert recorded["runner_args"]["queue_size"] == 7
    assert recorded["runner_args"]["timeout_seconds"] == 99.0
    assert recorded["app_args"]["runner"].runs_dir == runs_dir
    assert recorded["app_args"]["token"] is None
    assert len(uvicorn_calls) == 1
    assert uvicorn_calls[0]["app"] is app_sentinel
    assert uvicorn_calls[0]["host"] == "127.0.0.1"
    assert uvicorn_calls[0]["port"] == 8080
    out = capsys.readouterr().out
    assert "runs dir" in out
    assert str(runs_dir.resolve()) in out


def test_python_m_aibom_verify_help_seam() -> None:
    # JobRunner shells out to `python -m aibom verify ...`; nothing else in the
    # suite executes that command, so pin the cheapest real shape here. cli.py
    # lives at <repo>/server/src/aibom_server/cli.py, so parents[3] is the root
    # holding cli/src and verifier/src.
    repo_root = Path(cli.__file__).resolve().parents[3]
    pythonpath = os.pathsep.join(
        [str(repo_root / "cli" / "src"), str(repo_root / "verifier" / "src")]
    )
    if "PYTHONPATH" in os.environ:
        pythonpath += os.pathsep + os.environ["PYTHONPATH"]

    proc = subprocess.run(
        [sys.executable, "-m", "aibom", "verify", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": pythonpath},
    )

    assert proc.returncode == 0
    assert "usage" in proc.stdout
