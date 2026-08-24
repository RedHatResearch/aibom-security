"""Accept mode: start/config failures exit 0 with non-confirming VerificationResult."""

from __future__ import annotations

import argparse
import json

import pytest

from aibom_verifier import cli
from aibom_verifier.errors import CompareStartError
from aibom_verifier.nodes.verdict_synthesize import verdict_message
from aibom_verifier.types import ModelRef, RunResult, TestOutcome, VerificationResult

_ACCEPT_VERDICT = "insufficient_evidence"
_ACCEPT_MESSAGE = verdict_message(_ACCEPT_VERDICT, tests=[])


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    cli.add_arguments(parser)
    return parser.parse_args(argv)


def _expected_accept_payload(args: argparse.Namespace) -> dict[str, object]:
    payload = VerificationResult(
        target=args.target,
        base=args.base,
        verdict=_ACCEPT_VERDICT,
        message=_ACCEPT_MESSAGE,
    ).to_dict()
    payload["accept"] = True
    return payload


def test_add_arguments_accepts_flag():
    args = _parse(["org/model", "--accept"])
    assert args.accept is True


def test_add_arguments_accept_defaults_false():
    args = _parse(["org/model"])
    assert args.accept is False


def test_compare_start_error_with_accept_exits_0(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
):
    args = _parse(
        ["org/target-model", "--base", "org/base-model", "--cache-dir", str(tmp_path), "--accept"]
    )
    monkeypatch.delenv("AIBOM_ACCEPT", raising=False)

    def _raise(*a, **k):
        raise CompareStartError("repo_not_found", "missing")

    monkeypatch.setattr(cli, "run_compare", _raise)

    assert cli.run(args) == 0
    assert json.loads(capsys.readouterr().out) == _expected_accept_payload(args)


def test_aibom_accept_env_enables_accept_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
):
    args = _parse(["org/target-model", "--base", "org/base-model", "--cache-dir", str(tmp_path)])
    monkeypatch.setenv("AIBOM_ACCEPT", "1")

    def _raise(*a, **k):
        raise CompareStartError("gated_unauthenticated", "no access")

    monkeypatch.setattr(cli, "run_compare", _raise)

    assert cli.run(args) == 0
    assert json.loads(capsys.readouterr().out) == _expected_accept_payload(args)


def test_real_resolve_fail_with_accept_exits_0_and_emits_resolve_failed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
):
    from aibom_verifier.nodes import resolve_refs as resolve_refs_mod

    args = _parse(
        [
            "org/target-model",
            "--base",
            "org/base-model",
            "--cache-dir",
            str(tmp_path),
            "--accept",
        ]
    )
    monkeypatch.delenv("AIBOM_LOG_LEVEL", raising=False)
    monkeypatch.delenv("AIBOM_RUN_ID", raising=False)
    monkeypatch.delenv("AIBOM_ACCEPT", raising=False)

    def _raise_resolve(repo_id: str, revision: str | None = None, *, api=None):
        raise CompareStartError("gated_unauthenticated", "no access")

    monkeypatch.setattr(resolve_refs_mod, "resolve_commit", _raise_resolve)

    assert cli.run(args) == 0

    captured = capsys.readouterr()
    stderr_payloads = [json.loads(line) for line in captured.err.splitlines() if line]

    assert json.loads(captured.out) == _expected_accept_payload(args)
    assert [p["event"] for p in stderr_payloads] == ["run_start", "resolve_failed"]
    assert stderr_payloads[1]["error_code"] == "gated_unauthenticated"


def test_invalid_store_with_accept_exits_0_and_emits_run_failed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
):
    args = _parse(["org/model", "--store", "proxy", "--cache-dir", str(tmp_path)])
    monkeypatch.setenv("AIBOM_ACCEPT", "1")
    monkeypatch.delenv("AIBOM_LOG_LEVEL", raising=False)

    def _raise(**kwargs):
        raise ValueError("missing AIBOM_PG_DSN")

    monkeypatch.setattr(cli, "build_artifact_store", _raise)

    assert cli.run(args) == 0

    captured = capsys.readouterr()
    stderr_payloads = [json.loads(line) for line in captured.err.splitlines() if line]

    assert json.loads(captured.out) == _expected_accept_payload(args)
    assert [p["event"] for p in stderr_payloads] == ["run_failed"]
    assert stderr_payloads[0]["error_code"] == "invalid_store"
    assert stderr_payloads[0]["exit_code"] == 0


def test_invalid_log_level_with_accept_exits_0_and_emits_run_failed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    args = _parse(["org/model", "--accept"])
    monkeypatch.setenv("AIBOM_LOG_LEVEL", "TRACE")
    monkeypatch.delenv("AIBOM_ACCEPT", raising=False)
    monkeypatch.setattr(
        cli,
        "build_artifact_store",
        lambda **kwargs: pytest.fail("build_artifact_store should not run for invalid log level"),
    )

    assert cli.run(args) == 0

    captured = capsys.readouterr()
    stderr_payloads = [json.loads(line) for line in captured.err.splitlines() if line]

    assert json.loads(captured.out) == _expected_accept_payload(args)
    assert [p["event"] for p in stderr_payloads] == ["run_failed"]
    assert stderr_payloads[0]["error_code"] == "invalid_log_level"
    assert stderr_payloads[0]["exit_code"] == 0


def test_successful_compare_ignores_accept_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
):
    args = _parse(
        ["org/target-model", "--base", "org/base-model", "--cache-dir", str(tmp_path), "--accept"]
    )
    monkeypatch.delenv("AIBOM_ACCEPT", raising=False)
    fake = RunResult(
        target=ModelRef(repo_id="org/target-model", revision="main", sha="tsha"),
        base=ModelRef(repo_id="org/base-model", revision="main", sha="bsha"),
        base_source="cli",
        support_class="dense_supported",
        tests=[TestOutcome(test_id="resolve_refs", status="pass")],
        final_verdict="verified_derivative",
        cache={"hits": [], "misses": []},
    )
    monkeypatch.setattr(cli, "run_compare", lambda *a, **k: fake)

    assert cli.run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "verified_derivative"
    assert "accept" not in payload


def test_aibom_accept_yes_env_enables_accept_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
):
    args = _parse(["org/target-model", "--base", "org/base-model", "--cache-dir", str(tmp_path)])
    monkeypatch.setenv("AIBOM_ACCEPT", "yes")

    def _raise(*a, **k):
        raise CompareStartError("repo_not_found", "missing")

    monkeypatch.setattr(cli, "run_compare", _raise)

    assert cli.run(args) == 0
    assert json.loads(capsys.readouterr().out) == _expected_accept_payload(args)
