import argparse
import json
from uuid import UUID

import pytest
from hf_fakes import (
    fake_fetch_tensor_bytes,
    fake_list_tensor_names,
    fake_load_config_json,
    fake_resolve_commit,
    fake_tensor_shapes,
    read_stderr_jsonl,
)

from aibom_verifier import cli
from aibom_verifier.errors import CompareStartError
from aibom_verifier.mapping.block0 import BLOCK0_PREFIX, REQUIRED_SUFFIXES
from aibom_verifier.nodes import block0_shapes as block0_shapes_mod
from aibom_verifier.nodes import block0_values as block0_values_mod
from aibom_verifier.nodes import resolve_refs as resolve_refs_mod
from aibom_verifier.observer import StderrJsonlObserver
from aibom_verifier.types import ModelRef, RunResult, TestOutcome, VerificationResult

REQUIRED_NAMES = [BLOCK0_PREFIX + suffix for suffix in REQUIRED_SUFFIXES]
TARGET_REPO = "org/target-model"
BASE_REPO = "org/base-model"
TARGET_SHA = "target-sha"
BASE_SHA = "base-sha"


def _configure_real_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    shas = {TARGET_REPO: TARGET_SHA, BASE_REPO: BASE_SHA}
    configs = {
        TARGET_REPO: {"model_type": "llama"},
        BASE_REPO: {"model_type": "llama"},
    }
    default_shapes = {name: [4] for name in REQUIRED_NAMES}
    default_bytes = {name: b"identical-payload" for name in REQUIRED_NAMES}

    monkeypatch.setattr(resolve_refs_mod, "resolve_commit", fake_resolve_commit(shas))
    monkeypatch.setattr(resolve_refs_mod, "load_config_json", fake_load_config_json(configs))
    monkeypatch.setattr(
        resolve_refs_mod,
        "resolve_presumed_base",
        lambda repo_id, sha, *, api=None: BASE_REPO,
    )
    monkeypatch.setattr(
        block0_shapes_mod,
        "list_tensor_names",
        fake_list_tensor_names({TARGET_REPO: REQUIRED_NAMES, BASE_REPO: REQUIRED_NAMES}),
    )
    monkeypatch.setattr(
        block0_shapes_mod,
        "tensor_shapes",
        fake_tensor_shapes({TARGET_REPO: default_shapes, BASE_REPO: default_shapes}),
    )
    monkeypatch.setattr(
        block0_values_mod,
        "fetch_tensor_bytes",
        fake_fetch_tensor_bytes({TARGET_REPO: default_bytes, BASE_REPO: default_bytes}),
    )


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    cli.add_arguments(parser)
    return parser.parse_args(argv)


def _read_stderr_lines(capsys: pytest.CaptureFixture[str]) -> list[dict[str, object]]:
    return read_stderr_jsonl(capsys)


def _fake_run_result() -> RunResult:
    return RunResult(
        target=ModelRef(repo_id="org/target-model", revision="main", sha="target-sha"),
        base=ModelRef(repo_id="org/base-model", revision="main", sha="base-sha"),
        base_source="cli",
        support_class="dense_supported",
        tests=[
            TestOutcome(test_id="resolve_refs", status="pass"),
            TestOutcome(
                test_id="support_classify",
                status="pass",
                compatibility="compatible",
                detail={
                    "support_class": "dense_supported",
                    "target_model_type": "llama",
                    "base_model_type": "llama",
                },
            ),
            TestOutcome(
                test_id="block0_shapes",
                status="skip",
                skipped_because={"upstream": "support_classify", "reason": "gated"},
            ),
            TestOutcome(
                test_id="block0_values",
                status="error",
                reason_codes=["hub_error"],
            ),
        ],
        final_verdict="insufficient_evidence",
        cache={"hits": [], "misses": []},
    )


def test_invalid_log_level_emits_run_failed_and_skips_store_build(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    args = _parse(["org/target-model"])
    monkeypatch.setenv("AIBOM_LOG_LEVEL", "TRACE")
    monkeypatch.setattr(
        cli,
        "build_artifact_store",
        lambda **kwargs: pytest.fail("build_artifact_store should not run for invalid log level"),
    )

    assert cli.run(args) == 1

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    stderr_payloads = [json.loads(line) for line in captured.err.splitlines() if line]

    assert stdout_payload["ok"] is False
    assert stdout_payload["error"] == "invalid_log_level"
    assert "Invalid AIBOM_LOG_LEVEL" in stdout_payload["message"]
    assert [payload["event"] for payload in stderr_payloads] == ["run_failed"]
    assert stderr_payloads[0]["error_code"] == "invalid_log_level"
    assert "Invalid AIBOM_LOG_LEVEL" in str(stderr_payloads[0]["message"])
    assert stderr_payloads[0]["logger"] == "aibom_verifier.cli"
    assert stderr_payloads[0]["exit_code"] == 1
    assert isinstance(stderr_payloads[0]["duration_ms"], int)
    UUID(str(stderr_payloads[0]["run_id"]))


def test_runtime_store_failure_emits_run_failed_without_run_start(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    args = _parse(["org/target-model", "--store", "proxy"])
    monkeypatch.delenv("AIBOM_LOG_LEVEL", raising=False)

    def _raise_store_error(**kwargs):
        raise ValueError("missing AIBOM_PG_DSN")

    monkeypatch.setattr(cli, "build_artifact_store", _raise_store_error)

    assert cli.run(args) == 1

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    stderr_payloads = [json.loads(line) for line in captured.err.splitlines() if line]

    assert stdout_payload["ok"] is False
    assert stdout_payload["error"] == "invalid_store"
    assert stdout_payload["message"] == "missing AIBOM_PG_DSN"
    assert [payload["event"] for payload in stderr_payloads] == ["run_failed"]
    assert stderr_payloads[0]["error_code"] == "invalid_store"
    assert stderr_payloads[0]["message"] == "missing AIBOM_PG_DSN"
    assert stderr_payloads[0]["exit_code"] == 1
    UUID(str(stderr_payloads[0]["run_id"]))


def test_real_happy_path_emits_full_sequence_and_public_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
):
    args = _parse([TARGET_REPO, "--cache-dir", str(tmp_path)])
    monkeypatch.delenv("AIBOM_LOG_LEVEL", raising=False)
    monkeypatch.delenv("AIBOM_STORE", raising=False)
    _configure_real_happy_path(monkeypatch)

    assert cli.run(args) == 0

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    stderr_payloads = [json.loads(line) for line in captured.err.splitlines() if line]

    assert [payload["event"] for payload in stderr_payloads] == [
        "run_start",
        "resolve_ok",
        "test_started",
        "test_finished",
        "test_started",
        "test_finished",
        "test_started",
        "test_finished",
        "test_started",
        "test_finished",
        "run_finished",
    ]
    assert {payload["level"] for payload in stderr_payloads} == {"INFO"}
    run_ids = {str(payload["run_id"]) for payload in stderr_payloads}
    assert len(run_ids) == 1
    UUID(run_ids.pop())
    assert {payload["logger"] for payload in stderr_payloads} == {
        "aibom_verifier.cli",
        "aibom_verifier.orchestrator",
    }

    run_start = stderr_payloads[0]
    assert run_start["requested_target"] == TARGET_REPO
    assert run_start["requested_base"] is None
    assert run_start["revision_target"] is None
    assert run_start["revision_base"] is None
    assert run_start["backend"] == "local"
    assert run_start["store"] == "filesystem"
    assert run_start["ignore_cache"] is False

    resolve_ok = stderr_payloads[1]
    assert resolve_ok["target"] == {
        "repo_id": TARGET_REPO,
        "revision": "main",
        "sha": TARGET_SHA,
    }
    assert resolve_ok["base"] == {
        "repo_id": BASE_REPO,
        "revision": "main",
        "sha": BASE_SHA,
        "source": "card",
    }
    assert resolve_ok["cache"] == {"hits": 0, "misses": 2}

    for payload in stderr_payloads:
        if payload["event"] == "test_finished":
            assert "cache" in payload
            assert set(payload["cache"]) == {"hits", "misses"}
            assert isinstance(payload["cache"]["hits"], int)
            assert isinstance(payload["cache"]["misses"], int)

    run_finished = stderr_payloads[-1]
    assert run_finished["verdict"] == "verified_derivative"
    assert run_finished["exit_code"] == 0
    assert run_finished["tests_summary"] == {"pass": 4, "fail": 0, "skip": 0, "error": 0}

    assert isinstance(stdout_payload["message"], str)
    assert stdout_payload["message"]
    assert (
        stdout_payload
        == VerificationResult(
            target=TARGET_REPO,
            base=BASE_REPO,
            verdict="verified_derivative",
            message=stdout_payload["message"],
        ).to_dict()
    )


def test_success_emits_run_start_and_run_finished_and_forwards_observer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    args = _parse(["org/target-model", "--base", "org/base-model"])
    fake_store = object()
    captured_compare: dict[str, object] = {}
    monkeypatch.setenv("AIBOM_STORE", "proxy")
    monkeypatch.delenv("AIBOM_LOG_LEVEL", raising=False)
    monkeypatch.setattr(cli, "build_artifact_store", lambda **kwargs: fake_store)

    def _fake_compare(*compare_args, **compare_kwargs):
        captured_compare.update(compare_kwargs)
        return _fake_run_result()

    monkeypatch.setattr(cli, "run_compare", _fake_compare)

    assert cli.run(args) == 0

    stderr_payloads = _read_stderr_lines(capsys)

    assert [payload["event"] for payload in stderr_payloads] == ["run_start", "run_finished"]
    assert stderr_payloads[0]["requested_target"] == "org/target-model"
    assert stderr_payloads[0]["requested_base"] == "org/base-model"
    assert stderr_payloads[0]["revision_target"] is None
    assert stderr_payloads[0]["revision_base"] is None
    assert stderr_payloads[0]["backend"] == "local"
    assert stderr_payloads[0]["store"] == "proxy"
    assert stderr_payloads[0]["ignore_cache"] is False
    assert stderr_payloads[1]["verdict"] == "insufficient_evidence"
    assert stderr_payloads[1]["exit_code"] == 0
    assert stderr_payloads[1]["tests_summary"] == {"pass": 1, "fail": 0, "skip": 1, "error": 1}
    assert isinstance(stderr_payloads[1]["duration_ms"], int)
    assert captured_compare["store"] is fake_store
    assert captured_compare["observer"].__class__ is StderrJsonlObserver


def test_error_log_level_suppresses_info_events_for_successful_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    args = _parse(["org/target-model", "--base", "org/base-model"])
    monkeypatch.setenv("AIBOM_LOG_LEVEL", "ERROR")
    monkeypatch.setattr(cli, "build_artifact_store", lambda **kwargs: object())
    monkeypatch.setattr(cli, "run_compare", lambda *a, **k: _fake_run_result())

    assert cli.run(args) == 0

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    assert captured.err == ""
    assert stdout_payload["target"] == "org/target-model"
    assert stdout_payload["verdict"] == "insufficient_evidence"


def test_real_resolve_fail_emits_run_start_then_resolve_failed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
):
    args = _parse([TARGET_REPO, "--base", BASE_REPO, "--cache-dir", str(tmp_path)])
    monkeypatch.delenv("AIBOM_LOG_LEVEL", raising=False)

    def _raise_resolve(repo_id: str, revision: str | None = None, *, api=None):
        raise CompareStartError("gated_unauthenticated", "no access")

    monkeypatch.setattr(resolve_refs_mod, "resolve_commit", _raise_resolve)

    assert cli.run(args) == 1

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    stderr_payloads = [json.loads(line) for line in captured.err.splitlines() if line]

    assert stdout_payload["ok"] is False
    assert stdout_payload["error"] == "gated_unauthenticated"
    assert [payload["event"] for payload in stderr_payloads] == ["run_start", "resolve_failed"]
    assert stderr_payloads[0]["logger"] == "aibom_verifier.cli"
    assert stderr_payloads[1]["logger"] == "aibom_verifier.orchestrator"
    assert stderr_payloads[1]["level"] == "ERROR"
    assert stderr_payloads[1]["error_code"] == "gated_unauthenticated"
    assert stderr_payloads[1]["message"] == "no access"
    assert isinstance(stderr_payloads[1]["duration_ms"], int)
    run_ids = {str(payload["run_id"]) for payload in stderr_payloads}
    assert len(run_ids) == 1
    UUID(run_ids.pop())
    assert "run_finished" not in {payload["event"] for payload in stderr_payloads}


def test_cli_log_file_tee_does_not_change_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
):
    log_file = tmp_path / "verify.jsonl"
    monkeypatch.setenv("AIBOM_LOG_FILE", str(log_file))
    monkeypatch.delenv("AIBOM_LOG_LEVEL", raising=False)
    monkeypatch.delenv("AIBOM_STORE", raising=False)
    args = _parse([TARGET_REPO, "--cache-dir", str(tmp_path / "cache")])
    _configure_real_happy_path(monkeypatch)

    assert cli.run(args) == 0

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    assert stdout_payload["target"] == TARGET_REPO
    file_payloads = [json.loads(line) for line in log_file.read_text().splitlines() if line]
    stderr_payloads = [json.loads(line) for line in captured.err.splitlines() if line]
    assert file_payloads == stderr_payloads
    assert [payload["event"] for payload in file_payloads][0] == "run_start"
    assert [payload["event"] for payload in file_payloads][-1] == "run_finished"


def test_cli_observer_self_failure_still_prints_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    args = _parse(["org/target-model", "--base", "org/base-model"])
    monkeypatch.delenv("AIBOM_LOG_LEVEL", raising=False)
    monkeypatch.setattr(cli, "build_artifact_store", lambda **kwargs: object())
    monkeypatch.setattr(cli, "run_compare", lambda *a, **k: _fake_run_result())

    class _BoomObserver:
        def on_event(self, event: str, *, logger: str, **fields: object) -> None:
            raise RuntimeError(f"boom:{event}")

    monkeypatch.setattr(cli, "StderrJsonlObserver", _BoomObserver)

    assert cli.run(args) == 0

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    assert captured.err == ""
    assert stdout_payload["target"] == "org/target-model"
    assert stdout_payload["verdict"] == "insufficient_evidence"
