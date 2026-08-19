import json
from datetime import datetime

import pytest
from hf_fakes import read_stderr_jsonl

import aibom_verifier.run_log as run_log
from aibom_verifier import __version__
from aibom_verifier.observer import RecordingObserver, StderrJsonlObserver
from aibom_verifier.run_log import (
    configure_logging,
    emit,
    get_run_id,
    set_run_id,
    validate_log_level,
)
from aibom_verifier.types import POLICY_VERSION


def _read_stderr_lines(capsys: pytest.CaptureFixture[str]) -> list[dict[str, object]]:
    return read_stderr_jsonl(capsys, expect_empty_stdout=True)


class _BrokenStderr:
    def write(self, _: str) -> int:
        raise OSError("disk full")

    def flush(self) -> None:
        return None


def test_set_and_get_run_id():
    set_run_id("run-123")

    assert get_run_id() == "run-123"


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
def test_validate_log_level_accepts_known_values(level: str):
    assert validate_log_level(level) == level


def test_validate_log_level_rejects_invalid_value():
    with pytest.raises(ValueError, match="Invalid AIBOM_LOG_LEVEL"):
        validate_log_level("TRACE")


def test_configure_logging_error_suppresses_info_but_not_error(capsys: pytest.CaptureFixture[str]):
    set_run_id("run-error-threshold")
    configure_logging("ERROR")

    emit("run_start", logger="aibom_verifier.cli", level="INFO", backend="local")
    emit("run_failed", logger="aibom_verifier.cli", level="ERROR", error_code="invalid_store")

    payloads = _read_stderr_lines(capsys)

    assert [payload["event"] for payload in payloads] == ["run_failed"]
    assert payloads[0]["level"] == "ERROR"


def test_emit_writes_one_jsonl_line_with_common_envelope(capsys: pytest.CaptureFixture[str]):
    set_run_id("run-emit")
    configure_logging("INFO")

    emit(
        "run_start",
        logger="aibom_verifier.cli",
        level="INFO",
        requested_target="org/target",
        requested_base=None,
        backend="local",
    )

    payloads = _read_stderr_lines(capsys)

    assert len(payloads) == 1
    assert payloads[0] == {
        "ts": payloads[0]["ts"],
        "level": "INFO",
        "logger": "aibom_verifier.cli",
        "event": "run_start",
        "run_id": "run-emit",
        "policy_version": POLICY_VERSION,
        "tool_version": __version__,
        "requested_target": "org/target",
        "requested_base": None,
        "backend": "local",
    }
    assert isinstance(payloads[0]["ts"], str)
    assert datetime.fromisoformat(str(payloads[0]["ts"]).replace("Z", "+00:00")).tzinfo is not None


def test_emit_swallows_json_encoding_errors():
    set_run_id("run-encode-error")
    configure_logging("INFO")

    emit("run_start", logger="aibom_verifier.cli", level="INFO", bad_field=object())


def test_emit_without_run_id_is_silent_and_nonfatal(capsys: pytest.CaptureFixture[str]):
    run_log._RUN_ID.set(None)
    configure_logging("INFO")

    emit("run_start", logger="aibom_verifier.cli", level="INFO", backend="local")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_emit_keeps_envelope_keys_over_field_kwargs(capsys: pytest.CaptureFixture[str]):
    set_run_id("run-envelope")
    configure_logging("INFO")

    emit(
        "run_start",
        logger="aibom_verifier.cli",
        level="INFO",
        **{
            "run_id": "spoofed-id",
            "policy_version": "spoofed-policy",
            "tool_version": "0.0.0",
            "backend": "local",
        },
    )

    payloads = _read_stderr_lines(capsys)
    assert payloads[0]["event"] == "run_start"
    assert payloads[0]["run_id"] == "run-envelope"
    assert payloads[0]["policy_version"] != "spoofed-policy"
    assert payloads[0]["tool_version"] != "0.0.0"
    assert payloads[0]["backend"] == "local"


def test_emit_tees_jsonl_to_aibom_log_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
):
    log_file = tmp_path / "aibom.jsonl"
    monkeypatch.setenv("AIBOM_LOG_FILE", str(log_file))
    set_run_id("run-file-tee")
    configure_logging("INFO")

    emit("run_start", logger="aibom_verifier.cli", level="INFO", backend="local")

    payloads = _read_stderr_lines(capsys)
    file_payloads = [json.loads(line) for line in log_file.read_text().splitlines() if line]
    assert payloads == file_payloads
    assert file_payloads[0]["event"] == "run_start"
    assert file_payloads[0]["run_id"] == "run-file-tee"


def test_emit_swallows_log_file_write_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
):
    monkeypatch.setenv("AIBOM_LOG_FILE", str(tmp_path))  # directory, not a file
    set_run_id("run-file-io")
    configure_logging("INFO")

    emit("run_start", logger="aibom_verifier.cli", level="INFO", backend="local")

    payloads = _read_stderr_lines(capsys)
    assert [payload["event"] for payload in payloads] == ["run_start"]


def test_emit_skips_file_tee_when_log_file_unset(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
):
    monkeypatch.delenv("AIBOM_LOG_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    set_run_id("run-no-file")
    configure_logging("INFO")

    emit("run_start", logger="aibom_verifier.cli", level="INFO", backend="local")

    _read_stderr_lines(capsys)
    assert list(tmp_path.iterdir()) == []


def test_emit_swallows_stderr_write_errors(monkeypatch: pytest.MonkeyPatch):
    set_run_id("run-io-error")
    configure_logging("INFO")
    monkeypatch.setattr("sys.stderr", _BrokenStderr())

    emit("run_start", logger="aibom_verifier.cli", level="INFO", backend="local")


def test_emit_tees_log_file_after_stderr_write_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    log_file = tmp_path / "after-stderr-fail.jsonl"
    monkeypatch.setenv("AIBOM_LOG_FILE", str(log_file))
    set_run_id("run-stderr-fail-tee")
    configure_logging("INFO")
    monkeypatch.setattr("sys.stderr", _BrokenStderr())
    emit("run_start", logger="aibom_verifier.cli", level="INFO", backend="local")

    file_payloads = [json.loads(line) for line in log_file.read_text().splitlines() if line]
    assert [payload["event"] for payload in file_payloads] == ["run_start"]
    assert file_payloads[0]["run_id"] == "run-stderr-fail-tee"


def test_stderr_jsonl_observer_maps_catalog_events_to_levels(capsys: pytest.CaptureFixture[str]):
    set_run_id("run-observer")
    configure_logging("INFO")
    observer = StderrJsonlObserver()

    observer.on_event("run_start", logger="aibom_verifier.cli", backend="local")
    observer.on_event("exception", logger="aibom_verifier.orchestrator", test_id="block0_values")

    payloads = _read_stderr_lines(capsys)

    assert [payload["event"] for payload in payloads] == ["run_start", "exception"]
    assert [payload["level"] for payload in payloads] == ["INFO", "ERROR"]
    assert [payload["logger"] for payload in payloads] == [
        "aibom_verifier.cli",
        "aibom_verifier.orchestrator",
    ]


def test_recording_observer_captures_events():
    observer = RecordingObserver()

    observer.on_event("run_start", logger="aibom_verifier.cli", backend="local")
    observer.on_event("test_finished", logger="aibom_verifier.orchestrator", status="pass")

    assert observer.events == [
        {
            "event": "run_start",
            "logger": "aibom_verifier.cli",
            "fields": {"backend": "local"},
        },
        {
            "event": "test_finished",
            "logger": "aibom_verifier.orchestrator",
            "fields": {"status": "pass"},
        },
    ]
