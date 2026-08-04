"""run_test module entry + local ExecutionBackend wiring."""

from __future__ import annotations

import json

from aibom_verifier import run_test as run_test_mod
from aibom_verifier.slots import worker as worker_slot
from aibom_verifier.types import TestOutcome


def test_run_test_main_stub_registry_node(capsys, tmp_path, monkeypatch):
    def stub_node(inputs: dict, store) -> TestOutcome:
        store.put("seen", b"1")
        return TestOutcome(
            test_id="stub_node",
            status="pass",
            detail={"echo": inputs.get("value"), "has_api": "api" in inputs},
        )

    monkeypatch.setattr(worker_slot, "HfApi", lambda: object())
    code = run_test_mod.main(
        [
            "--node-id",
            "stub_node",
            "--inputs-json",
            json.dumps({"value": 42, "api": "must-be-stripped"}),
            "--store-dir",
            str(tmp_path),
        ],
        registry={"stub_node": stub_node},
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["test_id"] == "stub_node"
    assert payload["status"] == "pass"
    assert payload["detail"] == {"echo": 42, "has_api": True}
    assert (tmp_path / "seen").read_bytes() == b"1"


def test_run_test_main_invalid_inputs_json_exits_1(capsys):
    code = run_test_mod.main(
        ["--node-id", "x", "--inputs-json", "not-json"],
        registry={},
    )
    assert code == 1
    err = json.loads(capsys.readouterr().err)
    assert err["ok"] is False
    assert err["error"] == "run_test_failed"


def test_run_test_main_non_object_inputs_exits_1(capsys):
    code = run_test_mod.main(
        ["--node-id", "x", "--inputs-json", "[]"],
        registry={},
    )
    assert code == 1
    err = json.loads(capsys.readouterr().err)
    assert err["ok"] is False
    assert "JSON object" in err["message"]


def test_run_test_missing_node_exits_0_with_error_outcome(capsys, monkeypatch):
    monkeypatch.setattr(worker_slot, "HfApi", lambda: object())
    code = run_test_mod.main(
        ["--node-id", "missing", "--inputs-json", "{}"],
        registry={},
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["reason_codes"] == ["missing_node"]
