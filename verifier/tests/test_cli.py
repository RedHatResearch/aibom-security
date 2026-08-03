import argparse
import json

from aibom_verifier import cli
from aibom_verifier.types import ModelRef, RunResult, TestOutcome


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    cli.add_arguments(parser)
    return parser.parse_args(argv)


def test_add_arguments_happy_path_target_only():
    args = _parse(["org/model"])

    assert args.target == "org/model"
    assert args.base is None
    assert args.revision_target is None
    assert args.revision_base is None


def test_add_arguments_happy_path_all_options():
    args = _parse(
        [
            "org/target-model",
            "--base",
            "org/base-model",
            "--revision-target",
            "abc123",
            "--revision-base",
            "def456",
            "--cache-dir",
            "/tmp/cache",
        ]
    )

    assert args.target == "org/target-model"
    assert args.base == "org/base-model"
    assert args.revision_target == "abc123"
    assert args.revision_base == "def456"
    assert args.cache_dir == "/tmp/cache"


def test_run_prints_verification_result_json(monkeypatch, capsys, tmp_path):
    args = _parse(
        [
            "org/target-model",
            "--base",
            "org/base-model",
            "--cache-dir",
            str(tmp_path),
        ]
    )

    fake = RunResult(
        target=ModelRef(repo_id="org/target-model", revision="main", sha="tsha"),
        base=ModelRef(repo_id="org/base-model", revision="main", sha="bsha"),
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
            TestOutcome(test_id="arch_hash", status="pass", compatibility="compatible"),
            TestOutcome(test_id="block0_shapes", status="pass", compatibility="compatible"),
            TestOutcome(
                test_id="block0_values",
                status="fail",
                compatibility="incompatible",
                reason_codes=["byte_mismatch"],
            ),
        ],
        final_verdict="verified_derivative",
        cache={"hits": [], "misses": []},
    )
    monkeypatch.setattr(cli, "run_compare", lambda *a, **k: fake)

    exit_code = cli.run(args)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "org/target-model"
    assert payload["base"] == "org/base-model"
    assert payload["verdict"] == "verified_derivative"
    assert "fine-tune" in payload["message"]


def test_run_compare_start_error_returns_exit_1(monkeypatch, capsys, tmp_path):
    from aibom_verifier.errors import CompareStartError

    args = _parse(["org/target-model", "--base", "org/base-model", "--cache-dir", str(tmp_path)])

    def _raise(*a, **k):
        raise CompareStartError("repo_not_found", "missing")

    monkeypatch.setattr(cli, "run_compare", _raise)

    exit_code = cli.run(args)

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"] == "repo_not_found"
