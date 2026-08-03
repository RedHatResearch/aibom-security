import json

from aibom_verifier.types import ModelRef, RunResult, TestOutcome


def test_verify_subcommand_is_registered():
    from aibom.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["verify", "org/model"])

    assert args.command == "verify"
    assert args.target == "org/model"


def test_main_dispatches_verify_to_the_verifier_package(monkeypatch, capsys, tmp_path):
    from aibom.cli import main
    from aibom_verifier import cli as verify_cli

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
    monkeypatch.setattr(verify_cli, "run_compare", lambda *a, **k: fake)

    exit_code = main(
        [
            "verify",
            "org/target-model",
            "--base",
            "org/base-model",
            "--cache-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "org/target-model"
    assert payload["verdict"] == "verified_derivative"


def test_main_without_a_command_prints_help_and_fails(capsys):
    from aibom.cli import main

    exit_code = main([])

    assert exit_code == 1
    assert "usage" in capsys.readouterr().out.lower()
