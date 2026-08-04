import json

from aibom_verifier.types import ModelRef, RunResult, TestOutcome


def test_cache_sweep_subcommand_is_registered():
    from aibom.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["cache-sweep", "--max-age-days", "7"])

    assert args.command == "cache-sweep"
    assert args.max_age_days == 7


def test_main_dispatches_cache_sweep(monkeypatch, capsys):
    from aibom.cli import main
    from aibom_verifier import cli_cache_sweep as cache_sweep_cli

    def _fake_run(args):
        print(json.dumps({"deleted": 0, "max_age_days": args.max_age_days}))
        return 0

    monkeypatch.setattr(cache_sweep_cli, "run", _fake_run)
    assert main(["cache-sweep", "--max-age-days", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["max_age_days"] == 0


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
