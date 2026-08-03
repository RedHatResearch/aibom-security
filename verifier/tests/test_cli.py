import argparse
import json

from aibom_verifier import cli


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


def test_run_prints_valid_json_envelope_and_returns_not_implemented(capsys):
    args = _parse(["org/target-model", "--base", "org/base-model"])

    exit_code = cli.run(args)

    assert exit_code == cli.NOT_IMPLEMENTED_EXIT_CODE
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "org/target-model"
    assert payload["base"] == "org/base-model"
    assert payload["verdict"] == "insufficient_evidence"
