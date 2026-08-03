import json

from aibom.cli import build_parser, main


def test_verify_subcommand_is_registered():
    parser = build_parser()

    args = parser.parse_args(["verify", "org/model"])

    assert args.command == "verify"
    assert args.target == "org/model"


def test_main_dispatches_verify_to_the_verifier_package(capsys):
    exit_code = main(["verify", "org/target-model", "--base", "org/base-model"])

    assert exit_code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "org/target-model"


def test_main_without_a_command_prints_help_and_fails(capsys):
    exit_code = main([])

    assert exit_code == 1
    assert "usage" in capsys.readouterr().out.lower()
