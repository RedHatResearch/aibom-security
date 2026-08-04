import argparse
import json

from aibom_verifier import cli
from aibom_verifier.backends.ssh_local import SshLocalBackend
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
            "--store",
            "proxy",
            "--ignore-cache",
            "--backend",
            "ssh",
        ]
    )

    assert args.target == "org/target-model"
    assert args.base == "org/base-model"
    assert args.revision_target == "abc123"
    assert args.revision_base == "def456"
    assert args.cache_dir == "/tmp/cache"
    assert args.store == "proxy"
    assert args.ignore_cache is True
    assert args.backend == "ssh"


def test_add_arguments_backend_defaults_to_local():
    args = _parse(["org/model"])
    assert args.backend == "local"


def test_run_wires_ignore_cache_and_store_to_factory(monkeypatch, capsys):
    args = _parse(
        [
            "org/target-model",
            "--base",
            "org/base-model",
            "--store",
            "proxy",
            "--ignore-cache",
        ]
    )
    captured: dict = {}

    def _fake_build(**kwargs):
        captured.update(kwargs)
        return object()

    fake = RunResult(
        target=ModelRef(repo_id="org/target-model", revision="main", sha="tsha"),
        base=ModelRef(repo_id="org/base-model", revision="main", sha="bsha"),
        base_source="cli",
        support_class="dense_supported",
        tests=[TestOutcome(test_id="resolve_refs", status="pass")],
        final_verdict="insufficient_evidence",
        cache={"hits": [], "misses": []},
    )
    monkeypatch.setattr(cli, "build_artifact_store", _fake_build)
    monkeypatch.setattr(cli, "run_compare", lambda *a, **k: fake)

    assert cli.run(args) == 0
    assert captured["store"] == "proxy"
    assert captured["ignore_cache"] is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "insufficient_evidence"


def test_run_wires_ssh_backend(monkeypatch, tmp_path):
    args = _parse(
        [
            "org/target-model",
            "--base",
            "org/base-model",
            "--cache-dir",
            str(tmp_path),
            "--store",
            "proxy",
            "--ignore-cache",
            "--backend",
            "ssh",
        ]
    )
    captured: dict = {}

    def _fake_compare(*a, **kwargs):
        captured.update(kwargs)
        return RunResult(
            target=ModelRef(repo_id="org/target-model", revision="main", sha="tsha"),
            base=ModelRef(repo_id="org/base-model", revision="main", sha="bsha"),
            base_source="cli",
            support_class="dense_supported",
            tests=[TestOutcome(test_id="resolve_refs", status="pass")],
            final_verdict="insufficient_evidence",
            cache={"hits": [], "misses": []},
        )

    monkeypatch.setattr(cli, "build_artifact_store", lambda **kwargs: object())
    monkeypatch.setattr(cli, "run_compare", _fake_compare)
    assert cli.run(args) == 0
    backend = captured["backend"]
    assert isinstance(backend, SshLocalBackend)
    assert backend._store_dir == str(tmp_path)
    assert backend._store == "proxy"
    assert backend._ignore_cache is True


def test_run_wires_compose_backend(monkeypatch, tmp_path):
    args = _parse(
        [
            "org/target-model",
            "--base",
            "org/base-model",
            "--cache-dir",
            str(tmp_path),
            "--ignore-cache",
            "--backend",
            "compose",
        ]
    )
    captured: dict = {}
    built: dict = {}

    class FakeCompose:
        def __init__(self, **kwargs):
            built.update(kwargs)

    def _fake_compare(*a, **kwargs):
        captured.update(kwargs)
        return RunResult(
            target=ModelRef(repo_id="org/target-model", revision="main", sha="tsha"),
            base=ModelRef(repo_id="org/base-model", revision="main", sha="bsha"),
            base_source="cli",
            support_class="dense_supported",
            tests=[TestOutcome(test_id="resolve_refs", status="pass")],
            final_verdict="insufficient_evidence",
            cache={"hits": [], "misses": []},
        )

    monkeypatch.setattr(cli, "ComposeQueueBackend", FakeCompose)
    monkeypatch.setattr(cli, "build_artifact_store", lambda **kwargs: object())
    monkeypatch.setattr(cli, "run_compare", _fake_compare)
    assert cli.run(args) == 0
    assert isinstance(captured["backend"], FakeCompose)
    assert built["store_dir"] == str(tmp_path)
    assert built["ignore_cache"] is True


def test_add_arguments_accepts_compose_backend():
    args = _parse(["org/model", "--backend", "compose"])
    assert args.backend == "compose"


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


def test_verify_help_documents_poc_flags_and_exit_codes():
    parser = argparse.ArgumentParser(prog="aibom-verify")
    cli.add_arguments(parser)
    help_text = parser.format_help()

    assert "--backend" in help_text
    assert "--ignore-cache" in help_text
    assert "--store" in help_text
    assert cli.EXIT_CODE_EPILOG.strip() in help_text


def test_completed_compare_with_test_error_still_exits_0(monkeypatch, capsys, tmp_path):
    """Exit 0 means the compare finished; verdict/JSON carry non-confirming outcomes."""
    args = _parse(["org/target-model", "--base", "org/base-model", "--cache-dir", str(tmp_path)])
    fake = RunResult(
        target=ModelRef(repo_id="org/target-model", revision="main", sha="tsha"),
        base=ModelRef(repo_id="org/base-model", revision="main", sha="bsha"),
        base_source="cli",
        support_class="dense_supported",
        tests=[
            TestOutcome(test_id="resolve_refs", status="pass"),
            TestOutcome(
                test_id="block0_shapes",
                status="error",
                reason_codes=["hub_error"],
            ),
        ],
        final_verdict="insufficient_evidence",
        cache={"hits": [], "misses": []},
    )
    monkeypatch.setattr(cli, "run_compare", lambda *a, **k: fake)

    assert cli.run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "insufficient_evidence"


def test_invalid_store_config_returns_exit_1(monkeypatch, capsys, tmp_path):
    args = _parse(["org/model", "--store", "proxy", "--cache-dir", str(tmp_path)])

    def _raise(**kwargs):
        raise ValueError("missing AIBOM_PG_DSN")

    monkeypatch.setattr(cli, "build_artifact_store", _raise)

    assert cli.run(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"] == "invalid_store"
