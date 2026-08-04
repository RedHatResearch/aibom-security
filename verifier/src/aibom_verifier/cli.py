"""``verify`` subcommand: registered into the ``aibom`` umbrella CLI."""

from __future__ import annotations

import argparse
import json

from aibom_verifier.backends.compose_queue import ComposeQueueBackend
from aibom_verifier.backends.ssh_local import SshLocalBackend
from aibom_verifier.errors import CompareStartError
from aibom_verifier.nodes.verdict_synthesize import verdict_message
from aibom_verifier.planner import run_compare
from aibom_verifier.slots.execution_backend import ExecutionBackend
from aibom_verifier.store_factory import add_store_arguments, build_artifact_store
from aibom_verifier.types import POLICY_VERSION, RunResult, VerificationResult

NAME = "verify"
HELP = "Verify whether a model's declared base_model claim holds up against its weights"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Populate ``parser`` with the ``verify`` subcommand's arguments."""
    parser.add_argument(
        "target", help="Hugging Face repo id of the model being verified, e.g. org/model"
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Claimed base model repo id. Defaults to the target's model card base_model field.",
    )
    parser.add_argument(
        "--revision-target", default=None, help="Pin the target to a specific revision/commit SHA"
    )
    parser.add_argument(
        "--revision-base", default=None, help="Pin the base to a specific revision/commit SHA"
    )
    add_store_arguments(parser)
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Skip cache reads (still write new artifacts)",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "ssh", "compose"],
        default="local",
        help=(
            "Where detection tests run: local in-process (default), "
            "ssh (ssh localhost → python -m aibom_verifier.run_test), or "
            "compose (Redis list queue → worker; AIBOM_REDIS_URL)."
        ),
    )


def _error_envelope(error_code: str, message: str) -> dict[str, object]:
    return {
        "ok": False,
        "error": error_code,
        "message": message,
        "policy_version": POLICY_VERSION,
    }


def _exit_code_for_result(result: RunResult) -> int:
    if any(test.status == "error" for test in result.tests):
        return 1
    return 0


def _to_verification_result(result: RunResult) -> VerificationResult:
    return VerificationResult(
        target=result.target.repo_id,
        base=result.base.repo_id,
        verdict=result.final_verdict,
        message=verdict_message(result.final_verdict, tests=result.tests),
    )


def _build_backend(args: argparse.Namespace) -> ExecutionBackend | None:
    if args.backend == "ssh":
        return SshLocalBackend(
            store_dir=args.cache_dir,
            store=args.store,
            ignore_cache=bool(args.ignore_cache),
        )
    if args.backend == "compose":
        return ComposeQueueBackend(
            store_dir=args.cache_dir,
            store=args.store,
            ignore_cache=bool(args.ignore_cache),
        )
    return None


def run(args: argparse.Namespace) -> int:
    """Execute the ``verify`` subcommand and print a :class:`VerificationResult` as JSON."""
    store = build_artifact_store(
        store=args.store,
        cache_dir=args.cache_dir,
        ignore_cache=bool(args.ignore_cache),
    )
    try:
        result = run_compare(
            args.target,
            base_repo=args.base,
            revision_target=args.revision_target,
            revision_base=args.revision_base,
            store=store,
            backend=_build_backend(args),
        )
    except CompareStartError as exc:
        print(json.dumps(_error_envelope(exc.error_code, exc.message), indent=2))
        return 1

    public = _to_verification_result(result)
    print(json.dumps(public.to_dict(), indent=2))
    return _exit_code_for_result(result)
