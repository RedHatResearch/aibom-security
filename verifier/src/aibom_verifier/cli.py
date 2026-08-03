"""``verify`` subcommand: registered into the ``aibom`` umbrella CLI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from aibom_verifier.errors import CompareStartError
from aibom_verifier.nodes.verdict_synthesize import verdict_message
from aibom_verifier.planner import run_compare
from aibom_verifier.slots.artifact_store import FilesystemArtifactStore
from aibom_verifier.types import POLICY_VERSION, RunResult, VerificationResult

NAME = "verify"
HELP = "Verify whether a model's declared base_model claim holds up against its weights"

DEFAULT_CACHE_DIR = ".cache/aibom-verifier"


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
    parser.add_argument("--cache-dir", default=None, help="Override the local cache directory")
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Skip cache reads (still write new artifacts)",
    )


def _default_cache_dir() -> str:
    return os.environ.get("AIBOM_CACHE_DIR", DEFAULT_CACHE_DIR)


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


def run(args: argparse.Namespace) -> int:
    """Execute the ``verify`` subcommand and print a :class:`VerificationResult` as JSON."""
    cache_dir = args.cache_dir if args.cache_dir is not None else _default_cache_dir()
    store = FilesystemArtifactStore(
        base_dir=Path(cache_dir),
        ignore_cache=bool(getattr(args, "ignore_cache", False)),
    )
    try:
        result = run_compare(
            args.target,
            base_repo=args.base,
            revision_target=args.revision_target,
            revision_base=args.revision_base,
            store=store,
        )
    except CompareStartError as exc:
        print(json.dumps(_error_envelope(exc.error_code, exc.message), indent=2))
        return 1

    public = _to_verification_result(result)
    print(json.dumps(public.to_dict(), indent=2))
    return _exit_code_for_result(result)
