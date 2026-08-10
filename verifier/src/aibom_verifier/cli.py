"""``verify`` subcommand: registered into the ``aibom`` umbrella CLI."""

from __future__ import annotations

import argparse
import json
import os
from time import perf_counter_ns
from uuid import uuid4

from aibom_verifier.backends.compose_queue import ComposeQueueBackend
from aibom_verifier.backends.ssh_local import SshLocalBackend
from aibom_verifier.errors import CompareStartError
from aibom_verifier.nodes.verdict_synthesize import verdict_message
from aibom_verifier.observer import StderrJsonlObserver, safe_on_event
from aibom_verifier.planner import run_compare
from aibom_verifier.run_log import configure_logging, elapsed_ms, set_run_id
from aibom_verifier.slots.execution_backend import ExecutionBackend
from aibom_verifier.store_factory import (
    add_store_arguments,
    build_artifact_store,
    effective_store_kind,
)
from aibom_verifier.types import POLICY_VERSION, RunResult, VerificationResult

NAME = "verify"
HELP = "Verify whether a model's declared base_model claim holds up against its weights"

EXIT_CODE_EPILOG = """\
exit codes:
  0  compare completed (any verdict, including non-confirming)
  1  start/config error (resolve failed, gated, bad store config, …)
"""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Populate ``parser`` with the ``verify`` subcommand's arguments."""
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = EXIT_CODE_EPILOG
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
    add_store_arguments(
        parser,
        cache_dir_help=(
            "Filesystem cache directory when --store filesystem "
            "(default: AIBOM_CACHE_DIR or .cache/aibom-verifier; ignored for proxy)"
        ),
    )
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Skip cache reads and recompute; still write new artifacts to the store",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "ssh", "compose"],
        default="local",
        help=(
            "Where detection tests run: local in-process (default); "
            "ssh (ssh localhost → python -m aibom_verifier.run_test); "
            "compose (Redis list queue → worker; needs AIBOM_REDIS_URL)"
        ),
    )


def _error_envelope(error_code: str, message: str) -> dict[str, object]:
    return {
        "ok": False,
        "error": error_code,
        "message": message,
        "policy_version": POLICY_VERSION,
    }


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


_CLI_LOGGER = "aibom_verifier.cli"


def _tests_summary(result: RunResult) -> dict[str, int]:
    summary = {"pass": 0, "fail": 0, "skip": 0, "error": 0}
    for outcome in result.tests:
        if outcome.test_id == "resolve_refs":
            continue
        summary[outcome.status] += 1
    return summary


def _emit_run_failed(
    observer: StderrJsonlObserver,
    *,
    started_ns: int,
    error_code: str,
    message: str,
) -> int:
    safe_on_event(
        observer,
        "run_failed",
        logger=_CLI_LOGGER,
        exit_code=1,
        error_code=error_code,
        message=message,
        duration_ms=elapsed_ms(started_ns),
    )
    print(json.dumps(_error_envelope(error_code, message), indent=2))
    return 1


def run(args: argparse.Namespace) -> int:
    """Execute ``verify``: print JSON to stdout; exit codes in ``EXIT_CODE_EPILOG``."""
    started_ns = perf_counter_ns()
    set_run_id(str(uuid4()))
    observer = StderrJsonlObserver()
    log_level = os.environ.get("AIBOM_LOG_LEVEL", "INFO")

    try:
        configure_logging(log_level)
    except ValueError as exc:
        return _emit_run_failed(
            observer,
            started_ns=started_ns,
            error_code="invalid_log_level",
            message=str(exc),
        )

    try:
        store = build_artifact_store(
            store=args.store,
            cache_dir=args.cache_dir,
            ignore_cache=bool(args.ignore_cache),
        )
    except ValueError as exc:
        return _emit_run_failed(
            observer,
            started_ns=started_ns,
            error_code="invalid_store",
            message=str(exc),
        )

    safe_on_event(
        observer,
        "run_start",
        logger=_CLI_LOGGER,
        requested_target=args.target,
        requested_base=args.base,
        revision_target=args.revision_target,
        revision_base=args.revision_base,
        backend=args.backend,
        store=effective_store_kind(args.store),
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
            observer=observer,
        )
    except CompareStartError as exc:
        print(json.dumps(_error_envelope(exc.error_code, exc.message), indent=2))
        return 1

    safe_on_event(
        observer,
        "run_finished",
        logger=_CLI_LOGGER,
        verdict=result.final_verdict,
        exit_code=0,
        duration_ms=elapsed_ms(started_ns),
        tests_summary=_tests_summary(result),
    )

    public = _to_verification_result(result)
    print(json.dumps(public.to_dict(), indent=2))
    return 0
