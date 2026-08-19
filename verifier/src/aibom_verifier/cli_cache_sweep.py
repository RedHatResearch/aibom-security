"""``cache-sweep`` subcommand: registered into the ``aibom`` umbrella CLI."""

from __future__ import annotations

import argparse
import json
import os
from time import perf_counter_ns

from aibom_verifier.cache_sweep import sweep_store
from aibom_verifier.observer import StderrJsonlObserver, safe_on_event
from aibom_verifier.run_log import configure_logging, elapsed_ms, resolve_run_id, set_run_id
from aibom_verifier.store_factory import (
    add_store_arguments,
    build_artifact_store,
    effective_store_kind,
)
from aibom_verifier.types import POLICY_VERSION

NAME = "cache-sweep"
HELP = (
    "Delete stale proxy-store artifacts by last-access time "
    "(filesystem is a no-op). Compose sweeper calls this daily."
)

_CLI_LOGGER = "aibom_verifier.cli_cache_sweep"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Populate ``parser`` with the ``cache-sweep`` subcommand's arguments."""
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=30,
        help="Delete proxy entries with last_accessed_at older than this many days (default: 30)",
    )
    add_store_arguments(
        parser,
        cache_dir_help="Filesystem cache directory (ignored for --store proxy)",
    )


def _error_envelope(
    error_code: str,
    message: str,
    *,
    include_policy_version: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "ok": False,
        "error": error_code,
        "message": message,
    }
    if include_policy_version:
        payload["policy_version"] = POLICY_VERSION
    return payload


def _emit_sweep_failed(
    observer: StderrJsonlObserver,
    *,
    started_ns: int,
    error_code: str,
    message: str,
    include_policy_version: bool = False,
) -> int:
    safe_on_event(
        observer,
        "sweep_failed",
        logger=_CLI_LOGGER,
        exit_code=1,
        error_code=error_code,
        message=message,
        duration_ms=elapsed_ms(started_ns),
    )
    print(
        json.dumps(
            _error_envelope(
                error_code,
                message,
                include_policy_version=include_policy_version,
            ),
            indent=2,
        )
    )
    return 1


def run(args: argparse.Namespace) -> int:
    """Run LAT eviction and print ``{"deleted": N, "max_age_days": ...}``."""
    started_ns = perf_counter_ns()
    set_run_id(resolve_run_id())
    observer = StderrJsonlObserver()
    log_level = os.environ.get("AIBOM_LOG_LEVEL", "INFO")

    try:
        configure_logging(log_level)
    except ValueError as exc:
        return _emit_sweep_failed(
            observer,
            started_ns=started_ns,
            error_code="invalid_log_level",
            message=str(exc),
            include_policy_version=True,
        )

    if args.max_age_days < 0:
        return _emit_sweep_failed(
            observer,
            started_ns=started_ns,
            error_code="invalid_max_age_days",
            message="max_age_days must be >= 0",
        )

    try:
        store = build_artifact_store(
            store=args.store,
            cache_dir=args.cache_dir,
        )
    except ValueError as exc:
        return _emit_sweep_failed(
            observer,
            started_ns=started_ns,
            error_code="invalid_store",
            message=str(exc),
        )

    store_kind = effective_store_kind(args.store)

    safe_on_event(
        observer,
        "sweep_start",
        logger=_CLI_LOGGER,
        max_age_days=args.max_age_days,
        store=store_kind,
    )

    try:
        deleted = sweep_store(store, max_age_days=args.max_age_days)
    except Exception as exc:
        return _emit_sweep_failed(
            observer,
            started_ns=started_ns,
            error_code="sweep_error",
            message=str(exc),
        )

    safe_on_event(
        observer,
        "sweep_finished",
        logger=_CLI_LOGGER,
        max_age_days=args.max_age_days,
        store=store_kind,
        deleted=deleted,
        exit_code=0,
        duration_ms=elapsed_ms(started_ns),
    )

    print(json.dumps({"deleted": deleted, "max_age_days": args.max_age_days}, indent=2))
    return 0
