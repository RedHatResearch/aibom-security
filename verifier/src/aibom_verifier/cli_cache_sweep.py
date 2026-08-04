"""``cache-sweep`` subcommand: registered into the ``aibom`` umbrella CLI."""

from __future__ import annotations

import argparse
import json

from aibom_verifier.cache_sweep import sweep_store
from aibom_verifier.store_factory import add_store_arguments, build_artifact_store

NAME = "cache-sweep"
HELP = "Delete stale cached artifacts by last-access time (proxy store only; filesystem is a no-op)"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Populate ``parser`` with the ``cache-sweep`` subcommand's arguments."""
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=30,
        help="Delete entries with last_accessed_at older than this many days (default: 30)",
    )
    add_store_arguments(
        parser,
        cache_dir_help="Filesystem cache directory (ignored for store=proxy)",
        store_help_suffix=" filesystem sweep is a no-op.",
    )


def run(args: argparse.Namespace) -> int:
    """Run LAT eviction and print ``{"deleted": N, "max_age_days": ...}``."""
    if args.max_age_days < 0:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "invalid_max_age_days",
                    "message": "max_age_days must be >= 0",
                },
                indent=2,
            )
        )
        return 1
    try:
        store = build_artifact_store(store=args.store, cache_dir=args.cache_dir)
        deleted = sweep_store(store, max_age_days=args.max_age_days)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": "invalid_store", "message": str(exc)}, indent=2))
        return 1
    print(json.dumps({"deleted": deleted, "max_age_days": args.max_age_days}, indent=2))
    return 0
