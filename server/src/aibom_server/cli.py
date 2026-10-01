"""``aibom-serve`` entry point: argparse front for the REST verify server."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

import uvicorn

from aibom_server.app import create_app
from aibom_server.jobs import (
    DEFAULT_CONCURRENCY,
    DEFAULT_QUEUE_SIZE,
    DEFAULT_TIMEOUT_SECONDS,
    JobRunner,
)

HELP = "REST server: accept Hugging Face model IDs, run aibom verify, log to a runs directory"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--host",
        default=os.environ.get("AIBOM_BIND_HOST", "127.0.0.1"),
        help="Bind address (default: AIBOM_BIND_HOST or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("AIBOM_BIND_PORT", "8080")),
        help="Bind port (default: AIBOM_BIND_PORT or 8080)",
    )
    parser.add_argument(
        "--runs-dir",
        default=os.environ.get("AIBOM_RUNS_DIR", "runs"),
        help=(
            "Directory holding one per-run folder (job.json, telemetry.jsonl, result.json); "
            "mount a volume here (default: AIBOM_RUNS_DIR or ./runs)"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Worker threads running verify subprocesses (default: %(default)s)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=DEFAULT_QUEUE_SIZE,
        help="Max queued submissions before POST /verify answers 503 (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout-secs",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Kill a verify subprocess after this many seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("AIBOM_API_TOKEN"),
        help=(
            "Require 'Authorization: Bearer <token>' on POST /verify "
            "(default: AIBOM_API_TOKEN; unset disables auth)"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aibom-serve", description=HELP)
    add_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = JobRunner(
        args.runs_dir,
        concurrency=args.concurrency,
        queue_size=args.queue_size,
        timeout_seconds=args.timeout_secs,
    )
    app = create_app(runner=runner, token=args.token or None)
    print(f"aibom-serve: runs dir {runner.runs_dir.resolve()}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0
