"""Top-level argument parser for the ``aibom`` command.

Each subcommand is implemented in its own workspace package and registered
here — this module should stay a thin dispatcher, never own business logic.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from aibom_verifier import cli as verify_cli
from aibom_verifier import cli_cache_sweep as cache_sweep_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aibom", description="aibom-security command-line tools")
    subparsers = parser.add_subparsers(dest="command", required=False)

    verify_parser = subparsers.add_parser(verify_cli.NAME, help=verify_cli.HELP)
    verify_cli.add_arguments(verify_parser)

    sweep_parser = subparsers.add_parser(cache_sweep_cli.NAME, help=cache_sweep_cli.HELP)
    cache_sweep_cli.add_arguments(sweep_parser)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == verify_cli.NAME:
        return verify_cli.run(args)
    if args.command == cache_sweep_cli.NAME:
        return cache_sweep_cli.run(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
