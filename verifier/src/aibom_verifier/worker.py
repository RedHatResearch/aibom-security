"""Compose worker entry: ``python -m aibom_verifier.worker``."""

from __future__ import annotations

import sys

from aibom_verifier.backends.compose_queue import worker_main


def main(argv: list[str] | None = None) -> int:
    _ = argv  # reserved for future flags; Compose uses env today
    worker_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
