"""Compose worker entry: ``python -m aibom_verifier.worker``."""

from __future__ import annotations

import sys

from aibom_verifier.backends.compose_queue import worker_main


def main() -> int:
    worker_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
