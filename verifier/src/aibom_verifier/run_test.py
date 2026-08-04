"""Run one detection node by id; print :class:`TestOutcome` JSON on stdout.

CLI contract (SSH remotes call this entry; Compose workers call
:func:`~aibom_verifier.slots.worker.run_one_node` in-process via Redis)::

    python -m aibom_verifier.run_test --node-id ID --inputs-json '...' [--store-dir DIR]

Exit 0 when a ``TestOutcome`` is printed (pass/fail/skip, or status=error
for missing node / node exception). Exit 1 on transport/parse/crash before
an outcome is produced. Argparse misuse may exit 2.

Never requires a serialized ``api`` in inputs — this module constructs
:class:`~huggingface_hub.HfApi` locally (token from env / default login).
Remotes rebuild the artifact store from env when ``AIBOM_STORE=proxy``;
``--store-dir`` selects the filesystem cache path.
``--ignore-cache`` skips cache reads on the remote store as well.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from aibom_verifier.planner import DEFAULT_REGISTRY
from aibom_verifier.slots.worker import NodeFn, run_one_node
from aibom_verifier.store_factory import add_store_arguments, build_artifact_store


def build_run_test_argv(
    node_id: str,
    inputs: dict,
    *,
    python: str = "python",
    store_dir: str | None = None,
    store: str | None = None,
    ignore_cache: bool = False,
) -> list[str]:
    """Argv for ``python -m aibom_verifier.run_test`` (SSH remotes)."""
    remote: list[str] = [
        python,
        "-m",
        "aibom_verifier.run_test",
        "--node-id",
        node_id,
        "--inputs-json",
        json.dumps(inputs),
    ]
    if store_dir is not None:
        remote.extend(["--store-dir", store_dir])
    if store is not None:
        remote.extend(["--store", store])
    if ignore_cache:
        remote.append("--ignore-cache")
    return remote


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aibom_verifier.run_test",
        description="Run one aibom-verifier NodeFn and print TestOutcome JSON",
    )
    parser.add_argument("--node-id", required=True, help="Registry key of the node to run")
    parser.add_argument(
        "--inputs-json",
        required=True,
        help="JSON object of node inputs (do not include api; constructed here)",
    )
    add_store_arguments(
        parser,
        cache_dir_flag="--store-dir",
        cache_dir_help="Filesystem cache directory (ignored when AIBOM_STORE=proxy)",
    )
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Skip cache reads (still write new artifacts)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    registry: dict[str, NodeFn] | None = None,
) -> int:
    """Run one node; print outcome JSON. ``registry`` is injectable for tests."""
    args = build_parser().parse_args(argv)
    try:
        raw = json.loads(args.inputs_json)
        if not isinstance(raw, dict):
            raise ValueError("inputs-json must be a JSON object")
        store = build_artifact_store(
            store=args.store,
            cache_dir=args.cache_dir,
            ignore_cache=bool(args.ignore_cache),
        )
        outcome = run_one_node(
            args.node_id,
            raw,
            store=store,
            registry=registry or DEFAULT_REGISTRY,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": "run_test_failed", "message": str(exc)},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(outcome.to_dict()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
