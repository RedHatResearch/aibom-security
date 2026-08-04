"""Build an :class:`ArtifactStore` from CLI flags / environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from aibom_verifier.slots.artifact_store import ArtifactStore, FilesystemArtifactStore
from aibom_verifier.slots.proxy_store import ProxyArtifactStore

DEFAULT_CACHE_DIR = ".cache/aibom-verifier"

_STORE_HELP = (
    "Artifact store: filesystem (default, or AIBOM_STORE) or proxy "
    "(Postgres metadata + MinIO blobs; needs AIBOM_PG_DSN and AIBOM_MINIO_* "
    "— AIBOM_MINIO_ENDPOINT, AIBOM_MINIO_ACCESS_KEY, AIBOM_MINIO_SECRET_KEY; "
    "optional AIBOM_MINIO_BUCKET, AIBOM_MINIO_SECURE)."
)


def default_cache_dir() -> str:
    return os.environ.get("AIBOM_CACHE_DIR", DEFAULT_CACHE_DIR)


def add_store_arguments(
    parser: argparse.ArgumentParser,
    *,
    cache_dir_flag: str = "--cache-dir",
    cache_dir_help: str = "Override the local cache directory",
    store_help_suffix: str = "",
) -> None:
    """Add shared ``--store`` / cache-dir flags used by verify, cache-sweep, and run_test.

    ``cache_dir_flag`` defaults to ``--cache-dir``; ``run_test`` uses ``--store-dir``
    (same ``dest=\"cache_dir\"``) to match the remote CLI contract.
    """
    parser.add_argument(cache_dir_flag, default=None, dest="cache_dir", help=cache_dir_help)
    parser.add_argument(
        "--store",
        default=None,
        choices=["filesystem", "proxy"],
        help=_STORE_HELP + store_help_suffix,
    )


def build_artifact_store(
    *,
    store: str | None = None,
    cache_dir: str | Path | None = None,
    ignore_cache: bool = False,
) -> ArtifactStore:
    """Return filesystem or Postgres+MinIO proxy store.

    ``store`` defaults to ``AIBOM_STORE`` or ``filesystem``.
    Proxy needs ``AIBOM_PG_DSN`` and ``AIBOM_MINIO_*`` (see ``ProxyArtifactStore.from_env``).
    ``cache_dir`` applies only to the filesystem store; proxy ignores it.
    """
    kind = (store or os.environ.get("AIBOM_STORE") or "filesystem").lower()
    if kind == "proxy":
        return ProxyArtifactStore.from_env(ignore_cache=ignore_cache)
    if kind != "filesystem":
        raise ValueError(f"unknown store kind: {kind!r} (expected filesystem|proxy)")
    base = Path(cache_dir if cache_dir is not None else default_cache_dir())
    return FilesystemArtifactStore(base_dir=base, ignore_cache=ignore_cache)
