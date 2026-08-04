"""LAT-based cache eviction helpers for ArtifactStore backends."""

from __future__ import annotations


def sweep_store(store: object, *, max_age_days: int) -> int:
    """Delete entries older than ``max_age_days`` by last-access time.

    Proxies that implement ``sweep`` (e.g. :class:`ProxyArtifactStore`) delete
    metadata and blobs. Filesystem stores have no LAT metadata; this is a no-op
    that returns ``0``.
    """
    if max_age_days < 0:
        raise ValueError("max_age_days must be >= 0")
    sweep = getattr(store, "sweep", None)
    if not callable(sweep):
        return 0
    return sweep(max_age_days=max_age_days)
