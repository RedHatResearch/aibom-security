"""ProxyArtifactStore unit tests with in-memory fakes (no real PG/MinIO)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aibom_verifier.slots.proxy_store import (
    ArtifactMeta,
    InMemoryBlobBackend,
    InMemoryMetadataBackend,
    ProxyArtifactStore,
)


@pytest.fixture
def backends():
    meta = InMemoryMetadataBackend()
    blobs = InMemoryBlobBackend()
    return meta, blobs


def _with_lat(row: ArtifactMeta, last_accessed_at: datetime) -> ArtifactMeta:
    return ArtifactMeta(
        key=row.key,
        blob_object=row.blob_object,
        size_bytes=row.size_bytes,
        created_at=row.created_at,
        last_accessed_at=last_accessed_at,
    )


def test_put_then_get_round_trip(backends):
    meta, blobs = backends
    store = ProxyArtifactStore(meta, blobs)

    store.put("k1", b"payload")
    assert store.exists("k1") is True
    assert store.get("k1") == b"payload"


def test_ignore_cache_skips_reads_but_still_writes(backends):
    meta, blobs = backends
    store = ProxyArtifactStore(meta, blobs, ignore_cache=True)

    store.put("k1", b"payload")
    assert store.exists("k1") is False
    assert store.get("k1") is None
    reader = ProxyArtifactStore(meta, blobs, ignore_cache=False)
    assert reader.get("k1") == b"payload"


def test_get_and_exists_update_last_accessed_at(backends):
    meta, blobs = backends
    store = ProxyArtifactStore(meta, blobs)
    store.put("k1", b"payload")

    before = meta.get("k1")
    assert before is not None
    aged = _with_lat(before, datetime(2020, 1, 1, tzinfo=UTC))
    meta.put(aged)

    assert store.exists("k1") is True
    after_exists = meta.get("k1")
    assert after_exists is not None
    assert after_exists.last_accessed_at > aged.last_accessed_at

    meta.put(_with_lat(after_exists, datetime(2020, 1, 1, tzinfo=UTC)))
    assert store.get("k1") == b"payload"
    after_get = meta.get("k1")
    assert after_get is not None
    assert after_get.last_accessed_at > datetime(2020, 1, 1, tzinfo=UTC)


def test_exists_false_when_blob_missing(backends):
    meta, blobs = backends
    store = ProxyArtifactStore(meta, blobs)
    store.put("k1", b"payload")
    row = meta.get("k1")
    assert row is not None
    lat_before = row.last_accessed_at
    blobs.delete(row.blob_object)

    assert store.exists("k1") is False
    assert store.get("k1") is None
    after = meta.get("k1")
    assert after is not None
    assert after.last_accessed_at == lat_before


def test_put_overwrite_preserves_created_at(backends):
    meta, blobs = backends
    store = ProxyArtifactStore(meta, blobs)
    store.put("k1", b"v1")
    first = meta.get("k1")
    assert first is not None
    created = first.created_at

    store.put("k1", b"v2-longer")
    second = meta.get("k1")
    assert second is not None
    assert second.created_at == created
    assert second.size_bytes == len(b"v2-longer")
    assert store.get("k1") == b"v2-longer"


def test_put_rolls_back_new_blob_if_metadata_fails(backends):
    meta, blobs = backends

    class BoomMeta(InMemoryMetadataBackend):
        def put(self, meta: ArtifactMeta) -> None:
            raise RuntimeError("pg down")

    store = ProxyArtifactStore(BoomMeta(), blobs)
    with pytest.raises(RuntimeError, match="pg down"):
        store.put("k1", b"payload")
    assert blobs.objects == {}


def test_sweep_rejects_negative_max_age_days(backends):
    meta, blobs = backends
    store = ProxyArtifactStore(meta, blobs)
    store.put("k1", b"payload")
    with pytest.raises(ValueError, match="max_age_days"):
        store.sweep(max_age_days=-1)
    assert store.get("k1") == b"payload"


def test_sweep_deletes_stale_metadata_and_blobs(backends):
    meta, blobs = backends
    store = ProxyArtifactStore(meta, blobs)
    store.put("stale", b"old")
    store.put("fresh", b"new")

    stale_row = meta.get("stale")
    assert stale_row is not None
    meta.put(_with_lat(stale_row, datetime.now(UTC) - timedelta(days=40)))

    removed = store.sweep(max_age_days=30)
    assert removed == 1
    assert store.get("stale") is None
    assert store.get("fresh") == b"new"
    assert stale_row.blob_object not in blobs.objects


def test_sweep_max_age_days_zero_after_simulated_age(backends):
    meta, blobs = backends
    store = ProxyArtifactStore(meta, blobs)
    store.put("k1", b"payload")
    row = meta.get("k1")
    assert row is not None
    meta.put(_with_lat(row, datetime.now(UTC) - timedelta(seconds=1)))

    assert store.sweep(max_age_days=0) == 1
    assert store.exists("k1") is False
