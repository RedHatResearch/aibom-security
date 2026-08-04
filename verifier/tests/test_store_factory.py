"""Artifact store factory (filesystem vs proxy)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from minio.error import S3Error

from aibom_verifier.slots.artifact_store import FilesystemArtifactStore
from aibom_verifier.slots.proxy_store import (
    InMemoryBlobBackend,
    InMemoryMetadataBackend,
    MinioBlobBackend,
    ProxyArtifactStore,
    PsycopgMetadataBackend,
)
from aibom_verifier.store_factory import build_artifact_store


def test_build_filesystem_store(tmp_path: Path):
    store = build_artifact_store(store="filesystem", cache_dir=tmp_path, ignore_cache=True)
    assert isinstance(store, FilesystemArtifactStore)
    store.put("k", b"v")
    assert store.get("k") is None


def test_build_proxy_store_from_env(monkeypatch):
    meta = InMemoryMetadataBackend()
    blobs = InMemoryBlobBackend()

    def _fake_from_env(cls, *, ignore_cache=False):
        return cls(meta, blobs, ignore_cache=ignore_cache)

    monkeypatch.setattr(ProxyArtifactStore, "from_env", classmethod(_fake_from_env))
    store = build_artifact_store(store="proxy", ignore_cache=True)
    assert isinstance(store, ProxyArtifactStore)
    store.put("k", b"v")
    assert store.get("k") is None
    assert ProxyArtifactStore(meta, blobs).get("k") == b"v"


def test_build_proxy_store_from_aibom_store_env(monkeypatch):
    meta = InMemoryMetadataBackend()
    blobs = InMemoryBlobBackend()

    def _fake_from_env(cls, *, ignore_cache=False):
        return cls(meta, blobs, ignore_cache=ignore_cache)

    monkeypatch.setenv("AIBOM_STORE", "proxy")
    monkeypatch.setattr(ProxyArtifactStore, "from_env", classmethod(_fake_from_env))
    store = build_artifact_store(ignore_cache=False)
    assert isinstance(store, ProxyArtifactStore)


def test_build_unknown_store_raises():
    with pytest.raises(ValueError, match="unknown store kind"):
        build_artifact_store(store="redis")


def test_psycopg_from_env_requires_dsn(monkeypatch):
    monkeypatch.delenv("AIBOM_PG_DSN", raising=False)
    with pytest.raises(ValueError, match="AIBOM_PG_DSN"):
        PsycopgMetadataBackend.from_env()


def test_minio_from_env_requires_credentials(monkeypatch):
    monkeypatch.delenv("AIBOM_MINIO_ENDPOINT", raising=False)
    monkeypatch.delenv("AIBOM_MINIO_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AIBOM_MINIO_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="AIBOM_MINIO_ENDPOINT"):
        MinioBlobBackend.from_env()


def test_minio_get_missing_object_returns_none():
    backend = MinioBlobBackend.__new__(MinioBlobBackend)
    client = MagicMock()
    client.get_object.side_effect = S3Error(
        MagicMock(),
        "NoSuchKey",
        "missing",
        "/obj",
        "req",
        "host",
        "bucket",
        "obj",
    )
    backend._client = client
    backend._bucket = "bucket"
    assert backend.get("obj") is None


def test_minio_get_access_denied_raises():
    backend = MinioBlobBackend.__new__(MinioBlobBackend)
    client = MagicMock()
    client.get_object.side_effect = S3Error(
        MagicMock(),
        "AccessDenied",
        "denied",
        "/obj",
        "req",
        "host",
        "bucket",
        "obj",
    )
    backend._client = client
    backend._bucket = "bucket"
    with pytest.raises(S3Error) as exc_info:
        backend.get("obj")
    assert exc_info.value.code == "AccessDenied"
