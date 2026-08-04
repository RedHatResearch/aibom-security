"""Postgres + MinIO ArtifactStore (FR-A).

Unit tests inject :class:`InMemoryMetadataBackend` /
:class:`InMemoryBlobBackend`. Production uses psycopg + MinIO clients from env.
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from io import BytesIO
from typing import Protocol

import psycopg
from minio import Minio
from minio.error import S3Error

from aibom_verifier.slots.artifact_store import _sanitize_key

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS artifacts (
  key TEXT PRIMARY KEY,
  blob_object TEXT NOT NULL,
  size_bytes BIGINT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

SCHEMA_INDEX_SQL = "CREATE INDEX IF NOT EXISTS artifacts_lat ON artifacts (last_accessed_at)"


@dataclass(frozen=True, slots=True)
class ArtifactMeta:
    key: str
    blob_object: str
    size_bytes: int
    created_at: datetime
    last_accessed_at: datetime


class MetadataBackend(Protocol):
    def get(self, key: str) -> ArtifactMeta | None: ...

    def put(self, meta: ArtifactMeta) -> None: ...

    def touch(self, key: str, accessed_at: datetime) -> None: ...

    def delete(self, key: str) -> None: ...

    def list_older_than(self, cutoff: datetime) -> list[ArtifactMeta]: ...


class BlobBackend(Protocol):
    def put(self, object_name: str, data: bytes) -> None: ...

    def get(self, object_name: str) -> bytes | None: ...

    def exists(self, object_name: str) -> bool: ...

    def delete(self, object_name: str) -> None: ...


class InMemoryMetadataBackend:
    """Test double for Postgres artifact metadata."""

    def __init__(self) -> None:
        self._rows: dict[str, ArtifactMeta] = {}

    def get(self, key: str) -> ArtifactMeta | None:
        return self._rows.get(key)

    def put(self, meta: ArtifactMeta) -> None:
        existing = self._rows.get(meta.key)
        if existing is not None:
            meta = ArtifactMeta(
                key=meta.key,
                blob_object=meta.blob_object,
                size_bytes=meta.size_bytes,
                created_at=existing.created_at,
                last_accessed_at=meta.last_accessed_at,
            )
        self._rows[meta.key] = meta

    def touch(self, key: str, accessed_at: datetime) -> None:
        row = self._rows.get(key)
        if row is None:
            return
        self._rows[key] = ArtifactMeta(
            key=row.key,
            blob_object=row.blob_object,
            size_bytes=row.size_bytes,
            created_at=row.created_at,
            last_accessed_at=accessed_at,
        )

    def delete(self, key: str) -> None:
        self._rows.pop(key, None)

    def list_older_than(self, cutoff: datetime) -> list[ArtifactMeta]:
        return [row for row in self._rows.values() if row.last_accessed_at < cutoff]


class InMemoryBlobBackend:
    """Test double for MinIO blob storage."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put(self, object_name: str, data: bytes) -> None:
        self.objects[object_name] = data

    def get(self, object_name: str) -> bytes | None:
        return self.objects.get(object_name)

    def exists(self, object_name: str) -> bool:
        return object_name in self.objects

    def delete(self, object_name: str) -> None:
        self.objects.pop(object_name, None)


def _blob_object_name(key: str) -> str:
    return f"artifacts/{uuid.uuid4().hex}/{_sanitize_key(key)}"


def _row_to_meta(row: tuple) -> ArtifactMeta:
    return ArtifactMeta(
        key=row[0],
        blob_object=row[1],
        size_bytes=int(row[2]),
        created_at=row[3],
        last_accessed_at=row[4],
    )


class ProxyArtifactStore:
    """ArtifactStore backed by metadata + blob backends (Postgres + MinIO in prod)."""

    def __init__(
        self,
        metadata: MetadataBackend,
        blobs: BlobBackend,
        *,
        ignore_cache: bool = False,
    ) -> None:
        self._metadata = metadata
        self._blobs = blobs
        self._ignore_cache = ignore_cache

    @classmethod
    def from_env(cls, *, ignore_cache: bool = False) -> ProxyArtifactStore:
        """Build from ``AIBOM_PG_DSN`` and ``AIBOM_MINIO_*`` environment variables."""
        return cls(
            PsycopgMetadataBackend.from_env(),
            MinioBlobBackend.from_env(),
            ignore_cache=ignore_cache,
        )

    def _load(self, key: str) -> bytes | None:
        """Return cached bytes and refresh LAT, or None on miss / ignore_cache."""
        if self._ignore_cache:
            return None
        row = self._metadata.get(key)
        if row is None:
            return None
        data = self._blobs.get(row.blob_object)
        if data is None:
            return None
        self._metadata.touch(key, datetime.now(UTC))
        return data

    def exists(self, key: str) -> bool:
        if self._ignore_cache:
            return False
        row = self._metadata.get(key)
        if row is None:
            return False
        if not self._blobs.exists(row.blob_object):
            return False
        self._metadata.touch(key, datetime.now(UTC))
        return True

    def get(self, key: str) -> bytes | None:
        return self._load(key)

    def put(self, key: str, data: bytes) -> None:
        now = datetime.now(UTC)
        existing = self._metadata.get(key)
        if existing is not None:
            blob_object = existing.blob_object
            created_at = existing.created_at
            created_new_blob = False
        else:
            blob_object = _blob_object_name(key)
            created_at = now
            created_new_blob = True
        self._blobs.put(blob_object, data)
        try:
            self._metadata.put(
                ArtifactMeta(
                    key=key,
                    blob_object=blob_object,
                    size_bytes=len(data),
                    created_at=created_at,
                    last_accessed_at=now,
                )
            )
        except Exception:
            if created_new_blob:
                self._blobs.delete(blob_object)
            raise

    def sweep(self, max_age_days: int) -> int:
        """Delete metadata + blobs with ``last_accessed_at`` older than ``max_age_days``.

        Returns the number of keys removed.
        """
        if max_age_days < 0:
            raise ValueError("max_age_days must be >= 0")
        cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
        stale = self._metadata.list_older_than(cutoff)
        for row in stale:
            # Drop metadata first so a failed blob delete cannot leave a
            # row that points at a missing object; orphan blobs are tolerable.
            self._metadata.delete(row.key)
            self._blobs.delete(row.blob_object)
        return len(stale)


class PsycopgMetadataBackend:
    """Postgres metadata backend (psycopg).

    Holds one long-lived connection for the process lifetime (PoC; not a pool).
    """

    def __init__(self, conninfo: str) -> None:
        # connect_timeout avoids multi-minute OS hangs on a bad DSN / down Postgres.
        self._conn = psycopg.connect(conninfo, connect_timeout=5)
        self._conn.execute(SCHEMA_SQL)
        self._conn.execute(SCHEMA_INDEX_SQL)
        self._conn.commit()

    @classmethod
    def from_env(cls) -> PsycopgMetadataBackend:
        dsn = os.environ.get("AIBOM_PG_DSN")
        if not dsn:
            raise ValueError("AIBOM_PG_DSN is required for store=proxy")
        return cls(dsn)

    @contextmanager
    def _session(self, *, commit: bool = False):
        """Yield the shared connection; always end the transaction.

        Reads roll back so the long-lived connection is not left
        idle-in-transaction after a cache miss.
        """
        try:
            yield self._conn
            if commit:
                self._conn.commit()
            else:
                self._conn.rollback()
        except Exception:
            self._conn.rollback()
            raise

    def get(self, key: str) -> ArtifactMeta | None:
        with self._session() as conn:
            row = conn.execute(
                "SELECT key, blob_object, size_bytes, created_at, last_accessed_at "
                "FROM artifacts WHERE key = %s",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_meta(row)

    def put(self, meta: ArtifactMeta) -> None:
        with self._session(commit=True) as conn:
            conn.execute(
                """
                INSERT INTO artifacts (key, blob_object, size_bytes, created_at, last_accessed_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (key) DO UPDATE SET
                  blob_object = EXCLUDED.blob_object,
                  size_bytes = EXCLUDED.size_bytes,
                  last_accessed_at = EXCLUDED.last_accessed_at
                """,
                (
                    meta.key,
                    meta.blob_object,
                    meta.size_bytes,
                    meta.created_at,
                    meta.last_accessed_at,
                ),
            )

    def touch(self, key: str, accessed_at: datetime) -> None:
        with self._session(commit=True) as conn:
            conn.execute(
                "UPDATE artifacts SET last_accessed_at = %s WHERE key = %s",
                (accessed_at, key),
            )

    def delete(self, key: str) -> None:
        with self._session(commit=True) as conn:
            conn.execute("DELETE FROM artifacts WHERE key = %s", (key,))

    def list_older_than(self, cutoff: datetime) -> list[ArtifactMeta]:
        with self._session() as conn:
            rows = conn.execute(
                "SELECT key, blob_object, size_bytes, created_at, last_accessed_at "
                "FROM artifacts WHERE last_accessed_at < %s",
                (cutoff,),
            ).fetchall()
        return [_row_to_meta(row) for row in rows]


_MISSING_OBJECT_CODES = frozenset({"NoSuchKey", "NoSuchObject", "NoSuchBucket"})


class MinioBlobBackend:
    """MinIO / S3-compatible blob backend."""

    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ) -> None:
        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self._bucket = bucket
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)

    @classmethod
    def from_env(cls) -> MinioBlobBackend:
        endpoint = os.environ.get("AIBOM_MINIO_ENDPOINT")
        access_key = os.environ.get("AIBOM_MINIO_ACCESS_KEY")
        secret_key = os.environ.get("AIBOM_MINIO_SECRET_KEY")
        bucket = os.environ.get("AIBOM_MINIO_BUCKET", "aibom-artifacts")
        if not endpoint or not access_key or not secret_key:
            raise ValueError(
                "AIBOM_MINIO_ENDPOINT, AIBOM_MINIO_ACCESS_KEY, and "
                "AIBOM_MINIO_SECRET_KEY are required for store=proxy"
            )
        # Accepted truthy values: 1, true, yes, on (case-insensitive).
        secure = os.environ.get("AIBOM_MINIO_SECURE", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return cls(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            secure=secure,
        )

    def put(self, object_name: str, data: bytes) -> None:
        self._client.put_object(
            self._bucket,
            object_name,
            BytesIO(data),
            length=len(data),
        )

    def get(self, object_name: str) -> bytes | None:
        try:
            response = self._client.get_object(self._bucket, object_name)
        except S3Error as exc:
            if exc.code in _MISSING_OBJECT_CODES:
                return None
            raise
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def exists(self, object_name: str) -> bool:
        try:
            self._client.stat_object(self._bucket, object_name)
        except S3Error as exc:
            if exc.code in _MISSING_OBJECT_CODES:
                return False
            raise
        return True

    def delete(self, object_name: str) -> None:
        self._client.remove_object(self._bucket, object_name)
