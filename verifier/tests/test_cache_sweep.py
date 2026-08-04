"""cache-sweep CLI and LAT eviction helpers."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta

import pytest

from aibom_verifier import cache_sweep as cache_sweep_mod
from aibom_verifier import cli_cache_sweep as sweep_cli
from aibom_verifier.slots.artifact_store import FilesystemArtifactStore
from aibom_verifier.slots.proxy_store import (
    ArtifactMeta,
    InMemoryBlobBackend,
    InMemoryMetadataBackend,
    ProxyArtifactStore,
)


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sweep_cli.add_arguments(parser)
    return parser.parse_args(argv)


def _with_lat(row: ArtifactMeta, last_accessed_at: datetime) -> ArtifactMeta:
    return ArtifactMeta(
        key=row.key,
        blob_object=row.blob_object,
        size_bytes=row.size_bytes,
        created_at=row.created_at,
        last_accessed_at=last_accessed_at,
    )


def _aged_proxy(*, age: timedelta) -> ProxyArtifactStore:
    meta = InMemoryMetadataBackend()
    blobs = InMemoryBlobBackend()
    store = ProxyArtifactStore(meta, blobs)
    store.put("stale", b"old")
    row = meta.get("stale")
    assert row is not None
    meta.put(_with_lat(row, datetime.now(UTC) - age))
    return store


def test_add_arguments_defaults_max_age_days_30():
    args = _parse([])
    assert args.max_age_days == 30
    assert args.store is None


def test_sweep_store_calls_proxy_sweep():
    store = _aged_proxy(age=timedelta(seconds=1))
    assert cache_sweep_mod.sweep_store(store, max_age_days=0) == 1
    assert store.get("stale") is None


def test_sweep_store_filesystem_is_noop(tmp_path):
    store = FilesystemArtifactStore(base_dir=tmp_path)
    store.put("k1", b"payload")
    assert cache_sweep_mod.sweep_store(store, max_age_days=0) == 0
    assert store.get("k1") == b"payload"


def test_sweep_store_rejects_negative_max_age():
    store = ProxyArtifactStore(InMemoryMetadataBackend(), InMemoryBlobBackend())
    with pytest.raises(ValueError, match="max_age_days"):
        cache_sweep_mod.sweep_store(store, max_age_days=-1)


def test_cli_run_exits_0_and_reports_deleted(monkeypatch, capsys):
    store = _aged_proxy(age=timedelta(days=40))
    monkeypatch.setattr(sweep_cli, "build_artifact_store", lambda **kw: store)
    args = _parse(["--max-age-days", "30", "--store", "proxy"])
    assert sweep_cli.run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"deleted": 1, "max_age_days": 30}
    assert store.get("stale") is None


def test_cli_run_max_age_days_zero(monkeypatch, capsys):
    store = _aged_proxy(age=timedelta(seconds=1))
    monkeypatch.setattr(sweep_cli, "build_artifact_store", lambda **kw: store)
    assert sweep_cli.run(_parse(["--max-age-days", "0"])) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deleted"] == 1
    assert payload["max_age_days"] == 0


def test_cli_run_rejects_negative_max_age_days(capsys):
    assert sweep_cli.run(_parse(["--max-age-days", "-1"])) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"] == "invalid_max_age_days"


def test_cli_run_store_build_error(monkeypatch, capsys):
    def _boom(**kw):
        raise ValueError("unknown store kind: 'bogus'")

    monkeypatch.setattr(sweep_cli, "build_artifact_store", _boom)
    assert sweep_cli.run(_parse([])) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"] == "invalid_store"
