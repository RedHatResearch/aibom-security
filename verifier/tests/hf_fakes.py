"""Shared offline HF Hub fakes and stderr JSONL helpers for verifier tests."""

from __future__ import annotations

import json
from typing import Any

import pytest


def fake_resolve_commit(shas: dict[str, str]):
    def _fn(repo_id: str, revision: str | None = None, *, api=None):
        del revision, api
        return shas[repo_id]

    return _fn


def fake_load_config_json(configs: dict[str, dict]):
    def _fn(repo_id: str, sha: str, store, *, api=None):
        del api
        cache_key = f"config:{repo_id}:{sha}"
        cached = store.get(cache_key)
        if cached is not None:
            return json.loads(cached)
        store.put(cache_key, json.dumps(configs[repo_id]).encode("utf-8"))
        return configs[repo_id]

    return _fn


def fake_list_tensor_names(names_by_repo: dict[str, Any]):
    def _fn(repo_id: str, sha: str, store, *, api=None):
        del api
        names = names_by_repo[repo_id]
        if isinstance(names, Exception):
            raise names
        cache_key = f"st_meta:{repo_id}:{sha}"
        if store.get(cache_key) is None:
            store.put(cache_key, json.dumps(names).encode("utf-8"))
        return sorted(names)

    return _fn


def fake_tensor_shapes(shapes_by_repo: dict[str, dict[str, list[int]]]):
    def _fn(repo_id: str, sha: str, store, *, api=None):
        del sha, store, api
        return shapes_by_repo[repo_id]

    return _fn


def fake_fetch_tensor_bytes(bytes_by_repo: dict[str, dict[str, bytes]]):
    def _fn(repo_id: str, sha: str, tensor_name: str, store, *, api=None):
        del api
        cache_key = f"st_tensor:{repo_id}:{sha}:{tensor_name}"
        cached = store.get(cache_key)
        if cached is not None:
            return cached
        data = bytes_by_repo[repo_id][tensor_name]
        store.put(cache_key, data)
        return data

    return _fn


def read_stderr_jsonl(
    capsys: pytest.CaptureFixture[str],
    *,
    expect_empty_stdout: bool = False,
) -> list[dict[str, object]]:
    captured = capsys.readouterr()
    if expect_empty_stdout:
        assert captured.out == ""
    lines = [line for line in captured.err.splitlines() if line]
    return [json.loads(line) for line in lines]
