import json

import pytest

from aibom_verifier.errors import CompareStartError
from aibom_verifier.mapping.block0 import BLOCK0_PREFIX, REQUIRED_SUFFIXES
from aibom_verifier.nodes import block0_shapes as block0_shapes_mod
from aibom_verifier.nodes import block0_values as block0_values_mod
from aibom_verifier.nodes import resolve_refs as resolve_refs_mod
from aibom_verifier.planner import run_compare
from aibom_verifier.slots.artifact_store import InMemoryArtifactStore

REQUIRED_NAMES = [BLOCK0_PREFIX + suffix for suffix in REQUIRED_SUFFIXES]

TARGET_REPO = "org/target"
BASE_REPO = "org/base"
TARGET_SHA = "tsha"
BASE_SHA = "bsha"


def _fake_resolve_commit(shas: dict[str, str]):
    def _fn(repo_id: str, revision: str | None = None, *, api=None):
        return shas[repo_id]

    return _fn


def _fake_load_config_json(configs: dict[str, dict]):
    def _fn(repo_id: str, sha: str, store, *, api=None):
        cache_key = f"config:{repo_id}:{sha}"
        cached = store.get(cache_key)
        if cached is not None:
            return json.loads(cached)
        store.put(cache_key, json.dumps(configs[repo_id]).encode("utf-8"))
        return configs[repo_id]

    return _fn


def _fake_list_tensor_names(names_by_repo: dict):
    def _fn(repo_id: str, sha: str, store, *, api=None):
        names = names_by_repo[repo_id]
        if isinstance(names, Exception):
            raise names
        cache_key = f"st_meta:{repo_id}:{sha}"
        if store.get(cache_key) is None:
            store.put(cache_key, json.dumps(names).encode("utf-8"))
        return sorted(names)

    return _fn


def _fake_tensor_shapes(shapes_by_repo: dict[str, dict[str, list[int]]]):
    def _fn(repo_id: str, sha: str, store, *, api=None):
        return shapes_by_repo[repo_id]

    return _fn


def _fake_fetch_tensor_bytes(bytes_by_repo: dict[str, dict[str, bytes]]):
    def _fn(repo_id: str, sha: str, tensor_name: str, store, *, api=None):
        cache_key = f"st_tensor:{repo_id}:{sha}:{tensor_name}"
        cached = store.get(cache_key)
        if cached is not None:
            return cached
        data = bytes_by_repo[repo_id][tensor_name]
        store.put(cache_key, data)
        return data

    return _fn


@pytest.fixture
def scenario(monkeypatch):
    """Configure the full happy-path gate chain, with per-test overrides."""

    def _configure(
        *,
        target_model_type: str = "llama",
        base_model_type: str = "llama",
        target_config_extra: dict | None = None,
        base_config_extra: dict | None = None,
        target_names=None,
        base_names=None,
        list_tensor_names_error: Exception | None = None,
        target_shapes=None,
        base_shapes=None,
        target_tensor_bytes=None,
        base_tensor_bytes=None,
    ):
        shas = {TARGET_REPO: TARGET_SHA, BASE_REPO: BASE_SHA}
        configs = {
            TARGET_REPO: {"model_type": target_model_type, **(target_config_extra or {})},
            BASE_REPO: {"model_type": base_model_type, **(base_config_extra or {})},
        }
        monkeypatch.setattr(resolve_refs_mod, "resolve_commit", _fake_resolve_commit(shas))
        monkeypatch.setattr(resolve_refs_mod, "load_config_json", _fake_load_config_json(configs))

        if list_tensor_names_error is not None:
            names_by_repo = {
                TARGET_REPO: list_tensor_names_error,
                BASE_REPO: base_names if base_names is not None else REQUIRED_NAMES,
            }
        else:
            names_by_repo = {
                TARGET_REPO: target_names if target_names is not None else REQUIRED_NAMES,
                BASE_REPO: base_names if base_names is not None else REQUIRED_NAMES,
            }
        monkeypatch.setattr(
            block0_shapes_mod, "list_tensor_names", _fake_list_tensor_names(names_by_repo)
        )

        default_shapes = {name: [4] for name in REQUIRED_NAMES}
        shapes_by_repo = {
            TARGET_REPO: target_shapes if target_shapes is not None else default_shapes,
            BASE_REPO: base_shapes if base_shapes is not None else default_shapes,
        }
        monkeypatch.setattr(block0_shapes_mod, "tensor_shapes", _fake_tensor_shapes(shapes_by_repo))

        default_bytes = {name: b"identical-payload" for name in REQUIRED_NAMES}
        bytes_by_repo = {
            TARGET_REPO: target_tensor_bytes if target_tensor_bytes is not None else default_bytes,
            BASE_REPO: base_tensor_bytes if base_tensor_bytes is not None else default_bytes,
        }
        monkeypatch.setattr(
            block0_values_mod, "fetch_tensor_bytes", _fake_fetch_tensor_bytes(bytes_by_repo)
        )

    return _configure


def _outcomes_by_id(result):
    return {t.test_id: t for t in result.tests}


def test_cross_type_pair_is_unsupported_and_skips_downstream(scenario):
    scenario(target_model_type="llama", base_model_type="qwen2")
    store = InMemoryArtifactStore()

    result = run_compare(TARGET_REPO, base_repo=BASE_REPO, store=store, api=None)

    outcomes = _outcomes_by_id(result)
    assert outcomes["support_classify"].status == "pass"
    assert outcomes["support_classify"].compatibility == "unsupported"
    assert outcomes["arch_hash"].status == "skip"
    assert outcomes["arch_hash"].skipped_because == {
        "upstream": "support_classify",
        "reason": "unsupported",
    }
    assert outcomes["block0_shapes"].status == "skip"
    assert outcomes["block0_values"].status == "skip"
    assert result.final_verdict == "unsupported"


def test_arch_hash_mismatch_is_fraudulent_claim_and_skips_tensors(scenario):
    scenario(target_config_extra={"hidden_size": 2048}, base_config_extra={"hidden_size": 4096})
    store = InMemoryArtifactStore()

    result = run_compare(TARGET_REPO, base_repo=BASE_REPO, store=store, api=None)

    outcomes = _outcomes_by_id(result)
    assert outcomes["arch_hash"].status == "fail"
    assert outcomes["arch_hash"].reason_codes == ["arch_hash_mismatch"]
    assert outcomes["block0_shapes"].status == "skip"
    assert outcomes["block0_values"].status == "skip"
    assert result.final_verdict == "fraudulent_claim"


def test_shapes_fail_skips_values_and_verdict_is_incompatible(scenario):
    mismatched_shapes = {name: [4] for name in REQUIRED_NAMES}
    mismatched_shapes[REQUIRED_NAMES[0]] = [999]
    scenario(base_shapes=mismatched_shapes)
    store = InMemoryArtifactStore()

    result = run_compare(TARGET_REPO, base_repo=BASE_REPO, store=store, api=None)

    outcomes = _outcomes_by_id(result)
    assert outcomes["block0_shapes"].status == "fail"
    assert outcomes["block0_values"].status == "skip"
    assert result.final_verdict == "incompatible"


def test_not_safetensors_yields_insufficient_evidence(scenario):
    scenario(list_tensor_names_error=ValueError("not_safetensors"))
    store = InMemoryArtifactStore()

    result = run_compare(TARGET_REPO, base_repo=BASE_REPO, store=store, api=None)

    outcomes = _outcomes_by_id(result)
    assert outcomes["block0_shapes"].compatibility == "insufficient_evidence"
    assert outcomes["block0_values"].status == "skip"
    assert result.final_verdict == "insufficient_evidence"


def test_shapes_pass_values_differ_is_verified_derivative(scenario):
    common_bytes = {name: b"payload" for name in REQUIRED_NAMES}
    differing_bytes = dict(common_bytes)
    differing_bytes[REQUIRED_NAMES[0]] = b"different-payload"
    scenario(target_tensor_bytes=common_bytes, base_tensor_bytes=differing_bytes)
    store = InMemoryArtifactStore()

    result = run_compare(TARGET_REPO, base_repo=BASE_REPO, store=store, api=None)

    outcomes = _outcomes_by_id(result)
    assert outcomes["block0_shapes"].status == "pass"
    assert outcomes["block0_values"].status == "fail"
    assert result.final_verdict == "verified_derivative"


def test_shapes_pass_values_match_is_verified_derivative(scenario):
    scenario()
    store = InMemoryArtifactStore()

    result = run_compare(TARGET_REPO, base_repo=BASE_REPO, store=store, api=None)

    outcomes = _outcomes_by_id(result)
    assert outcomes["arch_hash"].status == "pass"
    assert outcomes["block0_shapes"].status == "pass"
    assert outcomes["block0_values"].status == "pass"
    assert result.final_verdict == "verified_derivative"
    assert result.support_class == "dense_supported"


def test_resolve_refs_compare_start_error_propagates(monkeypatch):
    def _raise_gated(repo_id: str, revision: str | None = None, *, api=None):
        raise CompareStartError("gated_unauthenticated", "no access")

    monkeypatch.setattr(resolve_refs_mod, "resolve_commit", _raise_gated)
    store = InMemoryArtifactStore()

    with pytest.raises(CompareStartError) as exc_info:
        run_compare(TARGET_REPO, base_repo=BASE_REPO, store=store, api=None)

    assert exc_info.value.error_code == "gated_unauthenticated"


def test_base_source_is_card_when_base_repo_not_given(scenario, monkeypatch):
    scenario()
    monkeypatch.setattr(
        resolve_refs_mod,
        "resolve_presumed_base",
        lambda repo_id, sha, *, api=None: BASE_REPO,
    )
    store = InMemoryArtifactStore()

    result = run_compare(TARGET_REPO, store=store, api=None)

    assert result.base.repo_id == BASE_REPO
    assert result.base_source == "card"
