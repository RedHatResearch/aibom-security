"""Orchestrator forward-walk + default rules (FR-B Task 2)."""

from typing import cast

import pytest
from hf_fakes import (
    fake_fetch_tensor_bytes,
    fake_list_tensor_names,
    fake_load_config_json,
    fake_resolve_commit,
    fake_tensor_shapes,
)
from huggingface_hub import HfApi

from aibom_verifier.mapping.block0 import BLOCK0_PREFIX, REQUIRED_SUFFIXES
from aibom_verifier.nodes import block0_shapes as block0_shapes_mod
from aibom_verifier.nodes import block0_values as block0_values_mod
from aibom_verifier.nodes import resolve_refs as resolve_refs_mod
from aibom_verifier.nodes.stub_float_score import stub_float_score_node
from aibom_verifier.orchestrator import run_test_run
from aibom_verifier.planner import DEFAULT_REGISTRY, run_compare
from aibom_verifier.rules import Requirement, Rule, default_rules_path, load_rules
from aibom_verifier.slots.artifact_store import InMemoryArtifactStore
from aibom_verifier.slots.proxy_store import (
    InMemoryBlobBackend,
    InMemoryMetadataBackend,
    ProxyArtifactStore,
)
from aibom_verifier.types import TestOutcome

_DUMMY_API = cast(HfApi, object())

REQUIRED_NAMES = [BLOCK0_PREFIX + suffix for suffix in REQUIRED_SUFFIXES]

TARGET_REPO = "org/target"
BASE_REPO = "org/base"
TARGET_SHA = "tsha"
BASE_SHA = "bsha"


@pytest.fixture
def scenario(monkeypatch):
    def _configure(*, base_shapes=None):
        shas = {TARGET_REPO: TARGET_SHA, BASE_REPO: BASE_SHA}
        configs = {
            TARGET_REPO: {"model_type": "llama"},
            BASE_REPO: {"model_type": "llama"},
        }
        monkeypatch.setattr(resolve_refs_mod, "resolve_commit", fake_resolve_commit(shas))
        monkeypatch.setattr(resolve_refs_mod, "load_config_json", fake_load_config_json(configs))

        names_by_repo = {TARGET_REPO: REQUIRED_NAMES, BASE_REPO: REQUIRED_NAMES}
        monkeypatch.setattr(
            block0_shapes_mod, "list_tensor_names", fake_list_tensor_names(names_by_repo)
        )

        default_shapes = {name: [4] for name in REQUIRED_NAMES}
        shapes_by_repo = {
            TARGET_REPO: default_shapes,
            BASE_REPO: base_shapes if base_shapes is not None else default_shapes,
        }
        monkeypatch.setattr(block0_shapes_mod, "tensor_shapes", fake_tensor_shapes(shapes_by_repo))

        default_bytes = {name: b"identical-payload" for name in REQUIRED_NAMES}
        bytes_by_repo = {TARGET_REPO: default_bytes, BASE_REPO: default_bytes}
        monkeypatch.setattr(
            block0_values_mod, "fetch_tensor_bytes", fake_fetch_tensor_bytes(bytes_by_repo)
        )

    return _configure


def _outcomes_by_id(result):
    return {t.test_id: t for t in result.tests}


def test_load_default_rules_maps_yaml_short_names():
    rules = load_rules()
    assert [r.test_id for r in rules] == [
        "support_classify",
        "arch_hash",
        "block0_shapes",
        "block0_values",
    ]
    assert rules[0].requires == []
    assert rules[1].requires == [
        Requirement(
            upstream_test_id="support_classify",
            kind="status",
            expect_status="pass",
        )
    ]
    assert default_rules_path().is_file()


def test_shapes_fail_skips_values_with_upstream(scenario):
    mismatched = {name: [4] for name in REQUIRED_NAMES}
    mismatched[REQUIRED_NAMES[0]] = [999]
    scenario(base_shapes=mismatched)
    store = InMemoryArtifactStore()

    result = run_compare(TARGET_REPO, base_repo=BASE_REPO, store=store, api=None)

    outcomes = _outcomes_by_id(result)
    assert outcomes["block0_shapes"].status == "fail"
    assert outcomes["block0_values"].status == "skip"
    assert outcomes["block0_values"].skipped_because == {
        "upstream": "block0_shapes",
        "reason": "incompatible",
    }
    assert result.final_verdict == "incompatible"


def test_run_test_run_happy_path_verified_derivative(scenario):
    scenario()
    store = InMemoryArtifactStore()
    registry = dict(DEFAULT_REGISTRY)
    result = run_test_run(
        TARGET_REPO,
        base_repo=BASE_REPO,
        store=store,
        rules=load_rules(),
        registry=registry,
        api=None,
    )

    outcomes = _outcomes_by_id(result)
    assert outcomes["block0_values"].status == "pass"
    assert result.final_verdict == "verified_derivative"


def test_float_score_rule_skips_downstream(scenario):
    """Stub float gate via custom rules (not in default YAML)."""
    scenario()

    def downstream_stub_node(inputs: dict, store) -> TestOutcome:
        del store, inputs
        return TestOutcome(test_id="downstream_stub", status="pass")

    rules = [
        *load_rules(),
        Rule(test_id="stub_float_score", requires=[]),
        Rule(
            test_id="downstream_stub",
            requires=[
                Requirement(
                    upstream_test_id="stub_float_score",
                    kind="score",
                    score_key="x",
                    threshold=0.8,
                )
            ],
        ),
    ]
    registry = {
        **DEFAULT_REGISTRY,
        "stub_float_score": stub_float_score_node,
        "downstream_stub": downstream_stub_node,
    }
    store = InMemoryArtifactStore()

    result = run_test_run(
        TARGET_REPO,
        base_repo=BASE_REPO,
        store=store,
        rules=rules,
        registry=registry,
        api=None,
        extra_inputs={"score_x": 0.5},
    )

    outcomes = _outcomes_by_id(result)
    assert outcomes["stub_float_score"].scores["x"] == 0.5
    assert outcomes["downstream_stub"].status == "skip"
    assert outcomes["downstream_stub"].skipped_because == {
        "upstream": "stub_float_score",
        "reason": "score_below_threshold",
    }


def test_float_score_rule_allows_downstream_when_above_threshold(scenario):
    scenario()

    def downstream_stub_node(inputs: dict, store) -> TestOutcome:
        del store, inputs
        return TestOutcome(test_id="downstream_stub", status="pass")

    rules = [
        *load_rules(),
        Rule(test_id="stub_float_score", requires=[]),
        Rule(
            test_id="downstream_stub",
            requires=[
                Requirement(
                    upstream_test_id="stub_float_score",
                    kind="score",
                    score_key="x",
                    threshold=0.8,
                )
            ],
        ),
    ]
    registry = {
        **DEFAULT_REGISTRY,
        "stub_float_score": stub_float_score_node,
        "downstream_stub": downstream_stub_node,
    }
    store = InMemoryArtifactStore()

    result = run_test_run(
        TARGET_REPO,
        base_repo=BASE_REPO,
        store=store,
        rules=rules,
        registry=registry,
        api=None,
        extra_inputs={"score_x": 0.9},
    )

    outcomes = _outcomes_by_id(result)
    assert outcomes["downstream_stub"].status == "pass"


def test_node_overrides_replace_registry_entry(scenario):
    scenario()
    store = InMemoryArtifactStore()
    calls: list[str] = []

    def override_support(inputs: dict, store) -> TestOutcome:
        del store
        calls.append("override")
        return TestOutcome(
            test_id="support_classify",
            status="pass",
            compatibility="compatible",
            detail={
                "support_class": "dense_supported",
                "target_model_type": inputs["target_config"]["model_type"],
                "base_model_type": inputs["base_config"]["model_type"],
            },
        )

    result = run_compare(
        TARGET_REPO,
        base_repo=BASE_REPO,
        store=store,
        api=None,
        node_overrides={"support_classify": override_support},
    )

    assert calls == ["override"]
    assert result.final_verdict == "verified_derivative"
    assert isinstance(result.cache["hits"], list)
    assert isinstance(result.cache["misses"], list)


def test_counting_store_records_second_run_hits(scenario):
    scenario()
    store = InMemoryArtifactStore()

    first = run_compare(TARGET_REPO, base_repo=BASE_REPO, store=store, api=None)
    second = run_compare(TARGET_REPO, base_repo=BASE_REPO, store=store, api=None)

    assert first.cache["misses"]
    assert len(second.cache["hits"]) > len(first.cache["hits"])


def test_proxy_store_reuse_across_run_compare(scenario):
    """Orchestrator-level reuse against a durable (fake) proxy store."""
    scenario()
    store = ProxyArtifactStore(InMemoryMetadataBackend(), InMemoryBlobBackend())

    first = run_compare(TARGET_REPO, base_repo=BASE_REPO, store=store, api=None)
    second = run_compare(TARGET_REPO, base_repo=BASE_REPO, store=store, api=None)

    assert first.cache["misses"]
    assert len(second.cache["hits"]) > len(first.cache["hits"])
    assert second.final_verdict == "verified_derivative"


def test_proxy_ignore_cache_still_writes(scenario):
    scenario()
    meta = InMemoryMetadataBackend()
    blobs = InMemoryBlobBackend()
    writer = ProxyArtifactStore(meta, blobs, ignore_cache=True)
    run_compare(TARGET_REPO, base_repo=BASE_REPO, store=writer, api=None)

    reader = ProxyArtifactStore(meta, blobs, ignore_cache=False)
    assert reader.exists(f"config:{TARGET_REPO}:{TARGET_SHA}") is True


def _bootstrap_resolve_refs(inputs: dict, store) -> TestOutcome:
    return TestOutcome(
        test_id="resolve_refs",
        status="pass",
        detail={
            "target_sha": "t",
            "base_sha": "b",
            "base_repo": "org/base",
            "base_source": "cli",
            "target_config": {"model_type": "llama"},
            "base_config": {"model_type": "llama"},
        },
    )


def test_orchestrator_strips_api_for_non_local_backend():
    class RemoteBackend:
        def __init__(self) -> None:
            self.last_inputs: dict | None = None

        def run(self, node_id: str, inputs: dict, store) -> TestOutcome:
            self.last_inputs = inputs
            if node_id == "support_classify":
                return TestOutcome(
                    test_id=node_id,
                    status="pass",
                    detail={"support_class": "dense_supported"},
                )
            return TestOutcome(test_id=node_id, status="pass")

    def must_not_run(inputs: dict, store) -> TestOutcome:
        raise AssertionError("local node must not run for non-local backend")

    remote = RemoteBackend()
    run_test_run(
        "org/target",
        base_repo="org/base",
        store=InMemoryArtifactStore(),
        rules=[
            Rule(test_id="support_classify", requires=[]),
            Rule(test_id="probe", requires=[]),
        ],
        registry={
            "resolve_refs": _bootstrap_resolve_refs,
            "support_classify": must_not_run,
            "probe": must_not_run,
        },
        backend=remote,
        api=_DUMMY_API,
        extra_inputs={"api": "should-strip", "keep": 1},
    )
    assert remote.last_inputs is not None
    assert "api" not in remote.last_inputs
    assert remote.last_inputs.get("keep") == 1


def test_orchestrator_local_backend_keeps_api():
    seen: dict | None = None

    def probe(inputs: dict, store) -> TestOutcome:
        nonlocal seen
        seen = dict(inputs)
        return TestOutcome(test_id="probe", status="pass")

    def support(inputs: dict, store) -> TestOutcome:
        return TestOutcome(
            test_id="support_classify",
            status="pass",
            detail={"support_class": "dense_supported"},
        )

    run_test_run(
        "org/target",
        base_repo="org/base",
        store=InMemoryArtifactStore(),
        rules=[
            Rule(test_id="support_classify", requires=[]),
            Rule(test_id="probe", requires=[]),
        ],
        registry={
            "resolve_refs": _bootstrap_resolve_refs,
            "support_classify": support,
            "probe": probe,
        },
        api=_DUMMY_API,
        extra_inputs={"api": "keep-me", "keep": 1},
    )
    assert seen is not None
    assert seen.get("api") == "keep-me"
    assert seen.get("keep") == 1


def test_orchestrator_backend_raise_becomes_error_outcome():
    class BoomBackend:
        def run(self, node_id: str, inputs: dict, store) -> TestOutcome:
            raise RuntimeError("ssh failed")

    result = run_test_run(
        "org/target",
        base_repo="org/base",
        store=InMemoryArtifactStore(),
        rules=[Rule(test_id="support_classify", requires=[])],
        registry={"resolve_refs": _bootstrap_resolve_refs},
        backend=BoomBackend(),
        api=_DUMMY_API,
    )
    outcome = next(t for t in result.tests if t.test_id == "support_classify")
    assert outcome.status == "error"
    assert outcome.reason_codes == ["backend_exception"]
    assert "ssh failed" in outcome.detail["message"]
