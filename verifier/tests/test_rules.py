"""Rule predicate evaluation and float stub gate (FR-B Task 1)."""

from pathlib import Path

import pytest

from aibom_verifier.nodes.stub_float_score import stub_float_score_node
from aibom_verifier.rules import Requirement, load_rules, requirement_satisfied
from aibom_verifier.slots.artifact_store import InMemoryArtifactStore
from aibom_verifier.types import TestOutcome


def test_boolean_status_requirement_allows_when_upstream_passes():
    upstream = TestOutcome(
        test_id="block0_shapes",
        status="pass",
        compatibility="compatible",
    )
    req = Requirement(
        upstream_test_id="block0_shapes",
        kind="status",
        expect_status="pass",
    )
    assert requirement_satisfied(req, {"block0_shapes": upstream}) is True


def test_boolean_status_requirement_blocks_when_upstream_status_not_pass():
    upstream = TestOutcome(
        test_id="block0_shapes",
        status="fail",
        compatibility="incompatible",
    )
    req = Requirement(
        upstream_test_id="block0_shapes",
        kind="status",
        expect_status="pass",
    )
    assert requirement_satisfied(req, {"block0_shapes": upstream}) is False


def test_status_requirement_fails_closed_when_upstream_missing():
    req = Requirement(
        upstream_test_id="block0_shapes",
        kind="status",
        expect_status="pass",
    )
    assert requirement_satisfied(req, {}) is False


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.9, True),
        (0.8, False),  # strict greater-than
        (0.7, False),
    ],
)
def test_float_score_gate_threshold(score: float, expected: bool):
    upstream = TestOutcome(
        test_id="stub_float_score",
        status="pass",
        scores={"x": score},
    )
    req = Requirement(
        upstream_test_id="stub_float_score",
        kind="score",
        score_key="x",
        threshold=0.8,
    )
    assert requirement_satisfied(req, {"stub_float_score": upstream}) is expected


def test_score_requirement_fails_closed_when_score_key_missing():
    upstream = TestOutcome(
        test_id="stub_float_score",
        status="pass",
        scores={},
    )
    req = Requirement(
        upstream_test_id="stub_float_score",
        kind="score",
        score_key="x",
        threshold=0.8,
    )
    assert requirement_satisfied(req, {"stub_float_score": upstream}) is False


def test_stub_float_score_node_puts_score_in_outcome():
    store = InMemoryArtifactStore()
    outcome = stub_float_score_node({"score_x": 0.9}, store)
    assert outcome.test_id == "stub_float_score"
    assert outcome.status == "pass"
    assert outcome.scores == {"x": 0.9}


def test_stub_node_outcome_satisfies_float_gate():
    store = InMemoryArtifactStore()
    outcome = stub_float_score_node({"score_x": 0.9}, store)
    req = Requirement(
        upstream_test_id="stub_float_score",
        kind="score",
        score_key="x",
        threshold=0.8,
    )
    assert requirement_satisfied(req, {outcome.test_id: outcome}) is True


def test_load_rules_rejects_non_mapping_requirement(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "rules:\n  - test_id: arch_hash\n    requires:\n      - not-a-mapping\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requirement must be a mapping"):
        load_rules(path)


def test_load_rules_maps_score_short_names(tmp_path: Path):
    path = tmp_path / "score.yaml"
    path.write_text(
        "rules:\n"
        "  - test_id: downstream\n"
        "    requires:\n"
        "      - upstream: stub_float_score\n"
        "        score: x\n"
        "        gt: 0.8\n",
        encoding="utf-8",
    )
    rules = load_rules(path)
    assert rules[0].requires == [
        Requirement(
            upstream_test_id="stub_float_score",
            kind="score",
            score_key="x",
            threshold=0.8,
        )
    ]
