"""Offline tests for ProvenanceBench manifest."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from pbench.eval import run_eval
from pbench.load import (
    bench_root,
    case_by_id,
    kind_index,
    load_cases,
    load_kinds,
    load_smoke_ids,
    model_pin,
)

VALID_Y = frozenset({"related", "unrelated", "unsupported"})
VALID_STACK = frozenset({"dense", "moe"})
VALID_FILL = frozenset({"hub", "partial", "script", "none"})
VALID_LABEL_CONFIDENCE = frozenset({"verified", "inferred", "disputed"})
REQUIRED_CASE_KEYS = frozenset({"id", "stack", "kind", "ref", "sus", "y", "cite"})
SMOKES_FIXTURES = Path(__file__).resolve().parents[2] / "smokes" / "fixtures.yaml"
ANCHOR_LABELS = {
    "pb-001": "positive_dense",
    "pb-002": "vocab_drift",
    "pb-003": "unrelated_cross_family",
}


def test_kinds_load() -> None:
    kinds = load_kinds()
    assert len(kinds) >= 30
    for row in kinds:
        assert row["fill"] in VALID_FILL
        assert row["stack"] in {"dense", "moe", "both"}


def test_pairs_schema() -> None:
    cases = load_cases()
    kinds = kind_index(load_kinds())
    assert len(cases) >= 10
    for case in cases:
        assert case.keys() >= REQUIRED_CASE_KEYS
        assert case["y"] in VALID_Y
        assert case["stack"] in VALID_STACK
        assert case["kind"] in kinds
        tier = case.get("label_confidence", "inferred")
        assert tier in VALID_LABEL_CONFIDENCE


def test_smoke_ids_exist() -> None:
    cases = case_by_id(load_cases())
    for cid in load_smoke_ids():
        assert cid in cases


def test_anchor_pairs_match_smokes_fixtures() -> None:
    fixtures = yaml.safe_load(SMOKES_FIXTURES.read_text(encoding="utf-8"))["pairs"]
    cases = case_by_id(load_cases())
    for case_id, label in ANCHOR_LABELS.items():
        case = cases[case_id]
        pair = fixtures[label]
        assert case["ref"] == pair["ref"]
        assert case["sus"] == pair["sus"]


def test_smoke_label_balance() -> None:
    cases = case_by_id(load_cases())
    smoke = load_smoke_ids()
    labels = [cases[cid]["y"] for cid in smoke]
    related = sum(label == "related" for label in labels)
    unrelated = sum(label == "unrelated" for label in labels)
    unsupported = sum(label == "unsupported" for label in labels)
    assert unrelated >= 2
    assert unsupported >= 2
    assert related <= len(smoke) // 2


def test_detect_does_not_import_bench_labels() -> None:
    detect_src = (bench_root() / "detect.py").read_text(encoding="utf-8")
    banned = ("load_cases", "pairs.yaml", "case_by_id", "pbench.load", "yaml.safe_load")
    for term in banned:
        assert term not in detect_src


def test_always_related_baseline_scores_low_bal_acc() -> None:
    root = bench_root()
    cheat = root / "detect_cheat_always_related.py"
    cheat.write_text(
        """\
from dataclasses import dataclass
from typing import Literal

Pred = Literal["related", "unrelated", "abstain"]

@dataclass(frozen=True)
class Score:
    pred: Pred
    confidence: float
    note: str = ""

def score(ref, sus, *, ref_rev=None, sus_rev=None, stack):
    _ = (ref, sus, ref_rev, sus_rev, stack)
    return Score(pred="related", confidence=1.0)
""",
        encoding="utf-8",
    )
    real_detect = root / "detect.py"
    backup = root / "detect.py.bak"
    real_detect.rename(backup)
    cheat.rename(real_detect)
    try:
        run_path = root / "runs" / "test-cheat.jsonl"
        if run_path.exists():
            run_path.unlink()
        result = run_eval(root, suite="smoke", out_path=run_path)
        assert result == 0
        summary = run_path.read_text(encoding="utf-8").strip().splitlines()[-1]
        assert '"bal_acc"' in summary
        trailer = json.loads(summary)
        assert trailer["bal_acc"] < 0.7
    finally:
        real_detect.rename(cheat)
        backup.rename(real_detect)
        if run_path.exists():
            run_path.unlink()


def test_model_pin_optional_rev() -> None:
    case = case_by_id(load_cases())["pb-001"]
    assert model_pin(case, "ref") == ("HuggingFaceTB/SmolLM2-1.7B", "effd688a1292")
    assert model_pin(case, "sus")[0].startswith("SultanR/")
    case_no_rev = case_by_id(load_cases())["pb-003"]
    assert model_pin(case_no_rev, "ref") == ("HuggingFaceTB/SmolLM2-1.7B", None)


def test_eval_smoke_dry_run() -> None:
    root = bench_root()
    run_path = root / "runs" / "test-latest.jsonl"
    triage_path = root / "runs" / "test-latest.triage.jsonl"
    for path in (run_path, triage_path):
        if path.exists():
            path.unlink()
    result = subprocess.run(
        [sys.executable, "-m", "pbench.eval", "--suite", "smoke", "--out", str(run_path)],
        cwd=root.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "SUMMARY acc=" in result.stdout
    assert "bal_acc=" in result.stdout
    assert "false_confirm=" in result.stdout
    assert run_path.is_file()
    assert triage_path.is_file()
    lines = run_path.read_text(encoding="utf-8").strip().splitlines()
    smoke_n = len(load_smoke_ids())
    assert len(lines) == smoke_n + 2
    meta = json.loads(lines[0])
    assert meta["type"] == "meta"
    assert "pairs_sha256" in meta
    case_row = json.loads(lines[1])
    assert case_row["type"] == "case"
    assert "ref_id" in case_row
    assert "cite" in case_row
    assert "label_confidence" in case_row
    trailer = json.loads(lines[-1])
    assert trailer["type"] == "trailer"
    assert "control_subset" in trailer
    assert "confusion" in trailer
    triage_lines = triage_path.read_text(encoding="utf-8").strip().splitlines()
    triage_case = json.loads(triage_lines[1])
    assert "y" not in triage_case
    assert "mismatch" in triage_case
