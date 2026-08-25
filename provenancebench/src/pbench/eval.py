"""Frozen eval harness for ProvenanceBench."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from pbench import load
from pbench.load import bench_root, label_confidence, manifest_hashes, model_pin

EVAL_VERSION = "pbench-eval/0.2"


def _load_detect(root: Path) -> ModuleType:
    path = root / "detect.py"
    spec = importlib.util.spec_from_file_location("bench_detect", path)
    if spec is None or spec.loader is None:
        msg = f"cannot load detector from {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "score"):
        msg = f"{path} must define score(...)"
        raise AttributeError(msg)
    return module


def _detect_sha256(root: Path) -> str:
    return hashlib.sha256((root / "detect.py").read_bytes()).hexdigest()[:12]


def _select_cases(
    cases: list[dict[str, Any]],
    *,
    suite: str,
    smoke_ids: list[str],
) -> list[dict[str, Any]]:
    if suite == "smoke":
        by_id = load.case_by_id(cases)
        return [by_id[cid] for cid in smoke_ids]
    if suite == "full":
        return cases
    msg = f"unknown suite: {suite}"
    raise ValueError(msg)


def _error_class(gold: str, pred: str) -> str | None:
    if gold == pred:
        return None
    if pred == "related" and gold == "unrelated":
        return "false_confirm"
    if pred == "related" and gold == "unsupported":
        return "unsupported_false_confirm"
    if gold == "related" and pred != "related":
        return "miss"
    if gold == "unrelated" and pred == "abstain":
        return "abstain_on_unrelated"
    if gold == "unsupported" and pred == "unrelated":
        return "reject_unsupported"
    if gold == "unsupported" and pred == "related":
        return "confirm_unsupported"
    return "other_mismatch"


def _is_control(case: dict[str, Any]) -> bool:
    kind = case.get("kind", "")
    y = case.get("y")
    return y in {"unrelated", "unsupported"} or (
        isinstance(kind, str) and kind.startswith("neg_")
    )


def _case_record(
    case: dict[str, Any],
    *,
    ref_id: str,
    ref_rev: str | None,
    sus_id: str,
    sus_rev: str | None,
    pred: str,
    confidence: float,
    note: str,
    ok: bool,
    suite: str,
    smoke_sha: str,
) -> dict[str, Any]:
    gold = case["y"]
    tier = label_confidence(case)
    return {
        "type": "case",
        "id": case["id"],
        "suite": suite,
        "stack": case["stack"],
        "kind": case["kind"],
        "ref_id": ref_id,
        "sus_id": sus_id,
        "ref_rev": ref_rev,
        "sus_rev": sus_rev,
        "claim": case.get("claim"),
        "cite": case["cite"],
        "label_confidence": tier,
        "y": gold,
        "pred": pred,
        "score": confidence,
        "note": note,
        "ok": ok,
        "error_class": _error_class(gold, pred),
        "control": _is_control(case),
        "smoke_sha": smoke_sha,
    }


def _triage_record(case_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "case",
        "id": case_row["id"],
        "suite": case_row["suite"],
        "stack": case_row["stack"],
        "kind": case_row["kind"],
        "ref_id": case_row["ref_id"],
        "sus_id": case_row["sus_id"],
        "ref_rev": case_row["ref_rev"],
        "sus_rev": case_row["sus_rev"],
        "claim": case_row["claim"],
        "cite": case_row["cite"],
        "label_confidence": case_row["label_confidence"],
        "pred": case_row["pred"],
        "score": case_row["score"],
        "note": case_row["note"],
        "mismatch": not case_row["ok"],
        "control": case_row["control"],
        "smoke_sha": case_row["smoke_sha"],
    }


def _summary_line(
    *,
    acc: float,
    bal_acc: float,
    cov: float,
    abstain: float,
    false_confirm: float,
    miss: float,
    unsupported_false_confirm: float,
    n: int,
    smoke_sha: str,
) -> str:
    return (
        f"SUMMARY acc={acc:.3f} bal_acc={bal_acc:.3f} cov={cov:.3f} abstain={abstain:.3f} "
        f"false_confirm={false_confirm:.3f} miss={miss:.3f} "
        f"unsupported_false_confirm={unsupported_false_confirm:.3f} n={n} "
        f"smoke_sha={smoke_sha}"
    )


def default_run_path(root: Path) -> Path:
    return root / "runs" / "latest.jsonl"


def triage_run_path(eval_path: Path) -> Path:
    return eval_path.with_name(f"{eval_path.stem}.triage{eval_path.suffix}")


def run_eval(
    root: Path | None = None,
    *,
    suite: str = "smoke",
    out_path: Path | None = None,
) -> int:
    root = root or bench_root()
    detect = _load_detect(root)
    cases = load.load_cases(root)
    smoke_ids, smoke_sha = load.load_smoke(root)
    selected = _select_cases(cases, suite=suite, smoke_ids=smoke_ids)
    out_path = out_path if out_path is not None else default_run_path(root)
    triage_path = triage_run_path(out_path)
    hashes = manifest_hashes(root)

    outcomes: list[dict[str, Any]] = []
    triage_rows: list[dict[str, Any]] = []
    correct = 0
    abstain_count = 0
    false_confirm = 0
    miss = 0
    unsupported_false_confirm = 0
    gold_unrelated = 0
    gold_related = 0
    gold_unsupported = 0
    related_ok = 0
    unrelated_ok = 0
    unsupported_ok = 0
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0})
    by_stack: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0})
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for case in selected:
        ref_id, ref_rev = model_pin(case, "ref")
        sus_id, sus_rev = model_pin(case, "sus")
        result = detect.score(
            ref_id,
            sus_id,
            ref_rev=ref_rev,
            sus_rev=sus_rev,
            stack=case["stack"],
        )
        gold = case["y"]
        if gold == "unrelated":
            gold_unrelated += 1
        elif gold == "related":
            gold_related += 1
        elif gold == "unsupported":
            gold_unsupported += 1
        pred = result.pred
        ok = pred == gold
        if ok:
            correct += 1
            if gold == "related":
                related_ok += 1
            elif gold == "unrelated":
                unrelated_ok += 1
            elif gold == "unsupported":
                unsupported_ok += 1
        if pred == "abstain":
            abstain_count += 1
        if pred == "related" and gold == "unrelated":
            false_confirm += 1
        if pred == "related" and gold == "unsupported":
            unsupported_false_confirm += 1
        if gold == "related" and pred != "related":
            miss += 1
        kind = case["kind"]
        stack = case["stack"]
        by_kind[kind]["n"] += 1
        by_stack[stack]["n"] += 1
        if ok:
            by_kind[kind]["ok"] += 1
            by_stack[stack]["ok"] += 1
        confusion[gold][pred] += 1

        row = _case_record(
            case,
            ref_id=ref_id,
            ref_rev=ref_rev,
            sus_id=sus_id,
            sus_rev=sus_rev,
            pred=pred,
            confidence=result.confidence,
            note=result.note,
            ok=ok,
            suite=suite,
            smoke_sha=smoke_sha,
        )
        outcomes.append(row)
        triage_rows.append(_triage_record(row))

    n = len(selected) or 1
    acc = correct / n
    cov = (n - abstain_count) / n
    abstain_rate = abstain_count / n
    false_confirm_rate = false_confirm / max(1, gold_unrelated)
    miss_rate = miss / max(1, gold_related)
    unsupported_false_confirm_rate = unsupported_false_confirm / max(1, gold_unsupported)
    class_rates: list[float] = []
    if gold_related:
        class_rates.append(related_ok / gold_related)
    if gold_unrelated:
        class_rates.append(unrelated_ok / gold_unrelated)
    if gold_unsupported:
        class_rates.append(unsupported_ok / gold_unsupported)
    bal_acc = sum(class_rates) / len(class_rates) if class_rates else 0.0
    by_kind_acc = {
        k: {"n": v["n"], "acc": (v["ok"] / v["n"] if v["n"] else 0.0)}
        for k, v in sorted(by_kind.items())
    }
    by_stack_acc = {
        k: {"n": v["n"], "acc": (v["ok"] / v["n"] if v["n"] else 0.0)}
        for k, v in sorted(by_stack.items())
    }
    control_subset = [row["id"] for row in outcomes if row["control"]]
    summary = _summary_line(
        acc=acc,
        bal_acc=bal_acc,
        cov=cov,
        abstain=abstain_rate,
        false_confirm=false_confirm_rate,
        miss=miss_rate,
        unsupported_false_confirm=unsupported_false_confirm_rate,
        n=len(selected),
        smoke_sha=smoke_sha,
    )
    meta = {
        "type": "meta",
        "eval_version": EVAL_VERSION,
        "suite": suite,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "n_cases": len(selected),
        "detect_sha256": _detect_sha256(root),
        **hashes,
    }
    trailer = {
        "type": "trailer",
        "summary": summary,
        "bal_acc": bal_acc,
        "acc": acc,
        "cov": cov,
        "abstain": abstain_rate,
        "false_confirm": false_confirm_rate,
        "miss": miss_rate,
        "unsupported_false_confirm": unsupported_false_confirm_rate,
        "by_kind": by_kind_acc,
        "by_stack": by_stack_acc,
        "confusion": {g: dict(preds) for g, preds in sorted(confusion.items())},
        "control_subset": control_subset,
    }
    triage_trailer = {
        "type": "trailer",
        "summary": summary,
        "control_subset": control_subset,
        "by_kind": by_kind_acc,
        "by_stack": by_stack_acc,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(meta, sort_keys=True) + "\n")
        for row in outcomes:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.write(json.dumps(trailer, sort_keys=True) + "\n")

    with triage_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(meta, sort_keys=True) + "\n")
        for row in triage_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.write(json.dumps(triage_trailer, sort_keys=True) + "\n")

    print(summary, file=sys.stdout)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ProvenanceBench dry-run eval")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="run artifact path (default: provenancebench/runs/latest.jsonl)",
    )
    parser.add_argument("--root", type=Path, default=None, help="bench data directory")
    args = parser.parse_args(argv)
    return run_eval(args.root, suite=args.suite, out_path=args.out)


if __name__ == "__main__":
    raise SystemExit(main())
