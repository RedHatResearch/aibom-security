"""Frozen v1 evaluation harness for ProvenanceBench."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from pbench import load
from pbench.load import bench_root, canonical_sha256, manifest_hashes, model_pin

EVAL_VERSION = "pbench-eval/1"
VALID_VERDICTS = frozenset({"related", "unrelated", "abstain"})


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


def _method_sha256(root: Path) -> str:
    return hashlib.sha256((root / "detect.py").read_bytes()).hexdigest()


def _apply_case_filters(
    selected: list[dict[str, Any]],
    *,
    stack: str | None,
    kinds: list[str] | None,
) -> list[dict[str, Any]]:
    if stack is not None and stack not in load.VALID_STACK:
        msg = f"stack must be dense or moe, got {stack!r}"
        raise ValueError(msg)
    kind_set = None
    if kinds is not None:
        if not kinds or any(not isinstance(kind, str) or not kind for kind in kinds):
            msg = "--kind values must be non-empty strings"
            raise ValueError(msg)
        kind_set = set(kinds)
    if stack is None and kind_set is None:
        return selected
    filtered = [
        case
        for case in selected
        if (stack is None or case["stack"] == stack)
        and (kind_set is None or case["kind"] in kind_set)
    ]
    if not filtered:
        parts: list[str] = []
        if stack is not None:
            parts.append(f"stack={stack}")
        if kind_set is not None:
            parts.append(f"kind in {{{', '.join(sorted(kind_set))}}}")
        msg = f"selection matched no cases after filters ({', '.join(parts)})"
        raise ValueError(msg)
    return filtered


def _resolve_ids(
    cases: list[dict[str, Any]],
    *,
    cohort: str | None,
    ids: list[str] | None,
    root: Path,
    stack: str | None = None,
    kinds: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if cohort is not None and ids is not None:
        msg = "--cohort and --ids are mutually exclusive"
        raise ValueError(msg)
    has_filters = stack is not None or bool(kinds)
    if cohort is None and ids is None and not has_filters:
        msg = "pass --cohort, --ids, or a --stack/--kind filter; no default cohort exists"
        raise ValueError(msg)
    selection: dict[str, Any]
    if cohort is not None:
        cohort_row = load.load_cohort(cohort, root)
        resolved_ids = list(cohort_row["ids"])
        selection = {
            "mode": "cohort",
            "requested": cohort,
            "resolved_ids": resolved_ids,
            "selection_sha256": canonical_sha256({"name": cohort, **cohort_row}),
        }
    elif ids is not None:
        resolved_ids = list(ids)
        selection = {
            "mode": "ids",
            "requested": resolved_ids,
            "resolved_ids": resolved_ids,
            "selection_sha256": canonical_sha256({"ids": resolved_ids}),
        }
    else:
        resolved_ids = [case["id"] for case in cases]
        selection = {
            "mode": "manifest",
            "requested": None,
            "resolved_ids": resolved_ids,
            "selection_sha256": canonical_sha256({"ids": resolved_ids}),
        }
    if not resolved_ids:
        msg = "selection resolved no case ids"
        raise ValueError(msg)
    by_id = load.case_by_id(cases)
    missing = [case_id for case_id in resolved_ids if case_id not in by_id]
    if missing:
        msg = f"unknown case ids: {', '.join(missing)}"
        raise ValueError(msg)
    if len(resolved_ids) != len(set(resolved_ids)):
        msg = "selection contains duplicate case ids"
        raise ValueError(msg)
    selected = [by_id[case_id] for case_id in resolved_ids]
    if has_filters:
        if kinds:
            known_kinds = load.kind_index(load.load_kinds(root))
            unknown = sorted({kind for kind in kinds if kind not in known_kinds})
            if unknown:
                msg = f"unknown kind(s): {', '.join(unknown)}"
                raise ValueError(msg)
        selected = _apply_case_filters(selected, stack=stack, kinds=kinds)
        selection["resolved_ids"] = [case["id"] for case in selected]
        selection["filters"] = {
            "stack": stack,
            "kinds": list(kinds) if kinds else None,
        }
    return selected, selection


def _correct(expected: str, verdict: str) -> bool:
    return expected == verdict or (expected == "unsupported" and verdict == "abstain")


def _error_class(expected: str, verdict: str) -> str | None:
    if _correct(expected, verdict):
        return None
    if verdict == "related" and expected == "unrelated":
        return "false_confirm"
    if verdict == "related" and expected == "unsupported":
        return "unsupported_false_confirm"
    if expected == "related":
        return "miss"
    if expected == "unrelated" and verdict == "abstain":
        return "abstain_on_unrelated"
    if expected == "unsupported" and verdict == "unrelated":
        return "reject_unsupported"
    return "other_mismatch"


def _result_fields(result: object) -> tuple[str, float | None, dict[str, Any], str, str | None]:
    verdict = getattr(result, "verdict", None)
    score = getattr(result, "score", None)
    diagnostics = getattr(result, "diagnostics", None)
    note = getattr(result, "note", "")
    abstain_reason = getattr(result, "abstain_reason", None)
    if verdict not in VALID_VERDICTS:
        msg = "detector result verdict must be related, unrelated, or abstain"
        raise ValueError(msg)
    if score is not None and (not isinstance(score, (int, float)) or isinstance(score, bool)):
        msg = "detector result score must be a number or null"
        raise ValueError(msg)
    if not isinstance(diagnostics, dict):
        msg = "detector result diagnostics must be a mapping"
        raise ValueError(msg)
    if not isinstance(note, str) or not (abstain_reason is None or isinstance(abstain_reason, str)):
        msg = "detector result note and abstain_reason must be strings or null"
        raise ValueError(msg)
    return verdict, float(score) if score is not None else None, diagnostics, note, abstain_reason


def _method_case_record(
    case: dict[str, Any],
    *,
    subject: str,
    subject_rev: str | None,
    reference: str,
    reference_rev: str | None,
    verdict: str,
    score: float | None,
    diagnostics: dict[str, Any],
    note: str,
    abstain_reason: str | None,
) -> dict[str, Any]:
    return {
        "type": "case",
        "id": case["id"],
        "stack": case["stack"],
        "subject": subject,
        "subject_rev": subject_rev,
        "reference": reference,
        "reference_rev": reference_rev,
        "verdict": verdict,
        "score": score,
        "diagnostics": diagnostics,
        "note": note,
        "abstain_reason": abstain_reason,
    }


def _review_case_record(method_case: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    verdict = method_case["verdict"]
    correct = _correct(expected, verdict)
    return {
        **method_case,
        "expected": expected,
        "kind": case["kind"],
        "evidence": case["evidence"],
        "context": case.get("context"),
        "lineage_scope": case.get("lineage_scope"),
        "challenge": case.get("challenge"),
        "correct": correct,
        "mismatch": not correct,
        "error_class": _error_class(expected, verdict),
    }


def _rates(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    by_expected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        by_expected[row["expected"]].append(row)
        confusion[row["expected"]][row["verdict"]] += 1
    n = len(rows)
    correct = sum(row["correct"] for row in rows)
    abstain = sum(row["verdict"] == "abstain" for row in rows)
    related = by_expected["related"]
    unrelated = by_expected["unrelated"]
    unsupported = by_expected["unsupported"]
    class_rates = [
        sum(row["correct"] for row in expected_rows) / len(expected_rows)
        for expected_rows in (related, unrelated, unsupported)
        if expected_rows
    ]

    def slices(field: str, *, challenges_only: bool = False) -> dict[str, dict[str, float | int]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if challenges_only:
                challenge = row.get("challenge")
                if challenge is None:
                    continue
                value = challenge["type"]
            else:
                value = row[field]
            groups[value].append(row)
        return {
            value: {"n": len(group), "acc": sum(row["correct"] for row in group) / len(group)}
            for value, group in sorted(groups.items())
        }

    return {
        "acc": correct / n,
        "bal_acc": sum(class_rates) / len(class_rates) if class_rates else 0.0,
        "cov": (n - abstain) / n,
        "abstain": abstain / n,
        "false_confirm": sum(row["error_class"] == "false_confirm" for row in unrelated)
        / max(1, len(unrelated)),
        "miss": sum(row["error_class"] == "miss" for row in related) / max(1, len(related)),
        "unsupported_false_confirm": sum(
            row["error_class"] == "unsupported_false_confirm" for row in unsupported
        )
        / max(1, len(unsupported)),
        "confusion": {expected: dict(verdicts) for expected, verdicts in sorted(confusion.items())},
        "by_kind": slices("kind"),
        "by_stack": slices("stack"),
        "by_challenge": slices("challenge", challenges_only=True),
    }


def _summary_line(metrics: dict[str, Any], n: int) -> str:
    return (
        f"SUMMARY acc={metrics['acc']:.3f} bal_acc={metrics['bal_acc']:.3f} "
        f"cov={metrics['cov']:.3f} abstain={metrics['abstain']:.3f} "
        f"false_confirm={metrics['false_confirm']:.3f} miss={metrics['miss']:.3f} "
        f"unsupported_false_confirm={metrics['unsupported_false_confirm']:.3f} n={n}"
    )


def default_run_path(root: Path, method_sha256: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return root / "runs" / f"{timestamp}-{method_sha256[:12]}.method.jsonl"


def review_run_path(method_path: Path) -> Path:
    suffix = ".method.jsonl"
    if not method_path.name.endswith(suffix):
        msg = "method artifact path must end with .method.jsonl"
        raise ValueError(msg)
    return method_path.with_name(f"{method_path.name.removesuffix(suffix)}.review.jsonl")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        msg = f"refusing to overwrite immutable artifact: {path}"
        raise FileExistsError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _assert_artifact_pair_is_new(method_path: Path, review_path: Path) -> None:
    for path in (method_path, review_path):
        if path.exists():
            msg = f"refusing to overwrite immutable artifact: {path}"
            raise FileExistsError(msg)


def run_eval(
    root: Path | None = None,
    *,
    cohort: str | None = None,
    ids: list[str] | None = None,
    stack: str | None = None,
    kinds: list[str] | None = None,
    out_path: Path | None = None,
) -> int:
    root = root or bench_root()
    detect = _load_detect(root)
    cases = load.load_cases(root)
    selected, selection = _resolve_ids(
        cases, cohort=cohort, ids=ids, root=root, stack=stack, kinds=kinds
    )
    method_sha256 = _method_sha256(root)
    out_path = out_path if out_path is not None else default_run_path(root, method_sha256)
    review_path = review_run_path(out_path)
    _assert_artifact_pair_is_new(out_path, review_path)
    hashes = {
        **manifest_hashes(root),
        "selection_sha256": selection["selection_sha256"],
        "method_sha256": method_sha256,
    }
    meta_base = {
        "eval_version": EVAL_VERSION,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "selection": selection,
        "hashes": hashes,
        "n_cases": len(selected),
    }

    method_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for case in selected:
        subject, subject_rev = model_pin(case, "subject")
        reference, reference_rev = model_pin(case, "reference")
        result = detect.score(
            subject,
            reference,
            subject_rev=subject_rev,
            reference_rev=reference_rev,
            stack=case["stack"],
        )
        verdict, score, diagnostics, note, abstain_reason = _result_fields(result)
        method_case = _method_case_record(
            case,
            subject=subject,
            subject_rev=subject_rev,
            reference=reference,
            reference_rev=reference_rev,
            verdict=verdict,
            score=score,
            diagnostics=diagnostics,
            note=note,
            abstain_reason=abstain_reason,
        )
        method_rows.append(method_case)
        review_rows.append(_review_case_record(method_case, case))

    metrics = _rates(review_rows)
    summary = _summary_line(metrics, len(review_rows))
    _write_jsonl(
        out_path,
        [
            {"type": "meta", "artifact": "method", **meta_base},
            *method_rows,
            {"type": "trailer", "artifact": "method", "n_cases": len(method_rows)},
        ],
    )
    _write_jsonl(
        review_path,
        [
            {"type": "meta", "artifact": "review", **meta_base},
            *review_rows,
            {"type": "trailer", "artifact": "review", "summary": summary, **metrics},
        ],
    )
    print(summary, file=sys.stdout)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ProvenanceBench v1 evaluation")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--cohort", help="named cohort")
    selection.add_argument("--ids", nargs="+", help="ordered explicit case ids")
    parser.add_argument(
        "--stack",
        choices=sorted(load.VALID_STACK),
        default=None,
        help="keep only dense or moe pairs (AND with --kind)",
    )
    parser.add_argument(
        "--kind",
        action="append",
        dest="kinds",
        default=None,
        metavar="KIND",
        help=(
            "keep pairs whose kind equals KIND (repeatable; OR within kind, AND with "
            "--stack). Quants: pass exact kinds quant and/or quant_gguf_rehost."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="new method artifact path, ending in .method.jsonl",
    )
    parser.add_argument("--root", type=Path, default=None, help="bench data directory")
    args = parser.parse_args(argv)
    return run_eval(
        args.root,
        cohort=args.cohort,
        ids=args.ids,
        stack=args.stack,
        kinds=args.kinds,
        out_path=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
