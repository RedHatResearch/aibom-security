"""MoE expert-lineage smoke — Experiment A (issue #26).

Builds eleven derived checkpoints from each cached parent MoE with known ground truth, writes
them to ``.scratch/moe-expert-lineage/<model>/<variant>/`` as safetensors, then runs the
label-blind weights-only detector over the written files. Scores the expectation registry in
config.py (pre-registered plus post-hoc claims) and writes ``results.json``.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import detect  # noqa: E402
import synth  # noqa: E402
from config import (  # noqa: E402
    ALPHA_TOL,
    ARCHES,
    BLOCK,
    EXACT_COS,
    EXPECTATIONS,
    MERGE_ALPHA,
    MERGE_GAIN_MIN,
    MERGED_VARIANT_ALPHA,
    MIN_COMPONENT,
    NEAR_EXACT_COS,
    NOISY_SIGMA,
    PRIMARY,
    RESULTS,
    SALIENCY_ALPHA,
    SCRATCH,
    SECONDARY,
    SECONDARY_BUDGET_S,
    SEED,
    SHALLOW_ALPHA,
    SKEWED_ALPHA,
    THREE_ALPHA,
    TIES_ALPHA,
    TIES_ALPHA_TOL,
    TIES_TRIM,
    VARIANTS,
)

COMMAND = "uv run --group smokes python smokes/moe-expert-lineage/smoke.py"
ORIGINS = {item["variant"]: item["origin"] for item in EXPECTATIONS}


# --------------------------------------------------------------------------- #
# variant build + detection
# --------------------------------------------------------------------------- #
def run_variant(
    model: dict,
    variant: str,
    layers: list[int],
    parent_layer,
    parent_reader,
    parent_desc,
) -> dict:
    rec: dict = {"variant": variant, "status": "ok"}
    arch = model["arch"]
    roles = ARCHES[arch]["roles"]
    experts = model["num_experts"]
    out_dir = SCRATCH / model["label"] / variant
    t0 = time.perf_counter()

    try:
        tensors: dict[str, torch.Tensor] = {}
        truth: dict[int, dict] = {}
        counts: dict[int, int] = {}
        for layer in layers:
            layer_experts, router = parent_layer(layer)
            built, built_router, ground_truth = synth.build_layer(
                variant,
                layer,
                layer_experts,
                router,
                synth.layer_rng(variant, layer, SEED),
            )
            truth[layer] = ground_truth
            parents = ground_truth.get("merge_parents")
            if parents:
                desc = parent_desc[layer]
                rec.setdefault("parent_pair_cos", {})[str(layer)] = {
                    f"{a},{b}": float(desc[a] @ desc[b])
                    for index, a in enumerate(parents)
                    for b in parents[index + 1 :]
                }
            counts[layer] = len(built)
            for slot, expert in enumerate(built):
                for role in roles:
                    tensors[synth.tensor_name(arch, layer, slot, role)] = expert[role]
            tensors[synth.router_name(arch, layer)] = built_router
        synth.write_variant(out_dir, arch, tensors, counts, SEED)
        del tensors
    except Exception as exc:  # noqa: BLE001 - one bad variant must not abort the run
        print(f"SKIP {model['label']}/{variant}: build failed: {type(exc).__name__}: {exc}")
        rec.update(
            status="skipped",
            reason=f"build: {type(exc).__name__}: {exc}",
            seconds=round(time.perf_counter() - t0, 2),
        )
        return rec

    rec["injected"] = {str(layer): truth[layer] for layer in layers}
    rec["file_bytes"] = sum(p.stat().st_size for p in out_dir.iterdir() if p.is_file())

    try:
        manifest = detect.read_manifest(out_dir)
        if manifest.get("arch") != arch:
            raise detect.DetectionError(f"manifest arch {manifest.get('arch')!r} != {arch!r}")
        if tuple(manifest.get("roles") or ()) != roles:
            raise detect.DetectionError(
                "manifest roles differ from the architecture; refusing to score"
            )
        if manifest.get("tensor_file") != "model.safetensors":
            raise detect.DetectionError(f"unsupported tensor_file {manifest.get('tensor_file')!r}")
        reader = detect.CheckpointReader(
            out_dir,
            detect.TensorSpec(**ARCHES[arch]),
            tensor_file=manifest["tensor_file"],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP {model['label']}/{variant}: {type(exc).__name__}: {exc}")
        rec.update(
            status="skipped",
            reason=f"{type(exc).__name__}: {exc}",
            seconds=round(time.perf_counter() - t0, 2),
        )
        return rec

    reports = []
    for layer in layers:
        try:
            count = manifest["layer_counts"][layer]
            if count != counts[layer]:
                raise detect.DetectionError(f"manifest says {count} experts, wrote {counts[layer]}")
            reports.append(
                detect.analyze_layer(
                    parent_reader,
                    layer,
                    experts,
                    reader,
                    layer,
                    count,
                    SEED,
                    exact_cos=EXACT_COS,
                    near_exact_cos=NEAR_EXACT_COS,
                    parent_desc=parent_desc[layer],
                )
            )
        except Exception as exc:  # noqa: BLE001 - per-layer skip, keep the rest
            print(f"SKIP {model['label']}/{variant} layer {layer}: {type(exc).__name__}: {exc}")
            reports.append(
                {
                    "layer": layer,
                    "status": "skipped",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    rec["layers"] = reports
    rec["summary"], rec["expectation"] = evaluate(variant, reports, truth, experts)
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    return rec


# --------------------------------------------------------------------------- #
# scoring against the expectation registry
# --------------------------------------------------------------------------- #
def _e_minus_two_survivors(
    reports: list[dict],
    truth: dict[int, dict],
    experts: int,
    *,
    missing_key: str,
    new_key: str | None = None,
) -> list[dict]:
    """Reports with E-2 exact survivors, the injected missing parents, and the
    injected new slot (or no new slot when ``new_key`` is None)."""
    return [
        report
        for report in reports
        if report["matched"] == experts - 2
        and report["missing"] == truth[report["layer"]][missing_key]
        and report["new"] == ([truth[report["layer"]][new_key]] if new_key else [])
    ]


def _in_band(alpha: list[float], expected: tuple[float, ...], tol: float) -> bool:
    return len(alpha) == len(expected) and all(
        abs(a - w) <= tol for a, w in zip(alpha, expected, strict=True)
    )


def _gain_ok(entry: dict) -> bool:
    return entry["gain"] is not None and entry["gain"] >= MERGE_GAIN_MIN


def _shadowed(entry: dict) -> bool:
    return entry["cos"] >= EXACT_COS


def evaluate(
    variant: str, reports: list[dict], truth: dict[int, dict], experts: int
) -> tuple[dict, dict]:
    scored = [r for r in reports if r.get("status") != "skipped"]
    total = len(reports)
    complete = len(scored) == total
    expectation = next(item for item in EXPECTATIONS if item["variant"] == variant)
    claim = expectation["claim"]
    # Every suspect is fitted; ``merges`` holds only the merged labels, so the
    # "no fit reaches gain >= 0.30" claims must be read off all classifications.
    fits = [entry for report in scored for entry in report["classifications"]]
    gains = [entry["gain"] for entry in fits if entry["gain"] is not None]
    undefined_gain_fits = len(fits) - len(gains)
    degenerate = [entry for entry in fits if entry["degenerate"]]
    over_gain = [gain for gain in gains if gain >= MERGE_GAIN_MIN]
    summary: dict = {
        "layers": total,
        "scored": len(scored),
        "fits": len(fits),
        "undefined_gain_fits": undefined_gain_fits,
        "degenerate_fits": len(degenerate),
    }

    if variant == "permuted":
        exact = [r for r in scored if r["matched"] == experts]
        perm = [
            r
            for r in exact
            if r["recovered_perm"] == {s: truth[r["layer"]]["perm"][s] for s in range(experts)}
        ]
        held = complete and len(exact) == total and len(perm) == total
        observed = (
            f"{len(exact)}/{total} layers at {experts}/{experts} exact; "
            f"permutation recovered on {len(perm)}/{total}"
        )
        summary.update(exact_layers=len(exact), permutation_layers=len(perm))

    elif variant == "pruned":
        structure = _e_minus_two_survivors(scored, truth, experts, missing_key="dropped")
        # Copies fit themselves with gain=None: those fits are "not applicable",
        # not "gain < 0.30". The clause is scored on the defined gains only, and
        # ``undefined_gain_fits`` records how much of it was N/A.
        held = complete and len(structure) == total and not over_gain
        gain_text = (
            f"{len(fits)} fits, {len(gains)} with defined gain, "
            f"{undefined_gain_fits} not applicable (exact copy), "
            f"defined gains >= {MERGE_GAIN_MIN}: {len(over_gain)}"
        )
        if gains:
            gain_text += f", max defined gain {max(gains):.4f}"
        observed = (
            f"{len(structure)}/{total} layers at {experts - 2}/{experts} exact, "
            f"missing == injected drops, no new expert; {gain_text}"
        )
        summary.update(
            exact_layers=len(structure),
            max_gain=max(gains) if gains else None,
            over_gain_fits=len(over_gain),
        )

    elif variant == "merged_shallow":
        good = []
        shadowed = 0
        for r in scored:
            merged = [c for c in r["classifications"] if c["label"] == "merged"]
            if len(merged) != 1:
                continue
            entry = merged[0]
            # Diagnostic, not a gate: on near-duplicate experts 5% is invisible to
            # cosine matching, and this flag is what the mixture fit rescues.
            if _shadowed(entry):
                shadowed += 1
            if _in_band(entry["alpha"], SHALLOW_ALPHA, ALPHA_TOL) and _gain_ok(entry):
                good.append(r)
        held = complete and len(good) == total
        observed = (
            f"{len(good)}/{total} layers: shallow merge flagged as merged with alpha in "
            f"band and gain >= {MERGE_GAIN_MIN}; cosine matching alone would have called "
            f"it a copy on {shadowed}/{total} layers"
        )
        summary.update(merged_layers=len(good), shadowed_layers=shadowed)

    elif variant in ("merged", "merged_permuted"):
        structure = _e_minus_two_survivors(
            scored, truth, experts, missing_key="merge_parents", new_key="suspect_slot"
        )
        aligned = [r for r in scored if r["new"] == [truth[r["layer"]]["suspect_slot"]]]
        # The fitted entry assigned to the injected merged slot, with the two
        # injected parents as its components.
        alphas = [
            entry["alpha"]
            for r in scored
            for entry in r["classifications"]
            if entry["suspect"] == truth[r["layer"]]["suspect_slot"]
            and entry["components"] == truth[r["layer"]]["merge_parents"]
            and len(entry["alpha"]) == 2
        ]
        in_band = [
            a
            for a in alphas
            if abs(a[0] - MERGE_ALPHA[0]) <= ALPHA_TOL and abs(a[1] - MERGE_ALPHA[1]) <= ALPHA_TOL
        ]
        merge_gains = [
            entry["gain"]
            for r in scored
            for entry in r["classifications"]
            if entry["suspect"] == truth[r["layer"]]["suspect_slot"]
            and entry["components"] == truth[r["layer"]]["merge_parents"]
            and entry["gain"] is not None
        ]
        over_merge = [g for g in merge_gains if g >= MERGE_GAIN_MIN]
        held = (
            complete
            and len(structure) == total
            and len(in_band) == total
            and len(over_merge) == total
        )
        if alphas:
            mean_a = sum(a[0] for a in alphas) / len(alphas)
            mean_b = sum(a[1] for a in alphas) / len(alphas)
            alpha_text = (
                f"alpha mean ({mean_a:.4f}, {mean_b:.4f}) in band on {len(in_band)}/{total}"
            )
        else:
            alpha_text = "no two-parent mixture fit"
        gain_text = (
            f"gain {min(merge_gains):.4f}..{max(merge_gains):.4f}, "
            f">= {MERGE_GAIN_MIN} on {len(over_merge)}/{total}"
            if merge_gains
            else "no gain"
        )
        observed = (
            f"{len(structure)}/{total} layers structure "
            f"({experts - 2} exact, 1 new at the injected slot, 2 missing); "
            f"new == injected slot on {len(aligned)}/{total}; {alpha_text}; {gain_text}"
        )
        summary.update(
            structure_layers=len(structure),
            alignment_layers=len(aligned),
            alpha_layers=len(in_band),
            alpha_mean=[
                sum(a[0] for a in alphas) / len(alphas),
                sum(a[1] for a in alphas) / len(alphas),
            ]
            if alphas
            else None,
            gain_min=min(merge_gains) if merge_gains else None,
            gain_max=max(merge_gains) if merge_gains else None,
        )

    elif variant in MERGED_VARIANT_ALPHA:
        expected, tol = MERGED_VARIANT_ALPHA[variant]
        parents = list(range(len(expected)))

        def kept_exact(report: dict) -> bool:
            layer_truth = truth[report["layer"]]
            return all(
                report["recovered_perm"].get(slot) == parent
                for slot, parent in zip(layer_truth["kept_slots"], layer_truth["kept"], strict=True)
            )

        good: list[dict] = []
        on_components: list[dict] = []
        alpha_errors: list[float] = []
        fit_gains: list[float] = []
        labels: dict[str, int] = {}
        alpha_lengths: dict[int, int] = {}
        top_parents: dict[int, int] = {}
        top_alphas: list[float] = []
        in_band = gain_ok = kept_ok = spread = shadowed = 0
        unique = kept_merged = 0
        for r in scored:
            layer_truth = truth[r["layer"]]
            kept = kept_exact(r)
            kept_ok += int(kept)
            injected_slot = layer_truth["suspect_slot"]
            merged = [c for c in r["classifications"] if c["label"] == "merged"]
            # Exactly one merged classification, and it must be the injected
            # suspect: a second label on a kept slot is the false-positive mode
            # this suite exists to catch.
            unique_here = len(merged) == 1 and merged[0]["suspect"] == injected_slot
            unique += int(unique_here)
            kept_merged += int(any(c["suspect"] in layer_truth["kept_slots"] for c in merged))
            injected = [
                entry for entry in r["classifications"] if entry["suspect"] == injected_slot
            ]
            if len(injected) != 1:
                continue
            entry = injected[0]
            labels[entry["label"]] = labels.get(entry["label"], 0) + 1
            if _shadowed(entry):
                shadowed += 1
            gain = entry["gain"]
            if gain is not None:
                fit_gains.append(gain)
            ok_gain = _gain_ok(entry)
            gain_ok += int(ok_gain)
            alpha_lengths[len(entry["alpha"])] = alpha_lengths.get(len(entry["alpha"]), 0) + 1
            if entry["alpha"]:
                top = max(range(len(entry["alpha"])), key=entry["alpha"].__getitem__)
                top_parent = entry["components"][top]
                top_parents[top_parent] = top_parents.get(top_parent, 0) + 1
                top_alphas.append(entry["alpha"][top])
            if entry["components"] != parents or len(entry["alpha"]) != len(expected):
                spread += 1
                continue
            on_components.append(entry)
            errors = [abs(a - w) for a, w in zip(entry["alpha"], expected, strict=True)]
            alpha_errors.extend(errors)
            band = _in_band(entry["alpha"], expected, tol)
            in_band += int(band)
            if entry["label"] == "merged" and band and ok_gain and kept and unique_here:
                good.append(r)
        held = complete and len(good) == total
        max_error = max(alpha_errors) if alpha_errors else None
        mean_alpha = (
            [
                sum(entry["alpha"][index] for entry in on_components) / len(on_components)
                for index in range(len(expected))
            ]
            if on_components
            else None
        )
        alpha_text = (
            f"alpha mean ({', '.join(f'{a:.4f}' for a in mean_alpha)}) in band on "
            f"{in_band}/{total}, max |dalpha| {max_error:.4f}"
            if mean_alpha is not None
            else "no on-components mixture fit"
        )
        gain_text = (
            f"gain {min(fit_gains):.4f}..{max(fit_gains):.4f}, >= {MERGE_GAIN_MIN} on "
            f"{gain_ok}/{total}"
            if fit_gains
            else "no gain"
        )
        top_text = (
            f"top component alpha {min(top_alphas):.4f}..{max(top_alphas):.4f} on parents "
            f"{sorted(top_parents)}"
            if top_alphas
            else "no fitted components"
        )
        label_text = ", ".join(f"{name} {count}" for name, count in sorted(labels.items()))
        observed = (
            f"{len(good)}/{total} layers pass; injected-suspect labels: "
            f"{label_text or 'none'}; {alpha_text}; {gain_text}; {top_text}; kept parents "
            f"exact on {kept_ok}/{total}; exactly one merged label on {unique}/{total}; "
            f"kept slot labelled merged on {kept_merged}/{total}; off-components spread on "
            f"{spread}/{total}; cos shadowed (>= {EXACT_COS}) on {shadowed}/{total}"
        )
        summary.update(
            merged_layers=len(good),
            unique_merged_layers=unique,
            kept_merged_layers=kept_merged,
            suspect_labels=labels,
            injected_alpha_lengths=alpha_lengths,
            injected_top_parents=top_parents,
            injected_top_alpha=([min(top_alphas), max(top_alphas)] if top_alphas else None),
            alpha_layers=in_band,
            alpha_error_max=max_error,
            alpha_mean=mean_alpha,
            gain_min=min(fit_gains) if fit_gains else None,
            gain_max=max(fit_gains) if fit_gains else None,
            gain_over_layers=gain_ok,
            kept_layers=kept_ok,
            components_spread_layers=spread,
            shadowed_layers=shadowed,
        )

    elif variant == "control":
        zero = [r for r in scored if r["matched"] == 0]
        max_gain = max(gains) if gains else None
        held = complete and len(zero) == total and not over_gain
        max_cos = max(
            (r["max_assigned_cos"] for r in scored if r["max_assigned_cos"] is not None),
            default=None,
        )
        gain_text = (
            f"{len(fits)} fits ({len(gains)} with gain, {len(degenerate)} degenerate), "
            f"max gain {max_gain:.4f}"
            if max_gain is not None
            else f"{len(fits)} fits, no defined gain"
        )
        cos_text = f"max assigned cos {max_cos:.4f}; " if max_cos is not None else ""
        observed = (
            f"max exact matches {max((r['matched'] for r in scored), default=-1)}/{experts}; "
            f"{cos_text}{gain_text}"
        )
        summary.update(
            zero_match_layers=len(zero),
            max_gain=max_gain,
            over_gain_fits=len(over_gain),
            max_assigned_cos=max_cos,
            max_pair_cos=max((r["max_pair_cos"] for r in scored), default=None),
        )

    else:
        raise ValueError(f"unknown variant {variant!r}")

    expectation_record = {
        "claim": claim,
        "origin": expectation["origin"],
        "held": bool(held),
        "observed": observed,
    }
    if "origin_ref" in expectation:
        expectation_record["origin_ref"] = expectation["origin_ref"]
    return summary, expectation_record


# --------------------------------------------------------------------------- #
# model driver
# --------------------------------------------------------------------------- #
def run_model(model: dict) -> dict:
    t0 = time.perf_counter()
    record: dict = {
        "label": model["label"],
        "repo": model["repo"],
        "revision": model["revision"],
        "arch": model["arch"],
        "num_experts": model["num_experts"],
        "status": "ok",
        "variants": {},
    }
    try:
        snapshot = synth.cached_snapshot(model["repo"], model["revision"])
        shards = synth.shard_index(snapshot)
        available = synth.discover_layers(shards, snapshot, model["arch"], model["num_experts"])
        layers = model["layers"] or available
        unusable = [layer for layer in layers if layer not in available]
        if unusable:
            raise RuntimeError(f"layers {unusable} lack a complete expert set")
        if not layers:
            raise RuntimeError("no MoE layer found")
        spec = detect.TensorSpec(**ARCHES[model["arch"]])
        parent_reader = detect.CheckpointReader(snapshot, spec, shards)
        parent_desc = {
            layer: detect.descriptors(parent_reader, layer, model["num_experts"], SEED)
            for layer in layers
        }
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP model {model['label']}: {type(exc).__name__}: {exc}")
        record.update(
            status="skipped",
            reason=f"{type(exc).__name__}: {exc}",
            seconds=round(time.perf_counter() - t0, 2),
        )
        return record

    record["layers"] = layers
    cache: dict[int, tuple] = {}

    def parent_layer(layer: int):
        if layer not in cache:
            cache[layer] = synth.load_layer(
                snapshot, shards, model["arch"], layer, model["num_experts"]
            )
        return cache[layer]

    for variant in VARIANTS:
        record["variants"][variant] = run_variant(
            model, variant, layers, parent_layer, parent_reader, parent_desc
        )

    record["seconds"] = round(time.perf_counter() - t0, 2)
    return record


# --------------------------------------------------------------------------- #
# structural-skip probe (extension, not in the spec)
# --------------------------------------------------------------------------- #
def robustness_probe() -> dict:
    """Malformed inputs must become DetectionError -> SKIP lines, never a crash."""
    result: dict = {"cases": [], "all_ok": True}
    try:
        root = SCRATCH / "_malformed"
        root.mkdir(parents=True, exist_ok=True)
        spec = detect.TensorSpec(**ARCHES["mixtral"])
        layer, experts, side = 0, 2, 48
        generator = torch.Generator().manual_seed(SEED)
        tensors = {
            spec.expert.format(layer=layer, expert=e, role=r): torch.randn(
                side, side, generator=generator
            )
            for e in range(experts)
            for r in spec.roles
        }
        tensors[spec.router.format(layer=layer)] = torch.randn(experts, side, generator=generator)
        manifest = {
            "arch": "mixtral",
            "seed": SEED,
            "roles": list(spec.roles),
            "layers": {str(layer): {"num_experts": experts}},
            "tensor_file": "model.safetensors",
        }

        def write(case: str, payload: dict, text: dict | None = None) -> Path:
            case_dir = root / case
            case_dir.mkdir(parents=True, exist_ok=True)
            stale = case_dir / "model.safetensors.index.json"
            if stale.exists():
                stale.unlink()
            synth.save_tensors(case_dir / "model.safetensors", payload)
            (case_dir / "manifest.json").write_text(
                json.dumps(text if text is not None else manifest, indent=2)
            )
            return case_dir

        def write_index(case_dir: Path, document: object) -> None:
            (case_dir / "model.safetensors.index.json").write_text(json.dumps(document))

        def probe(case: str, call) -> None:
            try:
                call()
            except detect.DetectionError as exc:
                result["cases"].append({"case": case, "skip": str(exc), "ok": True})
            except Exception as exc:  # noqa: BLE001 - anything else is a failed robustness claim
                result["cases"].append(
                    {"case": case, "skip": f"{type(exc).__name__}: {exc}", "ok": False}
                )
            else:
                result["cases"].append({"case": case, "skip": "no exception raised", "ok": False})

        def analyze(case_dir: Path, suspect_experts: int = experts) -> dict:
            return detect.analyze_layer(
                detect.CheckpointReader(case_dir, spec),
                layer,
                experts,
                detect.CheckpointReader(case_dir, spec),
                layer,
                suspect_experts,
                SEED,
            )

        full = write("missing_tensor", tensors)
        broken = dict(tensors)
        del broken[spec.expert.format(layer=layer, expert=0, role=spec.roles[0])]
        synth.save_tensors(full / "model.safetensors", broken)
        probe("missing_tensor", lambda: analyze(full))

        truncated = write("truncated_file", tensors)
        (truncated / "model.safetensors").write_bytes(b"not-a-safetensors-header")
        probe("truncated_file", lambda: analyze(truncated))

        bad_manifest = write("bad_manifest", tensors, {"layers": [], "arch": "mixtral"})
        probe("bad_manifest", lambda: detect.read_manifest(bad_manifest))

        # Untrusted shard index: a redirect must never open a file outside the
        # checkpoint, and the document must be an object.
        victim = root / "victim.safetensors"
        synth.save_tensors(victim, tensors)
        first = spec.expert.format(layer=layer, expert=0, role=spec.roles[0])

        traversal = write("traversal_index", tensors)
        write_index(traversal, {"weight_map": {first: "../victim.safetensors"}})
        probe("traversal_index", lambda: detect.CheckpointReader(traversal, spec))

        absolute = write("absolute_index", tensors)
        write_index(absolute, {"weight_map": {first: str(victim)}})
        probe("absolute_index", lambda: detect.CheckpointReader(absolute, spec))

        nonobject = write("nonobject_index", tensors)
        (nonobject / "model.safetensors.index.json").write_text("[]")
        probe("nonobject_index", lambda: detect.CheckpointReader(nonobject, spec))

        # A NaN outside the 32x32 descriptor window must still abstain.
        role = spec.roles[0]
        row, col = detect.block_origin(SEED, role, layer, (side, side))
        outside = next(
            (r, c)
            for r in range(side)
            for c in range(side)
            if not (row <= r < row + detect.BLOCK and col <= c < col + detect.BLOCK)
        )
        poisoned = write("nan_outside_window", tensors)
        payload = dict(tensors)
        name = spec.expert.format(layer=layer, expert=0, role=role)
        poisoned_tensor = payload[name].clone()
        poisoned_tensor[outside] = float("nan")
        payload[name] = poisoned_tensor
        synth.save_tensors(poisoned / "model.safetensors", payload)

        def nan_case() -> None:
            reader = detect.CheckpointReader(poisoned, spec)
            vector = detect.descriptor(reader, layer, 0, SEED)
            assert np.isfinite(vector).all(), "NaN reached the sampled descriptor window"
            detect.analyze_layer(reader, layer, experts, reader, layer, experts, SEED)

        probe("nan_outside_window", nan_case)

        # Manifest count lie: four expert tensors on disk, three declared.
        hidden = {
            spec.expert.format(layer=layer, expert=e, role=r): torch.randn(
                side, side, generator=generator
            )
            for e in range(experts + 2)
            for r in spec.roles
        }
        hidden[spec.router.format(layer=layer)] = torch.randn(experts, side, generator=generator)
        understated = write(
            "understated_count",
            hidden,
            {**manifest, "layers": {str(layer): {"num_experts": experts}}},
        )
        probe("understated_count", lambda: analyze(understated, experts))

        # Manifest type strictness.
        float_count = write(
            "float_num_experts",
            tensors,
            {**manifest, "layers": {str(layer): {"num_experts": 2.9}}},
        )
        probe("float_num_experts", lambda: detect.read_manifest(float_count))

        bool_count = write(
            "bool_num_experts",
            tensors,
            {**manifest, "layers": {str(layer): {"num_experts": True}}},
        )
        probe("bool_num_experts", lambda: detect.read_manifest(bool_count))

        aliased = write(
            "aliased_layer_key",
            tensors,
            {
                **manifest,
                "layers": {
                    str(layer): {"num_experts": experts},
                    f"0{layer}": {"num_experts": experts},
                },
            },
        )
        probe("aliased_layer_key", lambda: detect.read_manifest(aliased))

        deep = root / "deep_manifest"
        deep.mkdir(parents=True, exist_ok=True)
        (deep / "manifest.json").write_text("[" * 20_000 + "]" * 20_000)
        probe("deep_manifest", lambda: detect.read_manifest(deep))

        oversized = root / "oversized_manifest"
        oversized.mkdir(parents=True, exist_ok=True)
        (oversized / "manifest.json").write_text(" " * (detect.MAX_METADATA_BYTES + 1))
        probe("oversized_manifest", lambda: detect.read_manifest(oversized))
    except Exception as exc:  # noqa: BLE001 - the probe itself must never abort the run
        result["all_ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["all_ok"] = result["all_ok"] and all(case["ok"] for case in result["cases"])
    return result


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _pair_list(values: list[int]) -> str:
    return ", ".join(str(v) for v in values) if values else "-"


def print_model_table(record: dict) -> None:
    print(
        f"\n== {record['label']} — {record['repo']}@{record['revision'][:12]} "
        f"({record['arch']}, {len(record.get('layers', []))} layers, "
        f"{record['num_experts']} experts, {record.get('seconds', 'n/a')}s)"
    )
    if record["status"] != "ok":
        print(f"   skipped: {record.get('reason')}")
        return
    header = (
        f"{'variant':<15} {'layer':>5} {'matched':>8} {'cand':>5} {'missing':<12} {'new':<12} "
        f"{'maxcos':>9} {'fits':>4} {'alpha':<19} {'gain':>8}"
    )
    print(header)
    print("-" * len(header))
    for variant, vrec in record["variants"].items():
        if vrec["status"] != "ok":
            print(f"{variant:<15} {'-':>5} SKIP {vrec.get('reason')}")
            continue
        for report in vrec["layers"]:
            if report.get("status") == "skipped":
                print(f"{variant:<15} {report['layer']:>5} SKIP {report['reason']}")
                continue
            merges = [m for m in report["merges"] if m["gain"] is not None]
            best = max(merges, key=lambda m: m["gain"]) if merges else None
            alpha = ", ".join(f"{a:.4f}" for a in best["alpha"]) if best else "-"
            gain = f"{best['gain']:.4f}" if best else "-"
            max_cos = report["max_pair_cos"]
            print(
                f"{variant:<15} {report['layer']:>5} "
                f"{report['matched']:>3}/{report['parent_experts']:<4} "
                f"{len(report['candidate_pairs']):>5} "
                f"{_pair_list(report['missing']):<12} {_pair_list(report['new']):<12} "
                f"{max_cos:>9.6f} {len(report['classifications']):>4} "
                f"{alpha:<19} {gain:>8}"
            )


def print_expectations(models: list[dict]) -> None:
    print("\n== expectations (origin: post_hoc claims are restated after a run)")
    header = f"{'model':<14} {'variant':<15} {'origin':<14} {'verdict':<7} observed"
    print(header)
    print("-" * len(header))
    for record in models:
        if record["status"] != "ok":
            for variant in VARIANTS:
                print(
                    f"{record['label']:<14} {variant:<15} {ORIGINS[variant]:<14} {'SKIP':<7} "
                    f"model unavailable: {record.get('reason')}"
                )
            continue
        for variant, vrec in record["variants"].items():
            if vrec["status"] != "ok":
                print(
                    f"{record['label']:<14} {variant:<15} {ORIGINS[variant]:<14} {'SKIP':<7} "
                    f"{vrec.get('reason')}"
                )
                continue
            verdict = "HELD" if vrec["expectation"]["held"] else "MISSED"
            print(
                f"{record['label']:<14} {variant:<15} {ORIGINS[variant]:<14} {verdict:<7} "
                f"{vrec['expectation']['observed']}"
            )


def main() -> int:
    assert detect.BLOCK == BLOCK, "detector block size drifted from config"
    assert detect.EXACT_COS == EXACT_COS, "detector exact-cosine floor drifted from config"
    assert detect.NEAR_EXACT_COS == NEAR_EXACT_COS, "near-exact band drifted"
    assert detect.MERGE_GAIN_MIN == MERGE_GAIN_MIN, "merge-gain floor drifted"
    assert detect.MIN_COMPONENT == MIN_COMPONENT, "minimum mixture component drifted"
    if tuple(item["variant"] for item in EXPECTATIONS) != VARIANTS:
        raise SystemExit(
            "config.VARIANTS and config.EXPECTATIONS disagree: "
            f"{VARIANTS} vs {tuple(item['variant'] for item in EXPECTATIONS)}"
        )

    started = time.perf_counter()
    print(
        f"Experiment A — MoE expert lineage | seed {SEED} | block {BLOCK} | "
        f"exact cos >= {EXACT_COS} | merge gain >= {MERGE_GAIN_MIN} | "
        f"alpha bands {ALPHA_TOL} (TIES {TIES_ALPHA_TOL})"
    )
    print(f"scratch {SCRATCH}")

    models = [run_model(PRIMARY)]
    primary_seconds = models[0].get("seconds", float("inf"))
    print(f"\nprimary runtime {primary_seconds}s (scale-repeat budget {SECONDARY_BUDGET_S}s)")

    if models[0]["status"] == "ok" and primary_seconds < SECONDARY_BUDGET_S:
        models.append(run_model(SECONDARY))
    else:
        reason = (
            f"primary runtime {primary_seconds}s >= {SECONDARY_BUDGET_S}s budget"
            if models[0]["status"] == "ok"
            else "primary model unavailable"
        )
        print(f"SKIP model {SECONDARY['label']}: {reason}")
        models.append(
            {
                "label": SECONDARY["label"],
                "repo": SECONDARY["repo"],
                "revision": SECONDARY["revision"],
                "arch": SECONDARY["arch"],
                "num_experts": SECONDARY["num_experts"],
                "status": "skipped",
                "reason": reason,
                "variants": {},
            }
        )

    models[0]["robustness"] = robustness_probe()

    results = {
        "experiment": "A — MoE expert lineage",
        "command": COMMAND,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": SEED,
        "descriptor_block": BLOCK,
        "exact_cos": EXACT_COS,
        "near_exact_cos": NEAR_EXACT_COS,
        "merge_gain_min": MERGE_GAIN_MIN,
        "merge_alpha": list(MERGE_ALPHA),
        "shallow_alpha": list(SHALLOW_ALPHA),
        "skewed_alpha": list(SKEWED_ALPHA),
        "three_alpha": list(THREE_ALPHA),
        "saliency_alpha": list(SALIENCY_ALPHA),
        "noisy_sigma": NOISY_SIGMA,
        "ties_trim": TIES_TRIM,
        "ties_alpha": list(TIES_ALPHA),
        "ties_alpha_tol": TIES_ALPHA_TOL,
        "alpha_tol": ALPHA_TOL,
        "min_component": MIN_COMPONENT,
        "variants": list(VARIANTS),
        "expectations": list(EXPECTATIONS),
        "secondary_budget_s": SECONDARY_BUDGET_S,
        "models": models,
        "total_seconds": round(time.perf_counter() - started, 2),
    }
    # Write before printing: a reporting error must not lose the run.
    RESULTS.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")

    for record in models:
        print_model_table(record)
    print_expectations(models)

    robustness = models[0]["robustness"]
    print("\n== structural-skip probe")
    for case in robustness["cases"]:
        print(f"{'OK' if case['ok'] else 'FAIL':<5} {case['case']:<20} {case['skip']}")
    if robustness.get("error"):
        print(f"probe error: {robustness['error']}")

    print(f"\nwrote {RESULTS}")
    print(f"total {results['total_seconds']}s")
    return 0 if models[0]["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
