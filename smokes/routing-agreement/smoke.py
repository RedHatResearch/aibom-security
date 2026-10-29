"""Routing agreement smoke — smokes/routing-agreement/ · Experiment B.

Question: does preserving router precision while quantizing experts preserve
routing (the llama.cpp GGUF property), and how much does quantizing the router
itself hurt?

Method (frozen in ``config.py`` before the first run):
  * 24 prompts (6 each code/math/factual/narrative), 96 tokens each;
  * per-layer top-k expert sets captured from the ``router_logits`` tuple;
  * variants: base, experts_q4, all_q4, router_shuffled;
  * metric per layer: exact top-k set agreement and mean Jaccard over all
    (prompt, position) rows, overall = mean across layers;
  * analytic chance level for two independent uniform k-subsets of E experts.

``router_shuffled`` permutes router rows: that layer's logits permute, so the
selected ids become ``perm^-1(S)`` for base set ``S``, but the experts stay in
place, so the layer computes a different mixture and the control behaves like
random routing. Layer 0 is the exception — its input (the embedding) is
untouched — so its agreement has the closed form "fraction of base layer-0 sets
invariant under perm"; that exact prediction is reported next to the measured
value.

Run: ``uv run --group smokes python smokes/routing-agreement/smoke.py``
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import sys
import time
from datetime import UTC, datetime
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SMOKE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SMOKE_DIR))

import config as C  # noqa: E402
import quant as Q  # noqa: E402

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def peak_rss_gb() -> float:
    """Process peak RSS in GB (``ru_maxrss`` is bytes on macOS, KiB on Linux)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e9 if platform.system() == "Darwin" else raw * 1024 / 1e9


def layer_index(name: str) -> int:
    """Decoder-layer index parsed from a parameter name (``-1`` if absent)."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else -1


def router_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """Router gate matrices in decoder-layer order (never lexicographic)."""
    found = [(n, p) for n, p in model.named_parameters() if Q.is_router_param(n)]
    return sorted(found, key=lambda kv: layer_index(kv[0]))


def expert_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """All expert weight tensors (fused 3-D or per-expert 2-D), sorted by layer."""
    found = [(n, p) for n, p in model.named_parameters() if Q.is_expert_param(n)]
    return sorted(found, key=lambda kv: (layer_index(kv[0]), kv[0]))


def chance_levels(k: int, n_experts: int) -> dict:
    """Analytic chance for two independent uniform k-subsets of E experts.

    Exact-set agreement is ``1 / C(E, k)``. Mean Jaccard needs the distribution
    of the intersection size ``X ~ Hypergeometric(E, k, k)``:
    ``E[J] = sum_x P(X=x) * x / (2k - x)``.
    """
    total = comb(n_experts, k)
    exact = Fraction(1, total)
    jaccard = Fraction(0)
    for x in range(max(0, 2 * k - n_experts), k + 1):
        p_x = Fraction(comb(k, x) * comb(n_experts - k, k - x), total)
        jaccard += p_x * Fraction(x, 2 * k - x)
    return {
        "set_agreement": float(exact),
        "mean_jaccard": float(jaccard),
        "set_agreement_fraction": f"1/C({n_experts},{k}) = {exact}",
        "mean_jaccard_fraction": str(jaccard),
    }


# ---------------------------------------------------------------------------
# Probe set
# ---------------------------------------------------------------------------


def encode_prompt(tokenizer, text: str, seq_len: int) -> tuple[list[int], int, bool, bool]:
    """Tokenize, truncate or pad to ``seq_len``; return ids, real length, trunc, pad flags."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    n_real = len(ids)
    truncated = n_real > seq_len
    padded = n_real < seq_len
    ids = ids[:seq_len]
    if padded:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        ids = ids + [int(pad_id)] * (seq_len - n_real)
    return ids, min(n_real, seq_len), truncated, padded


def build_probe(tokenizer, seq_len: int) -> dict:
    """Fixed-order probe tensor plus padding mask and per-prompt bookkeeping."""
    rows, masks, records = [], [], []
    for prompt in C.PROMPTS:
        ids, n_real, truncated, padded = encode_prompt(tokenizer, prompt["text"], seq_len)
        rows.append(ids)
        masks.append([1] * n_real + [0] * (seq_len - n_real))
        records.append(
            {
                "domain": prompt["domain"],
                "text": prompt["text"],
                "tokens": len(tokenizer.encode(prompt["text"], add_special_tokens=False)),
                "used": n_real,
                "truncated": truncated,
                "padded": padded,
            }
        )
    return {
        "input_ids": torch.tensor(rows, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "position_mask": torch.tensor(masks, dtype=torch.bool).reshape(-1),
        "domain_index": torch.tensor(
            [C.DOMAINS.index(p["domain"]) for p in C.PROMPTS], dtype=torch.long
        ).repeat_interleave(seq_len),
        "records": records,
        "n_truncated": sum(r["truncated"] for r in records),
        "n_padded": sum(r["padded"] for r in records),
    }


# ---------------------------------------------------------------------------
# Routing capture
# ---------------------------------------------------------------------------


def capture_routing(model, probe: dict, top_k: int) -> tuple[np.ndarray, dict]:
    """Top-k expert ids per (layer, row) from the router-logits tuple.

    Returns an int16 array ``[layers, rows, k]`` plus bookkeeping about the
    tuple shape. Adapts (and reports) if the tuple is not exactly one entry per
    decoder layer.
    """
    n_layers = int(model.config.num_hidden_layers)
    with torch.inference_mode():
        out = model(
            input_ids=probe["input_ids"],
            attention_mask=probe["attention_mask"],
            output_router_logits=True,
            use_cache=False,
            logits_to_keep=1,
        )
    router = out.router_logits
    if router is None:
        raise RuntimeError("forward returned router_logits=None despite output_router_logits=True")
    entries = list(router)
    non_null = [i for i, x in enumerate(entries) if x is not None]
    used = non_null[:n_layers]
    info = {
        "tuple_len": len(entries),
        "non_null": len(non_null),
        "n_layers_config": n_layers,
        "used_layers": len(used),
        "adapted": len(entries) != n_layers or len(non_null) != n_layers,
    }
    if not used:
        raise RuntimeError("router_logits tuple has no non-null entries")

    per_layer = []
    n_experts = None
    for idx in used:
        logits = entries[idx].float()
        if logits.ndim != 2:
            raise RuntimeError(f"router logits for layer {idx} have shape {tuple(logits.shape)}")
        n_experts = int(logits.shape[-1]) if n_experts is None else n_experts
        if int(logits.shape[-1]) != n_experts:
            raise RuntimeError("inconsistent expert count across router logits")
        kk = min(top_k, n_experts)
        per_layer.append(torch.topk(logits, kk, dim=-1).indices.to(torch.int16))
    info["n_experts"] = n_experts
    info["top_k"] = int(min(top_k, n_experts))
    info["rows"] = int(entries[used[0]].shape[0])
    stacked = torch.stack(per_layer, dim=0).cpu().numpy()
    return stacked, info


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _jaccard_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Jaccard per row for two ``[M, k]`` index sets (unique ids, same k)."""
    inter = (a[:, :, None] == b[:, None, :]).any(axis=-1).sum(axis=1)
    union = a.shape[1] + b.shape[1] - inter
    return inter / union


def _set_equal_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact set equality per row (order-insensitive)."""
    return (np.sort(a, axis=-1) == np.sort(b, axis=-1)).all(axis=-1)


def compare(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> dict:
    """Per-layer set agreement + mean Jaccard over masked rows, and overall means."""
    n_layers = reference.shape[0]
    per_layer_set, per_layer_jac = [], []
    for layer in range(n_layers):
        ref = reference[layer][mask]
        cand = candidate[layer][mask]
        per_layer_set.append(float(_set_equal_rows(ref, cand).mean()))
        per_layer_jac.append(float(_jaccard_rows(ref, cand).mean()))
    return {
        "per_layer_set_agreement": per_layer_set,
        "per_layer_mean_jaccard": per_layer_jac,
        "overall_set_agreement": float(np.mean(per_layer_set)),
        "overall_mean_jaccard": float(np.mean(per_layer_jac)),
    }


def by_domain(reference: np.ndarray, candidate: np.ndarray, probe: dict) -> dict:
    """Overall set agreement per prompt domain (mean of per-layer means)."""
    out = {}
    for domain in C.DOMAINS:
        rows = (probe["domain_index"] == C.DOMAINS.index(domain)).numpy() & probe[
            "position_mask"
        ].numpy()
        per_layer = [
            float(_set_equal_rows(reference[layer][rows], candidate[layer][rows]).mean())
            for layer in range(reference.shape[0])
        ]
        out[domain] = float(np.mean(per_layer))
    return out


# ---------------------------------------------------------------------------
# Variant application (in place on the loaded weights)
# ---------------------------------------------------------------------------


def apply_experts_q4(model) -> int:
    """Quantize-dequantize every expert tensor in place; returns tensor count."""
    params = expert_params(model)
    if not params:
        raise Q.CheckpointTensorError("no expert tensors matched — checkpoint layout unexpected")
    for _, param in params:
        Q.quantize_dequantize_(param)
    return len(params)


def qdq_router(model) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    """Quantize-dequantize router gates, returning originals for restoration."""
    params = router_params(model)
    if not params:
        raise Q.CheckpointTensorError("no router gate tensors matched")
    saved = []
    for _, param in params:
        saved.append((param, param.detach().clone()))
        Q.quantize_dequantize_(param)
    return saved


def shuffle_routers(
    model, seed: int
) -> list[tuple[str, torch.nn.Parameter, torch.Tensor, torch.Tensor]]:
    """Permute router rows per layer with a seeded generator; return names, originals, perms."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for name, param in router_params(model):
        perm = torch.randperm(param.shape[0], generator=gen)
        saved = Q.apply_router_permutation_(param, perm)
        out.append((name, param, saved, perm))
    return out


def restore(saved: list[tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
    """Copy saved tensors back into their parameters."""
    for param, value in saved:
        with torch.no_grad():
            param.copy_(value)


# ---------------------------------------------------------------------------
# One model
# ---------------------------------------------------------------------------


def run_model(spec: dict, tokenizer, *, check_budgets: bool) -> dict:
    """Load one model, run every variant, return its results entry."""
    entry: dict = {
        "label": spec["label"],
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "dtype": spec["dtype"],
        "status": "ok",
    }
    t_load = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            spec["repo_id"],
            revision=spec["revision"],
            dtype=getattr(torch, spec["dtype"]),
            output_router_logits=True,
            attn_implementation="eager",
        )
    except Exception as exc:  # untrusted checkpoint: fail closed with a reported skip
        entry.update(status="skipped", skip_reason=f"load failed: {type(exc).__name__}: {exc}")
        return entry
    model.eval()
    load_s = time.perf_counter() - t_load
    entry["load_s"] = round(load_s, 2)
    entry["peak_rss_gb_after_load"] = round(peak_rss_gb(), 2)

    top_k_model = int(getattr(model.config, "num_experts_per_tok", spec["top_k"]))
    n_experts_model = int(
        getattr(model.config, "num_local_experts", None)
        or getattr(model.config, "num_experts", spec["n_experts"])
    )
    entry["n_layers"] = int(model.config.num_hidden_layers)
    entry["n_experts"] = n_experts_model
    entry["top_k"] = top_k_model
    entry["router_params"] = len(router_params(model))
    entry["expert_params"] = len(expert_params(model))

    probe = build_probe(tokenizer, C.SEQ_LEN)
    n_rows = int(probe["position_mask"].sum())
    entry["probe"] = {
        "prompts": len(C.PROMPTS),
        "seq_len": C.SEQ_LEN,
        "rows_scored": n_rows,
        "truncated_prompts": probe["n_truncated"],
        "padded_prompts": probe["n_padded"],
        "records": probe["records"],
    }

    t_cap = time.perf_counter()
    try:
        base, rl_info = capture_routing(model, probe, top_k_model)
    except Exception as exc:
        entry.update(
            status="skipped", skip_reason=f"routing capture failed: {type(exc).__name__}: {exc}"
        )
        return entry
    base_s = time.perf_counter() - t_cap
    entry["router_logits"] = rl_info
    entry["forward_s"] = round(base_s, 2)
    entry["peak_rss_gb_after_base"] = round(peak_rss_gb(), 2)

    k = rl_info["top_k"]
    n_experts = rl_info["n_experts"]
    entry["chance"] = chance_levels(k, n_experts)

    projection = load_s + base_s * len(C.EXECUTION_ORDER) * 1.5
    entry["projected_total_s"] = round(projection, 1)
    if check_budgets:
        if entry["peak_rss_gb_after_base"] > C.OLMOE_RAM_BUDGET_GB:
            entry.update(
                status="skipped",
                skip_reason=(
                    f"peak RSS {entry['peak_rss_gb_after_base']:.1f} GB exceeds "
                    f"{C.OLMOE_RAM_BUDGET_GB:.0f} GB budget"
                ),
            )
            return entry
        if projection > C.OLMOE_TIME_BUDGET_S:
            entry.update(
                status="skipped",
                skip_reason=(
                    f"projected {projection:.0f} s exceeds {C.OLMOE_TIME_BUDGET_S:.0f} s budget"
                ),
            )
            return entry

    mask = probe["position_mask"].numpy()
    reference = base
    variants: dict[str, dict] = {
        "base": {
            **compare(reference, reference, mask),
            "by_domain": by_domain(reference, reference, probe),
            "note": C.VARIANT_NOTES["base"],
        }
    }

    # Execution order: the router shuffle runs first, on pristine weights (its
    # router rows are restored afterwards), because it is a control on the base
    # model and must not be confounded with expert Q4. Then experts_q4 QDQs the
    # expert tensors and leaves them QDQ'd, and all_q4 only adds the router, so
    # all_q4 == base experts QDQ'd once + router QDQ'd once and no full-size
    # backup of the expert tensors is ever needed.
    shuffled: list[tuple[str, torch.nn.Parameter, torch.Tensor, torch.Tensor]] = []
    try:
        shuffled = shuffle_routers(model, C.SEED)
        shuffle_info = [
            {
                "layer": layer_index(name),
                "perm": perm.tolist(),
                "fixed_points": Q.permutation_fixed_points(perm),
            }
            for name, _param, _saved, perm in shuffled
        ]
        cand, _ = capture_routing(model, probe, top_k_model)
        metrics = compare(reference, cand, mask)
        # Layer 0 is the only layer whose input is untouched by the shuffle
        # (it is the embedding), so its agreement has a closed form: the fraction
        # of base layer-0 routing sets invariant under that layer's permutation.
        layer0_pred = Q.permutation_invariance(
            torch.from_numpy(reference[0][mask]),
            torch.tensor(shuffle_info[0]["perm"], dtype=torch.long),
        )
        variants["router_shuffled"] = {
            **metrics,
            "by_domain": by_domain(reference, cand, probe),
            "note": C.VARIANT_NOTES["router_shuffled"],
            "layer0_relabel_prediction": layer0_pred,
            "layer0_measured": metrics["per_layer_set_agreement"][0],
            "layer0_matches_prediction": bool(
                abs(layer0_pred - metrics["per_layer_set_agreement"][0]) < 1e-9
            ),
            "per_layer_fixed_points": [r["fixed_points"] for r in shuffle_info],
            "per_layer_permutations": [r["perm"] for r in shuffle_info],
        }
    except Exception as exc:
        entry["shuffle_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        restore([(p, saved) for _name, p, saved, _perm in shuffled])

    saved_router: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    try:
        t0 = time.perf_counter()
        n_expert_tensors = apply_experts_q4(model)
        qdq_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        cand, _ = capture_routing(model, probe, top_k_model)
        entry["variants_qdq_note"] = {
            "expert_tensors_quantized": n_expert_tensors,
            "experts_qdq_s": round(qdq_s, 2),
            "experts_q4_forward_s": round(time.perf_counter() - t0, 2),
        }
        variants["experts_q4"] = {
            **compare(reference, cand, mask),
            "by_domain": by_domain(reference, cand, probe),
            "note": C.VARIANT_NOTES["experts_q4"],
        }

        t0 = time.perf_counter()
        saved_router = qdq_router(model)
        router_qdq_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        cand, _ = capture_routing(model, probe, top_k_model)
        entry["variants_qdq_note"].update(
            {
                "router_tensors_quantized": len(saved_router),
                "router_qdq_s": round(router_qdq_s, 2),
                "all_q4_forward_s": round(time.perf_counter() - t0, 2),
            }
        )
        variants["all_q4"] = {
            **compare(reference, cand, mask),
            "by_domain": by_domain(reference, cand, probe),
            "note": C.VARIANT_NOTES["all_q4"],
        }
    except Exception as exc:
        entry["variant_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if saved_router:
            restore(saved_router)

    entry["variants"] = variants
    entry["router_shuffled_seed"] = C.SEED
    entry["elapsed_s"] = round(time.perf_counter() - t_load, 1)
    entry["peak_rss_gb"] = round(peak_rss_gb(), 2)
    del model
    return entry


# ---------------------------------------------------------------------------
# Expectations
# ---------------------------------------------------------------------------


def evaluate_expectations(model_entries: list[dict]) -> list[dict]:
    """Mark every pre-registered expectation held/missed/skipped."""
    by_label = {e["label"]: e for e in model_entries}
    out = []
    for exp in C.EXPECTATIONS:
        entry = by_label.get(exp["model"])
        record = {
            "id": exp["id"],
            "model": exp["model"],
            "variant": exp["variant"],
            "expectation": exp["text"],
            "kind": exp["kind"],
            "param": exp["param"],
        }
        if (
            entry is None
            or entry["status"] != "ok"
            or exp["variant"] not in entry.get("variants", {})
        ):
            record["verdict"] = "skipped"
            record["observed"] = None
            out.append(record)
            continue
        observed = entry["variants"][exp["variant"]]["overall_set_agreement"]
        record["observed"] = observed
        if exp["kind"] == "min":
            record["held"] = observed >= exp["param"]
        elif exp["kind"] == "lt_variant":
            other = entry["variants"][exp["param"]]["overall_set_agreement"]
            record["compared_to"] = {"variant": exp["param"], "overall_set_agreement": other}
            record["held"] = observed < other
        elif exp["kind"] == "near_chance":
            chance = entry["chance"]["set_agreement"]
            record["chance"] = chance
            record["delta_from_chance"] = observed - chance
            record["held"] = abs(observed - chance) <= exp["param"]
        else:  # pragma: no cover - config typo guard
            raise ValueError(f"unknown expectation kind {exp['kind']}")
        record["verdict"] = "held" if record["held"] else "missed"
        out.append(record)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_VARIANTS_SHOWN = ("experts_q4", "all_q4", "router_shuffled")


def print_model_table(entry: dict) -> None:
    """Per-layer + overall agreement table for one model."""
    v = entry["variants"]
    print(
        f"\nmodel {entry['repo_id']} @ {entry['revision'][:8]}  "
        f"dtype={entry['dtype']}  L={entry['n_layers']}  E={entry['n_experts']}  "
        f"k={entry['top_k']}  router_logits={entry['router_logits']['tuple_len']}"
        f"{' (ADAPTED)' if entry['router_logits']['adapted'] else ' (ok)'}"
    )
    p = entry["probe"]
    print(
        f"probe {p['prompts']} prompts x {p['seq_len']} tokens = {p['rows_scored']} rows"
        f"  truncated={p['truncated_prompts']}  padded={p['padded_prompts']}"
        f"  domains={','.join(C.DOMAINS)}"
    )
    print(
        f"chance  set={entry['chance']['set_agreement']:.6f}  "
        f"jaccard={entry['chance']['mean_jaccard']:.6f}   "
        f"({entry['chance']['set_agreement_fraction']}; "
        f"J={entry['chance']['mean_jaccard_fraction']})"
    )
    header = f"{'layer':>5} | " + " | ".join(f"{name:>15} {'J':>12}" for name in _VARIANTS_SHOWN)
    print(header)
    print("-" * len(header))
    for layer in range(entry["n_layers"]):
        cells = []
        for name in _VARIANTS_SHOWN:
            data = v.get(name)
            if data is None:
                cells.append(f"{'-':>15} {'-':>12}")
            else:
                cells.append(
                    f"{data['per_layer_set_agreement'][layer]:>15.6f} "
                    f"{data['per_layer_mean_jaccard'][layer]:>12.6f}"
                )
        print(f"{layer:>5} | " + " | ".join(cells))
    cells = []
    for name in _VARIANTS_SHOWN:
        data = v.get(name)
        if data is None:
            cells.append(f"{'-':>15} {'-':>12}")
        else:
            cells.append(
                f"{data['overall_set_agreement']:>15.6f} {data['overall_mean_jaccard']:>12.6f}"
            )
    print(f"{'mean':>5} | " + " | ".join(cells))

    shuffled = v.get("router_shuffled")
    if shuffled and shuffled.get("layer0_relabel_prediction") is not None:
        print(
            f"router_shuffled: layer-0 relabel prediction "
            f"{shuffled['layer0_relabel_prediction']:.6f} vs measured "
            f"{shuffled['layer0_measured']:.6f} "
            f"(exact={shuffled['layer0_matches_prediction']}); "
            f"fixed points per layer {shuffled['per_layer_fixed_points']}"
        )
    elif shuffled:
        print(f"router_shuffled: no layer-0 prediction ({entry.get('shuffle_error')})")
    if v.get("experts_q4") and v.get("all_q4"):
        print(
            "layer-0 isolation (routing independent of expert weights): "
            f"experts_q4={v['experts_q4']['per_layer_set_agreement'][0]:.6f}, "
            f"all_q4={v['all_q4']['per_layer_set_agreement'][0]:.6f}"
        )
    if entry.get("variants_qdq_note"):
        note = entry["variants_qdq_note"]
        print(
            f"qdq: {note.get('expert_tensors_quantized')} expert tensors "
            f"in {note.get('experts_qdq_s')}s; "
            f"{note.get('router_tensors_quantized')} router tensors "
            f"in {note.get('router_qdq_s')}s"
        )


def print_domain_table(entry: dict) -> None:
    """Overall set agreement split by prompt domain."""
    v = entry["variants"]
    header = f"{'domain':>10} | " + " | ".join(f"{n:>14}" for n in _VARIANTS_SHOWN)
    print(header)
    print("-" * len(header))
    for domain in C.DOMAINS:
        cells = []
        for name in _VARIANTS_SHOWN:
            data = v.get(name)
            cells.append(f"{data['by_domain'][domain]:.6f}" if data else f"{'-':>14}")
        print(f"{domain:>10} | " + " | ".join(cells))


def print_expectations(records: list[dict]) -> None:
    print(f"\n{'id':>3} | {'model':>11} | expectation{'':<42} | observed | verdict")
    print("-" * 108)
    for rec in records:
        observed = "-" if rec["observed"] is None else f"{rec['observed']:.6f}"
        print(
            f"{rec['id']:>3} | {rec['model']:>11} | {rec['expectation']:<54} | "
            f"{observed:>8} | {rec['verdict']}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Routing agreement smoke (Experiment B)")
    parser.add_argument("--out", type=Path, default=SMOKE_DIR / "results.json")
    parser.add_argument(
        "--olmoe",
        choices=("auto", "always", "never"),
        default="auto",
        help="auto: run the secondary model only if the RAM/time budgets hold",
    )
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 8))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    t_start = time.perf_counter()
    print(f"routing-agreement · seed={C.SEED} · threads={args.threads} · seq_len={C.SEQ_LEN}")

    entries: list[dict] = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            C.PRIMARY["repo_id"], revision=C.PRIMARY["revision"]
        )
        primary = run_model(C.PRIMARY, tokenizer, check_budgets=False)
        entries.append(primary)
        if primary["status"] == "ok":
            print_model_table(primary)
            print("\nper-domain overall set agreement")
            print_domain_table(primary)
        else:
            print(f"SKIP {C.PRIMARY['label']}: {primary.get('skip_reason')}")
    except Exception as exc:
        print(f"SKIP {C.PRIMARY['label']}: setup failed: {type(exc).__name__}: {exc}")

    if args.olmoe != "never":
        try:
            secondary_tokenizer = AutoTokenizer.from_pretrained(
                C.SECONDARY["repo_id"], revision=C.SECONDARY["revision"]
            )
            secondary = run_model(
                C.SECONDARY, secondary_tokenizer, check_budgets=args.olmoe == "auto"
            )
            entries.append(secondary)
            if secondary["status"] == "ok":
                print_model_table(secondary)
                print("\nper-domain overall set agreement")
                print_domain_table(secondary)
            else:
                print(f"\nSKIP {C.SECONDARY['label']}: {secondary.get('skip_reason')}")
        except Exception as exc:
            entries.append(
                {
                    "label": C.SECONDARY["label"],
                    "repo_id": C.SECONDARY["repo_id"],
                    "revision": C.SECONDARY["revision"],
                    "dtype": C.SECONDARY["dtype"],
                    "status": "skipped",
                    "skip_reason": f"setup failed: {type(exc).__name__}: {exc}",
                }
            )
            print(f"\nSKIP {C.SECONDARY['label']}: {type(exc).__name__}: {exc}")

    expectations = evaluate_expectations(entries)
    print_expectations(expectations)

    payload = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "smoke": "routing-agreement",
        "spec": ".scratch/specs/moe-p0-and-routing.md · Experiment B",
        "seed": C.SEED,
        "seq_len": C.SEQ_LEN,
        "domains": list(C.DOMAINS),
        "quant": {"bits": C.QUANT_BITS, "block": C.QUANT_BLOCK, "scheme": "symmetric RTN"},
        "variants": list(C.VARIANTS),
        "execution_order": list(C.EXECUTION_ORDER),
        "variant_notes": C.VARIANT_NOTES,
        "threads": args.threads,
        "models": entries,
        "expectations": expectations,
        "total_elapsed_s": round(time.perf_counter() - t_start, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}  total {payload['total_elapsed_s']}s")

    ran = [e for e in entries if e["status"] == "ok"]
    if not ran:
        print("FAIL: no model produced routing measurements")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
