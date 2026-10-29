"""Routing calibration smoke — smokes/routing-calibration/ · issue 69.

Question: is routing overlap (Jaccard of per-layer top-k expert sets) a usable
lineage signal? Three populations are measured with the same probe and QDQ
contract as ``smokes/routing-agreement`` (imported, never copied):

  * same model, requantized — base vs experts-4bit and vs experts+router-4bit;
  * same architecture, different checkpoint — OLMoE-0924 vs OLMoE-0924-SFT
    (the hard negative that decides the threshold's false-positive rate);
  * different family, matching E/k — TinyMixtral vs TinyLLama-4x1.1B-MoE
    (E=4, k=2). The 64-expert different-family cell is an explicit skip: no
    cached, weight-complete, non-OLMoE 64-expert checkpoint exists.

The threshold is frozen before the run in ``config.THRESHOLD_RULE`` and uses
only the same-model requant population (mean - 3 sd of the 16 per-layer mean
Jaccards). The negative population is never consulted for tuning.

Each checkpoint runs in its own subprocess and writes integer top-k arrays to
a temporary ``.npz``; the parent compares arrays and measures the run's true
peak RSS. This keeps peak memory at one checkpoint (the two OLMoE checkpoints
would otherwise sum to ~28 GB) and makes a load/capture failure a recorded
skip, never an aborted run. All loads are ``local_files_only=True`` with
``trust_remote_code=False``; the only checkpoint parser is transformers'
safetensors reader.

Run: ``uv run --group smokes python smokes/routing-calibration/smoke.py``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from math import comb
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

SMOKE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SMOKE_DIR.parent.parent
sys.path.insert(0, str(SMOKE_DIR))

import config as C  # noqa: E402

RA_DIR = SMOKE_DIR.parent / "routing-agreement"
sys.path.insert(0, str(RA_DIR))

import quant as Q  # noqa: E402  (shared frozen QDQ contract; never copied)

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


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
    exact = 1.0 / total
    jaccard = 0.0
    for x in range(max(0, 2 * k - n_experts), k + 1):
        p_x = comb(k, x) * comb(n_experts - k, k - x) / total
        jaccard += p_x * (x / (2 * k - x))
    return {
        "set_agreement": exact,
        "mean_jaccard": jaccard,
        "set_agreement_fraction": f"1/C({n_experts},{k}) = {exact}",
        "mean_jaccard_fraction": f"{jaccard:.12g}",
    }


def sanitize(text: str, limit: int = 1200) -> str:
    """Strip control characters and bound diagnostic text from untrusted code."""
    return _CTRL_RE.sub("?", text)[-limit:]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Probe set (same 24 prompts, 96 ids, one prefill batch, no generation)
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
    input_ids = torch.tensor(rows, dtype=torch.long)
    attention_mask = torch.tensor(masks, dtype=torch.long)
    h = hashlib.sha256()
    h.update(input_ids.numpy().astype("<i8").tobytes())
    h.update(attention_mask.numpy().astype("u1").tobytes())
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_mask": attention_mask.bool().reshape(-1),
        "records": records,
        "n_truncated": sum(r["truncated"] for r in records),
        "n_padded": sum(r["padded"] for r in records),
        "pad_id": int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None,
        "input_sha256": h.hexdigest(),
    }


def mask_from_records(records: list[dict], seq_len: int) -> np.ndarray:
    """Reconstruct the flattened scored-position mask from worker records."""
    return np.concatenate(
        [np.array([True] * r["used"] + [False] * (seq_len - r["used"])) for r in records]
    ).astype(bool)


# ---------------------------------------------------------------------------
# Routing capture
# ---------------------------------------------------------------------------


def capture_routing(model, probe: dict, top_k: int) -> tuple[np.ndarray, dict]:
    """Top-k expert ids per (layer, row) from the router-logits tuple.

    One forward pass over the whole probe batch; no generation, no chunking.
    Returns an int16 array ``[layers, rows, k]`` plus tuple bookkeeping.
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


def compare_pair(
    reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray, chance: dict
) -> dict:
    """Per-layer Jaccard mean/min/max/sd + set agreement, over masked rows.

    Layers are compared by index for ``min(reference_layers, candidate_layers)``
    layers. Cross-family depth alignment is index-only and reported as such.
    """
    n = min(reference.shape[0], candidate.shape[0])
    per_mean, per_min, per_max, per_sd, per_set = [], [], [], [], []
    for layer in range(n):
        ref = reference[layer][mask]
        cand = candidate[layer][mask]
        jac = _jaccard_rows(ref, cand)
        per_mean.append(float(jac.mean()))
        per_min.append(float(jac.min()))
        per_max.append(float(jac.max()))
        per_sd.append(float(jac.std(ddof=1)) if jac.size > 1 else 0.0)
        per_set.append(float(_set_equal_rows(ref, cand).mean()))

    overall_j = float(np.mean(per_mean))
    overall_set = float(np.mean(per_set))
    return {
        "layers_compared": int(n),
        "reference_layers": int(reference.shape[0]),
        "candidate_layers": int(candidate.shape[0]),
        "depth_aligned": bool(reference.shape[0] == candidate.shape[0]),
        "per_layer_mean_jaccard": per_mean,
        "per_layer_min_jaccard": per_min,
        "per_layer_max_jaccard": per_max,
        "per_layer_sd_jaccard": per_sd,
        "per_layer_set_agreement": per_set,
        "overall_mean_jaccard": overall_j,
        "overall_mean_jaccard_excl_layer0": float(np.mean(per_mean[1:])) if n > 1 else None,
        "overall_set_agreement": overall_set,
        "overall_set_agreement_excl_layer0": float(np.mean(per_set[1:])) if n > 1 else None,
        "layer0_mean_jaccard": per_mean[0],
        "layer0_set_agreement": per_set[0],
        "distribution": {
            "n_layers": int(n),
            "mean_of_layer_means": overall_j,
            "min_layer_mean": float(min(per_mean)),
            "max_layer_mean": float(max(per_mean)),
            "sd_layer_means": float(statistics.stdev(per_mean)) if n > 1 else 0.0,
        },
        "chance": chance,
        "jaccard_minus_chance": overall_j - chance["mean_jaccard"],
        "set_minus_chance": overall_set - chance["set_agreement"],
    }


def row_hit_stats(
    reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray, threshold: float
) -> dict:
    """Fraction of masked row-level Jaccards at or above ``threshold``."""
    n = min(reference.shape[0], candidate.shape[0])
    values = np.concatenate(
        [_jaccard_rows(reference[layer][mask], candidate[layer][mask]) for layer in range(n)]
    )
    hits = int((values >= threshold).sum())
    return {
        "rows": int(values.size),
        "at_or_above": hits,
        "fraction": float(hits / values.size),
    }


# ---------------------------------------------------------------------------
# Fingerprints — every one carries router precision and quant type
# ---------------------------------------------------------------------------


def fingerprint(
    array: np.ndarray,
    *,
    mask: np.ndarray,
    spec: dict,
    variant: str,
    router_precision: str,
    quant_type: str,
    quant_bits: int | None,
    quant_block: int | None,
    seq_len: int,
    probe_sha256: str,
) -> dict:
    """Content hash of the scored top-k ids plus every field needed to compare it."""
    scored = np.ascontiguousarray(array[:, mask, :])
    h = hashlib.sha256()
    h.update(f"{scored.shape}|{scored.dtype.str}|".encode())
    h.update(scored.tobytes())
    return {
        "label": spec["label"],
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "variant": variant,
        "model_dtype": spec["dtype"],
        "router_precision": router_precision,
        "quant_type": quant_type,
        "quant_bits": quant_bits,
        "quant_block": quant_block,
        "n_layers": int(scored.shape[0]),
        "rows_scored": int(scored.shape[1]),
        "seq_len": seq_len,
        "probe_input_sha256": probe_sha256,
        "top_k": int(scored.shape[2]),
        "n_experts": int(spec["n_experts"]),
        "sha256": h.hexdigest(),
    }


# ---------------------------------------------------------------------------
# Variant application (shared QDQ contract, in place)
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


def restore(saved: list[tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
    """Copy saved tensors back into their parameters."""
    for param, value in saved:
        with torch.no_grad():
            param.copy_(value)


# ---------------------------------------------------------------------------
# Worker: one checkpoint, one subprocess, one peak RSS
# ---------------------------------------------------------------------------


def local_weight_gb(spec: dict) -> float | None:
    """Checkpoint payload size from the cached safetensors index (no network).

    Mergekit indexes omit ``metadata.total_size``, so fall back to summing the
    cached files named in ``weight_map``. Returns None only if the cache is
    incomplete (the loader then fails cleanly and the worker is skipped).
    """
    index_path = try_to_load_from_cache(
        spec["repo_id"], "model.safetensors.index.json", revision=spec["revision"]
    )
    if isinstance(index_path, str) and os.path.exists(index_path):
        index = json.loads(Path(index_path).read_text())
        total = index.get("metadata", {}).get("total_size")
        if total:
            return float(total) / 1e9
        files = sorted(set(index.get("weight_map", {}).values()))
        sizes = []
        for name in files:
            path = try_to_load_from_cache(spec["repo_id"], name, revision=spec["revision"])
            if not isinstance(path, str) or not os.path.exists(path):
                return None
            sizes.append(os.path.getsize(path))
        if sizes:
            return float(sum(sizes)) / 1e9
    path = try_to_load_from_cache(spec["repo_id"], "model.safetensors", revision=spec["revision"])
    if isinstance(path, str) and os.path.exists(path):
        return os.path.getsize(path) / 1e9
    return None


def worker_main(label: str, out_dir: Path, threads: int) -> int:
    """Load one checkpoint, capture the requested variants, write arrays + meta."""
    spec = C.MODELS[label]
    torch.set_num_threads(threads)
    torch.manual_seed(C.SEED)
    t_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        spec["tokenizer"]["repo_id"],
        revision=spec["tokenizer"]["revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    probe = build_probe(tokenizer, C.SEQ_LEN)

    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        spec["repo_id"],
        revision=spec["revision"],
        dtype=getattr(torch, spec["dtype"]),
        output_router_logits=True,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()
    load_s = time.perf_counter() - t_load

    routers = router_params(model)
    experts = expert_params(model)
    if not routers:
        raise Q.CheckpointTensorError(
            "no router gate tensors matched — checkpoint layout unexpected"
        )
    base_router_precision = str(routers[0][1].dtype).replace("torch.", "")

    t_fwd = time.perf_counter()
    base, rl_info = capture_routing(model, probe, spec["top_k"])
    forward_s = time.perf_counter() - t_fwd
    chance = chance_levels(rl_info["top_k"], rl_info["n_experts"])

    # The fingerprint's n_experts/top_k come from the router logits, not the
    # pinned config: a config lie is visible in the record.
    measured = {
        **spec,
        "n_experts": rl_info["n_experts"],
        "top_k": rl_info["top_k"],
    }

    arrays: dict[str, np.ndarray] = {"base": base}
    fps: dict[str, dict] = {
        "base": fingerprint(
            base,
            mask=probe["position_mask"].numpy(),
            spec=measured,
            variant="base",
            router_precision=base_router_precision,
            quant_type="none",
            quant_bits=None,
            quant_block=None,
            seq_len=C.SEQ_LEN,
            probe_sha256=probe["input_sha256"],
        )
    }
    qdq_note: dict = {"router_tensors": len(routers), "expert_tensors": len(experts)}

    if spec["mode"] == "requant":
        t0 = time.perf_counter()
        n_exp = apply_experts_q4(model)
        qdq_note["expert_tensors_quantized"] = n_exp
        qdq_note["experts_qdq_s"] = round(time.perf_counter() - t0, 2)

        t0 = time.perf_counter()
        cand, _ = capture_routing(model, probe, spec["top_k"])
        arrays["experts_q4"] = cand
        fps["experts_q4"] = fingerprint(
            cand,
            mask=probe["position_mask"].numpy(),
            spec=measured,
            variant="experts_q4",
            router_precision=base_router_precision,
            quant_type="experts_qdq_symmetric_rtn_4bit_block32",
            quant_bits=C.QUANT_BITS,
            quant_block=C.QUANT_BLOCK,
            seq_len=C.SEQ_LEN,
            probe_sha256=probe["input_sha256"],
        )

        t0 = time.perf_counter()
        saved_router = qdq_router(model)
        qdq_note["router_tensors_quantized"] = len(saved_router)
        qdq_note["router_qdq_s"] = round(time.perf_counter() - t0, 2)

        t0 = time.perf_counter()
        cand, _ = capture_routing(model, probe, spec["top_k"])
        arrays["all_q4"] = cand
        fps["all_q4"] = fingerprint(
            cand,
            mask=probe["position_mask"].numpy(),
            spec=measured,
            variant="all_q4",
            router_precision="q4_block32",
            quant_type="experts_router_qdq_symmetric_rtn_4bit_block32",
            quant_bits=C.QUANT_BITS,
            quant_block=C.QUANT_BLOCK,
            seq_len=C.SEQ_LEN,
            probe_sha256=probe["input_sha256"],
        )
        restore(saved_router)

    np.savez_compressed(out_dir / f"{label}.npz", **arrays)

    meta = {
        "label": label,
        "status": "ok",
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "dtype": spec["dtype"],
        "mode": spec["mode"],
        "role": spec["role"],
        "tokenizer": spec["tokenizer"],
        "probe": {
            "prompts": len(probe["records"]),
            "seq_len": C.SEQ_LEN,
            "rows_scored": int(probe["position_mask"].sum()),
            "truncated_prompts": probe["n_truncated"],
            "padded_prompts": probe["n_padded"],
            "pad_id": probe["pad_id"],
            "input_sha256": probe["input_sha256"],
            "records": probe["records"],
        },
        "router_logits": rl_info,
        "n_layers": int(base.shape[0]),
        "n_experts": rl_info["n_experts"],
        "top_k": rl_info["top_k"],
        "router_params": len(routers),
        "expert_params": len(experts),
        "base_router_precision": base_router_precision,
        "chance": chance,
        "weight_gb": local_weight_gb(spec),
        "load_s": round(load_s, 2),
        "forward_s": round(forward_s, 2),
        "qdq_note": qdq_note,
        "fingerprints": fps,
        "peak_rss_gb": round(peak_rss_gb(), 3),
        "elapsed_s": round(time.perf_counter() - t_start, 1),
    }
    (out_dir / f"{label}.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[{label}] ok L={meta['n_layers']} E={meta['n_experts']} k={meta['top_k']} "
        f"rows={meta['probe']['rows_scored']} peak={meta['peak_rss_gb']}GB "
        f"elapsed={meta['elapsed_s']}s"
    )
    return 0


# ---------------------------------------------------------------------------
# Parent: dispatch workers, compare arrays, calibrate
# ---------------------------------------------------------------------------


def run_worker(label: str, out_dir: Path, threads: int) -> dict:
    """Spawn a worker subprocess; return its meta or a sanitized skip entry."""
    spec = C.MODELS[label]
    weight_gb = local_weight_gb(spec)
    if weight_gb is not None and weight_gb > C.MAX_WEIGHT_GB:
        return {
            "label": label,
            "status": "skipped",
            "skip_reason": (
                f"checkpoint payload {weight_gb:.1f} GB exceeds the {C.MAX_WEIGHT_GB:.0f} GB "
                f"per-model budget (process budget {C.RAM_BUDGET_GB:.0f} GB)"
            ),
        }

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        label,
        "--out-dir",
        str(out_dir),
        "--threads",
        str(threads),
    ]
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=C.PER_MODEL_TIMEOUT_S,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "label": label,
            "status": "skipped",
            "skip_reason": f"worker timed out after {C.PER_MODEL_TIMEOUT_S:.0f}s",
        }
    elapsed = round(time.perf_counter() - t0, 1)
    if proc.returncode != 0:
        return {
            "label": label,
            "status": "skipped",
            "skip_reason": (
                f"worker exited {proc.returncode}: {sanitize(proc.stderr or proc.stdout)}"
            ),
            "worker_s": elapsed,
        }

    meta_path = out_dir / f"{label}.json"
    npz_path = out_dir / f"{label}.npz"
    if not meta_path.exists() or not npz_path.exists():
        return {
            "label": label,
            "status": "skipped",
            "skip_reason": "worker exited 0 but wrote no capture",
            "worker_s": elapsed,
        }
    meta = json.loads(meta_path.read_text())
    arrays = np.load(npz_path, allow_pickle=False)
    meta["arrays"] = {name: arrays[name] for name in arrays.files}
    meta["worker_s"] = elapsed
    return meta


def pair_metrics(
    captures: dict[str, dict],
    *,
    label: str,
    reference: tuple[str, str],
    candidate: tuple[str, str],
    expected_experts: int,
    expected_top_k: int,
) -> dict:
    """Compare two captures, refusing to compare misaligned probes or E/k."""
    ref_model, ref_variant = reference
    cand_model, cand_variant = candidate
    ref_entry = captures.get(ref_model)
    cand_entry = captures.get(cand_model)
    base = {
        "label": label,
        "reference": {"model": ref_model, "variant": ref_variant},
        "candidate": {"model": cand_model, "variant": cand_variant},
    }
    if not ref_entry or ref_entry["status"] != "ok":
        return {
            **base,
            "status": "skipped",
            "skip_reason": f"{ref_model} capture unavailable",
        }
    if not cand_entry or cand_entry["status"] != "ok":
        return {
            **base,
            "status": "skipped",
            "skip_reason": f"{cand_model} capture unavailable",
        }
    if ref_variant not in ref_entry["arrays"] or cand_variant not in cand_entry["arrays"]:
        return {
            **base,
            "status": "skipped",
            "skip_reason": "requested variant was not captured",
        }

    if ref_entry["probe"]["input_sha256"] != cand_entry["probe"]["input_sha256"]:
        return {
            **base,
            "status": "skipped",
            "skip_reason": "probe token ids differ; refusing to compare misaligned rows",
        }
    if (ref_entry["n_experts"], ref_entry["top_k"]) != (
        expected_experts,
        expected_top_k,
    ):
        return {
            **base,
            "status": "skipped",
            "skip_reason": (
                f"reference E/k = {ref_entry['n_experts']}/{ref_entry['top_k']}, "
                f"expected {expected_experts}/{expected_top_k}"
            ),
        }
    if (cand_entry["n_experts"], cand_entry["top_k"]) != (
        ref_entry["n_experts"],
        ref_entry["top_k"],
    ):
        return {
            **base,
            "status": "skipped",
            "skip_reason": (
                f"candidate E/k = {cand_entry['n_experts']}/{cand_entry['top_k']}, "
                f"reference {ref_entry['n_experts']}/{ref_entry['top_k']}"
            ),
        }

    ref = ref_entry["arrays"][ref_variant]
    cand = cand_entry["arrays"][cand_variant]
    mask = mask_from_records(ref_entry["probe"]["records"], C.SEQ_LEN)
    chance = chance_levels(ref_entry["top_k"], ref_entry["n_experts"])
    metrics = compare_pair(ref, cand, mask, chance)
    return {
        **base,
        "status": "ok",
        "reference_fingerprint": ref_entry["fingerprints"][ref_variant],
        "candidate_fingerprint": cand_entry["fingerprints"][cand_variant],
        **metrics,
    }


def calibrate(captures: dict[str, dict], pairs: dict[str, dict]) -> dict:
    """Apply the pre-registered mean-3sd rule; report layer and row FPR/TPR."""
    rule = C.THRESHOLD_RULE
    mult = rule["sd_multiplier"]

    def one(primary_label: str) -> dict:
        prim = pairs.get(primary_label)
        neg = pairs.get(rule["negative_population"])
        if not prim or prim.get("status") != "ok":
            return {"status": "skipped", "skip_reason": f"{primary_label} unavailable"}
        if not neg or neg.get("status") != "ok":
            return {
                "status": "skipped",
                "skip_reason": "negative population unavailable",
            }

        layer_values = prim["per_layer_mean_jaccard"]
        neg_layers = neg["per_layer_mean_jaccard"]
        mean = float(statistics.mean(layer_values))
        sd = float(statistics.stdev(layer_values)) if len(layer_values) > 1 else 0.0
        threshold = mean - mult * sd
        tpr_layer = float(sum(v >= threshold for v in layer_values) / len(layer_values))
        fpr_layer = float(sum(v >= threshold for v in neg_layers) / len(neg_layers))

        prim_ref = captures[prim["reference"]["model"]]["arrays"][prim["reference"]["variant"]]
        prim_cand = captures[prim["candidate"]["model"]]["arrays"][prim["candidate"]["variant"]]
        prim_mask = mask_from_records(
            captures[prim["reference"]["model"]]["probe"]["records"], C.SEQ_LEN
        )
        neg_ref = captures[neg["reference"]["model"]]["arrays"][neg["reference"]["variant"]]
        neg_cand = captures[neg["candidate"]["model"]]["arrays"][neg["candidate"]["variant"]]
        neg_mask = mask_from_records(
            captures[neg["reference"]["model"]]["probe"]["records"], C.SEQ_LEN
        )

        result = {
            "status": "ok",
            "population": primary_label,
            "rule": f"mean - {mult} * sample_sd (ddof=1)",
            "mean": mean,
            "sd": sd,
            "threshold": threshold,
            "calibration_layers": len(layer_values),
            "negative_layers": len(neg_layers),
            "tpr_layer": tpr_layer,
            "tpr_row": row_hit_stats(prim_ref, prim_cand, prim_mask, threshold),
            "negative_population": rule["negative_population"],
            "negative_mean": float(statistics.mean(neg_layers)),
            "negative_min": float(min(neg_layers)),
            "negative_max": float(max(neg_layers)),
            "fpr_layer": fpr_layer,
            "fpr_row": row_hit_stats(neg_ref, neg_cand, neg_mask, threshold),
            "margin_mean": mean - float(statistics.mean(neg_layers)),
            "margin_worst_layer": float(min(layer_values)) - float(max(neg_layers)),
        }

        # Sensitivity only: the same rule on layers 1..L-1. Not the verdict.
        if len(layer_values) > 1:
            ex_mean = float(statistics.mean(layer_values[1:]))
            ex_sd = float(statistics.stdev(layer_values[1:]))
            ex_t = ex_mean - mult * ex_sd
            result["excl_layer0"] = {
                "mean": ex_mean,
                "sd": ex_sd,
                "threshold": ex_t,
                "tpr_layer": float(
                    sum(v >= ex_t for v in layer_values[1:]) / len(layer_values[1:])
                ),
                "fpr_layer": float(sum(v >= ex_t for v in neg_layers[1:]) / len(neg_layers[1:])),
                "not_used_for_verdict": True,
            }
        return result

    primary = one(rule["primary_population"])
    secondary = (
        one(rule["secondary_population"])
        if rule.get("secondary_population") in pairs
        else {"status": "skipped", "skip_reason": "no secondary calibration population"}
    )
    return {"rule": rule, "primary": primary, "secondary": secondary}


# The executed rule *is* the pre-registered object in config.py: results.json
# records this same dict, so the archive cannot drift from the code.
VERDICT_RULES = C.VERDICT_RULES


def decide_verdict(cal: dict) -> str:
    """Name the verdict by walking ``VERDICT_RULES``; no numbers live here."""
    primary = cal.get("primary", {})
    fallback = VERDICT_RULES["fallback"]
    if primary.get("status") != "ok":
        return fallback
    fpr = primary["fpr_layer"]
    tpr = primary["tpr_layer"]
    for name, band in VERDICT_RULES["bands"].items():
        if fpr <= band["fpr_layer_max"] and tpr >= band["tpr_layer_min"]:
            return name
    return fallback


def resolve(ctx: dict, path: str):
    """Resolve a dotted metric path, allowing list indices."""
    node = ctx
    for part in path.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict):
            node = node[part]
        else:
            raise KeyError(part)
    return node


def evaluate_expectations(ctx: dict) -> list[dict]:
    """Mark every pre-registered expectation held/missed/skipped."""
    out = []
    for exp in C.EXPECTATIONS:
        try:
            observed = resolve(ctx, exp["metric"])
        except (KeyError, IndexError, TypeError, ZeroDivisionError):
            out.append({**exp, "observed": None, "status": "skipped"})
            continue
        op, target = exp["op"], exp["value"]
        if op == "ge":
            ok = observed >= target
        elif op == "le":
            ok = observed <= target
        elif op == "gt":
            ok = observed > target
        elif op == "lt":
            ok = observed < target
        elif op == "between":
            ok = target[0] <= observed <= target[1]
        elif op == "eq":
            ok = abs(observed - target) <= exp.get("tol", 1e-9)
        elif op == "max_lt":
            ok = all(v < target for v in observed)
        else:  # pragma: no cover - frozen data
            raise ValueError(f"unknown expectation op {op!r}")
        out.append({**exp, "observed": observed, "status": "held" if ok else "missed"})
    return out


def derived_metrics(models: dict, pairs: dict) -> dict:
    """Differences used by the pre-registered expectations."""
    out: dict[str, float] = {}
    try:
        tiny = models["tinymixtral"]["variants"]
        out["tinymixtral_router_cost_jaccard"] = (
            tiny["experts_q4"]["overall_mean_jaccard"] - tiny["all_q4"]["overall_mean_jaccard"]
        )
    except KeyError:
        pass
    try:
        olm = models["olmoe_0924"]["variants"]
        out["olmoe_router_cost_jaccard"] = (
            olm["experts_q4"]["overall_mean_jaccard"] - olm["all_q4"]["overall_mean_jaccard"]
        )
        out["olmoe_jaccard_minus_set_agreement"] = (
            olm["experts_q4"]["overall_mean_jaccard"] - olm["experts_q4"]["overall_set_agreement"]
        )
    except KeyError:
        pass
    try:
        ckpt = pairs["olmoe_0924_vs_0924_sft"]
        out["checkpoint_minus_requant_jaccard"] = (
            ckpt["overall_mean_jaccard"]
            - models["olmoe_0924"]["variants"]["experts_q4"]["overall_mean_jaccard"]
        )
        out["checkpoint_minus_chance_jaccard"] = (
            ckpt["overall_mean_jaccard"] - ckpt["chance"]["mean_jaccard"]
        )
    except KeyError:
        pass
    try:
        fam = pairs["tinymixtral_vs_tinyllama_moe"]
        out["family_minus_chance_jaccard"] = (
            fam["overall_mean_jaccard"] - fam["chance"]["mean_jaccard"]
        )
    except KeyError:
        pass
    return out


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_pair_table(pair: dict) -> None:
    """Per-layer Jaccard distribution (mean/min/max/sd) and set agreement."""
    print(
        f"\n{pair['label']}  [{pair['reference']['model']}:{pair['reference']['variant']} -> "
        f"{pair['candidate']['model']}:{pair['candidate']['variant']}]"
    )
    print(
        f"  layers={pair['layers_compared']}  mean J={pair['overall_mean_jaccard']:.4f}  "
        f"excl0={pair['overall_mean_jaccard_excl_layer0']:.4f}  "
        f"set={pair['overall_set_agreement']:.4f}  chance J={pair['chance']['mean_jaccard']:.4f}"
    )
    print(
        f"  {'layer':>5} | {'mean J':>8} | {'min J':>8} | {'max J':>8} | {'sd J':>8} | {'set':>8}"
    )
    print("  " + "-" * 62)
    for layer in range(pair["layers_compared"]):
        print(
            f"  {layer:>5} | {pair['per_layer_mean_jaccard'][layer]:>8.4f} | "
            f"{pair['per_layer_min_jaccard'][layer]:>8.4f} | "
            f"{pair['per_layer_max_jaccard'][layer]:>8.4f} | "
            f"{pair['per_layer_sd_jaccard'][layer]:>8.4f} | "
            f"{pair['per_layer_set_agreement'][layer]:>8.4f}"
        )


def print_summary(models: dict, pairs: dict) -> None:
    print("\npopulation summary (J = mean per-layer mean Jaccard over scored rows)")
    print(
        f"  {'pair':<30} | {'L':>2} | {'mean J':>7} | {'min':>7} | {'max':>7} | "
        f"{'excl0':>7} | {'set':>7} | {'chance':>7}"
    )
    print("  " + "-" * 108)
    for pair in pairs.values():
        if pair.get("status") != "ok":
            continue
        print(
            f"  {pair['label']:<30} | {pair['layers_compared']:>2} | "
            f"{pair['overall_mean_jaccard']:>7.4f} | "
            f"{pair['distribution']['min_layer_mean']:>7.4f} | "
            f"{pair['distribution']['max_layer_mean']:>7.4f} | "
            f"{pair['overall_mean_jaccard_excl_layer0']:>7.4f} | "
            f"{pair['overall_set_agreement']:>7.4f} | {pair['chance']['mean_jaccard']:>7.4f}"
        )


def print_calibration(cal: dict) -> None:
    print("\ncalibration (threshold = mean - 3*sd of the primary same-model requant population)")
    for name in ("primary", "secondary"):
        entry = cal[name]
        if entry.get("status") != "ok":
            print(f"  {name:<9}: SKIP {entry.get('skip_reason')}")
            continue
        print(
            f"  {name:<9}: pop={entry['population']}  mean={entry['mean']:.4f}  "
            f"sd={entry['sd']:.4f}  T={entry['threshold']:.4f}  "
            f"TPR(layer)={entry['tpr_layer']:.4f}  TPR(row)={entry['tpr_row']['fraction']:.4f}"
        )
        print(
            f"             negative={entry['negative_population']}  "
            f"J mean/min/max={entry['negative_mean']:.4f}/{entry['negative_min']:.4f}/"
            f"{entry['negative_max']:.4f}  "
            f"FPR(layer)={entry['fpr_layer']:.4f} "
            f"({round(entry['fpr_layer'] * entry['negative_layers'])}/{entry['negative_layers']})  "
            f"FPR(row)={entry['fpr_row']['fraction']:.4f} "
            f"({entry['fpr_row']['at_or_above']}/{entry['fpr_row']['rows']})"
        )
        print(
            f"             margin(mean)={entry['margin_mean']:+.4f}  "
            f"margin(worst layer vs worst negative)={entry['margin_worst_layer']:+.4f}"
        )
        if "excl_layer0" in entry:
            ex = entry["excl_layer0"]
            print(
                f"             sensitivity excl layer 0: T={ex['threshold']:.4f}  "
                f"TPR={ex['tpr_layer']:.4f}  FPR={ex['fpr_layer']:.4f} (not the verdict)"
            )


def print_expectations(records: list[dict]) -> None:
    print(f"\n{'id':>3} | {'expectation':<78} | {'observed':>10} | verdict")
    print("-" * 118)
    for rec in records:
        observed = rec["observed"]
        shown = f"{observed:.4f}" if isinstance(observed, float) else str(observed)
        print(f"{rec['id']:>3} | {rec['text']:<78} | {shown:>10} | {rec['status']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Routing calibration smoke (issue 69)")
    parser.add_argument("--out", type=Path, default=SMOKE_DIR / "results.json")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 8))
    parser.add_argument("--worker", choices=tuple(C.MODELS), help=argparse.SUPPRESS)
    parser.add_argument("--out-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if args.out_dir is None:
            parser.error("--worker requires --out-dir")
        sys.exit(worker_main(args.worker, args.out_dir, args.threads))

    t_start = time.perf_counter()
    torch.set_num_threads(args.threads)
    print(
        f"routing-calibration · issue={C.ISSUE} · seed={C.SEED} · threads={args.threads} "
        f"· seq_len={C.SEQ_LEN} · prompts={len(C.PROMPTS)}"
    )
    print(
        "probe/QDQ reused from smokes/routing-agreement: "
        f"config sha256={sha256_file(REPO_ROOT / C.PROBE_SOURCE['config_path'])[:16]} "
        f"quant sha256={sha256_file(REPO_ROOT / C.PROBE_SOURCE['quant_path'])[:16]}"
    )

    captures: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="routing-cal-") as tmp:
        out_dir = Path(tmp)
        for label in C.WORKER_ORDER:
            if time.perf_counter() - t_start > C.TOTAL_TIME_BUDGET_S:
                captures[label] = {
                    "label": label,
                    "status": "skipped",
                    "skip_reason": f"total time budget {C.TOTAL_TIME_BUDGET_S:.0f}s exhausted",
                }
                continue
            print(f"\n[{label}] loading (offline) ...", flush=True)
            captures[label] = run_worker(label, out_dir, args.threads)
            if captures[label]["status"] != "ok":
                print(f"[{label}] SKIP {captures[label].get('skip_reason')}")
            else:
                print(
                    f"[{label}] peak RSS {captures[label]['peak_rss_gb']:.2f} GB, "
                    f"worker {captures[label]['worker_s']}s"
                )

    models: dict[str, dict] = {}
    for label, spec in C.MODELS.items():
        cap = captures.get(label)
        if not cap or cap.get("status") != "ok":
            models[label] = {
                "label": label,
                "repo_id": spec["repo_id"],
                "revision": spec["revision"],
                "dtype": spec["dtype"],
                "mode": spec["mode"],
                "role": spec["role"],
                "status": "skipped",
                "skip_reason": (cap or {}).get("skip_reason", "worker not run"),
            }
            continue
        mask = mask_from_records(cap["probe"]["records"], C.SEQ_LEN)
        base = cap["arrays"]["base"]
        variants = {"base": compare_pair(base, base, mask, cap["chance"])}
        for variant in ("experts_q4", "all_q4"):
            if variant in cap["arrays"]:
                variants[variant] = compare_pair(base, cap["arrays"][variant], mask, cap["chance"])
        models[label] = {
            "label": label,
            "repo_id": cap["repo_id"],
            "revision": cap["revision"],
            "dtype": cap["dtype"],
            "mode": cap["mode"],
            "role": cap["role"],
            "status": "ok",
            "tokenizer": cap["tokenizer"],
            "probe": cap["probe"],
            "router_logits": cap["router_logits"],
            "n_layers": cap["n_layers"],
            "n_experts": cap["n_experts"],
            "top_k": cap["top_k"],
            "base_router_precision": cap["base_router_precision"],
            "chance": cap["chance"],
            "weight_gb": cap["weight_gb"],
            "load_s": cap["load_s"],
            "forward_s": cap["forward_s"],
            "qdq_note": cap["qdq_note"],
            "fingerprints": cap["fingerprints"],
            "variants": variants,
            "peak_rss_gb": cap["peak_rss_gb"],
            "elapsed_s": cap["elapsed_s"],
            "worker_s": cap["worker_s"],
        }

    pairs: dict[str, dict] = {}
    for spec in C.REQUANT_PAIRS:
        pairs[spec["label"]] = pair_metrics(
            captures,
            label=spec["label"],
            reference=(spec["model"], "base"),
            candidate=(spec["model"], spec["candidate"]),
            expected_experts=C.MODELS[spec["model"]]["n_experts"],
            expected_top_k=C.MODELS[spec["model"]]["top_k"],
        )
        pairs[spec["label"]]["condition"] = spec["condition"]
    for spec in C.CHECKPOINT_PAIRS:
        pairs[spec["label"]] = pair_metrics(
            captures,
            label=spec["label"],
            reference=spec["reference"],
            candidate=spec["candidate"],
            expected_experts=spec["n_experts"],
            expected_top_k=spec["top_k"],
        )
        pairs[spec["label"]]["role"] = spec["role"]
    for spec in C.FAMILY_PAIRS:
        pairs[spec["label"]] = pair_metrics(
            captures,
            label=spec["label"],
            reference=spec["reference"],
            candidate=spec["candidate"],
            expected_experts=spec["n_experts"],
            expected_top_k=spec["top_k"],
        )
        pairs[spec["label"]]["role"] = spec["role"]

    for pair in pairs.values():
        if pair.get("status") == "ok":
            print_pair_table(pair)
        else:
            print(f"\n{pair['label']}  SKIP {pair.get('skip_reason')}")

    print_summary(models, pairs)
    cal = calibrate(captures, pairs)
    print_calibration(cal)
    verdict = decide_verdict(cal)
    ctx = {"models": models, "pairs": pairs, "calibration": cal, "verdict": verdict}
    ctx["derived"] = derived_metrics(models, pairs)
    expectations = evaluate_expectations(ctx)
    print_expectations(expectations)
    print(f"\nverdict: {verdict}")

    payload = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "smoke": "routing-calibration",
        "issue": C.ISSUE,
        "issue_url": C.ISSUE_URL,
        "seed": C.SEED,
        "seq_len": C.SEQ_LEN,
        "domains": list(C.DOMAINS),
        "quant": {
            "bits": C.QUANT_BITS,
            "block": C.QUANT_BLOCK,
            "scheme": "symmetric RTN, quantize-dequantize, block 32 along last dim",
            "source": C.PROBE_SOURCE["quant_path"],
        },
        "probe_source": {
            **C.PROBE_SOURCE,
            "config_sha256": sha256_file(REPO_ROOT / C.PROBE_SOURCE["config_path"]),
            "quant_sha256": sha256_file(REPO_ROOT / C.PROBE_SOURCE["quant_path"]),
        },
        "threshold_rule": C.THRESHOLD_RULE,
        "verdict_rules": C.VERDICT_RULES,
        "verdict": verdict,
        "populations": {
            "same_model_requantized": {
                "pairs": [p["label"] for p in C.REQUANT_PAIRS],
            },
            "same_architecture_different_checkpoint": {
                "pairs": [p["label"] for p in C.CHECKPOINT_PAIRS],
            },
            "different_family": {
                "pairs": [p["label"] for p in C.FAMILY_PAIRS],
                "skipped": list(C.FAMILY_SKIP_NOTES),
            },
        },
        "models": models,
        "pairs": pairs,
        "calibration": cal,
        "derived": ctx["derived"],
        "expectations": expectations,
        "fingerprint_schema": {
            "fields": [
                "label",
                "repo_id",
                "revision",
                "variant",
                "model_dtype",
                "router_precision",
                "quant_type",
                "quant_bits",
                "quant_block",
                "n_layers",
                "rows_scored",
                "seq_len",
                "probe_input_sha256",
                "top_k",
                "n_experts",
                "sha256",
            ],
            "note": (
                "router precision, quant type and probe identity travel with every "
                "fingerprint, so comparability is checkable from the fingerprint "
                "alone: compare only when n_experts, top_k, seq_len, "
                "probe_input_sha256, model_dtype and router_precision match; "
                "experts_q4 keeps the router at model dtype, all_q4 records "
                "router_precision='q4_block32', and a base fingerprint has "
                "quant_type='none'."
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threads": args.threads,
        },
        "budgets": {
            "ram_gb": C.RAM_BUDGET_GB,
            "max_weight_gb": C.MAX_WEIGHT_GB,
            "total_time_s": C.TOTAL_TIME_BUDGET_S,
            "per_model_timeout_s": C.PER_MODEL_TIMEOUT_S,
        },
        "max_model_peak_rss_gb": max(
            (m.get("peak_rss_gb", 0.0) for m in models.values()), default=0.0
        ),
        "parent_peak_rss_gb": round(peak_rss_gb(), 3),
        "total_elapsed_s": round(time.perf_counter() - t_start, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}  total {payload['total_elapsed_s']}s")

    # The archived rule must be the object decide_verdict walked, not a copy.
    assert payload["verdict_rules"] is VERDICT_RULES, "recorded rule != executed rule"

    ran = [pair for pair in pairs.values() if pair.get("status") == "ok"]
    if not ran:
        print("FAIL: no population produced routing measurements")
        sys.exit(1)
    if cal["primary"].get("status") != "ok":
        print(
            "FAIL: primary calibration unavailable, so the verdict is the "
            f"registered fallback {verdict!r}: {cal['primary'].get('skip_reason')}"
        )
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
