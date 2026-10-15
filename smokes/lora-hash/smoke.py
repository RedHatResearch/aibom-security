"""LoRA-Hash smoke — smokes/lora-hash/ · ICME 2026 · #26 survey.

From-scratch reimplementation of **LoRA-Hash** (He, Chen, Li, Feng, Ji,
"LoRA-Hash: Unveiling Model Identity from Heavy-Tailed Weight Distributions",
ICME 2026; camera PDF https://nesa.zju.edu.cn/download/icme2026__camera.pdf).
No official code exists and the public camera PDF withholds Appendices B/C/G,
so three gaps are filled here and pinned (see README.md): segment count
N=64 (hash length L=128, the paper's stated minimum for FPR < 1e-6), tau
calibration to a 1% FPR on unrelated-adapter negatives (the paper's own
protocol; tau is never hardcoded), and the attack severities.

Pipeline per adapter: per module dW = B @ A; discard modules with
max|dW| < epsilon; per-module z-score normalization; concatenate retained
modules in sorted-module order; N+1 equal contiguous segments; adjacent
max/min comparison bits per stream; H = concat(b_max, b_min), L = 2N = 128.
Matching: normalized Hamming distance D; same-adapter/plagiarized iff
D <= tau.

Corpus: ProvenanceBench has no LoRA-adapter pairs (feeds #48), so the corpus
is self-generated. Default mode is synthetic-only and offline (--selftest);
--corpus real derives pseudo-adapters (rank-8 SVD truncation of the real
W_subject - W_reference delta over q_proj/v_proj) from the pinned pbench
pairs pb-smoke-qwen3-sft and pb-smoke-smoltulu.

Stdout is a single JSON document; progress goes to stderr. CPU-only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from safetensors.numpy import load_file as st_load_file
from safetensors.numpy import save_file as st_save_file
from scipy.stats import rankdata

SMOKE_DIR = Path(__file__).resolve().parent
SMOKES_ROOT = SMOKE_DIR.parent
sys.path.insert(0, str(SMOKES_ROOT))

# Method constants. N (segments-1), epsilon and the 1% FPR target are the
# pinned gap-fills for the withheld appendices; see README.md.
N_SEGMENTS = 64  # N: N+1 segments -> 2N comparison bits
HASH_BITS = 2 * N_SEGMENTS  # L = 128, the paper's stated minimum for FPR < 1e-6
EPSILON = 1e-4  # layer-discard threshold on max|dW|
TARGET_FPR = 0.01  # tau calibration target on negative pairs
RANK = 8  # pseudo-adapter rank for the real corpus

# Synthetic-corpus constants (design-validated; see README "Corpus").
N_LAYERS = 28
Q_IN, V_IN, OUT_DIM = 32, 16, 32
DECAY_SIGMA = 0.8  # per-module spectral decay alpha ~ exp(0.8 * N(0,1))
SCALE_SIGMA = 0.6  # per-module heavy-tailed scale ~ exp(0.6 * N(0,1))
N_DEAD_MODULES = 2  # near-zero modules injected to exercise the epsilon discard
ATTACK_SEED_BASE = 10_000
SPLIT_SEED = 20261012  # negative cal/eval split (fixed, shared across pipelines)

# Attack grid. Everything except finetune_drift is the issue-spec severity
# list (Appendix C is withheld, so severities are ours); finetune_drift is
# this smoke's documented addition — the only attack family that perturbs
# *relative* module scales, which per-module Z-norm exists to absorb (without
# it the Z-norm ablation cannot show degradation; see README).
ATTACK_SPECS: tuple[tuple[str, float], ...] = (
    ("gaussian_noise", 0.05),
    ("gaussian_noise", 0.1),
    ("gaussian_noise", 0.2),
    ("magnitude_prune", 0.10),
    ("magnitude_prune", 0.30),
    ("magnitude_prune", 0.50),
    ("rank_truncate", 2),  # keep r/2 singular directions of dW
    ("rank_truncate", 4),  # keep r/4
    ("reparameterize", 0.5),  # A -> cA, B -> B/c
    ("reparameterize", 2.0),
    ("global_rescale", 0.7),
    ("global_rescale", 1.5),
    ("finetune_drift", 0.5),
    ("finetune_drift", 1.0),
)

# Ablation pipelines: the full method, the paper's Raw-Hash-without-Z-norm
# variant, and Concat-Hash (no dW reconstruction). Closed-set sanity targets
# from the paper: full 0.9166 AUC, no-Z-norm 0.8871, concat-only 0.55 — we
# check ordering/direction on our corpus, not the absolute values.
PIPELINES: tuple[tuple[str, dict[str, bool]], ...] = (
    ("full", {}),
    ("no_z_norm", {"znorm": False}),
    ("concat_only", {"reconstruct": False}),
)

# module name -> (A [r, in], B [out, r]); dW = B @ A.
Factors = dict[str, tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# Pure scoring core — no I/O, no fixture knowledge.
# ---------------------------------------------------------------------------


def lora_hash(
    factors: Factors,
    *,
    n_segments: int = N_SEGMENTS,
    epsilon: float = EPSILON,
    znorm: bool = True,
    reconstruct: bool = True,
) -> tuple[np.ndarray, int]:
    """LoRA-Hash of one adapter. Returns (bits int8[2N], dropped-module count).

    reconstruct=False is the Concat-Hash ablation (hash the raw A/B factors,
    no dW reconstruction, no epsilon discard — there is no dW to range-check).
    znorm=False is the Raw-Hash-without-Z-norm ablation.
    """
    chunks: list[np.ndarray] = []
    dropped = 0
    for name in sorted(factors):
        a_mat, b_mat = factors[name]
        if reconstruct:
            v = (b_mat @ a_mat).ravel()
            if np.max(np.abs(v)) < epsilon:
                dropped += 1
                continue
        else:
            v = np.concatenate([a_mat.ravel(), b_mat.ravel()])
        if znorm:
            std = v.std()
            v = (v - v.mean()) / (std if std > 1e-12 else 1.0)
        chunks.append(v)
    if not chunks:
        raise ValueError("adapter has no retained modules")
    big_v = np.concatenate(chunks)
    n_seg = n_segments + 1
    if big_v.size < n_seg:
        raise ValueError(f"hash vector too short ({big_v.size}) for {n_seg} segments")
    edges = np.linspace(0, big_v.size, n_seg + 1).astype(int)
    seg_max = np.array([big_v[edges[i] : edges[i + 1]].max() for i in range(n_seg)])
    seg_min = np.array([big_v[edges[i] : edges[i + 1]].min() for i in range(n_seg)])
    bits = np.concatenate(
        [
            (seg_max[:-1] > seg_max[1:]).astype(np.int8),
            (seg_min[:-1] > seg_min[1:]).astype(np.int8),
        ]
    )
    return bits, dropped


def hamming(h1: np.ndarray, h2: np.ndarray) -> float:
    """Normalized Hamming distance D = mismatches / L."""
    if h1.size != h2.size:
        raise ValueError(f"hash length mismatch: {h1.size} vs {h2.size}")
    return float(np.count_nonzero(h1 != h2) / h1.size)


def calibrate_tau(negative_distances: list[float], target_fpr: float = TARGET_FPR) -> float:
    """Paper protocol: tau at the target FPR over unrelated-adapter negatives."""
    if not negative_distances:
        raise ValueError("no negative pairs to calibrate tau")
    return float(np.quantile(negative_distances, target_fpr))


def auc_score(positive_d: list[float], negative_d: list[float]) -> float:
    """Mann-Whitney AUC: P(D_pos < D_neg) + 0.5 P(D_pos == D_neg).
    Positives are same-lineage pairs and must sit at SMALLER hash distance
    than negatives, so the upward Mann-Whitney statistic (P(D_pos > D_neg)
    + 0.5 on ties) is subtracted from 1.
    """
    if not positive_d or not negative_d:
        return float("nan")
    ranks = rankdata(np.asarray(positive_d + negative_d))
    n_pos = len(positive_d)
    n_neg = len(negative_d)
    up = (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(1.0 - up)


# ---------------------------------------------------------------------------
# Adapter I/O — PEFT adapter_model.safetensors layout or .npz.
# ---------------------------------------------------------------------------

_PEFT_KEY = re.compile(r"(?:base_model\.model\.)?(?P<module>.+)\.lora_(?P<side>[AB])\.weight")
_NPZ_KEY = re.compile(r"(?P<module>.+)\.lora_(?P<side>[AB])")


def _pair_up(raw: dict[tuple[str, str], np.ndarray], source: str) -> Factors:
    halves: dict[str, dict[str, np.ndarray]] = {}
    for (module, side), tensor in raw.items():
        halves.setdefault(module, {})[side] = np.asarray(tensor)
    factors: Factors = {}
    for module, sides in halves.items():
        if "A" not in sides or "B" not in sides:
            raise KeyError(f"{source}: module {module!r} missing lora_A or lora_B")
        factors[module] = (sides["A"], sides["B"])
    if not factors:
        raise ValueError(f"{source}: no LoRA A/B factor pairs found")
    return factors


def load_adapter(path: Path) -> Factors:
    """Load {module: (A, B)} from a PEFT adapter_model.safetensors (keys
    `...{module}.lora_A.weight` / `...{module}.lora_B.weight`, optional
    `base_model.model.` prefix) or an .npz with `{module}.lora_A` /
    `{module}.lora_B` keys."""
    if path.suffix == ".npz":
        with np.load(path) as archive:
            raw = {}
            for key in archive.files:
                match = _NPZ_KEY.fullmatch(key)
                if match:
                    raw[(match["module"], match["side"])] = archive[key]
        return _pair_up(raw, str(path))
    flat = st_load_file(str(path))
    raw = {}
    for key, tensor in flat.items():
        match = _PEFT_KEY.fullmatch(key)
        if match:
            raw[(match["module"], match["side"])] = tensor
    return _pair_up(raw, str(path))


# ---------------------------------------------------------------------------
# pbench manifest loader (provenancebench/pairs.yaml — orchestrator's
# manifest; model ids and pinned revisions are read at runtime, not inlined).
# ---------------------------------------------------------------------------


def find_pairs_yaml() -> Path:
    candidates = [SMOKE_DIR.parents[1] / "provenancebench" / "pairs.yaml"]
    cwd = Path.cwd()
    candidates += [base / "provenancebench" / "pairs.yaml" for base in (cwd, *cwd.parents)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("provenancebench/pairs.yaml not found near the smoke or cwd")


def load_pair(pair_id: str) -> dict:
    with open(find_pairs_yaml()) as handle:
        manifest = yaml.safe_load(handle)
    for entry in manifest.get("pairs", []):
        if entry.get("id") == pair_id:
            return entry
    raise KeyError(f"pair id {pair_id!r} not found in provenancebench/pairs.yaml")


# ---------------------------------------------------------------------------
# Corpus generation.
# ---------------------------------------------------------------------------


@dataclass
class CorpusRecord:
    rec_id: str
    kind: str  # "anchor" | "variant"
    anchor_id: str
    attack: str
    factors: Factors


@dataclass
class Corpus:
    mode: str
    records: list[CorpusRecord]
    provenance: dict


def attack_seed(master_seed: int, anchor_index: int) -> int:
    return ATTACK_SEED_BASE + master_seed * 137 + anchor_index * 17


def attack_name(kind: str, severity: float) -> str:
    if kind == "rank_truncate":
        return f"{kind}_r_over_{int(severity)}"  # e.g. r_over_2 = keep r/2
    return f"{kind}_{severity:g}"


def synth_adapter(
    rng: np.random.Generator,
    *,
    n_layers: int = N_LAYERS,
    rank: int = RANK,
    scale_sigma: float = SCALE_SIGMA,
    n_dead: int = N_DEAD_MODULES,
) -> Factors:
    """Random-factor adapter with the heavy-tailed per-module weight
    distributions the paper hashes: per-module spectral decay in A and a
    lognormal per-module output scale. The first `n_dead` modules in sorted
    order are squashed to ~1e-7 to exercise the epsilon discard path."""
    mods: Factors = {}
    for layer in range(n_layers):
        for role, d_in in (("q_proj", Q_IN), ("v_proj", V_IN)):
            alpha = float(np.exp(DECAY_SIGMA * rng.standard_normal()))
            decay = (np.arange(1, rank + 1) ** -alpha)[:, None]
            a_mat = decay * rng.standard_normal((rank, d_in)) / np.sqrt(d_in)
            b_mat = rng.standard_normal((OUT_DIM, rank)) / np.sqrt(rank)
            scale = float(np.exp(scale_sigma * rng.standard_normal()))
            mods[f"model.layers.{layer}.self_attn.{role}"] = (a_mat * scale, b_mat)
    for name in sorted(mods)[:n_dead]:
        a_mat, b_mat = mods[name]
        mods[name] = (a_mat * 1e-7, b_mat * 1e-7)
    return mods


def apply_attack(factors: Factors, kind: str, severity: float, seed: int) -> Factors:
    """One attacked variant of an adapter (our severities; Appendix C is
    withheld). Iterates in insertion order so RNG draws are deterministic."""
    rng_atk = np.random.default_rng(seed)
    out: Factors = {}
    for name, (a_mat, b_mat) in factors.items():
        a_new, b_new = a_mat.copy(), b_mat.copy()
        if kind == "gaussian_noise":
            a_new = a_new + severity * a_new.std() * rng_atk.standard_normal(a_new.shape)
            b_new = b_new + severity * b_new.std() * rng_atk.standard_normal(b_new.shape)
        elif kind == "magnitude_prune":
            for mat in (a_new, b_new):
                threshold = np.quantile(np.abs(mat), severity)
                mat[np.abs(mat) < threshold] = 0.0
        elif kind == "rank_truncate":
            keep = max(1, a_new.shape[0] // int(severity))
            u_mat, s_val, vt_mat = np.linalg.svd(b_new @ a_new, full_matrices=False)
            root = np.sqrt(s_val[:keep])
            a_new = np.diag(root) @ vt_mat[:keep]
            b_new = u_mat[:, :keep] @ np.diag(root)
        elif kind == "reparameterize":
            a_new, b_new = a_new * severity, b_new / severity
        elif kind == "global_rescale":
            a_new = a_new * severity
        elif kind == "finetune_drift":
            # continued-finetune simulation: per-module lognormal scale drift
            # plus a small additive update (this smoke's addition; see README)
            a_new = a_new * float(np.exp(severity * rng_atk.standard_normal()))
            a_new = a_new + 0.01 * a_new.std() * rng_atk.standard_normal(a_new.shape)
        else:
            raise ValueError(f"unknown attack kind: {kind}")
        out[name] = (a_new, b_new)
    return out


def build_synthetic_corpus(*, seed: int, n_anchors: int) -> Corpus:
    """Offline corpus: n_anchors unrelated random adapters, each with the full
    attack grid as variants. Positives = anchor vs self/variants; negatives =
    cross-anchor pairs."""
    rng = np.random.default_rng(seed)
    records: list[CorpusRecord] = []
    for index in range(n_anchors):
        factors = synth_adapter(rng)
        anchor_id = f"a{index:02d}"
        records.append(CorpusRecord(anchor_id, "anchor", anchor_id, "identity", factors))
        for kind, severity in ATTACK_SPECS:
            atk = attack_name(kind, severity)
            records.append(
                CorpusRecord(
                    f"{anchor_id}:{atk}",
                    "variant",
                    anchor_id,
                    atk,
                    apply_attack(factors, kind, severity, attack_seed(seed, index)),
                )
            )
    provenance = {
        "mode": "synthetic",
        "n_anchors": n_anchors,
        "modules_per_anchor": N_LAYERS * 2,
        "rank": RANK,
        "dead_modules_injected": N_DEAD_MODULES,
        "attacks": [attack_name(k, s) for k, s in ATTACK_SPECS],
        "note": (
            "offline random-factor adapters with heavy-tailed per-module scales; "
            "ProvenanceBench has no LoRA-adapter pairs (feeds #48), see README"
        ),
    }
    return Corpus("synthetic", records, provenance)


def pseudo_adapter_from_pair(
    pair_id: str, *, rank: int, cache_dir: Path | None
) -> tuple[Factors, dict]:
    """REAL lineage in adapter form: rank-r SVD truncation of
    W_subject - W_reference over q_proj/v_proj for a pinned pbench pair."""
    entry = load_pair(pair_id)
    # Real-corpus path only; --selftest never imports the Hub stack.
    from _lib.hub import load_tensors
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(
        entry["reference"],
        filename="config.json",
        cache_dir=cache_dir,
        revision=entry["reference_rev"],
    )
    with open(cfg_path) as handle:
        num_layers = int(json.load(handle)["num_hidden_layers"])
    names = {
        f"model.layers.{i}.self_attn.{role}.weight"
        for i in range(num_layers)
        for role in ("q_proj", "v_proj")
    }
    _log(f"[{pair_id}] loading subject {entry['subject']}@{entry['subject_rev'][:8]}")
    w_subject = load_tensors(
        entry["subject"], names, cache_dir=cache_dir, revision=entry["subject_rev"]
    )
    _log(f"[{pair_id}] loading reference {entry['reference']}@{entry['reference_rev'][:8]}")
    w_reference = load_tensors(
        entry["reference"], names, cache_dir=cache_dir, revision=entry["reference_rev"]
    )
    factors: Factors = {}
    for name in sorted(names):
        delta = w_subject[name] - w_reference[name]
        u_mat, s_val, vt_mat = np.linalg.svd(delta, full_matrices=False)
        keep = min(rank, s_val.size)
        root = np.sqrt(s_val[:keep])
        factors[name.removesuffix(".weight")] = (
            np.diag(root) @ vt_mat[:keep],
            u_mat[:, :keep] @ np.diag(root),
        )
    provenance = {
        key: entry[key] for key in ("id", "subject", "subject_rev", "reference", "reference_rev")
    }
    return factors, provenance


def build_real_corpus(
    cache_dir: Path | None, *, seed: int = 0, rank: int = RANK, n_synthetic: int = 8
) -> Corpus:
    """Real-delta corpus: pseudo-adapters from the two pinned pbench SFT pairs
    (pb-smoke-qwen3-sft anchor + pb-smoke-smoltulu cross-family negative),
    plus synthetic random-factor adapters as additional negatives."""
    records: list[CorpusRecord] = []
    pair_provenance = []
    for anchor_index, pair_id in enumerate(("pb-smoke-qwen3-sft", "pb-smoke-smoltulu")):
        factors, pair_prov = pseudo_adapter_from_pair(pair_id, rank=rank, cache_dir=cache_dir)
        pair_provenance.append(pair_prov)
        anchor_id = f"real:{pair_id}"
        records.append(CorpusRecord(anchor_id, "anchor", anchor_id, "identity", factors))
        for kind, severity in ATTACK_SPECS:
            atk = attack_name(kind, severity)
            records.append(
                CorpusRecord(
                    f"{anchor_id}:{atk}",
                    "variant",
                    anchor_id,
                    atk,
                    apply_attack(factors, kind, severity, attack_seed(seed, anchor_index)),
                )
            )
    rng = np.random.default_rng(seed + 1)
    for index in range(n_synthetic):
        anchor_id = f"synth:{index:02d}"
        records.append(CorpusRecord(anchor_id, "anchor", anchor_id, "identity", synth_adapter(rng)))
    provenance = {
        "mode": "real",
        "pseudo_adapter_construction": (
            f"rank-{rank} SVD truncation of W_subject - W_reference over q_proj/v_proj"
        ),
        "pairs": pair_provenance,
        "synthetic_negative_anchors": n_synthetic,
        "attacks": [attack_name(k, s) for k, s in ATTACK_SPECS],
        "note": (
            "ProvenanceBench has no LoRA-adapter pairs; the pseudo-adapters carry the "
            "real pair lineage in adapter form (see README, feeds issue #48)"
        ),
    }
    return Corpus("real", records, provenance)


# ---------------------------------------------------------------------------
# Evaluation.
# ---------------------------------------------------------------------------


def evaluate_corpus(corpus: Corpus, *, n_segments: int = N_SEGMENTS) -> dict:
    """Score the corpus under all three pipelines (full method + the two
    ablations). Each pipeline calibrates its own tau on a fixed half of the
    negative pairs; recall/FPR/AUC are reported per pipeline."""
    anchors = [rec for rec in corpus.records if rec.kind == "anchor"]
    anchor_pairs = [
        (anchors[i], anchors[j]) for i in range(len(anchors)) for j in range(i + 1, len(anchors))
    ]
    split_rng = np.random.default_rng(SPLIT_SEED)
    perm = split_rng.permutation(len(anchor_pairs))
    cal_idx = set(perm[: len(anchor_pairs) // 2].tolist())

    results: dict[str, dict] = {}
    for pipe_name, kwargs in PIPELINES:
        t0 = time.perf_counter()
        hashes: dict[str, np.ndarray] = {}
        dropped_total = 0
        hash_bits = 0
        for rec in corpus.records:
            bits, dropped = lora_hash(rec.factors, n_segments=n_segments, **kwargs)
            hashes[rec.rec_id] = bits
            dropped_total += dropped
            hash_bits = int(bits.size)
        hash_seconds = time.perf_counter() - t0

        positives = [
            {
                "anchor": a.rec_id,
                "variant": a.rec_id,
                "attack": "identity",
                "D_hamming": hamming(hashes[a.rec_id], hashes[a.rec_id]),
            }
            for a in anchors
        ]
        positives += [
            {
                "anchor": rec.anchor_id,
                "variant": rec.rec_id,
                "attack": rec.attack,
                "D_hamming": hamming(hashes[rec.anchor_id], hashes[rec.rec_id]),
            }
            for rec in corpus.records
            if rec.kind == "variant"
        ]
        negatives = [
            {
                "a": x.rec_id,
                "b": y.rec_id,
                "D_hamming": hamming(hashes[x.rec_id], hashes[y.rec_id]),
                "split": "cal" if index in cal_idx else "eval",
            }
            for index, (x, y) in enumerate(anchor_pairs)
        ]

        pos_d = [p["D_hamming"] for p in positives]
        cal_d = [n["D_hamming"] for n in negatives if n["split"] == "cal"]
        eval_d = [n["D_hamming"] for n in negatives if n["split"] == "eval"]
        all_neg_d = [n["D_hamming"] for n in negatives]
        tau = calibrate_tau(cal_d)
        for p in positives:
            p["match"] = p["D_hamming"] <= tau
        for n in negatives:
            n["match"] = n["D_hamming"] <= tau

        drift_d = [p["D_hamming"] for p in positives if p["attack"].startswith("finetune_drift")]
        per_attack: dict[str, list[float]] = {}
        for p in positives:
            if p["attack"] != "identity":
                per_attack.setdefault(p["attack"], []).append(p["D_hamming"])

        results[pipe_name] = {
            "tau": tau,
            "recall_at_1pct_fpr": float(np.mean([d <= tau for d in pos_d])),
            "fpr_eval": float(np.mean([d <= tau for d in eval_d])) if eval_d else None,
            "auc": auc_score(pos_d, all_neg_d),
            "mean_drift_D": float(np.mean(drift_d)) if drift_d else None,
            "n_positives": len(positives),
            "n_negatives": len(negatives),
            "n_negative_cal": len(cal_d),
            "n_negative_eval": len(eval_d),
            "hash_bits": hash_bits,
            "dropped_modules_total": dropped_total,
            "per_attack_mean_D": {k: float(np.mean(v)) for k, v in sorted(per_attack.items())},
            "positives": positives,
            "negatives": negatives,
            "hash_seconds": round(hash_seconds, 3),
        }
    return results


def adapter_io_roundtrip(factors: Factors) -> bool:
    """The adapter loaders must reproduce the in-memory hash bit-exactly
    through both supported on-disk layouts (.npz and PEFT safetensors)."""
    reference = lora_hash(factors)[0]
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        npz_path = tmpdir / "adapter.npz"
        np.savez(
            npz_path,
            **{
                f"{module}.lora_{side}": tensor
                for module, (a_mat, b_mat) in factors.items()
                for side, tensor in (("A", a_mat), ("B", b_mat))
            },
        )
        h_npz = lora_hash(load_adapter(npz_path))[0]
        st_path = tmpdir / "adapter_model.safetensors"
        st_save_file(
            {
                f"base_model.model.{module}.lora_{side}.weight": np.ascontiguousarray(tensor)
                for module, (a_mat, b_mat) in factors.items()
                for side, tensor in (("A", a_mat), ("B", b_mat))
            },
            str(st_path),
        )
        h_st = lora_hash(load_adapter(st_path))[0]
    return bool(np.array_equal(reference, h_npz) and np.array_equal(reference, h_st))


def selftest_checks(results: dict) -> dict[str, bool]:
    """Pre-registered acceptance criteria for --selftest (see README)."""
    full = results["full"]
    no_z = results["no_z_norm"]
    concat = results["concat_only"]
    return {
        "hash_length_is_128": full["hash_bits"] == HASH_BITS,
        "layer_discard_exercised": full["dropped_modules_total"] > 0,
        "recall_at_least_0.9": full["recall_at_1pct_fpr"] >= 0.9,
        "negatives_above_tau": full["fpr_eval"] is not None and full["fpr_eval"] <= 0.10,
        "ablation_concat_degrades_sharply": (
            concat["recall_at_1pct_fpr"] <= full["recall_at_1pct_fpr"] - 0.05
            and concat["auc"] <= full["auc"] - 0.05
        ),
        "ablation_no_z_norm_degrades": (
            no_z["auc"] <= full["auc"] + 1e-9 and no_z["mean_drift_D"] >= 5.0 * full["mean_drift_D"]
        ),
    }


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA-Hash smoke (smokes/lora-hash/)")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="offline synthetic corpus + acceptance checks; exits 0 on pass",
    )
    parser.add_argument(
        "--corpus",
        choices=("synthetic", "real"),
        default="synthetic",
        help="corpus mode (default: synthetic, offline)",
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="HF cache directory")
    parser.add_argument("--seed", type=int, default=0, help="corpus master seed")
    parser.add_argument("--anchors", type=int, default=12, help="synthetic anchor count")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="hash one standalone adapter file (.safetensors PEFT or .npz) and exit",
    )
    args = parser.parse_args()

    if args.adapter is not None:
        factors = load_adapter(args.adapter)
        bits, dropped = lora_hash(factors)
        print(
            json.dumps(
                {
                    "adapter": str(args.adapter),
                    "modules": len(factors),
                    "hash_bits": int(bits.size),
                    "dropped_modules": dropped,
                    "hash_hex": np.packbits(bits.astype(np.uint8)).tobytes().hex(),
                }
            )
        )
        return

    if args.selftest and args.corpus != "synthetic":
        parser.error("--selftest runs offline on the synthetic corpus only")

    t_start = time.perf_counter()
    corpus = (
        build_real_corpus(args.cache_dir, seed=args.seed)
        if args.corpus == "real"
        else build_synthetic_corpus(seed=args.seed, n_anchors=args.anchors)
    )
    corpus_seconds = time.perf_counter() - t_start

    results = evaluate_corpus(corpus)
    full = results["full"]

    output = {
        "smoke": "lora-hash",
        "paper": (
            "LoRA-Hash: Unveiling Model Identity from Heavy-Tailed Weight Distributions "
            "(He, Chen, Li, Feng, Ji; ICME 2026); no official code — reimplemented from "
            "the camera PDF, withheld-appendix gaps filled as documented in README.md"
        ),
        "params": {
            "n_segments": N_SEGMENTS,
            "hash_bits": HASH_BITS,
            "epsilon": EPSILON,
            "target_fpr": TARGET_FPR,
            "rank": RANK,
            "seed": args.seed,
        },
        "corpus": corpus.provenance,
        "tau": full["tau"],
        "recall_at_1pct_fpr": full["recall_at_1pct_fpr"],
        "fpr_eval": full["fpr_eval"],
        "pairs": full["positives"],
        "negatives": full["negatives"],
        "ablations": {
            name: {k: v for k, v in res.items() if k not in ("positives", "negatives")}
            for name, res in results.items()
        },
        "timing": {
            "corpus_seconds": round(corpus_seconds, 3),
            "hash_seconds": {name: res["hash_seconds"] for name, res in results.items()},
            "total_seconds": round(time.perf_counter() - t_start, 3),
        },
    }

    if args.selftest:
        checks = selftest_checks(results)
        checks["adapter_io_roundtrip"] = adapter_io_roundtrip(corpus.records[0].factors)
        passed = all(checks.values())
        output["selftest"] = {"checks": checks, "pass": passed}
        print(json.dumps(output, indent=2, default=float))
        sys.exit(0 if passed else 1)

    print(json.dumps(output, indent=2, default=float))


if __name__ == "__main__":
    main()
