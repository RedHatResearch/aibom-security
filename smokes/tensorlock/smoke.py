"""TensorLock smoke — smokes/tensorlock/ · doi:10.1145/3832119 · #26 survey.

Pairwise reimplementation of the white-box static signals of TensorLock
(Wu et al., ISSTA 2026): Eq. 1 s_MSA connectivity gate, Eq. 3 QPS quant
provenance, Eq. 7 spectral-entropy FT direction (Eq. 6 FT distance as
context). Clustering (HDBSCAN), merging (IPS/LASSO/TPS) and the MDST tree
are corpus-level and stay out; see README.md for scope and deviations.

Reimplemented from the paper equations only; the authors' artifact has no
license and was used strictly as a read-only behavioral cross-check.

Stdout is a single JSON document; progress goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from gguf import GGUFReader, dequantize
from huggingface_hub import hf_hub_download
from scipy.stats import kurtosis, skew, spearmanr

SMOKE_DIR = Path(__file__).resolve().parent
SMOKES_ROOT = SMOKE_DIR.parent
sys.path.insert(0, str(SMOKE_DIR))
sys.path.insert(0, str(SMOKES_ROOT))

from _lib.hub import (  # noqa: E402
    attn_tensor_names,
    load_config,
    load_tensors,
    mlp_tensor_names,
    parse_model_config,
)
from config import DENSE_PAIRS, QUANT_FIXTURES, REVISIONS, DensePair, QuantFixture  # noqa: E402

# Paper §3.4.1 / RQ6: connectivity threshold tau_conn and QPS bin count |B|.
TAU_CONN = 0.6
QPS_BINS = 512
# Stride-subsampling cap per layer vector (artifact's NMI_SAMPLE_POINTS);
# identical indices for both sides, so joint alignment is preserved.
NMI_SAMPLE_CAP = 500_000

MSA_ROLES = ("q", "k", "v", "o")
FFN_ROLES = ("gate", "up", "down")
ALL_ROLES = MSA_ROLES + FFN_ROLES

# Matrices: role -> per-layer 2-D float32 arrays (HF layout: out x in).
Matrices = dict[str, list[np.ndarray]]


# ---------------------------------------------------------------------------
# Pure scoring core — no I/O, no fixture knowledge. Takes plain Matrices
# dicts; a later ProvenanceBench detector reuses these directly.
# ---------------------------------------------------------------------------


def _safe_spearman(v1: np.ndarray, v2: np.ndarray) -> float:
    """Spearman rho; NaN (constant vector) counts as 0, artifact convention."""
    if v1.size < 2 or v2.size < 2:
        return 0.0
    rho = spearmanr(v1, v2).statistic
    return 0.0 if np.isnan(rho) else float(rho)


def msa_features(mats: Matrices) -> dict[str, dict[str, np.ndarray]]:
    """Eq. 1 features: per MSA role, per-layer std / skew / kurtosis of the
    flattened weight matrix. Returns role -> feature -> vector over layers."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for role in MSA_ROLES:
        layers = mats.get(role, [])
        if not layers:
            continue
        std = np.array([float(np.std(w)) for w in layers])
        sk = np.array([float(skew(w.ravel())) for w in layers])
        ku = np.array([float(kurtosis(w.ravel())) for w in layers])
        out[role] = {"std": std, "skew": sk, "kurtosis": ku}
    return out


def msa_similarity(
    feat_i: dict[str, dict[str, np.ndarray]], feat_j: dict[str, dict[str, np.ndarray]]
) -> dict:
    """Eq. 1 s_MSA: mean over 4 MSA roles of the mean over {std, skew,
    kurtosis} Spearman correlations, truncated to L_min layers."""
    per_role: dict[str, float] = {}
    for role in MSA_ROLES:
        if role not in feat_i or role not in feat_j:
            continue
        corrs = []
        for feat in ("std", "skew", "kurtosis"):
            vi, vj = feat_i[role][feat], feat_j[role][feat]
            n = min(vi.size, vj.size)  # truncate to L_min (paper §3.2)
            corrs.append(_safe_spearman(vi[:n], vj[:n]))
        per_role[role] = float(np.mean(corrs))
    score = float(np.mean(list(per_role.values()))) if per_role else 0.0
    return {"score": score, "per_role": per_role}


def _bin_labels(x: np.ndarray, bins: int) -> np.ndarray:
    """Equal-width bins over the vector's own [min, max] -> labels 0..bins-1."""
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.int64)
    edges = np.linspace(lo, hi, bins + 1)[1:-1]
    return np.digitize(x, edges)


def nmi(p: np.ndarray, q: np.ndarray, bins: int) -> float:
    """2·I(P;Q) / (H(P) + H(Q)) on element-aligned binnings of p and q
    (Eq. 3 normalization; matches sklearn's arithmetic NMI)."""
    pl = _bin_labels(p, bins)
    ql = _bin_labels(q, bins)
    n = pl.size
    joint = np.bincount(pl * bins + ql, minlength=bins * bins).astype(np.float64) / n
    j = joint.reshape(bins, bins)
    px = j.sum(axis=1)
    qx = j.sum(axis=0)

    def _h(v: np.ndarray) -> float:
        nz = v[v > 0]
        return float(-(nz * np.log(nz)).sum())

    h_p, h_q = _h(px), _h(qx)
    if h_p + h_q == 0.0:
        return 0.0
    denom = px[:, None] * qx[None, :]
    nz = j > 0
    mi = float((j[nz] * (np.log(j[nz]) - np.log(denom[nz]))).sum())
    return 2.0 * mi / (h_p + h_q)


def _stride_indices(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    step = n // cap
    return np.arange(0, n, step)[:cap]


def _layer_vector(mats: Matrices, layer: int) -> np.ndarray | None:
    """Flattened 'weight vector of layer l' (paper §3.3): the layer's seven
    projection matrices concatenated."""
    parts = [mats[role][layer].ravel() for role in ALL_ROLES if role in mats]
    return np.concatenate(parts) if parts else None


def _shapes_align(mats_i: Matrices, mats_j: Matrices) -> tuple[bool, str | None]:
    if set(mats_i) != set(mats_j):
        return False, f"role mismatch {sorted(mats_i)} vs {sorted(mats_j)}"
    for role in (r for r in ALL_ROLES if r in mats_i):
        if len(mats_i[role]) != len(mats_j[role]):
            return False, f"layer count mismatch for {role}"
        for li, (wi, wj) in enumerate(zip(mats_i[role], mats_j[role], strict=False)):
            if wi.shape != wj.shape:
                return False, f"shape mismatch {role}[{li}]: {wi.shape} vs {wj.shape}"
    return True, None


def qps_score(
    mats_p: Matrices, mats_c: Matrices, bins: int = QPS_BINS, cap: int = NMI_SAMPLE_CAP
) -> dict:
    """Eq. 3 QPS: arithmetic-normalized MI between the layer weight
    histograms of parent candidate and quantized child, averaged over
    layers (min/max per-layer NMI as context)."""
    ok, reason = _shapes_align(mats_p, mats_c)
    if not ok:
        return {"qps": None, "reason": reason}
    n_layers = min(len(mats_p[r]) for r in mats_p)
    per_layer = []
    for li in range(n_layers):
        vp = _layer_vector(mats_p, li)
        vc = _layer_vector(mats_c, li)
        idx = _stride_indices(vp.size, cap)
        per_layer.append(nmi(vp[idx], vc[idx], bins))
    return {
        "qps": float(np.mean(per_layer)),
        "per_layer_min": float(np.min(per_layer)),
        "per_layer_max": float(np.max(per_layer)),
    }


def spectral_entropy(w: np.ndarray) -> float:
    """H_sigma(W) (Eq. 7 pre-step): Shannon entropy of normalized singular
    values, natural log."""
    s = np.linalg.svd(w, compute_uv=False).astype(np.float64)
    total = s.sum()
    if total <= 0:
        return 0.0
    p = s / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _entropy_table(mats: Matrices) -> dict[str, list[float]]:
    """Per-role per-layer H_sigma over MSA + FFN matrices (Eq. 7 scope)."""
    table: dict[str, list[float]] = {}
    for role in ALL_ROLES:
        if role in mats:
            table[role] = [spectral_entropy(w) for w in mats[role]]
    return table


def ft_direction_score(tab_i: dict[str, list[float]], tab_j: dict[str, list[float]]) -> dict:
    """Eq. 7 S(m_i, m_j) = (1/2L) sum over layers and {MSA, FFN} of
    [H_sigma(W_j) - H_sigma(W_i)]. S > 0 marks i -> j a valid parent -> child
    direction. Also reports the artifact-style exp(H) (effective-rank) form
    and the per-role breakdown for robustness reading."""
    roles = [r for r in ALL_ROLES if r in tab_i and r in tab_j]
    diffs: list[float] = []
    exp_diffs: list[float] = []
    per_role: dict[str, float] = {}
    for role in roles:
        n = min(len(tab_i[role]), len(tab_j[role]))
        d = [tab_j[role][li] - tab_i[role][li] for li in range(n)]
        diffs.extend(d)
        exp_diffs.extend(
            [float(np.exp(tab_j[role][li]) - np.exp(tab_i[role][li])) for li in range(n)]
        )
        per_role[role] = float(np.mean(d))
    s = float(np.mean(diffs)) if diffs else 0.0
    s_exp = float(np.mean(exp_diffs)) if exp_diffs else 0.0
    return {
        "S": s,
        "direction_valid": s > 0,
        "S_effective_rank_form": s_exp,
        "sign_agrees_effective_rank_form": (s > 0) == (s_exp > 0),
        "per_role": per_role,
    }


def ft_distance(mats_i: Matrices, mats_j: Matrices) -> dict:
    """Eq. 6 A_dis = (1/2L) sum_l [||W_MSA,i - W_MSA,j||_2 + ||W_FFN,i - W_FFN,j||_2],
    per-layer block norms with the layer's matrices stacked. Context only:
    the MDST tree that consumes it is out of scope."""
    ok, reason = _shapes_align(mats_i, mats_j)
    if not ok:
        return {"a_dis": None, "reason": reason}
    n_layers = min(len(mats_i[r]) for r in mats_i)
    total = 0.0
    for li in range(n_layers):
        for roles in (MSA_ROLES, FFN_ROLES):
            sq = 0.0
            for role in roles:
                if role in mats_i:
                    d = mats_i[role][li].astype(np.float64) - mats_j[role][li].astype(np.float64)
                    sq += float((d * d).sum())
            total += float(np.sqrt(sq))
    return {"a_dis": total / (2.0 * n_layers)}


# ---------------------------------------------------------------------------
# Hub loading layer (smoke-only; the pure core above never touches I/O).
# ---------------------------------------------------------------------------


_HF_ROLES = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}

_GGUF_ROLES = {  # gguf tensor suffix -> our role key
    "attn_q": "q",
    "attn_k": "k",
    "attn_v": "v",
    "attn_output": "o",
    "ffn_gate": "gate",
    "ffn_up": "up",
    "ffn_down": "down",
}


def load_dense_matrices(repo_id: str, cache_dir: Path | None) -> Matrices:
    """Safetensors projection matrices for one checkpoint, rekeyed to
    role -> per-layer arrays (all seven roles, pinned revision)."""
    config = parse_model_config(repo_id, load_config(repo_id, cache_dir=cache_dir))
    n = config.num_layers
    names = attn_tensor_names(n) | mlp_tensor_names(n)
    tensors = load_tensors(repo_id, names, cache_dir=cache_dir, revision=REVISIONS.get(repo_id))
    out: Matrices = {}
    for role, prefix in _HF_ROLES.items():
        out[role] = [tensors[f"model.layers.{i}.{prefix}.weight"] for i in range(n)]
    return out


def load_gguf_matrices(repo_id: str, filename: str, cache_dir: Path | None) -> Matrices:
    """Dequantized GGUF projection matrices, rekeyed like the dense loader.
    gguf.dequantize returns HF-layout (out x in) arrays element-aligned with
    the source safetensors (verified against Qwen2.5-0.5B Q8_0: corr ~0.99998)."""
    path = hf_hub_download(
        repo_id, filename=filename, cache_dir=cache_dir, revision=REVISIONS.get(repo_id)
    )
    reader = GGUFReader(path)
    raw: dict[str, dict[int, np.ndarray]] = {role: {} for role in _GGUF_ROLES.values()}
    for tensor in reader.tensors:
        name = tensor.name if isinstance(tensor.name, str) else tensor.name.decode()
        m = re.fullmatch(r"blk\.(\d+)\.([a-z_]+)\.weight", name)
        if not m:
            continue
        role = _GGUF_ROLES.get(m.group(2))
        if role is None:
            continue
        w = dequantize(tensor.data, tensor.tensor_type)
        raw[role][int(m.group(1))] = np.ascontiguousarray(w, dtype=np.float32)
    out: Matrices = {}
    for role, layers in raw.items():
        if layers:
            out[role] = [layers[i] for i in range(max(layers) + 1)]
    return out


def used_revision(repo_id: str) -> str | None:
    """Commit sha actually used: every download call passes the pinned
    revision, so the pin is what the hub client resolves by construction."""
    return REVISIONS.get(repo_id)


# ---------------------------------------------------------------------------
# Fixture runners.
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run_dense_pair(pair: DensePair, cache_dir: Path | None) -> dict:
    """Gate the pair with s_MSA, then (if same-lineage) check Eq. 7 direction
    both ways and report Eq. 6 distance as context."""
    t0 = time.perf_counter()
    _log(f"[{pair.label}] loading {pair.parent} (all roles)")
    mats_p = load_dense_matrices(pair.parent, cache_dir)
    _log(f"[{pair.label}] loading {pair.child} (all roles)")
    mats_c = load_dense_matrices(pair.child, cache_dir)

    sim = msa_similarity(msa_features(mats_p), msa_features(mats_c))
    gate_pass = sim["score"] >= TAU_CONN

    entry: dict = {
        "label": pair.label,
        "parent": {"repo_id": pair.parent, "revision": used_revision(pair.parent)},
        "child": {"repo_id": pair.child, "revision": used_revision(pair.child)},
        "relation": pair.relation,
        "note": pair.note,
        "s_msa": sim,
        "gate": "pass" if gate_pass else "fail",
        "gate_threshold": TAU_CONN,
    }

    if gate_pass:
        tab_p = _entropy_table(mats_p)
        tab_c = _entropy_table(mats_c)
        fwd = ft_direction_score(tab_p, tab_c)  # parent -> child
        rev = ft_direction_score(tab_c, tab_p)
        entry["ft_direction"] = {
            "S_parent_to_child": fwd["S"],
            "parent_to_child_valid": fwd["direction_valid"],
            "S_child_to_parent": rev["S"],
            "sign_agrees_effective_rank_form": fwd["sign_agrees_effective_rank_form"],
            "per_role": fwd["per_role"],
        }
        entry["ft_distance"] = ft_distance(mats_p, mats_c)
        if pair.relation == "ft":
            entry["verdict"] = "correct" if fwd["direction_valid"] else "wrong_direction"
        else:
            entry["verdict"] = "gate_pass_on_unrelated"
    else:
        entry["verdict"] = "correct_rejection" if pair.relation == "unrelated" else "gate_miss"
    entry["compute_seconds"] = round(time.perf_counter() - t0, 1)
    _log(f"[{pair.label}] s_MSA={sim['score']:.3f} gate={entry['gate']} verdict={entry['verdict']}")
    return entry


def run_quant_fixture(fx: QuantFixture, cache_dir: Path | None) -> dict:
    """Load the GGUF child, gate each candidate with s_MSA (paper restricts
    QPS candidates to the connectivity cluster), then QPS-argmax over the
    gate-passing candidates."""
    t0 = time.perf_counter()
    _log(f"[{fx.label}] loading {fx.child_repo}/{fx.child_file}")
    child = load_gguf_matrices(fx.child_repo, fx.child_file, cache_dir)
    child_feat = msa_features(child)

    candidates = []
    for cand in fx.candidates:
        _log(f"[{fx.label}] candidate {cand}")
        mats = load_dense_matrices(cand, cache_dir)
        sim = msa_similarity(child_feat, msa_features(mats))
        passed = sim["score"] >= TAU_CONN
        entry = {
            "repo_id": cand,
            "revision": used_revision(cand),
            "s_msa": sim,
            "gate": "pass" if passed else "fail",
        }
        if passed:
            entry["qps"] = qps_score(mats, child)
        else:
            # Shape alignment note for cross-family candidates (QPS would
            # abstain anyway; the gate decides first, as in the paper).
            ok, reason = _shapes_align(mats, child)
            entry["qps"] = {"qps": None, "reason": f"gate failed; {reason or 'shapes align'}"}
        candidates.append(entry)
        del mats

    pool = [c for c in candidates if c["gate"] == "pass" and c["qps"]["qps"] is not None]
    argmax = max(pool, key=lambda c: c["qps"]["qps"])["repo_id"] if pool else None
    correct = argmax == fx.true_source if pool else None
    entry = {
        "label": fx.label,
        "child": {
            "repo_id": fx.child_repo,
            "file": fx.child_file,
            "quant": fx.child_quant,
            "revision": used_revision(fx.child_repo),
        },
        "true_source": fx.true_source,
        "candidates": candidates,
        "qps_argmax": argmax,
        "verdict": "correct"
        if correct
        else ("wrong_parent" if correct is False else "no_candidate"),
        "note": fx.note,
        "compute_seconds": round(time.perf_counter() - t0, 1),
    }
    _log(f"[{fx.label}] argmax={argmax} true={fx.true_source} verdict={entry['verdict']}")
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorLock smoke (smokes/tensorlock/)")
    parser.add_argument("--cache-dir", type=Path, default=None, help="HF cache directory")
    parser.add_argument("--only", default=None, help="run a single fixture label")
    args = parser.parse_args()

    output = {
        "smoke": "tensorlock",
        "paper": "TensorLock (Wu et al., ISSTA 2026), doi:10.1145/3832119",
        "scope": "pairwise: Eq.1 gate, Eq.3 QPS, Eq.7 FT direction (+ Eq.6 distance as context)",
        "params": {"tau_conn": TAU_CONN, "qps_bins": QPS_BINS, "nmi_sample_cap": NMI_SAMPLE_CAP},
        "pairs": [],
        "quant": [],
    }
    labels = {p.label for p in DENSE_PAIRS} | {f.label for f in QUANT_FIXTURES}
    if args.only and args.only not in labels:
        parser.error(f"unknown fixture label: {args.only}")
    pairs = [p for p in DENSE_PAIRS if not args.only or p.label == args.only]
    fixtures = [f for f in QUANT_FIXTURES if not args.only or f.label == args.only]
    for pair in pairs:
        output["pairs"].append(run_dense_pair(pair, args.cache_dir))
    for fx in fixtures:
        output["quant"].append(run_quant_fixture(fx, args.cache_dir))
    print(json.dumps(output, indent=2, default=float))


if __name__ == "__main__":
    main()
