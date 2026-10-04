"""Witness Overlap smoke — smokes/witness-overlap/ · arXiv:2609.31784 · #26 survey.

"Witness Overlap: Directional Provenance Inside Open-Weight Model Families",
Li et al., NeurIPS 2026 (arXiv:2609.31784).

Stage-2 provenance inside an open-weight family: given an anchor A, target B
and witness W (all same-family checkpoints), the weight deltas Δ_AB = B−A and
Δ_AW = W−A are compared per tensor with (1) the Frobenius cosine of the two
deltas and (2) the SVD top-k right-singular-subspace overlap. Scores are
aggregated block-role-wise (mean within each of the 7 llama-style projection
roles across layers, then mean across roles). The child-anchor score s_B
systematically exceeds the parent-anchor score s_A, so the predicted parent
of a pair is argmin_X s_X pooled over witnesses. Dually, the family root is
argmin_M of the mean of overlap(M; W1, W2) over unordered witness pairs.
ρ/η failure tags (cosine between the two descendant updates, norm ratio)
flag degenerate triplets. Reimplemented from the paper; the authors' code
artifact is an empty placeholder.

https://github.com/RedHatResearch/aibom-security/issues/26
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

SMOKE_DIR = Path(__file__).resolve().parent
SMOKES_ROOT = SMOKE_DIR.parent
sys.path.insert(0, str(SMOKE_DIR))
sys.path.insert(0, str(SMOKES_ROOT))

from _lib.hub import load_config, load_tensors, parse_model_config  # noqa: E402
from config import FAMILY_SETS, REVISIONS, TRIPLES, FamilySet, Triple  # noqa: E402

# Llama-style dense projection blocks (paper §3.1): everything except
# embeddings, lm_head and norms.
BLOCK_ROLES = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj")

# Failure-tag thresholds: paper's corpus-specific top-10% cutoffs (Table 11),
# applied as-is.
RHO_THRESHOLD = 0.0247
ETA_THRESHOLD = 45.16


# ---------------------------------------------------------------------------
# Pure scoring core — no I/O, no fixture knowledge. Takes plain dicts of
# name → 2-D float arrays; a later ProvenanceBench detector reuses these.
# ---------------------------------------------------------------------------


def block_role(tensor_name: str) -> str | None:
    """Return the projection role of a tensor name, or None if not one."""
    for role in BLOCK_ROLES:
        if f".{role}.weight" in tensor_name:
            return role
    return None


def select_block_tensors(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Keep only the 7 projection-role tensors (drops embed/lm_head/norms)."""
    return {name: t for name, t in tensors.items() if block_role(name) is not None}


def intersect_tensors(
    tensor_dicts: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, list[str]], list[str]]:
    """Intersect tensor names AND shapes across checkpoints.

    Returns (per-checkpoint filtered dict, dropped names sorted). Roles absent
    from the intersection simply drop out of the block aggregation.
    """
    common: set[str] | None = None
    for tensors in tensor_dicts.values():
        names = {
            name for name, t in tensors.items() if block_role(name) is not None and t.ndim == 2
        }
        common = names if common is None else (common & names)
    common = common or set()
    kept = sorted(
        name
        for name in common
        if all(
            tensor_dicts[c][name].shape == next(iter(tensor_dicts.values()))[name].shape
            for c in tensor_dicts
        )
    )
    dropped = sorted(
        {
            name
            for tensors in tensor_dicts.values()
            for name in tensors
            if block_role(name) is not None and name not in kept
        }
    )
    return {c: kept for c in tensor_dicts}, dropped


def frobenius_cosine(d1: np.ndarray, d2: np.ndarray) -> float | None:
    """⟨d1,d2⟩_F / (‖d1‖_F ‖d2‖_F); None on zero norms."""
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)
    n1 = float(np.linalg.norm(d1.ravel()))
    n2 = float(np.linalg.norm(d2.ravel()))
    if n1 == 0.0 or n2 == 0.0:
        return None
    return float(np.sum(d1 * d2) / (n1 * n2))


def right_singular_basis(delta: np.ndarray, k: int = 16) -> np.ndarray | None:
    """Top-k right-singular vectors of `delta` (columns are orthonormal).

    Never a full SVD: eigendecomposition of the SMALLER Gram side.
    If n <= m: eigh(ΔᵀΔ) (n×n) → top eigenvectors ARE the right basis V.
    Else: eigh(ΔΔᵀ) (m×m) → left basis U; V = QR(Δᵀ U_k) restores an
    orthonormal right basis. eigh is ascending → take last k, reversed.
    k is clamped to min(m, n). Returns None for a zero delta.
    """
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {delta.shape}")
    m, n = delta.shape
    if np.linalg.norm(delta.ravel()) == 0.0:
        return None
    kk = max(1, min(k, m, n))
    if n <= m:
        gram = delta.T @ delta  # n×n, smaller side
        evals, evecs = np.linalg.eigh(gram)
        return evecs[:, -kk:][:, ::-1]
    gram = delta @ delta.T  # m×m, smaller side
    _, evecs = np.linalg.eigh(gram)
    u_k = evecs[:, -kk:][:, ::-1]
    v, _ = np.linalg.qr(delta.T @ u_k)  # n×kk orthonormal right basis
    return v


def svd_subspace_overlap(d1: np.ndarray, d2: np.ndarray, k: int = 16) -> float | None:
    """(1/k)·‖V2ᵀV1‖²_F — mean squared cosine of principal angles between
    the top-k right-singular subspaces. None if either delta is zero."""
    v1 = right_singular_basis(d1, k)
    v2 = right_singular_basis(d2, k)
    if v1 is None or v2 is None:
        return None
    kk = min(v1.shape[1], v2.shape[1])
    if kk == 0:
        return None
    cross = v2.T @ v1  # kk×kk
    return float(np.sum(cross * cross) / kk)


class DeltaCache:
    """Caches per-delta Gram eigenbases so each (anchor, checkpoint) delta is
    decomposed once per tensor. V(−Δ) = V(Δ), so Δ_BA reuses Δ_AB's basis."""

    def __init__(self) -> None:
        self._bases: dict[tuple[str, str, str], np.ndarray | None] = {}

    def basis(self, name: str, a: str, b: str, delta: np.ndarray, k: int) -> np.ndarray | None:
        key = (name, min(a, b), max(a, b))  # sign-invariant: V(−Δ)=V(Δ)
        if key not in self._bases:
            self._bases[key] = right_singular_basis(delta, k)
        return self._bases[key]


def _block_mean(per_tensor: dict[str, float | None]) -> dict[str, float]:
    """Step 1 of block-aware aggregation: mean within each block role."""
    sums: dict[str, list[float]] = {}
    for name, value in per_tensor.items():
        if value is None:
            continue
        role = block_role(name)
        if role is not None:
            sums.setdefault(role, []).append(value)
    return {role: float(np.mean(vals)) for role, vals in sums.items()}


def _aggregate(per_tensor: dict[str, float | None]) -> float | None:
    """Block-aware MEAN (paper §3.3): mean within each block role across
    layers, then mean across roles. None if no scores survived."""
    role_means = _block_mean(per_tensor)
    if not role_means:
        return None
    return float(np.mean(list(role_means.values())))


def _overlap(
    tensors_a: dict[str, np.ndarray],
    tensors_b: dict[str, np.ndarray],
    tensors_w: dict[str, np.ndarray],
    k: int,
    cache: DeltaCache | None = None,
    label_a: str = "a",
    label_b: str = "b",
    label_w: str = "w",
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """overlap(A;B,W) for both score variants. Returns (per-tensor cos,
    per-tensor svd). label_a/b/w name the three checkpoints so the shared
    DeltaCache keys deltas by checkpoint pair, not by call-site role."""
    names = sorted(set(tensors_a) & set(tensors_b) & set(tensors_w))
    cache = cache or DeltaCache()
    cos: dict[str, float | None] = {}
    svd: dict[str, float | None] = {}
    for name in names:
        delta_ab = tensors_b[name].astype(np.float64) - tensors_a[name].astype(np.float64)
        delta_aw = tensors_w[name].astype(np.float64) - tensors_a[name].astype(np.float64)
        cos[name] = frobenius_cosine(delta_ab, delta_aw)
        v_ab = cache.basis(name, label_a, label_b, delta_ab, k)
        v_aw = cache.basis(name, label_a, label_w, delta_aw, k)
        svd[name] = (
            None
            if v_ab is None or v_aw is None
            else float(np.sum((v_aw.T @ v_ab) ** 2) / min(v_ab.shape[1], v_aw.shape[1]))
        )
    return cos, svd


def witness_scores(
    tensors_a: dict[str, np.ndarray],
    tensors_b: dict[str, np.ndarray],
    tensors_w: dict[str, np.ndarray],
    k: int = 16,
) -> dict:
    """s_A = overlap(A;B,W) and s_B = overlap(B;A,W), cos + svd variants,
    with per-role and per-tensor detail. Pure."""
    cache = DeltaCache()
    cos_a, svd_a = _overlap(tensors_a, tensors_b, tensors_w, k, cache, "A", "B", "W")
    cos_b, svd_b = _overlap(tensors_b, tensors_a, tensors_w, k, cache, "B", "A", "W")
    return {
        "s_A": {"cos": _aggregate(cos_a), "svd": _aggregate(svd_a)},
        "s_B": {"cos": _aggregate(cos_b), "svd": _aggregate(svd_b)},
        "per_role": {
            "s_A": {"cos": _block_mean(cos_a), "svd": _block_mean(svd_a)},
            "s_B": {"cos": _block_mean(cos_b), "svd": _block_mean(svd_b)},
        },
        "per_tensor": {"s_A": {"cos": cos_a, "svd": svd_a}, "s_B": {"cos": cos_b, "svd": svd_b}},
    }


def root_scores(
    family_tensors: dict[str, dict[str, np.ndarray]],
    k: int = 16,
) -> dict:
    """Per-member root score = mean of overlap(M; W1, W2) over all unordered
    pairs {W1, W2} of the other members (cos + svd). overlap is symmetric in
    target/witness, so each unordered pair is scored once. Pure."""
    members = sorted(family_tensors)
    cache = DeltaCache()
    pair_scores: dict[tuple[str, str, str], dict[str, float | None]] = {}
    for m in members:
        others = [o for o in members if o != m]
        for x in range(len(others)):
            for y in range(x + 1, len(others)):
                w1, w2 = others[x], others[y]
                key = (m, min(w1, w2), max(w1, w2))
                if key not in pair_scores:
                    cos, svd = _overlap(
                        family_tensors[m],
                        family_tensors[w1],
                        family_tensors[w2],
                        k,
                        cache,
                        m,
                        w1,
                        w2,
                    )
                    pair_scores[key] = {"cos": _aggregate(cos), "svd": _aggregate(svd)}
    out: dict[str, dict] = {}
    for m in members:
        others = [o for o in members if o != m]
        pair_keys = [
            (m, min(others[x], others[y]), max(others[x], others[y]))
            for x in range(len(others))
            for y in range(x + 1, len(others))
        ]
        member_out: dict = {}
        for variant in ("cos", "svd"):
            vals = [v for key in pair_keys if (v := pair_scores[key][variant]) is not None]
            member_out[variant] = float(np.mean(vals)) if vals else None
        member_out["pairs"] = [{"witnesses": [pk[1], pk[2]], **pair_scores[pk]} for pk in pair_keys]
        out[m] = member_out
    return out


def rho_eta(
    tensors_parent: dict[str, np.ndarray],
    tensors_child: dict[str, np.ndarray],
    tensors_witness: dict[str, np.ndarray],
) -> dict:
    """ρ = cosine between the two descendant updates from the parent
    (flattened concatenation of the selected tensors); η = ‖u_witness‖ /
    ‖u_child‖. Zero guards → None. Failure tags from the paper's cutoffs."""
    names = sorted(set(tensors_parent) & set(tensors_child) & set(tensors_witness))
    if not names:
        return {"rho": None, "eta": None, "failure_tag": "neither"}
    u_child = np.concatenate(
        [
            (tensors_child[n].astype(np.float64) - tensors_parent[n].astype(np.float64)).ravel()
            for n in names
        ]
    )
    u_witness = np.concatenate(
        [
            (tensors_witness[n].astype(np.float64) - tensors_parent[n].astype(np.float64)).ravel()
            for n in names
        ]
    )
    rho = frobenius_cosine(u_child.reshape(1, -1), u_witness.reshape(1, -1))
    n_child = float(np.linalg.norm(u_child))
    n_witness = float(np.linalg.norm(u_witness))
    eta = None if n_child == 0.0 else n_witness / n_child
    tag = "neither"
    if rho is not None and rho >= RHO_THRESHOLD and eta is not None and eta >= ETA_THRESHOLD:
        tag = "both"
    elif rho is not None and rho >= RHO_THRESHOLD:
        tag = "high-rho"
    elif eta is not None and eta >= ETA_THRESHOLD:
        tag = "high-eta"
    return {"rho": rho, "eta": eta, "failure_tag": tag}


# ---------------------------------------------------------------------------
# Hub loading layer (smoke-only; the pure core above never touches I/O).
# ---------------------------------------------------------------------------


def projection_tensor_names(num_layers: int) -> set[str]:
    """The 7 llama-style projection tensor names for all layers."""
    names: set[str] = set()
    for layer in range(num_layers):
        for role in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names.add(f"model.layers.{layer}.self_attn.{role}.weight")
        for role in ("up_proj", "gate_proj", "down_proj"):
            names.add(f"model.layers.{layer}.mlp.{role}.weight")
    return names


def resolve_revision(repo_id: str, cache_dir: Path | None) -> str | None:
    """Resolved commit sha of the snapshot actually used (dir name), or None
    if the snapshot path does not carry a sha (e.g. a local directory)."""
    model_dir = snapshot_download(
        repo_id,
        allow_patterns=["*.safetensors", "model.safetensors.index.json"],
        cache_dir=cache_dir,
    )
    name = Path(model_dir).name
    return name if re.fullmatch(r"[0-9a-f]{40}", name) else None


def load_checkpoint(
    repo_id: str, cache_dir: Path | None
) -> tuple[dict[str, np.ndarray], str | None]:
    """Config-gated projection tensors for one checkpoint + its revision sha.
    Loads only this checkpoint's own projection names at the pinned revision
    (config.REVISIONS); name∩shape intersection across checkpoints happens
    later (needed for allow_partial fixtures)."""
    cfg = parse_model_config(repo_id, load_config(repo_id, cache_dir))
    revision = REVISIONS.get(repo_id)
    tensors = load_tensors(
        repo_id, projection_tensor_names(cfg.num_layers), cache_dir=cache_dir, revision=revision
    )
    return tensors, revision if revision is not None else resolve_revision(repo_id, cache_dir)


def _gate_reason(configs: dict[str, object]) -> str | None:
    """Compat gate: hidden_size / num_layers / num_attention_heads /
    num_key_value_heads must match across checkpoints. Returns a reason string
    on mismatch, None when compatible."""
    fields = (
        "hidden_size",
        "num_layers",
        "num_attention_heads",
        "num_key_value_heads",
    )
    mismatches = []
    for field_name in fields:
        values = {repo: getattr(cfg, field_name) for repo, cfg in configs.items()}
        if len(set(values.values())) > 1:
            detail = ", ".join(f"{repo}={v}" for repo, v in sorted(values.items()))
            mismatches.append(f"{field_name} ({detail})")
    return "; ".join(mismatches) if mismatches else None


def _predicted(scores: dict[str, float | None], candidates: tuple[str, str]) -> str | None:
    """argmin over candidates with the pooled score; ties → first candidate."""
    if any(scores.get(c) is None for c in candidates):
        return None
    return min(candidates, key=lambda c: (scores[c], candidates.index(c)))


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def run_triple(triple: Triple, cache_dir: Path | None, k: int) -> dict:
    """Score one orientation fixture end-to-end and return its JSON entry."""
    t0 = time.perf_counter()
    active = list(triple.witnesses)

    entry = {
        "label": triple.label,
        "note": triple.note,
        "allow_partial": triple.allow_partial,
        "ground_truth_parent": triple.ground_truth_parent,
    }
    if not active:
        entry["gate"] = {"status": "abstain", "reason": "no witnesses configured"}
        entry["witnesses"] = []
        entry["pooled"] = None
        entry["compute_seconds"] = time.perf_counter() - t0
        return entry

    configs = {
        repo: parse_model_config(repo, load_config(repo, cache_dir))
        for repo in (triple.parent, triple.child, *(w.repo_id for w in active))
    }
    mismatch = _gate_reason(configs)
    if mismatch is not None and not triple.allow_partial:
        entry["gate"] = {"status": "abstain", "reason": f"config mismatch: {mismatch}"}
        entry["witnesses"] = []
        entry["pooled"] = None
        entry["compute_seconds"] = time.perf_counter() - t0
        return entry

    loaded = {repo: load_checkpoint(repo, cache_dir) for repo in (triple.parent, triple.child)}
    loaded.update({w.repo_id: load_checkpoint(w.repo_id, cache_dir) for w in active})
    tensors = {repo: tensors for repo, (tensors, _rev) in loaded.items()}
    revisions = {repo: rev for repo, (_t, rev) in loaded.items()}

    selected_names, dropped = intersect_tensors(tensors)
    reason = f"config mismatch ({mismatch}); " if mismatch is not None else ""
    reason += (
        f"{len(dropped)} of {len(selected_names[triple.parent]) + len(dropped)} "
        f"block tensors dropped"
    )
    entry["gate"] = {
        "status": "degraded" if (mismatch is not None or dropped) else "ok",
        "reason": reason if (mismatch is not None or dropped) else "compatible",
        "dropped_tensors": dropped,
        "tensor_count": len(selected_names[triple.parent]),
    }
    entry["repos"] = [
        {
            "repo_id": repo,
            "revision": revisions[repo],
            "role": role,
            "kind": kind,
        }
        for repo, role, kind in [
            (triple.parent, "parent", "endpoint"),
            (triple.child, "child", "endpoint"),
            *[(w.repo_id, "witness", w.kind) for w in active],
        ]
    ]

    parent_t = {n: tensors[triple.parent][n] for n in selected_names[triple.parent]}
    child_t = {n: tensors[triple.child][n] for n in selected_names[triple.child]}
    witness_results: list[dict] = []
    pooled: dict[str, list[float]] = {"cos_A": [], "cos_B": [], "svd_A": [], "svd_B": []}
    for witness in active:
        w_t = {n: tensors[witness.repo_id][n] for n in selected_names[witness.repo_id]}
        scores = witness_scores(parent_t, child_t, w_t, k=k)
        metrics = rho_eta(parent_t, child_t, w_t)
        s_a = {"cos": scores["s_A"]["cos"], "svd": scores["s_A"]["svd"]}
        s_b = {"cos": scores["s_B"]["cos"], "svd": scores["s_B"]["svd"]}
        predicted = {
            variant: _predicted(
                {triple.parent: s_a[variant], triple.child: s_b[variant]},
                (triple.parent, triple.child),
            )
            for variant in ("cos", "svd")
        }
        correct = (
            None
            if triple.ground_truth_parent is None
            else {
                variant: (
                    None
                    if predicted[variant] is None
                    else predicted[variant] == triple.ground_truth_parent
                )
                for variant in ("cos", "svd")
            }
        )
        witness_results.append(
            {
                "repo_id": witness.repo_id,
                "kind": witness.kind,
                "status": "scored",
                "s_A": s_a,
                "s_B": s_b,
                "per_role": scores["per_role"],
                "predicted_parent": predicted,
                "ground_truth_parent": triple.ground_truth_parent,
                "correct": correct,
                "rho": metrics["rho"],
                "eta": metrics["eta"],
                "failure_tag": metrics["failure_tag"],
            }
        )
        for variant in ("cos", "svd"):
            if s_a[variant] is not None:
                pooled[f"{variant}_A"].append(s_a[variant])
            if s_b[variant] is not None:
                pooled[f"{variant}_B"].append(s_b[variant])

    pooled_scores = {
        "s_A": {"cos": _mean(pooled["cos_A"]), "svd": _mean(pooled["svd_A"])},
        "s_B": {"cos": _mean(pooled["cos_B"]), "svd": _mean(pooled["svd_B"])},
    }
    pooled_predicted = {
        variant: _predicted(
            {
                triple.parent: pooled_scores["s_A"][variant],
                triple.child: pooled_scores["s_B"][variant],
            },
            (triple.parent, triple.child),
        )
        for variant in ("cos", "svd")
    }
    entry["witnesses"] = witness_results
    entry["pooled"] = {
        "n_witnesses": len(active),
        **pooled_scores,
        "predicted_parent": pooled_predicted,
        "ground_truth_parent": triple.ground_truth_parent,
        "correct": (
            None
            if triple.ground_truth_parent is None
            else {
                variant: (
                    None
                    if pooled_predicted[variant] is None
                    else pooled_predicted[variant] == triple.ground_truth_parent
                )
                for variant in ("cos", "svd")
            }
        ),
    }
    entry["compute_seconds"] = time.perf_counter() - t0
    return entry


def run_family(family: FamilySet, cache_dir: Path | None, k: int) -> dict:
    """Score one root-identification fixture and return its JSON entry."""
    t0 = time.perf_counter()
    active = list(family.members)

    entry = {"label": family.label, "note": family.note}
    if len(active) < 3:
        entry["gate"] = {
            "status": "abstain",
            "reason": f"only {len(active)} members; need >= 3 for root ID",
        }
        entry["members"] = []
        entry["root_scores"] = None
        entry["predicted_root"] = None
        entry["compute_seconds"] = time.perf_counter() - t0
        return entry

    configs = {
        m.repo_id: parse_model_config(m.repo_id, load_config(m.repo_id, cache_dir)) for m in active
    }
    mismatch = _gate_reason(configs)
    if mismatch is not None:
        entry["gate"] = {"status": "abstain", "reason": f"config mismatch: {mismatch}"}
        entry["members"] = []
        entry["root_scores"] = None
        entry["predicted_root"] = None
        entry["compute_seconds"] = time.perf_counter() - t0
        return entry

    loaded = {m.repo_id: load_checkpoint(m.repo_id, cache_dir) for m in active}
    tensors = {repo: t for repo, (t, _rev) in loaded.items()}
    revisions = {repo: rev for repo, (_t, rev) in loaded.items()}
    selected_names, dropped = intersect_tensors(tensors)
    selected = {
        repo: {n: tensors[repo][n] for n in names} for repo, names in selected_names.items()
    }
    entry["gate"] = {
        "status": "degraded" if dropped else "ok",
        "reason": (
            f"{len(dropped)} block tensors dropped by name/shape intersection"
            if dropped
            else "compatible"
        ),
        "dropped_tensors": dropped,
        "tensor_count": len(next(iter(selected_names.values()))),
    }
    entry["members"] = [
        {
            "repo_id": m.repo_id,
            "revision": revisions[m.repo_id],
            "kind": m.kind,
            "status": "member",
        }
        for m in active
    ]

    roots = root_scores(selected, k=k)
    scores_table = {
        m.repo_id: {"cos": roots[m.repo_id]["cos"], "svd": roots[m.repo_id]["svd"]} for m in active
    }
    candidates = tuple(m.repo_id for m in active)
    predicted_root = {
        variant: _predicted({c: scores_table[c][variant] for c in candidates}, candidates)
        for variant in ("cos", "svd")
    }
    ground_truth = family.ground_truth_root
    entry["root_scores"] = scores_table
    entry["root_pair_detail"] = {m.repo_id: roots[m.repo_id]["pairs"] for m in active}
    entry["predicted_root"] = predicted_root
    entry["ground_truth_root"] = ground_truth
    entry["correct"] = {
        variant: (
            None if predicted_root[variant] is None else predicted_root[variant] == ground_truth
        )
        for variant in ("cos", "svd")
    }
    entry["compute_seconds"] = time.perf_counter() - t0
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Witness Overlap smoke (smokes/witness-overlap/)")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--only", default=None, help="run a single fixture set by label")
    parser.add_argument("--k", type=int, default=16, help="SVD subspace dimension")
    args = parser.parse_args()

    triples = [t for t in TRIPLES if args.only in (None, t.label)]
    families = [f for f in FAMILY_SETS if args.only in (None, f.label)]
    if args.only is not None and not triples and not families:
        parser.error(f"unknown fixture label: {args.only}")

    total_start = time.perf_counter()
    triple_results = []
    for triple in triples:
        print(f"=== triple {triple.label}: {triple.parent} vs {triple.child} ===", file=sys.stderr)
        result = run_triple(triple, args.cache_dir, args.k)
        triple_results.append(result)
        gate = result["gate"]
        print(f"  gate={gate['status']} ({gate['reason']})", file=sys.stderr)

    family_results = []
    for family in families:
        print(f"=== family {family.label} ===", file=sys.stderr)
        result = run_family(family, args.cache_dir, args.k)
        family_results.append(result)
        gate = result["gate"]
        print(f"  gate={gate['status']} ({gate['reason']})", file=sys.stderr)

    output = {
        "method": "witness-overlap",
        "paper": "arXiv:2609.31784 (Li et al., NeurIPS 2026)",
        "issue": 26,
        "k": args.k,
        "triples": triple_results,
        "family_sets": family_results,
        "total_seconds": time.perf_counter() - total_start,
    }
    print(json.dumps(output, indent=2, default=float))


if __name__ == "__main__":
    main()
