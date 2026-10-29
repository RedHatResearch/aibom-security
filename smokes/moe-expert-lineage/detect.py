"""Label-blind expert-lineage detector for MoE checkpoints (Experiment A).

The detector sees only files and shapes: a suspect checkpoint directory (plain
safetensors + manifest, i.e. what a published derivative would ship) and the
pinned parent snapshot. It never reads variant labels, scenario names, or the
injected permutation / mixture -- those live with the caller.

Method (frozen before the run, see README.md):
- descriptor: one 32x32 window per projection, at an offset derived from
  sha256(seed | role | layer | shape) -- independent of the expert slot, so a
  permuted slot produces the same descriptor;
- per-layer cosine matrix (parent rows x suspect rows), Hungarian assignment
  maximising cosine;
- a pair is `exact` at cos >= EXACT_COS, otherwise `candidate`;
- for every suspect expert without an exact match, fit y = sum_i a_i x_i over
  the parent experts without an exact match (a_i >= 0, sum a_i = 1) and report
  gain = 1 - RSS(mix) / RSS(best single).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from scipy.optimize import linear_sum_assignment, nnls

BLOCK = 32
EXACT_COS = 0.9995
NEAR_EXACT_COS = 0.995
MERGE_GAIN_MIN = 0.30
MIN_COMPONENT = 0.04
CHUNK = 1 << 16


class DetectionError(Exception):
    """A checkpoint that cannot be scored; callers turn this into a SKIP line."""


@dataclass(frozen=True)
class TensorSpec:
    """Tensor naming for one architecture."""

    expert: str
    router: str
    roles: tuple[str, ...]


@dataclass
class CheckpointReader:
    """Read-only tensor access to a checkpoint directory (or HF snapshot)."""

    root: Path
    spec: TensorSpec
    shards: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if not self.shards:
            index = self.root / "model.safetensors.index.json"
            if index.is_file():
                try:
                    with index.open() as f:
                        weight_map = json.load(f).get("weight_map", {})
                except (OSError, json.JSONDecodeError) as exc:
                    raise DetectionError(f"unreadable shard index: {exc}") from exc
                if not isinstance(weight_map, dict):
                    raise DetectionError("shard index has no weight_map")
                self.shards = {str(k): str(v) for k, v in weight_map.items()}

    def tensor(self, name: str) -> torch.Tensor:
        path = (
            self.root / self.shards[name]
            if name in self.shards
            else self.root / "model.safetensors"
        )
        try:
            with safe_open(str(path), framework="pt") as f:
                return f.get_tensor(name)
        except Exception as exc:  # noqa: BLE001 - untrusted bytes, map to a structural skip
            raise DetectionError(f"{name}: {type(exc).__name__}: {exc}") from exc

    def expert(self, layer: int, expert: int) -> dict[str, torch.Tensor]:
        return {
            role: self.tensor(self.spec.expert.format(layer=layer, expert=expert, role=role))
            for role in self.spec.roles
        }

    def router(self, layer: int) -> torch.Tensor:
        return self.tensor(self.spec.router.format(layer=layer))


def read_manifest(variant_dir: Path) -> dict:
    """Validate a variant manifest; anything surprising is a structural skip."""
    path = Path(variant_dir) / "manifest.json"
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DetectionError(f"manifest unreadable: {exc}") from exc
    if not isinstance(raw, dict):
        raise DetectionError("manifest is not an object")
    layers = raw.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise DetectionError("manifest has no layers")
    counts: dict[int, int] = {}
    for key, value in layers.items():
        try:
            layer, count = int(key), int(value["num_experts"])
        except (TypeError, ValueError, KeyError) as exc:
            raise DetectionError(f"manifest layer {key!r} malformed: {exc}") from exc
        if count < 1 or layer < 0:
            raise DetectionError(f"manifest layer {key!r} out of range")
        counts[layer] = count
    raw["layer_counts"] = counts
    return raw


# --------------------------------------------------------------------------- #
# descriptors
# --------------------------------------------------------------------------- #
def block_origin(seed: int, role: str, layer: int, shape: tuple[int, int]) -> tuple[int, int]:
    """Slot-independent window offset for (seed, role, layer, shape)."""
    if min(shape) < BLOCK:
        raise DetectionError(f"shape {shape} smaller than the {BLOCK}x{BLOCK} block")
    key = f"{seed}|{role}|{layer}|{shape[0]}x{shape[1]}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    return int(rng.integers(0, shape[0] - BLOCK + 1)), int(rng.integers(0, shape[1] - BLOCK + 1))


def _block(tensor: torch.Tensor, seed: int, role: str, layer: int) -> np.ndarray:
    if tensor.ndim != 2:
        raise DetectionError(f"{role} layer {layer} is not 2-D")
    row, col = block_origin(seed, role, layer, (int(tensor.shape[0]), int(tensor.shape[1])))
    block = tensor[row : row + BLOCK, col : col + BLOCK].to(torch.float64).numpy()
    if not np.isfinite(block).all():
        raise DetectionError(f"{role} layer {layer} block has non-finite values")
    return block.reshape(-1)


def descriptor(reader: CheckpointReader, layer: int, expert: int, seed: int) -> np.ndarray:
    tensors = reader.expert(layer, expert)
    parts = [_block(tensors[role], seed, role, layer) for role in reader.spec.roles]
    vector = np.concatenate(parts)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise DetectionError(f"degenerate descriptor for layer {layer} expert {expert}")
    return vector / norm


def descriptors(reader: CheckpointReader, layer: int, num_experts: int, seed: int) -> np.ndarray:
    matrix = np.stack([descriptor(reader, layer, expert, seed) for expert in range(num_experts)])
    if not np.isfinite(matrix).all():
        raise DetectionError(f"non-finite descriptor matrix for layer {layer}")
    return matrix


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
def match(parent_desc: np.ndarray, suspect_desc: np.ndarray, exact_cos: float = EXACT_COS) -> dict:
    """Hungarian assignment maximising cosine, then exact / candidate split."""
    if parent_desc.shape[1] != suspect_desc.shape[1]:
        raise DetectionError("descriptor dimensions differ")
    cosine = parent_desc @ suspect_desc.T
    rows, cols = linear_sum_assignment(-cosine)
    exact, candidates = [], []
    for parent, suspect in zip(rows.tolist(), cols.tolist(), strict=True):
        record = {
            "parent": int(parent),
            "suspect": int(suspect),
            "cos": float(cosine[parent, suspect]),
        }
        (exact if record["cos"] >= exact_cos else candidates).append(record)
    matched_parents = {pair["parent"] for pair in exact}
    matched_suspects = {pair["suspect"] for pair in exact}
    return {
        "exact": exact,
        "candidates": candidates,
        "missing": [p for p in range(parent_desc.shape[0]) if p not in matched_parents],
        "new": [s for s in range(suspect_desc.shape[0]) if s not in matched_suspects],
        "max_pair_cos": float(cosine.max()) if cosine.size else float("nan"),
        "max_candidate_cos": max((pair["cos"] for pair in candidates), default=float("nan")),
    }


# --------------------------------------------------------------------------- #
# constrained mixture fit
# --------------------------------------------------------------------------- #
def _nnls_sum_to_one(gram: np.ndarray, cross: np.ndarray) -> tuple[np.ndarray, bool]:
    """argmin a'Ga - 2b'a s.t. a >= 0, then rescale to sum(a) = 1.

    Solved through the PSD square root of the Gram matrix so only the k x k
    normal equations are touched; equivalent to NNLS on the stacked tensors.
    """
    k = cross.size
    if k == 1:
        return np.ones(1), False
    values, vectors = np.linalg.eigh(gram)
    values = np.clip(values, 0.0, None)
    keep = values > max(float(values.max()), 1.0) * 1e-14
    if not keep.any():
        return np.zeros(k), True
    root = np.sqrt(values[keep])
    scale = root[:, None] * vectors[:, keep].T
    target = (vectors[:, keep].T @ cross) / root
    try:
        alpha, _ = nnls(scale, target, maxiter=100 * k)
    except RuntimeError:
        return np.zeros(k), True
    total = float(alpha.sum())
    if total <= 0.0:
        return np.zeros(k), True
    return alpha / total, False


def fit_mixtures(
    basis: list[list[torch.Tensor]],
    suspects: list[list[torch.Tensor]],
) -> list[dict]:
    """Fit each suspect as a convex mixture of the given parent experts.

    ``basis[i]`` and ``suspects[j]`` are the per-projection tensors of one expert;
    all tensors of a projection must share a shape. Statistics are accumulated in
    float64 over the concatenation of all projections.
    """
    if not basis or not suspects:
        return []
    projections = len(basis[0])
    for expert in basis + suspects:
        if len(expert) != projections:
            raise DetectionError("mixed projection counts in mixture fit")

    k, m = len(basis), len(suspects)
    gram = np.zeros((k, k))
    cross = np.zeros((k, m))
    energy = np.zeros(m)

    for projection in range(projections):
        shape = basis[0][projection].shape
        for expert in basis + suspects:
            if expert[projection].shape != shape:
                raise DetectionError("mixture fit needs identical expert shapes")
        flat_basis = [expert[projection].reshape(-1) for expert in basis]
        flat_suspects = [expert[projection].reshape(-1) for expert in suspects]
        numel = int(shape.numel())
        for start in range(0, numel, CHUNK):
            stop = min(start + CHUNK, numel)
            x = torch.stack([t[start:stop].to(torch.float64) for t in flat_basis])
            y = torch.stack([t[start:stop].to(torch.float64) for t in flat_suspects])
            gram += (x @ x.T).numpy()
            cross += (x @ y.T).numpy()
            energy += (y * y).sum(dim=1).numpy()

    fits = []
    for j in range(m):
        diagonal = np.diag(gram)
        usable = diagonal > 0.0
        if not usable.any():
            fits.append(
                {
                    "suspect_index": j,
                    "basis": [],
                    "alpha": [],
                    "gain": None,
                    "best_single": None,
                    "degenerate": True,
                }
            )
            continue
        best_single = energy[j] - float(
            np.max(np.where(usable, cross[:, j] ** 2 / np.where(usable, diagonal, 1.0), -np.inf))
        )
        best_index = int(
            np.argmax(np.where(usable, cross[:, j] ** 2 / np.where(usable, diagonal, 1.0), -np.inf))
        )
        alpha, degenerate = _nnls_sum_to_one(gram, cross[:, j])
        rss_mix = float(energy[j] - 2.0 * alpha @ cross[:, j] + alpha @ gram @ alpha)
        if best_single <= 1e-12 * max(float(energy[j]), 1e-30):
            gain = None
        else:
            gain = float(1.0 - rss_mix / best_single)
        fits.append(
            {
                "suspect_index": j,
                "basis": list(range(k)),
                "alpha": [float(a) for a in alpha],
                "gain": gain,
                "best_single": best_index,
                "degenerate": degenerate,
            }
        )
    return fits


# --------------------------------------------------------------------------- #
# per-layer analysis
# --------------------------------------------------------------------------- #
def analyze_layer(
    parent: CheckpointReader,
    parent_layer: int,
    parent_experts: int,
    suspect: CheckpointReader,
    suspect_layer: int,
    suspect_experts: int,
    seed: int,
    *,
    exact_cos: float = EXACT_COS,
    parent_desc: np.ndarray | None = None,
) -> dict:
    """Descriptors, assignment, exact/candidate split, and mixture fits for one layer."""
    router = suspect.router(suspect_layer)
    if router.ndim != 2 or int(router.shape[0]) != int(suspect_experts):
        raise DetectionError(
            f"router layer {suspect_layer} has shape {tuple(router.shape)}, "
            f"expected {suspect_experts} rows"
        )
    pdesc = (
        descriptors(parent, parent_layer, parent_experts, seed)
        if parent_desc is None
        else parent_desc
    )
    sdesc = descriptors(suspect, suspect_layer, suspect_experts, seed)
    result = match(pdesc, sdesc, exact_cos)

    # Fit every suspect against every parent. An exact copy fits itself with
    # alpha=1 and gain ~0, so the fit does not disturb copy detection; a shallow
    # merge (e.g. 0.95/0.05) still fits two parents with high gain even when
    # cosine matching alone calls it an exact copy.
    fits = fit_mixtures(
        [
            [parent.expert(parent_layer, p)[role] for role in parent.spec.roles]
            for p in range(parent_experts)
        ],
        [
            [suspect.expert(suspect_layer, s)[role] for role in suspect.spec.roles]
            for s in range(suspect_experts)
        ],
    )
    assigned = {pair["suspect"]: pair["cos"] for pair in result["exact"] + result["candidates"]}
    classifications = []
    for fit in fits:
        alpha = fit["alpha"]
        big = [i for i, a in enumerate(alpha) if a >= MIN_COMPONENT]
        gain = fit["gain"]
        if gain is not None and gain >= MERGE_GAIN_MIN and len(big) >= 2:
            label = "merged"
        elif assigned.get(fit["suspect_index"], 0.0) >= exact_cos:
            label = "exact"
        else:
            label = "unexplained"
        classifications.append(
            {
                "suspect": int(fit["suspect_index"]),
                "label": label,
                "cos": float(assigned.get(fit["suspect_index"], float("nan"))),
                "gain": gain,
                "basis": big,
                "alpha": [float(alpha[i]) for i in big],
            }
        )

    merges = [
        {
            "suspect": entry["suspect"],
            "basis": entry["basis"],
            "alpha": entry["alpha"],
            "gain": entry["gain"],
            "best_single_parent": None,
            "degenerate": False,
        }
        for entry in classifications
        if entry["label"] == "merged"
    ]
    near_exact = [
        entry
        for entry in classifications
        if entry["label"] == "merged" and entry["cos"] >= exact_cos
    ]

    return {
        "layer": int(suspect_layer),
        "parent_layer": int(parent_layer),
        "parent_experts": int(parent_experts),
        "suspect_experts": int(suspect_experts),
        "matched": len(result["exact"]),
        "exact_pairs": result["exact"],
        "candidate_pairs": result["candidates"],
        "missing": result["missing"],
        "new": result["new"],
        "recovered_perm": {int(pair["suspect"]): int(pair["parent"]) for pair in result["exact"]},
        "merges": merges,
        "classifications": classifications,
        "near_exact_merges": near_exact,
        "max_pair_cos": result["max_pair_cos"],
        "max_candidate_cos": result["max_candidate_cos"],
        "max_assigned_cos": max(
            (pair["cos"] for pair in result["exact"] + result["candidates"]),
            default=float("nan"),
        ),
    }
