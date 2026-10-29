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
- every suspect expert is fitted as y = sum_i a_i x_i over every parent expert
  (a_i >= 0, sum a_i = 1); gain = 1 - RSS(mix) / RSS(best single);
- a suspect whose fit has no gain (it is proportional to one parent) and one
  dominant component is an exact duplicate, even when the 1-1 assignment left
  it `new`.

The manifest and the shard index are untrusted metadata: both are size-capped,
every parse surprise maps to `DetectionError`, `weight_map` values must stay
inside the checkpoint directory, and non-finite tensor values that reach the
mixture fit are a structural abstain rather than a silent `gain: NaN`.
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
# A component below this share is fit noise, not a merge parent. Matches the
# low end of the pre-registered alpha band (SHALLOW_ALPHA[1] - ALPHA_TOL).
MIN_COMPONENT = 0.03
CHUNK = 1 << 16

# Bounds on attacker-declared structure, checked before anything is allocated.
MAX_METADATA_BYTES = 4 << 20  # manifest.json / shard index
MAX_TENSOR_BYTES = 256 << 20  # one materialised tensor
MAX_EXPERTS_PER_LAYER = 512
MAX_LAYERS = 256

_ITEMSIZE = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


class DetectionError(Exception):
    """A checkpoint that cannot be scored; callers turn this into a SKIP line."""


@dataclass(frozen=True)
class TensorSpec:
    """Tensor naming for one architecture."""

    expert: str
    router: str
    roles: tuple[str, ...]


# --------------------------------------------------------------------------- #
# untrusted metadata
# --------------------------------------------------------------------------- #
def _read_json(path: Path, what: str) -> object:
    """Parse a size-capped JSON metadata file; any surprise is a structural skip."""
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_METADATA_BYTES + 1)
    except OSError as exc:
        raise DetectionError(f"{what} unreadable: {exc}") from exc
    if len(data) > MAX_METADATA_BYTES:
        raise DetectionError(f"{what} exceeds {MAX_METADATA_BYTES} bytes")
    try:
        return json.loads(data)
    except (ValueError, TypeError, AttributeError, RecursionError) as exc:
        raise DetectionError(f"{what} unparseable: {type(exc).__name__}: {exc}") from exc


def _checkpoint_file(root: Path, value: object, what: str) -> Path:
    """Resolve a metadata-supplied shard filename to a direct child of ``root``.

    Only a plain filename is accepted: absolute paths, ``..``, any separator,
    NUL and empty/blank values are rejected outright. The file itself is not
    ``resolve()``-ed and compared to ``root`` because an HF cache snapshot
    materialises every shard as a symlink into the cache's ``blobs/`` tree;
    containment comes from the plain-filename rule, which leaves metadata no way
    to name a path outside the checkpoint directory.
    """
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        raise DetectionError(f"{what}: invalid filename {value!r}")
    if not value.endswith(".safetensors"):
        raise DetectionError(f"{what}: {value!r} is not a safetensors file")
    relative = Path(value)
    if relative.is_absolute() or relative.name != value or ".." in relative.parts:
        raise DetectionError(f"{what}: path escapes the checkpoint: {value!r}")
    return root / relative


@dataclass
class CheckpointReader:
    """Read-only tensor access to a checkpoint directory (or HF snapshot)."""

    root: Path
    spec: TensorSpec
    shards: dict[str, str] = field(default_factory=dict)
    tensor_file: str = "model.safetensors"
    _index: dict[str, Path] | None = field(default=None, init=False, repr=False)
    _files: tuple[Path, ...] = field(default=(), init=False, repr=False)
    _name_map: dict[str, Path] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if not self.root.is_dir():
            raise DetectionError(f"{self.root} is not a directory")
        if self.shards:
            self._index = {
                str(name): _checkpoint_file(self.root, value, "shard index")
                for name, value in self.shards.items()
            }
            self._files = tuple(sorted(set(self._index.values())))
        else:
            index = self.root / "model.safetensors.index.json"
            if index.is_file():
                document = _read_json(index, "shard index")
                if not isinstance(document, dict):
                    raise DetectionError("shard index is not an object")
                weight_map = document.get("weight_map")
                if not isinstance(weight_map, dict) or not weight_map:
                    raise DetectionError("shard index has no weight_map")
                self._index = {
                    str(name): _checkpoint_file(self.root, value, "shard index")
                    for name, value in weight_map.items()
                }
                self._files = tuple(sorted(set(self._index.values())))
            else:
                self._files = (_checkpoint_file(self.root, self.tensor_file, "tensor_file"),)

    def _scan(self) -> dict[str, Path]:
        mapping: dict[str, Path] = {}
        for path in self._files:
            try:
                with safe_open(str(path), framework="pt") as handle:
                    available = handle.keys()
            except Exception as exc:  # noqa: BLE001 - untrusted bytes
                raise DetectionError(f"{path.name}: {type(exc).__name__}: {exc}") from exc
            for name in available:
                mapping.setdefault(str(name), path)
        return mapping

    def names(self) -> dict[str, Path]:
        """Tensor name -> file, from the physical headers (the index is only a path list)."""
        if self._name_map is None:
            self._name_map = self._scan()
        return self._name_map

    def _locate(self, name: str) -> Path:
        path = self.names().get(name)
        if path is None:
            raise DetectionError(f"{name}: not in the checkpoint")
        return path

    def shape(self, name: str) -> tuple[int, ...]:
        """Declared shape without materialising the tensor."""
        path = self._locate(name)
        try:
            with safe_open(str(path), framework="pt") as handle:
                return tuple(int(v) for v in handle.get_slice(name).get_shape())
        except DetectionError:
            raise
        except Exception as exc:  # noqa: BLE001 - untrusted bytes
            raise DetectionError(f"{name}: {type(exc).__name__}: {exc}") from exc

    def tensor(self, name: str) -> torch.Tensor:
        path = self._locate(name)
        try:
            with safe_open(str(path), framework="pt") as handle:
                window = handle.get_slice(name)
                shape = tuple(int(v) for v in window.get_shape())
                dtype = str(window.get_dtype())
                itemsize = _ITEMSIZE.get(dtype)
                if itemsize is None:
                    raise DetectionError(f"{name}: unsupported dtype {dtype}")
                numel = 1
                for dim in shape:
                    numel *= dim
                if numel * itemsize > MAX_TENSOR_BYTES:
                    raise DetectionError(
                        f"{name}: {shape} {dtype} exceeds {MAX_TENSOR_BYTES} bytes"
                    )
                return handle.get_tensor(name)
        except DetectionError:
            raise
        except Exception as exc:  # noqa: BLE001 - untrusted bytes, map to a structural skip
            raise DetectionError(f"{name}: {type(exc).__name__}: {exc}") from exc

    def expert(self, layer: int, expert: int) -> dict[str, torch.Tensor]:
        return {
            role: self.tensor(self.spec.expert.format(layer=layer, expert=expert, role=role))
            for role in self.spec.roles
        }


def read_manifest(variant_dir: Path) -> dict:
    """Validate a variant manifest; anything surprising is a structural skip."""
    variant_dir = Path(variant_dir)
    raw = _read_json(variant_dir / "manifest.json", "manifest")
    if not isinstance(raw, dict):
        raise DetectionError("manifest is not an object")
    layers = raw.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise DetectionError("manifest has no layers")
    if len(layers) > MAX_LAYERS:
        raise DetectionError(f"manifest declares {len(layers)} layers > {MAX_LAYERS}")
    counts: dict[int, int] = {}
    for key, value in layers.items():
        if not isinstance(value, dict):
            raise DetectionError(f"manifest layer {key!r} is not an object")
        try:
            layer = int(key)
        except (TypeError, ValueError) as exc:
            raise DetectionError(f"manifest layer {key!r} malformed: {exc}") from exc
        if key != str(layer) or not 0 <= layer < MAX_LAYERS:
            raise DetectionError(f"manifest layer key {key!r} is not canonical")
        if layer in counts:
            raise DetectionError(f"manifest layer {key!r} duplicated after normalisation")
        count = value.get("num_experts")
        if not isinstance(count, int) or isinstance(count, bool):
            raise DetectionError(f"manifest layer {key!r} num_experts is not an int")
        if not 1 <= count <= MAX_EXPERTS_PER_LAYER:
            raise DetectionError(f"manifest layer {key!r} num_experts {count} out of range")
        counts[layer] = count
    raw["layer_counts"] = counts
    tensor_file = raw.get("tensor_file", "model.safetensors")
    _checkpoint_file(variant_dir, tensor_file, "manifest")
    raw["tensor_file"] = str(tensor_file)
    return raw


def expert_slots(reader: CheckpointReader, layer: int) -> set[int]:
    """Expert slots that have at least one projection tensor for ``layer``."""
    template = reader.spec.expert
    if (
        template.count("{layer}") != 1
        or template.count("{expert}") != 1
        or "{role}" not in template
    ):
        raise DetectionError(f"unsupported expert template {template!r}")
    head, tail = template.split("{layer}", 1)
    slot_prefix, role_tail = tail.split("{expert}", 1)
    role_before, role_after = role_tail.split("{role}", 1)
    prefix = f"{head}{layer}{slot_prefix}"
    roles = set(reader.spec.roles)
    slots: set[int] = set()
    for name in reader.names():
        if not (name.startswith(prefix) and name.endswith(role_after)):
            continue
        middle = name[len(prefix) : len(name) - len(role_after)]
        slot_text, separator, role = middle.partition(role_before)
        if separator and slot_text.isdigit() and role in roles:
            slots.add(int(slot_text))
    return slots


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
    if parent_desc.shape[0] == 0 or suspect_desc.shape[0] == 0:
        raise DetectionError("empty descriptor matrix")
    cosine = parent_desc @ suspect_desc.T
    if not np.isfinite(cosine).all():
        raise DetectionError("non-finite cosine matrix")
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
        "max_pair_cos": float(cosine.max()),
        "max_candidate_cos": max((pair["cos"] for pair in candidates), default=None),
        "suspect_max_cos": [float(value) for value in cosine.max(axis=0)],
    }


# --------------------------------------------------------------------------- #
# constrained mixture fit
# --------------------------------------------------------------------------- #
def _nnls_sum_to_one(gram: np.ndarray, cross: np.ndarray) -> tuple[np.ndarray, bool]:
    """argmin a'Ga - 2b'a s.t. a >= 0, then rescale to sum(a) = 1.

    Solved through the PSD square root of the Gram matrix so only the k x k
    normal equations are touched; equivalent to NNLS on the stacked tensors.
    A LAPACK/NNLS failure is a degenerate fit, not an exception.
    """
    k = cross.size
    if k == 1:
        return np.ones(1), False
    try:
        values, vectors = np.linalg.eigh(gram)
    except (np.linalg.LinAlgError, ValueError):
        return np.zeros(k), True
    values = np.clip(values, 0.0, None)
    keep = values > max(float(values.max()), 1.0) * 1e-14
    if not keep.any():
        return np.zeros(k), True
    root = np.sqrt(values[keep])
    scale = root[:, None] * vectors[:, keep].T
    target = (vectors[:, keep].T @ cross) / root
    try:
        alpha, _ = nnls(scale, target, maxiter=100 * k)
    except (RuntimeError, np.linalg.LinAlgError, ValueError):
        return np.zeros(k), True
    if not np.isfinite(alpha).all():
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

    # A NaN/Inf outside the descriptor window must not silently turn every fit
    # into gain=NaN (which the merge test would read as "not merged").
    if not (np.isfinite(gram).all() and np.isfinite(cross).all() and np.isfinite(energy).all()):
        raise DetectionError("mixture statistics are non-finite")

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
                    "best_single_parent": None,
                    "degenerate": True,
                }
            )
            continue
        scores = np.where(usable, cross[:, j] ** 2 / np.where(usable, diagonal, 1.0), -np.inf)
        best_single = energy[j] - float(np.max(scores))
        best_index = int(np.argmax(scores))
        alpha, degenerate = _nnls_sum_to_one(gram, cross[:, j])
        rss_mix = max(float(energy[j] - 2.0 * alpha @ cross[:, j] + alpha @ gram @ alpha), 0.0)
        if best_single <= 1e-12 * max(float(energy[j]), 1e-30):
            gain = None
        else:
            gain = min(float(1.0 - rss_mix / best_single), 1.0)
        fits.append(
            {
                "suspect_index": j,
                "basis": list(range(k)),
                "alpha": [float(a) for a in alpha],
                "gain": gain,
                "best_single_parent": best_index,
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
    near_exact_cos: float = NEAR_EXACT_COS,
    parent_desc: np.ndarray | None = None,
) -> dict:
    """Descriptors, assignment, exact/candidate split, and mixture fits for one layer."""
    if not 1 <= int(suspect_experts) <= MAX_EXPERTS_PER_LAYER:
        raise DetectionError(
            f"suspect layer {suspect_layer}: expert count {suspect_experts} out of range"
        )
    present = expert_slots(suspect, suspect_layer)
    if len(present) != int(suspect_experts):
        raise DetectionError(
            f"suspect layer {suspect_layer} has {len(present)} expert tensors on disk, "
            f"manifest declares {suspect_experts}"
        )
    router_shape = suspect.shape(suspect.spec.router.format(layer=suspect_layer))
    if len(router_shape) != 2 or router_shape[0] != int(suspect_experts):
        raise DetectionError(
            f"router layer {suspect_layer} has shape {router_shape}, "
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
    # gain None (RSS of the best single expert is already ~0), so the fit does
    # not disturb copy detection; a shallow merge (e.g. 0.95/0.05) still fits
    # two parents with high gain even when cosine matching alone calls it an
    # exact copy.
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
        suspect_index = int(fit["suspect_index"])
        alpha = fit["alpha"]
        big = [i for i, a in enumerate(alpha) if a >= MIN_COMPONENT]
        gain = fit["gain"]
        duplicate_of = None
        if gain is not None and gain >= MERGE_GAIN_MIN and len(big) >= 2:
            label = "merged"
        elif assigned.get(suspect_index, 0.0) >= exact_cos:
            label = "exact"
        elif gain is None and alpha and max(alpha) >= 1.0 - MIN_COMPONENT:
            # Unassigned exact duplicate: the fit is one parent at full weight.
            label = "exact"
            duplicate_of = fit["best_single_parent"]
        else:
            label = "unexplained"
        classifications.append(
            {
                "suspect": suspect_index,
                "label": label,
                "cos": float(assigned.get(suspect_index, result["suspect_max_cos"][suspect_index])),
                "max_parent_cos": float(result["suspect_max_cos"][suspect_index]),
                "gain": gain,
                "basis": big,
                "alpha": [float(alpha[i]) for i in big],
                "best_single_parent": fit["best_single_parent"],
                "degenerate": bool(fit["degenerate"]),
                "duplicate_of": duplicate_of,
            }
        )

    merges = [
        {
            "suspect": entry["suspect"],
            "basis": entry["basis"],
            "alpha": entry["alpha"],
            "gain": entry["gain"],
            "cos": entry["cos"],
            "best_single_parent": entry["best_single_parent"],
            "degenerate": entry["degenerate"],
        }
        for entry in classifications
        if entry["label"] == "merged"
    ]

    def merge_ref(entry: dict) -> dict:
        return {
            "suspect": entry["suspect"],
            "cos": entry["cos"],
            "alpha": entry["alpha"],
            "gain": entry["gain"],
        }

    near_exact = [merge_ref(e) for e in merges if e["cos"] >= near_exact_cos]
    shadowed = [merge_ref(e) for e in merges if e["cos"] >= exact_cos]

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
        "shadowed_merges": shadowed,
        "max_pair_cos": result["max_pair_cos"],
        "max_candidate_cos": result["max_candidate_cos"],
        "max_assigned_cos": max(
            (pair["cos"] for pair in result["exact"] + result["candidates"]),
            default=None,
        ),
    }
