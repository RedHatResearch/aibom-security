"""GATE spectral-fingerprint detector for ProvenanceBench.

Paper: "Model-intrinsic, Weight-level, and Training-free Spectral
Fingerprinting for Large Language Models" (He, Chen, Xiong, Zhao, Wu;
arXiv:2610.30626, 22 Oct 2026). For each gate-like FFN matrix the detector
normalizes the retained top-r singular values, takes their spectral entropy,
pools the per-layer profile into s=8 depth segments, and scores two
checkpoints with linear CKA (squared Pearson on the 1-D profiles). Verdict:
related iff CKA >= tau = 0.9; no forward passes or outputs are involved.

Provenance: the numeric core below is the same audited implementation as
``smokes/gate-spectral/`` (paper-only reimplementation from Eq. 1-6). That
core passed an independent numeric audit to machine epsilon and the smoke's
offline selftest; this file inlines it so the benchmark has no
dependency on smoke code. House conventions are the pinned ones: fp32 SVD
input and fp64 entropy arithmetic, EPS=1e-12 in the Eq. (2) denominator
only, r_eff = min(r, q), per-layer arithmetic mean over gate tensors, s=8
``numpy.array_split`` pooling, squared-Pearson CKA, tau=0.9.

MoE extension (ours; not in the paper): He et al. define the feature only
for dense gated-FFN models and do not evaluate MoE checkpoints. Here every
gate-like FFN tensor per layer (routed-expert gates, a shared-expert gate,
and the dense FFN gate of dense layers inside an MoE model; router/logit
matrices excluded) contributes the paper's per-tensor statistic, and the
layer scalar is their arithmetic mean. That mean is invariant to expert
duplication (naive upcycling), unbiased under random expert subsetting
(expectation over uniformly drawn subsets), and degenerates exactly to the
paper on dense models. Pooling, linear CKA, and tau=0.9 are unchanged. This
extension is a design choice of this reproduction, untested by the paper's
authors.

Non-gated architectures are scored through the paper's own dagger-style
down-projection fallback: GPT-2 ``transformer.h.N.mlp.c_proj`` (the paper's
† auxiliary), GPTNeoX ``mlp.dense_4h_to_h``, and Nemotron-H
``mixer.down_proj``. Core-cohort numbers for those pairs
(``pb-core-nemotron-h-4b``, ``pb-core-gpt2-fineweb-fp8``) therefore read a
substituted down projection, not the gate branch.

Label blindness: the detector reads only the repo ids, revisions, and stack
passed to ``score()``. It never opens the curated pair manifest, labels,
evidence, or transformation kinds; revisions are passed straight to
``huggingface_hub`` so every run scores the same bytes.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gguf.quants
import ml_dtypes
import numpy as np
from gguf import GGUFReader
from huggingface_hub import (
    HfApi,
    get_session,
    get_token,
    hf_hub_download,
    hf_hub_url,
    snapshot_download,
)
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError

Verdict = Literal["related", "unrelated", "abstain"]

#: Eq. (2) denominator epsilon (fp64-safe; typical singular sums are 1e2-1e4).
EPS = 1e-12
#: Paper default retained-rank budget.
R_DEFAULT = 128
#: Paper default depth-segment count.
SEGMENTS = 8
#: A-priori operating point (never tuned on labels).
TAU = 0.9
#: |r| flag for the anticorrelation-collision diagnostic.
ANTICORRELATION_R = -0.7
#: Roles a gate-like tensor can carry; used by the MoE sensitivity variants.
ROLE_ROUTED = "routed"
ROLE_SHARED = "shared"

#: Safetensors layout: 8-byte little-endian header length, JSON header, data.
_HEADER_BYTES = 8
#: Plausibility bound for a shard header (guards a non-safetensors response).
_HEADER_MAX_BYTES = 100 * 1024 * 1024
#: Largest element count one tensor may declare (int64 index space).
_MAX_ELEMENT_COUNT = 2**63
#: Timeout for one range GET: a stalled server must not hang the benchmark.
_RANGE_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class Result:
    verdict: Verdict
    score: float | None
    diagnostics: dict[str, object]
    note: str = ""
    abstain_reason: str | None = None


# ---------------------------------------------------------------------------
# Spectral core (identical semantics to smokes/gate-spectral/_gate.py).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorSpectrum:
    """One gate-like FFN tensor reduced to its singular-value spectrum.

    Computed at extraction time: the gate matrix itself is dropped as soon as
    its spectrum exists, so a whole checkpoint costs O(sv) instead of O(weights).
    """

    name: str
    role: str
    sv: np.ndarray
    shape: tuple[int, ...] = ()

    @property
    def q(self) -> int:
        """Paper's q = min(d_m, d_f) = SV count; invariant to orientation."""
        return int(self.sv.size)

    @property
    def r_eff(self) -> int:
        """Retained rank at the paper default budget (diagnostics only)."""
        return int(min(R_DEFAULT, self.q))

    @property
    def spectrum_fraction(self) -> float:
        """Retained fraction of the spectrum at the paper default budget."""
        return (self.r_eff / self.q) if self.q else 0.0


def _svd_spectrum(matrix: np.ndarray) -> np.ndarray:
    """Singular values of a gate matrix, descending (fp32 SVD, fp64 output)."""
    w = np.asarray(matrix, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"gate matrix must be 2-D, got shape {w.shape}")
    return np.linalg.svd(w, compute_uv=False).astype(np.float64)


def _entropy_feature(sv: Sequence[float] | np.ndarray, r: int = R_DEFAULT) -> dict:
    """Eq. (2)-(4): retained-top-r entropy feature, r_eff = min(r, q)."""
    spectrum = np.asarray(sv, dtype=np.float64)
    q = int(spectrum.size)
    r_eff = int(min(int(r), q))
    feature = {
        "q": q,
        "r_eff": r_eff,
        "spectrum_fraction": (r_eff / q) if q else 0.0,
        "all_zero": False,
        "H": 0.0,
        "f": 1.0,
    }
    if r_eff == 0:
        feature["all_zero"] = True
        return feature
    top = spectrum[:r_eff]
    total = float(top.sum())
    if not np.isfinite(total) or total <= 0.0:
        feature["all_zero"] = True
        return feature
    sigma_bar = top / (total + EPS)
    positive = sigma_bar[sigma_bar > 0.0]
    h = float(-np.sum(positive * np.log(positive)))
    feature["H"] = h
    feature["f"] = float(np.exp(h))
    return feature


def _layer_features(
    layers: Mapping[int, Sequence[TensorSpectrum]],
    r: int = R_DEFAULT,
    roles: Iterable[str] | None = None,
) -> list[dict]:
    """Per-layer scalars, shallow -> deep; arithmetic mean over gate tensors."""
    allowed = None if roles is None else frozenset(roles)
    features: list[dict] = []
    for layer in sorted(layers, key=int):
        tensors = [t for t in layers[layer] if allowed is None or t.role in allowed]
        if not tensors:
            continue
        per_tensor = []
        for tensor in tensors:
            feature = _entropy_feature(tensor.sv, r)
            feature["name"] = tensor.name
            feature["role"] = tensor.role
            per_tensor.append(feature)
        features.append(
            {
                "layer": int(layer),
                "f": float(np.mean([t["f"] for t in per_tensor])),
                "tensor_count": len(per_tensor),
                "tensors": per_tensor,
            }
        )
    return features


def _geometric_layer_features(
    layers: Mapping[int, Sequence[TensorSpectrum]], r: int = R_DEFAULT
) -> list[dict]:
    """Diagnostic variant: geometric mean of per-tensor exp(H) (never verdict)."""
    features: list[dict] = []
    for layer in sorted(layers, key=int):
        tensors = list(layers[layer])
        if not tensors:
            continue
        entropies = [_entropy_feature(tensor.sv, r)["H"] for tensor in tensors]
        features.append(
            {
                "layer": int(layer),
                "f": float(np.exp(np.mean(entropies))),
                "tensor_count": len(tensors),
            }
        )
    return features


def _concatenated_layer_features(
    layers: Mapping[int, Sequence[TensorSpectrum]], r: int = R_DEFAULT
) -> list[dict]:
    """Diagnostic variant: entropy of concatenated top-r_eff spectra (never verdict)."""
    features: list[dict] = []
    for layer in sorted(layers, key=int):
        tensors = list(layers[layer])
        if not tensors:
            continue
        parts = []
        for tensor in tensors:
            spectrum = np.asarray(tensor.sv, dtype=np.float64)
            parts.append(spectrum[: min(int(r), spectrum.size)])
        concatenated = np.concatenate(parts)
        feature = _entropy_feature(concatenated, r=int(concatenated.size))
        features.append(
            {
                "layer": int(layer),
                "f": feature["f"],
                "tensor_count": len(tensors),
            }
        )
    return features


def _segment_sizes(n_items: int, s: int = SEGMENTS) -> list[int]:
    """Sizes of ``numpy.array_split``'s contiguous ~equal groups (first get extras)."""
    return [int(group.size) for group in np.array_split(np.arange(n_items), s)]


def _pool_segments(values: Sequence[float] | np.ndarray, s: int = SEGMENTS) -> np.ndarray:
    """Eq. (5): mean-pool the depth profile into s contiguous ~equal segments."""
    profile = np.asarray(list(values), dtype=np.float64)
    if profile.size == 0:
        return np.full(s, np.nan)
    return np.array(
        [float(group.mean()) if group.size else np.nan for group in np.array_split(profile, s)]
    )


def _cka_linear_1d(
    a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray
) -> tuple[float | None, float | None]:
    """Linear CKA between 1-D profiles: (CKA, signed Pearson), or (None, None)."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"profiles must be equal-length 1-D vectors: {x.shape} vs {y.shape}")
    if x.size < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None, None
    xc = x - x.mean()
    yc = y - y.mean()
    xx = float(np.dot(xc, xc))
    yy = float(np.dot(yc, yc))
    if xx <= 0.0 or yy <= 0.0:
        return None, None
    r = float(np.dot(xc, yc) / np.sqrt(xx * yy))
    r = float(np.clip(r, -1.0, 1.0))
    return r * r, r


def _similarity(
    profile_a: Sequence[float] | np.ndarray, profile_b: Sequence[float] | np.ndarray
) -> dict:
    """CKA + signed Pearson + the anticorrelation-collision flag."""
    cka, signed = _cka_linear_1d(profile_a, profile_b)
    return {
        "cka": cka,
        "signed_pearson": signed,
        "anticorrelation_collision": bool(
            cka is not None and signed is not None and signed <= ANTICORRELATION_R and cka >= TAU
        ),
    }


def _verdict_for(cka: float | None, tau: float = TAU) -> str | None:
    """A-priori rule: related iff CKA >= tau; abstain for a missing CKA."""
    if cka is None:
        return None
    return "related" if cka >= tau else "unrelated"


def _profile_cv(profile: Sequence[float]) -> float:
    """Coefficient of variation of the pooled profile (std / mean)."""
    values = np.asarray(profile, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return float("nan")
    mean = float(values.mean())
    if mean == 0.0:
        return float("nan")
    return float(values.std() / mean)


def _compare(
    layers_a: Mapping[int, Sequence[TensorSpectrum]],
    layers_b: Mapping[int, Sequence[TensorSpectrum]],
    r: int = R_DEFAULT,
    s: int = SEGMENTS,
    roles: Iterable[str] | None = None,
) -> dict:
    """Profile both sides and score them (structural abstains included)."""
    features_a = _layer_features(layers_a, r, roles)
    features_b = _layer_features(layers_b, r, roles)
    profile_a = _pool_segments([layer["f"] for layer in features_a], s)
    profile_b = _pool_segments([layer["f"] for layer in features_b], s)
    sim = _similarity(profile_a, profile_b)
    if not features_a or not features_b:
        status = "no_gate_tensors"
    elif sim["cka"] is None:
        status = "degenerate_profile"
    else:
        status = "ok"
    return {
        "status": status,
        "profile": {"a": profile_a.tolist(), "b": profile_b.tolist()},
        "features": {"a": features_a, "b": features_b},
        "tensor_counts": {
            "a": sum(layer["tensor_count"] for layer in features_a),
            "b": sum(layer["tensor_count"] for layer in features_b),
        },
        **sim,
        "verdict": _verdict_for(sim["cka"]),
    }


def _diagnostic_variant(
    layers_a: Mapping[int, Sequence[TensorSpectrum]],
    layers_b: Mapping[int, Sequence[TensorSpectrum]],
    builder: Callable[[Mapping[int, Sequence[TensorSpectrum]], int], list[dict]],
    r: int = R_DEFAULT,
    s: int = SEGMENTS,
) -> dict:
    """Status + CKA of one diagnostic profile builder (no verdict)."""
    features_a = builder(layers_a, r)
    features_b = builder(layers_b, r)
    sim = _similarity(
        _pool_segments([layer["f"] for layer in features_a], s),
        _pool_segments([layer["f"] for layer in features_b], s),
    )
    if not features_a or not features_b:
        status = "no_gate_tensors"
    elif sim["cka"] is None:
        status = "degenerate_profile"
    else:
        status = "ok"
    return {"status": status, "cka": sim["cka"]}


def _sensitivity_panel(
    layers_a: Mapping[int, Sequence[TensorSpectrum]],
    layers_b: Mapping[int, Sequence[TensorSpectrum]],
    primary: dict,
    r: int = R_DEFAULT,
    s: int = SEGMENTS,
) -> dict:
    """Diagnostic variants of the primary comparison; never used for the verdict."""
    panel: dict[str, dict] = {}
    for variant_r in (64, 256):
        comparison = _compare(layers_a, layers_b, r=variant_r, s=s)
        panel[f"r{variant_r}"] = {"status": comparison["status"], "cka": comparison["cka"]}
    for variant_s in (4, 16):
        sim = _similarity(
            _pool_segments([layer["f"] for layer in primary["features"]["a"]], variant_s),
            _pool_segments([layer["f"] for layer in primary["features"]["b"]], variant_s),
        )
        panel[f"s{variant_s}"] = {
            "status": "ok" if sim["cka"] is not None else "degenerate_profile",
            "cka": sim["cka"],
        }
    for role, key in ((ROLE_ROUTED, "experts_only"), (ROLE_SHARED, "shared_only")):
        comparison = _compare(layers_a, layers_b, r=r, s=s, roles={role})
        if comparison["status"] == "no_gate_tensors":
            panel[key] = {"status": "unavailable", "cka": None}
        else:
            panel[key] = {"status": comparison["status"], "cka": comparison["cka"]}
    for key, builder in (
        ("geometric", _geometric_layer_features),
        ("concatenated", _concatenated_layer_features),
    ):
        panel[key] = _diagnostic_variant(layers_a, layers_b, builder, r, s)
    return panel


# ---------------------------------------------------------------------------
# Family-aware gate-tensor extraction (same table as smokes/gate-spectral).
# ---------------------------------------------------------------------------

AWQ_BITS = 4
AWQ_PACK_NUM = 32 // AWQ_BITS
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

#: Quant schemes with no dequantizer here; they abstain as format_unreadable.
#: (MXFP4 is supported: see ``dequantize_mxfp4``.)
UNSUPPORTED_QUANT_METHODS = frozenset({"aqlm", "gptq", "quip", "gsq"})

#: Download retry policy (matches smokes/independence-tests).
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_BACKOFF_SECONDS = (2.0, 8.0)

SAFETENSORS_DTYPES: dict[str, object] = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": ml_dtypes.bfloat16,
    "F8_E4M3": ml_dtypes.float8_e4m3fn,
    "F8_E5M2": ml_dtypes.float8_e5m2,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U64": np.uint64,
    "U32": np.uint32,
    "U16": np.uint16,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


class FormatUnreadable(Exception):
    """The checkpoint format cannot be consumed by this detector."""


@dataclass
class Extraction:
    """Per-layer gate spectra plus provenance for diagnostics."""

    layers: dict[int, list[TensorSpectrum]]
    family: str
    format: str
    source: dict

    @property
    def n_tensors(self) -> int:
        return sum(len(tensors) for tensors in self.layers.values())


# (regex, kind, role). Anchored so router names can never sneak in through a
# prefix match; ``is_router`` is an explicit second gate.
_NAME_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$"), "dense_gate", "dense"),
    (
        re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight$"),
        "expert_gate",
        "routed",
    ),
    (
        re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w1\.weight$"),
        "expert_gate",
        "routed",
    ),
    (
        # LFM2.5 discrete experts (pinned 5dd22602 layout; w1 is the gate).
        re.compile(r"^model\.layers\.(\d+)\.feed_forward\.experts\.(\d+)\.w1\.weight$"),
        "expert_gate",
        "routed",
    ),
    (
        re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.input_linear\.weight$"),
        "granite_fused",
        "routed",
    ),
    (re.compile(r"^blocks\.(\d+)\.ffn\.gate$"), "upcycle_routed", "routed"),
    (
        re.compile(r"^blocks\.(\d+)\.ffn\.shared_expert\.gate\.weight$"),
        "upcycle_shared",
        "shared",
    ),
    (re.compile(r"^blocks\.(\d+)\.ffn\.gate\.weight$"), "upcycle_dense", "dense"),
    (
        re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj(?:\.weight)?$"),
        "gptoss_fused",
        "routed",
    ),
    (
        re.compile(r"^model\.layers\.(\d+)\.feed_forward\.experts\.gate_up_proj(?:\.weight)?$"),
        "lfm2_fused",
        "routed",
    ),
    (re.compile(r"^model\.layers\.(\d+)\.feed_forward\.w1\.weight$"), "dense_gate", "dense"),
    # GPT-2 paper-† fallback. Two observed spellings for the same down
    # projection: the transformers export prefixes "transformer." while the
    # raw openai-community/gpt2 state dict has bare "h.N.mlp.c_proj.weight".
    (
        re.compile(r"^(?:transformer\.)?h\.(\d+)\.mlp\.c_proj\.weight$"),
        "gpt2_fallback",
        "dense",
    ),
    (
        # GPTNeoX/Pythia: no gate exists; the down projection is the disclosed
        # non-gated fallback (dense_h_to_4h is the UP branch and is never a gate).
        re.compile(r"^gpt_neox\.layers\.(\d+)\.mlp\.dense_4h_to_h\.weight$"),
        "gpt_neox_fallback",
        "dense",
    ),
    (
        # Qwen3.5 nests the decoder under language_model; anchored so the
        # mtp.layers.* speculative-decoding head can never match.
        re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate_proj\.weight$"),
        "dense_gate",
        "dense",
    ),
    (
        # Nemotron-H relu2 MLP: no gate exists; down projection as the
        # disclosed non-gated fallback.
        re.compile(r"^backbone\.layers\.(\d+)\.mixer\.down_proj\.weight$"),
        "nemotron_fallback",
        "dense",
    ),
    (
        # TinyMixtral legacy .bin layout: axis 0 is the expert axis but the rows
        # are already gate-only (no row split, no de-interleave).
        re.compile(r"^layers\.(\d+)\.moe\.gate_proj$"),
        "tinymixtral_stacked",
        "routed",
    ),
    (re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$"), "gguf_exps", "routed"),
    (re.compile(r"^blk\.(\d+)\.ffn_gate\.weight$"), "gguf_gate", "dense"),
)

_ROUTER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^model\.layers\.\d+\.mlp\.gate\.weight$"),  # Qwen3-MoE / OLMoE
    re.compile(r"^model\.layers\.\d+\.block_sparse_moe\.gate\.weight$"),  # Mixtral
    re.compile(r"^model\.layers\.\d+\.block_sparse_moe\.router\."),  # granite
    re.compile(r"^blocks\.\d+\.ffn\.router\."),  # AlexWortega upcycle
    re.compile(r"^model\.layers\.\d+\.mlp\.router\."),  # gpt-oss
    re.compile(r"^model\.layers\.\d+\.feed_forward\.gate\.weight$"),  # LFM2
    re.compile(r"^blk\.\d+\.ffn_gate_inp\.weight$"),  # GGUF (granite/LFM2)
)

#: Stacked-expert kinds: axis 0 is the expert axis and needs per-expert slicing.
_STACKED_KINDS = frozenset(
    {
        "granite_fused",
        "gptoss_fused",
        "lfm2_fused",
        "upcycle_routed",
        "gguf_exps",
        "tinymixtral_stacked",
    }
)

_FAMILY_PRIORITY: tuple[tuple[str, str], ...] = (
    ("granite_fused", "granite_moe"),
    ("gguf_exps", "granite_moe"),
    ("upcycle_routed", "upcycle"),
    ("upcycle_shared", "upcycle"),
    ("upcycle_dense", "upcycle"),
    ("gptoss_fused", "gpt_oss"),
    ("mxfp4_gate", "gpt_oss"),
    ("lfm2_fused", "lfm2"),
    ("tinymixtral_stacked", "tinymixtral"),
    ("expert_gate", "moe_experts"),
    ("dense_gate", "dense"),
    ("gguf_gate", "dense"),
    ("gpt2_fallback", "gpt2_fallback"),
    ("gpt_neox_fallback", "gpt_neox_fallback"),
    ("nemotron_fallback", "nemotron_h_fallback"),
)


def _is_router(name: str) -> bool:
    """True for router/logit matrices (excluded in every family)."""
    return any(pattern.match(name) for pattern in _ROUTER_PATTERNS)


def _classify(name: str) -> tuple[str, int, str] | None:
    """Map a weight name to (kind, integer layer, role), or None if not a gate."""
    if _is_router(name):
        return None
    for pattern, kind, role in _NAME_PATTERNS:
        match = pattern.match(name)
        if match:
            return kind, int(match.group(1)), role
    return None


def _family_from_kinds(kinds: Iterable[str]) -> str:
    """Family label from the gate-tensor kinds present (diagnostics only)."""
    present = set(kinds)
    for kind, family in _FAMILY_PRIORITY:
        if kind in present:
            return family
    return "unknown"


def _split_stacked(kind: str, name: str, array: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Slice a stacked expert tensor into (tensor name, gate matrix) pairs."""
    if array.ndim != 3:
        raise FormatUnreadable(f"{name}: expected a 3-D stacked tensor, got {array.shape}")
    n_experts, rows, _ = array.shape
    if kind == "granite_fused":
        # JetMoe-style fused gate+up: chunk(2) along dim -2, first chunk = gate.
        if rows % 2:
            raise FormatUnreadable(f"{name}: fused gate+up rows not even ({rows})")
        gates = array[:, : rows // 2, :]
    elif kind == "gptoss_fused":
        # Gate and up are interleaved on the last axis; even columns = gate.
        gates = array[:, :, ::2]
    elif kind == "lfm2_fused":
        if rows % 2:
            raise FormatUnreadable(f"{name}: fused gate+up rows not even ({rows})")
        gates = array[:, : rows // 2, :]
    else:  # upcycle_routed / gguf_exps / tinymixtral_stacked: gate-only expert stacks
        gates = array
    return [(f"{name}#expert{expert}", gates[expert]) for expert in range(n_experts)]


def _split_tensor(kind: str, name: str, array: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Expand one weight tensor into (tensor name, gate matrix) pairs."""
    if kind in _STACKED_KINDS:
        return _split_stacked(kind, name, array)
    return [(name, array)]


def _store_matrix(
    layers: dict[int, list[TensorSpectrum]], name: str, matrix: np.ndarray, role: str, layer: int
) -> None:
    """SVD one gate matrix now and keep only its spectrum (peak RAM = one matrix)."""
    spectrum = TensorSpectrum(
        name=name, role=role, sv=_svd_spectrum(matrix), shape=tuple(np.shape(matrix))
    )
    layers.setdefault(layer, []).append(spectrum)


def _accumulate(
    layers: dict[int, list[TensorSpectrum]],
    kinds: set[str],
    items: Iterable[tuple[str, np.ndarray]],
) -> None:
    """Classify, slice, SVD, and store one tensor at a time (only spectra are kept)."""
    for name, array in items:
        classified = _classify(name)
        if classified is None:
            continue
        kind, layer, role = classified
        kinds.add(kind)
        for tensor_name, matrix in _split_tensor(kind, name, array):
            _store_matrix(layers, tensor_name, matrix, role, layer)


def _finish_extraction(
    layers: dict[int, list[TensorSpectrum]], kinds: Iterable[str], fmt: str, source: dict
) -> Extraction:
    """Sorted per-layer spectra plus family/format/provenance in the contract shape."""
    for tensors_in_layer in layers.values():
        tensors_in_layer.sort(key=lambda tensor: tensor.name)
    return Extraction(layers=layers, family=_family_from_kinds(kinds), format=fmt, source=source)


def _dequantize_awq_gemm(
    qweight: np.ndarray,
    qzeros: np.ndarray,
    scales: np.ndarray,
    group_size: int,
    bits: int = AWQ_BITS,
) -> np.ndarray:
    """Dequantize an AWQ GEMM int4 layer (AutoAWQ ``dequantize_gemm`` semantics)."""
    shifts = np.arange(0, 32, bits, dtype=np.int32)
    mask = (1 << bits) - 1
    iweight = (qweight[:, :, None] >> shifts) & mask
    iweight = iweight.reshape(qweight.shape[0], -1)
    izeros = (qzeros[:, :, None] >> shifts) & mask
    izeros = izeros.reshape(qzeros.shape[0], -1)
    # Undo the [0,2,4,6,1,3,5,7] packing order within each int32 column:
    # AutoAWQ reverses per 8-wide block, so each block gets its own offset.
    n_cols = iweight.shape[1]
    if n_cols % AWQ_PACK_NUM:
        raise FormatUnreadable(f"AWQ packed columns {n_cols} are not a multiple of {AWQ_PACK_NUM}")
    blocks = np.arange(n_cols).reshape(-1, AWQ_PACK_NUM)
    reverse = blocks[:, AWQ_REVERSE_ORDER].reshape(-1)
    iweight = iweight[:, reverse]
    izeros = izeros[:, reverse]
    izeros = np.repeat(izeros, group_size, axis=0)
    scales_full = np.repeat(scales, group_size, axis=0).astype(np.float32)
    return (iweight - izeros).astype(np.float32) * scales_full


#: E2M1 FP4 codebook: index = 4-bit code, sign bit 3.
FP4_VALUES = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32
)
#: Packed FP4 codes (and dequantized elements) per E8M0 scale group.
MXFP4_GROUP = 32


def dequantize_mxfp4(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize MXFP4 (E2M1 codes, E8M0 block scales) into fp32.

    ``blocks`` is uint8 ``[..., rows, groups, 16]``: 16 bytes are 32 nibble
    codes, low nibble = even element. ``scales`` is uint8 ``[..., rows, groups]``
    with value ``2 ** (int(scale) - 127)``. The result ``[..., rows, groups*32]``
    keeps the GEMM's own layout (the input axis runs last inside each group), so
    the Hugging Face weight is ``result.swapaxes(-1, -2)``.
    """
    packed = np.asarray(blocks)
    scale_codes = np.asarray(scales)
    if packed.dtype != np.uint8 or scale_codes.dtype != np.uint8:
        raise FormatUnreadable(f"MXFP4 needs uint8 blocks/scales, got {packed.dtype}")
    if packed.ndim < 2 or packed.shape[-1] != 16:
        raise FormatUnreadable(f"MXFP4 blocks need 16 packed bytes per group, got {packed.shape}")
    if packed.shape[:-1] != scale_codes.shape:
        raise FormatUnreadable(
            f"MXFP4 blocks {packed.shape} and scales {scale_codes.shape} disagree"
        )
    codes = np.empty(packed.shape[:-1] + (MXFP4_GROUP,), dtype=np.uint8)
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = packed >> 4
    # ldexp is the reference's exact 2**(scale-127) scaling; unlike a multiply by
    # np.exp2 it cannot turn a zero code into 0*inf = nan at overflowed scales.
    exponent = scale_codes.astype(np.int32) - 127
    with np.errstate(over="ignore"):  # degenerate E8M0 codes overflow to inf, as in torch
        values = np.ldexp(FP4_VALUES[codes], exponent[..., None])
    return values.reshape(*values.shape[:-2], values.shape[-2] * MXFP4_GROUP)


#: Packed MXFP4 expert gate+up blocks (gpt-oss); scales sit beside them.
_MXFP4_GATE_BLOCKS = re.compile(r"\.experts\.gate_up_proj_blocks$")

#: Deepest ``layers.N`` index in a weight name (nested decoder layouts included).
_LAYER_IN_NAME = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _last_layer_index(name: str) -> int | None:
    """Integer layer index of a weight name, or None when it carries no depth."""
    matches = _LAYER_IN_NAME.findall(name)
    return int(matches[-1]) if matches else None


def _read_header(path: Path) -> tuple[dict, int]:
    """Header dict and absolute data offset of a safetensors shard.

    The header is untrusted: its length prefix, its JSON, and every entry are
    validated before any tensor bytes are sought, so a malformed shard can
    neither request a multi-gigabyte read nor crash outside ``FormatUnreadable``.
    """
    size = path.stat().st_size
    with open(path, "rb") as handle:
        prefix = handle.read(_HEADER_BYTES)
        if len(prefix) != _HEADER_BYTES:
            raise FormatUnreadable(
                f"{path.name}: shorter than the {_HEADER_BYTES}-byte header prefix"
            )
        header_len = int.from_bytes(prefix, "little")
        if not 0 < header_len <= _HEADER_MAX_BYTES:
            raise FormatUnreadable(f"{path.name}: implausible header length {header_len}")
        if _HEADER_BYTES + header_len > size:
            raise FormatUnreadable(
                f"{path.name}: header length {header_len} runs past the file end ({size} bytes)"
            )
        raw = handle.read(header_len)
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormatUnreadable(f"{path.name}: header is not valid JSON ({exc})") from exc
    if not isinstance(header, dict) or not all(
        isinstance(entry, dict) for entry in header.values()
    ):
        raise FormatUnreadable(f"{path.name}: header is not a dict of tensor entries")
    return header, _HEADER_BYTES + header_len


def _decode_tensor(
    raw: bytes, dtype: object, shape: tuple[int, ...], count: int, label: str
) -> np.ndarray:
    """Decode tensor bytes by safetensors dtype tag (bf16/float8 preserved)."""
    if len(raw) != count * np.dtype(dtype).itemsize:
        raise FormatUnreadable(f"{label}: byte length mismatch for {shape}")
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


@dataclass(frozen=True)
class _TensorRef:
    """Header entry for one tensor: where its bytes live and what they mean."""

    filename: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


def _tensor_ref(
    filename: str, dtype: object, shape: Iterable[object], data_offsets: Iterable[object]
) -> _TensorRef:
    """Typed reference from a Hub metadata object or a parsed header entry."""
    begin, end = (int(offset) for offset in data_offsets)
    return _TensorRef(
        filename=filename,
        dtype=str(dtype),
        shape=tuple(int(dimension) for dimension in shape),
        data_offsets=(begin, end),
    )


def _checked_layout(ref: _TensorRef) -> tuple[object, tuple[int, ...], int]:
    """Validated ``(dtype, shape, element count)`` for one tensor reference."""
    try:
        dtype = SAFETENSORS_DTYPES[ref.dtype]
    except (KeyError, TypeError) as exc:
        # TypeError covers an unhashable tag (e.g. a list); KeyError an unknown one.
        raise FormatUnreadable(
            f"{ref.filename}: unsupported safetensors dtype {ref.dtype!r}"
        ) from exc
    shape = ref.shape
    if any(dimension < 0 for dimension in shape):
        raise FormatUnreadable(f"{ref.filename}: negative shape dimension in {shape}")
    count = math.prod(shape)
    if not 0 < count <= _MAX_ELEMENT_COUNT:
        raise FormatUnreadable(f"{ref.filename}: implausible element count {count} for {shape}")
    begin, end = ref.data_offsets
    if begin < 0 or end < begin:
        raise FormatUnreadable(f"{ref.filename}: invalid data_offsets [{begin}, {end}]")
    if end - begin != count * np.dtype(dtype).itemsize:
        raise FormatUnreadable(f"{ref.filename}: byte length mismatch for {ref.dtype} {shape}")
    return dtype, shape, count


def _read_range_body(response, expected: int, label: str) -> bytes:
    """Read at most ``expected`` bytes of a partial-content response."""
    if "content-range" not in response.headers:
        raise RuntimeError(f"{label}: HTTP 206 without Content-Range")
    body = bytearray()
    for chunk in response.iter_bytes():
        body += chunk[: max(expected - len(body), 0)]
        if len(body) >= expected:
            break
    if len(body) != expected:
        raise RuntimeError(f"{label}: short read ({len(body)}/{expected} bytes)")
    return bytes(body)


class _SafetensorsReader:
    """Per-tensor random access to one pinned safetensors checkpoint."""

    def __init__(self, refs: Mapping[str, _TensorRef], files: Sequence[str]) -> None:
        self._refs = dict(refs)
        self.files = list(files)

    def names(self) -> list[str]:
        """Every tensor name in the checkpoint, sorted."""
        return sorted(self._refs)

    def __contains__(self, name: str) -> bool:
        return name in self._refs

    def dtype(self, name: str) -> str:
        """Safetensors dtype tag of one tensor."""
        return self._refs[name].dtype

    def read(self, name: str) -> np.ndarray:
        raise NotImplementedError


class _RangeReader(_SafetensorsReader):
    """HTTP range reads: only the tensors we ask for cross the wire.

    ``HfApi.get_safetensors_metadata`` supplies dtype/shape/data_offsets for
    every tensor, so a shard costs one 8-byte range GET (its header length) plus
    one request per tensor actually needed — no multi-hundred-GB snapshot.
    """

    def __init__(self, repo_id: str, revision: str | None, step: int | None = None) -> None:
        metadata = _retry_call(
            f"safetensors metadata {repo_id}",
            lambda: HfApi().get_safetensors_metadata(repo_id, revision=revision),
        )
        prefix = None if step is None else f"step_{step}/"  # never mix training steps
        refs: dict[str, _TensorRef] = {}
        for filename, file_metadata in metadata.files_metadata.items():
            if prefix is not None and not str(filename).startswith(prefix):
                continue
            for name, info in file_metadata.tensors.items():
                refs[name] = _tensor_ref(str(filename), info.dtype, info.shape, info.data_offsets)
        if not refs:
            raise RuntimeError(f"{repo_id}: no safetensors tensors at this revision")
        super().__init__(refs, sorted({ref.filename for ref in refs.values()}))
        self._repo_id = repo_id
        self._revision = revision
        self._token = get_token()
        self._urls: dict[str, str] = {}
        self._header_lengths: dict[str, int] = {}

    def _url(self, filename: str) -> str:
        if filename not in self._urls:
            self._urls[filename] = hf_hub_url(self._repo_id, filename, revision=self._revision)
        return self._urls[filename]

    def _get(self, filename: str, start: int, end: int) -> bytes:
        """One streaming inclusive-range GET with the Hub token and retry policy."""
        if start < 0 or end < start:
            raise RuntimeError(f"{filename!r}: invalid range [{start}, {end}]")
        url = self._url(filename)
        headers = {"Range": f"bytes={start}-{end}"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        expected = end - start + 1

        def fetch() -> bytes:
            # stream=True: the 206 check happens before any body byte is
            # buffered, so a server that ignores Range (200 plus the whole
            # shard, or an intermediary that strips the header) cannot make
            # this process hold a shard in memory.
            with get_session().stream(
                "GET", url, headers=headers, timeout=_RANGE_TIMEOUT_SECONDS
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(
                        f"{filename!r}: server ignored the range request "
                        f"(HTTP {response.status_code})"
                    )
                return _read_range_body(response, expected, repr(filename))

        return _retry_call(f"range {filename!r}[{start}:{end}]", fetch)

    def _header_length(self, filename: str) -> int:
        """u64 header-length prefix: one 8-byte GET per shard, cached."""
        if filename not in self._header_lengths:
            raw = self._get(filename, 0, _HEADER_BYTES - 1)
            if len(raw) != _HEADER_BYTES:
                raise FormatUnreadable(f"{filename}: short header prefix ({len(raw)} bytes)")
            header_len = int.from_bytes(raw, "little")
            if not 0 < header_len <= _HEADER_MAX_BYTES:
                raise FormatUnreadable(f"{filename}: implausible header length {header_len}")
            self._header_lengths[filename] = header_len
        return self._header_lengths[filename]

    def read(self, name: str) -> np.ndarray:
        ref = self._refs[name]
        dtype, shape, count = _checked_layout(ref)
        begin, end = ref.data_offsets
        start = _HEADER_BYTES + self._header_length(ref.filename) + begin
        raw = self._get(ref.filename, start, start + (end - begin) - 1)
        return _decode_tensor(raw, dtype, shape, count, name)


class _SnapshotReader(_SafetensorsReader):
    """Per-tensor reads from shard files already in the shared Hugging Face cache."""

    def __init__(self, root: Path, step: int | None) -> None:
        shards = sorted(root.rglob("*.safetensors"))
        if step is not None:
            shards = [
                shard for shard in shards if shard.relative_to(root).parts[0] == f"step_{step}"
            ]
        if not shards:
            raise FormatUnreadable(f"{root.name}: no *.safetensors in the pinned snapshot")
        self._shards: dict[str, tuple[Path, int, int]] = {}
        refs: dict[str, _TensorRef] = {}
        for shard in shards:
            filename = str(shard.relative_to(root))
            size = shard.stat().st_size
            header, data_start = _read_header(shard)
            self._shards[filename] = (shard, data_start, size)
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                try:
                    entry = (info["dtype"], info["shape"], info["data_offsets"])
                except (KeyError, TypeError) as exc:
                    raise FormatUnreadable(
                        f"{filename}: tensor {name!r} entry is malformed ({type(exc).__name__})"
                    ) from exc
                refs[name] = _tensor_ref(filename, *entry)
        super().__init__(refs, sorted(self._shards))

    def read(self, name: str) -> np.ndarray:
        ref = self._refs[name]
        dtype, shape, count = _checked_layout(ref)
        path, data_start, size = self._shards[ref.filename]
        begin, end = ref.data_offsets
        length = end - begin
        if length <= 0 or length > size - data_start:
            raise FormatUnreadable(
                f"{ref.filename}: tensor bytes [{begin}, {end}) run past the data section"
            )
        with open(path, "rb") as handle:
            handle.seek(data_start + begin)
            # length is validated positive: handle.read(-1) would buffer the
            # entire remaining shard, which must never be reachable.
            raw = handle.read(length)
        return _decode_tensor(raw, dtype, shape, count, name)


def _scaled_gate(
    array: np.ndarray, name: str, dtype_name: str, scale: np.ndarray | None
) -> np.ndarray:
    """Fold a compressed-tensors ``weight_scale`` into an FP8 gate when present."""
    if not dtype_name.startswith("F8") or scale is None:
        return array
    scale = np.asarray(scale, dtype=np.float32).reshape(-1, 1)
    if scale.shape[0] not in (1, array.shape[-2]):
        raise FormatUnreadable(
            f"{name}_scale: {scale.shape[0]} scale rows do not match {name} rows {array.shape[-2]}"
        )
    return array.astype(np.float32) * scale


def _gate_items(reader: _SafetensorsReader) -> Iterator[tuple[str, np.ndarray]]:
    """Gate-like tensors one at a time, fp8 scales folded in (streaming reads)."""
    for name in reader.names():
        if _classify(name) is None:
            continue
        scale_name = name + "_scale"
        scale = reader.read(scale_name) if scale_name in reader else None
        yield name, _scaled_gate(reader.read(name), name, reader.dtype(name), scale)


def _awq_items(reader: _SafetensorsReader) -> list[tuple[str, np.ndarray]]:
    """Dequantize int4-packed gate projections into ``.weight`` matrices."""
    packed: dict[str, dict[str, np.ndarray]] = {}
    for name in reader.names():
        base, _, field = name.rpartition(".")
        if field in ("qweight", "qzeros", "scales") and _classify(base + ".weight") is not None:
            packed.setdefault(base, {})[field] = reader.read(name)
    items = []
    for base, parts in packed.items():
        if not {"qweight", "qzeros", "scales"} <= parts.keys():
            raise FormatUnreadable(f"{base}: incomplete AWQ packed tensors {sorted(parts)}")
        qweight, qzeros, scales = (parts[field] for field in ("qweight", "qzeros", "scales"))
        if qzeros.shape[0] == 0 or qweight.shape[0] % qzeros.shape[0]:
            raise FormatUnreadable(
                f"{base}: AWQ qweight rows {qweight.shape[0]} do not group by "
                f"qzeros rows {qzeros.shape[0]}"
            )
        group_size = qweight.shape[0] // qzeros.shape[0]
        # dequantize returns [in, out]; HF .weight is [out, in].
        items.append(
            (base + ".weight", _dequantize_awq_gemm(qweight, qzeros, scales, group_size).T)
        )
    return items


def _extract_safetensors(
    reader: _SafetensorsReader,
    repo_id: str,
    revision: str | None,
    quant_method: str | None,
    step: int | None,
    *,
    range_reads: bool,
) -> Extraction:
    """Extract gate spectra from a reader (plain/FP8, AWQ int4, or MXFP4 blocks)."""
    layers: dict[int, list[TensorSpectrum]] = {}
    kinds: set[str] = set()
    plain_storage = False
    if quant_method == "awq":
        _accumulate(layers, kinds, _awq_items(reader))
    elif quant_method == "mxfp4":
        _accumulate_mxfp4(layers, kinds, reader)
        if not layers:
            # Declared MXFP4 but stored plain (a bf16 re-save with a stale
            # config): read the shards as they are, exactly as the smoke does.
            # Recorded in provenance because this module has no log channel.
            plain_storage = True
            _accumulate(layers, kinds, _gate_items(reader))
    else:
        _accumulate(layers, kinds, _gate_items(reader))
    source = {
        "repo_id": repo_id,
        "revision": revision,
        "files": list(reader.files),
        "quant_method": quant_method,
        "step": step,
        "fetch": "range" if range_reads else "snapshot",
    }
    if plain_storage:
        source["mxfp4_plain_storage"] = True
    return _finish_extraction(layers, kinds, "safetensors", source)


def _accumulate_mxfp4(
    layers: dict[int, list[TensorSpectrum]],
    kinds: set[str],
    reader: _SafetensorsReader,
) -> None:
    """Dequantize MXFP4 expert gate+up blocks into per-expert gate spectra.

    A config that declares ``quant_method='mxfp4'`` but ships no packed blocks
    stores plain weights (stale-config bf16 re-save); this function then keeps
    nothing and the caller re-reads the shards as plain. A missing ``_scales``
    beside real blocks is still structural: the bench raises, unlike the smoke,
    which skips such an entry (no pair hits either path today).
    """
    block_names = [name for name in reader.names() if _MXFP4_GATE_BLOCKS.search(name)]
    if not block_names:
        return
    for name in block_names:
        layer = _last_layer_index(name)
        if layer is None:
            raise FormatUnreadable(f"{name}: no layers.N index in an MXFP4 gate name")
        stem = name[: -len("_blocks")]
        scales_name = stem + "_scales"
        if scales_name not in reader:
            raise FormatUnreadable(f"{name}: missing {scales_name}")
        blocks = reader.read(name)
        scale_codes = reader.read(scales_name)
        if blocks.ndim != 4:
            raise FormatUnreadable(
                f"{name}: expected [experts, rows, groups, 16], got {blocks.shape}"
            )
        for expert in range(int(blocks.shape[0])):
            # GEMM layout [rows, groups*32] -> HF [in, out]; the even columns of
            # the last axis are the gate half of the fused gate+up weight.
            gate = dequantize_mxfp4(blocks[expert], scale_codes[expert]).T[:, ::2]
            _store_matrix(layers, f"{stem}#expert{expert}", gate, ROLE_ROUTED, layer)
        kinds.add("mxfp4_gate")


# ---------------------------------------------------------------------------
# Pinned Hub loading (shared HF cache; tight allow_patterns; retry on transients).
# ---------------------------------------------------------------------------


def _reset_hub_session() -> None:
    """Drop the shared httpx client: hub closes it on ConnectError but keeps a reference."""
    with suppress(Exception):
        from huggingface_hub.utils import close_session

        close_session()


def _retry_call(label: str, fn):
    """Run a Hub call with the independence-tests transient-error policy."""
    last_exc: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        if attempt > 1:
            _reset_hub_session()
            wait = DOWNLOAD_BACKOFF_SECONDS[min(attempt - 2, len(DOWNLOAD_BACKOFF_SECONDS) - 1)]
            time.sleep(wait)
        try:
            return fn()
        except FormatUnreadable:
            raise  # structural, not transient
        except Exception as exc:  # classified below
            transient = (
                "Server disconnected" in str(exc)
                or "client has been closed" in str(exc)
                or type(exc).__name__ in ("ConnectError", "ReadTimeout", "RemoteProtocolError")
            )
            if not transient:
                raise
            last_exc = exc
    raise last_exc  # all attempts transient-failed


def _snapshot(repo_id: str, revision: str | None, patterns: list[str]) -> Path:
    """Pinned-revision snapshot restricted to the patterns we need."""
    return Path(
        _retry_call(
            f"snapshot {repo_id}",
            lambda: snapshot_download(
                repo_id, allow_patterns=patterns, cache_dir=None, revision=revision
            ),
        )
    )


def _download_file(repo_id: str, revision: str | None, filename: str) -> Path:
    return Path(
        _retry_call(
            f"file {filename}",
            lambda: hf_hub_download(repo_id, filename=filename, cache_dir=None, revision=revision),
        )
    )


def _load_config(repo_id: str, revision: str | None) -> dict:
    """Pinned config.json (missing config is not fatal — it is metadata)."""
    try:
        path = _download_file(repo_id, revision, "config.json")
    except Exception:
        return {}
    with open(path) as handle:
        try:
            return json.load(handle)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FormatUnreadable(f"{repo_id}: config.json is not valid JSON ({exc})") from exc


def _quant_method(config: Mapping) -> str | None:
    """Declared ``quantization_config.quant_method``, or None when absent."""
    quant_config = config.get("quantization_config")
    return quant_config.get("quant_method") if isinstance(quant_config, dict) else None


def _repo_files(repo_id: str, revision: str | None) -> list[str]:
    return list(
        _retry_call(
            f"list {repo_id}",
            lambda: HfApi().list_repo_files(repo_id, revision=revision),
        )
    )


def _choose_gguf(files: Iterable[str]) -> str | None:
    """Q8_0 first (closest to the parent numerics); else any non-mmproj GGUF."""
    candidates = sorted(
        name for name in files if name.endswith(".gguf") and "mmproj" not in name.lower()
    )
    if not candidates:
        return None
    for name in candidates:
        if "Q8_0" in name:
            return name
    return candidates[0]


def _load_gguf(repo_id: str, revision: str | None, gguf_files: Iterable[str]) -> Extraction:
    """GGUF MoE/dense gates; any reader failure is a structural abstain."""
    try:
        return _read_gguf(repo_id, revision, gguf_files)
    except FormatUnreadable:
        raise
    except Exception as exc:
        # Untrusted container: GGUFReader raises ValueError for unknown quant
        # type codes and TypeError/OSError for malformed headers. One weird
        # file must never abort a whole benchmark run.
        raise FormatUnreadable(f"{repo_id}: GGUF unreadable ({type(exc).__name__}: {exc})") from exc


def _read_gguf(repo_id: str, revision: str | None, gguf_files: Iterable[str]) -> Extraction:
    """GGUF MoE/dense gates via GGUFReader + gguf.quants.dequantize."""
    filename = _choose_gguf(gguf_files)
    if filename is None:
        raise FormatUnreadable(f"{repo_id}: no usable .gguf file at this revision")
    path = _download_file(repo_id, revision, filename)
    reader = GGUFReader(str(path))
    layers: dict[int, list[TensorSpectrum]] = {}
    kinds: set[str] = set()
    quant_types: set[str] = set()
    for tensor in reader.tensors:
        if _classify(tensor.name) is None:
            continue
        # gguf-py already dequantizes into the logical numpy layout; the
        # reader's ``tensor.shape`` is reversed GGML file order.
        _accumulate(layers, kinds, [(tensor.name, _gguf_gate(tensor))])
        quant_types.add(tensor.tensor_type.name)
    source = {
        "repo_id": repo_id,
        "revision": revision,
        "file": filename,
        "quant_types": sorted(quant_types),
    }
    return _finish_extraction(layers, kinds, "gguf", source)


def _gguf_gate(tensor) -> np.ndarray:
    """Dequantize one GGUF tensor and check it against its declared size."""
    try:
        data = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
    except NotImplementedError as exc:
        raise FormatUnreadable(
            f"{tensor.name}: {tensor.tensor_type.name} dequant unsupported"
        ) from exc
    array = np.asarray(data, dtype=np.float32)
    if array.size != tensor.n_elements:
        raise FormatUnreadable(
            f"{tensor.name}: dequantized {array.size} values, expected {tensor.n_elements}"
        )
    return array


def _resolve_step(config: Mapping, files: Iterable[str]) -> int | None:
    """Training-step checkpoint directory (AlexWortega upcycle layout)."""
    steps = {
        int(match.group(1))
        for name in files
        if (match := re.match(r"step_(\d+)/", name)) is not None
    }
    if not steps:
        return None  # flat checkpoint layout; config "step" is not a directory
    config_step = config.get("step")
    try:
        preferred = int(config_step) if config_step is not None else None
    except (TypeError, ValueError):
        preferred = None  # an untrusted config value never selects a directory
    return preferred if preferred in steps else max(steps)


def _snapshot_reader(repo_id: str, revision: str | None, step: int | None) -> _SnapshotReader:
    """Cached-snapshot reader; blobs already in the HF cache are reused as-is."""
    patterns = (
        ["*.safetensors", "*.safetensors.index.json"]
        if step is None
        else [f"step_{step}/*.safetensors"]  # never mix intermediate training steps
    )
    return _SnapshotReader(_snapshot(repo_id, revision, patterns), step)


def _load_safetensors(
    repo_id: str, revision: str | None, quant_method: str | None, step: int | None
) -> Extraction:
    """Safetensors checkpoint: HTTP range reads first, cached snapshot on any failure.

    An untrusted header may fail in an unbounded number of ways (bad JSON,
    hostile offsets, a list where a dict belongs); every non-structural failure
    inside either attempt maps to ``FormatUnreadable`` so one odd checkpoint
    abstains instead of aborting the sweep.
    """
    try:
        try:
            reader = _RangeReader(repo_id, revision, step=step)
            return _extract_safetensors(
                reader, repo_id, revision, quant_method, step, range_reads=True
            )
        except FormatUnreadable:
            raise  # structural: the cached bytes would fail in exactly the same way
        except Exception:
            # Offline/gated/HTTP failure: read the shared cache instead.
            reader = _snapshot_reader(repo_id, revision, step)
            return _extract_safetensors(
                reader, repo_id, revision, quant_method, step, range_reads=False
            )
    except FormatUnreadable:
        raise
    except Exception as exc:  # noqa: BLE001 — untrusted checkpoint container
        raise FormatUnreadable(
            f"{repo_id}: safetensors unreadable ({type(exc).__name__}: {exc})"
        ) from exc


def _bin_extraction(
    items: Iterable[tuple[str, np.ndarray]], source: Mapping[str, object]
) -> Extraction:
    """Gate spectra from streamed state-dict tensors (torch-free, offline-testable)."""
    layers: dict[int, list[TensorSpectrum]] = {}
    kinds: set[str] = set()
    _accumulate(layers, kinds, items)
    if not layers:
        raise FormatUnreadable(f"{source.get('file')}: no gate-like FFN tensors in the state dict")
    return _finish_extraction(layers, kinds, "pytorch_bin", dict(source))


def _load_pytorch_bin(repo_id: str, revision: str | None) -> Extraction:
    """Legacy ``pytorch_model.bin`` checkpoint; any read failure is a structural abstain."""
    try:
        return _read_pytorch_bin(repo_id, revision)
    except FormatUnreadable:
        raise
    except Exception as exc:
        # Untrusted pickle: torch raises UnpicklingError/RuntimeError/OSError on
        # damaged or unsupported checkpoints. One bad file must not abort a run.
        raise FormatUnreadable(
            f"{repo_id}: pytorch_model.bin unreadable ({type(exc).__name__}: {exc})"
        ) from exc


def _read_pytorch_bin(repo_id: str, revision: str | None) -> Extraction:
    """Read a TinyMixtral-style flat state dict and keep only gate spectra.

    Single-file ``pytorch_model.bin`` only: sharded
    (``pytorch_model-NNNNN-of-NNNNN.bin``) and index-declared checkpoints are
    not joined here, unlike the smoke's shard- and index-aware ``_load_bin``.
    No current pair needs them; a sharded-only repo abstains as
    ``format_unreadable`` rather than scoring half a model.

    torch is imported here rather than at module scope: it is an optional
    dependency, and every other checkpoint format must keep working in an
    environment that does not ship it.
    """
    try:
        import torch
    except ImportError as exc:
        raise FormatUnreadable(
            f"{repo_id}: pytorch_model.bin needs torch, which is not installed"
        ) from exc
    filename = "pytorch_model.bin"
    path = _download_file(repo_id, revision, filename)
    # weights_only=True: never unpickle arbitrary objects from a Hub repo.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise FormatUnreadable(f"{repo_id}: {filename} is not a state dict")

    def gates() -> Iterator[tuple[str, np.ndarray]]:
        for name, value in checkpoint.items():
            if _classify(name) is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise FormatUnreadable(f"{filename}: {name} is not a tensor")
            yield name, value.detach().to(device="cpu", dtype=torch.float32).numpy()

    source = {"repo_id": repo_id, "revision": revision, "file": filename}
    return _bin_extraction(gates(), source)


def _load_gate_tensors(repo_id: str, revision: str | None) -> Extraction:
    """Extract every gate-like FFN tensor from one pinned checkpoint."""
    config = _load_config(repo_id, revision)
    quant_method = _quant_method(config)
    if quant_method in UNSUPPORTED_QUANT_METHODS:
        raise FormatUnreadable(
            f"{repo_id}: quant_method={quant_method!r} has no dequantizer in this detector"
        )
    try:
        files = _repo_files(repo_id, revision)
    except (GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError) as exc:
        # A license-gated, missing, or Hub-unresolvable checkpoint is a
        # structural abstain, not a scoring failure (e.g.
        # meta-llama/Llama-3.2-1B needs a token; by-SHA tree resolution
        # can also 404 transiently on the Hub side).
        raise FormatUnreadable(
            f"{repo_id}: inaccessible at this revision ({type(exc).__name__})"
        ) from exc
    if any(name.endswith(".safetensors") for name in files):
        # Includes gpt-oss MXFP4: its gates are packed ``gate_up_proj_blocks``,
        # not ``.weight`` matrices, and the safetensors loader decodes them.
        return _load_safetensors(repo_id, revision, quant_method, _resolve_step(config, files))
    gguf_files = [name for name in files if name.endswith(".gguf")]
    if gguf_files:
        return _load_gguf(repo_id, revision, gguf_files)
    if any(name.endswith("pytorch_model.bin") for name in files):
        return _load_pytorch_bin(repo_id, revision)
    raise FormatUnreadable(f"{repo_id}: no .safetensors, .gguf, or .bin weights at this revision")


# ---------------------------------------------------------------------------
# Diagnostics assembly and the harness entry point.
# ---------------------------------------------------------------------------


def _source_summary(extraction: Extraction) -> dict:
    """Compact provenance: source file(s) for the chosen quant path."""
    source: dict[str, object] = {"format": extraction.format}
    if extraction.format == "gguf":
        source["file"] = extraction.source.get("file")
        source["quant_types"] = extraction.source.get("quant_types", [])
    elif extraction.format == "safetensors":
        source["files"] = extraction.source.get("files", [])
        source["quant_method"] = extraction.source.get("quant_method")
        source["step"] = extraction.source.get("step")
        source["fetch"] = extraction.source.get("fetch")
        if extraction.source.get("mxfp4_plain_storage"):
            # Declared MXFP4, stored plain: the provenance record is the only
            # trace of the fallback (this module never writes to stderr).
            source["mxfp4_plain_storage"] = True
    elif extraction.format == "pytorch_bin":
        source["file"] = extraction.source.get("file")
    return source


def _extraction_summary(extraction: Extraction) -> dict:
    """Pre-scoring summary used by structural abstains."""
    return {
        "family": extraction.family,
        "n_layers": len(extraction.layers),
        "n_gate_tensors": extraction.n_tensors,
        "source": _source_summary(extraction),
    }


def _model_diagnostics(extraction: Extraction, profile: Sequence[float]) -> dict:
    """Per-side diagnostics: depth, per-layer r_eff/counts, fractions, raw profile."""
    layers = [extraction.layers[layer] for layer in sorted(extraction.layers, key=int)]
    r_effs = [tensor.r_eff for tensors in layers for tensor in tensors]
    fractions = [tensor.spectrum_fraction for tensors in layers for tensor in tensors]
    return {
        "family": extraction.family,
        "n_layers": len(layers),
        "n_gate_tensors": extraction.n_tensors,
        "layers": [int(layer) for layer in sorted(extraction.layers, key=int)],
        "r_eff_per_layer": [
            [min(tensor.r_eff for tensor in tensors), max(tensor.r_eff for tensor in tensors)]
            for tensors in layers
        ],
        "tensor_counts_per_layer": [len(tensors) for tensors in layers],
        "r_eff": {"min": min(r_effs) if r_effs else 0, "max": max(r_effs) if r_effs else 0},
        "spectrum_fraction": {
            "min": min(fractions) if fractions else 0.0,
            "max": max(fractions) if fractions else 0.0,
            "mean": float(np.mean(fractions)) if fractions else 0.0,
        },
        "profile": [float(value) for value in profile],
        "profile_cv": _profile_cv(profile),
        "source": _source_summary(extraction),
    }


def _structure_matched(*extractions: Extraction) -> bool:
    """Skeptic-audit diagnostic: every extracted tensor has the same q."""
    q_values = {
        tensor.q
        for extraction in extractions
        for tensors_in_layer in extraction.layers.values()
        for tensor in tensors_in_layer
    }
    return len(q_values) == 1


def _abstain(reason: str, detail: str, diagnostics: dict) -> Result:
    """Structural abstain in the harness contract shape."""
    detail = " ".join(detail.split())
    return Result(
        verdict="abstain",
        score=None,
        diagnostics={"reason": reason, "detail": detail, **diagnostics},
        note=f"GATE spectral abstain {reason}: {detail}",
        abstain_reason=reason,
    )


def score(
    subject: str,
    reference: str,
    *,
    subject_rev: str | None = None,
    reference_rev: str | None = None,
    stack: str,
) -> Result:
    """Score one pair from pinned weights only (ids, revisions, and stack)."""
    del stack  # stack is contract input; GATE derives family/structure from weights
    try:
        extraction_subject = _load_gate_tensors(subject, subject_rev)
        extraction_reference = _load_gate_tensors(reference, reference_rev)
    except (
        FormatUnreadable,
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    ) as exc:
        return _abstain("format_unreadable", str(exc), {})

    summaries = {
        "subject": _extraction_summary(extraction_subject),
        "reference": _extraction_summary(extraction_reference),
    }
    if extraction_subject.n_tensors == 0 or extraction_reference.n_tensors == 0:
        missing = "subject" if extraction_subject.n_tensors == 0 else "reference"
        return _abstain("no_gate_tensors", f"no gate-like FFN tensors on {missing}", summaries)

    spectra_subject = extraction_subject.layers
    spectra_reference = extraction_reference.layers
    primary = _compare(spectra_subject, spectra_reference, R_DEFAULT, SEGMENTS)
    if primary["status"] != "ok":
        diagnostics = {
            **summaries,
            "profile_cv": {
                "subject": _profile_cv(primary["profile"]["a"]),
                "reference": _profile_cv(primary["profile"]["b"]),
            },
        }
        return _abstain(
            primary["status"], f"primary comparison status={primary['status']}", diagnostics
        )

    panel = _sensitivity_panel(spectra_subject, spectra_reference, primary, R_DEFAULT, SEGMENTS)
    cka = float(primary["cka"])
    diagnostics = {
        "subject": _model_diagnostics(extraction_subject, primary["profile"]["a"]),
        "reference": _model_diagnostics(extraction_reference, primary["profile"]["b"]),
        "pair": {
            "signed_pearson": primary["signed_pearson"],
            "anticorrelation_collision": primary["anticorrelation_collision"],
            "structure_matched": _structure_matched(extraction_subject, extraction_reference),
            "layers_per_segment": {
                "subject": _segment_sizes(len(primary["features"]["a"]), SEGMENTS),
                "reference": _segment_sizes(len(primary["features"]["b"]), SEGMENTS),
            },
        },
        "sensitivity": panel,
    }
    return Result(
        verdict="related" if cka >= TAU else "unrelated",
        score=cka,
        diagnostics=diagnostics,
        note=f"GATE spectral r={R_DEFAULT} s={SEGMENTS} tau={TAU} cka={cka:.4f}",
        abstain_reason=None,
    )
