"""GATE spectral fingerprint core — pure numpy, no I/O, no fixture knowledge.

Reimplementation (paper only; the authors released no code) of the GATE
spectral feature from "Model-intrinsic, Weight-level, and Training-free
Spectral Fingerprinting for Large Language Models" (He, Chen, Xiong, Zhao,
Wu; arXiv:2610.30626v1). Paper-equation mapping:

  Eq. (1)  gated FFN; the feature consumes W_gate, the branch whose output
           passes the activation before the element-wise product with UP.
  Eq. (2)  sigma_bar_i = sigma_i / (sum_{j<=r_eff} sigma_j + EPS) — the
           normalization sums over the *retained* top-r spectrum only.
  Eq. (3)  H = -sum sigma_bar_i * ln(sigma_bar_i).
  Eq. (4)  f = exp(H) (natural log, so exp cancels the log base).
  Eq. (5)  p_k = mean(f[l] for l in contiguous ~equal depth group k).
  Eq. (6)  linear CKA; on 1-D mean-centered vectors it equals the squared
           Pearson correlation (sign-insensitive).

House conventions pinned by the implementation contract:

- fp32 SVD input (``numpy.linalg.svd(compute_uv=False)``, deterministic
  LAPACK), fp64 normalization/entropy arithmetic.
- ``EPS = 1e-12`` in the Eq. (2) denominator only. Eq. (3)'s log-epsilon is
  redundant for positive singular values and is omitted; exact zeros are
  skipped as ``lim x ln x = 0`` so rank-deficient matrices stay finite.
- ``r_eff = min(r, q)`` with ``q = min(d_m, d_f)``; r_eff is recorded per
  tensor because a different r_eff changes the max attainable entropy.
- all-zero W -> H = 0, f = 1, flagged (never NaN/inf).
- Layer scalar (MoE extension, disclosed; degenerates to the paper on dense
  models) = arithmetic mean of per-tensor f over every gate-like FFN tensor
  in that layer.
- Depth order is the integer layer index, never lexicographic.
- s = 8 contiguous ~equal segments with ``numpy.array_split`` semantics.
- Verdict: CKA >= 0.9 -> related. Sensitivities (r in 64/256, s in 4/16,
  routed-only/shared-only, geometric-mean and concatenated-spectrum layer
  variants) are diagnostics and never enter the verdict.

Everything here operates on plain spectra/profile arrays; extraction lives
in ``tensors.py`` and the CLI in ``smoke.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

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
#: profile_cv below this marks a low-signal depth profile (skeptic audit).
LOW_SIGNAL_CV = 0.02
#: Roles a gate-like tensor can carry; used by the MoE sensitivity variants.
ROLE_ROUTED = "routed"
ROLE_SHARED = "shared"


@dataclass(frozen=True)
class TensorSpectrum:
    """One gate-like FFN tensor reduced to its singular-value spectrum.

    ``shape`` carries the provenance the extraction diagnostics used to read off
    the matrix; it is recorded here because streaming extraction drops the
    matrix as soon as the spectrum exists. It defaults to ``()`` so a bare
    ``TensorSpectrum(name, role, sv)`` (selftests, synthetic fixtures) stays
    valid. ``q`` is derived, never stored, so it cannot drift from ``sv``: the
    full spectrum has exactly ``q = min(d_m, d_f)`` values.
    """

    name: str
    role: str
    sv: np.ndarray
    shape: tuple[int, ...] = ()

    @property
    def q(self) -> int:
        """Paper's q = min(d_m, d_f), invariant to orientation."""
        return int(self.sv.size)


def svd_spectrum(matrix: np.ndarray) -> np.ndarray:
    """Singular values of a gate matrix, descending, fp64.

    The SVD runs on the float32 cast (house convention) so the LAPACK path is
    deterministic and matches the transport precision of the safetensors read;
    the returned spectrum is float64 for the entropy arithmetic. Transposing W
    is unnecessary: singular values are invariant to it.
    """
    w = np.asarray(matrix, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"gate matrix must be 2-D, got shape {w.shape}")
    return np.linalg.svd(w, compute_uv=False).astype(np.float64)


def entropy_feature(sv: Sequence[float] | np.ndarray, r: int = R_DEFAULT) -> dict:
    """Paper-exact per-tensor feature from a descending singular spectrum.

    Eq. (2)-(4) with ``r_eff = min(r, q)``: normalize the retained top-r_eff
    values by their own sum plus EPS, take the natural-log entropy, and
    exponentiate. Exact zero singular values contribute nothing (the
    continuous limit of x*ln(x)); an all-zero matrix is the guarded case
    H = 0, f = 1 with ``all_zero`` set.
    """
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


def spectral_entropy_feature(matrix: np.ndarray, r: int = R_DEFAULT) -> dict:
    """``entropy_feature`` on a raw gate matrix (SVD + Eq. (2)-(4))."""
    return entropy_feature(svd_spectrum(matrix), r)


def layer_features(
    layers: Mapping[int, Sequence[TensorSpectrum]],
    r: int = R_DEFAULT,
    roles: Iterable[str] | None = None,
) -> list[dict]:
    """Per-layer scalars, shallow -> deep (Eq. (5) input).

    ``layers`` maps an integer physical depth index to that layer's gate-like
    tensors and their spectra. The layer scalar is the arithmetic mean of the
    per-tensor exp(H) values (MoE extension; on a dense model there is exactly
    one tensor, so it degenerates to the paper). ``roles`` restricts the tensor
    set for the routed-only / shared-only diagnostics.
    """
    allowed = None if roles is None else frozenset(roles)
    features: list[dict] = []
    for layer in sorted(layers, key=int):
        tensors = [t for t in layers[layer] if allowed is None or t.role in allowed]
        if not tensors:
            continue
        per_tensor = []
        for tensor in tensors:
            feature = entropy_feature(tensor.sv, r)
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


def geometric_layer_features(
    layers: Mapping[int, Sequence[TensorSpectrum]], r: int = R_DEFAULT
) -> list[dict]:
    """Per-layer geometric mean of the per-tensor ``exp(H)`` values.

    Diagnostic only (never the verdict): ``exp(mean H_t)`` is the geometric
    mean of the per-tensor ``exp(H_t)``, recorded for every MoE pair so a
    reviewer can see whether the arithmetic-mean rule hides layer diversity.
    """
    features: list[dict] = []
    for layer in sorted(layers, key=int):
        tensors = list(layers[layer])
        if not tensors:
            continue
        entropies = [entropy_feature(tensor.sv, r)["H"] for tensor in tensors]
        features.append(
            {
                "layer": int(layer),
                "f": float(np.exp(np.mean(entropies))),
                "tensor_count": len(tensors),
            }
        )
    return features


def concatenated_layer_features(
    layers: Mapping[int, Sequence[TensorSpectrum]], r: int = R_DEFAULT
) -> list[dict]:
    """Per-layer ``exp(H)`` over the concatenated top-r_eff spectra.

    Diagnostic only (never the verdict): the per-tensor retained top-r_eff
    spectra are concatenated and renormalized together, so the layer scalar
    measures diversity across the layer's tensors instead of averaging their
    individual entropies.
    """
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
        feature = entropy_feature(concatenated, r=int(concatenated.size))
        features.append(
            {
                "layer": int(layer),
                "f": feature["f"],
                "tensor_count": len(tensors),
            }
        )
    return features


def segment_sizes(n_items: int, s: int = SEGMENTS) -> list[int]:
    """Sizes of ``numpy.array_split``'s contiguous ~equal groups (first get extras)."""
    return [int(group.size) for group in np.array_split(np.arange(n_items), s)]


def pool_segments(values: Sequence[float] | np.ndarray, s: int = SEGMENTS) -> np.ndarray:
    """Eq. (5): mean-pool the depth profile into s contiguous ~equal segments.

    ``numpy.array_split`` semantics: group sizes differ by at most one and the
    first ``L mod s`` groups receive the extra layer. Fewer layers than
    segments leave empty groups as NaN, which the CKA zero-variance/NaN guard
    turns into a ``degenerate_profile`` abstain instead of a silent number.
    """
    profile = np.asarray(list(values), dtype=np.float64)
    if profile.size == 0:
        return np.full(s, np.nan)
    return np.array(
        [float(group.mean()) if group.size else np.nan for group in np.array_split(profile, s)]
    )


def cka_linear_1d(
    a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray
) -> tuple[float | None, float | None]:
    """Linear CKA between 1-D profiles (Eq. (6)): (CKA, signed Pearson).

    For mean-centered 1-D vectors the Frobenius CKA reduces to the squared
    Pearson coefficient, which is sign-insensitive by construction — the
    signed value is returned separately for the anticorrelation diagnostic.
    A constant (zero-variance) or non-finite profile has no defined
    correlation and returns ``(None, None)`` so the caller abstains rather
    than emitting NaN.
    """
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


def similarity(
    profile_a: Sequence[float] | np.ndarray, profile_b: Sequence[float] | np.ndarray
) -> dict:
    """CKA + signed Pearson + the anticorrelation-collision flag."""
    cka, signed = cka_linear_1d(profile_a, profile_b)
    return {
        "cka": cka,
        "signed_pearson": signed,
        "anticorrelation_collision": bool(
            cka is not None and signed is not None and signed <= ANTICORRELATION_R and cka >= TAU
        ),
    }


def verdict_for(cka: float | None, tau: float = TAU) -> str | None:
    """A-priori rule: related iff CKA >= tau; abstain for a missing CKA."""
    if cka is None:
        return None
    return "related" if cka >= tau else "unrelated"


def profile_cv(profile: Sequence[float]) -> float:
    """Coefficient of variation of the pooled profile (std / mean)."""
    values = np.asarray(profile, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return float("nan")
    mean = float(values.mean())
    if mean == 0.0:
        return float("nan")
    return float(values.std() / mean)


def profile_diagnostics(features: Sequence[dict], profile: Sequence[float]) -> dict:
    """Per-model diagnostics: depth, tensor counts, r_eff and spectrum fraction."""
    r_effs = [t["r_eff"] for layer in features for t in layer["tensors"]]
    fractions = [t["spectrum_fraction"] for layer in features for t in layer["tensors"]]
    counts = [layer["tensor_count"] for layer in features]
    return {
        "L": len(features),
        "n_gate_tensors": sum(counts),
        "tensor_counts": {
            "per_layer": counts,
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
        },
        "r_eff": {"min": min(r_effs) if r_effs else 0, "max": max(r_effs) if r_effs else 0},
        "r_eff_per_layer": [
            {
                "layer": layer["layer"],
                "min": min(t["r_eff"] for t in layer["tensors"]),
                "max": max(t["r_eff"] for t in layer["tensors"]),
            }
            for layer in features
        ],
        "spectrum_fraction": {
            "min": min(fractions) if fractions else 0.0,
            "max": max(fractions) if fractions else 0.0,
        },
        "profile_cv": profile_cv(profile),
        "layers_per_segment": segment_sizes(len(features), len(profile)),
    }


def compare(
    layers_a: Mapping[int, Sequence[TensorSpectrum]],
    layers_b: Mapping[int, Sequence[TensorSpectrum]],
    r: int = R_DEFAULT,
    s: int = SEGMENTS,
    roles: Iterable[str] | None = None,
) -> dict:
    """Profile both sides and score them (unsegmented abstains included).

    ``status`` is ``ok`` when a CKA exists, ``no_gate_tensors`` when a role
    filter (or an empty extraction) leaves a side without tensors, and
    ``degenerate_profile`` when a pooled vector is constant/NaN.
    """
    features_a = layer_features(layers_a, r, roles)
    features_b = layer_features(layers_b, r, roles)
    profile_a = pool_segments([layer["f"] for layer in features_a], s)
    profile_b = pool_segments([layer["f"] for layer in features_b], s)
    sim = similarity(profile_a, profile_b)
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
        "verdict": verdict_for(sim["cka"]),
    }


def _variant(comparison: dict) -> dict:
    """Compact sensitivity-panel entry from a full comparison."""
    return {
        "status": comparison["status"],
        "cka": comparison["cka"],
        "signed_pearson": comparison["signed_pearson"],
        "verdict": comparison["verdict"],
        "tensor_counts": comparison["tensor_counts"],
    }


def _diagnostic_variant(
    layers_a: Mapping[int, Sequence[TensorSpectrum]],
    layers_b: Mapping[int, Sequence[TensorSpectrum]],
    builder: Callable[[Mapping[int, Sequence[TensorSpectrum]], int], list[dict]],
    r: int = R_DEFAULT,
    s: int = SEGMENTS,
) -> dict:
    """Score one diagnostic profile builder (statuses mirror ``compare``)."""
    features_a = builder(layers_a, r)
    features_b = builder(layers_b, r)
    sim = similarity(
        pool_segments([layer["f"] for layer in features_a], s),
        pool_segments([layer["f"] for layer in features_b], s),
    )
    if not features_a or not features_b:
        status = "no_gate_tensors"
    elif sim["cka"] is None:
        status = "degenerate_profile"
    else:
        status = "ok"
    return {
        "status": status,
        **sim,
        "verdict": verdict_for(sim["cka"]),
        "tensor_counts": {
            "a": sum(layer["tensor_count"] for layer in features_a),
            "b": sum(layer["tensor_count"] for layer in features_b),
        },
    }


def sensitivity_panel(
    layers_a: Mapping[int, Sequence[TensorSpectrum]],
    layers_b: Mapping[int, Sequence[TensorSpectrum]],
    primary: dict,
    r: int = R_DEFAULT,
    s: int = SEGMENTS,
) -> dict:
    """Diagnostic variants of the primary comparison; never used for the verdict.

    Rank variants recompute Eq. (2)-(4) at r = 64 / 256 and repool at the
    primary s; segment variants repool the primary r features at s = 4 / 16;
    MoE variants restrict the tensor set to routed or shared gates, and the
    ``geometric`` / ``concatenated`` variants replace the per-layer arithmetic
    mean with the geometric mean of ``exp(H)`` and the entropy of the
    concatenated top-r_eff spectra. All entries carry their own CKA/verdict so
    a reviewer can see whether the primary call is sensitive to any of them
    (``rank_sensitive`` / ``moe_rule_sensitive`` flags are derived in the CLI).
    """
    panel: dict[str, dict] = {}
    for variant_r in (64, 256):
        panel[f"r{variant_r}"] = _variant(compare(layers_a, layers_b, r=variant_r, s=s))
    for variant_s in (4, 16):
        profile_a = pool_segments([layer["f"] for layer in primary["features"]["a"]], variant_s)
        profile_b = pool_segments([layer["f"] for layer in primary["features"]["b"]], variant_s)
        sim = similarity(profile_a, profile_b)
        panel[f"s{variant_s}"] = {
            "status": "ok" if sim["cka"] is not None else "degenerate_profile",
            **sim,
            "verdict": verdict_for(sim["cka"]),
            "tensor_counts": primary["tensor_counts"],
        }
    for role, key in ((ROLE_ROUTED, "experts_only"), (ROLE_SHARED, "shared_only")):
        comparison = compare(layers_a, layers_b, r=r, s=s, roles={role})
        if comparison["status"] == "no_gate_tensors":
            panel[key] = {
                "status": "unavailable",
                "cka": None,
                "signed_pearson": None,
                "verdict": None,
                "tensor_counts": comparison["tensor_counts"],
            }
        else:
            panel[key] = _variant(comparison)
    for key, builder in (
        ("geometric", geometric_layer_features),
        ("concatenated", concatenated_layer_features),
    ):
        panel[key] = _diagnostic_variant(layers_a, layers_b, builder, r, s)
    return panel
