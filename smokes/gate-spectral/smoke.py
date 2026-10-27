"""GATE spectral fingerprint smoke — smokes/gate-spectral/ · arXiv:2610.30626 · #26 survey.

Weight-level reimplementation of the GATE spectral fingerprint from
"Model-intrinsic, Weight-level, and Training-free Spectral Fingerprinting for
Large Language Models" (He, Chen, Xiong, Zhao, Wu; arXiv:2610.30626v1). No
code was released, so everything here is implemented from the paper's
equations (see ``_gate.py`` for the Eq. (2)-(6) mapping and README.md for the
method spec).

Pipeline per checkpoint:
  1. ``tensors.py`` extracts every gate-like FFN tensor (routers excluded) —
     dense ``mlp.gate_proj`` (including Qwen3.5's ``model.language_model``
     nesting), GPTNeoX ``dense_4h_to_h`` and Nemotron-H ``mixer.down_proj``
     non-gated † fallbacks, granite's fused ``input_linear`` (split at half
     the row count), upcycle routed/shared/dense gates, Mixtral ``w1``, LFM2
     fused and discrete experts, tinymixtral's stacked
     ``layers.N.moe.gate_proj`` (``pytorch_model.bin``), GGUF
     ``ffn_gate_exps``; AWQ int4 / FP8 / bf16 / MXFP4 / GGUF dequant
     included. Extraction streams spectra, one tensor at a time.
  2. top-r singular values -> Eq. (2) normalization -> spectral entropy ->
     f = exp(H) per tensor; layer scalar = mean over the layer's tensors.
  3. stack in integer depth order, mean-pool into s=8 contiguous segments.
  4. linear CKA = squared Pearson of the two 8-vectors; related iff
     CKA >= 0.9. Abstain (``no_gate_tensors`` / ``format_unreadable`` /
     ``degenerate_profile``) instead of emitting NaN.

Expectation outcomes match ``smokes/independence-tests``: the gate fails only
on a plain ``missed``; structural format abstains are ``skipped_format``.

Stdout is a single JSON document; progress goes to stderr. ``--selftest``
runs fully offline (synthetic tensors, no network).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

SMOKE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SMOKE_DIR))

import _gate  # noqa: E402
import tensors  # noqa: E402
from _gate import (  # noqa: E402
    EPS,
    LOW_SIGNAL_CV,
    R_DEFAULT,
    SEGMENTS,
    TAU,
    TensorSpectrum,
)

# Outcome values for the per-pair expectation check (the gate fails only on
# "missed"; see README "Expectation outcomes and the gate").
OUTCOME_HELD = "held"
OUTCOME_MISSED = "missed"
OUTCOME_SKIPPED_FORMAT = "skipped_format"

# Abstain reasons meaning "this smoke structurally cannot consume the
# pair's checkpoint", recorded as skipped_format rather than a gate miss.
FORMAT_OUT_OF_SCOPE_REASONS = frozenset({"no_gate_tensors", "format_unreadable"})

# Default fixture set — the nine-pair smoke cohort from
# provenancebench/pairs.yaml: (pbench pair id, fixture role, expected verdict).
DEFAULT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("pb-smoke-qwen25-awq", "quant_robustness", "related"),
    ("pb-smoke-qwen25-fp8", "quant_robustness", "related"),
    ("pb-smoke-smoltulu", "positive", "related"),
    ("pb-smoke-qwen3-sft", "positive", "related"),
    ("pb-smoke-llama-smol-cross", "hard_negative", "unrelated"),
    ("pb-smoke-granite-instruct", "positive", "related"),
    ("pb-smoke-granite-gguf", "quant_robustness", "related"),
    ("pb-smoke-granite-smol-cross", "hard_negative", "unrelated"),
    ("pb-smoke-qwen3-upcycle", "upcycle", "related"),
)

# Ad-hoc --pair runs: expected verdict from the pair's provenancebench label.
_EXPECTED_FROM_YAML = {"related": "related", "unrelated": "unrelated", "unsupported": "abstain"}


@dataclass(frozen=True)
class PairSpec:
    """One provenancebench pair (ids and pinned revisions from pairs.yaml)."""

    pair_id: str
    stack: str
    kind: str
    expected: str
    subject: str
    reference: str
    subject_rev: str
    reference_rev: str


# ---------------------------------------------------------------------------
# pairs.yaml + Hub helpers.
# ---------------------------------------------------------------------------


def _pairs_yaml_path() -> Path:
    for bases in (SMOKE_DIR.resolve().parents, Path.cwd().parents):
        for base in bases:
            candidate = base / "provenancebench" / "pairs.yaml"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError("provenancebench/pairs.yaml not found above this smoke")


def load_pair_specs() -> dict[str, PairSpec]:
    """All pairs from the orchestrator manifest (ids + pinned revisions)."""
    with open(_pairs_yaml_path()) as handle:
        raw = yaml.safe_load(handle)
    return {
        entry["id"]: PairSpec(
            pair_id=entry["id"],
            stack=entry["stack"],
            kind=entry["kind"],
            expected=entry["expected"],
            subject=entry["subject"],
            reference=entry["reference"],
            subject_rev=entry["subject_rev"],
            reference_rev=entry["reference_rev"],
        )
        for entry in raw["pairs"]
    }


# ---------------------------------------------------------------------------
# Pair scoring (I/O lives in tensors.py; everything below scores spectra).
# ---------------------------------------------------------------------------


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _structure_matched(*extractions: tensors.Extraction) -> bool:
    """Skeptic-audit diagnostic: every extracted tensor has the same q."""
    q_values = {
        tensor.q
        for extraction in extractions
        for tensors_in_layer in extraction.layers.values()
        for tensor in tensors_in_layer
    }
    return len(q_values) == 1


def _expectation_outcome(entry: dict, expected_verdict: str) -> tuple[str, str | None]:
    """Classify a scored pair against its pre-registered expectation.

    The exit code fails only on a plain miss. Pairs the smoke structurally
    cannot consume are ``skipped_format`` (recorded, not gating);
    ``degenerate_profile`` is not a format issue, so it remains a miss.
    """
    verdict = entry["verdict"]
    if verdict == expected_verdict:
        return OUTCOME_HELD, None
    if verdict == "abstain" and entry.get("reason") in FORMAT_OUT_OF_SCOPE_REASONS:
        return (
            OUTCOME_SKIPPED_FORMAT,
            f"format out of scope for this smoke (reason={entry['reason']}); "
            "verdict kept as abstain, neither held nor a gate miss",
        )
    return OUTCOME_MISSED, None


def run_pair(spec: PairSpec, role: str, expected_verdict: str, cache_dir: Path | None) -> dict:
    """Score one pair end-to-end and return its JSON entry."""
    started = time.perf_counter()
    entry: dict = {
        "pair_id": spec.pair_id,
        "role": role,
        "kind": spec.kind,
        "stack": spec.stack,
        "subject": {"repo_id": spec.subject, "revision": spec.subject_rev},
        "reference": {"repo_id": spec.reference, "revision": spec.reference_rev},
        "expected_verdict": expected_verdict,
    }
    try:
        entry.update(_pair_fields(spec, cache_dir))
    except tensors.FormatUnreadable as exc:
        entry.update({"verdict": "abstain", "reason": "format_unreadable", "detail": str(exc)})
    outcome, outcome_note = _expectation_outcome(entry, expected_verdict)
    entry["outcome"] = outcome
    entry["expectation_held"] = outcome == OUTCOME_HELD
    if outcome_note:
        entry["outcome_note"] = outcome_note
    entry["compute_seconds"] = round(time.perf_counter() - started, 1)
    _log(
        f"[{spec.pair_id}] verdict={entry['verdict']} cka={entry.get('cka')} "
        f"outcome={entry['outcome']} ({entry['compute_seconds']}s)"
    )
    return entry


def _pair_fields(spec: PairSpec, cache_dir: Path | None) -> dict:
    """Verdict and diagnostics for one pair; abstains are fields, not exceptions.

    Only ``FormatUnreadable`` escapes (the caller records it as an abstain);
    a pair the smoke cannot consume comes back as an abstain dict so the
    expectation check can classify it.
    """
    subject = tensors.load_gate_tensors(spec.subject, spec.subject_rev, cache_dir=cache_dir)
    reference = tensors.load_gate_tensors(spec.reference, spec.reference_rev, cache_dir=cache_dir)
    _log(
        f"[{spec.pair_id}] gates subject={subject.n_tensors} ({subject.family}) "
        f"reference={reference.n_tensors} ({reference.family})"
    )
    for side, extraction in (("subject", subject), ("reference", reference)):
        if extraction.n_tensors == 0:
            return {
                "verdict": "abstain",
                "reason": "no_gate_tensors",
                "detail": f"no gate-like FFN tensors found on {side}",
            }
    primary = _gate.compare(subject.layers, reference.layers, R_DEFAULT, SEGMENTS)
    if primary["status"] != "ok":
        return {
            "verdict": "abstain",
            "reason": primary["status"],
            "detail": f"primary comparison status={primary['status']}",
        }
    return _scored_fields(subject, reference, primary)


def _scored_fields(
    subject: tensors.Extraction, reference: tensors.Extraction, primary: dict
) -> dict:
    """Verdict, sensitivity panel, skeptic-audit flags, and per-side diagnostics."""
    panel = _gate.sensitivity_panel(subject.layers, reference.layers, primary, R_DEFAULT, SEGMENTS)
    subject_diagnostics = _side_diagnostics(
        subject, primary["features"]["a"], primary["profile"]["a"]
    )
    reference_diagnostics = _side_diagnostics(
        reference, primary["features"]["b"], primary["profile"]["b"]
    )
    flags = {
        "rank_sensitive": _variant_flipped(panel, primary, ("r64", "r256")),
        "moe_rule_sensitive": _variant_flipped(panel, primary, ("experts_only", "shared_only")),
        "low_signal_profile": min(
            subject_diagnostics["profile_cv"], reference_diagnostics["profile_cv"]
        )
        < LOW_SIGNAL_CV,
    }
    return {
        "verdict": primary["verdict"],
        "cka": primary["cka"],
        "signed_pearson": primary["signed_pearson"],
        "anticorrelation_collision": primary["anticorrelation_collision"],
        "structure_matched": _structure_matched(subject, reference),
        "layers_per_segment": {
            "subject": subject_diagnostics["layers_per_segment"],
            "reference": reference_diagnostics["layers_per_segment"],
        },
        "sensitivity": panel,
        "flags": flags,
        "diagnostics": {"subject": subject_diagnostics, "reference": reference_diagnostics},
    }


def _side_diagnostics(
    extraction: tensors.Extraction, features: list[dict], profile: list[float]
) -> dict:
    """Extraction diagnostics plus the pooled depth profile for one side."""
    return {
        **tensors.extraction_diagnostics(extraction, R_DEFAULT),
        **_gate.profile_diagnostics(features, profile),
        "profile": profile,
    }


def _variant_flipped(panel: dict, primary: dict, keys: tuple[str, ...]) -> bool:
    """True when any named panel variant scores and lands on a different verdict."""
    return any(
        panel[key]["status"] == "ok" and panel[key]["verdict"] != primary["verdict"] for key in keys
    )


# ---------------------------------------------------------------------------
# Offline selftest — synthetic tensors only, no network, no HF downloads.
# ---------------------------------------------------------------------------


def _rank_k_matrix(rng: np.random.Generator, rows: int, cols: int, k: int) -> np.ndarray:
    """Rank-k matrix with k equal nonzero singular values (f = k exactly)."""
    u, _ = np.linalg.qr(rng.standard_normal((rows, k)))
    v, _ = np.linalg.qr(rng.standard_normal((cols, k)))
    return u @ v.T


def _orthogonal(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.linalg.qr(rng.standard_normal((n, n)))[0]


def _awq_roundtrip_check(rng: np.random.Generator) -> dict:
    """Pack a synthetic HF-layout weight into the AutoAWQ GEMM int4 layout,
    dequantize with dequantize_awq_gemm, and compare (offline validation of
    the unpack/reorder path, same construction as independence-tests)."""
    out_dim, in_dim, group_size = 64, 256, 128
    groups = in_dim // group_size
    weight = rng.standard_normal((out_dim, in_dim)).astype(np.float32)
    grouped = weight.reshape(out_dim, groups, group_size)
    scale16 = (np.abs(grouped).max(axis=2, keepdims=True) / 7.0 + 1e-8).astype(np.float16)
    zero = 8  # |w|/scale <= 7, so q = round(w/s) + 8 stays in [1, 15]
    quantized = np.clip(np.round(grouped / scale16.astype(np.float32)) + zero, 0, 15).astype(
        np.int32
    )
    transposed = quantized.reshape(out_dim, in_dim).T.copy()  # [in, out]
    n_pack = out_dim // tensors.AWQ_PACK_NUM
    qweight = np.zeros((in_dim, n_pack), dtype=np.int32)
    qzeros = np.zeros((groups, n_pack), dtype=np.int32)
    zeros = np.full((groups, out_dim), zero, dtype=np.int32)
    packed_src = (
        np.arange(out_dim).reshape(-1, tensors.AWQ_PACK_NUM)[:, tensors.AWQ_ORDER].reshape(-1)
    )
    for col in range(n_pack):
        for k in range(tensors.AWQ_PACK_NUM):
            src = packed_src[col * tensors.AWQ_PACK_NUM + k]
            qweight[:, col] |= transposed[:, src] << (tensors.AWQ_BITS * k)
            qzeros[:, col] |= zeros[:, src] << (tensors.AWQ_BITS * k)
    scales = np.ascontiguousarray(scale16.reshape(out_dim, groups).T)  # [groups, out]
    dequantized = tensors.dequantize_awq_gemm(qweight, qzeros, scales, group_size).T
    error = np.abs(dequantized - weight)
    return {
        "max_abs_err": float(error.max()),
        "scale_max": float(scales.max()),
        "corr": float(np.corrcoef(dequantized.ravel(), weight.ravel())[0, 1]),
    }


def _gguf_roundtrip_check(rng: np.random.Generator) -> dict:
    """Quantize/dequantize a synthetic tensor with gguf-py (Q8_0)."""
    try:
        import gguf.quants as gguf_quants
        from gguf import GGMLQuantizationType
    except ImportError:
        return {"skipped": "gguf-py not installed"}
    weight = rng.standard_normal((16, 64)).astype(np.float32)
    packed = gguf_quants.quantize(weight, GGMLQuantizationType.Q8_0)
    restored = gguf_quants.dequantize(packed, GGMLQuantizationType.Q8_0)
    return {
        "corr": float(np.corrcoef(restored.ravel(), weight.ravel())[0, 1]),
        "shape_ok": tuple(restored.shape) == tuple(weight.shape),
        "max_abs_err": float(np.abs(restored - weight).max()),
    }


def _gguf_file_extraction_check() -> dict:
    """Reload a GGUFWriter-written 3-D tensor through the real ``_load_gguf``.

    ``GGUFReader`` exposes ``tensor.shape`` in reversed GGML file order while
    the dequantized array is already in the logical (numpy) layout; a
    regression that reshapes to ``tensor.shape`` would scramble every axis.
    The decoy router tensor must stay excluded (GGUF router name).
    """
    try:
        from gguf import GGMLQuantizationType, GGUFWriter
    except ImportError:
        return {"skipped": "gguf-py not installed"}
    values = np.arange(4 * 6 * 10, dtype=np.float32).reshape(4, 6, 10)
    router = np.ones((4, 10), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.gguf"
        writer = GGUFWriter(str(path), "llama")
        writer.add_tensor("blk.0.ffn_gate_exps.weight", values, raw_dtype=GGMLQuantizationType.F32)
        writer.add_tensor("blk.0.ffn_gate_inp.weight", router, raw_dtype=GGMLQuantizationType.F32)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        original = tensors._download_file
        tensors._download_file = lambda *args, **kwargs: path
        try:
            extraction = tensors._load_gguf("synthetic/repo", "rev", None, ["synthetic.gguf"])
        finally:
            tensors._download_file = original
    gates = extraction.layers.get(0, [])
    return {
        "n_experts": len(gates),
        "shapes": [list(gate.shape) for gate in gates],
        "values_ok": all(
            np.array_equal(gate.sv, _gate.svd_spectrum(values[e])) for e, gate in enumerate(gates)
        ),
        "router_excluded": all("gate_inp" not in gate.name for gate in gates),
        "quant_types": extraction.source["quant_types"],
    }


def _synthetic_family_checks(rng: np.random.Generator) -> list[tuple[str, bool, dict]]:
    """Extraction-layer properties: slicing, role tags, router exclusion."""
    checks: list[tuple[str, bool, dict]] = []

    # Granite fused gate+up: chunk(2) first half is the gate (contract C11).
    fused = rng.standard_normal((4, 64, 32)).astype(np.float32)
    extraction = tensors.extract_from_tensors(
        {
            "model.layers.3.block_sparse_moe.input_linear.weight": fused,
            "model.layers.3.block_sparse_moe.router.layer.weight": rng.standard_normal((4, 32)),
        }
    )
    layers = extraction.layers.get(3, [])
    checks.append(
        (
            "granite_fused_split",
            extraction.family == "granite_moe"
            and len(layers) == 4
            and all(
                layers[e].shape == (32, 32)
                and np.array_equal(layers[e].sv, _gate.svd_spectrum(fused[e, :32, :]))
                for e in range(4)
            ),
            {"family": extraction.family, "tensors": len(layers)},
        )
    )

    # Router exclusion: all collision patterns, LFM2's router, and GGUF's.
    router_names = [
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.block_sparse_moe.gate.weight",
        "model.layers.0.block_sparse_moe.router.layer.weight",
        "blocks.0.ffn.router.w",
        "model.layers.0.mlp.router.weight",
        "model.layers.0.feed_forward.gate.weight",
        "blk.0.ffn_gate_inp.weight",
    ]
    gate = rng.standard_normal((8, 6)).astype(np.float32)
    extraction = tensors.extract_from_tensors(
        {name: rng.standard_normal((4, 6)) for name in router_names}
        | {"model.layers.0.mlp.gate_proj.weight": gate}
    )
    names = [tensor.name for tensor in extraction.layers.get(0, [])]
    checks.append(
        (
            "router_exclusion",
            names == ["model.layers.0.mlp.gate_proj.weight"]
            and all(tensors.is_router(name) for name in router_names),
            {"extracted": names},
        )
    )

    # gpt-oss interleaved gate/up: even columns are the gate (contract C12).
    interleaved = np.arange(2 * 8 * 10, dtype=np.float32).reshape(2, 8, 10)
    extraction = tensors.extract_from_tensors(
        {"model.layers.1.mlp.experts.gate_up_proj": interleaved}
    )
    experts = extraction.layers.get(1, [])
    checks.append(
        (
            "gptoss_deinterleave",
            len(experts) == 2
            and all(
                experts[e].shape == (8, 5)
                and np.array_equal(experts[e].sv, _gate.svd_spectrum(interleaved[e][:, ::2]))
                for e in range(2)
            ),
            {"family": extraction.family, "shapes": [list(t.shape) for t in experts]},
        )
    )

    # Upcycle: heterogenous per-layer tensor sets and roles.
    extraction = tensors.extract_from_tensors(
        {
            "blocks.0.ffn.gate.weight": rng.standard_normal((16, 6)),
            "blocks.1.ffn.gate": rng.standard_normal((3, 8, 6)),
            "blocks.1.ffn.shared_expert.gate.weight": rng.standard_normal((12, 6)),
            "blocks.1.ffn.router.w": rng.standard_normal((3, 6)),
        }
    )
    layer0 = extraction.layers.get(0, [])
    layer1 = extraction.layers.get(1, [])
    checks.append(
        (
            "upcycle_roles",
            extraction.family == "upcycle"
            and [(t.role) for t in layer0] == ["dense"]
            and sorted(t.role for t in layer1) == ["routed", "routed", "routed", "shared"],
            {"layer0": [t.role for t in layer0], "layer1": [t.role for t in layer1]},
        )
    )

    # Mixtral w1 is the gate; block_sparse_moe.gate is the router.
    checks.append(
        (
            "mixtral_w1_classification",
            tensors.classify("model.layers.2.block_sparse_moe.experts.5.w1.weight")
            == ("expert_gate", 2, "routed")
            and tensors.classify("model.layers.2.block_sparse_moe.gate.weight") is None,
            {},
        )
    )
    checks.append(
        (
            "gguf_names",
            tensors.classify("blk.4.ffn_gate_exps.weight") == ("gguf_exps", 4, "routed")
            and tensors.classify("blk.4.ffn_gate.weight") == ("gguf_gate", 4, "dense")
            and tensors.classify("blk.4.ffn_gate_inp.weight") is None,
            {},
        )
    )

    # LFM2.5: the pinned checkpoint has discrete experts; other LFM releases
    # fuse gate+up into one stacked tensor. Both layouts must classify.
    checks.append(
        (
            "lfm2_layouts",
            tensors.classify("model.layers.2.feed_forward.experts.17.w1.weight")
            == ("expert_gate", 2, "routed")
            and tensors.classify("model.layers.0.feed_forward.w1.weight")
            == ("dense_gate", 0, "dense")
            and tensors.classify("model.layers.2.feed_forward.gate.weight") is None
            and tensors.classify("model.layers.2.feed_forward.experts.gate_up_proj")
            == ("lfm2_fused", 2, "routed"),
            {},
        )
    )

    # Step-directory resolution (upcycle) and GGUF file choice.
    step_files = ["step_1000/model.safetensors", "step_304999/model.safetensors"]
    checks.append(
        (
            "step_resolution",
            tensors._resolve_step({"step": 1000}, step_files) == 1000
            and tensors._resolve_step({}, step_files) == 304999
            and tensors._resolve_step({"step": 7}, ["model.safetensors"]) is None,
            {},
        )
    )
    checks.append(
        (
            "gguf_file_choice",
            tensors._choose_gguf(["b.Q4_K_M.gguf", "b.Q8_0.gguf", "mmproj.gguf"]) == "b.Q8_0.gguf"
            and tensors._choose_gguf(["b.Q4_K_M.gguf", "mmproj.gguf"]) == "b.Q4_K_M.gguf",
            {},
        )
    )
    return checks


def _mxfp4_dequant_check(rng: np.random.Generator) -> dict:
    """Bit-exact MXFP4 decode against transformers' LUT path (0 ULP).

    The reference kernel returns the HF layout ``[..., G*32, O]`` (its
    ``transpose(1, 2)``); the numpy dequantizer keeps the packed layout
    ``[..., O, G*32]``, so the comparison transposes the reference back.
    """
    try:
        import torch
        from transformers.integrations.mxfp4 import _convert_moe_packed_tensors
    except ImportError:
        return {"skipped": "transformers/torch not installed"}
    # E=2, O=8, G=3 groups of 32 values; the scale range keeps 2**x finite.
    blocks = rng.integers(0, 256, (2, 8, 3, 16), dtype=np.uint8)
    scales = rng.integers(107, 148, (2, 8, 3), dtype=np.uint8)
    dequantized = tensors.dequantize_mxfp4(blocks, scales)
    reference = _convert_moe_packed_tensors(
        torch.from_numpy(blocks), torch.from_numpy(scales), dtype=torch.float32
    ).numpy()
    reference = reference.transpose(0, 2, 1)
    shape_ok = dequantized.shape == (2, 8, 3 * 32)
    return {
        "shape_ok": shape_ok,
        "layout_ok": dequantized.shape == reference.shape,
        "exact_equal": bool(np.array_equal(dequantized, reference)),
        "max_abs_err": float(np.abs(dequantized - reference).max()) if shape_ok else None,
    }


def _bin_extraction_check(rng: np.random.Generator) -> dict:
    """Synthetic ``pytorch_model.bin`` through the real ``_load_bin`` path.

    ``torch.load`` is monkeypatched to return a tinymixtral-shaped state dict,
    so the check exercises name classification and the stacked pass-through
    (axis 0 = expert, no row split, no de-interleave) without touching disk
    weights. ``weights_only=True`` is part of the contract.
    """
    try:
        import torch
    except ImportError:
        return {"skipped": "torch not installed"}
    stacked = rng.standard_normal((2, 16, 8)).astype(np.float32)
    router = rng.standard_normal((6, 16)).astype(np.float32)
    state = {
        "layers.0.moe.gate_proj": torch.from_numpy(stacked),
        "layers.0.moe.router.weight": torch.from_numpy(router),
        "model.embed_tokens.weight": torch.zeros((4, 4)),
    }
    load_kwargs: dict = {}
    original_download = tensors._download_file
    original_load = torch.load
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pytorch_model.bin"
        path.write_bytes(b"synthetic")

        def fake_load(*args, **kwargs):
            load_kwargs.update(kwargs)
            return state

        tensors._download_file = lambda *args, **kwargs: path
        torch.load = fake_load
        try:
            extraction = tensors._load_bin("synthetic/bin", "rev", None, ["pytorch_model.bin"])
        finally:
            tensors._download_file = original_download
            torch.load = original_load
    gates = extraction.layers.get(0, [])
    return {
        "n_experts": len(gates),
        "names": [gate.name for gate in gates],
        "shapes": [list(gate.shape) for gate in gates],
        "values_ok": all(
            np.array_equal(gate.sv, _gate.svd_spectrum(stacked[e])) for e, gate in enumerate(gates)
        ),
        "router_excluded": all("router" not in gate.name for gate in gates),
        "weights_only": load_kwargs.get("weights_only"),
        "format": extraction.format,
    }


def _gguf_unknown_type_check() -> dict:
    """``GGUFReader``'s ValueError (unknown type code) becomes an abstain."""

    class _BadReader:
        def __init__(self, *args, **kwargs):
            raise ValueError("unknown GGML quantization type")

    original_reader = tensors.GGUFReader
    original_download = tensors._download_file
    tensors.GGUFReader = _BadReader
    tensors._download_file = lambda *args, **kwargs: Path("synthetic.gguf")
    try:
        try:
            tensors._load_gguf("synthetic/gguf", "rev", None, ["b.Q8_0.gguf"])
        except tensors.FormatUnreadable as exc:
            return {"raised": True, "detail": str(exc)}
    finally:
        tensors.GGUFReader = original_reader
        tensors._download_file = original_download
    return {"raised": False}


def _new_capability_checks(rng: np.random.Generator) -> list[tuple[str, bool, dict]]:
    """New families, MXFP4, stacked .bin streaming, and the GGUF type guard."""
    checks: list[tuple[str, bool, dict]] = []

    # New family patterns; dense_h_to_4h is the UP branch and MTP is a
    # speculative-decoding head, so both must stay unclassified.
    expected = {
        "gpt_neox.layers.3.mlp.dense_4h_to_h.weight": ("gpt_neox_fallback", 3, "dense"),
        "gpt_neox.layers.3.mlp.dense_h_to_4h.weight": None,
        "model.language_model.layers.5.mlp.gate_proj.weight": ("dense_gate", 5, "dense"),
        "backbone.layers.7.mixer.down_proj.weight": ("nemotron_fallback", 7, "dense"),
        "mtp.layers.0.mlp.gate_proj.weight": None,
    }
    actual = {name: tensors.classify(name) for name in expected}
    checks.append(("new_family_classification", actual == expected, {"classification": actual}))

    # tinymixtral: axis 0 is the expert axis and the tensor is already
    # gate-only, so no row split and no even-column de-interleave may run.
    stacked = rng.standard_normal((4, 6, 10)).astype(np.float32)
    extraction = tensors.extract_from_tensors({"layers.4.moe.gate_proj": stacked})
    experts = extraction.layers.get(4, [])
    checks.append(
        (
            "tinymixtral_stacked_passthrough",
            extraction.family == "tinymixtral"
            and [tensor.name for tensor in experts]
            == [f"layers.4.moe.gate_proj#expert{e}" for e in range(4)]
            and all(
                tensor.shape == stacked.shape[1:]
                and tensor.q == min(stacked.shape[1:])
                and np.array_equal(tensor.sv, _gate.svd_spectrum(stacked[e]))
                for e, tensor in enumerate(experts)
            ),
            {
                "family": extraction.family,
                "shapes": [list(tensor.shape) for tensor in experts],
                "q": [tensor.q for tensor in experts],
            },
        )
    )

    mxfp4 = _mxfp4_dequant_check(rng)
    if "skipped" in mxfp4:
        checks.append(("mxfp4_dequant", True, mxfp4))
    else:
        checks.append(
            (
                "mxfp4_dequant",
                mxfp4["shape_ok"] and mxfp4["layout_ok"] and mxfp4["exact_equal"],
                mxfp4,
            )
        )

    bin_check = _bin_extraction_check(rng)
    if "skipped" in bin_check:
        checks.append(("bin_stacked_extraction", True, bin_check))
    else:
        checks.append(
            (
                "bin_stacked_extraction",
                bin_check["n_experts"] == 2
                and bin_check["names"] == [f"layers.0.moe.gate_proj#expert{e}" for e in range(2)]
                and all(shape == [16, 8] for shape in bin_check["shapes"])
                and bin_check["values_ok"]
                and bin_check["router_excluded"]
                and bin_check["weights_only"] is True,
                bin_check,
            )
        )

    guard = _gguf_unknown_type_check()
    checks.append(("gguf_unknown_type_guard", guard["raised"], guard))
    return checks


def run_selftest() -> int:
    """Exercise the full scoring path offline; exit 0 on success."""
    started = time.perf_counter()
    checks: list[tuple[str, bool, dict]] = []
    rng = np.random.default_rng(20261026)

    # 1) Effective rank: k equal singular values -> f = exp(H) = k.
    rank_k = _rank_k_matrix(rng, 60, 40, 6)
    feature = _gate.spectral_entropy_feature(rank_k, R_DEFAULT)
    checks.append(
        (
            "rank_k_effective_rank",
            abs(feature["f"] - 6.0) < 1e-4 and feature["r_eff"] == 40,
            {"f": feature["f"], "r_eff": feature["r_eff"], "q": feature["q"]},
        )
    )

    # 2) Scale invariance: sigma_bar(c*W) = sigma_bar(W).
    base = _gate.spectral_entropy_feature(rank_k, R_DEFAULT)["f"]
    scaled = _gate.spectral_entropy_feature(1e3 * rank_k, R_DEFAULT)["f"]
    checks.append(
        (
            "scale_invariance",
            abs(scaled - base) / base < 1e-7,
            {"f": base, "f_scaled": scaled},
        )
    )

    # 3) Orthogonal reparameterization: sigma(P W Q) = sigma(W).
    left = _orthogonal(rng, 60)
    right = _orthogonal(rng, 40)
    reparam = _gate.spectral_entropy_feature(left @ rank_k @ right, R_DEFAULT)["f"]
    checks.append(
        (
            "orthogonal_reparam_invariance",
            abs(reparam - base) / base < 1e-6,
            {"f": base, "f_reparam": reparam},
        )
    )

    # 4) Permutation invariance: row/column shuffles leave the spectrum alone.
    permuted = rank_k[rng.permutation(60)][:, rng.permutation(40)]
    permuted_f = _gate.spectral_entropy_feature(permuted, R_DEFAULT)["f"]
    checks.append(
        (
            "permutation_invariance",
            abs(permuted_f - base) / base < 1e-9,
            {"f": base, "f_permuted": permuted_f},
        )
    )

    # 5) r_eff guard: q < r -> r_eff = q and f is bounded by r_eff.
    small = rng.standard_normal((40, 100))
    small_feature = _gate.spectral_entropy_feature(small, R_DEFAULT)
    checks.append(
        (
            "r_eff_guard",
            small_feature["q"] == 40
            and small_feature["r_eff"] == 40
            and small_feature["spectrum_fraction"] == 1.0
            and small_feature["f"] <= 40.0 + 1e-9,
            small_feature,
        )
    )

    # 5b) C1 discrimination: Eq. (2) must normalize over the retained top-r
    # only. q=256 with top-128 = 1.0 and a 0.5 tail gives f = 128 exactly;
    # normalizing over all q would give ~241.9.
    discriminating = np.concatenate([np.ones(128), np.full(128, 0.5)])
    c1_feature = _gate.entropy_feature(discriminating, R_DEFAULT)
    checks.append(
        (
            "top_r_normalization",
            abs(c1_feature["f"] - 128.0) < 1e-3,
            {"f": c1_feature["f"], "r_eff": c1_feature["r_eff"], "q": c1_feature["q"]},
        )
    )

    # 6) All-zero guard: H = 0, f = 1, flagged, never NaN.
    zero_feature = _gate.spectral_entropy_feature(np.zeros((8, 16)), R_DEFAULT)
    checks.append(
        (
            "all_zero_guard",
            zero_feature["all_zero"] and zero_feature["H"] == 0.0 and zero_feature["f"] == 1.0,
            zero_feature,
        )
    )

    # 7) CKA sign-insensitivity: r = -0.95 -> CKA = 0.9025, still "related".
    profile_a = rng.standard_normal(8)
    profile_a -= profile_a.mean()
    orthogonal = rng.standard_normal(8)
    orthogonal -= orthogonal.mean()
    orthogonal -= orthogonal.dot(profile_a) / profile_a.dot(profile_a) * profile_a
    unit_a = profile_a / np.linalg.norm(profile_a)
    unit_orthogonal = orthogonal / np.linalg.norm(orthogonal)
    profile_b = -0.95 * unit_a + np.sqrt(1.0 - 0.95**2) * unit_orthogonal
    cka, signed = _gate.cka_linear_1d(profile_a, profile_b)
    sim = _gate.similarity(profile_a, profile_b)
    checks.append(
        (
            "cka_sign_insensitivity",
            cka is not None
            and abs(cka - 0.9025) < 1e-12
            and signed is not None
            and abs(signed + 0.95) < 1e-12,
            {"cka": cka, "signed_pearson": signed},
        )
    )
    checks.append(
        (
            "anticorrelation_collision_flag",
            sim["anticorrelation_collision"] and sim["cka"] is not None and sim["cka"] >= TAU,
            sim,
        )
    )
    checks.append(
        (
            "verdict_threshold",
            _gate.verdict_for(0.9) == "related" and _gate.verdict_for(0.899999) == "unrelated",
            {"tau": TAU},
        )
    )

    # 8) Zero-variance abstain: a constant profile has no defined CKA.
    degenerate_cka, degenerate_r = _gate.cka_linear_1d(np.ones(8), np.arange(8.0))
    checks.append(
        (
            "zero_variance_abstain",
            degenerate_cka is None and degenerate_r is None,
            {"cka": degenerate_cka},
        )
    )

    # 9) Segment pooling: array_split semantics and remainder placement.
    values = np.arange(26, dtype=np.float64)
    pooled = _gate.pool_segments(values, SEGMENTS)
    expected = np.array([group.mean() for group in np.array_split(values, SEGMENTS)])
    sizes = _gate.segment_sizes(26, SEGMENTS)
    checks.append(
        (
            "segment_remainder",
            np.allclose(pooled, expected)
            and sizes == [4, 4, 3, 3, 3, 3, 3, 3]
            and pooled.shape == (8,),
            {"sizes": sizes, "profile": pooled.tolist()},
        )
    )

    # 10) Layer aggregation + integer depth ordering (MoE mean, dense degenerates).
    spectra = {
        5: [TensorSpectrum("late", "dense", _gate.svd_spectrum(_rank_k_matrix(rng, 20, 10, 2)))],
        1: [
            TensorSpectrum("early_a", "routed", _gate.svd_spectrum(_rank_k_matrix(rng, 20, 10, 8))),
            TensorSpectrum("early_b", "shared", _gate.svd_spectrum(_rank_k_matrix(rng, 20, 10, 2))),
        ],
    }
    layer_scalars = _gate.layer_features(spectra, R_DEFAULT)
    checks.append(
        (
            "layer_aggregation_and_depth_order",
            [layer["layer"] for layer in layer_scalars] == [1, 5]
            and abs(layer_scalars[0]["f"] - 5.0) < 1e-4
            and abs(layer_scalars[1]["f"] - 2.0) < 1e-4,
            {
                "layers": [layer["layer"] for layer in layer_scalars],
                "f": [layer["f"] for layer in layer_scalars],
            },
        )
    )

    # 10b) Depth order is numeric, not lexicographic: keys {2, 10} -> [2, 10].
    two_digit = {
        10: [TensorSpectrum("ten", "dense", _gate.svd_spectrum(_rank_k_matrix(rng, 12, 8, 3)))],
        2: [TensorSpectrum("two", "dense", _gate.svd_spectrum(_rank_k_matrix(rng, 12, 8, 3)))],
    }
    two_digit_order = [layer["layer"] for layer in _gate.layer_features(two_digit, R_DEFAULT)]
    checks.append(
        (
            "depth_order_two_digit",
            two_digit_order == [2, 10],
            {"order": two_digit_order},
        )
    )

    # 10c) MoE diagnostic variants on a two-tensor layer (diagnostics only).
    # Spectra [1,1,0,0] (f=2) and [1,1,1,1] (f=4): arithmetic mean 3,
    # geometric sqrt(2*4) = sqrt(8), concatenated spectrum has six equal
    # nonzero values renormalized together, so f = 6.
    moe_layer = {
        0: [
            TensorSpectrum("a", "routed", np.array([1.0, 1.0, 0.0, 0.0])),
            TensorSpectrum("b", "routed", np.array([1.0, 1.0, 1.0, 1.0])),
        ]
    }
    arithmetic = _gate.layer_features(moe_layer, R_DEFAULT)[0]["f"]
    geometric = _gate.geometric_layer_features(moe_layer, R_DEFAULT)[0]["f"]
    concatenated = _gate.concatenated_layer_features(moe_layer, R_DEFAULT)[0]["f"]
    checks.append(
        (
            "moe_diagnostic_variants",
            abs(arithmetic - 3.0) < 1e-12
            and abs(geometric - np.sqrt(8.0)) < 1e-12
            and abs(concatenated - 6.0) < 1e-12,
            {"arithmetic": arithmetic, "geometric": geometric, "concatenated": concatenated},
        )
    )

    # 11) End-to-end comparison statuses + sensitivity panel keys.
    related_a = {
        layer: [
            TensorSpectrum(
                f"a{layer}",
                "routed",
                _gate.svd_spectrum(_rank_k_matrix(rng, 16, 8, 2 + layer % 7)),
            )
        ]
        for layer in range(12)
    }
    related_b = {
        layer: [
            TensorSpectrum(
                f"b{layer}",
                "routed",
                _gate.svd_spectrum(_rank_k_matrix(rng, 16, 8, 2 + layer % 7)),
            )
        ]
        for layer in range(12)
    }
    related = _gate.compare(related_a, related_b, R_DEFAULT, SEGMENTS)
    panel = _gate.sensitivity_panel(related_a, related_b, related)
    degenerate = _gate.compare(
        {layer: [TensorSpectrum("x", "dense", np.array([1.0, 0.0, 0.0]))] for layer in range(8)},
        related_b,
        R_DEFAULT,
        SEGMENTS,
    )
    checks.append(
        (
            "comparison_and_sensitivity",
            related["status"] == "ok"
            and related["cka"] is not None
            and related["cka"] >= TAU
            and set(panel)
            == {
                "r64",
                "r256",
                "s4",
                "s16",
                "experts_only",
                "shared_only",
                "geometric",
                "concatenated",
            }
            and panel["experts_only"]["status"] == "ok"
            and panel["shared_only"]["status"] == "unavailable"
            and degenerate["status"] == "degenerate_profile",
            {
                "cka": related["cka"],
                "panel_status": {key: value["status"] for key, value in panel.items()},
                "degenerate": degenerate["status"],
            },
        )
    )

    # 12) Profile diagnostics shape.
    diagnostics = _gate.profile_diagnostics(related["features"]["a"], related["profile"]["a"])
    checks.append(
        (
            "profile_diagnostics",
            diagnostics["L"] == 12
            and diagnostics["n_gate_tensors"] == 12
            and diagnostics["layers_per_segment"] == [2, 2, 2, 2, 1, 1, 1, 1]
            and len(diagnostics["r_eff_per_layer"]) == 12
            and diagnostics["r_eff_per_layer"][0] == {"layer": 0, "min": 8, "max": 8},
            diagnostics | {"profile": None},
        )
    )

    # 13) AWQ int4 round-trip (validates unpack/reorder, contract C13).
    roundtrip = _awq_roundtrip_check(np.random.default_rng(7))
    checks.append(
        (
            "awq_roundtrip",
            roundtrip["corr"] > 0.99
            and roundtrip["max_abs_err"] <= 0.5 * roundtrip["scale_max"] + 1e-4,
            roundtrip,
        )
    )

    # 14) GGUF quantize/dequantize round-trip (skip-with-note if absent).
    gguf_roundtrip = _gguf_roundtrip_check(np.random.default_rng(11))
    if "skipped" in gguf_roundtrip:
        checks.append(("gguf_roundtrip", True, gguf_roundtrip))
    else:
        checks.append(
            (
                "gguf_roundtrip",
                gguf_roundtrip["corr"] > 0.999 and gguf_roundtrip["shape_ok"],
                gguf_roundtrip,
            )
        )

    # 14b) Synthetic .gguf written with GGUFWriter and reloaded through the
    # real extraction path (catches GGML-order reshape scrambles).
    gguf_file = _gguf_file_extraction_check()
    if "skipped" in gguf_file:
        checks.append(("gguf_file_extraction", True, gguf_file))
    else:
        checks.append(
            (
                "gguf_file_extraction",
                gguf_file["n_experts"] == 4
                and all(shape == [6, 10] for shape in gguf_file["shapes"])
                and gguf_file["values_ok"]
                and gguf_file["router_excluded"]
                and gguf_file["quant_types"] == ["F32"],
                gguf_file,
            )
        )

    # 15) Family extraction (granite split, routers, gpt-oss, upcycle, GGUF names).
    checks.extend(_synthetic_family_checks(rng))

    # 15b) New extraction surface: GPTNeoX/Qwen3.5/Nemotron patterns,
    # stacked `.bin` pass-through, MXFP4 dequant, GGUF type guard.
    checks.extend(_new_capability_checks(rng))

    # 16) Expectation-outcome classification (gate policy, offline).
    outcome_cases = [
        ({"verdict": "related"}, "related", OUTCOME_HELD),
        ({"verdict": "unrelated"}, "unrelated", OUTCOME_HELD),
        ({"verdict": "abstain", "reason": "format_unreadable"}, "related", OUTCOME_SKIPPED_FORMAT),
        ({"verdict": "abstain", "reason": "no_gate_tensors"}, "unrelated", OUTCOME_SKIPPED_FORMAT),
        ({"verdict": "abstain", "reason": "degenerate_profile"}, "related", OUTCOME_MISSED),
        ({"verdict": "unrelated"}, "related", OUTCOME_MISSED),
        ({"verdict": "related"}, "unrelated", OUTCOME_MISSED),
    ]
    checks.append(
        (
            "expectation_outcomes",
            all(
                _expectation_outcome(entry, expected)[0] == want
                for entry, expected, want in outcome_cases
            ),
            {"cases": len(outcome_cases), "outcomes": sorted({want for *_, want in outcome_cases})},
        )
    )

    output = {
        "smoke": "gate-spectral",
        "mode": "selftest",
        "params": {"r": R_DEFAULT, "s": SEGMENTS, "tau": TAU, "eps": EPS},
        "checks": [
            {"name": name, "passed": passed, "detail": detail} for name, passed, detail in checks
        ],
        "all_passed": all(passed for _, passed, _ in checks),
        "seconds": round(time.perf_counter() - started, 2),
    }
    print(json.dumps(output, indent=2, default=float))
    return 0 if output["all_passed"] else 1


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="GATE spectral smoke (smokes/gate-spectral/)")
    parser.add_argument(
        "--selftest", action="store_true", help="offline synthetic self-test (no network)"
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="HF cache directory")
    parser.add_argument(
        "--pair", default=None, help="run one ProvenanceBench pair id instead of the default set"
    )
    args = parser.parse_args()

    if args.selftest:
        sys.exit(run_selftest())

    specs = load_pair_specs()
    if args.pair:
        if args.pair not in specs:
            parser.error(f"unknown pair id: {args.pair} (not in provenancebench/pairs.yaml)")
        selected = [(args.pair, "adhoc", None)]
    else:
        selected = list(DEFAULT_PAIRS)

    started = time.perf_counter()
    entries = []
    for pair_id, role, expected in selected:
        spec = specs[pair_id]
        if expected is None:
            expected = _EXPECTED_FROM_YAML.get(spec.expected, "abstain")
        entries.append(run_pair(spec, role, expected, args.cache_dir))

    summary = {
        "pairs_total": len(entries),
        "held": [entry["pair_id"] for entry in entries if entry["outcome"] == OUTCOME_HELD],
        "missed": [entry["pair_id"] for entry in entries if entry["outcome"] == OUTCOME_MISSED],
        "skipped_format": [
            entry["pair_id"] for entry in entries if entry["outcome"] == OUTCOME_SKIPPED_FORMAT
        ],
    }
    output = {
        "smoke": "gate-spectral",
        "paper": "Model-intrinsic, Weight-level, and Training-free Spectral Fingerprinting "
        "for Large Language Models (He et al.), arXiv:2610.30626",
        "statistic": "GATE gate-projection spectral entropy, depth-segment pooling, linear CKA",
        "params": {
            "r": R_DEFAULT,
            "s": SEGMENTS,
            "tau": TAU,
            "eps": EPS,
            "sensitivity": {"r": [64, 256], "s": [4, 16]},
        },
        "pairs": entries,
        "summary": summary,
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    print(json.dumps(output, indent=2, default=float))
    # Gate: fail only on hard expectation misses; skipped_format is recorded
    # but does not fail the run.
    sys.exit(1 if summary["missed"] else 0)


if __name__ == "__main__":
    main()
