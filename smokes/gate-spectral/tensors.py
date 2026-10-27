"""Family-aware gate-tensor extraction for the GATE spectral smoke.

The pure spectral core (``_gate.py``) never sees a tensor name; this module
turns a checkpoint into per-layer singular-value spectra plus the MoE role
tags the diagnostics need. Spectra are computed at extraction time and the
float matrices are dropped immediately, so peak memory is one tensor instead
of the whole model (30B-scale MoE pairs are otherwise unusable). The family
table follows the pinned MoE naming census for the #26 workflow and the
implementation contract:

Dense llama-style      model.layers.N.mlp.gate_proj.weight
Qwen3.5 (nested)       model.language_model.layers.N.mlp.gate_proj.weight
Granite MoE (HF)       model.layers.N.block_sparse_moe.input_linear.weight
                       [E, 2*ffn, d] fused gate+up -> gate = chunk(2)[0]
Granite MoE (GGUF)     blk.N.ffn_gate_exps.weight [E, ffn, d] (already split)
Mixtral / llm-jp       model.layers.N.block_sparse_moe.experts.M.w1.weight
OLMoE / Qwen3-MoE      model.layers.N.mlp.experts.M.gate_proj.weight
gpt-oss (bf16 re-save) model.layers.N.mlp.experts.gate_up_proj
                       [E, d, 2*d] interleaved -> gate = [:, :, ::2]
gpt-oss (MXFP4)        ...experts.gate_up_proj_blocks / _scales -> dequant
LFM2.5 (fused)         model.layers.N.feed_forward.experts.gate_up_proj
                       -> gate = [:ffn, :]; dense L0-L1 feed_forward.w1.weight
LFM2.5 (discrete)      model.layers.N.feed_forward.experts.M.w1.weight
                       (one observed layout), dense L0-L1 feed_forward.w1.weight
AlexWortega upcycle    blocks.N.ffn.gate [E, ffn, d] (no ``.weight`` suffix),
                       blocks.N.ffn.shared_expert.gate.weight,
                       blocks.0.ffn.gate.weight (dense)
tinymixtral (.bin)     layers.N.moe.gate_proj [E, ffn, d] (stacked, gate only)
GPT-2 style (paper †)  transformer.h.N.mlp.c_proj.weight
GPTNeoX (†)            gpt_neox.layers.N.mlp.dense_4h_to_h.weight
                       (dense_h_to_4h is the UP projection, not the analog)
Nemotron-H (†)         backbone.layers.N.mixer.down_proj.weight (relu2 MLP)

The † families are non-gated architectures: their extraction is a disclosed
per-model fallback to the pre-activation-free down projection, not the paper's
gate branch. The paper only tested the analogous GPT-2 ``c_proj`` fallback, so
every GPTNeoX/Nemotron-H result carries the caveat in its diagnostics.

Router (/logit) matrices are excluded in every family — the five name
collisions the census warns about are ``mlp.gate.weight`` (Qwen3-MoE/OLMoE),
``block_sparse_moe.gate.weight`` (Mixtral), ``block_sparse_moe.router.*``
(granite), ``blocks.N.ffn.router.*`` (upcycle), ``mlp.router.*`` (gpt-oss),
plus LFM2's ``feed_forward.gate.weight`` and the GGUF ``blk.N.ffn_gate_inp.weight``.
A router maps d_m -> n_experts and
never passes an activation before the element-wise UP product (paper Eq. 1).

Quant paths: AWQ GEMM int4 (``dequantize_awq_gemm``, lifted verbatim from
``smokes/independence-tests/smoke.py``, AutoAWQ Apache-2.0 semantics), MXFP4
(``dequantize_mxfp4``, the gpt-oss packed 4-bit format), and FP8/bf16 via
``ml_dtypes`` casts, with a compressed-tensors per-output-channel
``weight_scale`` applied when stored beside an F8 tensor. GGUF goes through
``gguf.GGUFReader`` and ``gguf.quants.dequantize`` (MIT). The tinymixtral
``pytorch_model.bin`` path uses ``torch.load(weights_only=True)`` and imports
torch lazily (single-file, sharded, or index-declared state dicts).

Safetensors are read with a small header parser instead of the framework
loaders so float8/bf16 decode through ``ml_dtypes`` without torch and without
materializing non-gate tensors. When the Hub is reachable, gate tensors are
fetched with HTTP Range reads (one tensor at a time) so a 30B pair costs
gigabytes of traffic rather than a full snapshot; any range failure falls back
to ``snapshot_download`` and the local reader.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import gguf.quants
import ml_dtypes
import numpy as np
from _gate import R_DEFAULT, TensorSpectrum, svd_spectrum
from gguf import GGUFReader
from huggingface_hub import HfApi, get_session, hf_hub_download, hf_hub_url, snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from huggingface_hub.utils import get_token

# AWQ GEMM packing constants (AutoAWQ convention): 4-bit values, eight per
# int32 column, packed in the [0,2,4,6,1,3,5,7] order.
AWQ_BITS = 4
AWQ_PACK_NUM = 32 // AWQ_BITS
AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

#: Quant schemes whose dequantizer is not implemented here; a checkpoint that
#: declares one abstains as format_unreadable instead of scoring packed ints.
#: (mxfp4 is supported: see ``dequantize_mxfp4``.)
UNSUPPORTED_QUANT_METHODS = frozenset({"aqlm", "gptq", "quip", "gsq"})

#: MXFP4 (gpt-oss) constants: 32 values per E8M0-scaled block, 16 packed bytes.
MXFP4_BLOCK = 32
#: E2M1 code -> value (bit 3 is the sign); mirrors the OCP MX FP4 grid.
FP4_VALUES = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

#: Download retry policy (matches smokes/independence-tests).
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_BACKOFF_SECONDS = (2.0, 8.0)

#: PyTorch state-dict entry points (single file, sharded shards, index).
_BIN_INDEX = "pytorch_model.bin.index.json"
_BIN_SHARD_RE = re.compile(r"pytorch_model-\d+-of-\d+\.bin$")

#: Safetensors dtype tags -> numpy/ml_dtypes dtypes.
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
    """The checkpoint format cannot be consumed by this smoke."""


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

    def tensor_diagnostics(self, r: int = R_DEFAULT) -> list[dict]:
        """Per-tensor shape/q/r_eff/spectrum-fraction (contract diagnostics).

        Reads only the streamed record, so it works without any retained
        matrix.
        """
        out = []
        for layer in sorted(self.layers, key=int):
            for tensor in self.layers[layer]:
                q = tensor.q
                r_eff = min(int(r), q)
                out.append(
                    {
                        "name": tensor.name,
                        "layer": int(layer),
                        "role": tensor.role,
                        "shape": [int(d) for d in tensor.shape],
                        "q": q,
                        "r_eff": r_eff,
                        "spectrum_fraction": (r_eff / q) if q else 0.0,
                    }
                )
        return out


def _finish(
    layers: dict[int, list[TensorSpectrum]], kinds: Iterable[str], fmt: str, source: dict
) -> Extraction:
    """Sort each layer's spectra by name and assemble the extraction record."""
    for tensors_in_layer in layers.values():
        tensors_in_layer.sort(key=lambda tensor: tensor.name)
    return Extraction(layers=layers, family=_family_from_kinds(kinds), format=fmt, source=source)


# ---------------------------------------------------------------------------
# Name classification (family table + router exclusion) — pure, no I/O.
# ---------------------------------------------------------------------------

# (regex, kind, role). Anchored so router names can never sneak in through a
# prefix match; ``is_router`` is an explicit second gate (census C10).
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
        # LFM2.5 discrete experts (one observed layout; w1 is the gate).
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
    (
        # Qwen3.5 nests the decoder under language_model; the speculative
        # decoding head at ``mtp.layers.*`` is excluded by the anchor.
        re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate_proj\.weight$"),
        "dense_gate",
        "dense",
    ),
    (
        # GPTNeoX is non-gated (†): dense_4h_to_h is the DOWN projection the
        # paper's GPT-2 c_proj fallback stands in for. dense_h_to_4h is the
        # element-wise UP branch and is deliberately not matched.
        re.compile(r"^gpt_neox\.layers\.(\d+)\.mlp\.dense_4h_to_h\.weight$"),
        "gpt_neox_fallback",
        "dense",
    ),
    (
        # Nemotron-H relu2 MLP is non-gated (†); mixer.down_proj is the
        # disclosed analog. Never matches the routed_moe.* blocks.
        re.compile(r"^backbone\.layers\.(\d+)\.mixer\.down_proj\.weight$"),
        "nemotron_fallback",
        "dense",
    ),
    (
        # tinymixtral stores all experts in one flat Parameter (axis 0 =
        # expert) that is already gate-only: pass through, never split or
        # de-interleave.
        re.compile(r"^layers\.(\d+)\.moe\.gate_proj$"),
        "tinymixtral_stacked",
        "routed",
    ),
    # GPT-2 paper-† fallback. Two observed spellings for the same down
    # projection: the transformers export prefixes "transformer." while the
    # raw openai-community/gpt2 state dict has bare "h.N.mlp.c_proj.weight".
    (
        re.compile(r"^(?:transformer\.)?h\.(\d+)\.mlp\.c_proj\.weight$"),
        "gpt2_fallback",
        "dense",
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
#: ``tinymixtral_stacked`` is pass-through: the stored tensor is already
#: gate-only, so only the expert axis is unpacked (no fused split, no
#: interleaved de-interleave).
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
    ("lfm2_fused", "lfm2"),
    ("tinymixtral_stacked", "tinymixtral"),
    ("expert_gate", "moe_experts"),
    ("gpt_neox_fallback", "gpt_neox_fallback"),
    ("nemotron_fallback", "nemotron_h_fallback"),
    ("dense_gate", "dense"),
    ("gguf_gate", "dense"),
    ("gpt2_fallback", "gpt2_fallback"),
)


def is_router(name: str) -> bool:
    """True for router/logit matrices (excluded in every family)."""
    return any(pattern.match(name) for pattern in _ROUTER_PATTERNS)


def classify(name: str) -> tuple[str, int, str] | None:
    """Map a weight name to (kind, integer layer, role), or None if not a gate."""
    if is_router(name):
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


# ---------------------------------------------------------------------------
# Per-kind slicing into gate matrices.
# ---------------------------------------------------------------------------


def _split_stacked(kind: str, name: str, array: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Slice a stacked expert tensor into (tensor name, gate matrix) pairs."""
    if array.ndim != 3:
        raise FormatUnreadable(f"{name}: expected a 3-D stacked tensor, got {array.shape}")
    n_experts, rows, _ = array.shape
    if kind == "granite_fused":
        # JetMoe-style fused gate+up: chunk(2) along dim -2, first chunk = gate
        # (census C11; ffn_dim = rows // 2, 512 for granite-3.1-1a400m).
        if rows % 2:
            raise FormatUnreadable(f"{name}: fused gate+up rows not even ({rows})")
        gates = array[:, : rows // 2, :]
    elif kind == "gptoss_fused":
        # Gate and up are interleaved on the last axis; even columns = gate (C12).
        gates = array[:, :, ::2]
    elif kind == "lfm2_fused":
        if rows % 2:
            raise FormatUnreadable(f"{name}: fused gate+up rows not even ({rows})")
        gates = array[:, : rows // 2, :]
    else:
        # upcycle_routed / gguf_exps / tinymixtral_stacked: already gate-only
        # expert stacks; axis 0 is the only axis to unpack.
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
    """SVD one gate matrix now and keep only its spectrum (peak RAM = one matrix).

    The SVD runs on an fp32 contiguous copy (house convention) and the record
    keeps only the singular values plus shape/q provenance, so a checkpoint
    costs one gate matrix at a time rather than every matrix at once.
    """
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    layers.setdefault(layer, []).append(
        TensorSpectrum(name=name, role=role, sv=svd_spectrum(matrix), shape=tuple(matrix.shape))
    )


def _accumulate(
    layers: dict[int, list[TensorSpectrum]],
    kinds: set[str],
    items: Iterable[tuple[str, np.ndarray]],
) -> None:
    """Classify, slice, and store per-tensor spectra (no matrices retained)."""
    for name, array in items:
        classified = classify(name)
        if classified is None:
            continue
        kind, layer, role = classified
        kinds.add(kind)
        for tensor_name, matrix in _split_tensor(kind, name, array):
            _store_matrix(layers, tensor_name, matrix, role, layer)


def extract_from_tensors(tensors: Mapping[str, np.ndarray]) -> Extraction:
    """Gate tensors from an in-memory name->array mapping (offline/testable)."""
    layers: dict[int, list[TensorSpectrum]] = {}
    kinds: set[str] = set()
    _accumulate(layers, kinds, tensors.items())
    return _finish(layers, kinds, "memory", {})


# ---------------------------------------------------------------------------
# AWQ int4 dequantization — numpy port of AutoAWQ's dequantize_gemm
# (github.com/casper-hansen/AutoAWQ, awq/utils/packing_utils.py).
# Lifted verbatim from smokes/independence-tests/smoke.py (contract C13).
# ---------------------------------------------------------------------------


def dequantize_awq_gemm(
    qweight: np.ndarray,
    qzeros: np.ndarray,
    scales: np.ndarray,
    group_size: int,
    bits: int = AWQ_BITS,
) -> np.ndarray:
    """Dequantize an AWQ GEMM int4 layer.

    qweight [in, out/8] and qzeros [in/group_size, out/8] hold eight 4-bit
    values per int32 column (packed in AWQ_ORDER); scales is
    [in/group_size, out] float16. Returns the weight in AutoAWQ's
    [in_features, out_features] layout — transpose for the HF [out, in]
    `.weight` layout.
    """
    shifts = np.arange(0, 32, bits, dtype=np.int32)
    mask = (1 << bits) - 1
    iweight = (qweight[:, :, None] >> shifts) & mask
    iweight = iweight.reshape(qweight.shape[0], -1)
    izeros = (qzeros[:, :, None] >> shifts) & mask
    izeros = izeros.reshape(qzeros.shape[0], -1)
    # Undo the [0,2,4,6,1,3,5,7] packing order within each int32 column:
    # AutoAWQ reverses per 8-wide block (arange(n).view(-1, 8)[:, REVERSE]),
    # so each block gets its own offset, not a tiled pattern.
    n_cols = iweight.shape[1]
    blocks = np.arange(n_cols).reshape(-1, AWQ_PACK_NUM)
    reverse = blocks[:, AWQ_REVERSE_ORDER].reshape(-1)
    iweight = iweight[:, reverse]
    izeros = izeros[:, reverse]
    izeros = np.repeat(izeros, group_size, axis=0)
    scales_full = np.repeat(scales, group_size, axis=0).astype(np.float32)
    return (iweight - izeros).astype(np.float32) * scales_full


# ---------------------------------------------------------------------------
# MXFP4 (gpt-oss) dequantization — numpy port of the OCP MX FP4 block format.
# Bit-exact against transformers.integrations.mxfp4._convert_moe_packed_tensors
# on random fixtures (same LUT, same power-of-two scale, 0 ULP at fp32).
# ---------------------------------------------------------------------------


def dequantize_mxfp4(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize MXFP4 packed blocks to fp32.

    ``blocks`` is uint8 ``[..., groups, 16]``: each trailing 16 bytes hold 32
    E2M1 nibbles (low nibble = even element, high nibble = odd). ``scales`` is
    uint8 ``[..., groups]`` holding the E8M0 exponent ``2**(scale - 127)``.
    Returns fp32 ``[..., groups * 32]`` — the original GEMM orientation
    ``[E, out, in]``.
    """
    blocks = np.asarray(blocks)
    scales = np.asarray(scales)
    if blocks.ndim < 2 or blocks.shape[-1] != MXFP4_BLOCK // 2:
        raise FormatUnreadable(f"MXFP4 blocks: expected trailing 16 bytes, got {blocks.shape}")
    if blocks.dtype != np.uint8 or scales.dtype != np.uint8:
        raise FormatUnreadable(
            f"MXFP4 expects uint8 blocks/scales, got {blocks.dtype}/{scales.dtype}"
        )
    if scales.shape != blocks.shape[:-1]:
        raise FormatUnreadable(
            f"MXFP4 scales {scales.shape} do not match blocks {blocks.shape[:-1]}"
        )
    codes = np.empty(blocks.shape[:-1] + (MXFP4_BLOCK,), dtype=np.uint8)
    codes[..., 0::2] = blocks & 0x0F
    codes[..., 1::2] = blocks >> 4
    # E8M0 is unsigned: the exponent exceeds 127, so cast before subtracting.
    # ldexp is the reference's exact 2**(scale-127) scaling and, unlike a
    # multiply by np.exp2, cannot turn a zero code into 0*inf at degenerate
    # scales.
    exponent = scales.astype(np.int32) - 127
    with np.errstate(over="ignore"):
        values = np.ldexp(FP4_VALUES[codes], exponent[..., None])  # [..., groups, 32]
    return values.reshape(*blocks.shape[:-2], blocks.shape[-2] * MXFP4_BLOCK)


def _mxfp4_gates(base: str, blocks: np.ndarray, scales: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Per-expert gate matrices for gpt-oss ``gate_up_proj_blocks``.

    ``blocks``/``scales`` are ``[E, out, groups, 16]`` / ``[E, out, groups]``.
    The original GEMM layout is ``[E, out, groups*32]``; HF stores
    ``[E, in, out]`` (transpose of the last two axes), and gate = even columns
    of the interleaved gate/up axis. Experts are dequantized one at a time so
    the full fp32 gate_up tensor never exists.
    """
    if blocks.ndim != 4 or scales.ndim != 3:
        raise FormatUnreadable(
            f"{base}: expected 4-D blocks / 3-D scales, got {blocks.shape}/{scales.shape}"
        )
    n_experts, rows, groups, _ = blocks.shape
    if rows % 2:
        raise FormatUnreadable(f"{base}: fused gate+up rows not even ({rows})")
    if scales.shape != (n_experts, rows, groups):
        raise FormatUnreadable(f"{base}: scales {scales.shape} do not match blocks {blocks.shape}")
    gate_rows = rows // 2
    in_features = groups * MXFP4_BLOCK
    gates = []
    for expert in range(n_experts):
        packed = dequantize_mxfp4(blocks[expert], scales[expert])  # [out, in]
        if packed.shape != (rows, in_features):
            raise FormatUnreadable(
                f"{base}: dequantized {packed.shape}, expected {(rows, in_features)}"
            )
        # [in, out] == transpose of the GEMM layout; even columns are the gate.
        gate = packed.T[:, ::2]
        if gate.shape != (in_features, gate_rows):
            raise FormatUnreadable(
                f"{base}: gate shape {gate.shape}, expected {(in_features, gate_rows)}"
            )
        gates.append((f"{base}#expert{expert}", gate))
    return gates


#: Deepest ``layers.N`` index in a weight name (nested decoder layouts included).
_LAYER_IN_NAME = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
#: MXFP4 gate+up packed base name (gpt-oss experts), with any prefix.
_MXFP4_GATE_BASE = re.compile(r"\.experts\.gate_up_proj$")


def _mxfp4_gate_layer(base: str) -> int | None:
    """Layer index for an MXFP4 gate base, or None if it is not one.

    Standard ``model.layers.N.mlp.experts.gate_up_proj`` bases classify directly;
    a nested decoder prefix (Qwen3.5-style) still resolves through its deepest
    ``layers.N``, matching the bench copy.
    """
    classified = classify(base)
    if classified is not None:
        return classified[1] if classified[0] == "gptoss_fused" else None
    if not _MXFP4_GATE_BASE.search(base):
        return None
    matches = _LAYER_IN_NAME.findall(base)
    if not matches:
        raise FormatUnreadable(f"{base}: no layers.N index in an MXFP4 gate name")
    return int(matches[-1])


def _mxfp4_pairs(names: Iterable[str]) -> dict[str, dict[str, str]]:
    """Group MXFP4 ``gate_up_proj_{blocks,scales}`` names by gate base name."""
    bases: dict[str, dict[str, str]] = {}
    for name in names:
        for field in ("blocks", "scales"):
            if name.endswith(f"gate_up_proj_{field}"):
                base = name[: -len(f"_{field}")]
                bases.setdefault(base, {})[field] = name
    return {base: parts for base, parts in bases.items() if len(parts) == 2}


# ---------------------------------------------------------------------------
# Minimal safetensors reader (float8/bf16 through ml_dtypes, no torch).
# ---------------------------------------------------------------------------


def _read_header(path: Path) -> tuple[dict, int]:
    """Header dict and absolute data offset of a safetensors shard."""
    with open(path, "rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_len))
    return header, 8 + header_len


@dataclass(frozen=True)
class _ShardSource:
    """One safetensors shard: its header index plus a per-tensor byte reader."""

    index: Mapping[str, dict]
    read: Callable[[str], np.ndarray]


def _local_sources(shards: Iterable[Path]) -> Iterator[_ShardSource]:
    """Sources for shards on disk; headers are parsed as each shard is reached."""
    for shard in shards:
        header, data_start = _read_header(shard)

        def read_tensor(name: str, header=header, shard=shard, data_start=data_start) -> np.ndarray:
            return _read_tensor(shard, header[name], data_start)

        yield _ShardSource(header, read_tensor)


def _tensor_layout(info: Mapping, label: str) -> tuple[object, tuple[int, ...], int]:
    """Validated ``(dtype, shape, element count)`` from one header entry."""
    try:
        dtype = SAFETENSORS_DTYPES[info["dtype"]]
    except KeyError as exc:
        raise FormatUnreadable(f"{label}: unsupported safetensors dtype {info['dtype']}") from exc
    shape = tuple(int(d) for d in info["shape"])
    begin, end = (int(offset) for offset in info["data_offsets"])
    count = int(np.prod(shape, dtype=np.int64))
    if end - begin != count * np.dtype(dtype).itemsize:
        raise FormatUnreadable(f"{label}: byte length mismatch for {info['dtype']} {shape}")
    return dtype, shape, count


def _read_tensor(path: Path, info: Mapping, data_start: int) -> np.ndarray:
    """Read one tensor by header entry; native dtype (float8/bf16 preserved)."""
    dtype, shape, count = _tensor_layout(info, path.name)
    with open(path, "rb") as handle:
        handle.seek(data_start + int(info["data_offsets"][0]))
        array = np.fromfile(handle, dtype=dtype, count=count)
    return array.reshape(shape)


def _gate_names_in(header: Mapping[str, dict]) -> list[str]:
    """Names in a shard header that classify as gate-like tensors."""
    return [name for name in header if name != "__metadata__" and classify(name) is not None]


def _apply_fp8_scale(
    header: Mapping[str, dict],
    name: str,
    array: np.ndarray,
    read_tensor: Callable[[str], np.ndarray],
) -> np.ndarray:
    """Multiply an F8 gate by its compressed-tensors ``weight_scale`` if present.

    Compressed-tensors stores a per-output-channel bf16 scale ``[out, 1]``
    beside each F8 weight; applying it reconstructs the deployment-time weight
    matrix rather than the stored quantization grid (the same principle as the
    AWQ path applying its dequantization scales). A scalar scale broadcasts.
    """
    scale_name = name + "_scale"
    if not header[name]["dtype"].startswith("F8") or scale_name not in header:
        return array
    scale = np.asarray(read_tensor(scale_name), dtype=np.float32).reshape(-1, 1)
    if scale.shape[0] not in (1, array.shape[-2]):
        raise FormatUnreadable(
            f"{scale_name}: {scale.shape[0]} scale rows do not match {name} rows {array.shape[-2]}"
        )
    return array.astype(np.float32) * scale


def _plain_items(source: _ShardSource) -> Iterator[tuple[str, np.ndarray]]:
    """Gate tensors of one source, one at a time, fp8 scales folded in."""
    for name in _gate_names_in(source.index):
        yield name, _apply_fp8_scale(source.index, name, source.read(name), source.read)


def _awq_parts_in(names: Iterable[str]) -> dict[str, dict[str, str]]:
    """Group AWQ ``qweight``/``qzeros``/``scales`` names by their gate base."""
    parts: dict[str, dict[str, str]] = {}
    for name in names:
        base, _, field = name.rpartition(".")
        if field in ("qweight", "qzeros", "scales") and classify(base + ".weight"):
            parts.setdefault(base, {})[field] = name
    return parts


def _awq_packed(
    index: Mapping[str, dict], read: Callable[[str], np.ndarray]
) -> dict[str, dict[str, np.ndarray]]:
    """Read one source's packed AWQ tensors, grouped by their gate base name."""
    return {
        base: {field: read(name) for field, name in parts.items()}
        for base, parts in _awq_parts_in(index).items()
    }


def _awq_items(packed: Mapping[str, dict[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    """Dequantize collected int4-packed gate projections into ``.weight`` matrices."""
    items = []
    for base, parts in packed.items():
        if not {"qweight", "qzeros", "scales"} <= parts.keys():
            raise FormatUnreadable(f"{base}: incomplete AWQ packed tensors {sorted(parts)}")
        qweight, qzeros, scales = (parts[field] for field in ("qweight", "qzeros", "scales"))
        group_size = qweight.shape[0] // qzeros.shape[0]
        # dequantize returns [in, out]; HF .weight is [out, in].
        items.append((base + ".weight", dequantize_awq_gemm(qweight, qzeros, scales, group_size).T))
    return items


def _store_mxfp4(
    layers: dict[int, list[TensorSpectrum]], kinds: set[str], source: _ShardSource
) -> None:
    """Store the per-expert MXFP4 gates found in one source (gpt-oss layout)."""
    for base, parts in _mxfp4_pairs(source.index).items():
        layer = _mxfp4_gate_layer(base)
        if layer is None:
            continue
        gates = _mxfp4_gates(base, source.read(parts["blocks"]), source.read(parts["scales"]))
        kinds.add("gptoss_fused")
        for name, gate in gates:
            _store_matrix(layers, name, gate, "routed", layer)


def _load_plain(
    sources: Iterable[_ShardSource], layers: dict[int, list[TensorSpectrum]], kinds: set[str]
) -> None:
    """Stream gate tensors out of the sources, one tensor at a time."""
    for source in sources:
        _accumulate(layers, kinds, _plain_items(source))


def _load_awq(
    sources: Iterable[_ShardSource], layers: dict[int, list[TensorSpectrum]], kinds: set[str]
) -> None:
    """Dequantize int4-packed gate projections and store them as .weight gates."""
    packed: dict[str, dict[str, np.ndarray]] = {}
    for source in sources:
        for base, parts in _awq_packed(source.index, source.read).items():
            packed.setdefault(base, {}).update(parts)
    _accumulate(layers, kinds, _awq_items(packed))


def _load_mxfp4(
    sources: Iterable[_ShardSource], layers: dict[int, list[TensorSpectrum]], kinds: set[str]
) -> None:
    """Dequantize MXFP4 ``gate_up_proj_blocks`` per expert (gpt-oss reference)."""
    for source in sources:
        _store_mxfp4(layers, kinds, source)


# ---------------------------------------------------------------------------
# HTTP Range reads: fetch only the gate tensors, never a whole shard.
# ---------------------------------------------------------------------------

#: Safetensors header length prefix.
_HEADER_BYTES = 8
#: Plausibility bound for a shard header (guards a non-safetensors response).
_HEADER_MAX_BYTES = 100 * 1024 * 1024
_RANGE_TIMEOUT_SECONDS = 60


class _RangeUnavailable(Exception):
    """A range read could not produce the requested bytes (fall back to snapshot)."""


def _range_headers(byte_range: str) -> dict[str, str]:
    """Range headers (identity encoding) plus the Hub token when one is set."""
    headers = {"Range": byte_range, "Accept-Encoding": "identity"}
    # get_token() also sees a stored login, which HfApi().token can miss.
    token = get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _range_get(repo_id: str, revision: str, filename: str, byte_range: str) -> bytes:
    """One partial-content GET; a non-206 answer means no Range support."""
    response = get_session().get(
        hf_hub_url(repo_id, filename, revision=revision),
        headers=_range_headers(byte_range),
        timeout=_RANGE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    if response.status_code != 206:
        raise _RangeUnavailable(f"{filename}: HTTP {response.status_code} for {byte_range}")
    return response.content


def _remote_data_start(repo_id: str, revision: str, filename: str) -> int:
    """Absolute offset of a shard's data section (8 + header length).

    Only the 8-byte length prefix is fetched; dtype/shape/offsets come from
    ``HfApi().get_safetensors_metadata``, so the header JSON never downloads.
    """
    prefix = _range_get(repo_id, revision, filename, f"bytes=0-{_HEADER_BYTES - 1}")
    if len(prefix) != _HEADER_BYTES:
        raise _RangeUnavailable(f"{filename}: short header prefix ({len(prefix)} bytes)")
    header_len = int.from_bytes(prefix, "little")
    if not 0 < header_len <= _HEADER_MAX_BYTES:
        raise _RangeUnavailable(f"{filename}: implausible header length {header_len}")
    return _HEADER_BYTES + header_len


def _remote_tensor(
    repo_id: str,
    revision: str,
    filename: str,
    data_start: int,
    info: Mapping,
    name: str,
) -> np.ndarray:
    """One tensor's bytes via Range, decoded like ``_read_tensor``."""
    dtype, shape, _ = _tensor_layout(info, filename)
    begin, end = (int(offset) for offset in info["data_offsets"])
    raw = _range_get(
        repo_id, revision, filename, f"bytes={data_start + begin}-{data_start + end - 1}"
    )
    if len(raw) != end - begin:
        raise _RangeUnavailable(f"{filename}: short read for {name} ({len(raw)}/{end - begin})")
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


# ---------------------------------------------------------------------------
# Hub loading layer (smoke-only; the pure core above never touches I/O).
# ---------------------------------------------------------------------------


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _reset_hub_session() -> None:
    """Drop the shared httpx client: huggingface_hub closes it on
    ConnectError but its retry loop keeps the dead reference."""
    try:
        from huggingface_hub.utils import close_session

        close_session()
    except Exception:  # noqa: BLE001 — best-effort reset
        pass


def _retry_call(label: str, fn):
    """Run a Hub call with the independence-tests transient-error policy."""
    last_exc: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        if attempt > 1:
            _reset_hub_session()
            wait = DOWNLOAD_BACKOFF_SECONDS[min(attempt - 2, len(DOWNLOAD_BACKOFF_SECONDS) - 1)]
            _log(f"[download] retry {attempt}/{DOWNLOAD_ATTEMPTS} {label} in {wait:g}s")
            time.sleep(wait)
        try:
            return fn()
        except FormatUnreadable:
            raise  # structural, not transient
        except Exception as exc:  # noqa: BLE001 — classified below
            transient = (
                "Server disconnected" in str(exc)
                or "client has been closed" in str(exc)
                or type(exc).__name__ in ("ConnectError", "ReadTimeout", "RemoteProtocolError")
            )
            if not transient:
                raise
            last_exc = exc
    raise last_exc  # all attempts transient-failed


def _snapshot(repo_id: str, revision: str, cache_dir: Path | None, patterns: list[str]) -> Path:
    """Pinned-revision snapshot restricted to the patterns we need."""
    return Path(
        _retry_call(
            f"snapshot {repo_id}",
            lambda: snapshot_download(
                repo_id, allow_patterns=patterns, cache_dir=cache_dir, revision=revision
            ),
        )
    )


def _download_file(repo_id: str, revision: str, filename: str, cache_dir: Path | None) -> Path:
    return Path(
        _retry_call(
            f"file {filename}",
            lambda: hf_hub_download(
                repo_id, filename=filename, cache_dir=cache_dir, revision=revision
            ),
        )
    )


def load_config(repo_id: str, revision: str, cache_dir: Path | None) -> dict:
    """Pinned config.json (missing config is not fatal — it is metadata)."""
    try:
        path = _download_file(repo_id, revision, "config.json", cache_dir)
    except Exception:  # noqa: BLE001 — absent/inaccessible config is optional
        return {}
    with open(path) as handle:
        return json.load(handle)


def _quant_method(config: Mapping) -> str | None:
    """Declared ``quantization_config.quant_method``, or None when absent."""
    quant_config = config.get("quantization_config")
    return quant_config.get("quant_method") if isinstance(quant_config, dict) else None


def _repo_files(repo_id: str, revision: str) -> list[str]:
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


def _load_gguf(
    repo_id: str, revision: str, cache_dir: Path | None, gguf_files: Iterable[str]
) -> Extraction:
    """GGUF MoE/dense gates; any reader failure is a structural abstain."""
    try:
        return _read_gguf(repo_id, revision, cache_dir, gguf_files)
    except FormatUnreadable:
        raise
    except Exception as exc:  # noqa: BLE001 — untrusted container
        # gguf-py raises ValueError on unknown tensor type codes (e.g. the
        # Prism-fork 34-byte Q2_0 blocks) and TypeError/OSError on malformed
        # headers; one weird file must not abort the whole run.
        raise FormatUnreadable(f"{repo_id}: GGUF unreadable ({type(exc).__name__}: {exc})") from exc


def _read_gguf(
    repo_id: str, revision: str, cache_dir: Path | None, gguf_files: Iterable[str]
) -> Extraction:
    """GGUF MoE/dense gates via GGUFReader + gguf.quants.dequantize."""
    filename = _choose_gguf(gguf_files)
    if filename is None:
        raise FormatUnreadable(f"{repo_id}: no usable .gguf file at this revision")
    path = _download_file(repo_id, revision, filename, cache_dir)
    reader = GGUFReader(str(path))
    layers: dict[int, list[TensorSpectrum]] = {}
    kinds: set[str] = set()
    quant_types: set[str] = set()
    for tensor in reader.tensors:
        if classify(tensor.name) is None:
            continue
        # gguf-py dequantizes into the logical (numpy) layout already; the
        # reader's ``tensor.shape`` is the reversed GGML file order, so
        # reshaping to it would scramble every axis.
        _accumulate(layers, kinds, [(tensor.name, _gguf_gate(tensor))])
        quant_types.add(tensor.tensor_type.name)
    source = {
        "repo_id": repo_id,
        "revision": revision,
        "file": filename,
        "quant_types": sorted(quant_types),
    }
    return _finish(layers, kinds, "gguf", source)


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
    """Training-step checkpoint directory (AlexWortega upcycle layout).

    The weights live at ``step_<cfg.step>/model.safetensors``. Only repos that
    actually carry ``step_N/`` paths are treated as step checkpoints; within
    them the config key wins when it names an existing step, else the highest
    step is taken. This never mixes intermediate training steps.
    """
    steps = {
        int(match.group(1))
        for name in files
        if (match := re.match(r"step_(\d+)/", name)) is not None
    }
    if not steps:
        return None  # flat checkpoint layout; config "step" is not a directory
    config_step = config.get("step")
    return int(config_step) if config_step is not None and int(config_step) in steps else max(steps)


def _safetensors_index(repo_id: str, revision: str, step: int | None) -> dict[str, dict[str, dict]]:
    """Per-shard ``{tensor: {dtype, shape, data_offsets}}`` from Hub metadata.

    Filenames come from ``get_safetensors_metadata`` (one API call), restricted
    to the resolved training-step directory when the checkpoint is one.
    """
    meta = _retry_call(
        f"safetensors metadata {repo_id}",
        lambda: HfApi().get_safetensors_metadata(repo_id, revision=revision),
    )
    files = {
        filename: file_meta
        for filename, file_meta in meta.files_metadata.items()
        if step is None or filename.startswith(f"step_{step}/")
    }
    return {
        filename: {
            name: {
                "dtype": info.dtype,
                "shape": list(info.shape),
                "data_offsets": list(info.data_offsets),
            }
            for name, info in file_meta.tensors.items()
        }
        for filename, file_meta in files.items()
    }


def _load_safetensors_remote(
    repo_id: str, revision: str, quant_method: str | None, step: int | None
) -> Extraction:
    """Range-read only the gate tensors of a safetensors checkpoint.

    Headers come from Hub metadata and bytes from Range GETs, but the result is
    fed to the same ``_ShardSource`` loaders as the snapshot path, so both
    fetches produce the same spectra. Every failure surfaces as an exception
    (``_RangeUnavailable`` for Range misses, ``FormatUnreadable`` for structural
    problems) so the caller can fall back to ``snapshot_download``.
    """
    shards = _safetensors_index(repo_id, revision, step)
    if not shards:
        raise _RangeUnavailable(f"{repo_id}: no safetensors shards at this revision")
    sources = []
    for filename in sorted(shards):
        index = shards[filename]
        data_start = _remote_data_start(repo_id, revision, filename)

        def read_tensor(
            name: str,
            filename: str = filename,
            data_start: int = data_start,
            index: dict = index,
        ) -> np.ndarray:
            return _remote_tensor(repo_id, revision, filename, data_start, index[name], name)

        sources.append(_ShardSource(index, read_tensor))

    layers: dict[int, list[TensorSpectrum]] = {}
    kinds: set[str] = set()
    if quant_method == "awq":
        for source in sources:
            _accumulate(layers, kinds, _awq_items(_awq_packed(source.index, source.read)))
    elif quant_method == "mxfp4":
        _load_mxfp4(sources, layers, kinds)
        if not layers:
            # Declared MXFP4 but stored plain (bf16 re-save with a stale
            # config): read the shards as they are.
            _log(f"[mxfp4] {repo_id}: no gate_up_proj_blocks; reading shards as-is")
            _load_plain(sources, layers, kinds)
    else:
        _load_plain(sources, layers, kinds)
    source = {
        "repo_id": repo_id,
        "revision": revision,
        "files": sorted(shards),
        "quant_method": quant_method,
        "step": step,
        "fetch": "range",
    }
    return _finish(layers, kinds, "safetensors", source)


def _load_safetensors_snapshot(
    repo_id: str,
    revision: str,
    cache_dir: Path | None,
    quant_method: str | None,
    step: int | None,
) -> Extraction:
    """Full-snapshot safetensors reader (plain, AWQ int4, MXFP4, FP8/bf16)."""
    patterns = (
        ["*.safetensors", "*.safetensors.index.json"]
        if step is None
        else [f"step_{step}/*.safetensors"]  # never mix intermediate training steps
    )
    model_dir = _snapshot(repo_id, revision, cache_dir, patterns)
    shards = sorted(model_dir.rglob("*.safetensors"))
    if step is not None:
        shards = [
            shard for shard in shards if shard.relative_to(model_dir).parts[0] == f"step_{step}"
        ]
    if not shards:
        raise FormatUnreadable(f"{repo_id}: no *.safetensors in the pinned snapshot")
    layers: dict[int, list[TensorSpectrum]] = {}
    kinds: set[str] = set()
    if quant_method == "awq":
        _load_awq(_local_sources(shards), layers, kinds)
    else:
        if quant_method == "mxfp4":
            _load_mxfp4(_local_sources(shards), layers, kinds)
            if not layers:
                # Declared MXFP4 but stored plain (bf16 re-save with a stale
                # config): read the shards as they are.
                _log(f"[mxfp4] {repo_id}: no gate_up_proj_blocks; reading shards as-is")
        if not layers:
            _load_plain(_local_sources(shards), layers, kinds)
    source = {
        "repo_id": repo_id,
        "revision": revision,
        "files": [str(shard.relative_to(model_dir)) for shard in shards],
        "quant_method": quant_method,
        "step": step,
        "fetch": "snapshot",
    }
    return _finish(layers, kinds, "safetensors", source)


def _load_safetensors(
    repo_id: str,
    revision: str,
    cache_dir: Path | None,
    quant_method: str | None,
    step: int | None,
    range_read: bool = True,
) -> Extraction:
    """Safetensors checkpoint: range-read the gates when possible, else snapshot."""
    if range_read:
        try:
            return _load_safetensors_remote(repo_id, revision, quant_method, step)
        except FormatUnreadable:
            raise
        except Exception as exc:  # noqa: BLE001 — any range failure means fall back
            _log(
                f"[range] {repo_id}: {type(exc).__name__}: {exc} "
                "— falling back to snapshot_download"
            )
    return _load_safetensors_snapshot(repo_id, revision, cache_dir, quant_method, step)


def _bin_shards(files: Iterable[str], index: Mapping | None) -> list[str]:
    """PyTorch state-dict files (single, sharded, or index-declared)."""
    if index is not None:
        shards = sorted({str(name) for name in index.get("weight_map", {}).values()})
        if shards:
            return shards
    return sorted(
        name
        for name in files
        if name.endswith("pytorch_model.bin") or _BIN_SHARD_RE.fullmatch(Path(name).name)
    )


def _load_bin(
    repo_id: str, revision: str, cache_dir: Path | None, bin_files: Iterable[str]
) -> Extraction:
    """``pytorch_model.bin`` checkpoints via ``torch.load(weights_only=True)``.

    torch is imported lazily: the safetensors and GGUF paths never need it.
    ``weights_only=True`` refuses arbitrary pickle payloads, so this reader is
    safe on untrusted Hub repos. Every shard is loaded on its own so the peak
    is one shard's state dict plus one gate matrix.
    """
    names = list(bin_files)
    index: Mapping | None = None
    index_name = next((name for name in names if name.endswith(_BIN_INDEX)), None)
    if index_name is not None:
        index_path = _download_file(repo_id, revision, index_name, cache_dir)
        with open(index_path) as handle:
            index = json.load(handle)
        names = [name for name in names if name != index_name]
    shards = _bin_shards(names, index)
    if not shards:
        raise FormatUnreadable(f"{repo_id}: no pytorch_model.bin shards at this revision")
    try:
        import torch
    except ImportError as exc:
        raise FormatUnreadable(f"{repo_id}: pytorch_model.bin needs torch") from exc
    layers: dict[int, list[TensorSpectrum]] = {}
    kinds: set[str] = set()
    for shard_name in shards:
        path = _download_file(repo_id, revision, shard_name, cache_dir)
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 — any load failure is structural
            raise FormatUnreadable(
                f"{shard_name}: torch.load(weights_only=True) failed ({exc})"
            ) from exc
        if not isinstance(state, Mapping):
            raise FormatUnreadable(f"{shard_name}: state dict is not a mapping")

        def items(state=state):
            for name, tensor in state.items():
                if classify(name) is None:
                    continue
                yield name, tensor.detach().to(torch.float32).cpu().numpy()

        _accumulate(layers, kinds, items())
        del state
    source = {
        "repo_id": repo_id,
        "revision": revision,
        "files": shards,
        "quant_method": None,
    }
    return _finish(layers, kinds, "pytorch_bin", source)


def load_gate_tensors(
    repo_id: str,
    revision: str,
    cache_dir: Path | None = None,
    range_read: bool = True,
) -> Extraction:
    """Extract every gate-like FFN tensor from one pinned checkpoint."""
    config = load_config(repo_id, revision, cache_dir)
    quant_method = _quant_method(config)
    if quant_method in UNSUPPORTED_QUANT_METHODS:
        raise FormatUnreadable(
            f"{repo_id}: quant_method={quant_method!r} has no dequantizer in this smoke"
        )
    try:
        files = _repo_files(repo_id, revision)
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        # A license-gated or missing checkpoint is a structural abstain, not
        # a scoring failure (e.g. meta-llama/Llama-3.2-1B needs a token).
        raise FormatUnreadable(
            f"{repo_id}: inaccessible at this revision ({type(exc).__name__})"
        ) from exc
    if any(name.endswith(".safetensors") for name in files):
        step = _resolve_step(config, files)
        return _load_safetensors(
            repo_id, revision, cache_dir, quant_method, step, range_read=range_read
        )
    gguf_files = [name for name in files if name.endswith(".gguf")]
    if gguf_files:
        return _load_gguf(repo_id, revision, cache_dir, gguf_files)
    bin_files = [
        name
        for name in files
        if name.endswith("pytorch_model.bin")
        or name.endswith(_BIN_INDEX)
        or _BIN_SHARD_RE.fullmatch(Path(name).name)
    ]
    if bin_files:
        return _load_bin(repo_id, revision, cache_dir, bin_files)
    raise FormatUnreadable(f"{repo_id}: no .safetensors, .gguf or .bin weights at this revision")


def extraction_diagnostics(extraction: Extraction, r: int = R_DEFAULT) -> dict:
    """JSON-ready extraction summary: family, source, counts, per-tensor r_eff."""
    tensors = extraction.tensor_diagnostics(r)
    return {
        "family": extraction.family,
        "format": extraction.format,
        "source": extraction.source,
        "L": len(extraction.layers),
        "n_gate_tensors": extraction.n_tensors,
        "q_values": sorted({entry["q"] for entry in tensors}),
        "tensors": tensors,
    }
