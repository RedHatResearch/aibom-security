"""Symmetric 4-bit block quantization for the routing-agreement smoke.

Frozen numeric contract (spec Experiment B):

* symmetric round-to-nearest, no zero-point,
* block size 32 along the *last* dimension of the loaded tensor,
* quantize-dequantize: values are snapped to the grid and back to the model
  dtype, exactly what a weight-only Q4 loader with the same grid would feed the
  kernels.

On the loaded model the expert tensors are fused 3-D parameters
(``...experts.gate_up_proj`` and ``...experts.down_proj``, layout ``[E, out, in]``).
Their last dimension is the *input* dimension, and each expert's row is a whole
number of 32-wide blocks, so the block grid is identical to quantizing the
per-expert 2-D weights the checkpoint stores.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

BITS = 4
BLOCK = 32

# Smallest normal scale; guards an all-zero block from dividing by zero.
_SCALE_EPS = 1e-12


class CheckpointTensorError(ValueError):
    """Unexpected tensor shape/rank while quantizing — treat as malformed input."""


def quantize_dequantize(t: torch.Tensor, bits: int = BITS, block: int = BLOCK) -> torch.Tensor:
    """Snap ``t`` to a symmetric ``bits``-bit grid in 32-wide blocks, then restore.

    Blocks run along the last dimension; a ragged tail is zero-padded for the
    scale computation and cropped afterwards. Math runs in float32 and the
    result is cast back to ``t.dtype`` (float32 for TinyMixtral, bfloat16 for
    OLMoE), so a bf16 model keeps bf16's own representable grid on top of Q4.
    """
    if t.ndim < 2:
        raise CheckpointTensorError(f"expected rank >= 2 tensor, got shape {tuple(t.shape)}")
    if t.shape[-1] == 0 or t.numel() == 0:
        raise CheckpointTensorError(f"empty tensor with shape {tuple(t.shape)}")
    if block < 1 or bits < 2:
        raise CheckpointTensorError(f"invalid quantization contract bits={bits} block={block}")

    out_dtype = t.dtype
    x = t.detach().to(torch.float32)
    last = x.shape[-1]
    pad = (-last) % block
    if pad:
        x = F.pad(x, (0, pad))

    qmax = float(2 ** (bits - 1) - 1)
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // block, block)
    scale = (blocks.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(_SCALE_EPS)
    q = torch.round(blocks / scale).clamp_(-qmax, qmax)
    deq = (q * scale).reshape(*x.shape[:-1], x.shape[-1])
    if pad:
        deq = deq[..., :last]
    return deq.to(out_dtype)


_CHUNK_BYTES = 64 << 20  # fp32 working set per chunk


def quantize_dequantize_(
    t: torch.Tensor, bits: int = BITS, block: int = BLOCK, chunk_bytes: int = _CHUNK_BYTES
) -> None:
    """Quantize-dequantize a live parameter in place.

    Slices along dim 0 so the fp32 working copy stays small; rows are
    independent under last-dim blocking, so the result is bit-identical to
    quantizing the whole tensor at once, and a 14 GB model needs only a few tens
    of MB of scratch.
    """
    if t.ndim < 2:
        raise CheckpointTensorError(f"expected rank >= 2 tensor, got shape {tuple(t.shape)}")
    rows = max(1, t.shape[0])
    row_bytes = max(1, t[0].numel() * 4)
    step = max(1, int(chunk_bytes // row_bytes))
    with torch.no_grad():
        if step >= rows:
            t.copy_(quantize_dequantize(t, bits=bits, block=block))
            return
        for start in range(0, rows, step):
            chunk = t[start : start + step]
            chunk.copy_(quantize_dequantize(chunk, bits=bits, block=block))


def is_expert_param(name: str) -> bool:
    """Expert weight tensors: fused 3-D or per-expert 2-D layouts."""
    return ".experts." in name


def is_router_param(name: str) -> bool:
    """Router gate matrix (first dim = expert index)."""
    return name.endswith(".gate.weight")


def apply_router_permutation_(router: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Overwrite ``router`` rows with ``router[perm]`` and return the pre-shuffle clone."""
    if router.shape[0] != perm.numel():
        raise CheckpointTensorError(
            f"router rows {router.shape[0]} != permutation size {perm.numel()}"
        )
    if perm.min().item() < 0 or perm.max().item() >= perm.numel():
        raise CheckpointTensorError("router permutation out of range")
    with torch.no_grad():
        saved = router.detach().clone()
        router.copy_(router[perm])
    return saved


def permutation_fixed_points(perm: torch.Tensor) -> int:
    """Number of expert slots a router permutation leaves in place."""
    return int((perm == torch.arange(perm.numel(), device=perm.device)).sum().item())


def permutation_invariance(routing_sets: torch.Tensor, perm: torch.Tensor) -> float:
    """Fraction of rows whose top-k *set* maps onto itself under ``perm``.

    Layer-local prediction for ``router_shuffled``: with ``gate' = gate[perm]``
    the logits permute, so the layer's selected ids become ``perm^-1(S)`` while
    the experts stay in place. When that layer's input is unchanged (layer 0,
    whose input is the embedding) the measured agreement with the base routing
    equals this invariance fraction exactly. Deeper layers see drifted inputs,
    so no closed-form prediction applies there.
    """
    idx = routing_sets.to(torch.long)
    mapped = perm.to(idx.device)[idx]
    a = torch.sort(idx, dim=-1).values
    b = torch.sort(mapped, dim=-1).values
    return float((a == b).all(dim=-1).float().mean().item())
