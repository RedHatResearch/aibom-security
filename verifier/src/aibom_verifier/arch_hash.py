"""Canonical architecture fingerprint (FR-4).

Hashes a stable subset of ``config.json`` fields so two models with the same
dense architecture produce the same digest, and any change to a canonicalized
field changes the hash. This is a cheap hard-negative gate before tensor work.

Optional fields that HF configs often omit (while Transformers applies a
default) are normalized before hashing so an explicit default and an absent
key do not spuriously disagree.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Dense-transformer fields used as the architecture fingerprint. Missing keys
# are recorded as null so absences participate in the digest (after
# normalization below).
ARCH_HASH_FIELDS: tuple[str, ...] = (
    "model_type",
    "architectures",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "rope_scaling",
    "tie_word_embeddings",
    "attention_bias",
    "mlp_bias",
    "use_sliding_window",
    "sliding_window",
    "partial_rotary_factor",
)


def _normalize_arch_values(fields: dict[str, Any]) -> dict[str, Any]:
    """Fill Transformers-style defaults so absent vs explicit-default agree."""
    normalized = dict(fields)

    hidden_size = normalized.get("hidden_size")
    num_heads = normalized.get("num_attention_heads")
    can_derive_head_dim = (
        normalized.get("head_dim") is None
        and isinstance(hidden_size, int)
        and isinstance(num_heads, int)
        and num_heads > 0
    )
    if can_derive_head_dim:
        normalized["head_dim"] = hidden_size // num_heads

    if normalized.get("attention_bias") is None:
        normalized["attention_bias"] = False
    if normalized.get("mlp_bias") is None:
        normalized["mlp_bias"] = False
    if normalized.get("use_sliding_window") is None:
        normalized["use_sliding_window"] = False
    if normalized.get("tie_word_embeddings") is None:
        normalized["tie_word_embeddings"] = False
    if normalized.get("partial_rotary_factor") is None:
        normalized["partial_rotary_factor"] = 1.0

    return normalized


def canonicalize_arch_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract and normalize the canonical architecture fields from a config."""
    raw = {field: config.get(field) for field in ARCH_HASH_FIELDS}
    return _normalize_arch_values(raw)


def compute_arch_hash(config: dict[str, Any]) -> str:
    """Return the hex SHA-256 of the canonicalized architecture config."""
    canonical = canonicalize_arch_config(config)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
