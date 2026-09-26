"""Origin Tracer activation-assisted grey-box smoke.

This is a deliberately small, approximate reimplementation of the method in
arXiv:2505.19466.  It is survey code, not a verifier or a passive weight
fingerprint.  The real path loads both decoder-only models, runs deterministic
single-token probes, injects base layer inputs into candidate layers, and
reconstructs candidate intermediates through the base MLP with autograd.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


METHOD = "origin-tracer-approximation"
PAPER = "https://arxiv.org/abs/2505.19466"
FIXTURES = ROOT / "fixtures.yaml"

# These are intentionally ordinary words rather than a corpus.  The method's
# probe is one token, so tokenizers are checked and words are skipped when
# either tokenizer splits them.
DEFAULT_PROBE_WORDS = (
    "the",
    "a",
    "of",
    "to",
    "and",
    "in",
    "is",
    "for",
    "model",
    "data",
    "security",
    "origin",
    "trace",
    "attention",
    "vector",
    "rank",
    "low",
    "change",
    "base",
    "candidate",
    "activation",
    "layer",
    "hidden",
    "weight",
    "signal",
    "probe",
    "token",
    "sample",
    "reconstruct",
    "difference",
)

# Config-level hints are only used by --preflight.  The loaded-model check
# below also requires an explicit decoder layer/MLP/attention module layout.
SUPPORTED_MODEL_TYPES = {
    "baichuan",
    "gemma",
    "gemma2",
    "gemma3",
    "llama",
    "mistral",
    "mixtral",
    "olmo",
    "olmo2",
    "phi",
    "phi3",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "stablelm",
    "smollm",
    "smollm3",
}

LAYER_PATHS = (
    "model.layers",
    "transformer.h",
    "gpt_neox.layers",
    "model.decoder.layers",
)
MLP_NAMES = ("mlp", "feed_forward", "ffn", "feedforward", "block_sparse_moe")
ATTENTION_NAMES = ("self_attn", "attention", "attn")
INTERMEDIATE_NORM_NAMES = (
    "post_attention_layernorm",
    "post_attention_layer_norm",
    "post_attn_layernorm",
    "ln_2",
)
V_PARAMETER_NAMES = {"v_proj", "value_proj", "value"}
O_PARAMETER_NAMES = {"o_proj", "out_proj", "output_proj", "o"}
ATTENTION_PARAMETER_CONTAINERS = set(ATTENTION_NAMES)
MLP_PARAMETER_CONTAINERS = set(MLP_NAMES) | {"feed_forward", "feedforward"}
ROUTE_TOP_K_NAMES = ("top_k", "num_experts_per_tok", "num_experts_per_token")
ORACLE_RELATIVE_TOLERANCE = 1e-6
ORACLE_ABSOLUTE_TOLERANCE = 1e-8
MATERIAL_RECONSTRUCTION_RATIO = 0.9

DEVIATIONS = (
    "The paper does not fully specify probe vocabulary, cycle count, optimizer, or "
    "stopping criteria; these are explicit CLI settings.",
    "The implementation samples rows (word probes) at "
    "min(probe_count, hidden_size // 2), because the paper's half-hidden-size probe "
    "wording is underspecified.",
    "The implementation uses the decoder layer's post-attention residual as the "
    "intermediate and reconstructs it with the base post-attention norm plus MLP.",
    "The paper's exact rank threshold/calibration is not available; rank is the index "
    "after the largest consecutive log singular-value gap.",
    "Only common Transformers decoder layouts with explicit V/O projections are "
    "supported; this is not a passive/static detector.",
    "MoE tracing is experimental under --allow-moe and assumes unchanged router "
    "and expert functions; missing routes, oracle precision, material progress, or "
    "nonzero rank produces UNSUPPORTED; default config preflight rejects MoE models.",
)


class UnsupportedArchitecture(RuntimeError):
    """The loaded model is outside the supported decoder-layer layouts."""


@dataclass
class DecoderSpec:
    """Modules needed by the activation trace for one loaded model."""

    layer_path: str
    layers: list[Any]
    intermediate_norms: list[Any]
    mlps: list[Any]
    attentions: list[Any]


@dataclass
class ProbeBatch:
    words: list[str]
    base_ids: list[int]
    candidate_ids: list[int]
    skipped: dict[str, str]
    token_mismatches: list[dict[str, Any]]


def _fixture_pairs() -> dict[str, tuple[str, str]]:
    """Read both active and deferred shared fixtures without downloading them."""

    with FIXTURES.open() as handle:
        raw = yaml.safe_load(handle) or {}
    pairs: dict[str, tuple[str, str]] = {}
    for section in ("pairs", "deferred"):
        for label, spec in (raw.get(section) or {}).items():
            if isinstance(spec, dict) and spec.get("ref") and spec.get("sus"):
                pairs[str(label)] = (str(spec["ref"]), str(spec["sus"]))
    return pairs


def _pair_from_args(
    args: argparse.Namespace,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Resolve a fixture or explicit pair; return a human-readable error, not a parser exit."""

    if args.pair:
        pairs = _fixture_pairs()
        if args.pair not in pairs:
            return None, None, None, f"unknown fixture pair {args.pair!r}"
        fixture_base, fixture_candidate = pairs[args.pair]
        return (
            args.pair,
            args.base_model or fixture_base,
            args.candidate_model or fixture_candidate,
            None,
        )

    if bool(args.base_model) != bool(args.candidate_model):
        return (
            None,
            args.base_model,
            args.candidate_model,
            "a same-base comparison requires both --base-model and --candidate-model",
        )
    if not args.base_model:
        return (
            None,
            None,
            None,
            "a same-base comparison requires --base-model and --candidate-model",
        )
    return "custom", args.base_model, args.candidate_model, None


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    value = getattr(config, name, default)
    if value is None:
        return default
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def config_summary(config: Any) -> dict[str, Any]:
    """Keep preflight output small and JSON-safe."""

    return {
        "model_type": _config_value(config, "model_type"),
        "architectures": list(_config_value(config, "architectures", []) or []),
        "hidden_size": _config_value(config, "hidden_size"),
        "num_hidden_layers": _config_value(config, "num_hidden_layers"),
        "num_attention_heads": _config_value(config, "num_attention_heads"),
        "num_key_value_heads": _config_value(
            config,
            "num_key_value_heads",
            _config_value(config, "num_attention_heads"),
        ),
        "intermediate_size": _config_value(config, "intermediate_size"),
        "num_local_experts": _config_value(config, "num_local_experts"),
        "num_experts_per_tok": _config_value(
            config,
            "num_experts_per_tok",
            _config_value(config, "num_experts_per_token"),
        ),
        "vocab_size": _config_value(config, "vocab_size"),
        "max_position_embeddings": _config_value(config, "max_position_embeddings"),
        "base_model_name_or_path": _config_value(config, "base_model_name_or_path"),
    }


def _is_moe_config(config: Any) -> bool:
    model_type = str(_config_value(config, "model_type", "")).lower()
    architectures = " ".join(
        str(value).lower() for value in (_config_value(config, "architectures", []) or [])
    )
    expert_fields = (
        "num_local_experts",
        "num_experts",
        "num_experts_per_tok",
        "num_experts_per_token",
    )
    exposes_expert_config = any(_config_value(config, field) is not None for field in expert_fields)
    return (
        "moe" in model_type
        or "mixtral" in model_type
        or "moe" in architectures
        or exposes_expert_config
    )


def _architecture_hint(config: Any) -> bool:
    model_type = str(_config_value(config, "model_type", "")).lower()
    if model_type in SUPPORTED_MODEL_TYPES:
        return True
    architectures = " ".join(
        str(x).lower() for x in (_config_value(config, "architectures", []) or [])
    )
    return any(
        marker in architectures
        for marker in (
            "llama",
            "mistral",
            "mixtral",
            "gemma",
            "qwen",
            "olmo",
            "phi",
            "neox",
            "stablelm",
        )
    )


def _config_shape_mismatches(base_config: Any, candidate_config: Any) -> list[str]:
    mismatches: list[str] = []
    fields = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
    )
    for field in fields:
        base_value = _config_value(base_config, field)
        candidate_value = _config_value(candidate_config, field)
        if base_value is not None and candidate_value is not None and base_value != candidate_value:
            mismatches.append(f"{field} mismatch {base_value} vs {candidate_value}")
    return mismatches


def _hf_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir.expanduser())
    if args.revision:
        kwargs["revision"] = args.revision
    if args.local_files_only:
        kwargs["local_files_only"] = True
    return kwargs


def _preflight_configs(
    base_model: str,
    candidate_model: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Load only config.json and report shape/architecture applicability."""

    from transformers import AutoConfig

    base_config = AutoConfig.from_pretrained(base_model, **_hf_kwargs(args))
    candidate_config = AutoConfig.from_pretrained(candidate_model, **_hf_kwargs(args))
    base_summary = config_summary(base_config)
    candidate_summary = config_summary(candidate_config)
    base_is_moe = _is_moe_config(base_config)
    candidate_is_moe = _is_moe_config(candidate_config)
    shape_mismatches = _config_shape_mismatches(base_config, candidate_config)

    if (base_is_moe or candidate_is_moe) and not args.allow_moe:
        return {
            "status": "UNSUPPORTED",
            "reason": (
                "MoE architectures are unsupported: Origin Tracer assumes one "
                "unchanged dense MLP path and does not trace routers or experts"
            ),
            "base_config": base_summary,
            "candidate_config": candidate_summary,
            "base_is_moe": base_is_moe,
            "candidate_is_moe": candidate_is_moe,
            "shape_mismatches": shape_mismatches,
        }
    if shape_mismatches:
        return {
            "status": "INCOMPATIBLE",
            "reason": "cross-shape model pair: " + "; ".join(shape_mismatches),
            "base_config": base_summary,
            "candidate_config": candidate_summary,
            "shape_mismatches": shape_mismatches,
            "base_is_moe": base_is_moe,
            "candidate_is_moe": candidate_is_moe,
        }
    if not _architecture_hint(base_config) or not _architecture_hint(candidate_config):
        return {
            "status": "UNSUPPORTED",
            "reason": (
                "unsupported decoder-only architecture; expected a common "
                "Transformers decoder layout"
            ),
            "base_config": base_summary,
            "candidate_config": candidate_summary,
            "base_is_moe": base_is_moe,
            "candidate_is_moe": candidate_is_moe,
        }
    base_metadata = _config_value(candidate_config, "base_model_name_or_path")
    if args.require_base_metadata and not base_metadata:
        return {
            "status": "UNSUPPORTED",
            "reason": (
                "same-base metadata is missing; rerun without "
                "--require-base-metadata only when the pair is known to share a base"
            ),
            "base_config": base_summary,
            "base_is_moe": base_is_moe,
            "candidate_is_moe": candidate_is_moe,
            "candidate_config": candidate_summary,
        }
    return {
        "status": "PREFLIGHT_OK",
        "reason": (
            "config shapes and architecture hints are compatible; parameter and "
            "module checks require model weights"
        ),
        "base_is_moe": base_is_moe,
        "candidate_is_moe": candidate_is_moe,
        "base_config": base_summary,
        "candidate_config": candidate_summary,
        "same_base_metadata": base_metadata,
    }


def _get_path(obj: Any, path: str) -> Any:
    current = obj
    for name in path.split("."):
        current = getattr(current, name)
    return current


def _child_module(obj: Any, names: Iterable[str]) -> Any | None:
    for name in names:
        module = getattr(obj, name, None)
        if module is not None and callable(module):
            return module
    return None


def _find_decoder_layers(model: Any) -> tuple[str, list[Any]]:
    for path in LAYER_PATHS:
        try:
            layers = _get_path(model, path)
        except AttributeError:
            continue
        try:
            if len(layers) == 0:
                continue
            first = layers[0]
        except (TypeError, IndexError):
            continue
        if callable(first):
            return path, list(layers)
    raise UnsupportedArchitecture(
        "unsupported decoder-only architecture: no model.layers, transformer.h, "
        "gpt_neox.layers, or model.decoder.layers ModuleList"
    )


def inspect_decoder_spec(model: Any) -> DecoderSpec:
    """Find the exact modules needed for the base-MLP reconstruction."""

    layer_path, layers = _find_decoder_layers(model)
    intermediate_norms: list[Any] = []
    mlps: list[Any] = []
    attentions: list[Any] = []
    for index, layer in enumerate(layers):
        norm = _child_module(layer, INTERMEDIATE_NORM_NAMES)
        mlp = _child_module(layer, MLP_NAMES)
        attention = _child_module(layer, ATTENTION_NAMES)
        if norm is None or mlp is None or attention is None:
            raise UnsupportedArchitecture(
                f"layer {index} lacks an explicit post-attention norm, MLP, and attention module"
            )
        attention_names = {name.split(".")[0] for name, _ in attention.named_parameters()}
        has_v = bool(attention_names & V_PARAMETER_NAMES)
        has_o = bool(attention_names & O_PARAMETER_NAMES)
        if not has_v or not has_o:
            raise UnsupportedArchitecture(
                f"layer {index} does not expose separate attention V/O projections; "
                "fused-QKV or non-attention layouts are unsupported"
            )
        intermediate_norms.append(norm)
        mlps.append(mlp)
        attentions.append(attention)
    return DecoderSpec(
        layer_path=layer_path,
        layers=layers,
        intermediate_norms=intermediate_norms,
        mlps=mlps,
        attentions=attentions,
    )


def _layer_parameter_family(name: str) -> str:
    """Classify a changed state-dict key under the stated LoRA assumptions."""

    normalized = name.lower().replace("/", ".")
    parts = [part for part in normalized.split(".") if part]
    if any(part in MLP_PARAMETER_CONTAINERS for part in parts):
        return "mlp"
    attention_index = next(
        (index for index, part in enumerate(parts) if part in ATTENTION_PARAMETER_CONTAINERS),
        None,
    )
    if attention_index is None:
        return "other"
    attention_parts = set(parts[attention_index + 1 :])
    if attention_parts & V_PARAMETER_NAMES:
        return "attention_v"
    if attention_parts & O_PARAMETER_NAMES:
        return "attention_o"
    return "attention_other"


def parameter_change_report(base_model: Any, candidate_model: Any) -> dict[str, Any]:
    """Compare loaded weights without building a second tensor copy."""
    import torch

    base_state = base_model.state_dict()
    candidate_state = candidate_model.state_dict()
    base_keys = set(base_state)
    candidate_keys = set(candidate_state)
    missing = sorted(base_keys - candidate_keys)
    extra = sorted(candidate_keys - base_keys)
    shape_mismatches: list[str] = []
    changed: list[str] = []
    for name in sorted(base_keys & candidate_keys):
        base_tensor = base_state[name]
        candidate_tensor = candidate_state[name]
        if tuple(base_tensor.shape) != tuple(candidate_tensor.shape):
            shape_mismatches.append(
                f"{name}: {tuple(base_tensor.shape)} vs {tuple(candidate_tensor.shape)}"
            )
            continue
        if not bool(torch.equal(base_tensor, candidate_tensor)):
            changed.append(name)

    families = Counter(_layer_parameter_family(name) for name in changed)
    return {
        "missing_keys": missing[:16],
        "extra_keys": extra[:16],
        "missing_key_count": len(missing),
        "extra_key_count": len(extra),
        "shape_mismatches": shape_mismatches[:16],
        "shape_mismatch_count": len(shape_mismatches),
        "changed_parameter_count": len(changed),
        "changed_parameters": changed[:32],
        "changed_parameter_truncated": len(changed) > 32,
        "changed_families": dict(sorted(families.items())),
        "same_base_evidence": (
            "only attention V/O parameters differ"
            if changed and set(families) <= {"attention_v", "attention_o"}
            else "not established"
        ),
    }


def _parameter_compatibility(report: dict[str, Any]) -> tuple[str, str]:
    if report["missing_key_count"] or report["extra_key_count"] or report["shape_mismatch_count"]:
        return "INCOMPATIBLE", "state-dict keys or tensor shapes differ"
    families = report["changed_families"]
    if families.get("mlp", 0):
        return (
            "UNSUPPORTED",
            "MLP-targeted changes detected; Origin Tracer assumes unchanged MLP functions",
        )
    unsupported = set(families) - {"attention_v", "attention_o"}
    if unsupported:
        return (
            "UNSUPPORTED",
            "changes outside attention V/O detected; expected a V/O-only low-rank LoRA candidate",
        )
    if not families:
        return (
            "UNSUPPORTED",
            "no changed attention V/O parameters detected; same-base candidate "
            "applicability is unestablished",
        )
    return "OK", "same-base evidence: only attention V/O parameters differ"


def _tokenize_one(tokenizer: Any, word: str) -> list[int]:
    values = tokenizer.encode(word, add_special_tokens=False)
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]


def prepare_probes(
    base_tokenizer: Any,
    candidate_tokenizer: Any,
    words: Iterable[str],
    count: int,
) -> ProbeBatch:
    selected_words: list[str] = []
    base_ids: list[int] = []
    candidate_ids: list[int] = []
    skipped: dict[str, str] = {}
    token_mismatches: list[dict[str, Any]] = []
    for raw_word in words:
        word = raw_word.strip()
        if not word or word in selected_words:
            continue
        try:
            base_tokens = _tokenize_one(base_tokenizer, word)
            candidate_tokens = _tokenize_one(candidate_tokenizer, word)
        except Exception as exc:  # tokenizer-specific errors are applicability diagnostics
            skipped[word] = f"tokenization error: {type(exc).__name__}"
            continue
        if len(base_tokens) != 1 or len(candidate_tokens) != 1:
            skipped[word] = (
                f"not one token in both tokenizers ({len(base_tokens)} vs {len(candidate_tokens)})"
            )
            continue
        if base_tokens[0] != candidate_tokens[0]:
            token_mismatches.append(
                {
                    "word": word,
                    "base_token_id": base_tokens[0],
                    "candidate_token_id": candidate_tokens[0],
                }
            )
            continue
        selected_words.append(word)
        base_ids.append(base_tokens[0])
        candidate_ids.append(candidate_tokens[0])
        if len(selected_words) >= count:
            break
    return ProbeBatch(
        words=selected_words,
        base_ids=base_ids,
        candidate_ids=candidate_ids,
        skipped=skipped,
        token_mismatches=token_mismatches,
    )


def _first_tensor(value: Any) -> Any | None:
    """Extract the hidden-state tensor from common Transformers tuple outputs."""

    # Import lazily so --dry-run and --help do not require torch initialization.
    import torch

    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _is_routed_mlp(module: Any) -> bool:
    """Recognize sparse MLPs without depending on a Transformers class name."""

    return callable(getattr(module, "gate", None)) and (
        getattr(module, "experts", None) is not None
        or getattr(module, "num_experts", None) is not None
        or getattr(module, "num_local_experts", None) is not None
    )


def _module_metadata(module: Any) -> dict[str, Any]:
    module_type = type(module)
    metadata: dict[str, Any] = {
        "class": f"{module_type.__module__}.{module_type.__qualname__}",
        "has_gate": callable(getattr(module, "gate", None)),
        "has_experts": getattr(module, "experts", None) is not None,
        "is_routed_mlp": _is_routed_mlp(module),
    }
    for name in ("num_experts", "num_local_experts", "num_experts_per_tok", "top_k"):
        value = getattr(module, name, None)
        if value is not None:
            try:
                metadata[name] = int(value)
            except (TypeError, ValueError):
                metadata[name] = str(value)
    experts = getattr(module, "experts", None)
    if experts is not None:
        try:
            metadata["expert_count"] = len(experts)
        except TypeError:
            metadata["expert_count"] = None

    return metadata


def _router_top_k(gate: Any, mlp: Any) -> int | None:
    for module in (gate, mlp):
        for name in ROUTE_TOP_K_NAMES:
            value = getattr(module, name, None)
            if value is None:
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
    return None


def _route_tensors(value: Any) -> list[Any]:
    import torch

    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        tensors: list[Any] = []
        for item in value:
            tensors.extend(_route_tensors(item))
        return tensors
    if isinstance(value, dict):
        tensors = []
        for item in value.values():
            tensors.extend(_route_tensors(item))
        return tensors
    return []


def _route_rows(value: Any, batch_size: int) -> Any | None:
    import torch

    if not torch.is_tensor(value) or value.ndim < 2 or value.shape[-1] < 1:
        return None
    rows = value.detach().reshape(-1, value.shape[-1])
    if rows.shape[0] < batch_size:
        return None
    return rows[:batch_size]


def _unavailable_routes(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "top_k": None,
        "top_k_ids": [],
        "top_k_weights": [],
        "route_margin": [],
        "route_margin_kind": None,
    }


def _parse_router_output(output: Any, gate: Any, mlp: Any, batch_size: int) -> dict[str, Any]:
    """Parse Mixtral-style (logits, scores, indices) and logits-only gates."""

    import torch

    tensors = _route_tensors(output)
    if not tensors:
        return _unavailable_routes("router hook returned no tensors")

    integer_tensors = [
        tensor
        for tensor in tensors
        if not torch.is_floating_point(tensor) and not tensor.is_complex()
    ]
    index_tensor = None
    for tensor in integer_tensors:
        rows = _route_rows(tensor, batch_size)
        if rows is not None:
            index_tensor = rows
            break

    floating_tensors = [
        tensor for tensor in tensors if torch.is_floating_point(tensor) or tensor.is_complex()
    ]
    floating_rows = [
        (tensor, rows)
        for tensor in floating_tensors
        if (rows := _route_rows(tensor, batch_size)) is not None
    ]
    if not floating_rows and index_tensor is None:
        return _unavailable_routes("router hook returned no usable route tensors")

    logits_tensor = None
    logits_rows = None
    if floating_rows:
        candidate_logits_tensor, candidate_logits_rows = max(
            floating_rows, key=lambda item: item[1].shape[-1]
        )
        if index_tensor is None or candidate_logits_rows.shape[-1] > index_tensor.shape[-1]:
            logits_tensor = candidate_logits_tensor
            logits_rows = candidate_logits_rows

    top_k = int(index_tensor.shape[-1]) if index_tensor is not None else _router_top_k(gate, mlp)
    if top_k is None or top_k < 1:
        return _unavailable_routes(
            "router top-k is not exposed; cannot interpret logits-only gate output"
        )
    if index_tensor is None:
        if logits_rows is None:
            return _unavailable_routes("router indices are unavailable")
        top_k = min(top_k, int(logits_rows.shape[-1]))
        _, index_tensor = torch.topk(logits_rows, k=top_k, dim=-1)
    else:
        top_k = min(top_k, int(index_tensor.shape[-1]))
        index_tensor = index_tensor[:, :top_k]

    score_rows = None
    for tensor, rows in sorted(floating_rows, key=lambda item: item[1].shape[-1]):
        if rows.shape[-1] == top_k and tensor is not logits_tensor:
            score_rows = rows
            break
    if score_rows is None and logits_rows is not None:
        probabilities = torch.softmax(logits_rows, dim=-1)
        score_rows = probabilities.gather(-1, index_tensor.to(dtype=torch.long))
    if score_rows is None:
        return _unavailable_routes("router weights are unavailable")
    score_rows = score_rows[:, :top_k]

    route_margin = None
    route_margin_kind = None
    if logits_rows is not None and logits_rows.shape[-1] > top_k:
        sorted_logits = torch.sort(logits_rows, dim=-1, descending=True).values
        route_margin = sorted_logits[:, top_k - 1] - sorted_logits[:, top_k]
        route_margin_kind = "selected_vs_first_unselected_logit"
    elif score_rows.shape[-1] > 1:
        route_margin = score_rows[:, 0] - score_rows[:, 1]
        route_margin_kind = "top_weight_gap"

    return {
        "available": True,
        "reason": None,
        "top_k": int(top_k),
        "top_k_ids": [
            [int(item) for item in row]
            for row in index_tensor.detach().to(device="cpu", dtype=torch.long).tolist()
        ],
        "top_k_weights": [
            [float(item) for item in row]
            for row in score_rows.detach().to(device="cpu", dtype=torch.float32).tolist()
        ],
        "route_margin": (
            [
                float(item)
                for item in route_margin.detach().to(device="cpu", dtype=torch.float32).tolist()
            ]
            if route_margin is not None
            else []
        ),
        "route_margin_kind": route_margin_kind,
        "router_logits_available": logits_rows is not None,
    }


def _route_switch_diagnostics(
    base_routes: dict[str, Any], candidate_routes: dict[str, Any]
) -> dict[str, Any]:
    if not base_routes.get("available") or not candidate_routes.get("available"):
        return {
            "available": False,
            "route_switch_fraction": None,
            "reason": "base or candidate route diagnostics unavailable",
        }
    if base_routes.get("top_k") != candidate_routes.get("top_k"):
        return {
            "available": False,
            "route_switch_fraction": None,
            "reason": "base and candidate expose different top-k widths",
        }
    base_ids = base_routes.get("top_k_ids", [])
    candidate_ids = candidate_routes.get("top_k_ids", [])
    count = min(len(base_ids), len(candidate_ids))
    if count == 0:
        return {
            "available": False,
            "route_switch_fraction": None,
            "reason": "base or candidate route diagnostics contain no tokens",
        }
    switches = sum(
        1
        for base_row, candidate_row in zip(base_ids[:count], candidate_ids[:count], strict=True)
        if sorted(base_row) != sorted(candidate_row)
    )
    return {
        "available": True,
        "route_switch_fraction": float(switches / count),
        "tokens_compared": int(count),
        "base_top_k": base_routes.get("top_k"),
        "candidate_top_k": candidate_routes.get("top_k"),
    }


def _router_capture_hook(
    captures: dict[str, Any],
    gate: Any,
    mlp: Any,
    batch_size: int,
) -> Any:
    def capture(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
        try:
            captures["routes"] = _parse_router_output(output, gate, mlp, batch_size)
        except Exception as exc:
            captures["routes"] = _unavailable_routes(
                f"router output parsing failed: {type(exc).__name__}: {exc}"
            )

    return capture


def _direct_delta_diagnostics(delta: Any) -> dict[str, Any]:
    import torch

    matrix = delta.detach().reshape(delta.shape[0], -1).to(dtype=torch.float32, device="cpu")
    values = np.linalg.svd(matrix.numpy(), compute_uv=False)
    leading = float(values[0]) if len(values) else 0.0
    threshold = max(1e-12, leading * 1e-6)
    return {
        "available": True,
        "shape": [int(value) for value in matrix.shape],
        "l2_norm": float(torch.linalg.vector_norm(matrix).item()),
        "frobenius_norm": float(torch.linalg.vector_norm(matrix).item()),
        "mean_squared_delta": float(torch.mean(matrix**2).item()),
        "singular_values": [float(value) for value in values[:16]],
        "effective_rank": int(np.count_nonzero(values > threshold)),
        "effective_rank_threshold": float(threshold),
    }


def _oracle_diagnostics(
    norm: Any, mlp: Any, post_attention_residual: Any, target_output: Any
) -> dict[str, Any]:
    import torch

    with torch.no_grad():
        mlp_value = _first_tensor(mlp(norm(post_attention_residual)))
        if mlp_value is None:
            return {
                "available": False,
                "reason": "base MLP did not return a tensor for oracle evaluation",
            }
        oracle_output = post_attention_residual + mlp_value
        mse = torch.mean((oracle_output - target_output.detach()) ** 2)
        target_scale = torch.mean(target_output.detach() ** 2)
        mse_value = float(mse.detach().cpu().item())
        scale_value = float(target_scale.detach().cpu().item())
    threshold = max(ORACLE_ABSOLUTE_TOLERANCE, scale_value * ORACLE_RELATIVE_TOLERANCE)
    return {
        "available": True,
        "mse": mse_value,
        "target_mean_squared": scale_value,
        "relative_mse": float(mse_value / max(scale_value, 1e-30)),
        "near_numerical_precision": bool(mse_value <= threshold),
        "tolerance": float(threshold),
    }


def _reconstruction_progress(reconstruction: dict[str, float]) -> dict[str, Any]:
    initial = float(reconstruction["initial_mse"])
    final = float(reconstruction["final_mse"])
    if not np.isfinite(initial) or not np.isfinite(final):
        return {"status": "unavailable", "ratio": None, "material": False}
    if initial <= 0.0:
        return {
            "status": "no_error_to_reduce",
            "ratio": None,
            "material": False,
        }
    ratio = final / initial
    if not np.isfinite(ratio):
        return {"status": "unavailable", "ratio": None, "material": False}
    if ratio <= MATERIAL_RECONSTRUCTION_RATIO:
        status = "material"
        material = True
    elif ratio < 1.0 - 1e-6:
        status = "improved"
        material = False
    else:
        status = "no_progress"
        material = False
    return {"status": status, "ratio": float(ratio), "material": material}


def _hidden_from_hook(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any | None:
    import torch

    if args and torch.is_tensor(args[0]):
        return args[0]
    for key in ("hidden_states", "hidden_state", "x", "inputs"):  # common layer signatures
        value = kwargs.get(key)
        if torch.is_tensor(value):
            return value
    return None


def _base_layer_states(
    model: Any,
    spec: DecoderSpec,
    input_ids: Any,
    attention_mask: Any,
    selected_layers: list[int],
    *,
    capture_routes: bool = False,
) -> tuple[dict[int, Any], dict[int, Any], dict[int, dict[str, Any]]]:
    """Run base once and capture layer inputs, residuals, and optional routes."""

    captures: dict[int, Any] = {}
    route_captures: dict[int, dict[str, Any]] = {}
    handles: list[Any] = []
    batch_size = int(input_ids.shape[0])
    for index in selected_layers:
        module = spec.intermediate_norms[index]

        def capture_hook(
            _module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            *,
            layer_index: int = index,
        ) -> None:
            hidden = _hidden_from_hook(args, kwargs)
            if hidden is not None:
                captures[layer_index] = hidden.detach()

        handles.append(module.register_forward_pre_hook(capture_hook, with_kwargs=True))
        if capture_routes:
            mlp = spec.mlps[index]
            gate = getattr(mlp, "gate", None)
            if callable(gate):
                route_state: dict[str, Any] = {}
                route_captures[index] = route_state
                handles.append(
                    gate.register_forward_hook(
                        _router_capture_hook(route_state, gate, mlp, batch_size)
                    )
                )
            else:
                route_captures[index] = _unavailable_routes(
                    "selected MLP does not expose a callable gate module"
                )
    try:
        with __import__("torch").no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Transformers model did not return hidden states")
    missing = [index for index in selected_layers if index not in captures]
    if missing:
        raise RuntimeError(f"could not capture post-attention residual for layer(s) {missing}")
    if capture_routes:
        for index in selected_layers:
            route_captures.setdefault(
                index,
                _unavailable_routes("router hook did not produce route diagnostics"),
            )
            if "routes" in route_captures[index]:
                route_captures[index] = route_captures[index]["routes"]
            elif "available" not in route_captures[index]:
                route_captures[index] = _unavailable_routes(
                    "router hook did not produce route diagnostics"
                )
    inputs = {index: hidden_states[index].detach() for index in selected_layers}
    return inputs, captures, route_captures


def _candidate_layer_output(
    model: Any,
    spec: DecoderSpec,
    layer_index: int,
    injected_input: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    capture_routes: bool = False,
) -> tuple[Any, Any | None, dict[str, Any]]:
    """Run candidate, inject base input, and capture output/residual/routes."""

    import torch

    captured: dict[str, Any] = {}
    route_captures: dict[str, Any] = {}
    layer = spec.layers[layer_index]

    def inject_hook(
        _module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if args:
            replaced = (injected_input, *args[1:])
            return replaced, kwargs
        replaced_kwargs = dict(kwargs)
        if "hidden_states" in replaced_kwargs:
            replaced_kwargs["hidden_states"] = injected_input
        elif "hidden_state" in replaced_kwargs:
            replaced_kwargs["hidden_state"] = injected_input
        else:
            raise RuntimeError("decoder layer has no recognizable hidden-state argument")
        return args, replaced_kwargs

    def residual_hook(
        _module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        hidden = _hidden_from_hook(args, kwargs)
        if hidden is not None:
            captured["post_attention_residual"] = hidden.detach()

    def output_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
        hidden = _first_tensor(output)
        if hidden is not None:
            captured["hidden"] = hidden.detach()

    input_handle = layer.register_forward_pre_hook(inject_hook, with_kwargs=True)
    residual_handle = spec.intermediate_norms[layer_index].register_forward_pre_hook(
        residual_hook, with_kwargs=True
    )
    output_handle = layer.register_forward_hook(output_hook)
    route_handle = None
    if capture_routes:
        mlp = spec.mlps[layer_index]
        gate = getattr(mlp, "gate", None)
        if callable(gate):
            route_handle = gate.register_forward_hook(
                _router_capture_hook(
                    route_captures,
                    gate,
                    mlp,
                    int(input_ids.shape[0]),
                )
            )
        else:
            route_captures["routes"] = _unavailable_routes(
                "selected MLP does not expose a callable gate module"
            )
    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
    finally:
        input_handle.remove()
        residual_handle.remove()
        output_handle.remove()
        if route_handle is not None:
            route_handle.remove()
    if "hidden" not in captured:
        raise RuntimeError(f"could not capture candidate output for layer {layer_index}")
    if "routes" not in route_captures:
        route_captures["routes"] = _unavailable_routes(
            "router hook did not produce route diagnostics"
        )
    return (
        captured["hidden"],
        captured.get("post_attention_residual"),
        route_captures["routes"],
    )


def _reconstruct_intermediate(
    norm: Any,
    mlp: Any,
    initial: Any,
    target_output: Any,
    steps: int,
    learning_rate: float,
) -> tuple[Any, dict[str, float]]:
    """Optimize z so z + base_mlp(base_norm(z)) matches candidate output."""

    import torch

    z = initial.detach().clone().requires_grad_(True)
    target = target_output.detach()
    optimizer = torch.optim.Adam([z], lr=learning_rate)

    def forward_from_intermediate(value: Any) -> Any:
        mlp_value = _first_tensor(mlp(norm(value)))
        if mlp_value is None:
            raise RuntimeError("base MLP did not return a tensor")
        return value + mlp_value

    with torch.enable_grad():
        initial_prediction = forward_from_intermediate(z)
        initial_loss = torch.mean((initial_prediction - target) ** 2)
        best_z = z.detach().clone()
        best_loss = initial_loss.detach()
        best_step = 0
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_from_intermediate(z)
            loss = torch.mean((prediction - target) ** 2)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite reconstruction loss")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                current_prediction = forward_from_intermediate(z)
                current_loss = torch.mean((current_prediction - target) ** 2)
            if bool(torch.isfinite(current_loss)) and bool(current_loss < best_loss):
                best_z = z.detach().clone()
                best_loss = current_loss.detach()
                best_step = step + 1
        final_loss = best_loss
    return best_z, {
        "initial_mse": float(initial_loss.detach().cpu().item()),
        "final_mse": float(final_loss.detach().cpu().item()),
        "best_step": float(best_step),
        "steps": float(steps),
        "learning_rate": float(learning_rate),
    }


def _rank_from_matrix(matrix: np.ndarray) -> tuple[int, float, list[float]]:
    if matrix.ndim != 2:
        raise ValueError(f"rank matrix must be 2-D, got {matrix.shape}")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if len(singular_values) == 0 or float(np.max(np.abs(singular_values))) <= 1e-12:
        return 0, 0.0, []
    if len(singular_values) == 1:
        return 1, float("inf"), [float(singular_values[0])]
    scale = max(float(singular_values[0]), 1.0)
    epsilon = scale * 1e-12
    safe = np.maximum(singular_values, epsilon)
    gaps = np.log(safe[:-1] / safe[1:])
    split = int(np.argmax(gaps))
    return split + 1, float(gaps[split]), [float(value) for value in singular_values[:16]]


def sampled_rank_diagnostics(
    matrix: np.ndarray,
    cycles: int,
    hidden_size: int,
    seed: int,
) -> dict[str, Any]:
    """Sample deterministic probe rows and return the minimum cycle rank."""

    probe_count = matrix.shape[0]
    sample_size = min(probe_count, max(1, hidden_size // 2))
    rng = np.random.default_rng(seed)
    cycle_results: list[dict[str, Any]] = []
    ranks: list[int] = []
    for cycle in range(cycles):
        if sample_size < probe_count:
            indices = np.sort(rng.choice(probe_count, size=sample_size, replace=False))
        else:
            indices = np.arange(probe_count)
        rank, gap, singular_values = _rank_from_matrix(matrix[indices])
        ranks.append(rank)
        cycle_results.append(
            {
                "cycle": cycle,
                "sample_size": int(sample_size),
                "probe_indices": [int(index) for index in indices],
                "rank": int(rank),
                "largest_log_gap": gap,
                "singular_values": singular_values,
            }
        )
    return {
        "sample_size": int(sample_size),
        "cycles": cycle_results,
        "minimum_rank": int(min(ranks)) if ranks else None,
    }


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if device.type == "mps" and (
        getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("--device mps requested but MPS is unavailable")
    return device


def _dtype_for(torch: Any, value: str) -> Any | None:
    if value == "auto":
        return None
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def _probe_words(args: argparse.Namespace) -> list[str]:
    if args.probe_words:
        return [word.strip() for word in args.probe_words.split(",") if word.strip()]
    return list(DEFAULT_PROBE_WORDS)


def _selected_layers(spec: DecoderSpec, value: str) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(len(spec.layers)))
    selected: list[int] = []
    for token in value.split(","):
        index = int(token.strip())
        if index < 0:
            index += len(spec.layers)
        if index < 0 or index >= len(spec.layers):
            raise ValueError(f"layer index {token!r} is outside 0..{len(spec.layers) - 1}")
        if index not in selected:
            selected.append(index)
    if not selected:
        raise ValueError("--layers selected no layers")
    return selected


def _load_models_and_tokenizers(
    base_model_id: str,
    candidate_model_id: str,
    args: argparse.Namespace,
) -> tuple[Any, Any, Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _resolve_device(torch, args.device)
    model_kwargs = _hf_kwargs(args)
    model_dtype = _dtype_for(torch, args.dtype)
    if model_dtype is not None:
        model_kwargs["torch_dtype"] = model_dtype
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
    candidate_model = AutoModelForCausalLM.from_pretrained(candidate_model_id, **model_kwargs)
    base_model.to(device)
    candidate_model.to(device)
    base_model.eval()
    candidate_model.eval()
    for model in (base_model, candidate_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_id, **_hf_kwargs(args))
    candidate_tokenizer = AutoTokenizer.from_pretrained(candidate_model_id, **_hf_kwargs(args))
    return base_model, candidate_model, base_tokenizer, candidate_tokenizer, device


def _runtime_metadata() -> dict[str, Any]:
    import torch
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", None),
        "transformers": getattr(transformers, "__version__", None),
    }


def run_analysis(
    label: str,
    base_model_id: str,
    candidate_model_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    preflight = _preflight_configs(base_model_id, candidate_model_id, args)
    if preflight["status"] != "PREFLIGHT_OK":
        return {
            "method": METHOD,
            "paper": PAPER,
            "status": preflight["status"],
            "label": label,
            "base_model": base_model_id,
            "candidate_model": candidate_model_id,
            "reason": preflight["reason"],
            "preflight": preflight,
            "deviations": list(DEVIATIONS),
            "elapsed_seconds": time.perf_counter() - started,
        }

    import torch

    base_model, candidate_model, base_tokenizer, candidate_tokenizer, device = (
        _load_models_and_tokenizers(
            base_model_id,
            candidate_model_id,
            args,
        )
    )
    base_spec = inspect_decoder_spec(base_model)
    candidate_spec = inspect_decoder_spec(candidate_model)
    base_loaded_moe = any(_is_routed_mlp(mlp) for mlp in base_spec.mlps)
    candidate_loaded_moe = any(_is_routed_mlp(mlp) for mlp in candidate_spec.mlps)
    loaded_moe = base_loaded_moe or candidate_loaded_moe
    if loaded_moe and not args.allow_moe:
        return {
            "method": METHOD,
            "paper": PAPER,
            "status": "UNSUPPORTED",
            "label": label,
            "base_model": base_model_id,
            "candidate_model": candidate_model_id,
            "reason": (
                "loaded model exposes a routed sparse MLP; rerun with explicit "
                "--allow-moe for the experimental diagnostic path"
            ),
            "preflight": preflight,
            "runtime": _runtime_metadata(),
            "model_diagnostics": {
                "base_moe_layers": [
                    index for index, mlp in enumerate(base_spec.mlps) if _is_routed_mlp(mlp)
                ],
                "candidate_moe_layers": [
                    index for index, mlp in enumerate(candidate_spec.mlps) if _is_routed_mlp(mlp)
                ],
            },
            "deviations": list(DEVIATIONS),
            "elapsed_seconds": time.perf_counter() - started,
        }
    experimental_moe = bool(
        args.allow_moe
        and (preflight.get("base_is_moe") or preflight.get("candidate_is_moe") or loaded_moe)
    )
    if len(base_spec.layers) != len(candidate_spec.layers):
        return {
            "method": METHOD,
            "paper": PAPER,
            "status": "INCOMPATIBLE",
            "label": label,
            "base_model": base_model_id,
            "candidate_model": candidate_model_id,
            "reason": (
                f"decoder depth mismatch {len(base_spec.layers)} vs {len(candidate_spec.layers)}"
            ),
            "preflight": preflight,
            "deviations": list(DEVIATIONS),
            "elapsed_seconds": time.perf_counter() - started,
        }
    if base_spec.layer_path != candidate_spec.layer_path:
        return {
            "method": METHOD,
            "paper": PAPER,
            "status": "UNSUPPORTED",
            "label": label,
            "base_model": base_model_id,
            "candidate_model": candidate_model_id,
            "reason": "base and candidate use different decoder layer layouts",
            "base_layer_path": base_spec.layer_path,
            "candidate_layer_path": candidate_spec.layer_path,
            "preflight": preflight,
            "deviations": list(DEVIATIONS),
            "elapsed_seconds": time.perf_counter() - started,
        }

    parameter_report = parameter_change_report(base_model, candidate_model)
    parameter_status, parameter_reason = _parameter_compatibility(parameter_report)
    if parameter_status != "OK":
        return {
            "method": METHOD,
            "paper": PAPER,
            "status": parameter_status,
            "label": label,
            "base_model": base_model_id,
            "candidate_model": candidate_model_id,
            "reason": parameter_reason,
            "preflight": preflight,
            "parameter_diagnostics": parameter_report,
            "deviations": list(DEVIATIONS),
            "elapsed_seconds": time.perf_counter() - started,
        }

    probes = prepare_probes(
        base_tokenizer,
        candidate_tokenizer,
        _probe_words(args),
        args.probe_count,
    )
    if probes.token_mismatches:
        return {
            "method": METHOD,
            "paper": PAPER,
            "status": "UNSUPPORTED",
            "label": label,
            "base_model": base_model_id,
            "candidate_model": candidate_model_id,
            "reason": "same-base tokenizer condition is missing: probe token IDs differ",
            "probe_diagnostics": {
                "selected_words": probes.words,
                "token_mismatches": probes.token_mismatches,
                "skipped": probes.skipped,
            },
            "preflight": preflight,
            "parameter_diagnostics": parameter_report,
            "deviations": list(DEVIATIONS),
            "elapsed_seconds": time.perf_counter() - started,
        }
    if len(probes.words) < 2:
        return {
            "method": METHOD,
            "paper": PAPER,
            "status": "UNSUPPORTED",
            "label": label,
            "base_model": base_model_id,
            "candidate_model": candidate_model_id,
            "reason": f"fewer than two shared single-token probes ({len(probes.words)} available)",
            "probe_diagnostics": {
                "selected_words": probes.words,
                "skipped": probes.skipped,
                "token_mismatches": probes.token_mismatches,
            },
            "preflight": preflight,
            "parameter_diagnostics": parameter_report,
            "deviations": list(DEVIATIONS),
            "elapsed_seconds": time.perf_counter() - started,
        }

    selected_layers = _selected_layers(base_spec, args.layers)
    input_ids_base = torch.tensor(
        [[token] for token in probes.base_ids], dtype=torch.long, device=device
    )
    input_ids_candidate = torch.tensor(
        [[token] for token in probes.candidate_ids], dtype=torch.long, device=device
    )
    attention_mask = torch.ones_like(input_ids_base, device=device)
    base_inputs, base_intermediates, base_routes = _base_layer_states(
        base_model,
        base_spec,
        input_ids_base,
        attention_mask,
        selected_layers,
        capture_routes=args.allow_moe,
    )

    layer_results: list[dict[str, Any]] = []
    for layer_index in selected_layers:
        candidate_output, candidate_residual, candidate_routes = _candidate_layer_output(
            candidate_model,
            candidate_spec,
            layer_index,
            base_inputs[layer_index],
            input_ids_candidate,
            attention_mask,
            capture_routes=args.allow_moe,
        )
        direct_delta: dict[str, Any] | None = None
        oracle: dict[str, Any] | None = None
        route_diagnostics: dict[str, Any] | None = None
        candidate_residual_metadata: dict[str, Any] = {
            "available": candidate_residual is not None,
        }
        if candidate_residual is not None:
            candidate_residual_metadata["shape"] = [
                int(value) for value in candidate_residual.shape
            ]
        if args.allow_moe:
            if candidate_residual is None:
                direct_delta = {
                    "available": False,
                    "reason": "candidate post-attention residual was not captured",
                }
            else:
                try:
                    direct_delta = _direct_delta_diagnostics(
                        candidate_residual - base_intermediates[layer_index]
                    )
                except Exception as exc:
                    direct_delta = {
                        "available": False,
                        "reason": (f"direct delta evaluation failed: {type(exc).__name__}: {exc}"),
                    }
            try:
                oracle = _oracle_diagnostics(
                    base_spec.intermediate_norms[layer_index],
                    base_spec.mlps[layer_index],
                    candidate_residual
                    if candidate_residual is not None
                    else base_intermediates[layer_index],
                    candidate_output,
                )
            except Exception as exc:
                oracle = {
                    "available": False,
                    "reason": f"oracle evaluation failed: {type(exc).__name__}: {exc}",
                }
            route_diagnostics = {
                "base": base_routes.get(
                    layer_index,
                    _unavailable_routes("base router diagnostics were not captured"),
                ),
                "candidate": candidate_routes,
            }
            route_diagnostics["comparison"] = _route_switch_diagnostics(
                route_diagnostics["base"],
                route_diagnostics["candidate"],
            )

        try:
            reconstructed, reconstruction = _reconstruct_intermediate(
                base_spec.intermediate_norms[layer_index],
                base_spec.mlps[layer_index],
                base_intermediates[layer_index],
                candidate_output,
                args.reconstruction_steps,
                args.learning_rate,
            )
        except Exception as exc:
            if not experimental_moe:
                raise
            layer_results.append(
                {
                    "layer": layer_index,
                    "model_layer_class": (
                        f"{type(base_spec.layers[layer_index]).__module__}."
                        f"{type(base_spec.layers[layer_index]).__qualname__}"
                    ),
                    "candidate_model_layer_class": (
                        f"{type(candidate_spec.layers[layer_index]).__module__}."
                        f"{type(candidate_spec.layers[layer_index]).__qualname__}"
                    ),
                    "candidate_mlp": _module_metadata(candidate_spec.mlps[layer_index]),
                    "mlp": _module_metadata(base_spec.mlps[layer_index]),
                    "candidate_post_attention_residual": candidate_residual_metadata,
                    "reconstruction": {
                        "available": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    "reconstruction_progress": {
                        "status": "unavailable",
                        "ratio": None,
                        "material": False,
                    },
                    "direct_delta_diagnostics": direct_delta,
                    "oracle_diagnostics": oracle,
                    "route_diagnostics": route_diagnostics,
                    "rank_diagnostics": {"minimum_rank": None, "cycles": []},
                }
            )
            continue
        progress = _reconstruction_progress(reconstruction)
        delta = (base_intermediates[layer_index] - reconstructed).detach()
        matrix = delta.reshape(delta.shape[0], -1).to(dtype=torch.float32, device="cpu").numpy()
        rank_diagnostics = sampled_rank_diagnostics(
            matrix,
            args.cycles,
            int(matrix.shape[1]),
            args.seed + layer_index * 1009,
        )
        layer_results.append(
            {
                "candidate_model_layer_class": (
                    f"{type(candidate_spec.layers[layer_index]).__module__}."
                    f"{type(candidate_spec.layers[layer_index]).__qualname__}"
                ),
                "candidate_mlp": _module_metadata(candidate_spec.mlps[layer_index]),
                "layer": layer_index,
                "model_layer_class": (
                    f"{type(base_spec.layers[layer_index]).__module__}."
                    f"{type(base_spec.layers[layer_index]).__qualname__}"
                ),
                "candidate_post_attention_residual": candidate_residual_metadata,
                "mlp": _module_metadata(base_spec.mlps[layer_index]),
                "hidden_size": int(matrix.shape[1]),
                "probe_count": int(matrix.shape[0]),
                "reconstruction": reconstruction,
                "reconstruction_progress": progress,
                "direct_delta_diagnostics": direct_delta,
                "oracle_diagnostics": oracle,
                "route_diagnostics": route_diagnostics,
                "rank_source": "base_intermediate - reconstructed_intermediate",
                "rank_diagnostics": rank_diagnostics,
            }
        )

    layer_ranks = [
        int(layer["rank_diagnostics"]["minimum_rank"])
        for layer in layer_results
        if layer["rank_diagnostics"].get("minimum_rank") is not None
    ]
    rank_estimate = min(layer_ranks) if layer_ranks else None
    experimental_failures: list[str] = []
    if experimental_moe:
        for layer in layer_results:
            layer_label = f"layer {layer['layer']}"
            direct = layer.get("direct_delta_diagnostics") or {}
            if not direct.get("available", True):
                experimental_failures.append(
                    f"{layer_label}: candidate residual diagnostics unavailable"
                )
            oracle = layer.get("oracle_diagnostics") or {}
            if not oracle.get("available") or not oracle.get("near_numerical_precision"):
                experimental_failures.append(
                    f"{layer_label}: oracle MSE is not near numerical precision"
                )
            routes = layer.get("route_diagnostics") or {}
            if (
                not routes.get("base", {}).get("available")
                or not routes.get("candidate", {}).get("available")
                or not routes.get("comparison", {}).get("available")
            ):
                experimental_failures.append(f"{layer_label}: router diagnostics unavailable")
            progress = layer.get("reconstruction_progress") or {}
            if not progress.get("material"):
                experimental_failures.append(
                    f"{layer_label}: reconstruction made no material progress "
                    f"({progress.get('status')})"
                )
            layer_rank = layer["rank_diagnostics"].get("minimum_rank")
            if layer_rank is None or int(layer_rank) <= 0:
                experimental_failures.append(
                    f"{layer_label}: reconstructed rank is zero or unavailable"
                )
        if not layer_results:
            experimental_failures.append("no selected layers produced diagnostics")
    status = "OK"
    result_reason = None
    if experimental_moe:
        if experimental_failures:
            status = "UNSUPPORTED"
            result_reason = (
                "experimental MoE path did not satisfy strict controlled conditions: "
                + "; ".join(experimental_failures)
            )
        else:
            result_reason = (
                "experimental MoE result: unchanged routed MLP diagnostics, "
                "near-zero oracle error, material reconstruction progress, and nonzero rank"
            )
    result: dict[str, Any] = {
        "method": METHOD,
        "paper": PAPER,
        "status": status,
        "label": label,
        "base_model": base_model_id,
        "candidate_model": candidate_model_id,
        "device": str(device),
        "runtime": _runtime_metadata(),
        "experimental_moe": experimental_moe,
        "seed": args.seed,
        "configuration": {
            "probe_count_requested": args.probe_count,
            "probe_count_used": len(probes.words),
            "cycles": args.cycles,
            "reconstruction_steps": args.reconstruction_steps,
            "learning_rate": args.learning_rate,
            "layers": selected_layers,
            "allow_moe": args.allow_moe,
            "rank_aggregation": "minimum rank across cycles, then minimum across selected layers",
            "experimental_oracle_absolute_tolerance": ORACLE_ABSOLUTE_TOLERANCE,
            "experimental_oracle_relative_tolerance": ORACLE_RELATIVE_TOLERANCE,
            "experimental_material_reconstruction_ratio": MATERIAL_RECONSTRUCTION_RATIO,
        },
        "model_diagnostics": {
            "base_layer_path": base_spec.layer_path,
            "candidate_layer_path": candidate_spec.layer_path,
            "base_moe_layers": [
                index for index, mlp in enumerate(base_spec.mlps) if _is_routed_mlp(mlp)
            ],
            "candidate_moe_layers": [
                index for index, mlp in enumerate(candidate_spec.mlps) if _is_routed_mlp(mlp)
            ],
            "base_mlp_classes": [
                f"{type(mlp).__module__}.{type(mlp).__qualname__}" for mlp in base_spec.mlps
            ],
            "candidate_mlp_classes": [
                f"{type(mlp).__module__}.{type(mlp).__qualname__}" for mlp in candidate_spec.mlps
            ],
            "base_layers": len(base_spec.layers),
            "candidate_layers": len(candidate_spec.layers),
        },
        "probe_diagnostics": {
            "source": "--probe-words" if args.probe_words else "builtin deterministic words",
            "words": probes.words,
            "base_token_ids": probes.base_ids,
            "candidate_token_ids": probes.candidate_ids,
            "skipped": probes.skipped,
        },
        "parameter_diagnostics": parameter_report,
        "rank_estimate": rank_estimate,
        "layers": layer_results,
        "preflight": preflight,
        "deviations": list(DEVIATIONS),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if result_reason is not None:
        result["reason"] = result_reason
    return result


def _dry_plan(
    label: str | None,
    base_model: str | None,
    candidate_model: str | None,
    pair_error: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "method": METHOD,
        "paper": PAPER,
        "status": "DRY_RUN",
        "label": label,
        "base_model": base_model,
        "candidate_model": candidate_model,
        "pair_error": pair_error,
        "downloads": False,
        "configuration": {
            "probe_count": args.probe_count,
            "cycles": args.cycles,
            "reconstruction_steps": args.reconstruction_steps,
            "learning_rate": args.learning_rate,
            "layers": args.layers,
            "device": args.device,
            "dtype": args.dtype,
            "cache_dir": str(args.cache_dir) if args.cache_dir else None,
            "seed": args.seed,
            "allow_moe": args.allow_moe,
        },
        "probe_source": {
            "kind": "--probe-words" if args.probe_words else "builtin deterministic words",
            "words": _probe_words(args)[: args.probe_count],
            "single_token_required": True,
        },
        "deviations": list(DEVIATIONS),
        "note": (
            "Use --preflight for config-only Hub/local checks, or omit --dry-run "
            "for activation tracing."
        ),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Origin Tracer approximation: activation-assisted tracing for a specified "
            "same-base decoder-only model pair"
        )
    )
    parser.add_argument(
        "--pair", help="shared fixture label; explicit model IDs override fixture selection"
    )
    parser.add_argument(
        "--base-model", "--base-id", dest="base_model", help="base/reference model ID or local path"
    )
    parser.add_argument(
        "--candidate-model",
        "--candidate-id",
        dest="candidate_model",
        help="candidate/suspect model ID or local path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print a deterministic plan without model/config downloads",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="load only Transformers config.json files; do not load model weights or run probes",
    )
    parser.add_argument(
        "--allow-moe",
        action="store_true",
        help=(
            "experimental: trace through the base routed sparse MLP; "
            "requires unchanged router and expert functions"
        ),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="Transformers/HF cache directory"
    )
    parser.add_argument("--revision", default=None, help="optional Hub revision for both models")
    parser.add_argument(
        "--local-files-only", action="store_true", help="disable Hub network access"
    )
    parser.add_argument(
        "--probe-count",
        type=int,
        default=8,
        help="maximum shared one-token word probes (default: 8)",
    )
    parser.add_argument("--probe-words", help="comma-separated deterministic word override")
    parser.add_argument(
        "--cycles", type=int, default=4, help="random probe-subset cycles (default: 4)"
    )
    parser.add_argument(
        "--reconstruction-steps",
        type=int,
        default=24,
        help="Adam steps for each layer's intermediate reconstruction (default: 24)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="reconstruction Adam learning rate (default: 0.05)",
    )
    parser.add_argument(
        "--layers", default="0", help="comma-separated layer indices or 'all' (default: 0)"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="deterministic probe-subset seed (default: 0)"
    )
    parser.add_argument(
        "--require-base-metadata",
        action="store_true",
        help="abstain unless candidate config exposes base_model_name_or_path",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.probe_count < 2:
        parser.error("--probe-count must be at least 2")
    if args.cycles < 1:
        parser.error("--cycles must be at least 1")
    if args.reconstruction_steps < 1:
        parser.error("--reconstruction-steps must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    label, base_model, candidate_model, pair_error = _pair_from_args(args)

    if args.dry_run:
        print(
            json.dumps(
                _dry_plan(label, base_model, candidate_model, pair_error, args),
                indent=2,
                default=_json_default,
            )
        )
        return 0
    if pair_error:
        result = {
            "method": METHOD,
            "paper": PAPER,
            "status": "UNSUPPORTED",
            "label": label,
            "base_model": base_model,
            "candidate_model": candidate_model,
            "reason": pair_error,
            "deviations": list(DEVIATIONS),
        }
        print(json.dumps(result, indent=2, default=_json_default))
        return 0

    assert base_model is not None and candidate_model is not None
    try:
        if args.preflight:
            result = _preflight_configs(base_model, candidate_model, args)
            result.update(
                {
                    "method": METHOD,
                    "paper": PAPER,
                    "label": label,
                    "base_model": base_model,
                    "candidate_model": candidate_model,
                    "deviations": list(DEVIATIONS),
                }
            )
        else:
            random.seed(args.seed)
            np.random.seed(args.seed)
            result = run_analysis(label, base_model, candidate_model, args)
    except (UnsupportedArchitecture, ValueError) as exc:
        result = {
            "method": METHOD,
            "paper": PAPER,
            "status": "UNSUPPORTED",
            "label": label,
            "base_model": base_model,
            "candidate_model": candidate_model,
            "reason": str(exc),
            "deviations": list(DEVIATIONS),
        }
    except Exception as exc:  # model loading/network/runtime failures are surfaced as JSON
        result = {
            "method": METHOD,
            "paper": PAPER,
            "status": "ERROR",
            "label": label,
            "base_model": base_model,
            "candidate_model": candidate_model,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "deviations": list(DEVIATIONS),
        }
    print(json.dumps(result, indent=2, default=_json_default))
    return 0 if result.get("status") in {"OK", "PREFLIGHT_OK", "UNSUPPORTED", "INCOMPATIBLE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
