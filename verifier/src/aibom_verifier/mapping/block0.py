BLOCK0_PREFIX = "model.layers.0."

REQUIRED_SUFFIXES = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]

# Present on some dense families (e.g. Qwen2.5-style attn biases, Qwen3 QK-norm)
# but not required for a block0 plan to build; included only when both sides agree.
OPTIONAL_SUFFIXES = [
    "self_attn.q_proj.bias",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.bias",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
]


def build_block0_plan(target_names: list[str], base_names: list[str]) -> dict:
    """Plan which block-0 (`model.layers.0.*`) tensors to compare across target/base.

    Filters the full tensor name lists down to block-0 required/optional
    roles, and reports what's missing or one-sided so the caller can decide
    pass/fail/skip. Does not read or compare any tensor data itself.
    """
    target_suffixes = {
        name[len(BLOCK0_PREFIX) :] for name in target_names if name.startswith(BLOCK0_PREFIX)
    }
    base_suffixes = {
        name[len(BLOCK0_PREFIX) :] for name in base_names if name.startswith(BLOCK0_PREFIX)
    }

    pairs: list[dict[str, str]] = []
    missing_required_target: list[str] = []
    missing_required_base: list[str] = []
    one_sided: list[str] = []

    for suffix in REQUIRED_SUFFIXES:
        in_target = suffix in target_suffixes
        in_base = suffix in base_suffixes
        if in_target and in_base:
            pairs.append({"name": BLOCK0_PREFIX + suffix})
        if not in_target:
            missing_required_target.append(BLOCK0_PREFIX + suffix)
        if not in_base:
            missing_required_base.append(BLOCK0_PREFIX + suffix)

    for suffix in OPTIONAL_SUFFIXES:
        in_target = suffix in target_suffixes
        in_base = suffix in base_suffixes
        if in_target and in_base:
            pairs.append({"name": BLOCK0_PREFIX + suffix})
        elif in_target or in_base:
            one_sided.append(BLOCK0_PREFIX + suffix)

    return {
        "prefix": BLOCK0_PREFIX,
        "pairs": pairs,
        "missing_required_target": missing_required_target,
        "missing_required_base": missing_required_base,
        "one_sided": one_sided,
    }
