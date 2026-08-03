from aibom_verifier.mapping.block0 import BLOCK0_PREFIX, REQUIRED_SUFFIXES, build_block0_plan

REQUIRED_NAMES = [BLOCK0_PREFIX + suffix for suffix in REQUIRED_SUFFIXES]

NOISE_NAMES = [
    "model.layers.1.input_layernorm.weight",
    "model.embed_tokens.weight",
    "lm_head.weight",
]


def test_build_block0_plan_all_required_present_both_sides():
    plan = build_block0_plan(
        target_names=REQUIRED_NAMES + NOISE_NAMES,
        base_names=REQUIRED_NAMES + NOISE_NAMES,
    )

    assert plan["prefix"] == BLOCK0_PREFIX
    assert {p["name"] for p in plan["pairs"]} == set(REQUIRED_NAMES)
    assert plan["missing_required_target"] == []
    assert plan["missing_required_base"] == []
    assert plan["one_sided"] == []


def test_build_block0_plan_missing_required_on_target():
    missing = BLOCK0_PREFIX + "mlp.down_proj.weight"
    target_names = [n for n in REQUIRED_NAMES if n != missing]
    base_names = REQUIRED_NAMES

    plan = build_block0_plan(target_names=target_names, base_names=base_names)

    assert plan["missing_required_target"] == [missing]
    assert plan["missing_required_base"] == []
    assert missing not in {p["name"] for p in plan["pairs"]}
    assert len(plan["pairs"]) == len(REQUIRED_SUFFIXES) - 1


def test_build_block0_plan_missing_required_on_base():
    missing = BLOCK0_PREFIX + "self_attn.q_proj.weight"
    target_names = REQUIRED_NAMES
    base_names = [n for n in REQUIRED_NAMES if n != missing]

    plan = build_block0_plan(target_names=target_names, base_names=base_names)

    assert plan["missing_required_base"] == [missing]
    assert plan["missing_required_target"] == []


def test_build_block0_plan_optional_present_on_both_sides_is_paired():
    optional = BLOCK0_PREFIX + "self_attn.q_norm.weight"
    target_names = REQUIRED_NAMES + [optional]
    base_names = REQUIRED_NAMES + [optional]

    plan = build_block0_plan(target_names=target_names, base_names=base_names)

    assert optional in {p["name"] for p in plan["pairs"]}
    assert plan["one_sided"] == []


def test_build_block0_plan_optional_present_on_one_side_only_is_one_sided():
    optional = BLOCK0_PREFIX + "self_attn.k_norm.weight"
    target_names = REQUIRED_NAMES + [optional]
    base_names = REQUIRED_NAMES

    plan = build_block0_plan(target_names=target_names, base_names=base_names)

    assert plan["one_sided"] == [optional]
    assert optional not in {p["name"] for p in plan["pairs"]}


def test_build_block0_plan_ignores_non_block0_names():
    plan = build_block0_plan(
        target_names=REQUIRED_NAMES + NOISE_NAMES,
        base_names=REQUIRED_NAMES + NOISE_NAMES,
    )

    pair_names = {p["name"] for p in plan["pairs"]}
    for noise_name in NOISE_NAMES:
        assert noise_name not in pair_names
