"""Pinned fixtures for the MoE gate probe."""

REVISIONS = {
    "allenai/OLMoE-1B-7B-0924": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
    "allenai/OLMoE-1B-7B-0924-SFT": "215cc4f73147dd68bd11e7a7dcc56bac397f4221",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "31b70e2e869a7173562077fd711b654946d38674",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    "s3nh/TinyLLama-4x1.1B-MoE": "e82b7308fe3a401995169eb99a85492a36df5b83",
}

FIXTURES = [
    {
        "label": "olmoe_sft",
        "a": "allenai/OLMoE-1B-7B-0924",
        "b": "allenai/OLMoE-1B-7B-0924-SFT",
        "relation": "same_lineage_moe_sft",
        "expect_gate": True,
        "expect_score_min": 0.9,
    },
    {
        "label": "olmoe_neg_smollm",
        "a": "allenai/OLMoE-1B-7B-0924",
        "b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "relation": "cross_family_moe_dense",
        "expect_gate": False,
        "expect_score_max": 0.3,
    },
    {
        "label": "olmoe_neg_qwen3",
        "a": "allenai/OLMoE-1B-7B-0924-SFT",
        "b": "Qwen/Qwen3-0.6B",
        "relation": "cross_family_moe_dense",
        "expect_gate": False,
        "expect_score_max": 0.3,
    },
    {
        "label": "tinyllama_upcycle",
        "a": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "b": "s3nh/TinyLLama-4x1.1B-MoE",
        "relation": "card_claimed_dense_parent_upcycled_moe",
        "expect_gate": True,
    },
]
