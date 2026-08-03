from aibom_verifier.arch_hash import canonicalize_arch_config, compute_arch_hash


def _dense_config(**overrides):
    base = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": 128256,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "hidden_act": "silu",
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_scaling": None,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "mlp_bias": False,
        "use_sliding_window": False,
        "sliding_window": None,
        "partial_rotary_factor": 1.0,
        # Non-architecture noise that must not affect the hash.
        "torch_dtype": "bfloat16",
        "transformers_version": "4.45.0",
    }
    base.update(overrides)
    return base


def test_identical_architecture_fields_produce_identical_hash():
    left = _dense_config()
    right = _dense_config(torch_dtype="float16", transformers_version="9.9.9")
    assert compute_arch_hash(left) == compute_arch_hash(right)


def test_changing_canonical_field_changes_hash():
    left = _dense_config()
    right = _dense_config(hidden_size=4096)
    assert compute_arch_hash(left) != compute_arch_hash(right)


def test_canonicalize_includes_missing_fields_as_null_or_defaults():
    canonical = canonicalize_arch_config({"model_type": "llama"})
    assert canonical["model_type"] == "llama"
    assert canonical["hidden_size"] is None
    assert canonical["mlp_bias"] is False
    assert "torch_dtype" not in canonical


def test_absent_vs_explicit_default_fields_hash_equal():
    with_defaults = _dense_config(head_dim=64, mlp_bias=False, attention_bias=False)
    without_defaults = _dense_config()
    del without_defaults["head_dim"]
    del without_defaults["mlp_bias"]
    del without_defaults["attention_bias"]
    assert compute_arch_hash(with_defaults) == compute_arch_hash(without_defaults)
