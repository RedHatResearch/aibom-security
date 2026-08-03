SUPPORTED_DENSE = {"llama", "mistral", "qwen2", "qwen3"}


def classify_pair(target_model_type: str, base_model_type: str) -> str:
    """Classify a target/base pair for PoC test eligibility.

    Both sides must be an allowlisted dense architecture AND report the same
    `model_type` to be considered `dense_supported`. Anything else (unknown
    type, cross-type pair, one supported + one not) is `unsupported`.
    """
    if (
        target_model_type in SUPPORTED_DENSE
        and base_model_type in SUPPORTED_DENSE
        and target_model_type == base_model_type
    ):
        return "dense_supported"
    return "unsupported"
