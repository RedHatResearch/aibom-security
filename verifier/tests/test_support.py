import pytest

from aibom_verifier.support import SUPPORTED_DENSE, classify_pair


@pytest.mark.parametrize("model_type", sorted(SUPPORTED_DENSE))
def test_classify_pair_same_supported_type_is_dense_supported(model_type):
    assert classify_pair(model_type, model_type) == "dense_supported"


def test_classify_pair_cross_type_is_unsupported():
    assert classify_pair("llama", "qwen2") == "unsupported"


def test_classify_pair_unknown_type_is_unsupported():
    assert classify_pair("phi3", "phi3") == "unsupported"


def test_classify_pair_moe_family_is_unsupported():
    assert classify_pair("mixtral", "mixtral") == "unsupported"


def test_classify_pair_one_supported_one_not_is_unsupported():
    assert classify_pair("llama", "phi3") == "unsupported"
    assert classify_pair("phi3", "llama") == "unsupported"
