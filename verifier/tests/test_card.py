from aibom_verifier.errors import CompareStartError
from aibom_verifier.hf import card as card_mod


class _FakeInfo:
    def __init__(self, base_model):
        self.card_data = {"base_model": base_model}


def test_resolve_presumed_base_single_string(monkeypatch):
    monkeypatch.setattr(
        card_mod,
        "HfApi",
        lambda: type("Api", (), {"model_info": lambda self, *a, **k: _FakeInfo("org/base")})(),
    )
    assert card_mod.resolve_presumed_base("org/model", "sha") == "org/base"


def test_resolve_presumed_base_single_element_list(monkeypatch):
    monkeypatch.setattr(
        card_mod,
        "HfApi",
        lambda: type("Api", (), {"model_info": lambda self, *a, **k: _FakeInfo(["org/base"])})(),
    )
    assert card_mod.resolve_presumed_base("org/model", "sha") == "org/base"


def test_resolve_presumed_base_rejects_multi_parent_list(monkeypatch):
    monkeypatch.setattr(
        card_mod,
        "HfApi",
        lambda: type(
            "Api",
            (),
            {"model_info": lambda self, *a, **k: _FakeInfo(["org/base-a", "org/base-b"])},
        )(),
    )
    try:
        card_mod.resolve_presumed_base("org/model", "sha")
        raise AssertionError("expected CompareStartError")
    except CompareStartError as exc:
        assert exc.error_code == "ambiguous_base_model"
