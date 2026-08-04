"""Network acceptance: SmolLM2 → SmolTulu (honest fine-tune; not in default CI)."""

import pytest

from aibom_verifier.planner import run_compare
from aibom_verifier.slots.artifact_store import FilesystemArtifactStore

TARGET_REPO = "SultanR/SmolTulu-1.7b-Instruct"
TARGET_SHA = "038ba8a7e8a2bff3c6ae2b352fafbc698d539f93"
EXPECTED_BASE_REPO = "HuggingFaceTB/SmolLM2-1.7B"
BASE_SHA = "effd688a12921b4cc83e3312b6feb579f70f9c71"


@pytest.mark.network
def test_smol_compare_verified_derivative(tmp_path):
    store = FilesystemArtifactStore(base_dir=tmp_path / "cache")

    result = run_compare(
        TARGET_REPO,
        revision_target=TARGET_SHA,
        revision_base=BASE_SHA,
        store=store,
    )

    assert result.support_class == "dense_supported"
    assert result.base.repo_id == EXPECTED_BASE_REPO
    assert result.base.sha == BASE_SHA
    assert result.target.sha == TARGET_SHA
    assert result.base_source == "card"
    assert result.final_verdict == "verified_derivative"

    by_id = {t.test_id: t for t in result.tests}
    assert by_id["arch_hash"].status == "pass"
    assert by_id["block0_shapes"].status == "pass"
    assert by_id["block0_shapes"].compatibility == "compatible"
    assert by_id["block0_values"].status == "fail"
