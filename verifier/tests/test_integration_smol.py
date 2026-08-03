"""Network smoke for the SmolLM2 → SmolTulu fixture (local only; not CI)."""

import pytest

from aibom_verifier.planner import run_compare
from aibom_verifier.slots.artifact_store import FilesystemArtifactStore

TARGET_REPO = "SultanR/SmolTulu-1.7b-Instruct"
EXPECTED_BASE_REPO = "HuggingFaceTB/SmolLM2-1.7B"


@pytest.mark.network
def test_smol_compare_verified_derivative(tmp_path):
    store = FilesystemArtifactStore(base_dir=tmp_path / "cache")

    result = run_compare(TARGET_REPO, store=store)

    assert result.support_class == "dense_supported"
    assert result.base.repo_id == EXPECTED_BASE_REPO
    assert result.base_source == "card"
    assert result.final_verdict == "verified_derivative"

    by_id = {t.test_id: t for t in result.tests}
    assert by_id["arch_hash"].status == "pass"
    assert by_id["block0_shapes"].status == "pass"
    assert by_id["block0_shapes"].compatibility == "compatible"
    assert by_id["block0_values"].status == "fail"
