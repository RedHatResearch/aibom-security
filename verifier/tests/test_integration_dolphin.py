"""Network acceptance: Dolphin3.0 ↔ Llama-3.2-1B (not in default CI).

Pinned SHAs. Dolphin's card claims ``meta-llama/Llama-3.2-1B`` but expands
``vocab_size`` (128258 vs 128256), so ``arch_hash`` mismatches and the
verdict is ``fraudulent_claim`` under T1 taxonomy.
"""

import pytest

from aibom_verifier.planner import run_compare
from aibom_verifier.slots.artifact_store import FilesystemArtifactStore

TARGET_REPO = "dphn/Dolphin3.0-Llama3.2-1B"
TARGET_SHA = "e753b6ebd7adf87036eb6a3e6de68acca5850e2f"
EXPECTED_BASE_REPO = "meta-llama/Llama-3.2-1B"
BASE_SHA = "4e20de362430cd3b72f300e6b0f18e50e7166e08"


@pytest.mark.network
def test_dolphin_llama_fraudulent_claim_on_vocab_resize(tmp_path):
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
    assert result.final_verdict == "fraudulent_claim"

    by_id = {t.test_id: t for t in result.tests}
    assert by_id["arch_hash"].status == "fail"
    assert "arch_hash_mismatch" in by_id["arch_hash"].reason_codes
    assert by_id["block0_shapes"].status == "skip"
    assert by_id["block0_values"].status == "skip"
