"""Fixture manifest for the Witness Overlap smoke (smokes/witness-overlap/).

Pure data: repo ids, roles, and pinned revisions. No loading or scoring
logic lives here. The smoke reads TRIPLES / FAMILY_SETS and never hardcodes
a repo id.

Every id below was verified against the Hub on 2026-10-01 (resolution,
root-level safetensors, llama/qwen3 tensor naming, documented lineage via
base_model tag or model card, layout match with the family base). Revision
shas are pinned in REVISIONS so a deleted or force-pushed Hub repo cannot
silently change what the smoke scores; the smoke also reports the resolved
sha in its JSON output.

Fixture roles:
- kind="registry": listed in the paper's appendix checkpoint registry
  (arXiv:2609.31784 Table 16).
- kind="fresh": documented same-family descendant the paper never touched.
- kind="base": family root (root-ID family sets only).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Witness:
    """A third checkpoint used to orient an (anchor, target) pair."""

    repo_id: str
    kind: str  # "registry" | "fresh"


@dataclass(frozen=True)
class Triple:
    """One orientation fixture: two endpoints + witness set.

    ground_truth_parent is the repo id of the true parent, or None when the
    pair has no lineage ground truth (cross-family control).
    """

    label: str
    parent: str
    child: str
    ground_truth_parent: str | None
    witnesses: tuple[Witness, ...]
    allow_partial: bool = False
    note: str = ""


@dataclass(frozen=True)
class FamilySet:
    """One root-identification fixture: members of a single model family.

    The ground-truth root is the base checkpoint (kind="base"); remaining
    members are descendants. root_score/argmin runs over all members.
    """

    label: str
    members: tuple[Witness, ...]
    note: str = ""

    @property
    def ground_truth_root(self) -> str:
        roots = [m.repo_id for m in self.members if m.kind == "base"]
        if len(roots) != 1:
            raise ValueError(f"{self.label}: exactly one base member expected, got {roots}")
        return roots[0]


# ---------------------------------------------------------------------------
# Orientation triples (Stage-2 parent/child direction).
# ---------------------------------------------------------------------------

SMOLLM2_1_7B = "HuggingFaceTB/SmolLM2-1.7B"
SMOLTULU_1_7B = "SultanR/SmolTulu-1.7b-Instruct"
SMOLLM2_1_7B_INSTRUCT = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
NUEXTRACT_1_5_SMOL = "numind/NuExtract-1.5-smol"
FINETUNE_SMOLLM2_PYTHON = "Sungmanc/finetune_smollm2_python"
LLAMA_3_2_1B = "meta-llama/Llama-3.2-1B"
DOLPHIN3_0_LLAMA_3_2_1B = "dphn/Dolphin3.0-Llama3.2-1B"
BANGLALLAMA_3_2_1B = "BanglaLLM/BanglaLLama-3.2-1b-unolp-culturax-base-v0.0.1"
TITULM_LLAMA_3_2_1B = "hishab/titulm-llama-3.2-1b-v1.1"
QWEN3_0_6B_BASE = "Qwen/Qwen3-0.6B-Base"
QWEN3_0_6B = "Qwen/Qwen3-0.6B"
DREAMWRITER_0_6B = "mrfakename/dreamwriter-0.6b-beta"
QWEN3_0_6B_IPO = "AIPlans/Qwen3-0.6B-IPO"
QWEN3_0_6B_FINETOME = "timarni/qwen3_FineTome-100k"
SMOLLM2_135M = "HuggingFaceTB/SmolLM2-135M"
SMOLLM2_135M_INSTRUCT = "HuggingFaceTB/SmolLM2-135M-Instruct"
SMOLLM2_135M_SFT_ONLY = "HuggingFaceTB/smollm2-135M-SFT-Only"
SMOLLM2_135M_SFT_SMOLTALK = "ParitKansal/SmolLM2-135M-SFT-smoltalk"
SMOLLM2_135M_INSTRUCT_SMOL_COURSE = "puettmann/SmolLM2-135M-Instruct-Smol-Course"
SMOLCHESS = "nlpguy/smolchess"
# Stress-axis fixtures (Hub verification 2026-10-04; see smoke README).
QWEN25_0_5B = "Qwen/Qwen2.5-0.5B"
QWEN25_0_5B_INSTRUCT = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN25_0_5B_INSTRUCT_FP8 = "RedHatAI/Qwen2.5-0.5B-Instruct-FP8-dynamic"
QWEN25_0_5B_MED_PRE = "Rumiii/Qwen2.5-0.5B-Med-Pre-Trained-92k"
QWEN25_0_5B_MED_POST = "Rumiii/Qwen2.5-0.5B-Med-Post-Trained-92k"
DUOGUARD_0_5B = "DuoGuard/DuoGuard-0.5B"
SMOLLM2_135M_INSTRUCT_TLDR = "mnoukhov/SmolLM2-135M-Instruct_tldr-sft"
SMOLLM2_135M_SOUP1 = "ThomasTheMaker/smollm2-135m-soup1"
SMOLLM2_135M_SLERP = "lldois/SmolLM2-135M-Reasoning-SLERP-Champion"
LLAMA_3_2_1B_INSTRUCT_FINETOME = "arogov/Llama-3.2-1B-Instruct-FineTome-100k"
SMOLLM2_360M_INSTRUCT = "HuggingFaceTB/SmolLM2-360M-Instruct"
SMOLLM2_360M_BOOLQ_GRPO = "25b3nk/smollm2-360m-boolq-calibration-grpo2-v1"
SMOLLM2_360M_KD17B = "DSTI/SmolLM2-360M-AccidentReports-distilled-kd1.7B"

# Pinned revisions (Hub verification 2026-10-01; see smoke README).
REVISIONS: dict[str, str] = {
    SMOLLM2_1_7B: "effd688a12921b4cc83e3312b6feb579f70f9c71",
    SMOLTULU_1_7B: "038ba8a7e8a2bff3c6ae2b352fafbc698d539f93",
    SMOLLM2_1_7B_INSTRUCT: "31b70e2e869a7173562077fd711b654946d38674",
    NUEXTRACT_1_5_SMOL: "7b4eda06be472f905b06ee9b18feaf5dd227d4d5",
    FINETUNE_SMOLLM2_PYTHON: "0003559fc62134d1611249f2888adabc15ae81c0",
    LLAMA_3_2_1B: "4e20de362430cd3b72f300e6b0f18e50e7166e08",
    DOLPHIN3_0_LLAMA_3_2_1B: "e753b6ebd7adf87036eb6a3e6de68acca5850e2f",
    BANGLALLAMA_3_2_1B: "4f9913f8a90169b8027cd6362d5d9d63d30e5038",
    TITULM_LLAMA_3_2_1B: "521da712ac50c53ea2fb6ccdd9abeab8f79e0820",
    QWEN3_0_6B_BASE: "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    QWEN3_0_6B: "c1899de289a04d12100db370d81485cdf75e47ca",
    DREAMWRITER_0_6B: "a52d7f8a8dcbbca6bf7461c2c4cf7fe7ca4f1302",
    QWEN3_0_6B_IPO: "72bda5389820fb25db5f7c5b26fb40c7df11194e",
    QWEN3_0_6B_FINETOME: "5aad7b02e86ca800417c0339194af18605369f38",
    SMOLLM2_135M: "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    SMOLLM2_135M_INSTRUCT: "12fd25f77366fa6b3b4b768ec3050bf629380bac",
    SMOLLM2_135M_SFT_ONLY: "79528469cd11749bac3e8e9200fd9c192fbd8979",
    SMOLLM2_135M_SFT_SMOLTALK: "465c7183b6cc4a74b2bb07eecdb0d490170dfbf5",
    SMOLLM2_135M_INSTRUCT_SMOL_COURSE: "cddd21c33fc6a06161e8994984ff3dee7a875bf5",
    SMOLCHESS: "736dac94f4855a327ae5da5f184d41b5fbe65d95",
    QWEN25_0_5B: "060db6499f32faf8b98477b0a26969ef7d8b9987",
    QWEN25_0_5B_INSTRUCT: "7ae557604adf67be50417f59c2c2f167def9a775",
    QWEN25_0_5B_INSTRUCT_FP8: "67c5c3a4f39629d3653b715527f363d86929841e",
    QWEN25_0_5B_MED_PRE: "915fc914926974078c123044cefbf9c12d639877",
    QWEN25_0_5B_MED_POST: "a59899a5078afcdf4a0e6e075fa08dda30638434",
    DUOGUARD_0_5B: "44396c3576fdd5f844c64615489cdbb5b3b3f3ce",
    SMOLLM2_135M_INSTRUCT_TLDR: "c2a4cbaae3ed36f7dbb2da0ed1d52d12e8a1b5ee",
    SMOLLM2_135M_SOUP1: "b2b072b6e45dc183fe4e9fa7ea86047934661368",
    SMOLLM2_135M_SLERP: "e4a5e65492a2d2fdf5bb739610ba2aee3b20b2a7",
    LLAMA_3_2_1B_INSTRUCT_FINETOME: "d95d0a92bae163ac014e46039f0ec083efaf16b9",
    SMOLLM2_360M_INSTRUCT: "a10cc1512eabd3dde888204e902eca88bddb4951",
    SMOLLM2_360M_BOOLQ_GRPO: "275f355f586c3512f00a7b7f00b07f32320bad80",
    SMOLLM2_360M_KD17B: "092defc756cb86f37a08bf147e501908e44c578b",
}


TRIPLES: tuple[Triple, ...] = (
    Triple(
        label="positive_dense",
        parent=SMOLLM2_1_7B,
        child=SMOLTULU_1_7B,
        ground_truth_parent=SMOLLM2_1_7B,
        witnesses=(
            Witness(repo_id=SMOLLM2_1_7B_INSTRUCT, kind="registry"),
            Witness(repo_id=NUEXTRACT_1_5_SMOL, kind="fresh"),
        ),
        note="Healthy dense pair. Registry witness: SmolLM2-1.7B-Instruct is the "
        "base of the paper's SmolLM2-1.7B-Instruct registry family and a "
        "documented child of this pair's parent. Fresh witness: NuExtract-1.5-smol "
        "(extraction SFT of SmolLM2-1.7B, not in the paper registry).",
    ),
    Triple(
        label="vocab_drift",
        parent=LLAMA_3_2_1B,
        child=DOLPHIN3_0_LLAMA_3_2_1B,
        ground_truth_parent=LLAMA_3_2_1B,
        witnesses=(
            Witness(repo_id=BANGLALLAMA_3_2_1B, kind="fresh"),
            Witness(repo_id=TITULM_LLAMA_3_2_1B, kind="fresh"),
        ),
        note="Child resized vocab 128258 vs anchor 128256 (+2 ChatML tokens); the "
        "7-projection B-set is unaffected. The paper registry has NO "
        "Llama-3.2-1B-base family (only Llama-3.2-1B-Instruct), so both witnesses "
        "are fresh documented Llama-3.2-1B-base descendants: Bangla continual "
        "pretrains (culturax / Bangla corpora).",
    ),
    Triple(
        label="unrelated_cross_family",
        parent=SMOLLM2_1_7B,
        child=LLAMA_3_2_1B,
        ground_truth_parent=None,
        witnesses=(
            Witness(repo_id=SMOLLM2_1_7B_INSTRUCT, kind="registry"),
            Witness(repo_id=BANGLALLAMA_3_2_1B, kind="fresh"),
        ),
        allow_partial=True,
        note="Cross-family abstain/uninformative control (no lineage; expected: no "
        "coherent directional signal). First witness is same-family with the "
        "SmolLM2 endpoint, second with the Llama endpoint. Layer counts differ "
        "(24 vs 16) and kv heads differ (32 vs 8) -> allow_partial: run on the "
        "tensor-name-and-shape intersection, report status='degraded'.",
    ),
    Triple(
        label="qwen3",
        parent=QWEN3_0_6B_BASE,
        child=QWEN3_0_6B,
        ground_truth_parent=QWEN3_0_6B_BASE,
        witnesses=(
            Witness(repo_id=DREAMWRITER_0_6B, kind="registry"),
            Witness(repo_id=QWEN3_0_6B_IPO, kind="registry"),
            Witness(repo_id=QWEN3_0_6B_FINETOME, kind="fresh"),
        ),
        note="Paper-registry Qwen3-0.6B-Base family (dreamwriter, IPO) plus a "
        "fresh community SFT (FineTome-100k). Correct lineage per "
        "pb-smoke-qwen3-sft: the Base is the parent, Qwen3-0.6B the post-trained "
        "child.",
    ),
    Triple(
        label="tiny_smoke",
        parent=SMOLLM2_135M,
        child=SMOLLM2_135M_INSTRUCT,
        ground_truth_parent=SMOLLM2_135M,
        witnesses=(
            Witness(repo_id=SMOLLM2_135M_SFT_ONLY, kind="registry"),
            Witness(repo_id=SMOLLM2_135M_SFT_SMOLTALK, kind="registry"),
            Witness(repo_id=SMOLLM2_135M_INSTRUCT_SMOL_COURSE, kind="registry"),
            Witness(repo_id=SMOLCHESS, kind="fresh"),
        ),
        note="Expected-failure control: the paper's named high-rho/high-eta family "
        "(SmolLM2-135M). Either outcome - failure reproduced or not - is a valid "
        "replication result.",
    ),
    # --- stress-axis fixtures (verified 2026-10-04; see smoke README) ---
    Triple(
        label="quant_fp8",
        parent=QWEN25_0_5B,
        child=QWEN25_0_5B_INSTRUCT,
        ground_truth_parent=QWEN25_0_5B,
        witnesses=(
            Witness(repo_id=QWEN25_0_5B_INSTRUCT_FP8, kind="fresh"),
            Witness(repo_id=QWEN25_0_5B_MED_PRE, kind="fresh"),
        ),
        note="Axis 1 quantized-storage: FP8-dynamic witness stores all 168 "
        "projection .weight tensors as F8_E4M3 (verified via header range "
        "read); smoke loads them through torch->float32 unchanged. Same "
        "regime as bench pair pb-smoke-qwen25-fp8. Dense witness is the "
        "control. AWQ/GPTQ int4 rehosts break the name intersection "
        "(qweight/g_idx) and are out of scope without dequantize support.",
    ),
    Triple(
        label="tiny_delta",
        parent=QWEN25_0_5B,
        child=QWEN25_0_5B_MED_PRE,
        ground_truth_parent=QWEN25_0_5B,
        witnesses=(
            Witness(repo_id=QWEN25_0_5B_INSTRUCT, kind="fresh"),
            Witness(repo_id=QWEN25_0_5B_MED_POST, kind="fresh"),
        ),
        note="Axis 2 sensitivity floor: child is a documented ~23.6M-token "
        "CPT of the base (92k PubMed abstracts; arXiv:2506.09513). Either "
        "the minimal real delta still orients (floor below smallest real "
        "delta) or it flips (tagged sensitivity limit).",
    ),
    Triple(
        label="huge_delta",
        parent=QWEN25_0_5B,
        child=DUOGUARD_0_5B,
        ground_truth_parent=QWEN25_0_5B,
        witnesses=(
            Witness(repo_id=QWEN25_0_5B_INSTRUCT, kind="fresh"),
            Witness(repo_id=QWEN25_0_5B_MED_POST, kind="fresh"),
        ),
        note="Axis 3 specificity ceiling: child transplanted to a 12-way "
        "safety-classifier head (lm_head -> score (12,896), embed kept); "
        "7-role projections untouched. Tests whether extreme task shift "
        "with intact weight lineage still orients.",
    ),
    Triple(
        label="merge_soup",
        parent=SMOLLM2_135M_INSTRUCT,
        child=SMOLLM2_135M_INSTRUCT_TLDR,
        ground_truth_parent=SMOLLM2_135M_INSTRUCT,
        witnesses=(
            Witness(repo_id=SMOLLM2_135M_SOUP1, kind="fresh"),
            Witness(repo_id=SMOLLM2_135M_SLERP, kind="fresh"),
        ),
        note="Axis 4 merge/soup: soup1 is a card-documented linear merge "
        "(w=1) of 135M-Instruct + this pair's child + SmolLM2-135M; SLERP "
        "Champion is a t=0.3 SLERP of two DPO children of the anchor. "
        "Linear-combination deltas should still orient toward the anchor.",
    ),
    Triple(
        label="same_data_diff_lineage",
        parent=QWEN3_0_6B_BASE,
        child=QWEN3_0_6B_FINETOME,
        ground_truth_parent=QWEN3_0_6B_BASE,
        witnesses=(
            Witness(repo_id=LLAMA_3_2_1B_INSTRUCT_FINETOME, kind="fresh"),
            Witness(repo_id=DREAMWRITER_0_6B, kind="registry"),
        ),
        allow_partial=True,
        note="Axis 5 same-data adversary: first witness is a FineTome-100k "
        "SFT of Llama-3.2-1B-Instruct (same public dataset as the child, "
        "different lineage; config _name_or_path evidence). Zero tensor "
        "intersection with the Qwen3 pair -> must abstain/degrade, never "
        "emit confident lineage. dreamwriter is the positive control.",
    ),
    Triple(
        label="kd_student",
        parent=SMOLLM2_360M_INSTRUCT,
        child=SMOLLM2_360M_BOOLQ_GRPO,
        ground_truth_parent=SMOLLM2_360M_INSTRUCT,
        witnesses=(Witness(repo_id=SMOLLM2_360M_KD17B, kind="fresh"),),
        note="Axis 5b: witness was logit-KD distilled FROM "
        "SmolLM2-1.7B-Instruct outputs (teacher named in card) but its "
        "weights descend from the 360M anchor; probes that output-derived "
        "data does not create phantom orientation.",
    ),
)


# ---------------------------------------------------------------------------
# Root-identification family sets (Stage-2 root ID).
# ---------------------------------------------------------------------------

FAMILY_SETS: tuple[FamilySet, ...] = (
    FamilySet(
        label="smollm2_1_7b",
        members=(
            Witness(repo_id=SMOLLM2_1_7B, kind="base"),
            Witness(repo_id=SMOLTULU_1_7B, kind="registry"),
            Witness(repo_id=SMOLLM2_1_7B_INSTRUCT, kind="registry"),
            Witness(repo_id=NUEXTRACT_1_5_SMOL, kind="fresh"),
            Witness(repo_id=FINETUNE_SMOLLM2_PYTHON, kind="fresh"),
        ),
        note="Healthy control: root = SmolLM2-1.7B base, 4 documented descendants.",
    ),
    FamilySet(
        label="qwen3_0_6b",
        members=(
            Witness(repo_id=QWEN3_0_6B_BASE, kind="base"),
            Witness(repo_id=QWEN3_0_6B, kind="registry"),
            Witness(repo_id=DREAMWRITER_0_6B, kind="registry"),
            Witness(repo_id=QWEN3_0_6B_IPO, kind="registry"),
            Witness(repo_id=QWEN3_0_6B_FINETOME, kind="fresh"),
        ),
        note="Healthy control: root = Qwen3-0.6B-Base, 4 documented descendants.",
    ),
    FamilySet(
        label="smollm2_135m",
        members=(
            Witness(repo_id=SMOLLM2_135M, kind="base"),
            Witness(repo_id=SMOLLM2_135M_INSTRUCT, kind="registry"),
            Witness(repo_id=SMOLLM2_135M_SFT_ONLY, kind="registry"),
            Witness(repo_id=SMOLLM2_135M_SFT_SMOLTALK, kind="registry"),
            Witness(repo_id=SMOLCHESS, kind="fresh"),
        ),
        note="Failing control: paper reports frequent wrong roots in this family.",
    ),
)
