"""Fixture manifest for the TensorLock smoke (smokes/tensorlock/).

Dense Stage-2 pairs reuse the shared manifest (smokes/fixtures.yaml); their
labels match `pairs` / `deferred` there. Quant fixtures are TensorLock-specific
(GGUF children) and live only here, mirroring the witness-overlap precedent of
owning fixture dataclasses plus pinned REVISIONS.

Hub verification 2026-10-04: every repo resolves, carries the expected
weights, and the GGUF files below exist at the pinned revisions. Dense-pair
revisions reuse the shas verified for smokes/witness-overlap (2026-10-01).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Dense Stage-2 pairs (s_MSA connectivity gate + Eq. 7 FT direction).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DensePair:
    """One pairwise lineage fixture: expected parent → fine-tuned child."""

    label: str
    parent: str
    child: str
    relation: str  # "ft" (real lineage) | "unrelated"
    note: str = ""


SMOLLM2_1_7B = "HuggingFaceTB/SmolLM2-1.7B"
SMOLTULU_1_7B = "SultanR/SmolTulu-1.7b-Instruct"
LLAMA_3_2_1B = "meta-llama/Llama-3.2-1B"
DOLPHIN3_0_LLAMA_3_2_1B = "dphn/Dolphin3.0-Llama3.2-1B"
QWEN3_0_6B_BASE = "Qwen/Qwen3-0.6B-Base"
QWEN3_0_6B = "Qwen/Qwen3-0.6B"
SMOLLM2_135M = "HuggingFaceTB/SmolLM2-135M"
SMOLLM2_135M_INSTRUCT = "HuggingFaceTB/SmolLM2-135M-Instruct"
SMOLLM2_1_7B_INSTRUCT = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
QWEN25_0_5B = "Qwen/Qwen2.5-0.5B"
QWEN25_0_5B_INSTRUCT = "Qwen/Qwen2.5-0.5B-Instruct"
S1_MINI = "superwhisper/s1-mini"
SMOLLM2_360M = "HuggingFaceTB/SmolLM2-360M"

# Quantized children (Hub verification 2026-10-04).
# QuantFactory/Qwen2.5-0.5B-GGUF is apache-2.0 and quantizes Qwen/Qwen2.5-0.5B
# (base, not Instruct); bartowski/SmolLM2-1.7B-Instruct-GGUF quantizes
# HuggingFaceTB/SmolLM2-1.7B-Instruct.
QUANTFACTORY_QWEN25_0_5B_GGUF = "QuantFactory/Qwen2.5-0.5B-GGUF"
BARTOWSKI_SMOLLM2_1_7B_INSTRUCT_GGUF = "bartowski/SmolLM2-1.7B-Instruct-GGUF"
QWEN25_0_5B_Q8_0_FILE = "Qwen2.5-0.5B.Q8_0.gguf"
QWEN25_0_5B_Q4_K_M_FILE = "Qwen2.5-0.5B.Q4_K_M.gguf"
SMOLLM2_1_7B_INSTRUCT_Q8_0_FILE = "SmolLM2-1.7B-Instruct-Q8_0.gguf"


DENSE_PAIRS: tuple[DensePair, ...] = (
    DensePair(
        label="positive_dense",
        parent=SMOLLM2_1_7B,
        child=SMOLTULU_1_7B,
        relation="ft",
        note="fixtures.yaml pairs.positive_dense; heavy SFT child",
    ),
    DensePair(
        label="vocab_drift",
        parent=LLAMA_3_2_1B,
        child=DOLPHIN3_0_LLAMA_3_2_1B,
        relation="ft",
        note="fixtures.yaml pairs.vocab_drift; CPT + SFT child, vocab resize "
        "(MSA/FFN tensors untouched by the resize)",
    ),
    DensePair(
        label="dense_qwen3",
        parent=QWEN3_0_6B_BASE,
        child=QWEN3_0_6B,
        relation="ft",
        note="fixtures.yaml deferred.dense_qwen3; GQA + QK-norm family",
    ),
    DensePair(
        label="tiny_smoke",
        parent=SMOLLM2_135M,
        child=SMOLLM2_135M_INSTRUCT,
        relation="ft",
        note="fixtures.yaml deferred.tiny_smoke; smallest scale axis",
    ),
    DensePair(
        label="unrelated_cross_family",
        parent=SMOLLM2_1_7B,
        child=LLAMA_3_2_1B,
        relation="unrelated",
        note="fixtures.yaml pairs.unrelated_cross_family; negative control "
        "for the connectivity gate (16 vs 24 layers, hidden 2048 vs 2048)",
    ),
    DensePair(
        label="qwen3_sft_community",
        parent=QWEN3_0_6B,
        child=S1_MINI,
        relation="ft",
        note="ProvenanceBench pb-118: card declares base_model=Qwen/Qwen3-0.6B "
        "with base_model_relation=finetune — the base_model claim shape an "
        "AI BOM actually asserts",
    ),
    DensePair(
        label="qwen3_neg_cross",
        parent=S1_MINI,
        child=SMOLLM2_1_7B,
        relation="unrelated",
        note="issue #26 Dense Qwen3 negative (unrelated family). Gemma-3-1B is "
        "license-gated in this environment, so the pair is built from "
        "ProvenanceBench-grounded repos: pb-118's Qwen3-derived child vs the "
        "pb-003 cross-family control",
    ),
    DensePair(
        label="tiny_scale_negative",
        parent=SMOLLM2_135M,
        child=SMOLLM2_360M,
        relation="unrelated",
        note="issue #26 tiny negative: same family and training recipe, "
        "different scale — gate specificity within a family",
    ),
)


# ---------------------------------------------------------------------------
# Quant provenance fixtures (Eq. 3 QPS): a GGUF child plus candidate parents.
# The paper restricts candidates to the child's connectivity cluster; the
# smoke therefore gates each candidate with s_MSA first and takes the QPS
# argmax over gate-passing candidates only.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantFixture:
    """One quant-provenance fixture: GGUF child + candidate parents."""

    label: str
    child_repo: str
    child_file: str
    child_quant: str
    true_source: str  # repo the GGUF file was quantized from
    candidates: tuple[str, ...]
    note: str = ""


QUANT_FIXTURES: tuple[QuantFixture, ...] = (
    QuantFixture(
        label="quant_same_child_q8",
        child_repo=QUANTFACTORY_QWEN25_0_5B_GGUF,
        child_file=QWEN25_0_5B_Q8_0_FILE,
        child_quant="Q8_0",
        true_source=QWEN25_0_5B,
        candidates=(QWEN25_0_5B, QWEN25_0_5B_INSTRUCT, SMOLLM2_1_7B),
        note="issue #26 Quant positive: same child BF16 vs GGUF. Sibling "
        "Instruct is the same-family distractor; SmolLM2-1.7B the "
        "cross-family one (gate must reject it)",
    ),
    QuantFixture(
        label="quant_same_child_q4",
        child_repo=QUANTFACTORY_QWEN25_0_5B_GGUF,
        child_file=QWEN25_0_5B_Q4_K_M_FILE,
        child_quant="Q4_K_M",
        true_source=QWEN25_0_5B,
        candidates=(QWEN25_0_5B, QWEN25_0_5B_INSTRUCT, SMOLLM2_1_7B),
        note="quant-level axis: harsher Q4_K_M quantization noise vs the sibling's SFT delta",
    ),
    QuantFixture(
        label="quant_unrelated_gguf",
        child_repo=BARTOWSKI_SMOLLM2_1_7B_INSTRUCT_GGUF,
        child_file=SMOLLM2_1_7B_INSTRUCT_Q8_0_FILE,
        child_quant="Q8_0",
        true_source=SMOLLM2_1_7B_INSTRUCT,
        candidates=(QWEN25_0_5B, SMOLLM2_1_7B_INSTRUCT),
        note="issue #26 Quant negative: unrelated GGUF. Qwen2.5-0.5B must "
        "fail the gate; the true SmolLM2 source is the positive control",
    ),
)


# Pinned revisions (dense pairs: Hub verification 2026-10-01, shared with
# smokes/witness-overlap; GGUF repos: Hub verification 2026-10-04).
REVISIONS: dict[str, str] = {
    SMOLLM2_1_7B: "effd688a12921b4cc83e3312b6feb579f70f9c71",
    SMOLLM2_1_7B_INSTRUCT: "31b70e2e869a7173562077fd711b654946d38674",
    SMOLTULU_1_7B: "038ba8a7e8a2bff3c6ae2b352fafbc698d539f93",
    LLAMA_3_2_1B: "4e20de362430cd3b72f300e6b0f18e50e7166e08",
    DOLPHIN3_0_LLAMA_3_2_1B: "e753b6ebd7adf87036eb6a3e6de68acca5850e2f",
    QWEN3_0_6B_BASE: "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    QWEN3_0_6B: "c1899de289a04d12100db370d81485cdf75e47ca",
    SMOLLM2_135M: "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    SMOLLM2_135M_INSTRUCT: "12fd25f77366fa6b3b4b768ec3050bf629380bac",
    QWEN25_0_5B: "060db6499f32faf8b98477b0a26969ef7d8b9987",
    QWEN25_0_5B_INSTRUCT: "7ae557604adf67be50417f59c2c2f167def9a775",
    QUANTFACTORY_QWEN25_0_5B_GGUF: "808e875628eb5b8d1e30728425039f4067b6ae16",
    BARTOWSKI_SMOLLM2_1_7B_INSTRUCT_GGUF: "1f03464768bfcc0319fc50da8ff5fb20b6417ba2",
    S1_MINI: "88f6b15896c73bbb13a3b596e0afe8ea0d5150b4",
    SMOLLM2_360M: "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
}
