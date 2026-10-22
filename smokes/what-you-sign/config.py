"""Fixture manifest for the what-you-sign smoke (smokes/what-you-sign/).

Paper under test: arXiv:2610.19498 — "What You Sign Is Not What You Run:
The Verifiability Gap in Quantized LLM Supply Chains" (Liu & Tsai, v1
Oct 2026). Constructive cross-check: ggufpacker
(github.com/theadamdanielsson/ggufpacker, MIT).

Conventions follow smokes/tensorlock/config.py: pinned Hub revisions,
pre-registered expectations fixed before the first run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# The W under test: publisher f16 GGUF (paper's own pilot-scale choice is
# Qwen2.5-1.5B; we drop to 0.5B to stay well inside a laptop smoke).
# Rev verified on the Hub 2026-10-20; the f16 file is the only cold download
# (~994 MB). Quantized children of this repo (Q8_0/Q4_K_M) were already
# cached from the tensorlock / witness-overlap runs.
# ---------------------------------------------------------------------------

PUBLISHER_REPO = "bartowski/Qwen2.5-0.5B-Instruct-GGUF"
PUBLISHER_REVISION = "41ba88dbac95fed2528c92514c131d73eb5a174b"
F16_FILENAME = "Qwen2.5-0.5B-Instruct-f16.gguf"
# Recorded after download 2026-10-20; the smoke asserts it before quantizing.
F16_SHA256 = "f1ad9d1174ce6ab47b584d522634c47e411b75bffdffd9a4e106e21e882392e5"

# Published sibling artifacts from the same repo, used for the ggufpacker
# cross-check (PLAN_EXACT vs PLAN_NEAR claim) on the publisher's own files.
PUBLISHED_SIBLINGS = ("Qwen2.5-0.5B-Instruct-Q8_0.gguf", "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")

# ---------------------------------------------------------------------------
# Execution environments E (paper §V): three llama.cpp builds at one pinned
# commit, CPU-only, identical link profiles (GGML_METAL/ACCELERATE/OPENMP/
# BLAS=OFF, LLAMA_OPENSSL=OFF — Apple clang lacks -fopenmp; the libomp
# arm64-only link issue is documented in the build session).
#
#   quantize-default     arm64 / Apple clang, default FP contraction
#   quantize-nocontract  arm64 / Apple clang, PR #25353 patch applied to
#                        ggml/src/CMakeLists.txt: file-scoped
#                        set_source_files_properties(ggml-quants.c
#                        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
#   quantize-x86-default x86_64 under Rosetta 2, default contraction,
#                        baseline SSE2 (GGML_NATIVE=OFF, FMA/AVX/F16C/SSE42/
#                        BMI2 all OFF — object-level check: 0 FMA, 0 AVX
#                        instructions), i.e. the paper's "baseline x86-64 is
#                        uncontracted" premise.
#
# Binaries + sources live OUTSIDE the repo in .scratch/ (clone dir +
# git worktree + build dirs, llama.cpp commit d8d92973, 2026-10-20 tip);
# the smoke takes their root via --builds-root (default .scratch/llamacpp-builds).
# ---------------------------------------------------------------------------

LLAMA_CPP_COMMIT = "d8d92973c8c21bb4dd5d592f0fec26fe2cd1e654"

DEFAULT_BUILDS_ROOT = Path(__file__).resolve().parents[2] / ".scratch" / "llamacpp-builds"

BUILDS: tuple[dict[str, str], ...] = (
    {
        "label": "arm64-default",
        "binary": "quantize-default/llama-quantize",
        "arch": "arm64",
        "contraction": "on (clang default, NEON FMA contracted)",
    },
    {
        "label": "arm64-nocontract",
        "binary": "quantize-nocontract/llama-quantize",
        "arch": "arm64",
        "contraction": "off (PR #25353 file-scoped -ffp-contract=off on ggml-quants.c)",
    },
    {
        "label": "x86_64-default",
        "binary": "quantize-x86-default/llama-quantize",
        "arch": "x86_64 (Rosetta 2)",
        "contraction": "default; FMA unavailable (baseline SSE2 = uncontracted)",
    },
)


@dataclass(frozen=True)
class QuantPlan:
    """One quantization run: fixed W, one environment, one quant type."""

    label: str
    build_label: str
    quant_type: str
    # Pre-registered expectation, fixed before the first run (paper §V Tab. IV):
    #   "cross-arch default builds byte-differ for scale-search types;
    #    contraction-off restores cross-arch bit-identity; Q8_0 never diverges"
    expectation: str


# Paper §V types: Q4_K_M, Q5_K_M, IQ4_XS (scale-search) + Q8_0 (control).
QUANT_TYPES = ("Q4_K_M", "Q5_K_M", "IQ4_XS", "Q8_0")

# The hash-matrix expectation, pre-registered before any run:
# (build_a, build_b) -> expected byte-identical per quant type.
# Paper Fig. 2a at 1.5B (x86 gcc vs arm64 clang):
#   default vs default:        differ on Q4_K_M, Q5_K_M, IQ4_XS; Q8_0 equal
#   contraction-off (either) : equal on all four
# Our arm64-default vs arm64-nocontract pair is the paper's §V single-machine
# proxy (exact at 1.5B: x86 default == arm64 contraction-off); the
# x86_64-default column then tests the cross-arch claim on one machine.
EXPECTED_EQUAL: dict[tuple[str, str], dict[str, bool]] = {
    # arm64 default vs arm64 contraction-off: the paper's single-machine
    # proxy pair — divergence expected on all three scale-search types.
    ("arm64-default", "arm64-nocontract"): {
        "Q4_K_M": False,
        "Q5_K_M": False,
        "IQ4_XS": False,
        "Q8_0": True,
    },
    # x86_64 baseline default vs arm64 contraction-off: paper claims the
    # x86-64 baseline build (no FMA) is uncontracted, so bit-identical.
    ("x86_64-default", "arm64-nocontract"): {
        "Q4_K_M": True,
        "Q5_K_M": True,
        "IQ4_XS": True,
        "Q8_0": True,
    },
    # x86_64 baseline default vs arm64 default: divergence expected (the
    # arm64 default contracts FMA).
    ("x86_64-default", "arm64-default"): {
        "Q4_K_M": False,
        "Q5_K_M": False,
        "IQ4_XS": False,
        "Q8_0": True,
    },
}

# Paper §IV 'card' column, 0.5B edition: per-tensor payload-equality of the
# stock-preset regeneration vs the published Q4_K_M, per build.
# Measured 2026-10-20, identical across all three builds (deterministic
# under the pins).
CARD_ALONE_EXPECTED = {"matched": 126, "type_mismatched": 32, "total": 290}

# Paper §IV recipe gap, 0.5B edition: published Q4_K_M tensors whose
# per-tensor type differs from the stock Q4_K_M preset. Measured 2026-10-20.
RECIPE_GAP_EXPECTED = 32

# Paper §IV imatrix-dominates relation, 0.5B edition: with the full
# published recipe pinned, the residual published-vs-regenerated XOR
# fraction (publisher's unshipped imatrix) stays ≥ 0.10 per build — well
# above the ~8e-5 contraction beta. Measured ≈ 0.1483 on 2026-10-20; the
# bound is the relation, not the exact float.
IMATRIX_GAP_MIN_FRAC = 0.10

# Paper §VI detection floor, 0.5B edition: our measured floor must not
# exceed the paper's 0.02% Q4_K floor at 1.5B. Measured 0.01% (0.0001)
# on 2026-10-20.
FLOOR_MAX = 0.02

# Detection-floor probe (paper §VI): natural variance R_nat = arm64-default
# rebuild vs reference (arm64-nocontract, the designated measurement
# reference); tampering = reference + uniformly placed one-step nibble flips.
# Statistic: per-tensor tail fraction of |dw|/sigma > tau (paper tau=0.1).
DETECTOR_TAU = 0.1
DETECTOR_SEEDS = 5
# Budget ladder straddles the paper's 0.02% Q4_K floor from below, so the
# floor claim is falsifiable rather than only confirmed from above.
TAMPER_BUDGETS = (0.00002, 0.00005, 0.0001, 0.0002, 0.001, 0.005, 0.01, 0.02)
# Sparse large-edit probe (paper §VI.d, the EXP11 --large mode): k
# farthest-code edits placed at global positions over the type's concatenated
# weights, rng = default_rng([1000+s, k]) per seed s. Pre-registered: a single
# farthest-code edit already exceeds the honest max |dw|/sigma in every seed.
LARGE_EDIT_KS = (1, 5)

# Census re-check (paper §III, harness EXP02/EXP03): rerun the frozen top-200
# repo census. The harness lives outside the repo (.scratch, not committed);
# default overridable via --harness-root.
DEFAULT_HARNESS_ROOT = (
    Path(__file__).resolve().parents[2] / ".scratch" / "gguf-verifiability-artifact"
)
CENSUS_REPOS_FILE = "exp/EXP03-ecosystem-census/data/frozen_repos_top200.txt"
CENSUS_SCRIPT = "exp/EXP02-census-tool/scripts/census.py"
# Pre-registered expectations, re-verified against a fresh harness run on
# 2026-10-20: 0 bit-exact verifiable of the repos scored; 148 of the 197
# scored repos source-reconstructible; 3 repos now error out
# (deleted/gated/unreadable header, counted in n_error, not a failure);
# every scored repo lacks the env/version record.
CENSUS_EXPECTED = {"n_repos": 200, "bit_exact_verifiable": 0, "source_reconstructible": 148}
# ---------------------------------------------------------------------------
# b9222 trial-repo re-derivation (paper §IV Table III, the positive control):
# bartowski/Qwen_Qwen3.5-0.8B-GGUF ships its bf16, its imatrix (card
# coordinate C) and names release b9222 on the card — the trial family where
# the paper's full protocol re-derives the published Q4_K_M bit-for-bit.
# Rev verified on the Hub 2026-10-20; sha256s recorded after download.
# ---------------------------------------------------------------------------
B9222_COMMIT = "9a532ae4bab1b164052ce60a738f78538b421c66"  # llama.cpp tag b9222
TRIAL_REPO = "bartowski/Qwen_Qwen3.5-0.8B-GGUF"
TRIAL_REVISION = "f36b1ea49a332ede8fe5f389bbf5b3575ef71f48"
TRIAL_BF16 = "Qwen_Qwen3.5-0.8B-bf16.gguf"
TRIAL_BF16_SHA256 = "ad1549eedc613064971dcbbbfab6c9b7990984d1c9ab38f792c6f2ec1207bbc2"
TRIAL_IMATRIX = "Qwen_Qwen3.5-0.8B-imatrix.gguf"
TRIAL_IMATRIX_SHA256 = "e79b46b28c107fcdc0e39711075099f64738c00721120b11f30fc9a3d65322df"
TRIAL_PUBLISHED = "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"
TRIAL_PUBLISHED_SHA256 = "fb044e93939a70469c905781334f5de1e6c8b608ced6cbc8c9249bd4127d9526"
# b9222 build trees live outside the repo (.scratch, not committed): the
# smoke locates the binaries by globbing these trees (or takes a
# --b9222-builds-root with the quantize-* layout); build them from the
# pinned release with scripts/build_llamacpp.sh --b9222.
B9222_BUILD_GLOB = ".scratch/llama.cpp-b9222*/build*/bin/llama-quantize"
# Paper Table III, Qwen3.5-0.8B row, pre-registered (tolerance: exact) under
# the paper's per-tensor comparison (name-keyed payload XOR; metadata/KV
# excluded by design): run key -> (matched tensors, total tensors).
#   card-only (arm64 default contraction, stock preset): 140/335
#   arm64 contraction-off / x86_64 baseline, stock preset: 260/335
#   + full published recipe: 335/335 (bit-exact re-derivation)
B9222_TABLE_III_EXPECTED = {
    "arm64-default__stock": (140, 335),
    "arm64-nocontract__stock": (260, 335),
    "arm64-nocontract__recipe": (335, 335),
    "x86_64-default__stock": (260, 335),
    "x86_64-default__recipe": (335, 335),
}
# Same-binary reruns (paper: bit-deterministic over 10 repeats) and
# thread-count invariance (paper: 8- vs 6-thread outputs byte-identical).
REPEAT_RUNS = 3
THREAD_VARIANTS = (1, 8)
