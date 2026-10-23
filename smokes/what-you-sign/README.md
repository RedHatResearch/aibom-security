# What-you-sign smoke (smokes/what-you-sign/)

Runnable check for **"What You Sign Is Not What You Run: The Verifiability
Gap in Quantized LLM Supply Chains"** (Liu & Tsai,
[arXiv:2610.19498](https://arxiv.org/abs/2610.19498), v1 Oct 2026), with
[ggufpacker](https://github.com/theadamdanielsson/ggufpacker) as the
constructive-engineering cross-check. Survey deep-dive for
[#26](https://github.com/RedHatResearch/aibom-security/issues/26); runnable-check
context in [../README.md](../README.md).

The paper measures the GGUF **independent-verifiability gap**: across the
200 most-downloaded HF GGUF repos (198 in scope), **0/198** disclose enough
for a third party to re-derive a deployed quantized artifact bit-for-bit
(recorded-inputs IVR); with two card-absent coordinates supplied (the build's
FP-contraction setting, and for hybrid models the per-tensor recipe), **7/8**
Q4_K_M files re-derive bit-for-bit on x86 (5/8 arm64). Census instrument =
GGUF header probes (8–16 MB ranged reads) — same block-0 philosophy as the
aibom verifier.

This smoke reproduces, at 0.5B scale on one arm64 laptop (Apple M3 Pro),
the claims that are checkable without a second machine or full-model
downloads:

1. **§V cross-environment divergence (B>0)** — quantize one fixed W
   (bartowski Qwen2.5-0.5B-Instruct f16 GGUF, pinned revision + sha256) with
   three CPU-only builds of one llama.cpp commit:
   `arm64-default`, `arm64-nocontract` (PR #25353's file-scoped
   `-ffp-contract=off` on `ggml-quants.c`), and `x86_64-default` (Rosetta 2,
   baseline SSE2, no FMA — the paper's "baseline x86-64 is uncontracted"
   premise, verified at object level: 0 FMA/AVX instructions). Scale-search
   types (Q4_K_M, Q5_K_M, IQ4_XS) byte-differ between default builds and
   agree under contraction-off; Q8_0 never diverges.
2. **§V localization** — divergence confined to quantized weight tensors
   (KV/metadata region byte-identical), concentrated in k-quant scale bytes
   (superscale/min/blockscales regions), propagating into packed nibbles.
3. **§VI detection floor** — per-tensor tail fraction of |Δw|/σ > τ=0.1
   (σ = population std of the reference-side dequantized weights) separates
   benign contraction variance (R_nat = arm64-default vs arm64-nocontract)
   from uniformly placed one-step nibble flips (analytic positives, the
   harness EXP11 method); we probe a budget ladder around the paper's 0.02%
   Q4_K floor. The same probe also runs the max-statistic / sparse
   large-edit check (paper §VI.d, harness EXP11 `--large`): k ∈ {1, 5}
   farthest-code edits at global positions over the type's concatenated
   weights (rng `default_rng([1000+s, k])` per seed); a single farthest-code
   edit must exceed the honest max |Δw|/σ in every seed, five exceed it
   further.
4. **ggufpacker cross-check** — its EXACT/NEAR/BLOB plan decision,
   reimplemented on our regenerated files XORed against the publisher's own
   published Q8_0 / Q4_K_M siblings: Q8_0 regenerates zero-XOR from every
   environment (EXACT — asserted; no scale search), while the published
   Q4_K_M is non-congruent with the stock preset (the recipe gap, the
   card's missing per-tensor recipe coordinate — asserted).
5. **§III census re-score** (`--census`, ~10 min, network) — rerun the
   paper's own frozen instrument (harness `EXP02 census.py`, stdlib-only,
   8–16 MB ranged header reads) over the `EXP03` frozen top-200 repo list
   from `.scratch/gguf-verifiability-artifact` (overridable via
   `--harness-root`) and check the pre-registered counts: 0/200 bit-exact
   verifiable, 148 source-reconstructible, every scored repo missing the
   env/version record; repos that now error out (deleted/gated) are counted
   in `n_error`, and any drift from the pinned counts fails the run loudly.
6. **§IV Table III trial re-derivation** (`--b9222`, ~10 min cold cache)
   — the paper's positive control: `bartowski/Qwen_Qwen3.5-0.8B-GGUF`
   (pinned rev `f36b1ea`) ships its bf16, its imatrix and names release
   b9222 on the card. Two b9222 builds (arm64 PR #25353; x86_64 Rosetta
   baseline SSE2) re-derive the published Q4_K_M with the publisher's
   imatrix in stock-preset and full-recipe modes, plus the arm64-default
   card-only column: 260/335 stock (75 hybrid SSM/MTP type mismatches,
   every same-type tensor XORs to zero), 335/335 +recipe (bit-exact),
   140/335 card-only — the paper's Table III row exactly. Binaries are
   located in the `.scratch` build trees (or `--b9222-builds-root`);
   when missing the smoke exits 1 pointing at
   `scripts/build_llamacpp.sh --b9222`.

Pre-registered expectations (fixed in `config.py` before the first run,
from the paper's Fig. 2a at 1.5B) — every one of them is hard-asserted:
the smoke exits non-zero listing each failing cell on any drift.

| pair | Q4_K_M | Q5_K_M | IQ4_XS | Q8_0 |
|---|---|---|---|---|
| arm64-default vs arm64-nocontract | differ | differ | differ | equal |
| x86_64-default vs arm64-nocontract | equal | equal | equal | equal |
| x86_64-default vs arm64-default | differ | differ | differ | equal |

```bash
uv sync --all-packages --group smokes
uv run --group smokes python smokes/what-you-sign/smoke.py
# with the census re-score (needs the harness clone + network, ~2 min extra):
uv run --group smokes python smokes/what-you-sign/smoke.py --census
# with the b9222 Table III re-derivation (~10 min cold cache):
uv run --group smokes python smokes/what-you-sign/smoke.py --b9222
```

Stdout is a single JSON document (progress goes to stderr). Requires the
three llama-quantize builds under `.scratch/llamacpp-builds/` (outside the
repo; built at llama.cpp commit `d8d92973`, see `config.py` for the exact
build flags). Cold run downloads the f16 (~994 MB) + published siblings
(~530 + 310 MB) into the shared HF cache; the Rosetta x86_64 build is slow
(~3–5× native) — expect a full run in the tens of minutes.
Exit code 0 means every pre-registered expectation held (see Output
contract); any drift aborts with a non-zero exit naming expected vs
observed.

`--census` additionally requires the paper's harness clone at
`.scratch/gguf-verifiability-artifact` (outside the repo; overridable via
`--harness-root`) and network access for the 8–16 MB ranged header reads —
it adds ~10 min.

`--b9222` additionally requires the three b9222 llama-quantize builds: the
smoke locates them in the `.scratch/llama.cpp-b9222*` build trees, or takes
an explicit `--b9222-builds-root` (same `quantize-*` layout as
`--builds-root`). Rebuild either set from a pinned commit with
[scripts/build_llamacpp.sh](scripts/build_llamacpp.sh) — it is idempotent
(existing binaries are object-verified, not rebuilt, unless `--force`),
asserts 0 FMA in the nocontract `ggml-quants.c.o` and 0 FMA + 0 ymm in the
x86 one (llvm-objdump), and applies the PR #25353 patch in a git worktree
so the stock tree stays pristine. It adds the trial-repo downloads
(~1.5 GB bf16 + 1.1 MB imatrix + 580 MB published Q4_K_M, sha256-pinned)
and ~10 min of quantization.

## Deviations from the paper

1. **Scale**: 0.5B (Qwen2.5-0.5B-Instruct) instead of the paper's 1.5B pilot
   — same family, same publisher (bartowski), laptop-friendly. The paper's
   b10448 is pinned; we use the 2026-10-20 llama.cpp tip `d8d92973`, with
   PR #25353 still open upstream (re-verified), applied as a file-scoped
   CMake patch identical to the PR.
2. **Single machine, three builds** instead of two machines: the
   x86_64-default variant runs under Rosetta 2 with FMA/AVX disabled
   (object-verified), which reproduces the paper's "baseline x86-64 default
   build is uncontracted" premise; the arm64-default vs arm64-nocontract
   pair is the paper's own §V single-machine proxy.
3. **No imatrix on the §V replication side** (C=∅), like the paper's §V
   cross-architecture experiment; the imatrix enters separately, as the
   published-vs-regenerated gap in the recipe-pinned step.
4. **Detection-floor positives are analytic** (dequantize with every payload
   code +1 to get exact per-weight steps; large edits via the two
   farthest-code planes, all-0 and all-15 payload codes), not materialized
   tampered files — likewise the harness EXP11 `--large` method. Budgets are
   per-tensor, AUC is Mann–Whitney with tie-averaged ranks, floor = smallest
   budget with auc_min ≥ 0.99.
5. **ggufpacker cross-check is a reimplementation** of the plan decision
   (zero-XOR payload → EXACT; congruent layout + zstd-12 delta ≤ 50% of the
   original → NEAR) on name-keyed tensor payloads, not a run of the tool
   itself: the tool on one machine with one binary always reproduces its
   own outputs EXACT, so the informative comparison is
   regenerated-vs-published.
6. **Recipe pinning uses the full published type map** via
   `--tensor-type-file` (the paper's "+rec" pins the deployed file's types
   wholesale); a diff-only override file leaves unlisted tensors to preset
   heuristics that changed across llama.cpp releases.

## Results

First full run 2026-10-20, Apple M3 Pro (arm64), llama.cpp `d8d92973`
(2026-10-20 tip; PR #25353 still open), fixtures pinned in `config.py`.
Full read in the #26 deep-dive comment; numbers (each pinned in
`config.py` and hard-asserted on every run — see Output contract):

**Hash matrix — 12/12 pre-registered expectations held:**

| pair | Q4_K_M | Q5_K_M | IQ4_XS | Q8_0 |
|---|---|---|---|---|
| arm64-default vs arm64-nocontract | differ | differ | differ | equal |
| x86_64-default (Rosetta) vs arm64-nocontract | equal | equal | equal | equal |
| x86_64-default (Rosetta) vs arm64-default | differ | differ | differ | equal |

Same-binary determinism held on every rerun. The paper's §V claims
(scale-search divergence, contraction-off cross-arch bit-identity, Q8_0
control, x86-baseline-is-uncontracted) all reproduce at 0.5B on one
machine via the Rosetta build.

**Localization** (paper §V.b): metadata/KV region byte-identical in every
divergent cell; divergence confined to k-quant tensors with the highest
per-byte rate in scale regions — e.g. Q4_K_M arm64-default vs nocontract:
β_byte = 8.1e-5 over 24 tensors; per-byte rates superscale 2.4e-5,
**min 1.0e-3, blockscales 1.3e-3** (highest), nibble 6.5e-5 (74% of
differing bytes by count) — the same shape as the paper's 1.5B numbers
(scales highest per-byte rate, nibbles hold most differing bytes). Both
properties are asserted: metadata identity in every divergent cell and
blockscales beta > nibble beta in the k-quant cells.

**Detection floor** (paper §VI): Q4_K honest pooled tail 0.021% (paper
1.5B: 0.018%). The budget ladder straddles the floor from below: AUC
0.73 at 0.002%, 0.88 at 0.005%, and 1.0 from the 0.01% budget up in all
5 seeds — floor **0.01%** at 0.5B vs the paper's 0.02% at 1.5B,
consistent with their divergence-grows-with-size trend (asserted as
floor ≤ 0.02 with the straddle sanity — AUC 1.0 at every budget ≥ the
floor, strictly below at the smallest budget). Natural-variance
percentiles match the paper closely: median |Δw|/σ 0.0133 (paper: 0.014),
p99 0.28 (paper: 0.26); the max statistic is 0.75 vs the paper's 2.36 —
a scale effect (fewer weights, smaller extreme draw), not a distribution
mismatch. Thread invariance (1 vs 8 threads) and same-binary determinism
(3 repeats) are asserted in-run and held everywhere.

**Sparse large edits** (paper §VI.d, the EXP11 `--large` mode): on the same
Q4_K pair, a single farthest-code edit pushes the max statistic to 3.09
(mean over 5 seeds; min 1.59) vs the honest max 0.75, and five edits to
4.26 (min 3.63) — above the honest max in every seed, matching the paper's
1.5B regime where the max statistic flags even one sparse farthest-code
edit. The exceeds-honest-max outcome is asserted per k.

**Card-alone** (paper §IV "card" column): 126/290 tensors of the
published Q4_K_M re-derive from card information alone (stock preset +
published f16; 32 type-mismatched) — same regime as the paper's trial
repos (140/335, 141/339): a minority of tensors match, no file
re-derives. The 126/32/290 counts are pinned per build.

**ggufpacker cross-check**: published Q8_0 XORs to zero against our
regeneration from all three environments — plan EXACT, confirming
"Q8_0 is already deterministic everywhere" (no scale search). Published
Q4_K_M is non-congruent with the stock preset (recipe gap: 32/290 tensors
carry a different per-tensor type than the stock Q4_K_M preset), and with
the full published type map pinned still differs in 14.8% of payload
bytes: the published file records `quantize.imatrix.*` (168 entries) but
the repo ships no imatrix — the paper's third card-absent coordinate (C
must be shipped and hash-bound). Contraction differences account for only
β_byte ≈ 8e-5 between our arm64 builds — the imatrix dominates the
published-vs-regenerated delta (asserted: Q8_0 EXACT from every build,
Q4_K_M non-congruent, recipe gap = 32, residual XOR ≥ 10% per build).
The tool's own test suite
(`tests/test_roundtrip.py`, our `quantize-default` binary via
`GGUFPACKER_TEST_LLAMA_QUANTIZE`) passes 14/14: EXACT round-trip
bit-exact, flipped payload bytes → NEAR with a zstd delta, unmatchable
recipe → BLOB — the plan semantics confirmed by the tool itself, not just
our reimplementation.

**Census re-checks** (paper harness `census.py`, 2026-10-20): live top-30
— 0/29 bit-exact verifiable, env/version gap 29/29; frozen 200-repo list
(`exp/EXP03` `frozen_repos_top200.txt`) — **0 verifiable, 148
source-reconstructible of 197 scored** (paper: 0/198, 148/198; three repos
now error out — two without a GGUF, one with an unreadable header — but
the source-reconstructible count is identical). The `--census` step re-runs
the frozen-list scoring in-smoke (`census_rescore` in the output JSON,
~10 min of ranged header reads) and fails loudly on drift from the pinned
0 / 148 counts and on env/version gap ≠ every scored repo.

**Bit-exact re-derivation at the card-named release (b9222)** — the
paper's §IV positive control, reproduced on
`bartowski/Qwen_Qwen3.5-0.8B-GGUF` (a trial repo that ships its bf16,
its imatrix, and names release b9222 on the card; pinned rev
`f36b1ea`, b9222 = `9a532ae4`). Two b9222 builds (arm64 with the
PR #25353 patch; x86_64 under Rosetta, baseline SSE2, 0 FMA
object-verified), each quantizing with the published imatrix:

| run | build | mode | tensors | bit-exact |
|---|---|---|---|---|
| 1 | arm64-nocontract | stock preset | 260/335 | no (75 type mismatches) |
| 2 | arm64-nocontract | full recipe | **335/335** | **yes** |
| 3 | x86_64-default | stock preset | 260/335 | no (75 type mismatches) |
| 4 | x86_64-default | full recipe | **335/335** | **yes** |
| 5 | arm64 default contraction | stock preset | 140/335 | no |

Paper Table III reproduced exactly (their Qwen3.5-0.8B row: 140/335 card,
260/335 arm64-contraction-off/x86 stock, 335/335 +recipe). The preset
runs fail only on recipe, not arithmetic — every same-type tensor XORs to
zero even with the wrong preset; the 75 mismatches are the hybrid
SSM/MTP tensors the stock preset types differently. The two recipe
outputs are byte-identical to each other and differ from the published
file only in the `quantize.imatrix.file` KV string (publisher's build
path vs our cache path) — the metadata quirk the paper's per-tensor
protocol excludes by design. Now part of the smoke itself (`--b9222`,
`b9222_table_iii` in the output); the original exploration scripts and
logs remain in `.scratch/rederive9222/` (not committed).

## Output contract

One JSON object on stdout: `params` (pins, repeats, thread variants),
`quantize_matrix` (per build × type: sha256, bytes, seconds),
`hash_matrix_cells` (pre-registered expectation checks),
`structured_diffs` (per divergent cell: metadata identity, per-type beta,
per-region share/beta), `detection_floor` (honest tails incl.
natural-variance percentiles, per-budget AUC over a straddling ladder,
floor, `large_edits` per-k max-statistic vs the honest max),
`ggufpacker_crosscheck` (per published sibling × build: nonzero
XOR bytes, zstd-12 delta size, plan), `recipe_pinned_rederive` (KV
imatrix record, recipe gap size, per-build XOR vs published),
`card_alone_match_counts` (per build: matched / type-mismatched / total
tensors vs the published Q4_K_M), and — only with the respective flags —
`census_rescore` (n scored / errors, bit-exact verifiable,
source-reconstructible, env-version gap counts vs the pinned
expectations) and `b9222_table_iii` (pins, the five Table III cells with
matched / type-mismatched / nonzero-XOR vs the pre-registered numbers,
recipe whole-file digests, all-expectations-hold verdict).

Every expectation is **hard-asserted, not merely recorded**: a zero exit
code means every pre-registered pin below held on this run; any drift
raises `SystemExit` (exit 1) naming the expected vs observed values.

| pin (`config.py`) | value | kind |
|---|---|---|
| `EXPECTED_EQUAL` | the 3 × 4 hash-matrix table above | exact (discrete booleans) |
| `CARD_ALONE_EXPECTED` | 126 matched / 32 type-mismatched / 290 total per build | exact (discrete counts) |
| `RECIPE_GAP_EXPECTED` | 32 tensors differ from the stock preset | exact (discrete count) |
| `IMATRIX_GAP_MIN_FRAC` | recipe-pinned residual XOR ≥ 0.10 per build | relation bound (imatrix dominates; measured ≈ 0.1483) |
| `FLOOR_MAX` | detection floor ≤ 0.02 (paper's 0.02% Q4_K floor at 1.5B) | relation bound (measured 0.0001) |
| `CENSUS_EXPECTED` | 200 repos / 0 bit-exact / 148 source-reconstructible, plus env-version gap = every scored repo | exact (discrete counts) |
| `B9222_TABLE_III_EXPECTED` | the five Table III cells (140/260/335 of 335) | exact (discrete counts) |

Also asserted structurally: same-binary determinism over reruns,
thread-count invariance, every divergent cell's KV/metadata region
byte-identical with blockscale-per-byte divergence above the nibble rate
(k-quant cells), Q8_0 plan EXACT vs the published file from every build,
published Q4_K_M non-congruent with the stock preset, AUC 1.0 at every
budget ≥ the floor with the smallest budget strictly below it (the
ladder straddles the floor), and every sparse large-edit k exceeding the
honest max statistic in every seed.
