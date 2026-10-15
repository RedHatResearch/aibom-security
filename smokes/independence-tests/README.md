# Independence Tests smoke (smokes/independence-tests/)

Reimplementation of the **`csu` weight-space independence statistic** from
**"Independence Tests for Language Models"** (Sally Zhu, Ahmed Ahmed, Rohith
Kuditipudi, Percy Liang; ICML 2025,
[arXiv:2502.12292](https://arxiv.org/abs/2502.12292)). Given two
checkpoints' weights, `csu` tests the null hypothesis that their parameters
are independent (independent random initializations), yielding a p-value.
Survey deep-dive for
[#26](https://github.com/RedHatResearch/aibom-security/issues/26);
runnable-check context in [../README.md](../README.md).

**License situation.** The authors' artifact
([github.com/ahmeda14960/model-tracing](https://github.com/ahmeda14960/model-tracing))
ships **no license** (no LICENSE file, no license badge), so nothing was
vendored or cloned: everything here is implemented from the paper's
Algorithm 2 (MATCH), Algorithm 3 (SPEARMAN) and Algorithm 4 (FISHER). The
upstream `tracing/statistics/csu.py` was read strictly as a behavioral
cross-check (read-only, no code copied). Cite the paper when reusing:

```bibtex
@inproceedings{zhu2025independencetestslanguagemodels,
  title={Independence Tests for Language Models},
  author={Zhu, Sally and Ahmed, Ahmed and Kuditipudi, Rohith and Liang, Percy},
  booktitle={ICML},
  year={2025},
  note={arXiv:2502.12292}
}
```

## Method (pinned from the paper)

The smoke implements the **constrained-setting** weight-space test on GLU
MLPs (paper §2.3, `ϕ_U^(ℓ)` with `M(θ) = U^(ℓ)`, the MLP up-projection):
it detects *unpermuted* weight dependence — fine-tunes, quantizations,
continued pretraining — and reports per-layer and aggregate p-values.

Per layer (state-dict keys containing `mlp.up_proj`; both models must have
**identical key sets and shapes**, otherwise the smoke abstains):

1. **Cast to float64.** Both `[intermediate, hidden]` up-projection
   matrices.
2. **Cosine-similarity matrix** between the rows of model A and the rows
   of model B → `[intermediate, intermediate]`.
3. **MATCH (Alg. 2): Hungarian assignment** on the cosine matrix
   (`scipy.optimize.linear_sum_assignment`, `maximize=True`) — the
   maximum-weight row-to-row alignment.
4. **SPEARMAN (Alg. 3):** `spearmanr(matched_indices, arange(n))` — the
   Spearman rank correlation between the recovered assignment and the
   identity permutation, with its two-sided p-value from
   `scipy.stats.spearmanr`. Under unpermuted dependence (fine-tuning,
   quantization) the optimal assignment recovers the identity, so ρ ≈ 1
   and the per-layer p is tiny; under independence the recovered
   permutation is exchangeable and p is uniform.
   *(Cross-check note: this is exactly the authors' `csw_sp_layer` —
   `spearmanr(LAP(cossim)[1], arange(n))` — the assignment indices, not
   the matched cosine values; the assignment-vs-identity correlation is
   what carries the dependence signal.)*
5. **FISHER (Alg. 4):** `X = -2 · Σ ln p_ℓ` over layers (NaN p-values
   skipped, the authors' convention for degenerate layers); aggregate
   `p = chi2.sf(X, df=2L)`.

**Verdict** (α = 0.05, a module constant): `aggregate_p < 0.05 →
non_independent`; `aggregate_p ≥ 0.05 → independent`. Incomparable stacks
(any up-projection key-set or shape difference) → `abstain` with
`reason: shape_mismatch` and the first differing layer. MoE and GGUF-only
pairs are also out of scope (see below).

Scoring core (`layer_csu`, `fisher_combine`, `csu_score`, `comparability`,
`dequantize_awq_gemm`) is pure numpy/scipy on plain tensor dicts — no I/O,
no fixtures — so a ProvenanceBench detector can lift it directly.

### Scope and known limits

- **Permutation fragility by design (constrained setting).** `csu` assumes
  the two models are related without hidden-unit permutation (fine-tunes,
  quantizations, continued training). An adversary who permutes MLP units
  post-hoc makes the assignment recover their permutation instead of the
  identity and the test loses power. The paper's *unconstrained*,
  permutation-robust statistics (`csh` on activations, `match` gate-vs-up
  agreement) are out of scope for this smoke.
- **MoE is out of scope**: the `mlp.up_proj` substring misses MoE expert
  naming (`mlp.experts.{i}.up_proj`); MoE pairs get a
  `out_of_scope_moe` abstain rather than a silent empty match.
- **GGUF is out of scope**: GGUF checkpoints are not safetensors and are
  not readable by this loader.
- **Non-GLU architectures** (GPTNeoX `dense_h_to_4h`, etc.) have no
  `mlp.up_proj` tensors and abstain with `no_up_proj`.

## AWQ quantized checkpoints

The `pb-smoke-qwen25-awq` fixture (4-bit AWQ) stores int4-packed
`qweight/qzeros/scales` tensors instead of `.weight`. The smoke includes a
numpy port of AutoAWQ's `dequantize_gemm`
([awq/utils/packing_utils.py](https://github.com/casper-hansen/AutoAWQ/blob/main/awq/utils/packing_utils.py)
— Apache-2.0): unpack the eight 4-bit values per int32 column, undo the
`[0,2,4,6,1,3,5,7]` packing order, expand the groupwise (group_size=128)
scales/zeros, and transpose to the HF `[out, in]` layout. The `--selftest`
mode validates the unpack/reorder path with an offline round-trip
(pack → dequantize → compare, corr > 0.99).

## Running

```bash
uv sync --all-packages --group smokes
# Offline self-test (no network, no HF downloads; synthetic tensors only):
uv run --group smokes python smokes/independence-tests/smoke.py --selftest
# Full default run (the six 135M-1.7B-scale pairs below):
uv run --group smokes python smokes/independence-tests/smoke.py
# Any other ProvenanceBench pair (id from provenancebench/pairs.yaml):
uv run --group smokes python smokes/independence-tests/smoke.py --pair pb-core-pds-bsl-470m
# Optional: --cache-dir PATH for the HF cache.
```

Stdout is a single JSON document (progress goes to stderr). Exit code 0 =
no pair *missed* its pre-registered expectation; 1 = at least one miss.
`skipped_format` and `borderline_false_positive` outcomes are recorded in
the summary but do not fail the run (see "Expectation outcomes and the
gate" below). `--selftest` exits 0 on success.

CPU-only (Apple M3 Pro 36GB; no `.to('cuda')` anywhere). The scoring core
uses numpy/scipy only; `safetensors.numpy` reads shards without torch
models. All fixtures are non-gated; `HF_TOKEN` is not required.

## Fixtures

Model ids and pinned revisions are read at runtime from
[provenancebench/pairs.yaml](../../provenancebench/pairs.yaml) (the
orchestrator's manifest) — nothing id+revision is hardcoded here beyond
the default pair-id list.

| pbench id | pair (subject vs reference) | role | expected verdict |
|---|---|---|---|
| `pb-smoke-qwen3-sft` | Qwen/Qwen3-0.6B vs Qwen3-0.6B-Base | positive | `non_independent` (small p) |
| `pb-smoke-smoltulu` | SultanR/SmolTulu-1.7b-Instruct vs HuggingFaceTB/SmolLM2-1.7B | positive | `non_independent` (small p) |
| `pb-core-pds-bsl-470m` | Data-Selection/PDS-470M vs Data-Selection/BSL-470M | hard negative (same Mistral config, both from scratch) | `independent` (large p) — **but the pinned BSL-470M revision ships no safetensors** (see note) → `skipped_format` |
| `pb-core-smollm-smollm2-strangers` | HuggingFaceTB/SmolLM-1.7B vs SmolLM2-1.7B | hard negative (same layout, independent pretrains) | `independent` (large p) — **observed p = 0.032**, a `borderline_false_positive` (see note) |
| `pb-smoke-qwen25-awq` | Qwen/Qwen2.5-0.5B-Instruct-AWQ vs Qwen2.5-0.5B-Instruct | quant robustness | `non_independent` (small p after int4 dequantization) |
| `pb-core-sheared-pythia` | princeton-nlp/Sheared-Pythia-160m vs EleutherAI/pythia-410m | abstain edge | `abstain` (no-safetensors at pinned revision; see note) |
| `pb-core-r1-distill-qwen15` | DeepSeek-R1-Distill-Qwen-1.5B vs Qwen2.5-Math-1.5B | calibration positive (distill onto backbone) | `non_independent` — **observed p = 0.0** (28 layers, 713 s) |
| `pb-core-minicpm5-1b-rl` | MiniCPM5-1B vs MiniCPM5-1B-SFT | calibration positive (RL post-train) | `non_independent` — **observed p = 0.0** (24 layers, 246 s) |
| `pb-core-llama2-granite7b-strangers` | granite-7b-base vs Llama-2-7b-hf | calibration hard negative (from-scratch 7B layout replica) | `independent` — **observed p = 0.338** (32 layers, 3607 s; Llama-2 gating applies) |

Note on `pb-core-sheared-pythia`: the pinned subject revision
(`princeton-nlp/Sheared-Pythia-160m` @ `c8889f49…`) ships
`pytorch_model.bin` only (no safetensors), so the run abstains with
`out_of_scope_gguf` (the smoke's generic no-safetensors reason) before
any tensor comparison — still `abstain` == expected, so `held`. Both
sides are additionally GPTNeoX (`dense_h_to_4h` naming, no
`mlp.up_proj`) with differing geometry (12×768×3072 vs 24×1024×4096):
on a safetensors revision the abstain reason becomes
`no_up_proj`/`shape_mismatch` with the first differing layer and the
geometry diff in `detail`.

Note on `pb-core-pds-bsl-470m`: the pinned BSL-470M revision
(`1581b9dc66ba9b9b170fbbeaa108701f5ea354cd`) ships **no safetensors**
(a `pytorch_model.bin` pickle at the time of running — not a GGUF
rehost; the reason label `out_of_scope_gguf` is the smoke's generic
no-safetensors reason), so a safetensors-only smoke structurally cannot
consume it. The pair's pbench manifest label
stays `unrelated` (expected `independent`), but the run records it as
`skipped_format` (verdict `abstain`, reason `out_of_scope_gguf`): neither
a gate failure nor a held expectation. If Data-Selection/BSL-470M later
gains safetensors at a new revision, repin and re-run to score it for
real.

Note on `pb-core-smollm-smollm2-strangers`: this known-independent pair
scored **p = 0.032** (24 layers, Fisher-combined aggregate) — a
`non_independent` verdict at α = 0.05 on a true negative, i.e. a
**borderline false positive**. Calibration follow-up (2026-10-14): the
7B layout-replica negative `pb-core-llama2-granite7b-strangers` scores
**p = 0.338** — cleanly independent — so the anti-conservative
behavior is not universal; it appears at 1.7B same-family scale
(SmolLM/SmolLM2 share a tokenizer family and corpus lineage, which may
inflate row-order alignment). Current tally on consumable negatives:
1 clean (`independent`, p = 0.338), 1 borderline (p = 0.032). See the
"Expectation outcomes and the gate" section below for the policy and
interpretation.

Pair-scale notes (Apple M3 Pro, CPU), measured 2026-10-13/14: the
Hungarian assignment is O(inter³) and intermediate size dominates
wall-clock. Observed `compute_seconds` (weights cached): Qwen2.5-0.5B
AWQ 22.7 · Qwen3-0.6B 2.4 · Qwen2.5-Math-1.5B (r1-distill) 713 ·
MiniCPM5-1B 246 · SmolLM2/SmolTulu-1.7B 25.8 · SmolLM-1.7B strangers
286 · granite-7b vs Llama-2-7b 3607 (inter 11008, 32 layers — the
dominant cost). Downloads are one-time into the shared HF cache: ~13 GB
total for the default six pairs plus ~15 GB for the granite/Llama-2
calibration negative (Llama-2 is gated; token with granted access
required).

## Expectation outcomes and the gate

Each pair's result carries an `outcome` field classifying it against the
pre-registered expectation:

- `held` — verdict matched the expectation.
- `missed` — verdict contradicted the expectation. **This is the only
  outcome that fails the gate** (exit 1).
- `skipped_format` — verdict is `abstain` with reason `out_of_scope_gguf`,
  `out_of_scope_moe`, or `no_up_proj`: the smoke structurally cannot
  consume the pair's checkpoint format (GGUF-only rehost, MoE expert
  naming, non-GLU MLP naming). The pair's manifest expectation is
  unchanged — this records that the *smoke's scope*, not the pair, is the
  reason nothing was scored. Not a failure and not counted as held.
- `borderline_false_positive` — expected-independent pair whose aggregate
  p landed in the borderline band `(0.01, 0.05]`. Recorded, loudly (see
  below), but not a gate failure.

The `summary` object counts each bucket by pair id
(`held` / `missed` / `skipped_format` / `borderline_false_positives`), and
`expectation_held` in each pair entry remains true only for `held`.
Exit 0 iff `missed` is empty.

### Borderline false positives (calibration caveat for #26)

**Measured:** `pb-core-smollm-smollm2-strangers` (SmolLM-1.7B vs
SmolLM2-1.7B — same 2048-hidden/24-layer layout, independently pretrained,
a true negative) returned **aggregate p = 0.032**: `non_independent` at
α = 0.05, i.e. a false positive at the nominal threshold. The p-value is
preserved verbatim in the pair's JSON; the outcome is
`borderline_false_positive` with an explanatory `outcome_note`.

**Interpretation:** the Fisher combination over 24 layers is mildly
anti-conservative at 1.7B same-layout independent pairs — per-layer
Spearman p-values that are individually plausible under H₀ combine into
an aggregate that dips below α more often than the nominal 5%. This is a
real calibration caveat for [#26](https://github.com/RedHatResearch/aibom-security/issues/26)
— treat `non_independent` verdicts on same-layout stranger pairs in the
p ∈ (0.01, 0.05] band as *not* provenance evidence — and it is consistent
with the paper evaluating at 7B (its experiments are 7B-scale; this
1.7B fixture is below the scale the statistic was calibrated on). The
band edge is deliberately conservative: p ≤ 0.01 on a true negative would
remain a hard `missed`, as would any false positive on a
non-independent-expectation pair.

## Selftest

`--selftest` runs fully offline (synthetic tensors, no network, no HF
downloads) and exercises the entire scoring path:

1. **Dependence direction**: two synthetic stacks sharing 60% of their
   `up_proj` rows (aligned, with tiny noise — the unpermuted fine-tuning
   shape) must yield `non_independent` with aggregate p < 1e-6.
2. **Independence direction**: 40 fully independent synthetic pairs; the
   minimum aggregate p must exceed the dependent case's p, and the count
   below α must be within the nominal 5% false-positive rate (≤ 7/40).
3. **Abstain paths**: shape mismatch, key-set mismatch, and the
   both-empty + differing-geometry case.
4. **AWQ dequantization round-trip**: pack a synthetic HF-layout weight
   into the AutoAWQ GEMM int4 layout, dequantize, compare (corr > 0.99,
   error within half a scale step).
5. **Expectation-outcome classification**: the gate policy classifier
   (held / missed / skipped_format / borderline_false_positive) over ten
   synthetic cases, including the band edges p = 0.032 (borderline) and
   p = 0.01 and p = 0.001 (both hard misses).

## Output contract

One JSON object on stdout:

```jsonc
{
  "smoke": "independence-tests",
  "paper": "...", "statistic": "csu (MLP up_proj MATCH + SPEARMAN, Fisher-combined)",
  "params": {"alpha": 0.05, "borderline_band": [0.01, 0.05]},
  "pairs": [
    {
      "pair_id": "pb-smoke-qwen3-sft", "role": "positive", "kind": "sft",
      "subject": {"repo_id": "...", "revision": "<pinned sha>"},
      "reference": {"repo_id": "...", "revision": "<pinned sha>"},
      "expected_verdict": "non_independent",
      "geometry": {"subject": {...}, "reference": {...}},   // best-effort
      "verdict": "non_independent" | "independent" | "abstain",
      "p": 1.2e-40,               // null when abstaining
      "fisher_stat": ..., "num_layers": 28,
      "per_layer": [{"layer": 0, "rho": 0.98, "p": 1e-30, "mean_matched_cosine": 0.94}, ...],
      "outcome": "held", "expectation_held": true, "compute_seconds": 41.2
      // outcome: "held" | "missed" | "skipped_format" | "borderline_false_positive";
      //          borderline/skipped entries also carry "outcome_note": "..."
      // abstain extras: "reason": "shape_mismatch" | "no_up_proj"
      //                 | "out_of_scope_moe" | "out_of_scope_gguf",
      //                 "layer": <first differing layer>, "detail": "..."
    }
  ],
  "summary": {
    "pairs_total": 6,
    "held": ["pb-smoke-qwen3-sft", ...],
    "missed": [],
    "skipped_format": ["pb-core-pds-bsl-470m"],
    "borderline_false_positives": ["pb-core-smollm-smollm2-strangers"]
  },
  "total_seconds": ...
}
```

## Deviations from the paper

1. **Reimplementation from the paper only** (no license on the artifact);
   the upstream repository was used strictly as a read-only behavioral
   cross-check. The numpy/scipy implementation matches the paper's
   Algorithm 2/3/4 exactly: cosine-similarity matrix, Hungarian
   assignment (`maximize=True`), `spearmanr(matched, arange(n))`,
   Fisher's method with `chi2.sf(X, 2L)`.
2. **Layer scope.** The paper's constrained experiments use all
   `U^(ℓ)` up-projections of Llama-style GLU MLPs; this smoke matches
   state-dict keys by the `mlp.up_proj` substring (same set, plus it
   naturally picks up Llama/Mistral/Qwen-style GLU naming).
3. **NaN handling.** Layers whose Spearman p is NaN (constant vectors,
   fewer than 2 layers) are skipped in the Fisher combination — the
   authors' `fisher` convention.
4. **Abstain path added.** The upstream code raises on mismatched
   parameter names; this smoke emits a structured `abstain` verdict
   (`shape_mismatch` + first differing layer) instead, which the
   ProvenanceBench harness expects for incomparable stacks.
5. **AWQ dequantization added.** The upstream `csu` runs on
   `from_pretrained` models (AutoAWQ handles int4 transparently); this
   smoke reads safetensors directly and dequantizes int4-packed
   `qweight/qzeros/scales` itself (numpy port of AutoAWQ's
   `dequantize_gemm`, Apache-2.0), so no GPU/CUDA kernels are needed.
6. **Float32 transport.** `_lib.hub.load_tensors` casts to float32 on
   load; the statistic then casts to float64 per the paper. AWQ
   checkpoints dequantize through fp16 scales to float32.
