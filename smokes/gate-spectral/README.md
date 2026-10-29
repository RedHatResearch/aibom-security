# GATE spectral fingerprint smoke (smokes/gate-spectral/)

Weight-level reimplementation of the **GATE spectral fingerprint** from
**"Model-intrinsic, Weight-level, and Training-free Spectral Fingerprinting
for Large Language Models"** (Yanan He, Yanli Chen, Ling Xiong, Gejian Zhao,
Hanzhou Wu; Guizhou Normal University / BLCU / Shanghai University;
[arXiv:2610.30626v1](https://arxiv.org/abs/2610.30626), 22 Oct 2026).
Given two checkpoints, the method computes one spectral-entropy scalar per
gate projection, pools the depth profile into eight numbers, and decides
whether the checkpoints are related by a linear-CKA threshold. It needs no
forward passes, inputs, or outputs — a static weight check for
provenance/lineage evidence. Survey deep-dive for
[#26](https://github.com/RedHatResearch/aibom-security/issues/26);
runnable-check context in [../README.md](../README.md).

**License situation.** No author code was released (paper only, 5 pages).
Everything here is implemented from the paper's equations (Eq. 2–6);
nothing was vendored. `gguf`/`gguf-py` is MIT; the AWQ dequantizer is a
numpy port of AutoAWQ's `dequantize_gemm` (Apache-2.0) lifted verbatim from
[`smokes/independence-tests`](../independence-tests/smoke.py). Cite the paper
when reusing:

```bibtex
@article{he2026gate,
  title  = {Model-intrinsic, Weight-level, and Training-free Spectral
            Fingerprinting for Large Language Models},
  author = {Yanan He and Yanli Chen and Ling Xiong and Gejian Zhao and
            Hanzhou Wu},
  journal = {arXiv preprint arXiv:2610.30626},
  year   = {2026}
}
```

## What the paper claims

- **Population.** 29 pairs: 9 related (Qwen2.5-7B family, Mistral-7B→Instruct,
  Baichuan2, Llama-3.1-8B→Hermes-3/Aloe-Beta), 6 robust (pruning, GPTQ
  Int4/Int8, distillation), 14 unrelated (cross-family 7B-class, GPT-2-Large).
  Related min 0.9727, robust min 0.9108 (GPTQ Int8; Int4 0.9256), unrelated
  max 0.2288, separation gap 0.7439. Expanded set: 120 pairs / 25 checkpoints,
  AUROC 0.9870 at r=64; τ=0.9 gives TPR 90.00%, FPR 4.44%.
- **Defaults.** r=128, s=8. The rank ablation is non-monotonic
  (AUROC 0.9830@32, 0.9870@64, 0.9781@128, 0.9448@256); s=16 is fragile
  (robust-min collapses to 0.0318). τ=0.9 is an empirical operating point,
  not a tuned decision boundary.
- **Adversarial.** Targeted rank-4 perturbations at a 2 % relative Frobenius
  budget drop CKA to 0.6519 at +12.41 % held-out perplexity; a 3 %
  utility-constrained attack lands at 0.8910 (+3.08 % PPL).
- **Caveat that matters here.** Every paper negative differs in *both* gate
  dimensions and data regime; no same-architecture independently pretrained
  pair is tested. Our fixtures include such strangers, so a false positive on
  them is a population-fragility finding, not an implementation bug.

## Method (pinned from the paper)

For each gate-like FFN tensor `W_gate ∈ R^{d_m×d_f}` (the branch whose output
passes the activation before the element-wise product with UP; Eq. 1):

```text
q      = min(d_m, d_f)                      # paper's available rank
r_eff  = min(r, q)                          # default r = 128
sigma  = svd(W_gate)                        # descending singular values
sigma_bar_i = sigma_i / (sum_{j<=r_eff} sigma_j + eps)      # Eq. (2)
H      = -sum_i sigma_bar_i * ln(sigma_bar_i)                # Eq. (3)
f      = exp(H)                                              # Eq. (4)
p_k    = mean(f^(l) for l in segment_k)                      # Eq. (5)
CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)      # Eq. (6)
```

Implementation choices (all contract-pinned):

1. **SVD input/arithmetic.** fp32 SVD via deterministic
   `numpy.linalg.svd(compute_uv=False)` (full LAPACK; no randomized SVD, no
   seed), fp64 for normalization and entropy. Singular values are invariant
   to transpose, so the stored `[out, in]` orientation is irrelevant.
2. **Normalization.** Eq. (2) sums over the retained top-`r_eff` values only
   — not all q. `eps = 1e-12` enters the denominator only; the paper's
   `log(sigma_bar + eps)` is redundant for positive singular values and is
   omitted. Exact zero singular values contribute nothing (`lim x ln x = 0`),
   so rank-deficient matrices stay finite.
3. **All-zero guard.** `W = 0` yields `H = 0`, `f = 1`, flagged as `all_zero`
   — never NaN/inf.
4. **Layer scalar.** Arithmetic mean of the per-tensor `exp(H)` values. On a
   dense model there is one tensor per layer, so this degenerates exactly to
   the paper. The MoE extension is disclosed below.
5. **Depth order.** Integer layer index, never lexicographic. A layer's
   tensors are sorted by name for determinism.
6. **Pooling.** `numpy.array_split` semantics: s contiguous groups, sizes
   differ by at most one, the first `L mod s` groups get the extra layer.
   L=26 → `[4,4,3,3,3,3,3,3]`; the paper's prune-6 case (L=26, 0.9198) shows
   the placement is not load-bearing.
7. **Similarity.** Linear CKA on the two 8-vectors equals the squared Pearson
   correlation of the mean-centered vectors. It is sign-insensitive by
   construction; `signed_pearson` is always reported separately.
8. **Verdict.** Related iff `CKA >= 0.9` at r=128, s=8. No label-dependent
   tuning anywhere. Abstain reasons: `no_gate_tensors`, `format_unreadable`,
   `degenerate_profile` (constant pooled vector → never NaN).

Diagnostics per model: L, per-layer tensor counts and r_eff range,
per-tensor `r_eff` and `spectrum_fraction = r_eff/q`, pooled 8-vector,
`profile_cv = std/mean`.
Per pair: CKA, signed Pearson, `anticorrelation_collision`
(`r <= -0.7` while `CKA >= 0.9`), `structure_matched` (all q equal across
both sides), `layers_per_segment`, `low_signal_profile` (`profile_cv < 0.02`),
and the sensitivity panel below.

## MoE extension (ours; not in the paper)

> **MoE extension (ours; not in the paper).** He et al. (arXiv:2610.30626) define the GATE spectral feature only for dense decoder-only models with gated FFNs — one gate projection per layer — and do not evaluate MoE checkpoints. ProvenanceBench's MoE cohort requires a rule. We extend the feature as follows: for each layer, every gate-like FFN tensor (routed-expert gates, the shared-expert gate where present, and the dense FFN gate for dense layers inside MoE models; router/logit matrices excluded) contributes the paper's per-tensor statistic — top-min(r, q) singular values normalized over the retained set, spectral entropy, exp(H) — and the layer scalar is the arithmetic mean of these values. Segmented pooling (s=8), linear CKA, and the τ=0.9 operating point are unchanged. On dense models this degenerates exactly to the paper's definition. The arithmetic mean is chosen because it is exactly invariant to expert duplication (naive upcycling), unbiased under random expert subsetting (expectation over uniformly drawn subsets), and introduces no expert-count-dependent term; shared-only, expert-only, geometric-mean, and concatenated-spectrum variants are recorded in diagnostics for every MoE pair. This extension is a design choice of this reproduction and is untested by the paper's authors; MoE results should be read with that in mind.

Routers are excluded in every family — a router maps `d_m -> n_experts` and
never passes an activation before the UP product. The five name collisions
handled explicitly: `mlp.gate.weight` (Qwen3-MoE/OLMoE),
`block_sparse_moe.gate.weight` (Mixtral), `block_sparse_moe.router.*`
(granite), `blocks.N.ffn.router.*` (upcycle), `mlp.router.*` (gpt-oss), plus
LFM2's `feed_forward.gate.weight` and the GGUF `blk.N.ffn_gate_inp.weight`.

Sensitivity variants (computed for every pair, **never used for the
verdict**): r ∈ {64, 256} recompute the entropy feature; s ∈ {4, 16} repool
the primary profile; `experts_only` / `shared_only` restrict the tensor set
by role; `geometric` replaces the layer arithmetic mean with the geometric
mean of `exp(H)`, and `concatenated` renormalizes the layer's concatenated
top-`r_eff` spectra together before taking the entropy (a diversity probe).
`rank_sensitive` / `moe_rule_sensitive` flags mark any variant
whose verdict differs from the primary one.

## Fixtures

Model ids and pinned revisions come from
[provenancebench/pairs.yaml](../../provenancebench/pairs.yaml) at runtime —
only the pair-id list is hardcoded here. Nine smoke-cohort pairs, all
≤ 1.7B parameters:

| pbench id | subject vs reference | stack | kind | expected |
|---|---|---|---|---|
| `pb-smoke-qwen25-awq` | Qwen2.5-0.5B-Instruct-AWQ vs Qwen2.5-0.5B-Instruct | dense | quant | related |
| `pb-smoke-qwen25-fp8` | Qwen2.5-0.5B-Instruct-FP8-dynamic vs Qwen2.5-0.5B-Instruct | dense | quant | related |
| `pb-smoke-smoltulu` | SmolTulu-1.7b-Instruct vs SmolLM2-1.7B | dense | sft | related |
| `pb-smoke-qwen3-sft` | Qwen3-0.6B vs Qwen3-0.6B-Base | dense | sft | related |
| `pb-smoke-llama-smol-cross` | Llama-3.2-1B vs SmolLM2-1.7B | dense | neg_cross | unrelated |
| `pb-smoke-granite-instruct` | granite-3.1-1b-a400m-instruct vs …-base | moe | sft | related |
| `pb-smoke-granite-gguf` | QuantFactory granite-3.1-1b-a400m-base-GGUF vs …-base | moe | quant_gguf_rehost | related |
| `pb-smoke-granite-smol-cross` | granite-3.1-1b-a400m-base vs SmolLM2-1.7B | moe vs dense | neg_cross | unrelated |
| `pb-smoke-qwen3-upcycle` | moe-600m-qwen3-upcycle vs Qwen3-0.6B-Base | moe | upcycle_naive | related |

The `kind` column is the literal `kind` from `pairs.yaml`. The two `quant`
rows are both a quantized subject against its unquantized reference — AWQ
int4 for `pb-smoke-qwen25-awq`, FP8-dynamic for `pb-smoke-qwen25-fp8`; the
format detail lives here rather than in the table.

Extraction patterns exercised by members of this cohort: dense
`mlp.gate_proj`; granite's fused `block_sparse_moe.input_linear`
`[32,1024,1024]` split at 512 rows (`chunk(2)[0]`); granite GGUF
`blk.N.ffn_gate_exps.weight`; the upcycle model's `blocks.N.ffn.gate`
(stacked routed), `shared_expert.gate.weight`, and dense
`blocks.0.ffn.gate.weight` under `step_<cfg.step>/model.safetensors`. The
loader also implements Mixtral/llm-jp `experts.M.w1` (where
`block_sparse_moe.gate` is the router), OLMoE/Qwen3-MoE discrete experts,
Qwen3.5's nested `model.language_model.layers.N.mlp.gate_proj.weight`
(anchored so the `mtp.layers.*` speculative-decoding head never matches),
gpt-oss even-column de-interleave plus MXFP4 `gate_up_proj_blocks` /
`_scales` (`dequantize_mxfp4`), LFM2 fused (`experts.gate_up_proj`) and
discrete (`feed_forward.experts.M.w1`) layouts, tinymixtral's stacked
`layers.N.moe.gate_proj` read from `pytorch_model.bin`, and the GPT-2
`c_proj` fallback.

Two further families are **non-gated** and use a disclosed down-projection
fallback (†, extending the paper's GPT-2 `c_proj` footnote): GPTNeoX
`gpt_neox.layers.N.mlp.dense_4h_to_h.weight` (`dense_h_to_4h` is the UP
branch and is deliberately never extracted) and Nemotron-H
`backbone.layers.N.mixer.down_proj.weight` (relu2 MLP, no gate). The paper
only tested the GPT-2 `c_proj` fallback, so every GPTNeoX/Nemotron-H use is
disclosed per model in diagnostics as an out-of-paper substitution.

`provenancebench/detect.py` inlines the same audited core (kept semantically
identical; edit both when changing extraction).

Pre-registered interpretations (fixed before the first run):

- Related pairs must be `related`; unrelated pairs must be `unrelated`.
- A cross-family negative scoring ≥ τ is a documented population-fragility
  finding, never a reason to move τ.
- Quant pairs in the 0.85–0.95 band are an honest measurement of the paper's
  thin margin (its own GPTQ-Int4 point is 0.9256, only 0.0256 above τ).
- The partition-upcycle pair may legitimately land below τ: slicing one
  3072-neuron MLP into 6×384 routed + 768 shared experts and then 10B tokens
  of continued pretraining exceeds the paper's tested transforms. The
  `experts_only` / `shared_only` variants attribute which tensor set moved.

## Running

```bash
# the nine-pair smoke cohort (cold run downloads ~20–24 GB at full-checkpoint
# sizes into the shared HF cache; less when Range reads fetch only gates)
uv run --group smokes python smokes/gate-spectral/smoke.py

# one pair ad hoc (ids from provenancebench/pairs.yaml)
uv run --group smokes python smokes/gate-spectral/smoke.py --pair pb-smoke-granite-gguf

# offline property tests, no network
uv run --group smokes python smokes/gate-spectral/smoke.py --selftest

# optional cache location
uv run --group smokes python smokes/gate-spectral/smoke.py --cache-dir /path/to/hf-cache
```

Stdout is one JSON document (progress goes to stderr). Exit 0 iff no pair
*missed* its pre-registered expectation; exit 1 on at least one miss.
`skipped_format` (a structural format abstain: no gate tensors or an
unreadable/unsupported checkpoint) is recorded in the summary but does not
fail the run. `--selftest` exits 0 on success. CPU-only (Apple M3 Pro 36 GB,
no CUDA); numpy/gguf/ml_dtypes/huggingface_hub/pyyaml, with `torch` imported
lazily and only when a `pytorch_model.bin` checkpoint is read.

## Cost

The granite GGUF pair's file choice and its pinning caveat are under
Deviations (#4); the loader records the chosen filename and every tensor's
quant type in diagnostics. All pairs reuse the shared HF cache; the unique
download for the nine-pair smoke cohort is roughly 20–24 GB at
full-checkpoint sizes, less when Range reads fetch only the gate tensors
(GGUF pairs stay full-file).

| pair | checkpoints downloaded | approx. |
|---|---|---|
| `pb-smoke-qwen25-awq` | Qwen2.5-0.5B-Instruct-AWQ + Qwen2.5-0.5B-Instruct | ~2 GB (int4 subject is smaller) |
| `pb-smoke-qwen25-fp8` | FP8-dynamic rehost + Qwen2.5-0.5B-Instruct | ~2 GB |
| `pb-smoke-smoltulu` | SmolTulu-1.7b + SmolLM2-1.7B | ~6.6 GB |
| `pb-smoke-qwen3-sft` | Qwen3-0.6B + Qwen3-0.6B-Base | ~2.4 GB |
| `pb-smoke-llama-smol-cross` | Llama-3.2-1B + SmolLM2-1.7B | ~5.8 GB (Llama is gated) |
| `pb-smoke-granite-instruct` | granite-3.1-1b-a400m instruct + base | ~5.4 GB |
| `pb-smoke-granite-gguf` | granite GGUF Q8_0 (~1.4 GB) + granite base | ~4.1 GB |
| `pb-smoke-granite-smol-cross` | granite base + SmolLM2-1.7B | ~6.0 GB |
| `pb-smoke-qwen3-upcycle` | upcycle fp32 (~2.4 GB) + Qwen3-0.6B-Base (~1.2 GB) | ~3.6 GB |

Extraction streams: each gate matrix is reduced to its singular spectrum as
soon as it is sliced, and only the spectrum plus name/role/shape metadata is
retained. Peak RAM is therefore one tensor plus spectra, not every gate
matrix in the checkpoint. That is what makes core-cohort pairs at 30B scale
feasible — e.g. the Qwen3-30B REAM pair (48 layers × 96 experts in the
subject, 48 × 128 in the reference; `[768,2048]` gates) would otherwise need
≈67.6 GB of fp32 gates in RAM for both sides together; streaming peaks at
one expert matrix, and Range reads cut that pair's traffic to ≈38.7 GB
instead of a 107.6 GB snapshot. Both figures are computed from tensor
shapes, not measured.

Safetensors shards are fetched with HTTP Range reads, one gate tensor at a
time: an 8-byte GET supplies the header length, `HfApi().get_safetensors_metadata`
supplies dtype/shape/offsets, and only the gate ranges (plus an FP8
`weight_scale` where present) transfer. Any Range failure — no partial-content
support, gated or redirected URL, short read — logs a `[range]` line and falls
back to `snapshot_download` plus the same local reader, so the shared HF cache
is the fallback path and the result is identical either way. GGUF stays
full-file (it is one file with no per-tensor index).

Runtime is dominated by SVDs and cold downloads; the granite pairs are the
heaviest (768 gate slices per checkpoint ⇒ 768 SVDs of `[512,1024]`). Once
cached, a full run is minutes on CPU — no GPU.

## Expectation outcomes and the gate

Each pair carries an `outcome`:

- `held` — verdict matched the pre-registered expectation.
- `missed` — the verdict differs and the abstain (if any) was not structural.
- `skipped_format` — `abstain` with `no_gate_tensors` or `format_unreadable`:
  the smoke structurally cannot consume the checkpoint (unknown naming, an
  unreadable file, an unsupported quantizer). Neither held nor a miss.

`degenerate_profile` is an abstain but not a format issue (a constant pooled
vector cannot be scored), so it is a miss when it contradicts an expectation.
The summary lists each bucket by pair id; `expectation_held` is true only for
`held`; exit 0 iff `missed` is empty.

### Default-cohort outcome (measured 2026-10-26)

Artifact: `provenancebench/runs/20261026T230514311838Z-c4720f53c3c6.*` — a
pbench-eval artifact; no smoke-CLI artifact exists for this cohort.

Recorded after the first run — the pre-registered part is the interpretation
list under Fixtures — so this is an observation, not a prediction. Two of
the nine pairs miss, both characterized method-population findings rather
than implementation defects:

- **`pb-smoke-qwen25-awq` (~0.41, missed).** AWQ's recipe folds a per-input-channel smoothing
  scale into the linear and its inverse into the preceding norm (`scale_ln_fcs`; AWQ Eq. 4):
  `W_gate ← W_gate·diag(s)` and `post_attention_layernorm ← γ/s`, where `γ` is that norm's
  existing gain. That is a change of basis, not noise, and `exp(H)` is not invariant to it: the
  smoothing is per input channel, so it reshapes the singular-value distribution that the
  statistic measures. Measured on the pinned weights (AWQ audit, 29 Oct 2026, script kept with
  the survey notes): recovering `s` from the norm ratio matches the per-column weight ratio at r
  = 0.9973–0.9998 over 24 layers (layer 0: 0.9995) and spans 0.20–5.13 across layers. The
  shipped path scores **0.4090**; dividing `s` out scores **0.9998** on the same pair. The
  decisive control is that applying `s` to the *unquantized* BF16 parent, with no quantization
  at all, already scores **0.389** — close to the 0.4090 the shipped path produces, so the fold
  alone accounts for the miss and int4 noise does not (int4-only controls score 0.99996 or
  better). The dequantizer was checked independently and is correct: an independent numpy unpack
  matches `tensors.dequantize_awq_gemm` bit-for-bit (max abs difference 0.0 over 24 gate
  tensors), 99.92 % of 4-bit codes match codes predicted from the BF16 parent on the stored
  grid, and the zero point is required (dropping it gives 3.06 relative Frobenius). The paper
  tested GPTQ only, which carries no such fold, so this class is outside its population. The
  miss landed below the pre-registered 0.85–0.95 band; the basis reading was developed after the
  result.
- **`pb-smoke-qwen3-upcycle` (~0.84, missed).** The partition upcycle carves
  the parent's 3072-neuron MLP into 6×384 routed plus 768 shared experts and
  then trains 10B tokens, exceeding the paper's tested transforms. The miss
  was pre-registered in the interpretation list above.

The default run therefore exits 1 with exactly these two misses; the other
seven pairs hold (the two unrelated hard negatives stay well below τ). The
same core's bench runs are the core cohort in
`provenancebench/runs/20261027T012533790066Z-15a9bc583089.*`, with the
gpt2-fineweb pair re-scored in
`provenancebench/runs/20261027T140739956978Z-466779299e34.*`.

## Scope and known limits

- **Dense-primary.** The paper evaluates dense decoder-only gated-FFN models.
  The MoE rule is ours (see the disclosure block) and every MoE result must be
  read as an extension, not a paper reproduction.
- **Population gap.** The paper has no same-architecture independently
  pretrained negative; ours include one. `Llama-3.2-1B` vs `SmolLM2-1.7B` is
  the same-geometry case — both share q=2048, with gate shapes `[8192, 2048]`
  and `[6144, 2048]` — and granite-MoE vs SmolLM2 is the cross-stack case
  (q=512 vs q=2048). The paper's structure-matched subset is AUROC 1.0000 at
  r=128 but 0.9800 at r=64, so collision risk on shared U-shaped depth
  profiles is real and untested by the authors.
- **τ headroom.** The paper's robust-min (GPTQ Int8 0.9108) clears τ by only
  0.0108, and Int4 (0.9256) by 0.0256 — an ordering that is not a bit-width
  effect (see the quantization bullets below); unrelated max 0.2288 was
  achieved only on cross-dimension, cross-data-regime pairs. A quantized pair
  landing in 0.85–0.95 is the paper's margin measured honestly, not a bug —
  and at that distance from τ the ordering between two quantized variants
  carries no information (see the Int8/Int4 bullet below).
- **Sign-insensitivity.** CKA squares the Pearson coefficient: an
  anticorrelated depth profile can score as high as its mirror image. The
  signed value and `anticorrelation_collision` flag are emitted for every
  pair; the paper saw 23/120 negative pairs, none below −0.9.
- **Rank sensitivity.** Beyond r=128 the top of the spectrum covers a larger
  fraction of the bulk and fingerprints converge (AUROC 0.9448 at r=256).
  `spectrum_fraction` is recorded per tensor: granite experts q=512 (25 %),
  upcycle routed q=384 (33 %), Qwen3 q=1024 (12.5 %), Llama/SmolLM q=2048
  (6.25 %). Comparing sides with very different fractions is a known
  confound.
- **MoE aggregation.** Averaging 32 experts shrinks profile variance
  (~1/32), flattening the curve toward the generic random-bulk shape. The
  upcycle model is a *partition* upcycle (not a copy) followed by 10B tokens
  of continued pretraining, so it may not match its dense parent at τ. The
  mean is unbiased only under uniformly random expert subsetting; a
  saliency-ranked drop (REAP) biases each layer's scalar by roughly
  (n_drop/n_keep)·δ, which is visible in the granite-reap miss (0.8163).
- **Quantization.** What moves this statistic is the recipe, not the bit width: AWQ's
  per-channel smoothing fold changes the basis and therefore the spectrum (mechanism in the miss
  analysis above), while its int4 quantization error alone leaves the score at 0.99996 or
  better. FP8-dynamic is near-lossless for spectra; GGUF k-quants add block-structured noise.
  MXFP4 (gpt-oss) is dequantized in numpy — E2M1 nibbles, E8M0 block exponents, gate = even
  columns after the layout transpose — and is bit-exact against
  `transformers.integrations.mxfp4` on random fixtures. AQLM/GPTQ/QuIP/GSQ have no dequantizer
  here and abstain as `format_unreadable`.
- **Bit-width ordering.** The paper's Int8 < Int4 result does not support a bit-width
  conclusion. Its GPTQ-Int8 row (0.9108) sits 0.0148 *below* its GPTQ-Int4 row (0.9256) against
  the same source model: n = 1 per bit width, and a 0.0148 gap is far smaller than the spread
  this metric shows under its own parameters (the recorded panels move the AWQ pair over
  0.095–0.573 across r, 0.257–0.409 across s). In our own controls more bits only reduce
  spectral flattening (int2 +0.93 %, int4 +0.032 %, int8 +0.0001 % mean Δf/f, from the 29 Oct
  2026 AWQ audit, not from a shipped script), and int4 at 12.75 % weight error still scores
  0.99996, so we would expect Int8 at or above Int4 rather than below. That is consistent with a
  recipe or operating-point effect, but it does not establish one: our pairing of the two Hub
  repositories that match the paper's model names and configs is an inference, the paper prints
  no repository URLs, and we did not recompute its 7B CKA values. Treat the inversion as
  unexplained.
- **Formats.** Safetensors shards are parsed header-first (8-byte header
  length + JSON + tensor bytes), or range-read per gate tensor (mechanism
  under Cost), so only gate tensors are materialized and float8/bf16 decode
  through `ml_dtypes` without torch. `pytorch_model.bin` (single-file, sharded `pytorch_model-NNNNN-of-NNNNN.bin`,
  or index-declared) is read with `torch.load(weights_only=True)`; torch is
  imported lazily and the pickle-safe mode refuses arbitrary payloads (that
  guarantee requires torch ≥ 2.6: earlier releases had a `weights_only`
  bypass, CVE-2025-32434, so pin the floor in any env where `.bin` is scored). GGUF
  goes through `gguf.GGUFReader`; an unknown GGML type code (a fork's
  non-standard blocks) becomes an abstain, not a crash. The GPT-2-style
  `mlp.c_proj` fallback from the paper's † footnote, and the disclosed
  GPTNeoX `dense_4h_to_h` / Nemotron-H `mixer.down_proj` down-projection
  fallbacks, are implemented and disclosed per model; the smoke cohort itself
  uses none of the three.
- **FP8 transport.** The stored fp8 weights are cast to fp32 with
  `ml_dtypes.float8_e4m3fn`; when the compressed-tensors checkpoint stores a
  `weight_scale` beside a weight (bf16, per-output-channel `[out, 1]`), the
  rows are multiplied by that scale after the cast, so the feature measures
  the deployment-time weight matrix rather than the stored quantization grid
  (the same principle as the AWQ path applying its dequantization scales).

## Output contract

One JSON object on stdout:

```jsonc
{
  "smoke": "gate-spectral",
  "paper": "...", "statistic": "GATE gate-projection spectral entropy, ...",
  "params": {"r": 128, "s": 8, "tau": 0.9, "eps": 1e-12,
             "sensitivity": {"r": [64, 256], "s": [4, 16]}},
  "pairs": [
    {
      "pair_id": "pb-smoke-granite-instruct", "role": "positive", "kind": "sft",
      "stack": "moe",
      "subject": {"repo_id": "...", "revision": "<pinned sha>"},
      "reference": {"repo_id": "...", "revision": "<pinned sha>"},
      "expected_verdict": "related",
      "verdict": "related" | "unrelated" | "abstain",
      "cka": 0.9997, "signed_pearson": 0.9999,
      "anticorrelation_collision": false, "structure_matched": true,
      "layers_per_segment": {"subject": [3,3,3,3,3,3,3,3],
                             "reference": [3,3,3,3,3,3,3,3]},
      "sensitivity": {"r64": {...}, "r256": {...}, "s4": {...}, "s16": {...},
                      "experts_only": {...}, "shared_only": {"status": "unavailable"},
                      "geometric": {...}, "concatenated": {...}},
      "flags": {"rank_sensitive": false, "moe_rule_sensitive": false,
                "low_signal_profile": false},
      "diagnostics": {
        "subject": {"family": "granite_moe", "format": "safetensors",
                    "source": {...}, "L": 24, "n_gate_tensors": 768,
                    "q_values": [512], "tensors": [{"name": "...",
                    "layer": 0, "role": "routed", "shape": [512, 1024],
                    "q": 512, "r_eff": 128, "spectrum_fraction": 0.25}, ...],
                    "profile_cv": 0.0031, "profile": [/* 8 values */],
                    "layers_per_segment": [3,3,3,3,3,3,3,3]},
        "reference": {...}
      },
      "outcome": "held", "expectation_held": true, "compute_seconds": 41.2
      // abstain extras: "reason": "no_gate_tensors" | "format_unreadable"
      //                 | "degenerate_profile", "detail": "..."
    }
  ],
  "summary": {"pairs_total": 9, "held": [...], "missed": [...],
              "skipped_format": [...]},
  "total_seconds": ...
}
```

## Selftest

`--selftest` runs fully offline (synthetic tensors, no network, no HF
downloads) and covers every property the contract pre-registers:

1. rank-k with k equal singular values ⇒ `exp(H) = k`;
2. scale invariance (`cW` has the same normalized spectrum);
3. orthogonal-reparameterization invariance (`σ(PWQ) = σ(W)`);
4. row/column permutation invariance;
5. CKA sign-insensitivity (`r = −0.95 ⇒ CKA = 0.9025`) and the
   anticorrelation-collision flag;
6. verdict threshold (`0.9 ⇒ related`, `0.899999 ⇒ unrelated`);
7. zero-variance abstain (constant profile ⇒ no CKA, never NaN);
8. `r_eff = min(r, q)` guard and `f <= r_eff`;
9. top-r normalization discrimination (`q=256`, top-128 = 1.0 and a 0.5
   tail ⇒ `f = 128`; normalizing over all q would give ≈ 241.9);
10. all-zero guard (`H = 0`, `f = 1`, flagged);
11. segment pooling remainder (`L=26 → [4,4,3,3,3,3,3,3]`);
12. layer aggregation (arithmetic mean of `exp(H)`) and integer depth order
    (keys `{2, 10}` ⇒ `[2, 10]`, never lexicographic);
13. MoE diagnostic variants on a two-tensor layer (arithmetic, geometric,
    concatenated-spectrum scalars);
14. end-to-end comparison statuses and the eight sensitivity variants;
15. profile diagnostics (L, counts, `layers_per_segment`);
16. AWQ int4 pack → dequantize round-trip (corr > 0.99);
17. GGUF Q8_0 quantize/dequantize round-trip (skipped with a note if
    gguf-py is unavailable) and a GGUFWriter-written 3-D tensor reloaded
    through the real extraction path (logical shape/values, router excluded);
18. family extraction: granite fused split, all router collisions,
    gpt-oss even-column de-interleave, upcycle roles, Mixtral `w1`, LFM2
    fused + discrete layouts, GGUF names, `step_<N>` directory resolution,
    and GGUF file choice;
19. new-family classification (GPTNeoX `dense_4h_to_h` matched while
    `dense_h_to_4h` and `mtp.layers.*` stay unclassified, Qwen3.5 nesting,
    Nemotron-H `mixer.down_proj`) and tinymixtral stacked pass-through
    (one spectrum per expert, no row split, no de-interleave);
20. MXFP4 dequantization bit-exact against
    `transformers.integrations.mxfp4._convert_moe_packed_tensors` (0 ULP),
    `.bin` stacked extraction through the real `_load_bin` with
    `torch.load(weights_only=True)`, and the GGUF unknown-type-code guard
    (`GGUFReader` `ValueError` ⇒ `FormatUnreadable`);
21. expectation-outcome classification;
22. malicious-shard rejection: five untrusted safetensors headers (negative
    shape dimension with negative offsets, header length 2^62, truncated
    header, list-shaped header, tensor bytes past the file end) each raise
    `FormatUnreadable` instead of allocating or reading to EOF;
23. a by-SHA `RevisionNotFoundError` from the Hub becomes an abstain entry
    (`skipped_format`), never a run abort;
24. a non-206 range response is rejected before the body is read (streaming),
    so a server that ignores `Range` cannot buffer a whole shard.

## Deviations from the paper

1. **Reimplementation from the paper only** (no released code), including
   Eq. (2)'s retained-only normalization and the equation-level defaults
   r=128, s=8, τ=0.9.
2. **MoE extension is ours** and disclosed verbatim above; the paper never
   evaluates an MoE checkpoint.
3. **Zero singular values** are skipped as `lim x ln x = 0` rather than fed
   through `log(σ̄ + ε)`; with a denominator-only ε this keeps rank-deficient
   matrices finite without changing positive spectra.
4. **GGUF file choice.** The GGUF repo pins a revision but not a file; Q8_0 is
   chosen (nearest to the parent numerics) and the filename plus per-tensor
   quant types are recorded in diagnostics.
5. **Non-gated fallbacks.** The paper's † GPT-2 `c_proj` auxiliary is
   implemented and disclosed per model; the same disclosed substitution
   covers GPTNeoX `dense_4h_to_h` and Nemotron-H `mixer.down_proj`, which
   have no gate branch. None of the three is used by the smoke cohort.
6. **Out-of-scope formats.** AQLM/GPTQ/QuIP/GSQ checkpoints abstain as
   `format_unreadable`. `pytorch_model.bin` (tinymixtral/GPTNeoX) and
   MXFP4 (gpt-oss) checkpoints are read — `.bin` via
   `torch.load(weights_only=True)`, MXFP4 via the numpy dequantizer — so
   they are scored rather than silently substituted.
