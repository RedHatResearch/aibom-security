# Routing agreement under expert/router quantization — Experiment B

**Question.** Does preserving router precision while quantizing experts preserve
routing (the llama.cpp GGUF property)? And how much does quantizing the router
itself hurt?

Not product code; not CI. Spec: [`.scratch/specs/moe-p0-and-routing.md`](../../.scratch/specs/moe-p0-and-routing.md)
(Experiment B). This README was written (pre-registration included) **before**
the first run; results were appended afterwards without changing any band.

## Method

- Probe model: `Isotonic/TinyMixtral-4x248M-MoE` @ `1e3516176a6279ce93923060fcad848dc043af79`,
  `AutoModelForCausalLM`, `dtype=torch.float32`, `output_router_logits=True`, CPU,
  `attn_implementation="eager"`, `eval()` + `inference_mode()`.
- Probe set (`config.PROMPTS`, exact text recorded there): 24 prompts, 6 each
  code / math / factual / narrative, fixed order, tokenized without special
  tokens and truncated or padded to 96 ids. All 24 tokenize to ≥ 96 ids with the
  Mistral tokenizer, so TinyMixtral scores 24 × 96 = 2304 rows with no padding;
  with OLMoE's GPTNeoX tokenizer 4 prompts land just under 96 ids, so they are
  right-padded and their pad positions are excluded (2284 rows scored).
- Routing capture: `forward(..., output_router_logits=True)` returns a tuple with
  one router-logit tensor `[rows, E]` per decoder layer; per layer take
  `torch.topk(logits.float(), k=-1)` → expert ids per (prompt, position).
  The tuple length is checked against `config.num_hidden_layers`; a mismatch is
  adapted (non-null entries, first `n_layers`) and reported in `results.json`
  (`router_logits.adapted`).
- Metric per layer: fraction of scored rows whose top-k expert **set** equals the
  base set (order-insensitive exact match), plus mean Jaccard over the same rows.
  Overall = unweighted mean over layers. Per-domain overall numbers are reported
  as a diagnostic.
- Chance level: analytic, for two independent uniform k-subsets of E experts —
  `P(exact set) = 1/C(E,k)`, `E[J] = Σ_x P(X=x)·x/(2k−x)` with
  `X ~ Hypergeometric(E,k,k)` (exact rational arithmetic).

### Quantization contract (frozen)

Symmetric round-to-nearest, 4-bit, block size 32 along the **last** dimension,
quantize-dequantize (snap to grid, then back to the model dtype). `qmax = 7`,
`scale = max|block| / 7`, `q = clamp(round(x/scale), −7, 7)`, `x̂ = q·scale`;
an all-zero block uses `scale = 1e-12`. The in-place application slices along
dim 0 (≤ 64 MB fp32 working set per chunk); rows are independent under
last-dim blocking, so this is bit-identical to quantizing the whole tensor at
once and keeps the 14 GB OLMoE load out of a second full-size fp32 copy. On the
loaded model both checkpoints
expose **fused 3-D expert tensors** in transformers 5.15
(`...mlp.experts.gate_up_proj` `[E, out, in]`, `...mlp.experts.down_proj`), not
the per-expert `experts.{E}.w1.weight` modules the checkpoint stores; the last
dimension is a whole number of 32-wide blocks and each expert row is a separate
row of blocks, so the grid is identical to quantizing the per-expert 2-D
weights.

Variants, captured in this order on the loaded weights:

| # | variant | what changes |
|---|---|---|
| 1 | `base` | nothing (reference) |
| 2 | `router_shuffled` | per-layer seeded `randperm(E)` permuting router rows, restored after capture (pristine experts) |
| 3 | `experts_q4` | all `.experts.*` tensors QDQ'd in place, router untouched |
| 4 | `all_q4` | router gate weight (`*.gate.weight`) QDQ'd on top of the already-QDQ'd experts (= base experts QDQ'd once + router QDQ'd once) |

Order note: the shuffle runs first so the control is not confounded with expert
Q4, and its router rows are restored immediately. `experts_q4` then leaves the
(large) expert tensors QDQ'd and `all_q4` only adds the router, so every variant
is reached from pristine weights with only the router (a few hundred KB per
layer) backed up. No full-size expert copy is kept, which is what lets the OLMoE
repeat fit in the RAM budget.

`router_shuffled` permutes the rows of each router matrix, which permutes the
router logits; the selected ids become `perm⁻¹(S)` for the base set `S`, but the
**experts stay in place**, so the layer computes a different mixture and hidden
states drift from there on. It is therefore a genuine random-routing control
whose whole-model agreement should sit near the analytic chance level, with one
exception that gives an exact check: layer 0's input (the embedding) is untouched
by the shuffle, so its agreement equals, by construction, the fraction of base
layer-0 routing sets left invariant by that permutation — computed from the base
capture and the drawn permutation, not a fitted parameter. Both the layer-0
prediction and the per-layer fixed-point counts are reported.

Runtime/determinism: seed `20261029` for permutations, `torch.set_num_threads(8)`,
no dropout, `use_cache=False`. Secondary model `allenai/OLMoE-1B-7B-0924` @
`6d84c48581ece794365f2b8e9cfb043c68ade9c5` is loaded in bfloat16 and repeated
with k=8, E=64 **only if** peak RSS stays under 20 GB and the projected wall
time under 600 s; otherwise it is skipped with the reason recorded in
`results.json`.

```bash
uv run --group smokes python smokes/routing-agreement/smoke.py
```

## Pre-registered expectations (written before the first run)

| id | model | variant | expectation |
|---|---|---|---|
| E1 | TinyMixtral | `experts_q4` | overall set agreement vs base **≥ 0.95** |
| E2 | TinyMixtral | `all_q4` | overall set agreement **<** `experts_q4` (direction only; magnitude reported) |
| E3 | TinyMixtral | `router_shuffled` | overall set agreement within **0.06** of chance (1/6 ≈ 0.1667) |
| E4 | OLMoE | `experts_q4` | overall set agreement vs base **≥ 0.95** (k=8, E=64) |
| E5 | OLMoE | `all_q4` | overall set agreement **<** `experts_q4` |
| E6 | OLMoE | `router_shuffled` | overall set agreement within **0.01** of chance (≈ 2.26e-10) |

Rationale recorded before the run: expert Q4 with 32-wide blocks is a small
relative perturbation (for a Gaussian block, block max ≈ 2.2σ, step ≈ 0.31σ,
RMS error ≈ 0.09σ, i.e. ≈ 20 dB), and the router logit gaps between the k-th and
(k+1)-th expert were expected to be large compared with that, so routing should
mostly survive; the router matrix is only 4×1024
(TinyMixtral) / 64×2048 (OLMoE) parameters but every row sets one expert's
score, so its Q4 error moves logits more directly and should cost more than
expert quantization. Note that `1/C(E,k)` for OLMoE (2.26e-10) makes E6 a
near-exact-zero test rather than a band.

## Results

Recorded run: `2026-10-29T18:04:12Z`, total 191.8 s, exit code 0, macOS arm64
(12 CPU cores, 36 GB RAM), torch 2.13.0, transformers 5.15.0, 8 threads, seed
20261029. `results.json` is the machine-readable record of exactly this run;
two further runs on the same variant order produced bit-identical tables (only
timings differed), one of them before the QDQ chunking change.

Routing tuple shape: **12 entries for TinyMixtral's 12 layers and 16 for OLMoE's
16 layers, all non-null** — no adaptation was needed for either model.

### Secondary-model budgets

| check | budget | observed | verdict |
|---|---|---|---|
| OLMoE RSS after load | ≤ 20 GB | 12.32 GB | pass |
| OLMoE RSS after base capture (the gate value) | ≤ 20 GB | 16.25 GB | pass |
| OLMoE whole-process peak (`ru_maxrss`) | reported | 17.62 GB | pass |
| OLMoE wall time | ≤ 600 s | 275 s projected, 179.1 s actual | pass |

Note on the peak: across the four runs of this smoke the gate value ranged
12.3–18.9 GB and the whole-process peak 17.4–20.7 GB, so the headroom against
"~20 GB" is small and run-dependent even though only one model copy is ever
held. The excess over the after-load figure appears on the first forward and is
consistent with oneDNN creating blocked copies of the weights (inference, not
measured directly); it never changed a routing number (the three runs on the
final variant order are bit-identical).
Total wall time per run was 191.8–214 s, well inside the 600 s budget.

### TinyMixtral-4x248M-MoE — L=12, E=4, k=2, fp32, 2304 rows

Chance: set agreement `1/C(4,2) = 1/6 = 0.166667`, mean Jaccard `7/18 = 0.388889`.

| layer | experts_q4 set | experts_q4 J | all_q4 set | all_q4 J | router_shuffled set | router_shuffled J |
|---|---|---|---|---|---|---|
| 0 | 1.000000 | 1.000000 | 0.905816 | 0.937211 | 0.000000 | 0.333333 |
| 1 | 0.980903 | 0.987269 | 0.765625 | 0.842593 | 0.012587 | 0.199653 |
| 2 | 0.973090 | 0.982060 | 0.792969 | 0.861545 | 0.044271 | 0.349826 |
| 3 | 0.951389 | 0.967593 | 0.783420 | 0.854745 | 0.305990 | 0.523727 |
| 4 | 0.948351 | 0.965567 | 0.763455 | 0.841869 | 0.038628 | 0.252459 |
| 5 | 0.939236 | 0.959346 | 0.792969 | 0.861400 | 0.066406 | 0.359954 |
| 6 | 0.934896 | 0.956597 | 0.796875 | 0.864005 | 0.752604 | 0.834346 |
| 7 | 0.937500 | 0.958333 | 0.813368 | 0.875434 | 0.269965 | 0.504774 |
| 8 | 0.940538 | 0.960359 | 0.842014 | 0.894387 | 0.797743 | 0.865017 |
| 9 | 0.932726 | 0.955150 | 0.835069 | 0.890046 | 0.020399 | 0.193142 |
| 10 | 0.945747 | 0.963831 | 0.851562 | 0.900897 | 0.123264 | 0.398003 |
| 11 | 0.964844 | 0.976562 | 0.878038 | 0.918547 | 0.053385 | 0.333333 |
| **mean** | **0.954102** | **0.969389** | **0.818432** | **0.878557** | **0.207104** | **0.428964** |

Structural checks: `experts_q4` layer 0 is exactly `1.000000` (layer-0 routing
does not depend on expert weights), so `all_q4` layer 0 `= 0.905816` isolates
router Q4 alone. `router_shuffled` layer-0 relabel prediction (from the drawn
permutation, a 3-cycle) is `0.000000` and measured is `0.000000` — exact. Fixed
points per layer: `[1, 0, 1, 2, 0, 1, 4, 2, 4, 0, 2, 0]` (layers 6 and 8 drew the
identity permutation).

Per-domain overall set agreement (TinyMixtral):

| domain | experts_q4 | all_q4 | router_shuffled |
|---|---|---|---|
| code | 0.962529 | 0.851707 | 0.214265 |
| math | 0.957176 | 0.833333 | 0.205874 |
| factual | 0.951389 | 0.808304 | 0.212384 |
| narrative | 0.945312 | 0.780382 | 0.195891 |

### OLMoE-1B-7B-0924 — L=16, E=64, k=8, bf16, 2284 rows

Chance: set agreement `1/C(64,8) = 1/4 426 165 368 ≈ 2.259e-10`, mean Jaccard
`2810058493/39835488312 ≈ 0.070542`. Four prompts tokenize below 96 ids with the
GPTNeoX tokenizer (they were 98–99 ids under the Mistral tokenizer); they are
right-padded with pad id 1 and the 20 pad positions are excluded from scoring
(24×96 − 20 = 2284 rows).

| layer | experts_q4 set | experts_q4 J | all_q4 set | all_q4 J | router_shuffled set | router_shuffled J |
|---|---|---|---|---|---|---|
| 0 | 1.000000 | 1.000000 | 0.721541 | 0.936719 | 0.000000 | 0.078976 |
| 1 | 0.809982 | 0.957385 | 0.605517 | 0.907603 | 0.000000 | 0.080939 |
| 2 | 0.723730 | 0.936194 | 0.536778 | 0.887948 | 0.000000 | 0.069099 |
| 3 | 0.712785 | 0.932608 | 0.558231 | 0.896862 | 0.000000 | 0.065590 |
| 4 | 0.742119 | 0.940903 | 0.594133 | 0.904012 | 0.000000 | 0.068516 |
| 5 | 0.698336 | 0.928093 | 0.586690 | 0.897157 | 0.000000 | 0.067404 |
| 6 | 0.719790 | 0.932572 | 0.611646 | 0.906297 | 0.000000 | 0.071584 |
| 7 | 0.720228 | 0.934835 | 0.588441 | 0.901842 | 0.000000 | 0.094191 |
| 8 | 0.720665 | 0.932997 | 0.607268 | 0.904529 | 0.000000 | 0.070566 |
| 9 | 0.781524 | 0.948465 | 0.682574 | 0.924636 | 0.000000 | 0.073989 |
| 10 | 0.775832 | 0.947984 | 0.664623 | 0.920561 | 0.000000 | 0.054583 |
| 11 | 0.772767 | 0.947781 | 0.656743 | 0.919585 | 0.000000 | 0.091185 |
| 12 | 0.788967 | 0.951717 | 0.648424 | 0.917060 | 0.000000 | 0.069352 |
| 13 | 0.729422 | 0.938251 | 0.624343 | 0.910931 | 0.000000 | 0.069496 |
| 14 | 0.749562 | 0.942960 | 0.605517 | 0.906804 | 0.000000 | 0.063404 |
| 15 | 0.748687 | 0.936707 | 0.618651 | 0.904645 | 0.000000 | 0.046599 |
| **mean** | **0.762150** | **0.944341** | **0.619445** | **0.909199** | **0.000000** | **0.070967** |

Structural checks: `experts_q4` layer 0 `= 1.000000` (exact), so `all_q4` layer 0
`= 0.721541` is router Q4 in isolation — 27.8% of rows flip from the router
quantization alone. `router_shuffled` layer-0 relabel prediction `0.000000` =
measured `0.000000` (layer 0 drew 4 fixed points, but no invariant 8-of-64 set
occurred among the 2284 rows). Fixed points per layer:
`[4, 4, 0, 3, 2, 0, 1, 1, 2, 1, 0, 2, 2, 1, 2, 0]`.

Per-domain overall set agreement (OLMoE):

| domain | experts_q4 | all_q4 | router_shuffled |
|---|---|---|---|
| code | 0.785807 | 0.655707 | 0.000000 |
| math | 0.733370 | 0.585978 | 0.000000 |
| factual | 0.765597 | 0.622868 | 0.000000 |
| narrative | 0.763889 | 0.613281 | 0.000000 |

### Pre-registered expectations vs outcome

| id | model | expectation | observed | verdict |
|---|---|---|---|---|
| E1 | TinyMixtral | `experts_q4` overall set agreement ≥ 0.95 | 0.954102 | **held** (margin +0.004) |
| E2 | TinyMixtral | `all_q4` < `experts_q4` | 0.818432 < 0.954102 | **held** |
| E3 | TinyMixtral | `router_shuffled` within 0.06 of chance 0.1667 | 0.207104 (+0.0404) | **held** |
| E4 | OLMoE | `experts_q4` overall set agreement ≥ 0.95 | 0.762150 | **missed** (−0.188) |
| E5 | OLMoE | `all_q4` < `experts_q4` | 0.619445 < 0.762150 | **held** |
| E6 | OLMoE | `router_shuffled` within 0.01 of chance ≈ 2.3e-10 | 0.000000 | **held** |

E4 is a real miss, not a measurement defect: the layer-0 identity check is exact
(`experts_q4` layer 0 = 1.000000) and the deviation is present at every layer
(0.698–0.810). At k=8, E=64 the 8th-ranked expert's logit margin is much thinner
than the 2nd-ranked expert's at k=2, E=4 (inference from the margin structure;
margins were not separately measured here), so the same Q4 grid flips many more
rows. The band was extrapolated from the k=2 intuition and was simply wrong at
k=8; it is recorded as a miss and not re-tuned. E1 held by 0.004, i.e. the band
was almost wrong there too. E6 is a weak test by construction (chance is ~0, and
the control measures exactly 0); it is the per-layer structure and the layer-0
closed form that carry information for that variant.

### Implementation surprises (reported, not tuned away)

1. **Fused expert tensors.** transformers 5.15 does not instantiate per-expert
   `experts.{E}.w1/w2/w3` modules; it fuses each layer into
   `experts.gate_up_proj` `[E, 2I, H]` and `experts.down_proj` `[E, H, I]`
   (`w1`/`w3` concatenated along the output axis). QDQ therefore runs on the
   fused 3-D parameters, block 32 along the last (input) dimension. Each expert
   row is a separate row of blocks, so the quantization grid is identical to the
   per-expert 2-D weights the spec names; only the tensor names/shapes differ.
2. **`router_shuffled` is not a relabelling no-op.** Permuting gate rows permutes
   the ids (to `perm⁻¹(S)`) but the experts stay in place, so the layer computes
   a different mixture and hidden states drift. That is why it behaves like a
   random-routing control; only layer 0 has the closed-form prediction
   (`permutation_invariance`), and it matches exactly for both models. An earlier
   draft of this smoke assumed the whole forward was invariant and predicted
   every layer from the base capture; that prediction was wrong (0.232 predicted
   vs 0.206 measured overall, with per-layer errors up to 0.25) and only layer 0
   matched exactly, which is what exposed the error. The assumption was corrected
   before the recorded run, and no threshold changed.
3. **Order of variants matters for confounding.** The shuffle is captured first,
   on pristine weights, with router rows restored immediately; only then are the
   experts QDQ'd and the router QDQ'd on top. This keeps `router_shuffled` free
   of expert-Q4 contamination and needs no 14 GB expert backup (`results.json`
   → `execution_order`).
4. **Tokenizer mismatch for the secondary model.** OLMoE's tokenizer is not in
   the original local snapshot (downloaded on first use); with GPTNeoX
   tokenization 4 prompts fall below 96 ids, so they are padded and 20 positions
   are dropped from scoring (2284 vs 2304 rows). TinyMixtral has no padding.
5. **bf16 quantization path.** OLMoE expert tensors are QDQ'd in float32 and
   stored back into the bf16 parameters, so the dequantized values are then
   rounded to bf16 (≈ 0.2% relative per weight) on top of the Q4 error
   (≈ 0.09σ_w RMS, i.e. ≈ 9% of a weight's standard deviation). The bf16
   baseline itself is not compared against fp32 here; all OLMoE numbers are
   *relative to the bf16 base*, as specified.

## Interpretation

Keeping the router in full precision while quantizing experts largely preserves
routing at k=2/E=4 and only partly at k=8/E=64: expert-only Q4 holds
**0.954** of TinyMixtral's (prompt, position) routing sets and **0.762** of
OLMoE's, while quantizing the router too drops those to **0.818** and **0.619**.
The router is 4096 parameters per layer against ≈ 50.3 M in that layer's
experts (0.008%) yet costs far more routing change than the experts: at layer 0,
where routing is provably independent of expert weights (`experts_q4` =
1.000000 exactly), router Q4 alone flips 9.4% of TinyMixtral rows and 27.8% of
OLMoE rows, and deeper `all_q4` layers stay 12–24% (TinyMixtral) and 32–46%
(OLMoE) off base with no monotone depth trend. That is the llama.cpp "keep the router at higher
precision" property reproduced in kind — expert Q4 is comparatively benign,
router Q4 is the dominant routing perturbation, and it is worse when more
experts compete (k=8 of 64). The shuffled control behaves as random routing
overall (TinyMixtral 0.207 vs chance 0.167, OLMoE 0.000 vs 2.3e-10 with mean
Jaccard 0.0710 vs 0.0705), although on TinyMixtral its per-layer values span
0.00–0.80 because layers 6 and 8 drew the identity permutation — those layers
show that input drift alone from upstream random routing moves ≈20–25% of rows,
which is the right scale for reading the `all_q4` numbers. Per-domain results
are flat — within 2 points for TinyMixtral `experts_q4` (0.945–0.963) and
within 6 points for OLMoE (0.733–0.786) — so the degradation tracks the router
geometry rather than the prompt domain.

## Files

- `config.py` — pinned models/revisions, 24 probe prompts (verbatim), quant
  contract, variant order, pre-registered expectations.
- `quant.py` — symmetric 4-bit block-32 QDQ, in-place application, router
  permutation helpers, permutation-invariance prediction.
- `smoke.py` — load, capture routing, run variants, metrics, table + `results.json`.
- `results.json` — machine-readable record of the run above.
