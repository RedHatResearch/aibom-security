# Routing agreement under expert/router quantization — Experiment B

**Question.** Does preserving router precision while quantizing experts preserve
routing (the llama.cpp GGUF property)? And how much does quantizing the router
itself hurt?

Not product code; not CI. Spec: [`.scratch/specs/moe-p0-and-routing.md`](../../.scratch/specs/moe-p0-and-routing.md)
(Experiment B). The E1-E3 pre-registration was written before the first run;
results were appended afterwards without changing any band. E4-E6 are
extensions, labelled as such below.

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
  `torch.topk(logits, min(top_k, E), dim=-1)` → expert ids per (prompt, position).
  The tuple length is checked against `config.num_hidden_layers`; a mismatch is
  adapted (non-null entries, first `n_layers`) and reported in `results.json`
  (`router_logits.adapted`). The recorded `n_layers` and all per-layer indexing
  come from the captured tuple (`router_logits.used_layers`), and each array is
  indexed by its own length, so a short or padded tuple prints short instead of
  raising.
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
states drift from there on. It is a negative control, but its null is **not** a
fresh uniform k-subset per token: each layer draws one fixed permutation and
applies it to every token. With that layer's input held fixed, the agreement is
exactly the fraction of its base sets invariant under the drawn permutation — a
single draw from a very broad distribution over cycle types. For E=4, k=2 the
identity gives 1, a transposition or double transposition gives 1/3, and a
3-cycle or 4-cycle gives 0; the mean over a random permutation is 1/6 with large
single-draw variance. Layers > 0 also see drifted inputs, so their agreement is
that invariance fraction plus drift, and the whole-model mean is a **mixture
across the drawn cycle types**, not an estimate concentrated at chance.

The calibrated check is layer 0: its input (the embedding) is untouched by the
shuffle, so its agreement equals, by construction, the fraction of base layer-0
routing sets left invariant by that permutation — computed from the base capture
and the drawn permutation, not a fitted parameter, and compared exactly. Both
that prediction and the per-layer fixed-point counts are reported; the identity
draws (fixed points = E) show how much row churn input drift alone causes.

Runtime/determinism: seed `20261029` for permutations,
`torch.set_num_threads(min(8, CPU count))`, no dropout, `use_cache=False`. Both
models and both tokenizers load with `local_files_only=True` and
`trust_remote_code=False`, so the run never touches the network; if the pinned
snapshot is incomplete the model is skipped with the reason recorded in
`results.json` rather than downloaded. A variant that raises (for example a
malformed expert/router shape) is skipped the same way — the model entry gets
`status: partial` and a `variant_skips` reason, `main` prints a
`SKIP <model>/<variant>` line, and the remaining variants still report.
Secondary model `allenai/OLMoE-1B-7B-0924` @ `6d84c48581ece794365f2b8e9cfb043c68ade9c5`
is loaded in bfloat16 and repeated with k=8, E=64 **only if** peak RSS stays
under 20 GB and the projected wall time under 600 s; otherwise it is skipped
with the reason recorded in `results.json`.

```bash
uv run --group smokes python smokes/routing-agreement/smoke.py
```

## Expectations: E1-E3 pre-registered, E4-E6 extensions

| id | model | variant | expectation |
|---|---|---|---|
| E1 | TinyMixtral | `experts_q4` | overall set agreement vs base **≥ 0.95** |
| E2 | TinyMixtral | `all_q4` | overall set agreement **<** `experts_q4` (direction only; magnitude reported) |
| E3 | TinyMixtral | `router_shuffled` | overall set agreement within **0.06** of chance (1/6 ≈ 0.1667) |
| E4 | OLMoE | `experts_q4` | overall set agreement vs base **≥ 0.95** (k=8, E=64) |
| E5 | OLMoE | `all_q4` | overall set agreement **<** `experts_q4` |
| E6 | OLMoE | `router_shuffled` | overall set agreement within **0.01** of chance (≈ 2.26e-10) |

E1-E3 are the spec's pre-registered table. **E4-E6 are extensions** added for
the OLMoE repeat, which that table did not cover; their bands reuse the E1-E3
forms rather than a new calibration: E4 carries E1's 0.95 band over unchanged
(no k/E rescaling), E5 repeats E2's ordering direction, and E6 is E3's
near-chance form with the absolute tolerance narrowed from ±0.06 to ±0.01;
with the analytic chance at k=8, E=64 at ≈2.26e-10 this is effectively a "≤0.01
above zero" test, not a two-sided chance band.

E3 (and E6) is **not a calibrated chance test**: `router_shuffled` draws one
fixed permutation per layer and applies it to every token, so its overall
agreement is a mixture over the drawn cycle types (on TinyMixtral, identity
draws measure 0.75–0.80 and non-identity draws 0.00–0.31), not a per-token
uniform k-subset concentrated at 1/6. The ±0.06 band is a fixed-permutation
control band; the calibrated check for that variant is the exact layer-0 relabel
prediction reported next to the measured value.

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

The recorded run exited 0. `results.json` is the machine-readable record of the
run (gitignored; regenerated by any rerun): generation timestamp, per-model load/forward/elapsed timings, peak RSS, and
every routing number below. Timings and memory vary between runs and are not
repeated in this prose; the routing numbers are copied from the record after the
run (the model and tokenizer loads are offline — `local_files_only=True`).

Routing tuple shape: **12 entries for TinyMixtral's 12 layers and 16 for OLMoE's
16 layers, all non-null** — no adaptation was needed for either model.

### Secondary-model budgets

Rule precedence: for this smoke Experiment B's "load in bf16, ~14 GB" **governs**
over the shared "use streaming reads (`safetensors.safe_open`), do not load
14 GB into RAM at once" instruction — the experiment needs the whole model
resident for full forward passes, and the RSS/time gates below bound the load.

| check | budget | outcome | recorded in `results.json` |
|---|---|---|---|
| OLMoE RSS after load | ≤ 20 GB | pass | `peak_rss_gb_after_load` |
| OLMoE RSS after base capture (the gate value) | ≤ 20 GB | pass | `peak_rss_gb_after_base` |
| OLMoE whole-process peak (`ru_maxrss`) | reported, not gated | reported | `peak_rss_gb` |
| OLMoE projected wall time (gate) | ≤ 600 s | pass | `projected_total_s` |
| OLMoE actual wall time | reported | — | `elapsed_s` |

Note on the peak: the gate reads RSS after the base capture and is a pre-flight
heuristic, not a bound on the run — the same run's final high-water mark is
higher and is recorded as `peak_rss_gb`. The after-load figure is below the
checkpoint's 13.838 GB of bf16 payload because the safetensors mappings are not
fully resident right after load (`[INFERENCE]` on the mmap explanation);
`ru_maxrss` is page-residency dependent, and repeated runs of this smoke have
moved the peak by several GB while the routing tables stayed bit-identical.

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

Layer-0 sensitivity (post-hoc diagnostic; the pre-registered metric and band are
unchanged): the overall metric is an unweighted mean over layers and `experts_q4`
layer 0 is structurally `1.000000`, so **excluding layer 0 the other 11 layers
average `0.949929`** — below the 0.95 band. E1's +0.004 margin is the
structurally perfect layer's 1/12 contribution, not evidence that the 11
non-trivial layers clear the band. (The same diagnostic for the other variants:
`all_q4` `0.810488`, `router_shuffled` `0.225931`.) The same diagnostic is
recorded in `results.json` as `overall_set_agreement_excl_layer0`. The metric
and band are left exactly as pre-registered.

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
GPTNeoX tokenizer (95, 92, 84 and 93 ids; the same texts are 99, 98, 110 and 100
ids under the Mistral tokenizer); they are right-padded with pad id 1 and the 20
pad positions are excluded from scoring (24×96 − 20 = 2284 rows).

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

Layer-0 sensitivity (post-hoc diagnostic; the pre-registered metric is
unchanged): **excluding layer 0 the other 15 layers average `0.746293`** for
`experts_q4` (`all_q4` `0.612639`, `router_shuffled` `0.000000`). E4 is a miss
with or without the structurally perfect layer.

Per-domain overall set agreement (OLMoE):

| domain | experts_q4 | all_q4 | router_shuffled |
|---|---|---|---|
| code | 0.785807 | 0.655707 | 0.000000 |
| math | 0.733370 | 0.585978 | 0.000000 |
| factual | 0.765597 | 0.622868 | 0.000000 |
| narrative | 0.763889 | 0.613281 | 0.000000 |

### Expectations vs outcome

| id | model | expectation | observed | verdict |
|---|---|---|---|---|
| E1 | TinyMixtral | `experts_q4` overall set agreement ≥ 0.95 | 0.954102 (excl. layer 0: 0.949929) | **held** (margin +0.004) |
| E2 | TinyMixtral | `all_q4` < `experts_q4` | 0.818432 < 0.954102 | **held** |
| E3 | TinyMixtral | `router_shuffled` within 0.06 of chance 0.1667 | 0.207104 (+0.0404) | **held** |
| E4 | OLMoE | `experts_q4` overall set agreement ≥ 0.95 | 0.762150 (excl. layer 0: 0.746293) | **missed** (−0.188) |
| E5 | OLMoE | `all_q4` < `experts_q4` | 0.619445 < 0.762150 | **held** |
| E6 | OLMoE | `router_shuffled` within 0.01 of chance ≈ 2.3e-10 | 0.000000 | **held** |

E4 is a real miss, not a measurement defect: the layer-0 identity check is exact
(`experts_q4` layer 0 = 1.000000) and the deviation is present at every layer
(0.698–0.810). At k=8, E=64 the 8th-ranked expert's logit margin is much thinner
than the 2nd-ranked expert's at k=2, E=4 (inference from the margin structure;
margins were not separately measured here), so the same Q4 grid flips many more
rows. The band was extrapolated from the k=2 intuition and was simply wrong at
k=8; it is recorded as a miss and not re-tuned. E1 held by 0.004, but that
margin is entirely layer 0's structural `1.000000`: excluding layer 0 the mean is
`0.949929`, below the band, so the 11 non-trivial layers do not clear it either
and E1 is reported as fragile rather than comfortable. E3's band also passed on
a fixed-permutation mixture (identity draws 0.75–0.80, non-identity draws
0.00–0.31), not on a calibrated null; the exact layer-0 predictions, not the
mean, are the check for that control. E6 is a weak test by construction (chance
is ~0, and the control measures exactly 0); it is the per-layer structure and
the layer-0 closed form that carry information for that variant.

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
   a different mixture and hidden states drift. Each layer applies one fixed
   permutation to every token, so the control's per-layer value is that layer's
   invariance fraction (plus drift for layers > 0) and the overall mean is a
   mixture over the drawn cycle types; only layer 0 has the closed-form
   prediction (`permutation_invariance`), and it matches exactly for both models.
   An earlier draft of this smoke assumed the whole forward was invariant and
   predicted every layer from the base capture; that prediction was wrong (0.232
   predicted vs 0.206 measured overall, with per-layer errors up to 0.25) and
   only layer 0 matched exactly, which is what exposed the error. The assumption
   was corrected before the recorded run, and no threshold changed.
3. **Order of variants matters for confounding.** The shuffle is captured first,
   on pristine weights, with router rows restored immediately; only then are the
   experts QDQ'd and the router QDQ'd on top. This keeps `router_shuffled` free
   of expert-Q4 contamination and needs no 14 GB expert backup (`results.json`
   → `execution_order`).
4. **Tokenizer mismatch for the secondary model.** OLMoE uses its own GPTNeoX
   tokenizer (present in the pinned local snapshot; loads are offline, so a
   missing snapshot is a reported skip, not a download); with GPTNeoX
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
OLMoE's (**0.9499** and **0.746** excluding layer 0, where routing is
expert-independent by construction — the TinyMixtral figure is then just below
the 0.95 band), while quantizing the router too drops those to **0.818** and
**0.619**.
The router is 4096 parameters per layer against ≈ 50.3 M in that layer's
experts (0.008%) yet costs far more routing change than the experts: at layer 0,
where routing is provably independent of expert weights (`experts_q4` =
1.000000 exactly), router Q4 alone flips 9.4% of TinyMixtral rows and 27.8% of
OLMoE rows, and deeper `all_q4` layers stay 12–24% (TinyMixtral) and 32–46%
(OLMoE) off base with no monotone depth trend. That is the llama.cpp "keep the router at higher
precision" property reproduced in kind — expert Q4 is comparatively benign,
router Q4 is the dominant routing perturbation, and it is worse when more
experts compete (k=8 of 64). The shuffled control is a negative control, not a
calibrated near-chance test: each layer applies one fixed permutation to every
token, so the overall mean is a mixture over the drawn cycle types (TinyMixtral
0.207, OLMoE 0.000; mean Jaccard 0.429 vs chance 0.389, and 0.0710 vs 0.0705).
On TinyMixtral its per-layer values span 0.00–0.80 because layers 6 and 8 drew
the identity permutation and measure 0.753/0.798 — those layers show that input
drift alone from upstream random routing moves ≈20–25% of rows, which is the
right scale for reading the `all_q4` numbers — while layer 0 drew a 3-cycle and
measures exactly its closed-form 0.000. E3's band passed because this fixed
mixture landed at 0.207, not because the control concentrates at chance; the
exact layer-0 predictions are what calibrate it. Per-domain results
are flat — within 2 points for TinyMixtral `experts_q4` (0.945–0.963) and
within 6 points for OLMoE (0.733–0.786) — so the degradation tracks the router
geometry rather than the prompt domain.

## Files

- `config.py` — pinned models/revisions, 24 probe prompts (verbatim), variant
  order, expectations (E1-E3 pre-registered, E4-E6 extensions).
- `quant.py` — frozen symmetric 4-bit block-32 QDQ constants and helpers, in-place
  application, router permutation helpers, permutation-invariance prediction.
- `smoke.py` — load, capture routing, run variants, metrics, table + `results.json`.
- `results.json` — machine-readable record of the latest run, gitignored
  (timings, RSS, routing tables, verdicts).
