# MoE expert lineage smoke

- Issue: [#26](https://github.com/RedHatResearch/aibom-security/issues/26) — Experiment A of
  [`.scratch/specs/moe-p0-and-routing.md`](../../.scratch/specs/moe-p0-and-routing.md)
- Sibling: [`routing-agreement/`](../routing-agreement/) (Experiment B, same spec)

## What this checks

Whether a weights-only, label-blind detector can recover **expert-level derivation** when the
ground truth is known. Eleven derived checkpoints are synthesised from a cached parent MoE, each
with a known edit, written to disk as plain safetensors, and then scored by a detector that sees
only the files: it must say which parent experts survive, which slots were permuted, which
experts are gone, and which "new" expert is a convex mixture of two or three parents (with the
mixture weights and an explained-variance gain).

The point is the *detector contract*, not the model: synthetic derivation gives exact ground
truth, so a miss is a property of the descriptor / matching / fit, not of a noisy label.

## Fixtures

| role | repo | revision | arch | layers | experts |
|---|---|---|---|---|---|
| primary | `Isotonic/TinyMixtral-4x248M-MoE` | `1e3516176a6279ce93923060fcad848dc043af79` | mixtral | all 12 | 4 (top-2, hidden 1024, inter 4096, fp32) |
| scale repeat | `allenai/OLMoE-1B-7B-0924` | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` | olmoe | 0 and 8 | 64 (top-8, hidden 2048, bf16) |

Both snapshots are resolved by path inside the local HF cache; nothing is downloaded. Tensor names
come from the architecture table in [config.py](config.py):
`model.layers.{L}.block_sparse_moe.experts.{E}.{w1,w2,w3}.weight` + `…block_sparse_moe.gate.weight`
for Mixtral, `model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight` + `…mlp.gate.weight`
for OLMoE. The scale repeat runs only if the primary phase takes < 600 s.

## Pre-registration (2026-10-29, before the first run; `origin` marks the post-hoc row)

Seed `20261029` for every permutation, prune choice and Gaussian control tensor. Thresholds are
fixed a priori: exact match at cos ≥ **0.9995**, mixture gain ≥ **0.30**, injected α = **0.60/0.40**
with a ±0.02 band, shallow merge at α = **0.95/0.05** with the same ±0.02 band. The component floor
that decides whether a fitted α counts as a merge parent is **0.03** (`SHALLOW_ALPHA[1] − ALPHA_TOL`;
re-baselined from 0.04, see below).

| variant | origin | expectation |
|---|---|---|
| `permuted` | `pre_registered` | every expert matched at cos ≥ 0.9995 in every layer; recovered π equals the injected π |
| `pruned` | `pre_registered` | the E−2 survivors matched, 2 experts reported missing, and no mixture fit reaching gain ≥ 0.30 |
| `merged` | `pre_registered` | E−2 untouched experts matched, one new expert reported and explained by the 0.60/0.40 mixture of the two missing parents (α within ±0.02, gain ≥ 0.30) |
| `merged_shallow` | `post_hoc` | exactly one merged expert with α within ±0.02 of 0.95/0.05 and gain ≥ 0.30, whether or not cosine matching would have called it a copy |
| `merged_permuted` | `pre_registered` | same as `merged` after alignment: E−2 exact matches, one new expert at the injected slot, α within ±0.02, gain ≥ 0.30 |
| `control` | `pre_registered` | 0 experts matched at cos ≥ 0.9995; no gain ≥ 0.30 |

Every row is pre-registered except `merged_shallow`, which was restated after the first run; its
original claim, the 0/2 OLMoE miss and the rationale are in “Second pre-registration, after a
miss” below. [`config.py`](config.py) carries `origin` on each claim and `results.json` copies the
registry plus `origin_ref` on the post-hoc row.

For TinyMixtral (E = 4) the concrete form is: 4/4 matched (permuted), 2/2 survivors + 2 missing
(pruned), 2 matched + 1 new (merged, merged_permuted). The claims are written for general E so
the same rules score the OLMoE repeat (E = 64).

Reasoning, fixed before the run:

- `permuted` and `pruned` are near-tautological for a value-based descriptor: a copied tensor
  must reach cos = 1.0. They are the sanity floor — a miss there means the descriptor or the
  slot-independence assumption is broken, not that the model is hard.
- `merged` is the real question. For near-orthogonal parents a 0.6/0.4 mixture has cos ≈ 0.83
  with its dominant parent (OLMoE: 0.819/0.848), far below 0.9995, so it must surface as a *new*
  expert; the constrained fit — run over every parent, not only the missing ones — should return
  the exact weights because the mixture lies inside the simplex. On a model with near-duplicate
  experts the same mixture stays close to its dominant parent, which is the case `merged_shallow`
  isolates.
- `control` fixes the false-positive floor: Gaussian noise at the parent's per-projection std
  shares nothing with any parent, so 0 exact matches and small/negative mixture gains are
  expected — the mixture is constrained to the simplex, so it cannot even rescale a parent.

### Harder merges — second pre-registration (written before the second run)

Five variants were added on top of the original six so the detector meets shapes it has not
seen: unequal weights, three parents, a saliency-shaped weighting, merge-then-finetune drift, and
a deliberately non-convex elementwise merge. Existing thresholds do not move: exact match at
cos ≥ **0.9995**, mixture gain ≥ **0.30**, component floor **0.03**, ±0.02 band for every
injected global weight. New constants, all fixed before the run:

| constant | value | meaning |
|---|---|---|
| `SKEWED_ALPHA` | 0.80/0.20 | unequal two-parent merge |
| `THREE_ALPHA` | 0.50/0.30/0.20 | plain three-parent merge |
| `SALIENCY_ALPHA` | 0.70/0.20/0.10 | REAM-shaped saliency weighting (injected, not estimated) |
| `NOISY_SIGMA` | 0.01 | per-element Gaussian drift as a fraction of the merged expert's own per-role std |
| `TIES_TRIM` | 0.20 | fraction of each parent's delta-to-mean trimmed by magnitude |
| `TIES_ALPHA` | 1/3 each | summary target for the fitted α, **not** a construction weight |
| `TIES_ALPHA_TOL` | ±0.10 | wider band because TIES has no global α |

| variant | origin | expectation |
|---|---|---|
| `merged_skewed` | `pre_registered` | every layer flags exactly one merged expert, α within ±0.02 of 0.80/0.20, gain ≥ 0.30; the E−2 untouched experts match exactly. On a near-duplicate MoE the merged expert may also clear the exact-cosine floor; that shadowing is reported, not scored as a failure. |
| `merged_three` | `pre_registered` | exactly one merged expert with parents 0, 1, 2 as components, α within ±0.02 of 0.50/0.30/0.20, gain ≥ 0.30; the E−3 untouched experts match exactly. A fit that spreads weight onto a third-party parent is a miss and is the identifiability limit this variant probes. |
| `merged_saliency` | `pre_registered` | same structure with α within ±0.02 of 0.70/0.20/0.10 and gain ≥ 0.30; the 0.10 component is still 3.3× the component floor. |
| `merged_noisy` | `pre_registered` | exactly one merged expert, α within ±0.02 of 0.60/0.40, gain ≥ 0.30; the E−2 untouched experts match exactly. The merge is no longer exact, so gain must drop from ~1.0 but stay above the floor. |
| `merged_ties` | `pre_registered` | exactly one merged expert with parents 0, 1, 2 as components, each α within ±0.10 of 1/3, gain ≥ 0.30; the E−3 untouched experts match exactly. **A legitimate miss is any other honest outcome** — gain below 0.30, a different α, or a different label — because TIES is outside the global convex span. It is expected when the merged parents are near-orthogonal (OLMoE), where no global α exists to recover. |

Model-specific expectations, fixed before the run:

- **TinyMixtral**: `merged_skewed`, `merged_three`, `merged_saliency` and `merged_noisy` should
  hold; the skewed merge may shadow its dominant parent because the experts are near-duplicates.
  `merged_noisy` should keep α near 0.60/0.40 (the noise averages over millions of elements) and
  lose only a few percent of gain, because the best-single residual on near-collinear parents is
  already small. `merged_ties` is the borderline case: the elementwise subsets are not a global
  mixture, so gain depends on the three parents' mutual cosines and may land either side of 0.30.
- **OLMoE**: the four convex merges should hold; `merged_ties` is expected to miss the gain floor
  (parents are near-orthogonal, so the per-element sign choices do not collapse to one α). That
  miss is the finding, not a process failure.
- **RNG streams**: the new variants take ordinals 6–10 in `VARIANT_ORDINAL`; the original six
  keep ordinals 0–5, so their streams, and therefore their rebuilt files, are unchanged.
- **Scale-repeat gate**: `SECONDARY_BUDGET_S` moves **120 → 300 s** because the suite grew from
  six to eleven variants at the same per-variant cost. This gate bounds runtime only; it is not
  a detector threshold and does not touch any verdict. The change is disclosed here rather than
  silently absorbed.
- **Detector**: reused unchanged. If a variant exposes a real bug, the change and its reason are
  recorded in `## Results` and in the task report.

## Method notes

Variant construction (all in memory from the parent, then written to
`.scratch/moe-expert-lineage/<model>/<variant>/` — tensors only plus `manifest.json`; not a
loadable model):

- RNG stream per variant and layer: `np.random.default_rng([seed, VARIANT_ORDINAL[variant], layer])`.
  The ordinal map lives in [config.py](config.py) and is keyed by variant name, not by tuple
  position, so inserting or reordering `VARIANTS` cannot re-key the streams of the other variants.
- `permuted`: per-layer π; expert tensors and router rows permuted identically.
- `pruned`: 2 experts per layer dropped, chosen by the seeded RNG (not a fixed pair); survivors
  copied verbatim in parent order; router rows kept for survivors only.
- `merged`: experts 0 and 1 replaced by `0.6·E0 + 0.4·E1` in all three projections, E1 dropped →
  E−1 experts in slot order `[mix, E2, …, E(E−1)]`. Router row 0 is mixed with the same 0.6/0.4
  weights and rows for E2… are kept (the spec fixes only the expert projections; the detector
  shape-validates the router against the declared expert count but does not score it).
- `merged_permuted`: `merged`, then a permutation of its E−1 slots (tensors and router rows).
- `merged_skewed`: experts 0 and 1 replaced by `0.8·E0 + 0.2·E1`, E1 dropped (E−1 experts);
  router row 0 mixed with the same weights.
- `merged_three`: experts 0, 1 and 2 replaced by `0.5·E0 + 0.3·E1 + 0.2·E2`, E1/E2 dropped
  (E−2 experts); router row 0 mixed with the same weights.
- `merged_saliency`: `0.7·E0 + 0.2·E1 + 0.1·E2` over the same three parents, E−2 experts. This
  is *REAM-shaped* — the weights simulate what a saliency scorer would assign — but no saliency
  is estimated; it isolates the arithmetic, not the saliency heuristic.
- `merged_noisy`: the `merged` 0.6/0.4 expert, then every element gets seeded Gaussian noise at
  1% of that projection's own population std (`NOISY_SIGMA`), computed in float64 and cast back
  to the checkpoint dtype. The router stays the exact 0.6/0.4 mix (the detector does not score
  it). Drift is injected once, not iterated, which models merge-then-finetune drift rather than
  a full fine-tuning trajectory.
- `merged_ties`: three-parent TIES-shaped merge. Take the elementwise mean of parents 0, 1, 2;
  per parent and projection, zero the smallest-magnitude 20% of its delta to that mean; per
  element elect the sign of the largest surviving |delta|; average the surviving deltas carrying
  that sign; add the average back to the mean. Router rows get the same operation. Disclosures:
  the base is the parents' mean (not a pretrained base), the trim is per projection tensor,
  sign election is by largest magnitude rather than the sum of task vectors (the deltas sum to
  zero, so a sum-based election would be degenerate), and an element with no surviving delta
  falls back to the mean. With three parents this reduces elementwise to "pick a parent or the
  average of a subset", which is inside the hull but not a global convex mixture.
- `control`: every expert replaced by seeded Gaussian tensors of the same shape at the parent's
  per-projection std for that layer; router untouched.
- Field recorded as ground truth (in `results.json` only, never inside the variant directory):
  π, kept/dropped slots, merge parents/weights/slot, plus the drift sigma for `merged_noisy` and
  the trim fraction/kind for `merged_ties`. For `merged_permuted`, `suspect_slot` and `kept` are
  recomputed in the written checkpoint's slot space (the detector's `new` list must equal
  `[suspect_slot]`, which is what the scoring checks). The aligned harder merges also record
  `kept_slots`, so a shadowed merge does not hide a missing survivor.

Detector (label-blind; reads only `manifest.json` and the safetensors shards of the suspect, plus
the pinned parent snapshot):

1. **Descriptor** per expert: one 32×32 window from each projection, at an offset derived from
   `sha256(seed|role|layer|HxW)` — independent of the expert slot — flattened and L2-normalised
   (3 × 1024 = 3072 dims).
2. **Matching**: per-layer cosine matrix (parent rows × suspect rows), Hungarian assignment
   maximising cosine. A pair is `exact` at cos ≥ 0.9995, otherwise `candidate`.
3. **Verdict per layer**: `matched` counts exact pairs; `missing` = parent experts without an
   exact match; `new` = suspect experts without an exact match. `new` is an assignment-level
   result: an unassigned suspect whose fit is one parent at full weight is still classified
   `exact`, with `duplicate_of` naming that parent and `cos` set to its maximum parent cosine
   (never NaN) — duplicated or collapsed experts are plausible in upcycled MoEs.
4. **Mixture fit** for every suspect over every parent: `y ≈ Σ αᵢ xᵢ`, α ≥ 0 and Σα = 1, solved
   by NNLS on the float64 Gram matrix of the concatenated projections and then normalised. A
   suspect is labelled `merged` when `gain = 1 − RSS(mix)/RSS(best single) ≥ 0.30` and at least
   two fitted components are ≥ 0.03. An exact copy fit has `gain = null`: the free-scale
   best-single RSS is already ~0, so the ratio is undefined. A near-duplicate basis can instead
   fit one parent at α = 0.999915 with 8.5e−5 on a second, which is why the label needs a
   component floor as well as a gain floor. Fitting every suspect over every parent (rather than
   only the unmatched suspects over the missing parents) is what catches a shallow merge whose
   cosine still clears the exact floor. "Best single" is the free-scale least-squares fit of `y`
   on one basis expert — a conservative bar, since the mixture may not rescale while the single
   expert may, so control gains can (and do) come out negative.
5. **Reporting**: each fit carries `best_single_parent` and `degenerate`; merged entries are also
   listed in `near_exact_merges` when cos ≥ 0.995 (reported only) and in `shadowed_merges` when
   cos ≥ 0.9995 (matching alone would have called it a copy).

Disclosures — parts that are ours, not in the spec:

- **Router handling** for `merged` / `merged_permuted` (same 0.6/0.4 weights on the merged router
  row) and untouched for `control`. No detector stage scores the router; `analyze_layer` reads its
  declared shape from the safetensors slice (without materialising the tensor) and requires the row
  count to match the declared expert count — a malformed header is a `SKIP`.
- **`best single` comparator** uses a free scale factor rather than a simplex vertex, as described
  above.
- **All-suspect fit**: the mixture fit runs for every suspect over every parent, not only the
  unmatched suspects over the unmatched parents. This is what catches a shallow merge whose
  cosine still clears the exact floor.
- **Duplicate label**: an unassigned suspect whose fit is one parent at full weight is classified
  `exact` with `duplicate_of` naming that parent, even though the assignment-level `new` list
  still contains it; duplicated or collapsed experts are plausible in upcycled MoEs.
- **Untrusted-metadata hardening** (`CheckpointReader`, `read_manifest`): `manifest.json` and the
  shard index are size-capped at 4 MiB; every parse failure (`UnicodeDecodeError`, malformed JSON,
  `RecursionError`, non-object document) maps to `DetectionError`, i.e. a structural abstain;
  `weight_map` values and `tensor_file` must be plain `*.safetensors` filenames inside the
  checkpoint directory (absolute paths, `..` and separators are rejected); a tensor is bounded
  before it is materialised (256 MiB, from slice metadata); `num_experts` must be a positive int
  (bools and floats rejected) and bounded by 512; the manifest may declare at most 256 layers, with
  canonical `"<int>"` keys and no aliases. The expert tensors physically present for a layer are
  counted from the checkpoint's own keys and must equal the declared count, so an understated
  manifest cannot hide an expert. Non-finite tensor values that reach the mixture fit raise
  `DetectionError` rather than producing a `NaN` gain. These bounds are extensions for the
  database-extractor reuse; none of them is on the label path.
- **Structural-skip probe** (`robustness_probe` in [smoke.py](smoke.py)): a deliberately malformed
  mini-checkpoint tree is scored to prove a malformed input becomes a `SKIP` line and a
  `DetectionError`, never a crash. 13 cases: missing tensor, truncated file, manifest without
  layers, relative and absolute index redirects, non-object index, a NaN outside the 32×32 window,
  an understated expert count, float and bool `num_experts`, an aliased layer key, a deeply nested
  manifest, and a 4 MiB manifest. Results go to `results.json` under `models[0].robustness`. It is
  not part of the pre-registered expectations.
- **Scale repeat gate**: the 600 s budget is compared against the measured wall time of the whole
  primary phase (build + write + detect), which is what the spec's "runtime" is taken to mean.

### Re-baselines (review follow-up, same day)

- `MIN_COMPONENT` **0.04 → 0.03**. The shallow claim injects α = 0.05 and the pre-registered band
  is ±0.02, so the band admits a 0.03 second component; a 0.04 component floor silently narrowed
  the claim it was supposed to score. 0.03 = `SHALLOW_ALPHA[1] − ALPHA_TOL` makes the two
  consistent. This changes no label, matched/missing/new count, α verdict or gain in either run:
  the observed shallow component is 0.050000001 while the fitted weights are accurate to ~1e−8
  (fp32) and 4e−6 (bf16), so nothing sits near the floor. The only payload effect is that the
  `components` list of noisy control fits now reports fitted components down to 0.03 instead of
  0.04.
- Variant RNG streams are now keyed by the frozen ordinal map instead of tuple position. That
  restores the stream the six-variant results were built on (inserting `merged_permuted` had
  silently re-keyed `control`) and makes the next insertion a no-op for existing variants.
- Reporting fixes that do not move a verdict: `gain` is clamped to the simplex optimum (RSS ≥ 0,
  gain ≤ 1); the "no fit ≥ 0.30" checks and `max gain` are read from **all** classifications, not
  only entries labelled `merged`; unassigned suspects report their maximum parent cosine instead
  of NaN; `results.json` is strict JSON (`allow_nan=False`).
- Second-suite reporting fix (same day, does not move a verdict): the first pass of the harder
  merges printed no α/gain for `merged_ties` on TinyMixtral because the injected fit was off the
  expected three-parent basis, and the scorer had skipped off-basis entries. The scorer now
  records the injected suspect's label, gain, α-length, top component and top component weight
  regardless of basis; the verdict rule (one `merged` label, on-basis, in-band, gain ≥ 0.30,
  kept parents exact) is unchanged. The committed `results.json` is from the final pass after the
  rebase; every verdict and every original number is identical.
- Scale-repeat gate raised **300 → 600 s** in the review follow-up: runtime only, disclosed here
  rather than silently absorbed. It gates the OLMoE repeat, never a verdict.

## Results

Exit 0, seed 20261029, thresholds and bands exactly as pre-registered apart from the documented
0.04 → 0.03 component floor (that guard was re-baselined for consistency; nothing else was
retuned) and the disclosed scale-repeat gate (raised to 600 s in the review follow-up, and it
cleared, so the OLMoE repeat ran). Wall-clock timings and artifact sizes are not quoted here
because they change on every rerun; read them from the run's
`results.json` (gitignored) (`generated_utc`, each model's `seconds`, `total_seconds`, and each
variant's `file_bytes`). Every scored number below is read from that file. The detector is
unchanged from the six-variant run; only `config.py` (new constants/expectations), `synth.py` (new
construction) and `smoke.py` (new scoring/diagnostics) moved.

**TinyMixtral-4x248M-MoE** (12 layers × 4 experts; `matched` is per layer; `fits` counts every
suspect fitted):

| variant | layers | matched | missing / new | fitted α | gain | merged cos | max pair cos | fits | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `permuted` | 12 | 4/4 each (48/48) | – / – | – | – | – | 1.000000 | 48 | HELD |
| `pruned` | 12 | 2/4 each (24/24) | 2 / 0 per layer, missing == injected drops | – | no defined gain (24 N/A) | – | 1.000000 | 24 | HELD |
| `merged` | 12 | 2/4 + 1 new each | 2 ([0,1]) / 1 ([0]) | 0.600000 / 0.400000 | 0.9999999999998–0.9999999999999 | 0.997130–0.992996 | 1.000000 | 36 | HELD |
| `merged_shallow` | 12 | 3/4 each (36/36) | 1 ([1]) / 0 | 0.950000 / 0.050000 | 0.9999999999808–1.000000 | 0.999895–0.999956 | 1.000000 | 36 | HELD |
| `merged_permuted` | 12 | 2/4 + 1 new each | 2 ([0,1]) / 1 (injected slot) | 0.600000 / 0.400000 | 0.9999999999998–0.9999999999999 | 0.997130–0.992996 | 1.000000 | 36 | HELD |
| `control` | 12 | 0/4 each (0/48) | 4 / 4 per layer | 14 of 48 fits degenerate (α = 0) | max −1.10e−8 | – | 0.043741 | 48 | HELD |

Recovered π equalled the injected π on 12/12 `permuted` layers, and on 12/12 `merged_permuted`
layers the detector's `new` slot equalled the injected `suspect_slot` (`[1,0,0,2,2,0,0,0,1,1,1,2]`
by layer) with the two kept parents in the matching slots. Injected drops were matched exactly:
L0 `[1,2]`, L1 `[1,3]`, L2 `[1,2]`, L3 `[2,3]`, L4–L7/L10/L11 `[0,1]`, L8 `[2,3]`, L9 `[1,3]`.
Fitted α was 0.600000006 / 0.399999994 on every merged layer; the shallow fit returned
0.949999999 / 0.050000001 with gain ≥ 0.9999999999808.

**OLMoE-1B-7B-0924** (layers 0 and 8 × 64 experts):

| variant | layers | matched | missing / new | fitted α | gain | merged cos | max pair cos | fits | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `permuted` | 2 | 64/64 each (128/128) | – / – | – | – | – | 1.000000 | 128 | HELD |
| `pruned` | 2 | 62/64 each (124/124) | 2 / 0 (L0 `[26,38]`, L8 `[47,55]`) | – | no defined gain (124 N/A) | – | 1.000000 | 124 | HELD |
| `merged` | 2 | 62/64 + 1 new | 2 ([0,1]) / 1 ([0]) | 0.599998 / 0.399993 | 0.9999907–0.9999911 | 0.818566 / 0.847733 | 1.000000 | 126 | HELD |
| `merged_shallow` | 2 | 62/64 each | 2 ([0,1]) / 1 ([0]) | 0.949981 / 0.050003 | 0.998910–0.998985 | 0.998436 / 0.998781 | 1.000000 | 126 | HELD |
| `merged_permuted` | 2 | 62/64 + 1 new | 2 ([0,1]) / 1 (L0 slot 16, L8 slot 13) | 0.599998 / 0.399993 | 0.9999907–0.9999911 | 0.818566 / 0.847733 | 1.000000 | 126 | HELD |
| `control` | 2 | 0/64 each (0/128) | 64 / 64 | 0 of 128 fits degenerate, all α spread thin | max −0.035182 | – | 0.067209 | 128 | HELD |

`merged cos` is the cosine the detector assigned to the merged suspect (its assigned pair, or its
maximum parent cosine when the assignment left it `new`); `max pair cos` is the largest entry of
that model's parent × suspect matrices. Per-layer files: `results.json` holds every layer's matched
count, missing/new lists, recovered π, candidate pairs with cosines, and for every fit its α, gain,
`best_single_parent`, `degenerate` flag and (for duplicates) `duplicate_of`. There are no `NaN`
tokens in the file.

**Merged-expert descriptor cosine per layer** — the margin that matters:

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TinyMixtral | .997130 | .995812 | .995707 | .994878 | .994753 | .994558 | .994571 | .993857 | .994205 | .993385 | .994554 | .992996 |
| TinyMixtral, 0.95/0.05 | .999956 | .999937 | .999935 | .999923 | .999921 | .999918 | .999918 | .999907 | .999913 | .999900 | .999919 | .999895 |

OLMoE layers 0 / 8: 0.818566 / 0.847733 for 0.6/0.4 and 0.998436 / 0.998781 for 0.95/0.05.
Exact-match floor 0.9995 in both runs.

### Harder merges (second suite)

`max |dα|` is the largest layer × component deviation from the injected global weights. For
`merged_ties` the α column is not an injected weight: it reports what the global fit actually did,
because the construction has no global α.

**TinyMixtral-4x248M-MoE** (12 layers × 4 experts):

| variant | layers | matched | fitted α | max \|dα\| | gain | merged cos | label | verdict |
|---|---|---|---|---|---|---|---|---|
| `merged_skewed` | 12 | 2/4 + 1 new each | 0.800000 / 0.200000 | 3.7e−9 | 0.999999999999–1.000000 | 0.998289–0.999288 | merged 12/12 | HELD |
| `merged_three` | 12 | 1/4 + 1 new each | 0.500000 / 0.300000 / 0.200000 | 8.2e−9 | 0.9999999999998–1.000000 | 0.988306–0.992861 | merged 12/12 | HELD |
| `merged_saliency` | 12 | 1/4 + 1 new each | 0.700000 / 0.200000 / 0.100000 | 7.9e−9 | 0.9999999999992–0.99999999999996 | 0.996760–0.998206 | merged 12/12 | HELD |
| `merged_noisy` | 12 | 2/4 + 1 new each | 0.599987 / 0.399997 | 4.3e−5 | 0.983777–0.992531 | 0.992952–0.997081 | merged 12/12 | HELD |
| `merged_ties` | 12 | 1/4 + 1 new each | off-basis: α₂ 0.9616–0.9856, α₀ ≤ 0.0384, α₁ < 0.03 | n/a | 0.008495–0.021996 | 0.824329–0.924424 | unexplained 12/12 | MISSED |

**OLMoE-1B-7B-0924** (layers 0 and 8 × 64 experts; cosines are per layer):

| variant | layers | matched | fitted α | max \|dα\| | gain | merged cos | label | verdict |
|---|---|---|---|---|---|---|---|---|
| `merged_skewed` | 2 | 62/64 + 1 new | 0.799992 / 0.199995 | 8.4e−6 | 0.9999496–0.9999529 | 0.966648 / 0.973570 | merged 2/2 | HELD |
| `merged_three` | 2 | 61/64 + 1 new | 0.499990 / 0.300002 / 0.199998 | 1.1e−5 | 0.9999913–0.9999917 | 0.803852 / 0.827042 | merged 2/2 | HELD |
| `merged_saliency` | 2 | 61/64 + 1 new | 0.699986 / 0.200002 / 0.100000 | 1.4e−5 | 0.9999674–0.9999694 | 0.948785 / 0.957348 | merged 2/2 | HELD |
| `merged_noisy` | 2 | 62/64 + 1 new | 0.599969 / 0.399976 | 3.7e−5 | 0.9996359–0.9996530 | 0.818556 / 0.847736 | merged 2/2 | HELD |
| `merged_ties` | 2 | 61/64 + 1 new | 0.351162 / 0.331776 / 0.301031 | 0.037734 | 0.150889–0.156387 | 0.320959 / 0.362532 | unexplained 2/2 | MISSED |

What the numbers say:

- **Every convex injected merge held.** Unequal two-parent (0.80/0.20), three-parent
  (0.50/0.30/0.20), saliency-shaped (0.70/0.20/0.10) and noisy (0.60/0.40 + 1% drift) merges
  were all labelled `merged` on every layer of both models, with every kept parent matched
  exactly. The two convex three-parent fits never spread onto the kept expert 3
  (`components_spread_layers = 0` on 12/12 and 2/2), even on TinyMixtral where parent 2 is nearly
  orthogonal to the near-duplicate pair 0↔1 (0↔2 cosine −0.030…0.025). An exact convex target is
  identifiable even when the parent set is overcomplete.
- **The skewed merge did not shadow on TinyMixtral.** Its cosines are 0.998289–0.999288, below
  the 0.9995 exact floor, so matching called it `new` on 12/12 (0/12 shadowed). This is the
  contrast with 0.95/0.05: 20% of the second parent is enough to leave the exact-copy band even
  when the two parents are near-duplicates (0↔1 cosine 0.9546–0.9819), while 5% is not.
- **1% drift is comfortably inside the working envelope.** Against the exact merge, gain fell
  from ≥ 0.9999999999998 to 0.983777–0.992531 on TinyMixtral (a 0.0075–0.0162 drop) and from
  0.9999907–0.9999911 to 0.9996359–0.9996530 on OLMoE (a 3.4e−4 drop). The weights barely moved:
  max |dα| 4.3e−5 (fp32) and 3.7e−5 (bf16), because the noise averages over millions of
  elements. The fit stays well above the 0.30 floor; 1% is not the boundary.
- **TIES is the boundary, and the two models fail for different reasons.** On OLMoE the parents
  are near-orthogonal (all pairwise cosines ≤ 0.022), so the global fit stays on the intended
  three-parent basis and returns α = 0.351/0.332/0.301 — within 0.038 of the 1/3 summary target —
  but gain is only 0.1509–0.1564, below 0.30. The detector therefore calls it `unexplained`,
  which is the honest answer: an elementwise sign-elected merge is not a global convex mixture.
  On TinyMixtral the construction itself collapses: parents 0↔1 are near-duplicates
  (0.9546–0.9819) while 0↔2 and 1↔2 are ≈ 0 (−0.0322…0.0300), so after trimming, parent 2's
  large delta wins the sign election almost everywhere. The merge becomes parent 2 alone
  (top component 0.9616–0.9856, gain 0.0085–0.0220, cosine to its parent 0.824–0.924), and the
  fit reports a one- or two-component basis rather than the intended three. That is a miss of
  the pre-registered summary target and exactly the legitimate-miss clause: TIES is outside the
  global convex span, and the detector reports the residual instead of forcing a merge label.
- **Plain-language boundary.** The detector's mixture contract works whenever the derived expert
  is a fixed convex combination of parents: unequal weights, three parents, a small 0.10
  component, and 1% elementwise drift are all inside it. It stops when the combination is
  elementwise non-convex (TIES): with near-orthogonal parents there is no global α and gain
  falls to ~0.15; with heterogeneous parent deltas the construction can collapse to a single
  parent, which the fit correctly calls `unexplained`, not `merged`. A 1% drift is far from the
  gain floor; the fit would need noise comparable to the parent-difference signal before the
  gain approached 0.30, which this suite did not probe.

**Structural-skip probe**: 13/13 cases raised `DetectionError` and were recorded as skip reasons —
missing tensor, truncated safetensors file, manifest without layers, relative and absolute
`weight_map` redirects, non-object index, NaN outside the sampled window, understated expert count,
float and bool `num_experts`, aliased layer key, deeply nested manifest, oversized manifest. No
case crashed the run.

Reading:

Twenty of the twenty-two expectation verdicts (eleven variants × two models) held: ten
pre-registered claims plus the post-hoc `merged_shallow` restatement, all with the a-priori
thresholds. The two misses are `merged_ties` on both models, which is the legitimate-miss
case written before the run: the detector reports a residual instead of forcing a merge. Every
convex injected merge — including unequal weights, three parents, a 0.10 component and 1% drift —
held on both the 4-expert fp32 model and the 64-expert bf16 model. The weak point remains the
0.0024 cosine headroom that a collapsed-expert MoE leaves between a 0.6/0.4 merge and an "exact
copy" verdict; the new boundary is the non-convex elementwise merge.

- Every convex harder merge held on both models: the three-parent fits returned all three
  injected weights within ±0.02 (max error 1.4e−5 on bf16), the 0.10 saliency component was not
  swallowed by the 0.03 component floor, and the noisy merge stayed within 4.3e−5 of 0.60/0.40
  while losing at most 0.0162 of gain. The 1% drift is far inside the envelope. The exact-merge
  gains are ≈ 1.0 because the construction is exact; they validate the estimator, not a claim
  that real-world merges are convex.
- The TIES miss is informative in both directions. On OLMoE the fitted α is close to the 1/3
  summary target (max error 0.038) but gain is 0.151–0.156, so the fit says "no global convex
  mixture explains this" — the correct reading of an elementwise sign-elected merge. On
  TinyMixtral the construction itself degenerates to parent 2 (top α 0.962–0.986, gain
  0.0085–0.0220) because the other two parents are near-duplicates of each other and nearly
  orthogonal to it; trimming removes their small deltas and sign election keeps the large one.
  A detector cannot recover a three-parent mixture that the merge script did not produce.
- One pre-run estimate in the second suite was optimistic: `merged_ties` was expected to be
  "borderline" on TinyMixtral, with gain possibly either side of 0.30. It landed far below
  (0.0085–0.0220) because the three parents are not exchangeable. The registered claim and the
  legitimate-miss clause were left untouched; this is a miss, not a re-baseline.
- Every original six-variant expectation held again with the same numbers, confirming that
  ordinals 6–10 did not re-key any existing RNG stream.
- One pre-run estimate was wrong and is recorded rather than fixed: the reasoning above expected
  the merged expert at cos ≈ 0.83, which is what OLMoE produced (0.819/0.848) but not TinyMixtral
  (0.993–0.997). The estimate assumed near-orthogonal experts; TinyMixtral violates that. The
  registered threshold and band were left untouched and the expectation still held, but the
  assumed margin was 20× larger than the observed one.
- The real finding is in the margin, not the verdict. TinyMixtral's experts are near-duplicates:
  on layer 0 the mean off-diagonal descriptor cosine is 0.4912, the maximum 0.9908, and experts
  0↔1 sit at 0.9819 (full-tensor cosine 0.974–0.984 across the three projections). Post-hoc
  diagnostic, not on the verdict path: a 0.6/0.4 mixture of two such experts is *almost* its
  dominant parent, which is why the merged expert lands at 0.993–0.997 — only 0.0024 under the
  0.9995 exact floor. The `merged_shallow` variant (0.95/0.05) confirms the consequence directly:
  on TinyMixtral the merged expert sits at cosine 0.9999, the Hungarian matcher pairs it with its
  dominant parent and calls it an exact copy on **12/12** layers (`matched` 3/4, `missing [1]`, no
  `new` expert), while the mixture fit still recovers α = 0.95/0.05 with gain ≥ 0.9999999999808.
  On OLMoE the same merge drops to 0.9984 (matcher calls it `new`: `matched` 62/64,
  `missing [0,1]`, `new [0]`) because its experts are almost orthogonal (mean descriptor cosine
  0.001 on L0 / 0.0005 on L8, max 0.063 / 0.076). Practical consequence: on a fine-tuned
  MoE with collapsed experts, a cosine "exact" verdict is not evidence that the expert was copied;
  the mixture fit (α with gain ≥ 0.30, run over every suspect) is the component that still
  separates. Both regimes sit inside the reported 0.995 near-exact band (TinyMixtral 12/12 shallow
  layers, OLMoE 2/2), but only TinyMixtral's cross the 0.9995 shadowing line (`shadowed_merges`
  12/12 vs 0/2).
- The control's separation is comfortable (0.043741 / 0.067209 max cosine against parents over
  48 / 128 fits; max gain −1.10e−8 / −0.035182), with caveats. iid Gaussian noise is the
  easiest possible negative, and 14/48 TinyMixtral control fits were degenerate (NNLS returns
  α = 0, i.e. "no non-negative combination explains this"), which is the right answer for noise
  but makes the maximum gain uninformative there. The pruned survivors are the opposite case: all
  24 / 124 fits have `gain = null` because an exact copy is explained by its own parent. Those
  fits are scored as **not applicable**, not as negative gains: `summary.undefined_gain_fits` is
  24/24 and 124/124, `over_gain_fits` is 0, and the clause is reported as "no defined gain
  ≥ 0.30" over an empty defined set rather than as a vacuous pass.
- The pruned-fit behaviour is also the documented limit of the estimator: with exactly duplicated
  parents the simplex α is not identifiable, and the descriptor is a 32×32 sample per projection,
  so "exact" is a descriptor-exact verdict. A derivative that matches only the sampled windows
  would be called exact; the sampling is disclosed above and the fit is the cross-check.
- Scope: the ground truth is synthetic and the variants were built by the same person who wrote
  the detector, so this probe fixes the detector contract and the failure mode above; it says
  nothing about detecting a genuine merge produced by a third-party script, which would also
  perturb the router and possibly rescale.

**Second pre-registration, after a miss (recorded, not hidden).** The shallow-merge claim was
first written as "the E−2 untouched experts match exactly and the merge is left as an exact copy".
On TinyMixtral it held 12/12; on OLMoE it missed 0/2, because 5% of a *distinct* expert already
drops the descriptor cosine to 0.9984 — below the 0.9995 floor — so matching alone calls the merge
`new`, not a copy. The claim bundled a model-specific premise (near-duplicate experts) into a
model-agnostic test. It was restated as: exactly one expert is flagged `merged` with α in band and
gain ≥ 0.30, *whether or not* cosine matching would have called it a copy, with the copy call
reported as a diagnostic (`shadowed_merges`). No threshold or band changed, and the restated claim
is what `config.EXPECTATIONS` scores. Observed shadowing is the finding: **12/12** TinyMixtral
layers would have been called copies by matching alone, **0/2** on OLMoE.

**Third pre-registration outcome (TIES), recorded not hidden.** `merged_ties` was registered with
an explicit legitimate-miss clause and missed on both models: OLMoE stayed on the three-parent
basis with α within 0.038 of 1/3 but gain 0.151–0.156, and TinyMixtral went off-basis and
collapsed to parent 2 (top α 0.962–0.986, gain 0.0085–0.0220). No threshold, band, component floor
or label rule changed. The detector was reused unchanged. The only code changes are the new
construction/constants, the scoring and diagnostics for the new variants, and the disclosed
scale-repeat gate.

## Ops

```bash
uv run --group smokes python smokes/moe-expert-lineage/smoke.py
```

Artifacts: `.scratch/moe-expert-lineage/<model>/<variant>/` (safetensors + manifest; 1.1–2.4 GB per
TinyMixtral variant and 1.56–1.61 GB per OLMoE variant, 33.6 GiB total — delete the tree to reclaim
space) and [`results.json`](results.json) next to this file. Exit 0 means the smoke ran and every
expectation is labelled held or missed — a miss is a finding, not a process failure; exit 1 only if
the primary parent is unusable. `results.json` is written before any table is printed, so a
reporting error cannot lose a run. Stdout is the per-layer table plus the expectation verdicts; the
JSON holds the same records (matched count, missing/new lists, recovered π, fitted α and gain per
variant and layer).
