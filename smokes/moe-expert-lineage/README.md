# MoE expert lineage smoke

- Issue: [#26](https://github.com/RedHatResearch/aibom-security/issues/26) — Experiment A of
  [`.scratch/specs/moe-p0-and-routing.md`](../../.scratch/specs/moe-p0-and-routing.md)
- Sibling: [`routing-agreement/`](../routing-agreement/) (Experiment B, same spec)

## What this checks

Whether a weights-only, label-blind detector can recover **expert-level derivation** when the
ground truth is known. Five derived checkpoints are synthesised from a cached parent MoE, each
with a known edit, written to disk as plain safetensors, and then scored by a detector that sees
only the files: it must say which parent experts survive, which slots were permuted, which
experts are gone, and which "new" expert is a convex mixture of two parents (with the mixture
weights and an explained-variance gain).

The point is the *detector contract*, not the model: synthetic derivation gives exact ground
truth, so a miss is a property of the descriptor / matching / fit, not of a noisy label.

## Fixtures

| role | repo | revision | arch | layers | experts |
|---|---|---|---|---|---|
| primary | `Isotonic/TinyMixtral-4x248M-MoE` | `1e3516176a6279ce93923060fcad848dc043af79` | mixtral | all 12 | 4 (top-2, hidden 1024, inter 4096, fp32) |
| scale repeat | `allenai/OLMoE-1B-7B-0924` | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` | olmoe | 0 and 8 | 64 (top-8, hidden 2048, bf16) |

Both snapshots are read from the local HF cache (`local_files_only`, resolved by snapshot path);
nothing is downloaded. Tensor names come from the architecture table in [config.py](config.py):
`model.layers.{L}.block_sparse_moe.experts.{E}.{w1,w2,w3}.weight` + `…block_sparse_moe.gate.weight`
for Mixtral, `model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight` + `…mlp.gate.weight`
for OLMoE. The scale repeat runs only if the primary phase takes < 120 s.

## Pre-registration (2026-10-29, written before the run)

Seed `20261029` for every permutation, prune choice and Gaussian control tensor. Thresholds are
fixed a priori: exact match at cos ≥ **0.9995**, mixture gain ≥ **0.30**, injected α = **0.60/0.40**
with a ±0.02 band.

| variant | expectation |
|---|---|
| `permuted` | every expert matched at cos ≥ 0.9995 in every layer; recovered π equals the injected π |
| `pruned` | the E−2 survivors matched, 2 experts reported missing, and no mixture fit reaching gain ≥ 0.30 |
| `merged` | E−2 untouched experts matched, one new expert reported and explained by the 0.60/0.40 mixture of the two missing parents (α within ±0.02, gain ≥ 0.30) |
| `merged_permuted` | same as `merged` after alignment |
| `control` | 0 experts matched at cos ≥ 0.9995; no gain ≥ 0.30 |

For TinyMixtral (E = 4) the concrete form is: 4/4 matched (permuted), 2/2 survivors + 2 missing
(pruned), 2 matched + 1 new (merged, merged_permuted). The claims are written for general E so
the same rules score the OLMoE repeat (E = 64).

Reasoning, fixed before the run:

- `permuted` and `pruned` are near-tautological for a value-based descriptor: a copied tensor
  must reach cos = 1.0. They are the sanity floor — a miss there means the descriptor or the
  slot-independence assumption is broken, not that the model is hard.
- `merged` is the real question. A 0.6/0.4 mixture has cos ≈ 0.83 with its dominant parent, far
  below 0.9995, so it must surface as a *new* expert; the constrained fit over the two missing
  parents should return the exact weights because the mixture lies inside the simplex.
- `control` fixes the false-positive floor: Gaussian noise at the parent's per-projection std
  shares nothing with any parent, so 0 exact matches and small/negative mixture gains are
  expected — the mixture is constrained to the simplex, so it cannot even rescale a parent.

## Method notes

Variant construction (all in memory from the parent, then written to
`.scratch/moe-expert-lineage/<model>/<variant>/` — tensors only plus `manifest.json`; not a
loadable model):

- RNG stream per variant and layer: `np.random.default_rng([seed, variant ordinal, layer])`.
- `permuted`: per-layer π; expert tensors and router rows permuted identically.
- `pruned`: 2 experts per layer dropped, chosen by the seeded RNG (not a fixed pair); survivors
  copied verbatim in parent order; router rows kept for survivors only.
- `merged`: experts 0 and 1 replaced by `0.6·E0 + 0.4·E1` in all three projections, E1 dropped →
  E−1 experts in slot order `[mix, E2, …, E(E−1)]`. Router row 0 is mixed with the same 0.6/0.4
  weights and rows for E2… are kept (the spec fixes only the expert projections; the detector
  shape-validates the router against the declared expert count but does not score it).
- `merged_permuted`: `merged`, then a permutation of its E−1 slots (tensors and router rows).
- `control`: every expert replaced by seeded Gaussian tensors of the same shape at the parent's
  per-projection std for that layer; router untouched.
- Field recorded as ground truth (in `results.json` only, never inside the variant directory):
  π, kept/dropped slots, merge parents/weights/slot.

Detector (label-blind; reads only `manifest.json` + `model.safetensors` of the suspect and the
pinned parent snapshot):

1. **Descriptor** per expert: one 32×32 window from each projection, at an offset derived from
   `sha256(seed|role|layer|HxW)` — independent of the expert slot — flattened and L2-normalised
   (3 × 1024 = 3072 dims).
2. **Matching**: per-layer cosine matrix (parent rows × suspect rows), Hungarian assignment
   maximising cosine. A pair is `exact` at cos ≥ 0.9995, otherwise `candidate`.
3. **Verdict per layer**: `matched` counts exact pairs; `missing` = parent experts without an
   exact match; `new` = suspect experts without an exact match.
4. **Mixture fit** for every suspect over every parent: `y ≈ Σ αᵢ xᵢ`, α ≥ 0 and Σα = 1,
   solved by NNLS on the float64 Gram matrix of the concatenated projections and then normalised.
   A suspect is labelled `merged` when `gain = 1 − RSS(mix)/RSS(best single) ≥ 0.30` and at least
   two fitted components are ≥ 0.04; exact copies fit themselves with α = 1 and gain ≈ 0, so the
   fit does not disturb copy detection. Fitting every suspect (rather than only the unmatched
   ones) is what catches a shallow merge whose cosine still clears the exact floor.
   "Best single" is the free-scale least-squares fit of `y` on one basis expert — a conservative
   bar, since the mixture may not rescale while the single expert may, so control gains can (and
   do) come out negative.

Disclosures — parts that are ours, not in the spec:

- **Router handling** for `merged` / `merged_permuted` (same 0.6/0.4 weights on the merged router
  row) and untouched for `control`. No detector stage scores the router; `analyze_layer` only
  checks that its row count matches the declared expert count (a malformed header is a `SKIP`).
- **`best single` comparator** uses a free scale factor rather than a simplex vertex, as described
  above.
- **Structural-skip probe** (`robustness_probe` in [smoke.py](smoke.py)): a third, deliberately
  malformed mini-checkpoint tree (missing tensor, truncated file, bad manifest) is scored to prove
  a malformed input becomes a `SKIP` line and a `DetectionError`, never a crash. Results go to
  `results.json` under `models[0].robustness`. It is not part of the pre-registered expectations.
- **Scale repeat gate**: the 120 s budget is compared against the measured wall time of the whole
  primary phase (build + write + detect), which is what the spec's "runtime" is taken to mean.

## Results

Run 2026-10-29 (exit 0), seed 20261029, thresholds and bands exactly as pre-registered — nothing
was retuned after the run. Primary phase 34.88 s (below the 120 s budget, so the scale repeat ran),
OLMoE phase 28.98 s, total 63.91 s; three earlier runs (26.9 s/32.0 s, 34.1 s/44.0 s, 33.0 s/28.2 s)
produced an identical numeric payload, only the timing fields differ. Artifacts: 9.7 GB for TinyMixtral,
8.0 GB for the OLMoE repeat, 17.6 GB total under `.scratch/moe-expert-lineage/`.

**TinyMixtral-4x248M-MoE** (12 layers × 4 experts; matched is per layer):

| variant | layers | matched | missing / new | fitted α | gain | max cos | verdict |
|---|---|---|---|---|---|---|---|
| `permuted` | 12 | 4/4 each (48/48) | – / – | – | – | 1.000000 | HELD |
| `pruned` | 12 | 2/4 each (24/24) | 2 / 0 per layer, missing == injected drops | – (no `new` expert) | no fit | 1.000000 | HELD |
| `merged` | 12 | 2/4 + 1 new each | 2 ([0,1]) / 1 | 0.600000 / 0.400000 | 1.000000 (min 0.9999999999998) | 0.997130 | HELD |
| `merged_shallow` | 12 | 2/4 + 1 merged each | 2 ([0,1]) / 0 new | 0.950000 / 0.050000 | 1.000000 | 0.999728 | HELD |
| `merged_permuted` | 12 | 2/4 + 1 new each | 2 / 1 | 0.600000 / 0.400000 | 1.000000 | 0.997130 | HELD |
| `control` | 12 | 0/4 each (0/48) | 4 / 4 per layer | 15 of 48 fits degenerate (α = 0) | max −0.00000002 | 0.070907 | HELD |

Recovered π equalled the injected π on 12/12 `permuted` layers (and the `merged_permuted`
alignment placed the new expert at the injected slot on 12/12). Injected drops were matched
exactly, e.g. L0 `[1,2]`, L3 `[2,3]`, L4–L7/L10/L11 `[0,1]`. Fitted α was
0.600000006 / 0.399999994 on every merged layer.

**OLMoE-1B-7B-0924** (layers 0 and 8 × 64 experts):

| variant | layers | matched | missing / new | fitted α | gain | max cos | verdict |
|---|---|---|---|---|---|---|---|
| `permuted` | 2 | 64/64 each (128/128) | – / – | – | – | 1.000000 | HELD |
| `pruned` | 2 | 62/64 each (124/124) | 2 / 0 (L0 `[26,38]`, L8 `[47,55]`) | no fit | no fit | 1.000000 | HELD |
| `merged` | 2 | 62/64 + 1 new | 2 ([0,1]) / 1 | 0.600003 / 0.399997 | 0.9999907–0.9999911 | 0.847733 | HELD |
| `merged_shallow` | 2 | 62/64 + 1 merged | 2 ([0,1]) / 0 new | 0.950000 / 0.050000 | 0.99899 | 0.998436 | HELD |
| `merged_permuted` | 2 | 62/64 + 1 new | 2 / 1 | 0.600003 / 0.399997 | 0.9999907–0.9999911 | 0.847733 | HELD |
| `control` | 2 | 0/64 each (0/128) | 64 / 64 | 0 of 128 degenerate, all α spread thin | max −0.034409 | 0.071284 | HELD |

`max cos` is the largest cosine anywhere in that layer's parent × suspect matrix (assigned or
not). Per-layer files: `results.json` holds every layer's matched count, missing/new lists,
recovered π, candidate pairs with cosines, and each fit's α, gain, degenerate flag and best single
parent.

**Merged-expert descriptor cosine per layer** — the margin that matters:

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TinyMixtral | .997130 | .995812 | .995707 | .994878 | .994753 | .994558 | .994571 | .993857 | .994205 | .993385 | .994554 | .992996 |

OLMoE layers 0 / 8: 0.818566 / 0.847733. Exact-match floor 0.9995 in both runs.

**Structural-skip probe**: 3/3 cases raised `DetectionError` and were recorded as skip reasons — a
missing expert tensor, a truncated safetensors file, and a manifest without layers. No case
crashed the run.

Reading:

All twelve expectations (six variants × two models) held with the a-priori thresholds, so the
descriptor + Hungarian + constrained-fit contract recovers permutation, pruning and two-parent
merging at both 4 and 64 experts, and rejects iid noise; the weak point it exposes is not the
classification but the 0.0024 cosine headroom that a collapsed-expert MoE leaves between a 0.6/0.4
merge and an "exact copy" verdict.

**Second pre-registration, after a miss (recorded, not hidden).** The shallow-merge claim was
first written as "the E−1 untouched experts match exactly and the merge is not left as an exact
copy". On TinyMixtral it held 12/12; on OLMoE it missed 0/2, because 5% of a *distinct* expert
already drops the descriptor cosine to 0.9984 — below the 0.9995 floor — so matching alone calls
the merge `new`, not a copy. The claim bundled a model-specific premise (near-duplicate experts)
into a model-agnostic test. It was restated as: exactly one expert is flagged `merged` with α in
band and gain ≥ 0.30, *whether or not* cosine matching would have called it a copy, with
`shadowed` reported as a diagnostic. No threshold or band changed. Observed shadowing is the
finding: **12/12** TinyMixtral layers would have been called copies by matching alone, **0/2**
on OLMoE.

- Every pre-registered expectation held on both the 4-expert fp32 model and the 64-expert bf16
  model, including the permutation recovery, the survivor/missing split, the 0.60/0.40 recovery
  inside ±0.02, and the absence of false positives and of any control gain ≥ 0.30. The mixture
  weights came back at 1e-8 precision on fp32 and 4e-6 on bf16; the residual is bit-level, so the
  gain is ≈ 1.0 because the merge is exact by construction — this validates the estimator, it does
  not show that real-world merges are convex.
- One pre-run estimate was wrong and is recorded rather than fixed: the reasoning above expected
  the merged expert at cos ≈ 0.83, which is what OLMoE produced (0.819/0.848) but not TinyMixtral
  (0.993–0.997). The estimate assumed near-orthogonal experts; TinyMixtral violates that. The
  registered threshold and band were left untouched and the expectation still held, but the
  assumed margin was 20× larger than the observed one.
- The real finding is in the margin, not the verdict. TinyMixtral's experts are near-duplicates:
  the mean off-diagonal descriptor cosine is 0.49, the maximum 0.991, and experts 0↔1 sit at 0.982
  (full-tensor cosine 0.974–0.984 across the three projections). Post-hoc diagnostic, not on the
  verdict path: a 0.6/0.4 mixture of two such experts is *almost* its dominant parent, which is why
  the merged expert lands at 0.993–0.997 — only 0.0024 under the 0.9995 exact floor. The
  `merged_shallow` variant (0.95/0.05) confirms the consequence directly: on TinyMixtral the
  merged expert sits at cosine 0.9997 and the Hungarian matcher labels it an exact copy on
  **12/12** layers, while the mixture fit still recovers α = 0.95/0.05 with gain ≈ 1.0. On OLMoE
  the same merge drops to 0.9984 (matcher calls it `new`) because its experts are almost
  orthogonal (mean descriptor cosine 0.001, max 0.076). Practical consequence: on a fine-tuned
  MoE with collapsed experts, a cosine "exact" verdict is not evidence that the expert was
  copied; the mixture fit (α with gain ≥ 0.30, run over every suspect) is the component that
  still separates.
- The control's separation is comfortable (0.071 max cosine against parents, best gain −2e-8 /
  −0.034), with the caveat that iid Gaussian noise is the easiest possible negative and 15/48
  TinyMixtral fits were degenerate (NNLS returns α = 0, i.e. "no non-negative combination
  explains this"), which is the right answer for noise but makes the maximum gain uninformative
  there.
- Scope: the ground truth is synthetic and the variants were built by the same person who wrote
  the detector, so this probe fixes the detector contract and the failure mode above; it says
  nothing about detecting a genuine merge produced by a third-party script, which would also
  perturb the router and possibly rescale.

## Ops

```bash
uv run --group smokes python smokes/moe-expert-lineage/smoke.py
```

Artifacts: `.scratch/moe-expert-lineage/<model>/<variant>/` (safetensors + manifest; ~2 GB per
TinyMixtral variant and ~8 GB for the OLMoE repeat at bf16 — delete the tree to reclaim space) and
[`results.json`](results.json) next to this file. Exit 0 means the smoke ran and every expectation
is labelled held or missed — a miss is a finding, not a process failure; exit 1 only if the
primary parent is unusable. Stdout is the per-layer table plus the expectation verdicts; the JSON
holds the same records (matched count, missing/new lists, recovered π, fitted α and gain per
variant and layer).
