# MoE real-pair consistency smoke

- Issue: [#68](https://github.com/RedHatResearch/aibom-security/issues/68) — first contact of the
  Experiment A detector with published Hub artifacts.
- Detector: [`../moe-expert-lineage/detect.py`](../moe-expert-lineage/detect.py), imported through
  `sys.path` (not copied). Thresholds and the descriptor are exactly the frozen ones from that
  smoke; nothing here redefines or retunes them.
- Pair roster: [`../../provenancebench/pairs.yaml`](../../provenancebench/pairs.yaml).

## What this checks

Experiment A synthesises derivatives with known ground truth. This smoke asks a weaker but real
question: on two **published** derivatives with pinned revisions, do the detector's per-layer
counts agree with the recipe the model card states — and where does it refuse to answer?

There is no ground truth here. "Agreement" means the survivor / missing / merge split is what the
published method predicts, not that the files were produced by a known injection.

| pair id | stack | parent (reference) | derivative (subject) | published recipe |
|---|---|---|---|---|
| `pb-core-granite-reap-025` | moe | `ibm-granite/granite-3.1-1b-a400m-instruct` @ `0da7a48` | `sasa2000/granite-3.1-1b-a400m-instruct-reap-pruned-025` @ `d8985e7` | REAP expert drop 32 → 24 (ratio 0.25); 8 pruned, survivors copied, no merge |
| `pb-core-qwen3-30b-ream` | moe | `Qwen/Qwen3-30B-A3B-Instruct-2507` @ `0d7cf23` | `SamsungSAILMontreal/Qwen3-30B-A3B-Instruct-2507-REAM` @ `91483d8` | REAM merge 128 → 96: 96 centroids by REAP saliency, 32 non-centroid experts absorbed, weighted average after permutation alignment, non-centroid gate rows removed |

Granite is 24 layers × 32/24 experts, hidden 1024, 2 projections per expert: every layer is scored.
Qwen3-30B-A3B is 48 layers × 128/96 experts, hidden 2048, 3 projections per expert: layers 0, 24
and 47 (first, middle, last) are sampled, and a layer is skipped with the failing projection if it
would not fit the 20 min / 20 GB budget.

## Pre-registration (2026-10-29, written before the run)

Thresholds are the frozen detector constants, not choices made here: exact match at cosine
≥ **0.9995**, merge at gain ≥ **0.30**, merge component floor **0.03**, descriptor = one 32×32
window per projection at a slot-independent offset, seed **20261029**.

**`pb-core-granite-reap-025`** — *subset of 24 with 8 missing and no merge above gain 0.30.*

- per layer: `matched == 24` (every survivor is a verbatim copy), `missing == 8`,
  `new == 0`, no suspect labelled `merged`, no `unexplained`, no exact duplicate;
- structural: 32 parent experts, 24 subject experts in every layer.

**`pb-core-qwen3-30b-ream`** — a range, because REAM keeps most centroids as singletons and only
32 of 128 parents are absorbed into groups:

- per sampled layer: structural `128 → 96` experts;
- `matched` ∈ **[64, 96]** — 96 − 32 singletons survive verbatim at worst, all 96 if every merge
  still clears the cosine floor;
- `missing` ∈ **[32, 64]** and `missing == 128 − matched` (an assignment identity, checked);
- `merged` fits (gain ≥ 0.30, ≥ 2 components ≥ 0.03) ∈ **[0, 32]**;
- every one of the 96 suspects carries exactly one label (`exact` / `merged` / `unexplained`);
- the fitted α and gain of each merge are **reported, not predicted**. REAM permutes group members
  against the centroid before averaging, so a convex fit in the original slot space is not
  guaranteed; a low `merged` count with the leftovers in `unexplained` is the predicted failure
  mode and is recorded as an observation, not a crash.

## Method notes

Scoring is label-blind: `score()` in [smoke.py](smoke.py) receives the subject and reference repo
ids, their pinned revisions, the stack string and the layer sample that `config.SAMPLING`
pre-registers per pair before the run. It never sees the pair id, the expected label, the evidence
block or the recipe; the harness attaches the id and judges afterwards. `pairs.yaml` is read for the
structural pins only — `expected`, `evidence` and `challenge` are never loaded.

Reads go through `detect.CheckpointReader` (one materialised tensor at a time, `MAX_TENSOR_BYTES`
cap). A layer is served in one of two ways, decided by the budget gate: whole shards are fetched
when they fit the remaining budget (the granite pair is a single 2.06 GB `model.safetensors`),
otherwise only the tensors of that layer are read by HTTP `Range` into a compact local mini-shard
(the same safetensors format, header rewritten to contain exactly the fetched tensors, so the
reader cannot silently address an un-fetched one). The shard index is used only as a download
hint; the reader re-derives names from the safetensors headers. GraniteMoe stores all experts of a
layer in one tensor (`[32, 1024, 1024]`), so `BatchedExpertView` presents dim 0 through the
reader's per-expert API; Qwen3-MoE is natively per-expert
(`model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight`) and uses `detect.TensorSpec`
unchanged.

Per sampled layer the driver records: matched count, missing list, new list, every merge fit with
its α and gain, the label counts, the maximum parent×suspect cosine, and elapsed time. A
`DetectionError` (structural surprise) becomes an explicit `ABSTAIN` line for that layer; it never
aborts the run. Budget gates are evaluated **before any payload byte is read** (only the small
shard headers are fetched first, to size the exact tensor ranges): a layer whose projected download
+ analysis time exceeds the remaining wall-clock budget, or whose run would push past 90 % of the
RSS budget, is skipped with that projection in `results.json`.

Disclosures — parts that are ours, not in the Experiment A spec or the REAM/REAP papers:

- **Batched-expert adapter** (`BatchedExpertView`, ~90 lines): GraniteMoe's dim-0 expert layout is
  not expressible in `detect.TensorSpec`. The adapter derives the expert count from the physical
  tensor shape, maps logical per-expert names onto slices, and caches the one batched tensor per
  (layer, role). It does not touch scores, thresholds or matching.
- **Layer sampling for the 30B pair** (0, 24, 47 instead of all 48) and the shard-level budget
  gate; both are recorded per pair in `results.json`.
- **HTTP ranged materialisation** (~180 lines, `plan_ranged` / `run_ranged`): the remote shard
  header (8-byte length, then JSON) is size-capped at the detector's 4 MiB metadata cap, every
  `data_offsets` pair is validated against the shard's `Content-Range` total, only `206` responses
  are accepted, and a group of tensor ranges is at most 64 MiB. Any surprise is a `DetectionError`
  (a recorded skip), never a crash. The mini-shard is written to `.scratch/moe-real-pairs/` and
  reused on a rerun, keyed by repo, revision and layer.
- **Default download-rate assumption** (2.5 MB/s) used for the first budget projection, replaced by
  the measured rate afterwards. This affects only which layers are attempted, never a score.
- **Index as a download hint only.** `pairs.yaml` pins the revision; the index decides which
  shard files to fetch, and the hardened reader re-derives names from the safetensors headers.

## Ops

```bash
uv run --group smokes python smokes/moe-real-pairs/smoke.py
# or one pair:
uv run --group smokes python smokes/moe-real-pairs/smoke.py --pair pb-core-granite-reap-025
```

Exit 0 means every pair has an outcome (`scored`, `skipped` or `unavailable`) recorded in
[`results.json`](results.json); a skipped pair is not a failure. Exit 1 is reserved for an
unexpected exception. Whole shards land in the shared Hugging Face cache; ranged reads land as
mini-shards under `.scratch/moe-real-pairs/` (6.0 GB for the three sampled 30B layers, delete the
tree to reclaim it).

## Results

Run 2026-10-29T21:10:02Z (exit 0), seed 20261029, thresholds exactly the frozen detector constants
(`exact_cos` 0.9995, `merge_gain_min` 0.30, `min_component` 0.03, 32×32 descriptor blocks), no
threshold or band touched before or after the run. Wall 45.69 s of the 1200 s budget; peak RSS
2698.9 MB of 20 GB. Every number in this section is read from the committed
[`results.json`](results.json).

This pass is **cache-only**: both pairs report `downloads.bytes = 0` with an empty `calls` list,
because the granite shards and the three ranged mini-shards were already on disk. The 30B layers
still took the ranged path (their full shards do not fit the remaining budget), but `plan_ranged`
found every mini-shard present, so not a byte was fetched. Download accounting records actual
fetches only — repo, revision, bytes and seconds per call — so a cache hit contributes nothing and
`bytes = 0` means exactly that.

A separate **cold** observation, made once with an empty scratch tree, is not committed in this
tree: it scored layers 0 and 24 and skipped layer 47 with a budget projection, and its timings and
byte counts are deliberately not quoted here because no artifact in the repository backs them.
That skip is the behaviour the shard-level gate exists for (see Method notes); the committed pass
itself abstained on nothing.

**`pb-core-granite-reap-025`** — 24 layers × (32 parent, 24 subject) experts; 576/768 expert pairs.

| layers | matched | missing | new | merged | unexplained | duplicates | max pair cos | max gain | per layer |
|---|---|---|---|---|---|---|---|---|---|
| 24/24 | **24/32 every layer** (576) | **8 every layer** (192 total) | 0 | 0 | 0 | 0 | 1.000000 | `null` (no residual) | 0.63–0.88 s |

**`pb-core-qwen3-30b-ream`** — 48 layers, layers 0 / 24 / 47 sampled; 128 parent, 96 subject experts.

| layer | matched | missing | new (suspect slots) | merged | unexplained | dups | max pair cos | max gain | elapsed |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 93/128 | 35 | `[1, 35, 78]` | **0** | 3 | 0 | 1.000000 | −1.2250 | 9.17 s |
| 24 | 93/128 | 35 | `[7, 61, 83]` | **0** | 3 | 0 | 1.000000 | −1.5312 | 8.63 s |
| 47 | 94/128 | 34 | `[7, 81]` | **0** | 2 | 0 | 1.000000 | −1.6247 | 8.36 s |

Missing lists are full in `results.json`; layer 0 begins `[1, 5, 6, 9, 10, 14, 15, 20, 31, 32, …]`.
The non-verbatim suspects (the ones in `new`) are the only experts that are not exact copies; the
other 93–94 match the exact floor with cosine 0.999994–1.000000:

| layer | suspect | dominant parent | cos | α (components ≥ 0.03) | gain | label |
|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 0.936602 | 0.9508 | −8.8408 | unexplained |
| 0 | 35 | 51 | 0.935599 | 0.9420 | −7.6124 | unexplained |
| 0 | 78 | 106 | 0.983291 | 0.9824 | −1.2250 | unexplained |
| 24 | 7 | 10 | 0.996596 | 0.9566 | −9.0997 | unexplained |
| 24 | 61 | 78 | 0.998957 | 0.9947 | −1.5312 | unexplained |
| 24 | 83 | 111 | 0.993496 | 0.9419 | −8.1495 | unexplained |
| 47 | 7 | 8 | 0.998571 | 0.9796 | −7.4535 | unexplained |
| 47 | 81 | 110 | 0.998316 | 0.9827 | −8.2237 | unexplained |

Every one of those dominant parents is itself in that layer's `missing` list, i.e. the original
centroid weights are gone as exact copies. `near_exact_merges` and `shadowed_merges` are empty on
all three layers.

### Agreement with the published recipes

**Granite REAP: agrees, exactly.** The card claims 32 → 24 with 8 pruned and survivors copied.
Observed: 24 verbatim survivors per layer at cosine 1.000000, exactly 8 missing parents per layer
(192 drops over 24 layers), no new expert, no candidate pair below the exact floor, no mixture fit
above gain 0.30, and no duplicate. The 8 dropped parents differ per layer (layer 0
`[6, 9, 13, 17, 19, 25, 26, 31]`, layer 1 `[1, 2, 20, 21, 22, 23, 27, 28]`, …), and over the 24
layers every parent slot 0–31 is dropped somewhere. The detector's `recovered_perm` is a strictly
increasing injection (subject slots 0–23 mapped onto an increasing 24-subset of parent slots 0–31),
which is what a drop that keeps the surviving order leaves.

**REAM: structural counts agree; the merge mechanism is present but not convex in slot space.**
The card claims 128 → 96 by merging, k=96 centroids, 32 non-centroid experts absorbed. Observed:
96 subject slots and 128 parent slots on every sampled layer; 93–94 of the 96 suspects are copies
at the exact floor (cos 0.999994–1.000000); 2–3 sit at cosine 0.936–0.999 from their closest parent,
which is where REAM's group averages sit. The missing count is not published, so the check is the
arithmetic: `missing = 128 − matched = 34–35`, which decomposes as **32 slots removed by the
128 → 96 reduction + 2–3 original centroids whose own weights were replaced by an average**. That
is consistent with the recipe and with the method's "few big clusters, many singletons": the 32
absorbed experts appear to be spread over 3 groups on layers 0/24 and 2 on layer 47 (one
non-verbatim centroid per group that received members — an inference from the counts, not
something the detector names).

**Where the detector does not fire: the merge test.** None of the eight non-exact suspects is
labelled `merged`: the best convex mixture in the original slot space is *worse* than the best
free-scaled single parent (gain −1.2 to −9.1), and only one component clears the 0.03 floor
(α = 0.94–0.99 on the dominant parent, the rest below the floor). This is the failure mode the pre-registration
named: REAM permutes each group member against the centroid (Hungarian on hidden activations and
weights) before averaging, and it weights the average by REAP saliency, so the result is a centroid
plus a small, neuron-aligned perturbation — not `Σ αᵢxᵢ` over the parents in their original
orientation. Cosine still lands at 0.936–0.999, so the merged experts are visible as `new`
(candidate pairs), but the detector correctly declines to call them convex merges. A detector that
wanted to name the group would need permutation-aware fitting; that is out of scope here and is
recorded as an extension, not a threshold change.

**Abstentions.** The committed pass raised no `DetectionError`: `skipped_layers` is empty for both
pairs, and all 24 granite layers plus all three sampled 30B layers are scored in `results.json`.
Every structural bound in the reader held on every sampled layer of both real checkpoints (32/24
and 128/96 experts, a single-file and a 16/10-shard layout, BF16 throughout). The only non-scored
layer seen at all is the uncommitted cold pass's layer-47 budget skip described above; a skip is
not a structural abstention.

### Reading

- The cheap pair is a clean control for the descriptor and the matching stage: bit-identical
  copies at cosine 1.000000 (max deviation 4e−15), the exact drop count, and zero false merge
  labels.
- The 30B pair shows what "consistency, not truth" buys: the detector recovers the *shape* of the
  REAM edit (96 survivors, 32 slots gone, 2–3 centroids whose weights were replaced by a group
  average) without recovering the merge weights, because the published recipe applies a permutation
  before averaging. The counts agree with the recipe; the mechanism is flagged as unexplained by
  design of the estimator.
- Layer sampling cannot see the whole model: layers 0/24/47 all show the same pattern (93–94
  verbatim, 2–3 replacements), but nothing here rules out a different pattern in the 45 unsampled
  layers.
- The granite pair is a single-file checkpoint; the Qwen pair is 16/10 shards. Both were read
  through the same reader, with the batched adapter only for GraniteMoe's dim-0 expert tensor.

**Reporting fix (disclosed, no measurement changed).** The first version of the judge counted a
budget-skipped layer as a failure of its `no structural abstention` check, which made an earlier,
uncommitted pass report `MISSED` for the 30B pair even though every scored layer held. The check is
now "no `DetectionError` on an attempted layer", and coverage (`scored` / `sampled` / `skipped` with
reasons) is reported separately. No score, count, threshold or fitted value changed; the raw layer
records are identical before and after.

Artifacts: `.scratch/moe-real-pairs/` holds the ranged mini-shards for the three sampled 30B layers
(measured 6.0 GB for the recorded tree: 3.4 GB parent, 2.6 GB subject; delete it to reclaim the
space). Weight shards live in the shared Hugging Face cache; the granite subject
`model.safetensors` (2,064,910,056 bytes) was fetched during setup and the parent snapshot was
already cached.

