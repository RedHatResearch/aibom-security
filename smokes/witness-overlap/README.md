# Witness Overlap smoke (smokes/witness-overlap/)

Passive, weight-level replication of **Witness Overlap** (Stage-2 directional
provenance inside open-weight model families) from
["Witness Overlap: Directional Provenance Inside Open-Weight Model Families"](https://arxiv.org/abs/2609.31784)
(Li et al., NeurIPS 2026). Survey deep-dive for [#26](https://github.com/RedHatResearch/aibom-security/issues/26);
runnable-check context in [../README.md](../README.md).

The paper's anonymous code artifact is an empty placeholder, so everything below is
reimplemented from the paper text only.

```bash
uv sync --all-packages --group smokes
uv run --group smokes python smokes/witness-overlap/smoke.py            # all fixtures
uv run --group smokes python smokes/witness-overlap/smoke.py --only qwen3  # one fixture set
```

Stdout is a single JSON document (progress goes to stderr). `--cache-dir`, `--only
LABEL`, `--k` (SVD subspace dimension, default 16) are supported. First cold run
downloads ~46 GB of weights into the shared HF cache.

## Method (pinned from the paper)

Given same-family checkpoints — anchor `A`, target `B`, witness `W` — per tensor
`ℓ` of the dense projection block set `B` (per-layer q/k/v/o/up/gate/down_proj;
embeddings, lm_head and norms excluded):

- deltas `Δ_AB = W_B − W_A`, `Δ_AW = W_W − W_A`
- **Frobenius cosine**: `⟨Δ_AB, Δ_AW⟩_F / (‖Δ_AB‖_F ‖Δ_AW‖_F)`
- **SVD top-k overlap** (k=16): with `V_AB, V_AW` the top-k right-singular bases,
  `(1/k)·‖V_AWᵀ V_AB‖²_F`

Aggregation is the paper's block-aware mean: average per-tensor scores within each
block role across layers, then average across the 7 roles.

- **Orientation**: `s_A = overlap(A;B,W)`, `s_B = overlap(B;A,W)`; the predicted
  parent is `argmin_X s_X` (the lower-overlap anchor is the parent). With several
  witnesses, pool by mean then argmin. The signal is the parent/child asymmetry
  (paper Table 10 reference means: parent-anchor 0.0232 vs child-anchor 0.5527 cos).
- **Root identification**: `root_score(M) = mean of overlap(M; W1, W2)` over
  unordered pairs of the other family members; root = `argmin_M`.
- **ρ/η failure tags** (diagnostics, never decisions): `ρ` = cosine between the two
  descendant updates from the parent; `η = ‖witness−parent‖ / ‖child−parent‖`;
  high-ρ `ρ ≥ 0.0247`, high-η `η ≥ 45.16` (paper's top-10% corpus cutoffs, Table 11).

Scoring core (`frobenius_cosine`, `right_singular_basis`, `svd_subspace_overlap`,
`witness_scores`, `root_scores`, `rho_eta`) is pure numpy on tensor dicts — no I/O,
no fixtures, no torch — so a ProvenanceBench detector can lift it directly. The SVD
path never forms a full SVD: top-k right bases come from an eigendecomposition of
the smaller Gram side, with `V(−Δ) = V(Δ)` basis reuse across anchors.

## Fixtures

All ids verified against the Hub (base fixtures 2026-10-01, stress-axis
fixtures 2026-10-04; resolution, root-level safetensors, llama/qwen3 naming,
documented lineage via `base_model` tag or model card, layout match);
revisions pinned in [config.py](config.py) (`REVISIONS`). Everything is
≤2B parameters.

| label | pair (parent → child) | witnesses | expectation |
|---|---|---|---|
| `positive_dense` | HuggingFaceTB/SmolLM2-1.7B → SultanR/SmolTulu-1.7b-Instruct | SmolLM2-1.7B-Instruct (registry), numind/NuExtract-1.5-smol (fresh) | healthy: parent-anchor ≪ child-anchor |
| `vocab_drift` | meta-llama/Llama-3.2-1B → dphn/Dolphin3.0-Llama3.2-1B (vocab 128258 vs 128256) | BanglaLLM/BanglaLLama-3.2-1b-unolp-culturax-base, hishab/titulm-llama-3.2-1b-v1.1 (both fresh) | healthy; block set unaffected by the vocab resize |
| `qwen3` | Qwen/Qwen3-0.6B-Base → Qwen/Qwen3-0.6B | mrfakename/dreamwriter-0.6b-beta, AIPlans/Qwen3-0.6B-IPO (registry), timarni/qwen3_FineTome-100k (fresh) | healthy; paper-registry family |
| `tiny_smoke` | HuggingFaceTB/SmolLM2-135M → SmolLM2-135M-Instruct | smollm2-135M-SFT-Only, ParitKansal/SmolLM2-135M-SFT-smoltalk, puettmann/SmolLM2-135M-Instruct-Smol-Course (registry), nlpguy/smolchess (fresh) | paper's named high-ρ/high-η failure family |

Stress-axis fixtures (2026-10-04) probe the method beyond same-family dense
geometry. Each isolates one axis; expectations were fixed before the runs:

| label | pair (parent → child) | witnesses | stress axis |
|---|---|---|---|
| `quant_fp8` | Qwen/Qwen2.5-0.5B → Qwen2.5-0.5B-Instruct | RedHatAI/Qwen2.5-0.5B-Instruct-FP8-dynamic, Rumiii/Qwen2.5-0.5B-Med-Pre-Trained-92k (both fresh) | quantized storage: all 168 projections stored as F8_E4M3; same regime as bench pair `pb-smoke-qwen25-fp8` |
| `tiny_delta` | Qwen/Qwen2.5-0.5B → Rumiii/Qwen2.5-0.5B-Med-Pre-Trained-92k | Qwen2.5-0.5B-Instruct, Rumiii/Qwen2.5-0.5B-Med-Post-Trained-92k (both fresh) | sensitivity floor: child is a documented ~23.6M-token CPT of the base (arXiv:2506.09513) |
| `huge_delta` | Qwen/Qwen2.5-0.5B → DuoGuard/DuoGuard-0.5B | Qwen2.5-0.5B-Instruct, Med-Post-Trained-92k (both fresh) | specificity ceiling: child has a transplanted 12-way classifier head (lm_head → score), projections intact |
| `merge_soup` | HuggingFaceTB/SmolLM2-135M-Instruct → mnoukhov/SmolLM2-135M-Instruct_tldr-sft | ThomasTheMaker/smollm2-135m-soup1, lldois/SmolLM2-135M-Reasoning-SLERP-Champion (both fresh) | merge/soup: soup1 is a card-documented linear merge of the anchor, the child, and SmolLM2-135M; SLERP-Champion is a t=0.3 SLERP of two DPO children of the anchor |
| `same_data_diff_lineage` | Qwen/Qwen3-0.6B-Base → timarni/qwen3_FineTome-100k | arogov/Llama-3.2-1B-Instruct-FineTome-100k (fresh), mrfakename/dreamwriter-0.6b-beta (registry) | same-data adversary: first witness is a FineTome-100k SFT of Llama-3.2-1B-Instruct — same public dataset as the child, different lineage (`allow_partial`) |
| `kd_student` | HuggingFaceTB/SmolLM2-360M-Instruct → 25b3nk/smollm2-360m-boolq-calibration-grpo2-v1 | DSTI/SmolLM2-360M-AccidentReports-distilled-kd1.7B (fresh) | distilled data: witness weights descend from the 360M anchor but were logit-KD-distilled from SmolLM2-1.7B-Instruct outputs |

Root-identification family sets: `smollm2_1_7b` (base + SmolTulu + Instruct +
NuExtract + Sungmanc/finetune_smollm2_python), `qwen3_0_6b` (base + Qwen3-0.6B +
dreamwriter + AIPlans/Qwen3-0.6B-IPO + timarni/qwen3_FineTome-100k),
`smollm2_135m` (failing control: base + Instruct + SFT-Only + smoltalk + smolchess).

## Deviations from the paper

1. **Tensor matching**: the paper says "corresponding" tensors without naming
   rules — the smoke intersects the 7 projection-role tensor names *and shapes*
   across the checkpoints involved.
2. **dtype**: unspecified in the paper — safetensors load as float32
   (`smokes/_lib/hub.py`), deltas and accumulations in float64 numpy.
3. **k clamp**: k=16 clamps to `min(m, n)` per tensor (tiny projections).
4. **ρ/η scope**: computed on the flattened concatenation of the block-set
   tensors (Proposition 4.1 uses the full trainable collection; block set as a
   tractable proxy; tags only, never decisions).
5. **Thresholds**: ρ/η cutoffs applied as-is from the paper's corpus.
6. **Hardware**: numpy CPU on ≤2B models (paper: L40S GPUs, 176 checkpoints).
7. **Witness substitutions** (the paper's checkpoint registry needed these
   stand-ins and drops; every replacement re-verified against the Hub):
   - `gotoplanb/SmolLM2-FT-MyDataset` is 404 (repo gone) — covered by the other
     135M registry witnesses; no substitution needed.
   - `lhoestq/finetune_smollm2_python`: LoRA adapter at root, merged weights only
     in a subfolder (unusable by the root-level-glob shared loader) — same-recipe
     `Sungmanc/finetune_smollm2_python` used in the SmolLM2-1.7B family set.
   - `ldqvinh/qwen3-0.6b-base-grpo-v2-2048` and `LastTransformer/qlora-…-dpo`:
     LoRA-only — dropped from the qwen3 witness pool.
   - `intalio/Qwen3-NoThink-0.6b` (paper-registry) is a template-only rehost:
     its safetensors are byte-identical to Qwen/Qwen3-0.6B (max|diff| = 0 on
     sampled tensors; identical root scores over all 196 block tensors). Dropped
     from the qwen3 fixtures — a witness identical to the child gives s_A = 1.0
     and a zero-norm Δ_BW (s_B undefined). Noteworthy for the survey: the
     zero-delta guard makes witness overlap a clean weight-identity/rehost
     detector.
   - `Qwen/Qwen3-Embedding-0.6B` was first picked as a fresh qwen3 witness
     (config reads `Qwen3ForCausalLM`), but its safetensors serialize without the
     `model.` key prefix (`layers.0.mlp...`) and drop the lm_head — nonstandard
     for the shared loader, and an embedding-objective post-train rather than an
     LM fine-tune. Replaced by `timarni/qwen3_FineTome-100k` (community SFT of
     Qwen3-0.6B-Base, standard naming, verified by safetensors key inspection).
   - The paper registry has **no Llama-3.2-1B-base family** (only
     Llama-3.2-1B-Instruct), so `vocab_drift` witnesses are fresh documented
     Llama-3.2-1B-base descendants (Bangla continual pretrains).
   - Rejected: unsloth mirrors of bases (Δ≈0 degenerate witnesses), GGUF/ONNX
     rehosts, and `nyuuzyou/SmolLM2-1.7B-Eagle` (EAGLE draft-head alters tensor
     roles). For the stress axis the base+Instruct merge and quantization
     rejections were revisited: `merge_soup` now covers a documented merge
     directly, and FP8-dynamic was added as `quant_fp8` (torch-backed load
     upcasts F8_E4M3 to float32; see Results). AWQ/GPTQ int4 remains out of
     scope — `qweight`/`qzeros`/`g_idx` packing stores no `.weight` projection
     tensors, so the name intersection is empty without dequantize support.
   - `arogov/Llama-3.2-1B-Instruct-FineTome-100k` (`same_data_diff_lineage`
     witness) has the weakest lineage evidence in the fixtures: repo name plus
     config `_name_or_path`, no model card. Acceptable because the fixture's
     assertion is abstention (zero tensor intersection with the Qwen3 pair),
     not a lineage call.
8. **Revision pinning**: the fixtures pass pins every repo to the sha resolved on
   2026-10-01 (stress-axis fixtures: 2026-10-04); one sha in the pre-run
   verification notes was corrupted (`nlpguy/smolchess`) and 404'd at first
   run — corrected against the Hub API and all pins (21 base + 13 stress)
   re-verified before the recorded runs.

## Results

All runs on the pinned revisions above, numpy CPU (arm64 laptop), k=16,
2026-10-01. Orientation scores are `s_parent-anchor / s_child-anchor`
(cos; SVD in parentheses); the decision is argmin, so a healthy pair shows
parent ≪ child.

| fixture | gate | orientation (cos, svd) | verdict |
|---|---|---|---|
| `positive_dense` | ok, 168T | witnesses 0.073/0.950 and 0.001/0.999; pooled 0.037/0.974 (svd 0.073/0.924) | **correct** |
| `vocab_drift` | ok, 112T | witnesses 0.002/0.781 and 0.007/0.660; pooled 0.004/0.721 (svd 0.022/0.200) | **correct** — vocab resize does not touch the block set |
| `qwen3` | ok, 196T | witnesses 0.020/0.994, 0.002/0.997, 0.008/0.759; pooled 0.010/0.916 (svd 0.042/0.820) | **correct** |
| `unrelated_cross_family` | degraded, 80T (k/v dropped: 32 vs 8 kv heads, 24 vs 16 layers) | SmolLM2-side witness 0.001/0.998, Llama-side witness 1.000/0.000; pooled 0.501/0.499 | **no signal** — witnesses vote along family lines and cancel; no parent is fabricated |
| `tiny_smoke` | ok, 210T | 3 independent witnesses 0.000–0.036/0.996–0.997; `smollm2-135M-SFT-Only` 1.000/0.001 (ρ=0.9999, η=1.0); pooled 0.268/0.748 | **pooled correct; SFT-Only witness fails**, tagged `high-rho` |

| family set | gate | root scores (cos, ascending) | verdict |
|---|---|---|---|
| `smollm2_1_7b` | ok, 168T | base 0.013 ≪ NuExtract 0.051 < Instruct 0.280 < SmolTulu 0.622 < finetune_smollm2_python 0.976 | **root = SmolLM2-1.7B** (cos+svd) |
| `qwen3_0_6b` | ok, 196T | base 0.009 ≪ IPO 0.148 < dreamwriter 0.200 < FineTome 0.812 < Qwen3-0.6B 0.875 | **root = Qwen3-0.6B-Base** (cos+svd) |
| `smollm2_135m` | ok, 210T | base 0.179 < smoltalk 0.299 < smolchess 0.347 < SFT-Only 0.498 ≈ Instruct 0.499 | **root = SmolLM2-135M** (cos+svd) — correct even in the paper's failure family |

Reading of the runs:

- The parent/child asymmetry replicates the paper's Table 10 reference numbers
  (parent-anchor 0.000–0.073 vs child-anchor 0.660–0.999 across healthy
  witnesses here; 0.023 vs 0.553 mean there). SVD separation is consistently
  weaker than cos but preserves every decision.
- The `tiny_smoke` failure is the paper's core caveat reproduced: a witness that
  is not an independent fine-tune (`smollm2-135M-SFT-Only` is the official SFT
  stage of the very child being oriented, ρ≈1, η≈1) inverts the decision. The
  ρ/η tag fires (`high-rho`) exactly as the paper's failure taxonomy predicts.
  Pooling over independent witnesses absorbs it.
- Root identification is robust in all three families, including the 135M
  failure family, matching the paper's 16/16 root-ID vs ~95% one-witness
  orientation.
- Cross-family negative control: each witness orients toward its own family
  (cos 0.998/1.000 on the respective sides) and the pooled vote collapses to a
  coin flip — the method degrades to noise rather than fabricating a shared
  parent.
- Ops signal (numpy CPU, compute only): 135M family 45 s, 0.6B family 251 s,
  1.7B family 4966 s (~83 min; 5 members × 168 tensors, eigh on ≤2048² Grams);
  triples 73–1547 s.

### Stress-axis runs (2026-10-04)

Same scoring path, pinned revisions from the stress fixture block. Scores are
`s_parent-anchor / s_child-anchor` (cos; SVD in parentheses):

| fixture | gate | orientation (cos, svd) | verdict |
|---|---|---|---|
| `quant_fp8` | ok, 168T | FP8-dynamic witness −0.199/+0.199 (svd 0.0723/0.0723, margin 5·10⁻⁷), tagged `high-eta` (η≈7.4·10⁴); Med-Pre control 0.000/0.955; pooled −0.099/+0.577 (svd 0.048/0.410) | **correct** — cosine sign carries orientation through F8_E4M3 storage noise; the SVD variant's separation collapses to its noise floor |
| `tiny_delta` | ok, 168T | Instruct witness 0.000/0.286 (correct); Med-Post witness 0.807/0.049 — inverts toward the child, tagged `high-rho`; pooled 0.404/0.167 (svd 0.340/0.100) | **wrong (pooled flips)** — the sensitivity floor is real: a ~23.6M-token CPT child with a witness sharing that delta orients toward the child, and the tag fires |
| `huge_delta` | ok, 168T | witnesses 0.009/0.740 and 0.003/0.953; pooled 0.006/0.847 (svd 0.039/0.797) | **correct** — classifier-head transplant with intact projections orients as strongly as a routine SFT |
| `merge_soup` | ok, 210T | soup1 (merge contains the child) 0.072/0.052 — wrong on cos (margin 0.02), right on svd (margin 0.001), tagged `high-rho`; SLERP witness 0.005/0.153; pooled 0.038/0.102 (svd 0.067/0.088) | **pooled correct** — a merge that contains the child is a borderline witness; pooling with an independent witness recovers |
| `same_data_diff_lineage` | degraded, 0T (196 dropped: hidden_size 1024 vs 2048) | all scores `null`, no predicted parent | **abstain** — same-dataset/different-lineage witness cannot fabricate lineage; the triple-wide gate also silences the dreamwriter positive control (conservative by design) |
| `kd_student` | ok, 224T | KD witness −0.001/+0.315 (svd 0.018/0.577) | **correct** — weights-descent dominates; distillation from another family's teacher outputs creates no phantom orientation |

Reading of the stress runs:

- **Quantized storage (FP8-dynamic)**: orientation survives on cosine — the
  witness delta is dominated by quantization noise (η≈7.4·10⁴) yet the
  parent/child asymmetry keeps its sign. At the witness level the SVD variant
  does not separate: margin 5·10⁻⁷ at k=16 (2–9·10⁻⁷ across k ∈ {8,16,32},
  landing on the correct side each time — chance-level), while the dense
  control separates by 0.7 and carries the pooled SVD verdict (margin 0.36).
  Operative rule for the `pb-smoke-qwen25-fp8` regime: cosine orients a
  quantized witness; a quantized-only witness pool leaves SVD without usable
  signal. (AWQ/GPTQ int4 rehosts do not even reach the gate:
  `qweight`/`g_idx` packing removes all `.weight` projection tensors, so
  dequantize support would be needed first.)
- **Sensitivity floor (tiny delta)**: the first genuinely wrong pooled answer in
  the smoke. A child ~23.6M CPT tokens from its base, witnessed by a sibling
  that shares the same CPT stage (Med-Post), orients toward the child. The
  `high-rho` tag fires on exactly that witness — the paper's failure taxonomy
  catches what the decision misses, which is the intended division of labor.
- **Specificity ceiling (huge delta)**: not reached. Head-transplant surgery
  leaves orientation margins (0.006/0.847) comparable to routine SFTs; the
  method tracks weight lineage, not task.
- **Merges**: a linear merge that includes the child as an equal-weight
  component is a near-tie (cos wrong by 0.02, svd right by 0.001, `high-rho`).
  A merge of *other* descendants (SLERP) behaves like a normal witness.
  Practical rule: witnesses that may contain the oriented child need the ρ
  tag and an independent second witness.
- **Same-data adversary**: zero tensor intersection (Qwen3 1024-wide vs Llama
  2048-wide) → total abstention, no fabricated lineage. Caveat recorded: the
  gate intersects across the whole triple, so the dreamwriter positive control
  is silenced too; per-witness intersection would score it. Conservative but
  total — data similarity never masquerades as lineage.
- **Distilled student**: KD from a different family's teacher does not transfer
  orientation (−0.001/+0.315 toward the true weight parent).
- Ops: 6 stress triples, 760 s compute total, ~13 min wall-clock (0.5B triples
  99–232 s; 135M `merge_soup` 79 s; 360M `kd_student` 157 s; the Llama-1B
  download dominated its wall-clock).

## Robustness of the judgment calls

Because the paper's code artifact is empty, the reimplementation rests on a few
documented readings. Two sweeps (cosine-only where noted, cached weights,
throwaway drivers — no change to the smoke) check that no decision depends on
them:

- **Aggregation × block set** — 4 aggregations (role-then-flat mean [pinned],
  flat mean over all tensors, median, element-count-weighted mean) × 3 block
  sets (all 7 projection roles [pinned], attention-only q/k/v/o, MLP-only
  up/gate/down) × all 8 base fixture sets: **96/96 decision cells unchanged**
  (per-witness and pooled orientation, family roots). The baseline variant
  reproduced the recorded results exactly (max abs err 0.0). Re-run on the 5
  scoreable stress triples after the stress runs (2026-10-04): **60/60 cells
  unchanged**, self-check exact — the `tiny_delta` wrong answer and the
  `merge_soup` near-tie are properties of the method, not of the aggregation
  choices. `same_data_diff_lineage` is excluded: its abstention is structural
  (0-tensor intersection), upstream of any aggregation.
- **SVD subspace dimension** — k ∈ {8, 16, 32} on the four small base fixture
  sets (svd variant): **zero flips**; k=16 reproduced the recorded svd fields
  bit-exactly. The 135M-SFT-Only degenerate witness keeps its failure character
  at every k. Re-run on the two thin-svd-margin stress triples
  (`quant_fp8`, `merge_soup`): **zero flips**, k=16 bit-exact; the FP8
  witness's svd margin stays at the 10⁻⁷ scale at every k (correct side,
  chance-level) and soup1's at 10⁻³.

Scores shift (e.g. up with k) but orderings never change. Thinnest margins
observed anywhere: SmolLM2-1.7B root gap 0.035 under MLP/size-weighted (still a
4× ratio over NuExtract), Qwen3 root gap ≈0.035 on svd (constant across k), and
the cross-family pooled vote stays ≈0.50 under every variant — the no-signal
behavior is itself robust. The ρ/η failure tags were not swept: they are
advisory diagnostics, not inputs to any decision.

## ProvenanceBench readiness (analysis only)

Over the 34 pairs in [`provenancebench/pairs.yaml`](../../provenancebench/pairs.yaml),
witness overlap would need a same-family third checkpoint with a comparable dense
layout — the method's premise. Classification:

- **Witness-addressable** (dense stack, comparable layout, same-family witnesses
  exist on the Hub): `pb-smoke-qwen25-fp8` (FP8 storage demonstrated in the
  `quant_fp8` stress run), `pb-smoke-smoltulu` (demonstrated here),
  `pb-smoke-qwen3-sft` (demonstrated here), `pb-core-tulu3-sft`,
  `pb-core-llama2-aqlm-ft`, `pb-core-r1-distill-llama8`, `pb-core-r1-abliterated`,
  `pb-core-daredevil-abliterated`. The last five are compute-heavy (≥7B triples)
  but structurally the same as the demonstrated cases. Note on quant pairs: as
  bench *pairs* these are near-zero-delta rehosts of their reference, so the
  operative detector is the zero-delta identity guard, not orientation — the
  stress run exercises their storage format flowing through the full scoring
  path. `pb-smoke-qwen25-awq` is **not addressable as-is**: int4 packing
  (`qweight`/`qzeros`/`g_idx`) stores no `.weight` projection tensors, so it
  needs dequantize support before any verdict.
- **Witness-scarce but method-applicable**: `pb-core-gemma-aqlm`,
  `pb-core-r1-distill-qwen15`, `pb-core-minicpm5-1b-rl` — no obvious third
  same-family tune with comparable layout on the Hub today; would need a witness
  hunt first.
- **Abstain by construction**: MoE-stack pairs (the paper's block set is undefined
  for expert tensors: `granite-*`, `gpt-oss-*`, `qwen3-upcycle`, `du-from-dense`,
  `lfm25-*`, `tinymixtral`, `fs/btx strangers`, `granite-reap`, `qwen3-30b-ream`,
  `lfm25-abliterated`, `lfm25-opus-distil`); GGUF-only repos (`granite-gguf`,
  `lfm25-gguf`, `ternary-bonsai`, `qwen35-unsloth-gguf`, `qwen35-4b-gsq`);
  layout-changing transforms (`sheared-pythia`, `sheared-llama-pruned`,
  `nemotron-h-4b`, `smollm2-medit-upscale`); cross-family negatives
  (`llama-smol-cross`, `granite-smol-cross`, `smollm-smollm2-strangers`,
  `llama2-granite7b-strangers`, `pds-bsl`, `gpt2-fineweb-fp8`); and
  `pb-core-pythia-step65000` (same-run sequential checkpoints violate the
  independent-fine-tune premise even though the layout is comparable).

A ProvenanceBench detector would reuse the pure scoring core with a
witness-resolution step (same-family search + layout gate, as the smoke's
`_gate_reason`/`intersect_tensors` already model) and abstain otherwise.

## Output contract

One JSON object on stdout per run: per triple `{label, repos+revisions, gate
{status, reason, dropped_tensors, tensor_count}, per-witness {repo_id, kind, s_A,
s_B (cos+svd), per_role, predicted_parent, correct, rho, eta, failure_tag},
pooled {…, predicted_parent, correct}, compute_seconds}`; per family set
`{label, members+revisions, gate, root_scores (cos+svd per member),
predicted_root, ground_truth_root, correct, compute_seconds}`. Gate statuses:
`ok` / `degraded` (allow_partial, name∩shape intersection) / `abstain`
(layout mismatch without allow_partial, or <3 family members).
