# Routing overlap as a lineage signal — calibration

**Question (issue [#69](https://github.com/RedHatResearch/aibom-security/issues/69)).**
Is routing overlap — the Jaccard overlap of per-layer top-k expert sets on a
fixed probe — a usable lineage signal? Specifically: does the same checkpoint
after 4-bit weight requantization stay much closer to itself than a *different
checkpoint of the same architecture*, and how does a different family compare?

This README and every band/threshold in `config.py` were written **before**
the first run. Results are appended below afterwards; nothing is re-tuned on
the negative population.

Not product code; not CI.

## Method

- Probe: the 24-prompt set (6 each code/math/factual/narrative), 96 ids each,
  from `smokes/routing-agreement/config.py`, imported by file path — not
  copied. Prompts are tokenized without special tokens, truncated/padded to
  96, and fed as **one prefill batch**; no generation, no chunking.
- Capture: `output_router_logits=True`, `use_cache=False`,
  `logits_to_keep=1`, one forward, then `topk` per layer. The scored rows are
  the real (non-pad) positions: 2304 for TinyMixtral/TinyLLama, 2284 for
  OLMoE (four prompts tokenize below 96 ids under the GPTNeoX tokenizer and
  are right-padded; pad id 1).
- Metric: per-layer mean Jaccard of the top-k **sets** over scored rows, plus
  exact top-k set agreement as a diagnostic. Overall = unweighted mean across
  compared layers. `excl0` drops layer 0 (whose routing is structurally
  independent of expert weights under expert-only QDQ).
- Quantization contract: frozen in `smokes/routing-agreement/quant.py`, which
  is imported (never copied): symmetric round-to-nearest, 4-bit, block 32
  along the last dimension, quantize-dequantize back to the model dtype. The
  SHA256 of both the probe config and `quant.py` is recorded in
  `results.json`.
- Determinism: seed recorded (`20261029`), `model.eval()`, no dropout, CPU
  only, `torch.set_num_threads(8)` (override with `--threads`). All loads use
  `local_files_only=True`, `trust_remote_code=False`, and the worker
  environment sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`; a missing
  snapshot is a recorded skip, never a download.
- Isolation and budget: each checkpoint runs in its own subprocess, so peak
  RSS is per-checkpoint and a malicious/corrupt checkpoint can only fail its
  own worker (recorded skip). The **enforced** gate is the payload budget
  (16 GB per model); 20 GB is the planning figure for a worker's process RSS
  (measured and reported, not enforced — see *Results*); the total budget is
  1200 s. A model that does not fit the payload budget is skipped with the
  reason in `results.json`. The only checkpoint parser is transformers'
  local safetensors reader.

### Checkpoints and populations

| population | reference | candidate | E / k | role |
|---|---|---|---|---|
| same model, requantized | TinyMixtral `base` | TinyMixtral `experts_q4` | 4 / 2 | expert-only Q4 |
| same model, requantized | TinyMixtral `base` | TinyMixtral `all_q4` | 4 / 2 | experts + router Q4 |
| same model, requantized | OLMoE-0924 `base` | OLMoE-0924 `experts_q4` | 64 / 8 | expert-only Q4 |
| same model, requantized | OLMoE-0924 `base` | OLMoE-0924 `all_q4` | 64 / 8 | experts + router Q4 |
| same architecture, different checkpoint | OLMoE-0924 `base` | OLMoE-0924-SFT `base` | 64 / 8 | **hard negative** |
| different family | TinyMixtral `base` | TinyLLama-4x1.1B-MoE `base` | 4 / 2 | true negative (index-aligned for 12 layers) |

Pins:

| label | repo | revision | dtype |
|---|---|---|---|
| `tinymixtral` | `Isotonic/TinyMixtral-4x248M-MoE` | `1e3516176a6279ce93923060fcad848dc043af79` | fp32 |
| `olmoe_0924` | `allenai/OLMoE-1B-7B-0924` | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` | bf16 |
| `olmoe_0924_sft` | `allenai/OLMoE-1B-7B-0924-SFT` | `215cc4f73147dd68bd11e7a7dcc56bac397f4221` | bf16 |
| `tinyllama_moe` | `s3nh/TinyLLama-4x1.1B-MoE` | `e82b7308fe3a401995169eb99a85492a36df5b83` | bf16 |

The 64-expert different-family cell is an explicit skip: no cached,
weight-complete, non-OLMoE 64-expert MoE exists here. `deepseek-moe-16b-base`
has only its index file cached and would need ~32 GB bf16; `Qwen1.5-MoE-A2.7B`
has 60 experts (E mismatch) and no cached weights. Comparing different E values
would silently map unrelated expert ids, so it is not done.

Tokenizer notes (both are snapshots that ship no tokenizer files):

- `OLMoE-0924-SFT` uses the `OLMoE-0924` tokenizer; the weights/config are
  pinned separately, the token ids are identical.
- `TinyLLama-4x1.1B-MoE` uses the Llama-2 SentencePiece tokenizer. The only
  cached SentencePiece file is TinyMixtral's `tokenizer.model`, whose SHA256
  (`dadfd56d…34055`) is the published Mistral/Llama-2 32k tokenizer blob; its
  special ids (unk 0, bos 1, eos 2) and vocab (32000) match the TinyLLama
  config. The **same** token ids are fed to both models, so the comparison is
  at least token-identical by construction of the probe. A tokenizer mismatch
  would *also* produce low overlap, so the observed chance-level result cannot
  tell the two apart; this is an inference from the blob hash, disclosed rather
  than assumed silently (see *Limitations vs #69*).

## Threshold rule (frozen before the run)

- **Unit**: per-layer mean Jaccard over all scored rows.
- **Calibration population**: OLMoE-0924 `experts_q4` (the same model as the
  negative, realistic GGUF-style requantization with the router kept at model
  precision).
- **Statistic**: `T = mean − 3 × sample_sd (ddof=1)` across the 16 per-layer
  means.
- **Negative population**: OLMoE-0924 vs OLMoE-0924-SFT, both `base`.
- **FPR**: fraction of the 16 negative layers at or above `T` (primary), and
  separately the fraction of scored negative rows at or above `T`
  (diagnostic). Higher Jaccard means "same lineage", so a negative above `T`
  is a false positive.
- **TPR**: fraction of the 16 calibration layers at or above their own `T`.
- A secondary threshold uses the same rule on OLMoE `all_q4` (router also
  Q4). A layer-0-excluded variant of the same rule is reported as sensitivity
  only, **not** used for the verdict.
- The negative population is never consulted to choose `T`; it only measures
  the resulting FPR.

**Decision rule (pre-registered, `config.VERDICT_RULES`):**

| verdict | band |
|---|---|
| `usable_lineage_signal` | `fpr_layer <= 0.0` and `tpr_layer >= 0.90` |
| `inference_tier_only` | `fpr_layer <= 0.25` and `tpr_layer >= 0.80` |
| `too_brittle` | otherwise, **including no calibration** |

`smoke.py` walks that config object itself — no threshold is restated in code —
and `results.json` records the same object, so the archived rule and the
executed rule cannot drift apart. The bands are stored as numbers rather than
as expression strings so they can be executed; the values are unchanged from
the pre-registration, and the earlier string form is in the git history.
There is no fourth verdict: a run whose primary calibration is unavailable
publishes the registered fallback (`too_brittle`) and exits non-zero.

The verdict concerns the **Jaccard** statistic. Exact top-k set agreement is
reported alongside because at high k it collapses much faster; it is not the
calibrated metric.

### Nuance recorded before the run

OLMoE-0924-SFT is the supervised fine-tune released with the 0924 base: the
two checkpoints share ancestry. Treating it as the ticket's
"different-checkpoint" negative means a high overlap is a false positive for
*"same exact checkpoint"* but a true positive for *"same lineage / descendant
of"*. Both readings are reported; the threshold is calibrated only against
the requantized self.

## Pre-registered expectations

Based on the prior routing-agreement run (TinyMixtral expert-Q4 J = 0.9694,
router-Q4 J = 0.8786; OLMoE expert-Q4 J = 0.9443 / set 0.7621, router-Q4
J = 0.9092; chance J = 0.3889 at E=4,k=2 and 0.0705 at E=64,k=8):

| id | expectation | band |
|---|---|---|
| E1 | TinyMixtral expert-Q4 keeps overall mean J | ≥ 0.95 |
| E2 | TinyMixtral expert-Q4 layer 0 is exactly 1.0 | \|x−1\| ≤ 1e-9 |
| E3 | TinyMixtral router-Q4 J < expert-Q4 J | difference > 0 |
| E4 | OLMoE expert-Q4 overall mean J | in [0.90, 0.98] |
| E5 | OLMoE expert-Q4 layer 0 is exactly 1.0 | \|x−1\| ≤ 1e-9 |
| E6 | OLMoE router-Q4 J < expert-Q4 J | difference > 0 |
| E7 | Different checkpoint less similar than requantization | difference < 0 |
| E8 | Different checkpoint stays above chance | difference > 0 |
| E9 | No negative layer reaches the primary threshold | layer FPR = 0 |
| E10 | Different family within 0.10 of its analytic chance J | \|x−chance\| ≤ 0.10 |
| E11 | Primary threshold stays above 0.60 | ≥ 0.60 |
| E12 | At least 12/16 requant layers clear their own threshold | TPR ≥ 0.75 |
| E13 | OLMoE exact set agreement at k=8 (the brittle metric) | in [0.70, 0.85] |

Rationale: expert Q4 with 32-wide blocks is a small relative perturbation, so
the requantized self should stay near J ≈ 0.95; the router's 64×2048 matrix
sets every expert's score directly and should cost more; a different
checkpoint (even a fine-tune) should lose more routing than requantization;
different families should land at chance because expert indices are arbitrary
labels.

## Fingerprint schema note

Every fingerprint (base, `experts_q4`, `all_q4`) records:

```
label, repo_id, revision, variant, model_dtype,
router_precision, quant_type, quant_bits, quant_block,
n_layers, rows_scored, seq_len, probe_input_sha256,
top_k, n_experts, sha256
```

`sha256` hashes the int16 top-k ids **on the scored rows only**. Probe identity
travels with the fingerprint: `seq_len` and `probe_input_sha256` repeat the
hash that the enclosing `probe` block records, so a future run can prove it
used the same probe without trusting that block. Compare fingerprints only when
`n_experts`, `top_k`, `seq_len`, `probe_input_sha256`, `model_dtype` and
`router_precision` match — all six are now in the fingerprint itself. `base`
has `quant_type="none"`; `experts_q4` keeps `router_precision` at the model
dtype; `all_q4` records `router_precision="q4_block32"`. A base-vs-Q4
comparison is a requantization experiment, not a lineage comparison.

## Run

```bash
uv run --group smokes python smokes/routing-calibration/smoke.py
```

Outputs `results.json` next to this README and prints per-layer tables,
population summaries, the calibration, and the pre-registered expectations.

Exit status: `0` when the primary calibration ran and produced the published
verdict; `1` when no population produced routing measurements, or when the
primary calibration is unavailable (the verdict is then the registered
`too_brittle` fallback). `results.json` is written in every case that reaches
the end of the parent process, so a degraded run is still inspectable.

## Results

Recorded run: `2026-10-29T23:06:51Z`, exit code 0, total **405.5 s**, macOS
arm64 (Python 3.12.13, torch 2.13.0, transformers 5.15.0, numpy 2.5.2),
8 threads, seed 20261029. All four checkpoints loaded offline, and this run
reproduces the earlier recorded numbers bit for bit. Peak per-checkpoint RSS
**22.80 GB** (`olmoe_0924_sft`) was *above* the 20 GB process planning budget
on this run: the gate actually enforced is the payload check (13.84 GB ≤ 16 GB
per model), and the peak itself is not reproducible — the same worker measured
10.0–16.0 GB on the earlier recorded runs and 22.8 GB here while other smoke
jobs were running on the machine, so treat peak RSS as a diagnostic of page
residency, not as a contract.

Budget outcomes:

| worker | payload GB | peak RSS GB | worker s |
|---|---:|---:|---:|
| `olmoe_0924` | 13.84 | 8.45 | 272.2 |
| `olmoe_0924_sft` | 13.84 | 22.80 | 48.0 |
| `tinymixtral` | 2.80 | 6.11 | 12.1 |
| `tinyllama_moe` | 6.77 | 7.46 | 72.9 |

Reproducibility check: the four requantization cells are **bit-identical** to
the independent `smokes/routing-agreement` run (`ΔJ = Δset = 0` for
TinyMixtral and OLMoE, expert-Q4 and router-Q4), even though the captures now
run in separate subprocesses. That validates the shared QDQ contract and the
determinism claim.

### Population summary

Chance: TinyMixtral (E=4, k=2) mean J = 0.388889, exact set = 1/6 = 0.166667;
OLMoE (E=64, k=8) mean J = 0.070542, exact set = 2.259e-10.

| pair | L | mean J | min layer | max layer | excl0 J | set agr | chance J |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tinymixtral_experts_q4` | 12 | 0.9694 | 0.9552 | 1.0000 | 0.9666 | 0.9541 | 0.3889 |
| `tinymixtral_all_q4` | 12 | 0.8786 | 0.8419 | 0.9372 | 0.8732 | 0.8184 | 0.3889 |
| `olmoe_0924_experts_q4` | 16 | 0.9443 | 0.9281 | 1.0000 | 0.9406 | 0.7621 | 0.0705 |
| `olmoe_0924_all_q4` | 16 | 0.9092 | 0.8879 | 0.9367 | 0.9074 | 0.6194 | 0.0705 |
| `olmoe_0924_vs_0924_sft` | 16 | 0.8414 | 0.8201 | 0.8702 | 0.8410 | 0.3985 | 0.0705 |
| `tinymixtral_vs_tinyllama_moe` | 12 | 0.3907 | 0.3513 | 0.4170 | 0.3904 | 0.1616 | 0.3889 |

Readings:

- **Same model, requantized.** Expert-only Q4 preserves nearly all routing
  overlap: J = 0.9694 (TinyMixtral) and 0.9443 (OLMoE), with exact-set
  agreement 0.9541 / 0.7621. Quantizing the router too costs another 9.1 / 3.5
  Jaccard points. At layer 0, where expert QDQ is structurally the identity
  (`J = 1.0000` exactly on both models), router Q4 alone gives 0.9372 / 0.9367:
  the router is the dominant perturbation, as in the prior smoke.
- **Same architecture, different checkpoint (hard negative).** J = 0.8414,
  exact set = 0.3985. This is 11.9× chance in Jaccard and 5.6× chance in set
  agreement: a fine-tune retains a great deal of routing, so it is a genuinely
  hard negative, not a near-chance one. It is nonetheless below **every** one
  of the 16 requantized layers (worst requant layer 0.9281; best negative layer
  0.8702). The SFT card's `base_model` tag confirms it is a fine-tune of
  0924, so this is the closest same-architecture negative available here; an
  unrelated checkpoint should land lower, making the measured FPR
  conservative.
- **Different family.** J = 0.3907 against 0.3889 chance (+0.0018), exact set
  0.1616 against 0.1667 chance. Indistinguishable from independent routing, as
  a lineage signal requires. Depth is index-aligned for the first 12 layers
  only (12 vs 22); TPs are not claimed beyond that. Chance-level overlap does
  not by itself identify the cause: a tokenizer mismatch would produce the same
  value, so this cell supports "no lineage signal detected", not "the models
  are independent and the probe was right" — see *Limitations vs #69*.

### Calibration

Primary rule: `T = mean − 3 × sample_sd (ddof=1)` over OLMoE-0924
`experts_q4`'s 16 per-layer mean Jaccards. Negative: 0924 vs 0924-SFT.
FPR is the fraction of negative layers/rows at or above `T`.

| calibration | mean | sd | T | TPR layer | TPR row | negative mean [min, max] | FPR layer | FPR row | margin mean | margin worst layer |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| primary (`experts_q4`) | 0.9443 | 0.0169 | **0.8936** | 1.0000 | 0.7621 | 0.8414 [0.8201, 0.8702] | **0.0000 (0/16)** | 0.3985 (14562/36544) | +0.1029 | +0.0579 |
| secondary (`all_q4`, router Q4) | 0.9092 | 0.0121 | **0.8730** | 1.0000 | 0.6194 | 0.8414 [0.8201, 0.8702] | **0.0000 (0/16)** | 0.3985 (14562/36544) | +0.0678 | +0.0177 |
| primary, excl layer 0 (sensitivity) | 0.9406 | 0.0084 | 0.9153 | 1.0000 | — | 0.8410 [0.8201, 0.8702] | 0.0000 (0/16) | — | — | — |

The layer-level separation is real but not wide: the most similar negative
layer (0.8702) sits 0.0234 below the primary threshold, and the noisiest
requantized layer (0.9281) sits 0.0345 above it. With a Q4 router the margin
nearly disappears: the secondary threshold is 0.8730 and the same negative
layer is 0.0028 below it. The layer-0-excluded sensitivity raises the
threshold to 0.9153 and still gives 16/16 TPR and 0/16 FPR, so the verdict
does not depend on the structurally perfect layer.

**TPR 16/16 is arithmetic, not evidence.** `T` is placed three sample sds
below the calibration mean, so a one-sided 3-sd tail leaves nearly all 16
layers above it whatever the routing is doing; 16/16 reports the construction
of the rule. The evidential claim in this row is the negative side: 0/16
negative layers reach `T`, the most similar negative layer is 0.0234 below it,
and the worst-layer positive-to-negative gap is +0.0579 (mean gap +0.1029).

**What 0/16 is worth at DB scale.** Zero false positives in 16 layers of one
negative pair is not a catalogue-scale FPR. The 95% Wilson upper bound for
0/16 is **0.1936** (one-sided 95%: 0.1446). Treated as independent Bernoulli
layer decisions, the point estimate of false accepts is 0 per 1,000 and per
10,000 models, while that bound allows up to **≈194 per 1,000** and
**≈1,936 per 10,000**. Those are limits on what this sample *fails to rule
out*, not a measured catalogue FPR: the 16 layer values come from a single
pair, are strongly correlated, and the pair is a descendant rather than an
unrelated model. A whole-model gate (all 16 layers above `T`) would shrink the
independence-based false-accept probability to 0.1936¹⁶ ≈ 4e-12 — precisely
why independence is not used to make a DB-scale claim. See *Limitations vs
#69*.

**Why row-level FPR is high (0.3985).** At k=8 a row's Jaccard is 1.0 only
when all eight ids match; the next possible value is 7/9 = 0.7778, below
0.8936. Thus the row-level "FPR at T" is exactly the negative population's
exact-set agreement (0.3985), and the row-level "TPR at T" is exactly the
requantized self's exact-set agreement (0.7621). The layer mean aggregates
partial overlap (7/9, 6/10, …) and is the statistic that separates the
populations; a per-row gate at this threshold does not.

### Pre-registered expectations

13 held, 0 missed, 0 skipped.

| id | expectation | observed | verdict |
|---|---|---|---|
| E1 | TinyMixtral expert-Q4 J ≥ 0.95 | 0.9694 | held |
| E2 | TinyMixtral expert-Q4 layer 0 = 1.0 | 1.0000 | held |
| E3 | TinyMixtral router-Q4 J < expert-Q4 J | +0.0908 | held |
| E4 | OLMoE expert-Q4 J in [0.90, 0.98] | 0.9443 | held |
| E5 | OLMoE expert-Q4 layer 0 = 1.0 | 1.0000 | held |
| E6 | OLMoE router-Q4 J < expert-Q4 J | +0.0351 | held |
| E7 | Different checkpoint less similar than requantization | −0.1029 | held |
| E8 | Different checkpoint above chance | +0.7709 | held |
| E9 | No negative layer reaches the primary threshold | 0/16 | held |
| E10 | Different family within 0.10 of chance | +0.0018 | held |
| E11 | Primary threshold > 0.60 | 0.8936 | held |
| E12 | TPR ≥ 0.75 | 1.0000 | held |
| E13 | OLMoE exact set agreement in [0.70, 0.85] | 0.7621 | held |

### Per-layer Jaccard distributions (mean / min / max of row J, plus set agreement)

#### `tinymixtral_experts_q4`

| layer | mean J | min J | max J | sd J | set agr |
|---|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 1 | 0.9873 | 0.3333 | 1.0000 | 0.0913 | 0.9809 |
| 2 | 0.9821 | 0.3333 | 1.0000 | 0.1079 | 0.9731 |
| 3 | 0.9676 | 0.3333 | 1.0000 | 0.1434 | 0.9514 |
| 4 | 0.9656 | 0.3333 | 1.0000 | 0.1476 | 0.9484 |
| 5 | 0.9593 | 0.0000 | 1.0000 | 0.1600 | 0.9392 |
| 6 | 0.9566 | 0.3333 | 1.0000 | 0.1645 | 0.9349 |
| 7 | 0.9583 | 0.3333 | 1.0000 | 0.1614 | 0.9375 |
| 8 | 0.9604 | 0.3333 | 1.0000 | 0.1577 | 0.9405 |
| 9 | 0.9552 | 0.3333 | 1.0000 | 0.1670 | 0.9327 |
| 10 | 0.9638 | 0.3333 | 1.0000 | 0.1510 | 0.9457 |
| 11 | 0.9766 | 0.3333 | 1.0000 | 0.1228 | 0.9648 |

#### `tinymixtral_all_q4`

| layer | mean J | min J | max J | sd J | set agr |
|---|---:|---:|---:|---:|---:|
| 0 | 0.9372 | 0.3333 | 1.0000 | 0.1948 | 0.9058 |
| 1 | 0.8426 | 0.0000 | 1.0000 | 0.2852 | 0.7656 |
| 2 | 0.8615 | 0.0000 | 1.0000 | 0.2713 | 0.7930 |
| 3 | 0.8547 | 0.0000 | 1.0000 | 0.2768 | 0.7834 |
| 4 | 0.8419 | 0.0000 | 1.0000 | 0.2844 | 0.7635 |
| 5 | 0.8614 | 0.0000 | 1.0000 | 0.2717 | 0.7930 |
| 6 | 0.8640 | 0.0000 | 1.0000 | 0.2698 | 0.7969 |
| 7 | 0.8754 | 0.0000 | 1.0000 | 0.2602 | 0.8134 |
| 8 | 0.8944 | 0.0000 | 1.0000 | 0.2441 | 0.8420 |
| 9 | 0.8900 | 0.3333 | 1.0000 | 0.2475 | 0.8351 |
| 10 | 0.9009 | 0.0000 | 1.0000 | 0.2375 | 0.8516 |
| 11 | 0.9185 | 0.0000 | 1.0000 | 0.2187 | 0.8780 |

#### `olmoe_0924_experts_q4`

| layer | mean J | min J | max J | sd J | set agr |
|---|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 1 | 0.9574 | 0.6000 | 1.0000 | 0.0884 | 0.8100 |
| 2 | 0.9362 | 0.6000 | 1.0000 | 0.1053 | 0.7237 |
| 3 | 0.9326 | 0.4545 | 1.0000 | 0.1090 | 0.7128 |
| 4 | 0.9409 | 0.6000 | 1.0000 | 0.1018 | 0.7421 |
| 5 | 0.9281 | 0.2308 | 1.0000 | 0.1150 | 0.6983 |
| 6 | 0.9326 | 0.1429 | 1.0000 | 0.1144 | 0.7198 |
| 7 | 0.9348 | 0.3333 | 1.0000 | 0.1082 | 0.7202 |
| 8 | 0.9330 | 0.3333 | 1.0000 | 0.1126 | 0.7207 |
| 9 | 0.9485 | 0.2308 | 1.0000 | 0.1008 | 0.7815 |
| 10 | 0.9480 | 0.4545 | 1.0000 | 0.0991 | 0.7758 |
| 11 | 0.9478 | 0.3333 | 1.0000 | 0.0983 | 0.7728 |
| 12 | 0.9517 | 0.4545 | 1.0000 | 0.0948 | 0.7890 |
| 13 | 0.9383 | 0.4545 | 1.0000 | 0.1029 | 0.7294 |
| 14 | 0.9430 | 0.4545 | 1.0000 | 0.1000 | 0.7496 |
| 15 | 0.9367 | 0.0667 | 1.0000 | 0.1200 | 0.7487 |

#### `olmoe_0924_all_q4`

| layer | mean J | min J | max J | sd J | set agr |
|---|---:|---:|---:|---:|---:|
| 0 | 0.9367 | 0.6000 | 1.0000 | 0.1031 | 0.7215 |
| 1 | 0.9076 | 0.4545 | 1.0000 | 0.1180 | 0.6055 |
| 2 | 0.8879 | 0.4545 | 1.0000 | 0.1269 | 0.5368 |
| 3 | 0.8969 | 0.4545 | 1.0000 | 0.1196 | 0.5582 |
| 4 | 0.9040 | 0.4545 | 1.0000 | 0.1204 | 0.5941 |
| 5 | 0.8972 | 0.3333 | 1.0000 | 0.1308 | 0.5867 |
| 6 | 0.9063 | 0.1429 | 1.0000 | 0.1250 | 0.6116 |
| 7 | 0.9018 | 0.3333 | 1.0000 | 0.1232 | 0.5884 |
| 8 | 0.9045 | 0.3333 | 1.0000 | 0.1252 | 0.6073 |
| 9 | 0.9246 | 0.2308 | 1.0000 | 0.1150 | 0.6826 |
| 10 | 0.9206 | 0.4545 | 1.0000 | 0.1158 | 0.6646 |
| 11 | 0.9196 | 0.3333 | 1.0000 | 0.1147 | 0.6567 |
| 12 | 0.9171 | 0.4545 | 1.0000 | 0.1162 | 0.6484 |
| 13 | 0.9109 | 0.4545 | 1.0000 | 0.1188 | 0.6243 |
| 14 | 0.9068 | 0.4545 | 1.0000 | 0.1198 | 0.6055 |
| 15 | 0.9046 | 0.0667 | 1.0000 | 0.1338 | 0.6187 |

#### `olmoe_0924_vs_0924_sft`

| layer | mean J | min J | max J | sd J | set agr |
|---|---:|---:|---:|---:|---:|
| 0 | 0.8468 | 0.4545 | 1.0000 | 0.1356 | 0.3954 |
| 1 | 0.8311 | 0.3333 | 1.0000 | 0.1423 | 0.3590 |
| 2 | 0.8290 | 0.0667 | 1.0000 | 0.1504 | 0.3687 |
| 3 | 0.8201 | 0.2308 | 1.0000 | 0.1469 | 0.3310 |
| 4 | 0.8239 | 0.0000 | 1.0000 | 0.1497 | 0.3454 |
| 5 | 0.8221 | 0.1429 | 1.0000 | 0.1486 | 0.3415 |
| 6 | 0.8378 | 0.1429 | 1.0000 | 0.1426 | 0.3818 |
| 7 | 0.8324 | 0.2308 | 1.0000 | 0.1529 | 0.3884 |
| 8 | 0.8344 | 0.3333 | 1.0000 | 0.1436 | 0.3722 |
| 9 | 0.8692 | 0.3333 | 1.0000 | 0.1347 | 0.4759 |
| 10 | 0.8702 | 0.3333 | 1.0000 | 0.1352 | 0.4829 |
| 11 | 0.8650 | 0.3333 | 1.0000 | 0.1383 | 0.4689 |
| 12 | 0.8616 | 0.3333 | 1.0000 | 0.1387 | 0.4575 |
| 13 | 0.8491 | 0.3333 | 1.0000 | 0.1440 | 0.4243 |
| 14 | 0.8414 | 0.3333 | 1.0000 | 0.1459 | 0.3997 |
| 15 | 0.8280 | 0.1429 | 1.0000 | 0.1587 | 0.3831 |

#### `tinymixtral_vs_tinyllama_moe`

| layer | mean J | min J | max J | sd J | set agr |
|---|---:|---:|---:|---:|---:|
| 0 | 0.3941 | 0.0000 | 1.0000 | 0.3049 | 0.1753 |
| 1 | 0.3513 | 0.0000 | 1.0000 | 0.2843 | 0.1306 |
| 2 | 0.4002 | 0.0000 | 1.0000 | 0.2989 | 0.1740 |
| 3 | 0.3957 | 0.0000 | 1.0000 | 0.2975 | 0.1697 |
| 4 | 0.4025 | 0.0000 | 1.0000 | 0.2990 | 0.1758 |
| 5 | 0.3997 | 0.0000 | 1.0000 | 0.2928 | 0.1684 |
| 6 | 0.3947 | 0.0000 | 1.0000 | 0.2973 | 0.1688 |
| 7 | 0.4170 | 0.0000 | 1.0000 | 0.3036 | 0.1905 |
| 8 | 0.3932 | 0.0000 | 1.0000 | 0.2905 | 0.1619 |
| 9 | 0.3932 | 0.0000 | 1.0000 | 0.2762 | 0.1497 |
| 10 | 0.3744 | 0.0000 | 1.0000 | 0.2684 | 0.1311 |
| 11 | 0.3725 | 0.0000 | 1.0000 | 0.2845 | 0.1432 |

### Verdict and recommendation

Pre-registered verdict: **`usable_lineage_signal`** — layer FPR = 0/16,
layer TPR = 16/16, threshold 0.8936 (E11 floor 0.60). The qualifiers below are
part of the reading, not a re-tuned rule, and *Limitations vs #69* states what
the cell coverage does not support.

- **Layer-mean Jaccard is a usable lineage fingerprint.** It separates the
  requantized self from the closest same-architecture negative (a fine-tune
  of the same base) with zero layer false positives at a threshold fixed
  before the run, and the different family sits at chance-level overlap (with
  the caveat in *Limitations vs #69*: the cause of that overlap is not
  identified by the number alone).
- **Exact top-k set agreement is too brittle at high k.** The requantized
  self loses 23.8 points of exact-set agreement at k=8 (1.0 → 0.762) while
  losing only 5.6 points of Jaccard (1.0 → 0.944); the negative is 0.399 in
  set agreement.
- **Row-level gating is inference-tier at best.** At k=8 a row at this
  threshold must match all eight ids, so 39.9% of negative rows still pass;
  the layer mean is the discriminating unit.
- **Router precision is load-bearing.** Router Q4 lowers OLMoE's statistic to
  0.9092 and leaves only 0.0028 of margin to the most similar negative layer;
  a Q4-router fingerprint is not comparable to a full-precision one. Every
  fingerprint therefore records `router_precision` and `quant_type`.

**Recommendation.** Use routing overlap as a corroborating, aggregate lineage
signal at the inference tier: hash the per-layer top-k sets, compute the
per-layer mean Jaccard against a pinned reference probe, and compare only
fingerprints with the same `n_experts`, `top_k`, probe, model dtype and router
precision. With a full-precision router the fixed mean−3sd threshold (0.8936
for OLMoE E=64/k=8) accepts all 16 requantized layers and rejects all 16
fine-tuned-checkpoint layers, with 0.0345 of margin above and 0.0234 below;
at E=4/k=2 the separation is much wider (0.9694 vs 0.3907, the latter at
chance). Do not use a per-row threshold, do not compare fingerprints across
different router precisions, and do not treat exact top-k set agreement as
the gate at high k. The measured FPR is conservative for a truly unrelated
same-architecture checkpoint, because the negative used here is a fine-tune
that still shares 0.8414 Jaccard with its base.

## Limitations vs #69

Issue #69 asked for a Jaccard distribution for requantizations of one model
*versus two genuinely different models, on at least two families*, and for a
threshold with an estimated false-positive rate at catalogue scale. This run
delivers the threshold and pre-registers it; the contrast side and the
catalogue scale are only partly delivered. Each gap is stated here instead of
being left implicit.

### Cell coverage

| family | E / k | requantized self | different checkpoint | different family |
|---|---|---|---|---|
| TinyMixtral-4x248M-MoE | 4 / 2 | `experts_q4`, `all_q4` | none | TinyLLama-4x1.1B-MoE (J 0.3907) |
| OLMoE-1B-7B-0924 | 64 / 8 | `experts_q4`, `all_q4` | OLMoE-0924-SFT (J 0.8414) | none (explicit skip) |

So the "two genuinely different models on at least two families" cell exists
once, at E=4/k=2, and not twice. The matrix doubles as the per-family table the
ticket asks for: one row per family, the value of every contrast that exists,
and empty cells marked as gaps rather than filled.
`FAMILY_SKIP_NOTES` in `config.py` is the machine-readable form of the E=64
skip.

**Why E=64 has no genuine negative.** No cached, weight-complete, non-OLMoE
64-expert MoE exists on this machine: `deepseek-moe-16b-base` has only its
index file cached and needs ~32 GB bf16 (over the 20 GB process budget /
16 GB per-model budget), and `Qwen1.5-MoE-A2.7B` has 60 experts, so its expert
id space is not comparable. Comparing different E values would silently map
unrelated expert ids, so it is not done.

The only same-architecture E=64 negative is 0924-SFT, which **shares ancestry**
with 0924 — it is a descendant, not a genuinely different model. Its J = 0.8414
therefore *overstates* the overlap an unrelated same-architecture checkpoint
would show, which makes the measured 0/16 FPR conservative in the safe
direction. That is an inference about direction, not a second measurement, and
the E=64 configuration carrying the verdict has **no** different-model
measurement at all. The verdict is supported by requant-vs-descendant only.

### FPR has no DB-scale upper bound

Issue #26 sizes the catalogue at 1.82M Hub models with 0.54M declared
dependency edges (Chen et al., ICSE 2026, as of 2025-06). Nothing in this run
measures false accepts at that scale, and the sample cannot bound it:

- **One negative pair, so n = 1 at the model level.** The 16 "trials" behind
  0/16 are layers of a single model pair, and those layer values are strongly
  correlated (that pair spans 0.8201–0.8702), so the effective sample size is
  nearer 1 than 16.
- **The negative is a descendant.** It is the hardest available negative, not a
  typical one, so its overlap is an upper end of what an unrelated checkpoint
  would show.
- **The catalogue is heterogeneous.** It mixes requantizations, fine-tunes and
  unrelated models in unknown proportions, and a per-layer decision needs a
  pinned reference per candidate.

The binomial (Wilson) bound for 0/16, stated only to show what this sample
rules out: 95% two-sided upper bound **0.1936**, one-sided 95% **0.1446**.
Under an independence assumption that does not hold here, the point estimate of
false accepts is 0 per 1,000 and per 10,000 candidate models, while the 95%
bound allows up to **≈194 per 1,000** and **≈1,936 per 10,000** single-test
decisions (one decision per candidate).
Used as a whole-model gate (all 16 layers above `T`), independence would put the
false-accept probability at 0.1936¹⁶ ≈ 4e-12. Those figures are the reason the
DB-scale FPR is reported as *unbounded*, not as "≈0": a per-layer independence
assumption is what makes the second number look small, and it is false.

### TinyLLama cell: chance does not identify the cause

J = 0.3907 against chance 0.3889 (+0.0018) is what independent routing looks
like, but a **tokenizer mismatch would produce the same value**: feeding the
wrong token ids to any MoE also destroys overlap. Issue #26 records
cross-tokenizer null inflation of 0.027–0.128 by family, so a tokenizer
artifact is a live alternative here, not a hypothetical one. The cell
therefore supports "no lineage signal detected at this probe", not "the two
models are independent *and* the probe tokenized correctly". The tokenizer
argument in *Method* — that TinyMixtral's `tokenizer.model` SHA256 is the
published Mistral/Llama-2 32k blob and matches TinyLLama's vocab and special
ids — is an inference recorded before the run, not a measured control. A
tokenizer-identity check on both sides would be needed to close this cell.

### TPR is by construction

`T = mean − 3 × sample_sd` is placed relative to the calibration population
itself, so 16/16 TPR is mechanical: a one-sided 3-sd tail leaves nearly all 16
layers above `T` whatever the routing is doing. It carries no evidential
weight about signal quality. The evidential content of the calibration row is
the negative side — 0/16 negative layers at `T`, the most similar negative
layer 0.0234 below it, and a worst-layer population gap of +0.0579.

## Files

- `config.py` — pinned checkpoints/revisions, populations, budget, threshold
  and verdict rules, pre-registered expectations.
- `smoke.py` — subprocess worker + parent comparison/calibration; imports the
  QDQ contract and probe from `smokes/routing-agreement`.
- `results.json` — machine-readable record of the run above (per-layer
  mean/min/max/sd Jaccard, set agreement, fingerprints carrying router
  precision, quant type and probe identity, calibration, expectations).
