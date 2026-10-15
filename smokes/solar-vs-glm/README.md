# Solar-vs-GLM dispute smoke (smokes/solar-vs-glm/)

Methodological reproduction of the **Solar-vs-GLM verdict-reversal dispute**
for the [#26](https://github.com/RedHatResearch/aibom-security/issues/26)
fingerprint survey. The methodology is ported **with citation; no code is
vendored** — both dispute repositories are unlicensed, so everything below is
a reimplementation from their published analyses (READMEs and figures).

## The dispute in one paragraph

In January 2026, [sionic-ai/solar-vs-glm](https://github.com/sionic-ai/solar-vs-glm)
published an accusation that Solar-Open-100B was derived from GLM-4.5-Air,
based on LayerNorm weight vectors showing ~0.99 raw cosine similarity between
the two models at matched layers, versus supposedly low within-model
across-layer similarity. Ko's rebuttal
([hyunwoongko/solar-vs-glm-vs-phi](https://github.com/hyunwoongko/solar-vs-glm-vs-phi))
reversed the verdict on three grounds: (1) **mean-centering** — because
LayerNorm γ is initialized at 1.0 and stays positive, the shared mean
direction dominates the raw dot product; after centering (Pearson), Solar–GLM
collapses to r ≈ 0.01 while the genuinely derived pair GLM–INTELLECT stays at
r ≈ 1.0; (2) the original within-model baseline was an **artifact of layer-0
`input_layernorm`** (compared only against layers 10/20/30/40), which is
atypical because it directly faces the raw embedding distribution, while
layer-0 `post_attention_layernorm` shows ≥0.92 cosine against later layers in
the same models; and (3) a **controlled GPT-2 experiment** — four tiny models
trained from scratch on distinct data streams with either ones- or random-init
LayerNorm — showing that ones-init cross-model LN cosine stays ≈0.999 while
random-init stays ≈0 regardless of any data or derivation. The accusation
repo itself was later rewritten into a
[retraction](https://github.com/sionic-ai/solar-vs-glm) applying the same
centered-cosine correction. **Caution statement: raw LayerNorm cosine
similarity is an initialization prior, not provenance evidence; only
mean-centered similarity (or byte-identity) carries signal.**

## What this smoke computes

For each pbench fixture pair (subject vs reference), over the layer grid
{0, n//4, n//2, 3n//4, n−1} and the tensor roles `input_layernorm.weight`,
`post_attention_layernorm.weight`, `self_attn.k_proj.weight`,
`self_attn.v_proj.weight`:

| metric | definition | notes |
|---|---|---|
| `cosine` | `cos(a.flat, b.flat)` | the accusation's headline metric; inflated by the shared mean direction on LayerNorm tensors |
| `centered_cosine` | cos of `(a − mean(a))` vs `(b − mean(b))` | the rebuttal's fix |
| `pearson` | mean of z-score products | mathematically equals `centered_cosine`; computed independently as a cross-check (`pearson_minus_centered` reported, ~1e-15) |
| `mean_abs_diff` | `mean(|a − b|)` | |
| `rel_l2` | `‖a − b‖ / ‖a‖` | **asymmetric**: `a` is always the pbench reference model; `rel_l2(a,b) ≠ rel_l2(b,a)` |
| `cv_diff` | `|std(a)/|mean(a)| − std(b)/|mean(b)||` | div-by-zero guarded (None when either mean vanishes) |
| `p99_abs_diff` | `|p99(|a|) − p99(|b|)|` | |

**Divergence check (the dispute's headline):** LayerNorm pairs are classified
`divergent` when raw cosine > 0.9 **and** centered cosine < 0.1 (the
Solar–GLM unrelated shape), `aligned` when both > 0.9 (the GLM–INTELLECT
derived shape).

**Layer-0 caveat record:** per model, layer-0 `input_layernorm` cosine
against layer-0 `post_attention_layernorm` is recorded separately, so the
accusation's artifact (layer-0 input LN being atypical) is directly visible.

**Byte-identity tier** (the accusation repo's `definitive_proof.py`): SHA-256
of the raw safetensors bytes of every grid LayerNorm tensor on both sides;
exact matches are reported (`identical: true`) — the only tier that
constitutes proof of derivation by this methodology.

**GPT-2 controlled experiment** (always runs, offline, CPU): four tiny GPT-2s
(`vocab_size=256, n_layer=12, n_head=8, n_embd=512`) trained from scratch,
300 Adam steps at lr 1e-3 on distinct arithmetic streams; two with
LayerNorm ones-init, two with random-init LN. Asserts ones-init cross-model
LN cosine > 0.99 and rand-init |cosine| < 0.2.

Skip handling: role/shape/layer-count mismatches (e.g. the head-expanded
MedIT upscale case) produce explicit `skipped` records, never a crash.
A pinned revision that ships **no safetensors at all** is likewise a
structural skip: the pair entry gets `status: "skipped"` with `reason:
"gguf_only_no_safetensors"` (mirroring the independence-tests smoke's
`skipped_format` policy), and the pair stays in the manifest.

### Download retry and the closed-client bug

Checkpoint downloads are wrapped in a small retry: up to **3 attempts**
with exponential backoff **2 s / 8 s**, covering transient
`httpx.ConnectError` (DNS/network hiccups) and the
`RuntimeError: Cannot send a request, as the client has been closed`
that `huggingface_hub` itself induces on such errors — its backoff
helper closes the shared `httpx.Client` on `ConnectError` (defensively,
for SSL errors) but then retries through the client object it captured
*before* closing it, and the resulting `RuntimeError` is not in its
retryable exception set. Before every retry this smoke forces a fresh
hub session (`huggingface_hub.utils.close_session()`), so each attempt
sends through a newly created, open client.

### Gate semantics

The `summary` object reports `counts: {ok, skipped, error}` and
`by_status` (pair labels per bucket). The run **exits 1 if any pair
ends in `status: "error"`**; `skipped` pairs (format-out-of-scope like
the no-safetensors BSL-470M, or shape/layer-count non-comparability) do not
fail the run. A failing GPT-2 control logs a warning but, as before,
does not by itself change the exit code.

## How to run

```bash
uv sync --all-packages --group smokes
uv run --group smokes python smokes/solar-vs-glm/smoke.py             # default fixtures
uv run --group smokes python smokes/solar-vs-glm/smoke.py --selftest  # offline, exit 0
uv run --group smokes python smokes/solar-vs-glm/smoke.py --full      # + granite/llama2 tier
uv run --group smokes python smokes/solar-vs-glm/smoke.py --only pos_qwen3_sft
uv run --group smokes python smokes/solar-vs-glm/smoke.py --gpt2-steps 100   # shorter control
uv run --group smokes python smokes/solar-vs-glm/smoke.py --no-gpt2          # skip the control
```

Stdout is one JSON object per result line (`gpt2_control`, one `pair` per
fixture, then `summary`); progress goes to stderr. Model ids and pinned
revisions are read from `provenancebench/pairs.yaml` at runtime. All
computation is CPU-only (Apple M3 Pro 36GB targets; no CUDA anywhere).
`meta-llama/Llama-2-7b-hf` (the `--full` tier) is manually gated on the Hub:
a set `HF_TOKEN` with granted access is required for that tier; the default
fixtures are all ungated. A missing token therefore only degrades `--full`.

## Fixtures (from provenancebench/pairs.yaml)

| label | pbench pair id | relation | role in this smoke |
|---|---|---|---|
| `neg_pds_bsl` | `pb-core-pds-bsl-470m` | unrelated | default negative: same Mistral config, from-scratch on different data — the Solar-vs-GLM structural situation in miniature. **Structurally unconsumable**: the pinned BSL-470M revision `1581b9dc` ships no safetensors (a `pytorch_model.bin` pickle — not a GGUF rehost; the skip reason label is the smoke's generic no-safetensors reason), so the pair always lands in `skipped` with reason `gguf_only_no_safetensors`; if Data-Selection/BSL-470M later gains safetensors at a new revision, repin and re-run to score it for real |
| `neg_smollm_smollm2` | `pb-core-smollm-smollm2-strangers` | unrelated | default negative: same layout, SmolLM2 trained from scratch |
| `pos_qwen3_sft` | `pb-smoke-qwen3-sft` | related | positive: instruct post-train of the base — the GLM–INTELLECT tier |
| `pos_smoltulu` | `pb-smoke-smoltulu` | related | positive: SFT/DPO of SmolLM2-1.7B |
| `caveat_medit_upscale` | `pb-core-smollm2-medit-upscale` | unsupported | layer-0 caveat case: attention heads expanded 32→64, so k/v projections are shape-mismatched — must produce explicit skip records, not a crash |
| `neg_granite_llama2` (`--full` only) | `pb-core-llama2-granite7b-strangers` | unrelated | dispute analog: Granite-7B-base is a from-scratch Llama-2-7B layout replica — the exact Solar-vs-GLM situation (~28 GB download; off by default) |

## Pass criteria

- `--selftest` exits 0: synthetic ones+noise pairs must show raw cosine > 0.9
  with centered cosine < 0.1 (divergent), affine copies must classify
  aligned, Pearson must equal centered cosine to 1e-10, zero-mean and
  layer-grid guards hold, and the reduced-step (30) GPT-2 control must pass
  both init-mode assertions.
- **Gate: the run exits non-zero if any pair ends in `status: "error"`**
  (zero-error required for green). The summary carries
  `counts: {ok, skipped, error}`; skipped pairs do not fail the run.
- Default fixtures: negatives should show the divergent pattern on LayerNorm
  tensors (raw high, centered low) and **no** byte-identical norm tensors;
  positives should show aligned LayerNorm tensors (both raw and centered
  high), typically with byte-identical norm tensors where post-training left
  LayerNorm untouched (the `definitive_proof.py` tier).
- `neg_pds_bsl` must land in `skipped` with reason
  `gguf_only_no_safetensors` (the pinned BSL-470M revision has no safetensors).
- `caveat_medit_upscale` must complete with explicit shape-mismatch skip
  records for k/v projections (LayerNorm tensors still comparable). Observed
  in the 2026-10-13 run: all 10 grid LayerNorm tensors are **byte-identical**
  (`n_byte_identical: 10`) between MedIT-Upscale-2B and SmolLM2-1.7B-Instruct
  while every k/v projection is shape-incompatible — the strongest byte-tier
  signature of the depth-upscale construction.
- GPT-2 control (300 steps): `ones_init_min_cosine > 0.99` and
  `rand_init_max_abs_cosine < 0.2`.

## Provenance and license notes

- Methodology ported with citation from
  [sionic-ai/solar-vs-glm](https://github.com/sionic-ai/solar-vs-glm)
  (accusation, later self-retracted) and
  [hyunwoongko/solar-vs-glm-vs-phi](https://github.com/hyunwoongko/solar-vs-glm-vs-phi)
  (rebuttal, including the `train_gpt2.py` controlled experiment). Both
  repositories carry **no license**; no code is vendored — the metrics and
  training loop are reimplemented from their published descriptions.
- Model checkpoints are fetched from the Hub at the pinned revisions in
  `provenancebench/pairs.yaml`; per-model licenses are those repos' own
  (Apache-2.0 / Llama-2 community license / etc.). The `--full` tier
  involves the gated Llama-2-7b-hf.

## Timing

Recorded per result line (`compute_seconds`) and overall (`wall_seconds` in
the summary). Expect (Apple M3 Pro, CPU): GPT-2 control ~2–4 min at 300
steps; each default fixture is dominated by download time on first run
(checkpoints 0.5–3.4 GB), with scoring itself seconds per pair; the `--full`
granite/llama2 tier adds ~28 GB of downloads plus ~1–2 min of scoring.
