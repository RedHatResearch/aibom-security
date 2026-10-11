# Origin Tracer smoke

- Paper: [Origin Tracer, arXiv:2505.19466](https://arxiv.org/abs/2505.19466)
- Code: [`smokes/origin-tracer/`](https://github.com/RedHatResearch/aibom-security/tree/main/smokes/origin-tracer)

This is a survey smoke, not verifier product code. Origin Tracer is an
activation-assisted grey-box method: it needs forward passes, not just
weights. For a specified base/candidate pair it runs one-token probes
through the base, injects each base layer input into the candidate layer,
reconstructs the candidate post-attention intermediate by optimizing it
through the base MLP, and estimates a low-rank signal from the
base-versus-reconstructed intermediate differences.

## Assumptions

- Both models are same-shape decoder-only Transformers.
- The candidate is derived from the specified base with unchanged MLP
  functions.
- The meaningful change is a low-rank LoRA-style update to attention V/O
  projections.
- Both tokenizers produce the same IDs for the selected one-token word
  probes.
- MoE models are rejected by the default config preflight before weight
  download. `--allow-moe` is an explicit experimental diagnostic path, not
  general MoE support.
- Families whose layer forward does not match the reconstruction formula
  `z_residual + mlp(norm(z_residual))` are excluded: gemma2, gemma3, and
  olmo2 norm the attention output before the residual add and wrap the MLP
  in `pre_feedforward_layernorm` / `post_feedforward_layernorm`. Running
  the reconstruction on those layouts fabricates rank, so preflight and the
  loaded-model check both reject them. The fixtures exercise llama-layout
  families only.

The smoke checks these assumptions and emits `OK`, `INCOMPATIBLE`, or
`UNSUPPORTED` JSON rather than treating an arbitrary fine-tune as a valid
Origin Tracer pair. MLP-targeted or other non-V/O changes are
`UNSUPPORTED`. Cross-shape pairs are `INCOMPATIBLE`. An omitted
base/candidate pair is an explicit same-base applicability failure.

The shared SFT fixture pairs are not LoRA positives. Use them for
applicability and abstention checks only. The default MoE rejection is the
safety boundary: do not add `--allow-moe` when a general applicability
result is wanted.

## Run without downloads

`--help` imports no model and `--dry-run` performs no Hub, config, or model
download:

```bash
uv run --group smokes python smokes/origin-tracer/smoke.py --help
uv run --group smokes python smokes/origin-tracer/smoke.py --dry-run
uv run --group smokes python smokes/origin-tracer/smoke.py --dry-run --pair unrelated_cross_family
```

`--preflight` downloads or reads only `config.json` for the specified pair.
It does not load weights or run probes:

```bash
uv run --group smokes python smokes/origin-tracer/smoke.py \
  --preflight \
  --base-model HuggingFaceTB/SmolLM2-135M \
  --candidate-model HuggingFaceTB/SmolLM2-135M-Instruct
```

A fixture label may be used instead of both IDs. Active and deferred labels
from [`../fixtures.yaml`](../fixtures.yaml) are accepted (read through
`smokes/_lib/pairs.py`):

```bash
uv run --group smokes python smokes/origin-tracer/smoke.py --preflight --pair moe
```

`--allow-moe` is opt-in and experimental. It enables the narrow
routed-sparse path only when the loaded MLP exposes a gate/experts
interface such as Mixtral's `mlp.gate` and `mlp.experts`. It does not claim
general MoE support or expert lineage.

## Activation trace

The real path needs both model weights and tokenizers. Defaults are small:
eight probes, four subset cycles, 24 reconstruction steps, layer `0` only.
Increase them explicitly for an evaluation. All output is JSON.

```bash
uv run --group smokes python smokes/origin-tracer/smoke.py \
  --base-model <base-id-or-local-path> \
  --candidate-model <candidate-id-or-local-path> \
  --layers 0,1 \
  --probe-count 16 \
  --cycles 8 \
  --reconstruction-steps 48 \
  --learning-rate 0.05 \
  --device cpu \
  --cache-dir ~/.cache/huggingface
```

`--probe-count` is at least 3. A 2-row difference matrix has a single
log-gap, so any nonzero result would report rank 1 regardless of the data.

Each cycle samples `min(probe_count, hidden_size // 2)` probe rows. When the
sample covers every row, the cycles would decompose the identical matrix,
so one cycle is computed and reported with `sampled_all_rows: true` instead
of identical copies.

The dense path keeps the original reconstruction and rank estimate. Under
`--allow-moe`, each selected layer additionally reports:

- the candidate post-attention residual `Y_c` diagnostics and direct
  residual-delta SVD;
- router route captures from both models, with top-k IDs, weights, margins,
  and route switches;
- an oracle check that the base routed MLP reproduces the candidate output
  at the captured candidate residual.

The direct residual delta is diagnostic evidence only. The reported rank
stays the SVD of `base_intermediate - reconstructed_intermediate`. An
experimental MoE result is `UNSUPPORTED`, never a successful rank-0 result,
when route diagnostics are missing, oracle MSE is not near numerical
precision, reconstruction makes no material progress, or the reconstructed
rank is zero. Experimental `OK` requires all of those checks plus exact
V/O-only state changes.

Useful controls: `--pair`, `--base-id`/`--candidate-id`, `--probe-words
word1,word2,...`, `--layers all`, `--dtype {auto,float32,float16,bfloat16}`,
`--seed`, `--revision`, `--local-files-only`.
`--require-base-metadata` makes the preflight abstain unless the candidate
config contains `base_model_name_or_path`. Without it, same-base evidence
comes from equal non-V/O tensors plus the explicit pair supplied by the
operator.

The trace uses Transformers hooks and autograd. It captures the base layer
input and post-attention residual, replaces the candidate layer input with
the base input, and captures the candidate layer output. For each layer it
optimizes `z` through the base post-attention norm and MLP (or the base
routed sparse MLP under `--allow-moe`) so `z + MLP_base(norm_base(z))`
matches the candidate output. Rows of `base_intermediate - z` are sampled,
decomposed with SVD, and split after the largest consecutive log
singular-value gap. The reported rank is the minimum cycle rank, then the
minimum over selected layers.

Determinism comes from a seeded `numpy` Generator in the sampling step and
the fixed probe word list. No global RNG is seeded because nothing consumes
it.

## Opt-in TinyMixtral MoE experiment

[`moe_smoke.py`](moe_smoke.py) is the focused opt-in experiment. It loads
`Isotonic/TinyMixtral-4x248M-MoE` from the network or the local cache,
writes a temporary candidate with exactly one scalar update to one attention
`v_proj` weight, and delegates to the existing implementation with
`--allow-moe`. It does not duplicate the Origin Tracer algorithm and is not
run by the repository's default smoke commands.

```bash
uv run --group smokes python smokes/origin-tracer/moe_smoke.py \
  --base-model Isotonic/TinyMixtral-4x248M-MoE \
  --layer 0 \
  --delta 0.001 \
  --layers 0 \
  --probe-count 8 \
  --cycles 4 \
  --reconstruction-steps 24 \
  --learning-rate 0.05 \
  --device cpu \
  --dtype float32 \
  --cache-dir ~/.cache/huggingface/hub
```

The layer whose V weight gets the scalar update is matched by exact path
segment, so `--layer 1` does not collide with layers 10 and 11.

Use `--candidate-dir` to keep the generated candidate, or
`--local-files-only` to require an already cached model. A small edit with
`--delta 0.001` and `--learning-rate 0.05` produced equal initial/final
reconstruction MSE, `best_step=0`, and rank `0`; the strict path correctly
returned `UNSUPPORTED`. That is a reconstruction failure with no progress,
not a valid zero-rank MoE result.

The same controlled experiment passed with a smaller reconstruction step
and a larger V edit:

```text
delta 0.05 · learning_rate 0.0001 · 48 steps · layer 0
exactly one V scalar changed · route_switch_fraction 0.0
oracle MSE 0.0 · final/initial MSE 0.0233 · reconstructed rank 1
status: OK (experimental only)
```

This demonstrates a narrow route-stable control with unchanged router and
unchanged experts, not general MoE lineage support. Only a run satisfying
every strict condition may emit experimental `OK`.

## Deviations and limitations

The paper does not publish enough hyperparameters for byte-for-byte
reproduction. This smoke exposes probe count, cycles, reconstruction steps,
optimizer learning rate, layers, device, dtype, seed, and cache explicitly.
It uses a deterministic built-in word list (or `--probe-words`) and keeps a
word only when both tokenizers produce one identical token. The paper's
phrase "half-hidden-size probe subsets" is ambiguous; this implementation
samples rows at `min(probe_count, hidden_size // 2)`. It uses the layer's
post-attention residual as the intermediate, reconstructs with the base
post-attention norm plus MLP, and splits at the largest consecutive log
singular-value gap because no calibration or null threshold is available.
Supported module layouts are narrower than the Transformers decoder family.
These choices are reported in every JSON result under `deviations`.

This smoke is not passive or static: it needs executable forward passes,
activation hooks, and differentiable reconstruction. It is not proof that a
candidate was produced by LoRA, and it must not be used as a product
verifier or as a positive claim for an arbitrary SFT pair.

## Minimal evaluation matrix

Use a small same-family model pair first, then include explicit abstention
cases. Record the complete JSON output and the exact parameter settings.

| Case | Expected result | Purpose |
|---|---|---|
| Same base + controlled attention V/O LoRA merge (same tokenizer) | `OK`, finite per-layer diagnostics and rank estimates | Method-positive smoke; construct this pair separately because shared fixtures do not establish a LoRA positive |
| Same base + MLP LoRA or dense MLP fine-tune | `UNSUPPORTED` with MLP-targeted reason | Confirms the unchanged-MLP precondition is enforced |
| Same-family but different hidden/depth/vocabulary shape | `INCOMPATIBLE` | Cross-shape guard |
| MoE pair (OLMoE, Mixtral, Qwen-MoE) | `UNSUPPORTED` by default before weights; `--allow-moe` stays `UNSUPPORTED` unless routes, oracle, material progress, and nonzero rank all pass | Default safety boundary plus truthful experimental abstention |
| gemma2 / gemma3 / olmo2 pair | `UNSUPPORTED` | Reconstruction-formula guard |
| Different decoder architecture or fused-QKV model | `UNSUPPORTED` | Module applicability guard |
| Same-shape pair with tokenizer ID drift | `UNSUPPORTED` | Same-base probe condition guard |
| Missing one or both model IDs | `UNSUPPORTED` (or `DRY_RUN` for `--dry-run`) | Missing same-base condition and no-download path |

No permanent test fixture downloads model weights. Use `--dry-run` for
deterministic command-path checks and `--preflight --local-files-only` when
small local configs are available.
