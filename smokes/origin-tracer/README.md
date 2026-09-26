# Origin Tracer smoke

- Paper: [Origin Tracer, arXiv:2505.19466](https://arxiv.org/abs/2505.19466)
- Code: [`smokes/origin-tracer/`](https://github.com/RedHatResearch/aibom-security/tree/main/smokes/origin-tracer)

This is a **survey smoke implementation**, not verifier product code. Origin Tracer is an activation-assisted grey-box method: it is not passive/static and cannot infer lineage from weights alone. For a specified base/candidate pair it runs one-token probes through the base, injects each base layer input into the candidate layer, reconstructs the candidate post-attention intermediate by optimizing it through the base MLP, and estimates a low-rank signal from the base-versus-reconstructed intermediate differences.

The implementation assumes:

- both models are same-shape decoder-only Transformers;
- the candidate is derived from the specified base, with unchanged MLP functions;
- the meaningful change is a low-rank LoRA-style update to attention V/O projections;
- both tokenizers produce the same IDs for the selected one-token word probes;
- **MoE models are rejected by the default config preflight before weight download**;
  `--allow-moe` is an explicit, experimental diagnostic path, not general MoE support.

The smoke checks those assumptions and emits `OK`, `INCOMPATIBLE`, or `UNSUPPORTED`
JSON rather than silently treating an arbitrary fine-tune as a valid Origin Tracer
pair. MLP-targeted or other non-V/O changes are `UNSUPPORTED`, cross-shape pairs are
`INCOMPATIBLE`, and an omitted base/candidate pair is an explicit same-base
applicability failure. The existing shared SFT fixture pairs are **not claimed to be
valid LoRA positives**; use them for applicability/abstention checks only. The
default MoE rejection is the safety boundary: do not add `--allow-moe` when a
general applicability result is wanted.

## Run without downloads

`--help` imports no model and `--dry-run` performs no Hub/config/model download:

```bash
uv run --group smokes python smokes/origin-tracer/smoke.py --help
uv run --group smokes python smokes/origin-tracer/smoke.py --dry-run
uv run --group smokes python smokes/origin-tracer/smoke.py --dry-run --pair unrelated_cross_family
```

`--preflight` downloads or reads only `config.json` for the specified pair. It does not load weights or execute probes:

```bash
uv run --group smokes python smokes/origin-tracer/smoke.py \
  --preflight \
  --base-model HuggingFaceTB/SmolLM2-135M \
  --candidate-model HuggingFaceTB/SmolLM2-135M-Instruct
```

A fixture label may be used instead of both IDs. Active and deferred labels from [`../fixtures.yaml`](../fixtures.yaml) are accepted:

```bash
uv run --group smokes python smokes/origin-tracer/smoke.py \
  --preflight --pair unrelated_cross_family
```

`--allow-moe` is intentionally opt-in and experimental. It enables the narrow
routed-sparse path only when the loaded MLP exposes a gate/experts interface such
as installed Mixtral's `mlp.gate` and `mlp.experts`; it does not claim general MoE
support or expert lineage.

## Activation trace

The real path requires both model weights and tokenizers. Defaults are intentionally small: eight probes, four subset cycles, 24 reconstruction steps, and layer `0` only. Increase them explicitly for an evaluation. All output is JSON.

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
The dense path retains the original reconstruction and rank estimate. With
`--allow-moe`, each selected layer additionally reports:

- the candidate post-attention residual `Y_c` diagnostics and direct
  `Y_c - Y_b` Frobenius/L2 norm, singular values, and effective numerical rank;
- an oracle MSE comparing the candidate layer output with
  `Y_c + MLP_base(norm_base(Y_c))`;
- base/candidate router top-k IDs and weights, route-switch fraction, and a
  selected-vs-unselected route margin when the runtime exposes it;
- reconstruction initial/final MSE, final-to-initial ratio, best optimizer step,
  and a `material`/`improved`/`no_progress` status;
- runtime, decoder-layer, and MLP class metadata.

The direct residual delta is diagnostic evidence only; it is **not** the reported
Origin Tracer rank estimate. The reported rank remains the SVD of
`base_intermediate - reconstructed_intermediate`. An experimental MoE result is
`UNSUPPORTED` (never a successful rank-0 result) when route diagnostics are
missing, oracle MSE is not near numerical precision, reconstruction makes no
material progress, or the reconstructed rank is zero. Experimental `OK` requires
all of those checks plus exact V/O-only state changes.

Useful controls are `--pair`, `--base-id`/`--candidate-id`, `--probe-words word1,word2,...`, `--layers all`, `--dtype {auto,float32,float16,bfloat16}`, `--seed`, `--revision`, and `--local-files-only`. `--require-base-metadata` makes the preflight abstain unless the candidate config contains `base_model_name_or_path`; without it, same-base evidence is established from equal non-V/O tensors plus the explicit pair supplied by the operator.

The trace uses Transformers hooks and autograd. It captures the base layer input and post-attention residual, replaces the corresponding candidate layer input with the base input, and captures the candidate layer output. For each layer it optimizes `z` through the base post-attention norm and MLP (or the base routed sparse MLP under `--allow-moe`) so `z + MLP_base(norm_base(z))` matches that candidate output. The matrix of `base_intermediate - z` rows is sampled repeatedly, SVD'd, and split after its largest consecutive log singular-value gap. The reported rank is the minimum cycle rank, then the minimum over selected layers.

## Opt-in TinyMixtral MoE experiment

[`moe_smoke.py`](moe_smoke.py) is the focused, opt-in experiment. It is clearly
network/heavy: it loads `Isotonic/TinyMixtral-4x248M-MoE`, writes a temporary
candidate with exactly one scalar update to one attention `v_proj` weight, and
then delegates to the existing implementation with `--allow-moe`. It does not
duplicate the Origin Tracer algorithm and it is not run by the repository's
default smoke commands.

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

Use `--candidate-dir` to retain the generated candidate, or
`--local-files-only` to require an already cached model. A small edit with
`--delta 0.001` and `--learning-rate 0.05` produced equal initial/final
reconstruction MSE, `best_step=0`, and rank `0`; the strict path correctly
returned `UNSUPPORTED`. That is reconstruction failure/no progress, **not** a
valid zero-rank MoE result.

The same controlled TinyMixtral experiment passed under a smaller reconstruction
step and larger V edit:

```text
delta 0.05 · learning_rate 0.0001 · 48 steps · layer 0
exactly one V scalar changed · route_switch_fraction 0.0
oracle MSE 0.0 · final/initial MSE 0.0233 · reconstructed rank 1
status: OK (experimental only)
```

This demonstrates a narrow route-stable, unchanged-router/unchanged-expert
control—not general MoE lineage support. Only a run satisfying every strict
condition may emit experimental `OK`.

## Exact deviations and limitations

The paper does not publish enough hyperparameters for byte-for-byte reproduction. This smoke therefore exposes probe count, cycles, reconstruction steps, optimizer learning rate, layers, device, dtype, seed, and cache explicitly. It uses a deterministic built-in word list (or `--probe-words`) and retains words only when both tokenizers produce one identical token. The paper's phrase “half-hidden-size probe subsets” is ambiguous; this implementation samples rows at `min(probe_count, hidden_size // 2)`. It uses the layer's post-attention residual as the intermediate, reconstructs with the base post-attention norm plus MLP, and uses the largest consecutive log singular-value gap without an unavailable calibration/null threshold. Supported module layouts are intentionally narrower than all Transformers architectures. These choices are reported in every JSON result under `deviations`.

This smoke is not passive/static: it needs executable model forward passes, activation hooks, and differentiable reconstruction. It is not a proof that a candidate was produced by LoRA, and it must not be used as a product verifier or as a positive claim for an arbitrary SFT pair.

## Minimal evaluation matrix

Use a small same-family model pair first, then include explicit abstention cases. Record the complete JSON output and the exact parameter settings.

| Case | Expected applicability/result | Purpose |
|---|---|---|
| Same base + controlled attention V/O LoRA merge (same tokenizer) | `OK`, finite per-layer diagnostics and rank estimates | Method-positive smoke; construct this pair separately because shared fixtures do not establish a LoRA positive |
| Same base + MLP LoRA or dense MLP fine-tune | `UNSUPPORTED` with MLP-targeted reason | Confirms the unchanged-MLP precondition is enforced |
| Same-family but different hidden/depth/vocabulary shape | `INCOMPATIBLE` | Cross-shape guard |
| MoE pair (OLMoE, Mixtral, Qwen-MoE) | `UNSUPPORTED` by default before weights; `--allow-moe` remains `UNSUPPORTED` unless routes, oracle, material progress, and nonzero rank all pass | Default safety boundary plus truthful experimental abstention |
| Different decoder architecture or fused-QKV model | `UNSUPPORTED` | Module applicability guard |
| Same-shape pair with tokenizer ID drift | `UNSUPPORTED` | Same-base probe condition guard |
| Missing one or both model IDs | `UNSUPPORTED` (or `DRY_RUN` for `--dry-run`) | Missing same-base condition and no-download path |

No permanent test fixture downloads model weights. Use `--dry-run` for deterministic command-path checks and `--preflight --local-files-only` when small local configs are available.
