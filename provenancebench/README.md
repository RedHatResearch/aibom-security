# ProvenanceBench

ProvenanceBench is a small fixed benchmark for passive, static-whitebox model
provenance checks. Each case asks whether a `subject` checkpoint is related to
a `reference` checkpoint.

The set is a pilot and grows when a new case class is added: 42 pairs, the
original 40 plus two same-architecture stranger rows. Existing rows are never
edited, so results stay comparable across methods and iterations; new rows are
appended with their own pins.

## Scope

The benchmark uses model weights, configuration, and MoE structure where
available. It does not use model outputs, watermarking, or parent retrieval.

Background and usage notes live on the
[wiki](https://github.com/RedHatResearch/aibom-security/wiki/ProvenanceBench).

## Files

| File | Purpose |
| --- | --- |
| `pairs.yaml` | Pair manifest and expected result |
| `kinds.yaml` | Transformation kinds used by the manifest |
| `cohorts.yaml` | Ordered `smoke`, `core`, and `all` cohorts |
| `detect.py` | Detector under test |
| `src/pbench/` | Loader and evaluation harness |
| `runs/` | Local JSONL run artifacts; ignored by git |

## Run

```bash
uv sync --all-packages
uv run pbench-eval --cohort smoke
uv run pbench-eval --cohort core
uv run pbench-eval --cohort all
```

The harness resolves bench data relative to this checkout. `pbench-eval` is
not usable from an installed wheel; run it from a clone of this repository.

Runtime dependencies are declared in `provenancebench/pyproject.toml`;
`uv sync --all-packages` installs them. `torch` is optional and only needed for
legacy `pytorch_model.bin` checkpoints.

Run `smoke` first, then `core`, then `all`. For a focused run:

```bash
uv run pbench-eval --ids pb-smoke-smoltulu pb-core-tulu3-sft
uv run pbench-eval --stack moe --kind quant
```

Each run writes a detector-only `.method.jsonl` artifact and a matching
`.review.jsonl` artifact. The detector sees only the pair IDs, revisions, and
stack. It does not receive labels or evidence.

When a detector downloads models, it should use the normal Hugging Face cache,
which is shared across runs and cloned checkouts. Set `HF_HOME` or
`HF_HUB_CACHE` only when you want to move that cache; do not put model files in
this repository.

## Pair fields

The important fields are:

| Field | Meaning |
| --- | --- |
| `subject`, `reference` | Checkpoints being compared |
| `subject_rev`, `reference_rev` | Immutable Hub revisions |
| `expected` | `related`, `unrelated`, or `unsupported` |
| `evidence` | Sources and short rationale for the expected result |
| `lineage_scope` | `direct` or `transitive` for related cases |
| `challenge` | Optional control metadata |

`unsupported` means the pair cannot be compared by the intended method. A
detector answers that case with `abstain`.

## Detector contract

Implement `score()` in `detect.py`:

```python
def score(
    subject: str,
    reference: str,
    *,
    subject_rev: str | None = None,
    reference_rev: str | None = None,
    stack: str,
) -> Result:
```

`subject` and `reference` are Hub repo ids; `subject_rev` and `reference_rev`
are the full immutable SHAs pinned in `pairs.yaml` — pass them to
`huggingface_hub` (for example `snapshot_download(repo_id, revision=rev)`)
so every run scores the same bytes. `stack` is `dense` or `moe`.

It returns:

```python
Result(
    verdict="related" | "unrelated" | "abstain",
    score=float | None,
    diagnostics={...},
    note="",
    abstain_reason=None,
)
```

`score` is null when the method abstains; `note` is a short free-text line and
`abstain_reason` a short token — the current detector emits `format_unreadable`,
`no_gate_tensors`, or `degenerate_profile`. The harness never passes labels,
evidence, or kind to the detector.

The current detector is the GATE spectral fingerprint: for every gate-like FFN
matrix it takes the fp32 SVD at extraction time and keeps only the singular
values, pools the per-layer spectral-entropy profile into eight depth segments,
and returns `related` when the linear-CKA similarity of the two profiles is at
least 0.9 and `unrelated` below it. Weights are read without full snapshots:
safetensors shards are fetched tensor by tensor over HTTP range requests (with a
cached snapshot as fallback), GGUF files are dequantized with `gguf.quants`,
legacy `pytorch_model.bin` checkpoints (TinyMixtral layout, single-file only:
sharded or index-declared `.bin` checkpoints abstain as `format_unreadable`,
unlike the smoke's shard-aware loader) go through
`torch.load(..., weights_only=True)`, AWQ int4 and FP8 compressed-tensors gates
are dequantized in-process, and packed MXFP4 expert blocks are dequantized before
the SVD. It abstains only when a checkpoint is structurally unreadable, has no
gate-like FFN tensors, or the two profiles are degenerate (`degenerate_profile`:
a constant pooled profile leaves no variance to correlate).

Non-gated architectures are scored through the paper's own †-style
down-projection fallback (GPT-2 `mlp.c_proj`, GPTNeoX `mlp.dense_4h_to_h`,
Nemotron-H `mixer.down_proj`), so core-cohort numbers for those pairs
(`pb-core-nemotron-h-4b`, `pb-core-gpt2-fineweb-fp8`) read a substituted down
projection, not the gate branch.
