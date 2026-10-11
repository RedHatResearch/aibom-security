# ProvenanceBench

ProvenanceBench is a small fixed benchmark for passive, static-whitebox model
provenance checks. Each case asks whether a `subject` checkpoint is related to
a `reference` checkpoint.

The current 40-row set is a pilot. The detector can change; the pair list stays
fixed so results can be compared across methods and iterations.

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

Implement `score()` in `detect.py`. It returns:

```python
Result(
    verdict="related" | "unrelated" | "abstain",
    score=float | None,
    diagnostics={...},
)
```

The current detector is a stub that abstains on every case.
