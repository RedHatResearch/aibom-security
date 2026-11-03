# ProvenanceBench

ProvenanceBench is a small benchmark for passive, static-whitebox model
provenance checks. Each case asks whether a `subject` checkpoint is related to
a `reference` checkpoint.

The set is a pilot and grows as rows are added: 42 pairs, the original 40 plus two
same-architecture stranger rows. Existing rows are never edited, so results stay
comparable across methods and iterations; new rows are appended with their own
pins. One of the two new rows replicates an existing relation at a second scale,
which its `context` field records.

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
| `src/pbench/fastpath.py` | Extraction-cache executor, seam wrappers, run sidecar |
| `src/pbench/fastpath_store.py` | Extraction-cache on-disk store |
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

## Extraction cache (fastpath)

`fastpath.py` is a harness-only executor: the harness loads `detect.py`, then wraps
its `_load_gate_tensors` and `_svd_spectrum` in-process. The file is not edited, so
`method_sha256` still names the detector alone.

| Tier | Key | Value |
| --- | --- | --- |
| A: provenance replay | `repo_id@revision` | one serialized `Extraction`: layer → name, role, shape, spectrum |
| B: content identity | `sha256(fp32 matrix bytes):rows x cols` | the full fp32 singular-value spectrum |

Revisions are full 40-hex SHAs, so a tier A key names immutable content and a hit reads
zero weight bytes; only pinned revisions are cached, and a row is bound to the `detect.py`
sha256 that wrote it, so another detector misses. Tier B hashes the decoded matrix in
canonical C-order, so no matrix is SVD'd twice — across pair sides, formats decoding to
the same fp32 weights, and runs — and a decode change is a new key. `_svd_spectrum` widens
fp32 singular values to fp64, so storing fp32 bits and widening on read is exact, and the
whole spectrum is stored, not a top-`r` truncation. A replay also carries the provenance
recorded when it was computed, including `source.fetch` — how those bytes were obtained,
not this run's transport.

The store root holds `index.json` (authoritative, atomically replaced), `spectra.f32`
(append-only flat little-endian fp32; a torn tail is truncated at the next writable
open), and `store.lock` (one writer; a second process opens read-only and skips writes).
The default root is `.cache/fastpath/` in this checkout, ignored by git;
`PB_FASTPATH_STORE` overrides it and `PB_NO_FASTPATH=1` disables the cache.

### Open-time validation

At open the store verifies a sha256 of the blob's durable region and recomputes every
recorded probe row with the detector's own frozen call. Probes are ill-conditioned,
rank-deficient, or real gate shapes; a Gaussian row is only the cheapest pre-check,
blind to the change that matters. A digest or probe mismatch, a short blob, an unreadable
or malformed index, and a different store spec all quarantine the files as
`*.refused-<utc>` and rebuild, so a corrupted or truncated store is never served.

### Disclosure sidecar

A cached run writes `<stem>.fastpath.json` beside its `.method.jsonl` and
`.review.jsonl`, recording the executor's and store module's sha256, the detector
sha256, the wrapped seams, the relative store root, the validation result, the numeric
stack, and the hit and miss counters. Nothing is added to an artifact's `meta`: the
artifact proves the detector is frozen, not the executor, and the sidecar pins it.

Verdict-path numerics, verdicts, abstain reasons, diagnostics, and probe selection are
unchanged, and nothing is persisted for a failed or abstained extraction or a raising
SVD, so a later run re-reads the checkpoint instead of replaying a failure.

```bash
# cold run: populates the store
uv run pbench-eval --cohort core
# warm re-run: every side replays from (repo, revision), no weight bytes are read
uv run pbench-eval --cohort core
# reference run with the cache disabled
PB_NO_FASTPATH=1 uv run pbench-eval --cohort core
```
