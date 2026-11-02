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

### What it is

`fastpath.py` is a harness-only executor. The harness loads `detect.py`, then installs
the executor, which wraps two functions of the loaded module — `_load_gate_tensors` and
`_svd_spectrum` — in-process. `detect.py` is not edited by this: its sha256 still names
the method artifact, and `PB_NO_FASTPATH=1` runs the same detector without the wrappers.
The executor only decides whether to compute something or to return bits an earlier call
produced; it never substitutes a different computation.

### The two tiers

| Tier | Key | Value | Why |
| --- | --- | --- | --- |
| A: provenance replay | `repo_id@revision` | serialized `Extraction`: layer → name, role, shape, spectrum | Revisions are full 40-hex SHAs, so the key names immutable content and a replay reads zero weight bytes. Rows are bound to the sha256 of the `detect.py` that wrote them and are a miss under any other detector. |
| B: content identity | `sha256(fp32 matrix bytes):rows x cols` | the full fp32 singular-value spectrum | A cold run never SVDs a matrix it has already seen: across the two sides of a pair, across formats that decode to the same fp32 weights, and across runs. The key hashes the decoded matrix, so a decode change is a new key, not a stale hit. |

Tier A is keyed only for pinned revisions. An unpinned extraction (`revision is None`)
is neither read from nor written to the store. Tier B hashes the matrix in canonical
C-order, so two element-wise identical inputs whose layouts differ share one key.

### Why it is bit-identical

The detector's `_svd_spectrum` takes the fp32 SVD and widens the singular values to
fp64. Those values are exactly fp32-representable, so storing the fp32 bits and widening
on read reproduces the frozen call's output exactly, and the whole spectrum is stored
rather than a top-`r` truncation, so every rank cutoff and tail statistic is still
available. A cache hit therefore returns the bits an earlier call produced; the store's
validation exists to test the antecedent of that guarantee — that the numeric stack did
not change.

### Store layout and environment

The store root holds three things:

| Path | Role |
| --- | --- |
| `index.json` | Authoritative index, replaced atomically (`index.json.tmp` → fsync → rename → directory fsync). |
| `spectra.f32` | Append-only flat little-endian fp32 spectra. An interrupted append leaves unreferenced bytes, which the next open truncates. |
| `store.lock` | One writer at a time. A second process opens read-only and skips writes. |

The default root is `.cache/fastpath/` under this checkout, which is ignored by git.
`PB_FASTPATH_STORE` overrides it; `PB_NO_FASTPATH=1` disables the cache entirely and
runs the detector's own calls.

### Open-time validation

Each store records probe rows and recomputes them at open with the detector's own frozen
call. The probe set is: a fixed-seed Gaussian 64², kept only as the cheapest pre-check
and labelled `blind` in the index; Hilbert 64² and 256², which are ill-conditioned;
a rank-one 96×192; an all-zero 32×48; and integer-arithmetic matrices at the real cohort
gate shapes 768×2048 and 2880×2880. Any mismatch quarantines the store files as
`*.refused-<utc>` and rebuilds the store; so does an unreadable or malformed index,
a truncated blob, or a different store spec.

The probe set is not a single Gaussian canary because a Gaussian canary is blind to the
change that matters. Measured: fp64 `dgesdd` and `dgesvd` agree bitwise on 200 of 200
Gaussian inputs, and the orientation change that really differs (transposing the input)
flips no fp32 value on Gaussian 64², 256², or 1024² inputs. Well-conditioned inputs round
the difference away, which is why the probes include ill-conditioned, rank-deficient, and
real-shape rows.

### Disclosure sidecar

A cached run writes `<stem>.fastpath.json` next to its `.method.jsonl` and `.review.jsonl`.
It records the executor's sha256, the store module's sha256, the detector's sha256, the
seams that were wrapped, the store root (relative, never absolute), the open-time
validation result, the numeric stack, and the hit and miss counters.

Nothing is added to an artifact's `meta`, so the artifact schema and every case record
stay byte-identical. Read that carefully: the artifact proves the detector is frozen, not
that the executor was. The sidecar is what pins the executor for a later reader.

### What it does not change

Verdict-path numerics, verdicts, abstain reasons, diagnostics, and probe selection are
unchanged. A failed or abstained extraction is never persisted, so a later run re-reads
the checkpoint rather than replaying a failure; a spectrum is stored only after the
frozen call returned it normally.

### Running it

```bash
# cold run: populates the store
uv run pbench-eval --cohort core
# warm re-run: every side replays from (repo, revision), no weight bytes are read
uv run pbench-eval --cohort core
# reference run with the cache disabled
PB_NO_FASTPATH=1 uv run pbench-eval --cohort core
```
