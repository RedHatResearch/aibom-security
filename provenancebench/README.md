# ProvenanceBench

**Stub benchmark** — taxonomy + seed pairs for [#36](https://github.com/RedHatResearch/aibom-security/issues/36). Full pair import from [#45](https://github.com/RedHatResearch/aibom-security/issues/45) is follow-up work.

Fixed pairs for passive, weight-level `base_model` lineage verification on Hugging Face checkpoints.

## Files

| File | Role |
|------|------|
| `kinds.yaml` | Transformation kinds and coverage gaps (`fill`) |
| `pairs.yaml` | Labeled `ref` / `sus` cases |
| `smoke.txt` | Ordered ids for fast autoresearch loops |
| `detect.py` | **Agent edits this** — scoring logic |
| `src/pbench/load.py` | Read bench YAML; resolve `*_rev` pins |
| `src/pbench/eval.py` | **Frozen** harness (`pbench-eval`) |
| `runs/latest.jsonl` | Full eval artifact: meta + per-case snapshot + trailer (gitignored) |
| `runs/latest.triage.jsonl` | Triage artifact: same run without gold `y` (gitignored) |
| `.cache/hub/` | Local Hub weight cache (gitignored) |

Anchors `pb-001`–`pb-003` align with [`smokes/fixtures.yaml`](../smokes/fixtures.yaml).

## Fields

| Field | Meaning |
|-------|---------|
| `stack` | `dense` or `moe` — weight layout |
| `kind` | Transformation type (see `kinds.yaml`) |
| `y` | Ground truth: `related`, `unrelated`, or `unsupported` (cannot compare) |
| `pred` | Detector output: `related`, `unrelated`, or `abstain` |
| `ref` / `sus` | Hugging Face repo ids |
| `ref_rev` / `sus_rev` | Optional git SHA on that repo — pin weights for reproducibility |
| `claim` | Optional HF `base_model` tag — audit only, not ground truth |
| `cite` | Why we believe `y` |

## What `load` vs `eval` do

**`load.py`** — bench *data* only. Reads `kinds.yaml`, `pairs.yaml`, `smoke.txt`. Resolves `model_pin(case, "ref"|"sus")` → `(repo_id, revision)`. No scoring, no Hub downloads.

**`eval.py`** — frozen *harness*. Loads a suite, calls `detect.score()` per case, writes `runs/latest.jsonl`, prints one `SUMMARY` line. Autoresearch agents must not edit this file.

**`detect.py`** — *method under test*. Returns `pred` (`related` | `unrelated` | `abstain`) and `confidence` (float). Gold `y` uses `unsupported` when the correct answer is “cannot compare”; detector abstains with `pred=abstain`.

## Caching

Weights are not re-downloaded every run if you:

1. Set `HF_HUB_CACHE` to `provenancebench/.cache/hub` (or let detect default to that path).
2. Pin `ref_rev` / `sus_rev` so the cache key is stable across Hub `main` moves.

The bench does not call the Hub API during dry-run stub eval. Network hits happen only inside `detect.py` once you implement real signals.

## Run

```bash
uv sync --all-packages
uv run pbench-eval --suite smoke
```

Writes `provenancebench/runs/latest.jsonl` and `latest.triage.jsonl`. Each eval row embeds ref/sus, revisions, `kind`, `cite`, `claim`, and `label_confidence` so triage never needs `pairs.yaml`. Triage reads `.triage.jsonl` only (no gold `y`).

## Autoresearch

Edit `detect.py` only. Do not edit `src/pbench/eval.py`.

`detect.score()` receives **ref/sus, revisions, and stack** — never ground-truth `y`, `kind`, or `claim`. Cheating by reading labels from the case dict is blocked by the harness.

Optimize on **`--suite full`** before claiming a win; smoke is for fast iteration only.

Primary score: **`bal_acc`** (macro per-class recall). A pass also needs low `false_confirm`, `miss`, and `unsupported_false_confirm`. Abstain-all should not look like success (`cov` = fraction answered).
