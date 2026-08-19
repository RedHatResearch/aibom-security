# AGENTS.md — verifier package

Context for `aibom_verifier` and the architecture-validation PoC. Root [AGENTS.md](../AGENTS.md) covers repo-wide workflow, board protocol, and commands.

## Package role

Library implementing the provenance verification pipeline. Not installed standalone — consumed by the `aibom` CLI in `../cli/`.

## Key modules

Paths under `src/aibom_verifier/`:

| Area | Path |
|------|------|
| CLI verify entry | `cli.py` |
| Orchestrator + rules | `orchestrator.py`, `rules.py`, `data/t1_default.yaml` |
| Execution backends | `backends/local.py`, `backends/ssh_local.py`, `backends/compose_queue.py` |
| Remote single-node | `run_test.py`, `worker.py` |
| Proxy store | `slots/proxy_store.py`, `store_factory.py` |
| Structured logs | `run_log.py`, `observer.py`, `worker_log.py` |
| PoC compose stack | `../docker-compose.yml` (repo root) |

## Verify CLI (common paths)

```bash
# Default: local in-process, filesystem cache
uv run aibom verify org/target --base org/base

# Shared proxy store (env AIBOM_STORE=proxy or flag)
uv run aibom verify org/target --base org/base --store proxy --ignore-cache

# Compose workers (needs compose up + AIBOM_REDIS_URL on host)
uv run aibom verify org/target --base org/base --store proxy --backend compose
```

- `target` — HF repo id being verified.
- `--base` — overrides model card `base_model` when set.
- `--ignore-cache` — skip artifact **reads**; still **writes** to store.
- `--backend` — `local` (default), `ssh`, `compose`.

Host proxy env: root `.env.example`. Container services use Docker DNS (`postgres`, `minio`, `redis`).

## Architecture PoC (throwaway-ok)

Prefer working prototype over polish. Stack may be thrown away later.

- Postgres + MinIO proxy store; 30-day LAT cache sweep (`aibom cache-sweep --max-age-days 30`), also run by a Compose sweeper timer.
- PoC entry is CLI-first: `aibom verify` uses proxy and can submit to Compose/SSH backends.
- Compose job queue: plain Redis list + JSON (`LPUSH`/`BRPOP`); **no RQ**; **no Dask** in M1.
- Worker replicas: `docker compose up -d --scale worker=N`.
- SSH-to-localhost: thin demo (`ssh localhost` → `run_test`); not the OpenShift target.
- OpenShift AI / MetaCentrum: documented upgrade paths on `ExecutionBackend` only — not M1 demos until cluster architecture is known.

PoC orchestration code lives in this package; root `docker-compose.yml` builds the same image.

## Detection and orchestration facts

- Block-0 / first-layer checks: safetensors headers + ranged tensor bytes — not full checkpoint downloads.
- Live T1→T2 gates on boolean pass; float-threshold gating is an orchestrator seam (stub + unit tests; `> 0.8` example).
- Test deps: small YAML/JSON rule list with forward walk — not an RPM/SAT solver.
- `arch_hash` normalizes Transformers absent-vs-default config fields (e.g. `head_dim`, `mlp_bias`) before hashing.
- FR-8 abandoned: no new lineage JSON fields; `VerificationResult` / `RunResult` are the public contract.
- Reference smoke pair: `SultanR/SmolTulu-1.7b-Instruct` vs `HuggingFaceTB/SmolLM2-1.7B`.

## Logging

- **stdout** — `VerificationResult` JSON (pipeline/consumer parse target).
- **stderr** — JSONL telemetry (`run_id`, events). Do not move JSONL to stdout.
- Optional `AIBOM_LOG_FILE` tees the same JSONL (best-effort; write failures do not fail verify).
- Host CLI honors `AIBOM_RUN_ID` when set; otherwise mints a UUID.
- Compose worker JSONL is on container stderr (`json-file` driver). Collect with `docker compose logs --no-log-prefix worker`.
- See #42 for pipeline telemetry scope; #43 for orchestration (M1.2).

## Smokes vs verifier

| Location | Purpose |
|----------|---------|
| `../smokes/<topic>/` | Survey runnable checks (#26 / #17); not the product verifier |
| `../.scratch/<tool>/` | Clones of upstream tools — not committed product code |
| `tests/` here | Shipped unit/integration tests for verifier |

Smokes need HF network and multi-GB downloads for dense models; not CI. Each topic README links paper, #26 (or #17) note, and run command.

When a smoke graduates into the verifier, move logic here and thin or delete the smoke wrapper.

## Tests in this package

```bash
uv run pytest verifier/tests -m "not network"
```

Prefer fakeredis for compose-queue tests; no real Redis required for offline CI.
