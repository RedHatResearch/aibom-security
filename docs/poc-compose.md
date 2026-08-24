# Local Compose PoC

Throwaway-ok laptop stack for architecture validation: Postgres + MinIO + Redis + workers + sweeper. Not a supported production deployment. The stack may be replaced later.

For running verify as a **host pipeline leaf step**, use the Docker image contract in [job-contract.md](job-contract.md) instead.

## Start

Published ports bind to `127.0.0.1`. Data is ephemeral (no named volumes). Container services use Docker DNS; the host CLI uses `.env.example` (`localhost` published ports).

```bash
docker compose up -d --scale worker=2
cp .env.example .env   # local-only defaults; do not commit secrets
set -a && source .env && set +a   # required; copy alone does not set env
uv run aibom verify org/model --base org/base --store proxy --backend compose
```

## Telemetry

Worker JSONL is on container stderr. Collect it with:

```bash
docker compose logs --no-log-prefix worker
```

The host CLI honors `AIBOM_RUN_ID` when set. Optional `AIBOM_LOG_FILE` tees the same JSONL to a file (best-effort; write failures do not fail verify). See [job-contract.md](job-contract.md) for the full stdout/stderr contract.

## Internals

- Proxy store: Postgres metadata + MinIO blobs; 30-day LAT cache sweep (`aibom cache-sweep --max-age-days 30`), also run by a Compose sweeper timer.
- Job queue: plain Redis list + JSON (`LPUSH`/`BRPOP`); no RQ or Dask.
- Worker replicas: `docker compose up -d --scale worker=N`.
- SSH-to-localhost backend is a thin demo only; not the OpenShift target.

Deeper module map and backend notes: [verifier/AGENTS.md](../verifier/AGENTS.md).
