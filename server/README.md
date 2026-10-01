# aibom-server

REST wrapper around `aibom verify` for external pipelines (e.g. Tekton): a
pipeline POSTs Hugging Face model IDs, the server runs each through the full
verify happy path (model card `base_model` claim → support/arch gates →
block-0 checks → verdict), and per-run logs accumulate in a runs directory
for offline analysis of which models the checks can actually verify.

Fire-and-forget by design: `POST /verify` returns `202` + `run_id`
immediately; verdicts and telemetry are read from the runs directory, not
from the API.

## API

| Route | Method | Behavior |
|---|---|---|
| `/verify` | POST | `{"model_id": "org/model"}` → `202 {"run_id": …, "status": "queued"}`; `401` when a token is configured and the request is unauthenticated (auth runs before body validation: an unauthenticated request gets `401` even if its body is also malformed); `413` request body too large (auth and size are checked before the body is read); `422` malformed id; `503` when the queue is full |
| `/healthz` | GET | `{"status": "ok", "pending": N}` — `pending` is queued + running jobs |

There are deliberately no result/query endpoints. Each run lands on disk:

```text
runs/<run_id>/
  job.json        # model_id, status (queued → running → done | error), exit_code, timestamps
  telemetry.jsonl # verify stderr: JSONL envelope (run_start … run_finished), shared run_id
  result.json     # verify stdout: VerificationResult JSON
```

Error runs (verify timeout, subprocess spawn failure, output-persist failure,
worker crash, restart interruption) write an `error` message in `job.json`
instead of `exit_code`, and may lack `result.json` / `telemetry.jsonl`.

Each job subprocesses `aibom verify <model_id> --accept`: the base is the
model card's `base_model` claim, and accept mode turns resolve failures (no
claim, gated repo, unknown repo) into logged non-confirming runs instead of
errors. The subprocess inherits the environment, so `HF_TOKEN`,
`AIBOM_CACHE_DIR`, `AIBOM_LOG_LEVEL` and other verify settings pass through.

## Run locally

```bash
uv sync --all-packages
uv run aibom-serve --runs-dir ./runs
curl -sS -X POST localhost:8080/verify \
  -H 'content-type: application/json' -d '{"model_id": "org/model"}'
```

## Deploy (single container on a lab machine)

Generate an API token with `openssl rand -hex 32` and export it before
`docker run`; the server refuses to start if `AIBOM_API_TOKEN` is set but
empty.

```bash
docker build -t aibom-security .
sudo mkdir -p /srv/aibom/runs
sudo chown 999:999 /srv/aibom/runs   # image runs as uid 999 (nonroot)
docker run -d --name aibom-serve \
  -p 8080:8080 \
  -v /srv/aibom/runs:/runs \
  -e AIBOM_RUNS_DIR=/runs \
  -e HF_TOKEN="$HF_TOKEN" \
  -e AIBOM_API_TOKEN="$AIBOM_API_TOKEN" \
  --entrypoint aibom-serve \
  aibom-security --host 0.0.0.0 --port 8080
```

Mount the runs volume and read `runs/*/job.json`, `result.json`, and
`telemetry.jsonl` for the relevance analysis. The server checks at startup
that the runs dir is writable and refuses to boot otherwise, so the `chown`
above is required. Point `AIBOM_CACHE_DIR` at a volume too if you want a warm
artifact cache across container restarts.

## Configuration

| Flag | Env | Default | Purpose |
|---|---|---|---|
| `--host` | `AIBOM_BIND_HOST` | `127.0.0.1` | Bind address (use `0.0.0.0` in containers) |
| `--port` | `AIBOM_BIND_PORT` | `8080` | Bind port |
| `--runs-dir` | `AIBOM_RUNS_DIR` | `runs` | Per-run log directory; mount a volume here |
| `--concurrency` | — | `1` | Parallel verify subprocesses |
| `--queue-size` | — | `100` | Queued jobs before `POST /verify` answers `503` |
| `--timeout-secs` | — | `3600` | Kill a verify subprocess after this many seconds |
| `--token` | `AIBOM_API_TOKEN` | off | Require `Authorization: Bearer <token>` on `POST /verify` |

## Notes

- Workers are daemon threads, so a restart can kill jobs mid-flight; on
  startup, run dirs left `queued` or `running` by a previous process are
  marked `error` with `error: "interrupted by restart"` in their `job.json`.
- Sized for trickle load (default one worker); raise `--concurrency` if the
  pipeline pushes more.
- The verify step itself follows `docs/job-contract.md`; no new public JSON
  fields beyond the `run_id`/`status` ack.
- Issue #50 tracks this server.
