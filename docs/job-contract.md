# Host pipeline job contract

This page is the integration contract for running the **Docker image** as a leaf step in an external verification pipeline. The image entrypoint is `aibom` -- the same CLI as `uv run aibom` in development.

This tool does not ship a queue product or host payload adapter.

## Primary invocation

```bash
docker build -t aibom-security .
docker run --rm aibom-security verify <target> [options]
```

With Hugging Face token for gated models:

```bash
docker run --rm -e HF_TOKEN="$HF_TOKEN" aibom-security verify org/model --base org/base
```

With pipeline telemetry and optional accept mode:

```bash
docker run --rm \
  -e HF_TOKEN="$HF_TOKEN" \
  -e AIBOM_RUN_ID="pipeline-run-42" \
  -e AIBOM_ACCEPT=1 \
  aibom-security verify org/model --base org/base
```

Collect **container stderr** for JSONL telemetry. Mount a volume only if you use `AIBOM_LOG_FILE` inside the container.

## Command argv

Subcommand: `verify`


| Argument            | Required | Description                                                                             |
| ------------------- | -------- | --------------------------------------------------------------------------------------- |
| `target`            | yes      | Hugging Face repo id of the model being verified                                        |
| `--base`            | no       | Claimed base repo id; defaults to the target model card `base_model`                    |
| `--revision-target` | no       | Pin target to a commit SHA or revision                                                  |
| `--revision-base`   | no       | Pin base to a commit SHA or revision                                                    |
| `--store`           | no       | `filesystem` (default) or `proxy` (Postgres + MinIO)                                    |
| `--cache-dir`       | no       | Local filesystem cache when `--store filesystem`                                        |
| `--ignore-cache`    | no       | Skip cache reads; still writes new artifacts                                            |
| `--backend`         | no       | `local` (default), `ssh`, or `compose` (Redis workers)                                  |
| `--accept`          | no       | On start/config failure, exit 0 with non-confirming stdout (alias for `AIBOM_ACCEPT` truthy values) |


Run `docker run --rm aibom-security verify --help` for the live flag list.

## Environment variables


| Variable          | When                    | Purpose                                                               |
| ----------------- | ----------------------- | --------------------------------------------------------------------- |
| `HF_TOKEN`        | Gated/private Hub repos | Hugging Face API token                                                |
| `AIBOM_RUN_ID`    | Optional                | Correlation id on every JSONL line; auto-minted if unset              |
| `AIBOM_LOG_FILE`  | Optional                | Tee stderr JSONL to a file (best-effort; failures do not fail verify) |
| `AIBOM_LOG_LEVEL` | Optional                | `DEBUG`, `INFO` (default), `WARNING`, `ERROR`                         |
| `AIBOM_ACCEPT`    | Optional                | `1` / `true` / `yes` — accept mode (see below)                        |
| `AIBOM_STORE`     | Optional                | Default store kind when `--store` is omitted                          |
| `AIBOM_CACHE_DIR` | Optional                | Default cache directory for filesystem store                          |
| `AIBOM_PG_DSN`    | Proxy store             | Postgres DSN                                                          |
| `AIBOM_MINIO_*`   | Proxy store             | MinIO endpoint, credentials, bucket, secure flag                      |
| `AIBOM_REDIS_URL` | `--backend compose`     | Redis URL for worker queue                                            |




## stdout vs stderr


| Stream     | Content                                                                                                                                      |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **stdout** | `VerificationResult` JSON on completed compare; `{ok: false, error, message}` on start/config failure (default); or non-confirming `VerificationResult` with `"accept": true` when accept mode is on |
| **stderr** | JSONL telemetry — one JSON object per line. Primary signal for pipeline observability.                                                       |


Every stderr line includes envelope fields: `ts`, `level`, `logger`, `event`, `run_id`, `policy_version`, `tool_version`, plus event-specific fields.

Log sink failures never fail verify.

## Exit codes


| Code  | Meaning                                                                                                       |
| ----- | ------------------------------------------------------------------------------------------------------------- |
| **0** | Compare completed (any verdict, including non-confirming), **or** start/config failure when accept mode is on |
| **1** | Start/config error (resolve failed, gated repo, bad store config, invalid log level, …)                       |


Verdicts are **not** mapped to exit codes. A `fraudulent_claim` or `insufficient_evidence` verdict still exits 0 when the compare finished.

### Accept mode (`AIBOM_ACCEPT=1` or `--accept`)

For container jobs that must not fail the host step on start/config errors:

- stderr still emits `resolve_failed` or `run_failed` with the real error
- process exits **0**
- stdout is a non-confirming `VerificationResult` (`insufficient_evidence`), never `verified_derivative`
- stdout may include `"accept": true`

Default image/CLI behavior (exit 1 on start errors) is unchanged when accept mode is off.

## JSONL events

Typical successful run sequence:

```text
run_start → resolve_ok → test_started → test_finished → … → run_finished
```

Failure paths:


| Event            | When                                                                               |
| ---------------- | ---------------------------------------------------------------------------------- |
| `run_failed`     | Invalid log level, invalid store config, or other CLI pre-start failure            |
| `resolve_failed` | Cannot pin or resolve target/base (gated, missing card field, repo not found, …)   |
| `test_skipped`   | Gate skipped a downstream test                                                     |
| `exception`      | In-process node raised (local backend)                                             |
| `run_finished`   | Compare completed; includes `verdict`, `exit_code`, `tests_summary`, `duration_ms` |


`resolve_ok` carries pinned `target` and `base` objects (repo id, revision, sha; base includes `source`). `test_finished` includes `test_id`, `status`, `reason_codes`, and `duration_ms`.

## stdout shapes

**Completed compare** (`VerificationResult`):

```json
{
  "target": "org/model",
  "base": "org/base",
  "verdict": "verified_derivative",
  "message": "…"
}
```

**Start/config failure (default)**:

```json
{
  "ok": false,
  "error": "gated_unauthenticated",
  "message": "…",
  "policy_version": "t1-1"
}
```

**Start/config failure (accept mode)**:

```json
{
  "target": "org/model",
  "base": "org/base",
  "verdict": "insufficient_evidence",
  "message": "Insufficient evidence to confirm or refute the claimed base relationship.",
  "accept": true
}
```

When `--base` was not passed, `base` may be `null` because resolve never completed.

## References

- Structured logging: GitHub issue #42
- Redis Compose PoC (internal fan-out): root `docker-compose.yml`, `verifier/AGENTS.md`

