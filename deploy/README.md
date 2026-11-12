# Deploying the aibom verify REST service

`server/` (`aibom-serve`) wraps `aibom verify` in one long-running HTTP
service: `POST /verify {"model_id":"<org>/<model>"}` queues the run and answers
`202 {"run_id","status":"queued"}`; no endpoint returns a verdict. Each run
writes `job.json`, `telemetry.jsonl` and `result.json` under the runs
directory. `provision.sh` creates the resources on a private COS VM (Cloud NAT
egress, tcp/8080 open to the caller's range only, tcp/22 over IAP),
`startup.sh` starts the container on boot, `deploy/tekton/` holds the Task.

## Prerequisites

- A GCP project that can hold a VPC, subnet, Cloud Router and NAT, firewall
  rules, a service account, an Artifact Registry repository and one VM.
- A route from the calling cluster to the VM's subnet: Shared VPC attachment,
  peering or IPsec tunnel. A firewall rule alone does not make it reachable.
- `gcloud`, authenticated against that project; Docker for image builds.

## Configuration

| Item | Value (from `server/src/aibom_server/`) |
|---|---|
| Port / bind | `8080`; `127.0.0.1` unless `--host 0.0.0.0` (the unit passes it) |
| `POST /verify` | `202` queued; `401` bad token; `413` body over 4 KiB; `422` malformed id; `503` queue full |
| Env | `AIBOM_API_TOKEN` (set-but-empty aborts startup; unset disables auth) and `AIBOM_RUNS_DIR` (default `runs`); `AIBOM_CACHE_DIR`, `HF_TOKEN`, `AIBOM_LOG_LEVEL`, `AIBOM_LOG_FILE` pass through to verify |
| Must stay unset | `AIBOM_STORE`; `proxy` needs Postgres, MinIO and Redis |

Deployment values come from `deploy/gce/site.env` (git-ignored; copy from
`site.env.example`): `PROJECT` and `CLIENT_CIDR` required, the rest defaulted in
`provision.sh`.

## Deploy

```bash
cp deploy/gce/site.env.example deploy/gce/site.env   # fill PROJECT, CLIENT_CIDR
PUSH_IMAGE=1 ./deploy/gce/provision.sh
```

Re-runs are safe (describe-then-create); an existing VM only gets its
`aibom-image` metadata refreshed, and `PUSH_IMAGE=1` builds for `linux/amd64`
and pushes. `startup.sh` writes the token into `serve.env` (0600, root), writes
the unit, then starts it. The tag lives in instance metadata:

```bash
gcloud compute instances add-metadata aibom-serve --zone=us-east1-b \
  --metadata=aibom-image=<region>-docker.pkg.dev/<project>/aibom/aibom-security:<tag>
gcloud compute instances reset aibom-serve --zone=us-east1-b
```

## Run artifacts

| File | Contents |
|---|---|
| `job.json` | `model_id`, `status` (`queued` → `running` → `done`/`error`), `exit_code`, timestamps |
| `telemetry.jsonl` | verify stderr, one JSON object per event, all with the same `run_id` |
| `result.json` | verify stdout: `VerificationResult` only — `target`, `base`, `verdict`, `message` |

A run that could not start writes `"accept": true` and an
`insufficient_evidence` verdict, exiting 0, so the target still leaves a record.
Per-test detail is not persisted: `tests[]`, reason codes and the architecture
config never reach stdout (`cli.py` prints `VerificationResult`); a
MoE-versus-dense split therefore needs a `config.json` fetch per model
(`model_type`, `architectures`, `num_experts` / `num_local_experts`).

## Operations

```bash
gcloud compute ssh aibom-serve --zone=us-east1-b --tunnel-through-iap \
  --command 'curl -sS http://127.0.0.1:8080/healthz'      # health, queue depth
mkdir -p ~/aibom-runs                                     # pull the run archive
gcloud compute scp --recurse --tunnel-through-iap --zone=us-east1-b \
  aibom-serve:/mnt/stateful_partition/aibom/runs/* ~/aibom-runs/
```

The token sits in `serve.env` (0600, root); read it over IAP SSH and store it
as a Secret in the pipeline namespace (`kubectl -n <pipeline-namespace> create
secret generic aibom-api-token --from-literal=token=<value>`). Rotate by
deleting `serve.env` and rebooting, then updating the Secret; keep it out of
instance metadata and git.

A manual `docker pull` over SSH is unauthenticated: the credential helper runs
under the unit's `HOME=/var/lib/aibom`, so let systemd pull or run
`sudo HOME=/var/lib/aibom docker-credential-gcr configure-docker
--registries $REGION-docker.pkg.dev` first.

## Tekton wiring

`deploy/tekton/aibom-provenance-probe.yaml` is a standalone Task: one step, no
workspaces, three results (`ack`, `http-code`, `run-id`), exit 0 on every
reachable outcome unless `fail-on-error` is `"true"`. `endpoint` has no default.

```yaml
    - name: aibom-provenance-probe
      runAfter:
        - <task that produces the model id>
      onError: continue        # the run continues even if this pod never starts
      timeout: 2m
      taskRef:
        resolver: git
        params:
          - { name: url,        value: https://github.com/RedHatResearch/aibom-security }
          - { name: revision,   value: <commit-sha> }
          - { name: pathInRepo, value: deploy/tekton/aibom-provenance-probe.yaml }
      params:
        - { name: model-id, value: "$(tasks.<task>.results.<result>)" }
        - { name: endpoint, value: "http://<vm-private-ip>:8080" }
```

Use `onError: continue`, not a `finally` task: `finally` accepts no `onError`,
and a failure there fails the PipelineRun. No Tekton `retries` — mutually
exclusive with `onError: continue`; the `503` retry loop runs in the step.

## Caveats

- `/var` is `noexec` on COS, so scripts there need an interpreter: the health
  gate is `ExecStartPost=/bin/bash /var/lib/aibom/healthz`. The image runs as
  uid 999; `startup.sh` chowns `runs/` and `cache/` each boot.
- Build with `--platform=linux/amd64 --provenance=false --sbom=false`
  (`provision.sh` passes these; a manual build must too). Docker
  Desktop's buildx otherwise adds an attestation entry (`unknown/unknown`) that
  COS's dockerd resolves instead of the image, leaving an empty `/etc/passwd`
  and `unable to find user nonroot`; the platform flag is needed on arm64 hosts.
- `-p 8080:8080` publishes past the host firewall, so the VPC rule is the only
  access control. No TLS; traffic stays on the private subnet.
- Single node, no HA: a reboot kills in-flight verifies (`status: "error"`,
  `error: "interrupted by restart"`), and state is on the boot disk.
