# aibom-security

Passive, weight-level verification of whether an LLM actually comes from the base model it claims. Instead of trusting a Hugging Face model card's `base_model` field, `aibom-security` inspects the weights directly and abstains rather than guessing when it can't tell.

An AI BOM is only useful if lineage claims can be checked against the weights.

A Red Hat Research project. Issues and milestones track what's being worked on; the [wiki](../../wiki) holds finished write-ups once an issue is closed (state of the art, standards research, design decisions).

## Current focus

See the [Milestone 1 board](../../milestone/1) for the active spec and requirements, and the [icebox](../../milestone/2) for deferred ideas.

## Quickstart (local)

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --all-packages
uv run aibom verify meta-llama/Llama-3.2-1B --base someorg/some-finetune
```

## Pipeline integration (Docker)

For a leaf step in an external verification pipeline:

```bash
docker build -t aibom-security .
docker run --rm aibom-security verify org/model --base org/base
```

For gated Hub repos, pass `HF_TOKEN`. **stdout** is the `VerificationResult` JSON; **stderr** is JSONL telemetry. Verdicts are not mapped to exit codes. Full contract (argv, env, accept mode, events): [docs/job-contract.md](docs/job-contract.md).

To run the verifier as a long-running REST service that an external pipeline (e.g. Tekton) can POST Hugging Face model IDs to, see [server/README.md](server/README.md).

## Repo layout

Monorepo — each top-level directory is an independently buildable component.

```
aibom-security/
├── cli/                 # the `aibom` umbrella command
├── verifier/            # aibom_verifier: the provenance verification pipeline
├── server/              # aibom-serve: REST verify server for pipeline submissions
├── smokes/              # survey runnable checks (#26); not product / not CI
├── docker-compose.yml   # local PoC stack only (see docs/poc-compose.md)
└── pyproject.toml       # uv workspace root
```

## Development

```bash
uv sync --all-packages
uv run pytest -m "not network"
uv run pytest -m network
uv run ruff check .
uv run ruff format .
uv run ty check
```

Contributing workflow and PR conventions: [AGENTS.md](AGENTS.md).

## More documentation

| Doc | For |
|-----|-----|
| [docs/job-contract.md](docs/job-contract.md) | Host pipeline integrators |
| [docs/poc-compose.md](docs/poc-compose.md) | Local Compose PoC (laptop only) |
| [verifier/AGENTS.md](verifier/AGENTS.md) | Verifier internals and backends |
| [smokes/README.md](smokes/README.md) | Fingerprint survey checks |
| [AGENTS.md](AGENTS.md) | Board protocol and agent workflow |
