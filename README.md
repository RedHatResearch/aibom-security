# aibom-security

Passive, weight-level verification of whether an LLM actually comes from the base model it claims. Instead of trusting a Hugging Face model card's `base_model` field, `aibom-security` inspects the weights directly and abstains rather than guessing when it can't tell.

A Red Hat Research project. Issues and milestones track what's being worked on; the [wiki](../../wiki) holds finished write-ups once an issue is closed (state of the art, standards research, design decisions).

## Current focus

See the [Milestone 1 board](../../milestone/1) for the active spec and requirements, and the [icebox](../../milestone/2) for deferred ideas.

## Repo layout

Monorepo — each top-level directory is an independently buildable component.

```
aibom-security/
├── cli/         # the `aibom` umbrella command
├── verifier/    # aibom_verifier: the provenance verification pipeline
└── pyproject.toml   # uv workspace root
```

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --all-packages
uv run aibom verify meta-llama/Llama-3.2-1B --base someorg/some-finetune
```

Or via Docker:

```bash
docker build -t aibom-security .
docker run --rm aibom-security verify meta-llama/Llama-3.2-1B --base someorg/some-finetune
```

Architecture PoC stack (Postgres + MinIO + Redis + workers + sweeper).
Local laptop only; published ports bind to `127.0.0.1`. Data is ephemeral
(no named volumes). Container services use Docker DNS; host CLI uses
`.env.example` (`localhost` published ports).

```bash
docker compose up -d --scale worker=2
cp .env.example .env   # local-only defaults; do not commit secrets
set -a && source .env && set +a   # required; copy alone does not set env
uv run aibom verify org/model --base org/base --store proxy --backend compose
```

## Development

```bash
uv sync --all-packages          # install everything
uv run pytest -m "not network"  # unit tests (offline)
uv run pytest -m network        # integration tests that hit the HF Hub
uv run ruff check .             # lint
uv run ruff format .            # format
uv run ty check                 # type check
```
