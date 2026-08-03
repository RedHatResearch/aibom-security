# AGENTS.md

## Project

`aibom-security` — passive, weight-level verification of LLM `base_model` provenance claims. Red Hat Research.
uv workspace monorepo: `verifier/` (`aibom_verifier` library) + `cli/` (`aibom` umbrella CLI, depends on `verifier`).
Day-to-day development uses `uv`; the Dockerfile is for colleague/repro convenience, not the primary dev path.

When explaining why this work matters (docs, issues, pitch), lead with compliance: an AI BOM is only useful if its lineage claims can be checked against the weights.

## Source of truth

GitHub Issues and Milestones — not this file — define current scope and spec.

- https://github.com/RedHatResearch/aibom-security/milestones
- Read the linked issue in full before starting work on it.
- Update the issue's "What I Got Stuck On" field as you go, not just at the end.
- Once an issue is closed, research/no-code write-ups move to the [wiki](../../wiki).

## Commands

```bash
uv sync --all-packages          # install everything
uv run pytest -m "not network"  # unit tests, offline, what CI runs
uv run pytest -m network        # integration tests that hit the HF Hub
uv run ruff check .             # lint
uv run ruff format .            # format
uv run ty check                 # type check
```

## Workflow

- Always: feature branch + PR. Never push to `main` directly.
- Always: every issue has a milestone; use the feature/bug/research issue form templates.
- Always: features ship with a happy-path test; bug fixes ship with a regression test (fails before, passes after).
- Ideas parked for later get the `icelog` label + `Icebox` milestone — frozen, don't act on them without asking first.
- Ask first: force-pushing, rewriting shared history, picking up `Icebox` items, pushing to remote.
- Never: commit secrets/HF tokens, add `Co-authored-by` or any AI/agent attribution to commits or PRs.
- Do not push to remote unless explicitly asked; commit only when requested.
- Keep the README short and human; put agent/process guidance here (and in `CLAUDE.md`), not in the README.
- Prefer plain, non-marketing prose in user-facing docs and GitHub issue copy; avoid AI-flavored taglines.
