# AGENTS.md

## Project

`aibom-security` — passive, weight-level verification of LLM `base_model` provenance claims. Red Hat Research.
uv workspace monorepo: `verifier/` (`aibom_verifier` library) + `cli/` (`aibom` umbrella CLI, depends on `verifier`).
Day-to-day development uses `uv`; the Dockerfile is for colleague/repro convenience, not the primary dev path.

When explaining why this work matters (docs, issues, pitch), lead with compliance: an AI BOM is only useful if its lineage claims can be checked against the weights.

## Source of truth

GitHub Issues and Milestones are the project whiteboard — for humans and agents. Concise plans, decisions, blockers, and progress live on the issue, not in chat-only memory or local scratch files.

- https://github.com/RedHatResearch/aibom-security/milestones
- This file is the rulebook (how to work). Issues are the board (what/why/status).
- Prefer `gh` for all board operations (no GitHub MCP required).
- Research/no-code findings move to the [wiki](../../wiki) when done; the issue stays as the discussion log and links to the wiki page.

## Board protocol

Use `gh` against `RedHatResearch/aibom-security`. Replace `N` with the issue number.

### 1. Load context (always before coding)

```bash
gh issue view N --comments
gh issue list --milestone "Milestone 1 — Passive Provenance Verifier (T1)" --state open
```

Read the body, labels, milestone, and comments. Treat `Plan:` / `Decision:` / `Context:` / `Stuck:` / `Done:` comments as the session history colleagues will also read.

### 2. Claim the issue

```bash
gh issue edit N --add-assignee @me \
  --remove-label "blocked" \
  --add-label "in-progress"
```

Open a **draft PR** early once there is a branch. That is the in-progress signal on GitHub.

**PR linking (required for agents):**

| Stage | Issue keyword in body | PR milestone |
|---|---|---|
| Draft / WIP | `Refs #N` (does not close) | Same milestone as the issue |
| Ready to merge (issue fully done in this PR) | `Fixes #N` / `Closes #N` | Keep the same milestone |
| Multi-issue PR | `Fixes` only for issues this PR fully completes; `Refs` for partial | Primary issue’s milestone |

Set the milestone on the **PR** (GitHub milestone field), not only in prose. Prefer `gh`:

```bash
gh pr create --draft \
  --milestone "Milestone …" \
  --title "…" \
  --body "$(cat <<'EOF'
## What & why
Refs #N

## Test plan
…

## Out of scope / follow-ups
…
EOF
)"
```

Before marking ready / merge: switch `Refs #N` → `Fixes #N` when the issue is fully done in that PR; confirm the PR milestone still matches the issue.

### 3. Work and log (shared with colleagues)

- **What I Got Stuck On** (issue body): durable blockers only — what is blocking and what would unblock it. Not a daily diary.
- **Comments** (short colleague-visible chronology): a few sentences or tight bullets. Prefer one `Decision:` / `Plan:` over essay threads. Long design stays rare on the board. Session prompts, agent scaffolding, approve rituals, and “next human action” lists stay in chat — not on the issue.

```bash
gh issue comment N --body "Plan: …"
gh issue comment N --body "Context: …"      # research links, notes colleagues need
gh issue comment N --body "Decision: …"     # choices and why
gh issue comment N --body "Stuck: …"        # mirror/update the Stuck On field when blocked
gh issue comment N --body "Done: …"         # what shipped / how to verify
```

When blocked or waiting on a human decision:

```bash
gh issue edit N --remove-label "in-progress" --add-label "blocked"
```

Say why in Stuck On / a `Stuck:` comment (hard block) or a short `Context:` / `Decision:` comment (needs a human call).

### Board visuals (issues / PR bodies / comments)

Use visuals when they clarify a **branch or contrast** that prose or a table would bury (gates, access taxonomy, rejector vs confirmer, two mechanisms). Skip decorative diagrams and do not backfill every deep-dive with the same load→score flowchart.

| Prefer | Avoid |
|---|---|
| One small Mermaid flowchart for a non-obvious control path | Mermaid that only restates a numbered list |
| Math (`$…$` / `$$…$$`) for real formulas in research notes | geoJSON / topoJSON / STL (irrelevant here) |
| Tables for signal matrices and coverage | Large pasted screenshots when a fence would do |

Keep Mermaid GitHub-safe: simple `flowchart`/`graph`, few nodes, avoid fancy Unicode (`≤`, `≫`) and reserved ids (`end`). Prefer editing the issue/comment in place when you authored it; otherwise paste a ready block for a human.

### 4. Parent dashboard

If a parent/epic issue has a checklist (e.g. MVP requirements), check off the item when that child work is actually complete — don't check early.

### 5. Ship (code issues)

- Branch: `<type>/<N>-<short-slug>` (see below).
- Commits while WIP: `Refs #N` (does not close).
- Draft PR: `Refs #N` in the body **and** set the PR milestone to match the issue (see PR linking under Claim).
- PR ready to merge: `Fixes #N` / `Closes #N` only when that issue is fully done in the PR; keep the milestone.
- Ask before every commit and before opening/pushing a PR.
- If merge used `Refs` only (issue stayed open): post `Done:`, clear `in-progress` / `blocked`, and close the issue manually.

### 6. Closeout — by issue kind

| Kind | Done means | How to close |
|---|---|---|
| Feature / bug (code) | Tests land; PR merges | `Fixes #N` on merge, or close manually after merge |
| Research / SOTA | Findings promoted to wiki; issue links the page | `Done:` comment + wiki URL in body; close |
| Planning / pitch / requirements | Text accepted; no code required | `Done:` or `Decision:` comment; close |
| Needs a human call first | Waiting on review | `blocked` until answered, then resume or close |

```bash
gh issue edit N --remove-label "in-progress" --remove-label "blocked"
gh issue close N --reason completed
```

### 7. Parking (icelog)

```bash
gh issue edit N --add-label "icelog" \
  --remove-label "in-progress" --remove-label "blocked"
```

Also put the issue on the **Icebox** milestone. Do not implement parked issues without asking first.

## Status labels

At most one of these on an open issue:

| Label | Meaning |
|---|---|
| `in-progress` | Assignee and/or draft PR; actively working |
| `blocked` | Cannot proceed, or waiting on human review/decision |

No status label on an open issue = ready to pick up. Closed = done. Parked = `icelog` + Icebox.

Type labels stay as today (`enhancement`, `bug`, `question`, `icelog`, …).

## Branch names

De facto pattern (Conventional Branch + issue id):

```text
<type>/<N>-<short-kebab-slug>
```

Examples: `feat/5-resolve-refs`, `fix/12-shape-mismatch`, `chore/2-checklist-sync`, `docs/1-elevator-pitch`.

Types: `feat`, `fix`, `chore`, `docs`, `test`, `refactor`. Lowercase, hyphens, include the issue number when the work is tracked.

## Commands

```bash
uv sync --all-packages          # install everything
uv run pytest -m "not network"  # unit tests, offline, what CI runs
uv run pytest -m network        # integration tests that hit the HF Hub
uv run ruff check .             # lint
uv run ruff format .            # format
uv run ty check                 # type check
```

## Smokes (#26 / #17 survey)

`smokes/<topic>/` holds **our** runnable checks for the fingerprint survey — reimplementations and probes, not the product verifier. Index: [`smokes/README.md`](smokes/README.md). Shared Hub pairs: [`smokes/fixtures.yaml`](smokes/fixtures.yaml).

| Put here | Put in `.scratch/` | Put in `verifier/` |
|---|---|---|
| Our smoke scripts + topic README | Clones of upstream tools (GhostSpec, HuRef repo, MPK, …) | Shipped tests and verify nodes |

```bash
uv sync --all-packages
uv run --group smokes python smokes/<topic>/smoke.py
```

- Not CI; needs HF network and multi-GB downloads for dense models.
- Each topic README links paper, #26 (or #17) note, and run command.
- When a smoke graduates into the verifier, move logic to `verifier/` and keep the smoke as a thin wrapper or delete it.
- Issue deep-dive **Fixture / smoke** sections should link `smokes/<topic>/` when our code exists.

## Workflow (always / ask / never)

- Always: feature branch + PR for code. Never push to `main` directly.
- Always: every issue has a milestone; use the feature/bug/research issue form templates.
- Always: features ship with a happy-path test; bug fixes ship with a regression test (fails before, passes after).
- Always: follow the board protocol above when touching issues.
- Always before pushing code: run `uv run ruff check .`, `uv run ruff format --check .` (or format), `uv run ty check`, and `uv run pytest -m "not network"` locally; fix failures first.
- Match process to the task. Prefer a suitable Cursor skill (e.g. writing-plans, finishing-a-development-branch, simplify / code-review when shipping code, verify-before-done) over a fixed ritual every turn. When code changed: simplify, review deeply (not rubber-stamp), fix findings, and verify before calling done.
- Ask first: force-pushing, rewriting shared history, picking up `Icebox` / `icelog` items, pushing to remote, committing.
- Never: commit secrets/HF tokens, add `Co-authored-by` or any AI/agent attribution to commits or PRs.
- Keep the README short and human; put agent/process guidance here (and in `CLAUDE.md`), not in the README.
- Prefer plain, non-marketing prose in user-facing docs and GitHub issue copy; avoid AI-flavored taglines.

## Learned User Preferences

- When an issue has unclear scope or missing schema/docs, summarize current status and ask clarifying questions before implementing.
- Prefer documenting or exposing existing verifier outputs over inventing new public JSON field names without a grounded draft.
- Keep this file usable by everyone on the repo: do not mention personal machines, home-directory paths, or other local repositories outside this workspace.
- When presenting design choices, use a table with tradeoffs and how close each option is to the stated brief.
- Fit process to the task (see Workflow skills bullet); do not run plan → implement → review on every turn.
- Issue/PR comments: short Plan/Decision/Context/Stuck/Done only. No scaffolding, session prompts, or board-side approve / next-action rituals. Report blockers in chat first.
- Board Mermaid only when it earns its keep (branch/contrast); see Board visuals.

## Learned Workspace Facts

- Run the CLI as `uv run aibom ...`; bare `aibom` is not on PATH by default in this uv workspace.
- Milestone 1 includes the T1 verifier CLI plus an architecture-validation PoC (Compose stack, Postgres+MinIO result proxy, Compose worker replicas and SSH-to-localhost demos, 30-day cache sweeper). The stack may be throwaway later; prefer working prototype over polish. OpenShift AI / MetaCentrum stay documented upgrade paths on the same ExecutionBackend interface, not M1 demos.
- PoC orchestration (proxy store, backends, Compose-related code) lives under `verifier/`; root `docker-compose.yml` may still sit at the repo root.
- PoC entry is CLI-first: `aibom verify` uses the proxy and can submit work to Compose/SSH backends; cache cleanup is one `aibom cache-sweep` path (`--max-age-days 30`) also run by a Compose timer.
- SSH-to-localhost demo uses a thin SSH backend (`ssh localhost` → `run-test` entrypoint); no Dask in M1.
- Compose job queue is plain Redis list + JSON (`LPUSH`/`BRPOP`); no RQ.
- Block-0 / first-layer checks use safetensors headers plus ranged tensor byte reads, not full model downloads; the two live detection tests stay boolean (float-threshold gating is an orchestrator seam for later tests).
- Survey runnable checks live under `smokes/<topic>/` (see **Smokes** section). External tool clones: `.scratch/<tool>/`.
- Test dependencies are a small YAML/JSON rule list with forward walk; live T1→T2 gates on boolean pass; float `> 0.8` is proven on a stub only.
- FR-8 inventing new lineage JSON fields is abandoned; current `VerificationResult` / `RunResult` output is enough for M1.
- Reference smoke pair: `SultanR/SmolTulu-1.7b-Instruct` vs `HuggingFaceTB/SmolLM2-1.7B`.
- `arch_hash` normalizes Transformers absent-vs-default config fields (for example `head_dim`, `mlp_bias`) before hashing.
