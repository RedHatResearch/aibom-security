# AGENTS.md

Agent-focused instructions for this repo. Human onboarding stays in [README.md](README.md).

## Project overview

`aibom-security` — passive, weight-level verification of LLM `base_model` provenance claims. Red Hat Research.

uv workspace monorepo: `verifier/` (`aibom_verifier`) + `cli/` (`aibom` umbrella CLI). Day-to-day dev uses `uv`; the Dockerfile is a packaging wrapper for colleagues, not the primary dev path.

When explaining why this work matters, lead with compliance: an AI BOM is only useful if lineage claims can be checked against the weights.

**Source of truth:** This file is the rulebook (how to work). [GitHub Issues and Milestones](https://github.com/RedHatResearch/aibom-security/milestones) are the board (what/why/status). Prefer `gh` against `RedHatResearch/aibom-security` (no GitHub MCP required).

## Repository layout

```text
aibom-security/
├── cli/              # thin `aibom` umbrella; verify impl in verifier/
├── verifier/         # aibom_verifier — see verifier/AGENTS.md
├── smokes/           # survey runnable checks (#26 / #17) — smokes/README.md
├── docker-compose.yml
└── pyproject.toml
```

## Setup and commands

```bash
uv sync --all-packages          # install everything
uv run aibom …                  # CLI (not on PATH by default)
uv run pytest -m "not network"  # unit tests, offline — what CI runs
uv run pytest -m network        # integration tests (HF Hub)
uv run ruff check .
uv run ruff format .
uv run ty check
```

Architecture PoC (local laptop): `docker compose up -d --scale worker=2`, host CLI with `.env` from `.env.example` (`AIBOM_STORE=proxy`, ports on `127.0.0.1`).

Survey smokes: `uv sync --all-packages --group smokes` then `uv run --group smokes python smokes/<topic>/smoke.py` — see [smokes/README.md](smokes/README.md), pairs in [smokes/fixtures.yaml](smokes/fixtures.yaml).

## Testing instructions

- Every feature ships with a happy-path test; every bug fix ships with a regression test (fails before, passes after).
- Run `uv run pytest -m "not network"` before push; fix failures first.
- Mark network tests with `@pytest.mark.network`; CI offline suite must stay green.

## Code style

- Match surrounding code: naming, imports at top of file, minimal scope diffs.
- Run `ruff check` and `ruff format` before push; `ty check` on changed Python.
- No inline imports unless a documented circular-import exception exists.
- Plain prose in user-facing docs and issue copy; avoid marketing or AI-flavored taglines.
- Keep README short; process guidance lives here (and [CLAUDE.md](CLAUDE.md) → `@AGENTS.md`).

## Security

- Never commit secrets, HF tokens, or credentials in code, issues, or PRs.
- Never add `Co-authored-by` or AI/agent attribution to commits or PRs.
- Issues, PRs, milestones, and wiki are **public**. See **Public board copy** under Board protocol.

## Workflow

| Always | Ask first | Never |
|--------|-----------|-------|
| Feature branch + PR for code; never push to `main` | Force-push, rewrite shared history, push to remote, commit | Commit secrets; agent co-authorship on git objects |
| Every open issue has a milestone; use feature/bug/research issue form templates | Pick up Icebox / `icelog` items | Put private email, DMs, or chat-only context on the board |
| Follow board protocol when touching issues | | Skip offline pytest / ruff / ty before push (code changes) |
| `gh` for board operations | | Invent public JSON field names without a grounded draft |

- Match process to the task — do not run plan → implement → review on every turn.
- When shipping code: prefer suitable skills (`verify-before-done`, `finishing-a-development-branch`, simplify / code-review); simplify, review deeply (not rubber-stamp), fix findings, verify before calling done.

**Agent definition of done (code):** `ruff check .`, `ruff format --check .` (or format), `ty check`, `uv run pytest -m "not network"` pass; scope matches the issue; no unrelated edits.

## Git and pull requests

**Branches:** `<type>/<N>-<short-kebab-slug>` — types: `feat`, `fix`, `chore`, `docs`, `test`, `refactor`.

Examples: `feat/5-resolve-refs`, `fix/12-shape-mismatch`, `chore/2-checklist-sync`.

**PR linking:**

| Stage | Issue keyword in body | PR milestone |
|-------|----------------------|--------------|
| Draft / WIP | `Refs #N` | Same as issue |
| Ready to merge (issue fully done) | `Fixes #N` / `Closes #N` | Same as issue |
| Multi-issue PR | `Fixes` only for issues fully done in that PR; `Refs` for partial | Primary issue’s milestone |

Set milestone on the **PR** (GitHub field), not only in prose. Before merge: switch `Refs #N` → `Fixes #N` when the issue is fully done. Ask before every commit and before opening/pushing a PR.

```bash
gh pr create --draft --milestone "…" --title "…" \
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

## Board protocol

Concise plans, decisions, blockers, and progress live on the issue — not chat-only memory. Research/no-code findings move to the [wiki](../../wiki) when done; the issue links the wiki page.

### Load context (before coding)

```bash
gh issue view N --comments
gh issue list --milestone "M1.2 — Pipeline telemetry & ops" --state open
```

Treat `Plan:` / `Decision:` / `Context:` / `Stuck:` / `Done:` comments as session history.

### Claim an issue

```bash
gh issue edit N --add-assignee @me \
  --remove-label "blocked" \
  --add-label "in-progress"
```

Open a **draft PR** early once there is a branch.

### Work and log

- **What I Got Stuck On** (issue body): durable blockers only.
- **Comments:** short bullets. Prefixes: `Plan:` · `Context:` · `Decision:` · `Stuck:` · `Done:`
- Session prompts, agent scaffolding, approve rituals, and “next human action” lists stay in **chat** — not on the issue.

```bash
gh issue comment N --body "Plan: …"
gh issue comment N --body "Context: …"
gh issue comment N --body "Decision: …"
gh issue comment N --body "Stuck: …"
gh issue comment N --body "Done: …"
```

When blocked:

```bash
gh issue edit N --remove-label "in-progress" --add-label "blocked"
```

Say why in Stuck On or a `Stuck:` comment (hard block) or a short `Context:` / `Decision:` comment (needs a human call).

### Board Mermaid and diagrams (issues / PRs / comments)

Use **Mermaid** (pick the **diagram type that fits the content** — do not default to `flowchart`), **tables**, or **math** when they clarify a **branch or contrast** prose would bury. Encouraged when it earns its keep — not decorative filler.

**Choose the type by what you are explaining:**

| What you are showing | Mermaid type (examples) |
|----------------------|-------------------------|
| Control flow, gates, pipelines, async handoff | `flowchart` / `graph` |
| Message or call order between actors (CLI → queue → worker) | `sequenceDiagram` |
| States and transitions (job/run/test lifecycle, outcomes) | `stateDiagram-v2` |
| Types, interfaces, plugin seams | `classDiagram` |
| Schema / store / metadata relationships | `erDiagram` |
| Component layout (services, pods, backends) | `block-beta` or `flowchart` with subgraphs |
| Timeline of milestones or phased rollout | `timeline` or `gantt` |
| Hierarchy of concepts or decision tree | `mindmap` or `flowchart` |
| Tradeoffs on two axes | `quadrantChart` |
| Simple share breakdown | `pie` (or a table if clearer) |
| Branch/merge or release sequencing | `gitGraph` |

If Mermaid is the wrong tool, use a **table** (signal matrices, coverage) or **math** (`$…$` / `$$…$$`) for formulas.

| Prefer | Avoid |
|--------|--------|
| The Mermaid type that matches the structure (see table above) | Always using `flowchart` when sequence/state/ER fits better |
| One small diagram for a non-obvious path or contrast | Mermaid that only restates a numbered list |
| Math for real formulas in research notes | geoJSON / topoJSON / STL (irrelevant here) |
| Tables for signal matrices and coverage | Large pasted screenshots when a fence would do |

GitHub-safe Mermaid: keep diagrams small; few nodes; avoid fancy Unicode (`≤`, `≫`) and reserved ids (`end`). Prefer editing the issue/comment in place when you authored it; otherwise paste a ready block for a human.

### Parent epics

Check off epic checklist items only when child work is actually complete.

### Ship (code issues)

- Commits while WIP: `Refs #N` (does not close).
- Draft PR: `Refs #N` in the body **and** PR milestone matches the issue.
- If merge used `Refs` only (issue stayed open): post `Done:`, clear `in-progress` / `blocked`, close manually.

### Closeout

| Kind | Done means | How to close |
|------|------------|--------------|
| Feature / bug | Tests land; PR merges | `Fixes #N` on merge, or close manually |
| Research / SOTA | Wiki promoted | `Done:` + wiki URL; close |
| Planning | Text accepted | `Decision:` or `Done:`; close |
| Blocked on human | Waiting on review | `blocked` until answered |

```bash
gh issue edit N --remove-label "in-progress" --remove-label "blocked"
gh issue close N --reason completed
```

### Parking (icelog)

```bash
gh issue edit N --add-label "icelog" \
  --remove-label "in-progress" --remove-label "blocked"
```

Also move to **Icebox** milestone. Do not implement without asking.

### Status labels

At most one status label on an open issue:

| Label | Meaning |
|-------|---------|
| `in-progress` | Assignee and/or draft PR |
| `blocked` | Cannot proceed, or waiting on human review/decision |

No status label = ready to pick up. Parked = `icelog` + Icebox. Type labels stay as today (`enhancement`, `bug`, `question`, …).

### Public board copy

Issues, PRs, and wiki pages are public. Write for any colleague or external reader.

| On the board | Off the board |
|--------------|---------------|
| Requirements, acceptance criteria, test plans | Private email, DM, or 1:1 quotes |
| `Decision:` / `Context:` with **public** links | “Per email from …”, named private messages |
| Neutral technical phrasing of stakeholder needs | Pasted mail threads or chat transcripts |

Translate private context into neutral technical language — not who said it or which channel.

**Scope authority:** Prefer the latest **agreed** written spec on the issue over older informal messages. If sources conflict, post a short `Decision:` — do not silently shrink scope.

## Package-specific guidance

- **[verifier/AGENTS.md](verifier/AGENTS.md)** — orchestrator, PoC stack, execution backends, proxy store, CLI verify path.
- **[smokes/README.md](smokes/README.md)** — fingerprint survey runnable checks. Issue deep-dives: link `smokes/<topic>/` when our smoke code exists.

## Learned preferences

- Unclear scope or missing schema: summarize status and ask clarifying questions before implementing.
- Prefer exposing existing verifier outputs over inventing new public JSON fields without a grounded draft.
- Do not mention personal machines, home-directory paths, or repos outside this workspace in shared docs.
- Design choices: use a table with tradeoffs vs the stated brief.
- Fit process to the task; do not run plan → implement → review on every turn.
- Issue/PR comments: short prefixed bullets only. No scaffolding or board-side approve rituals. Report hard blockers in chat first.
- On GitHub issues/PRs: pick the **Mermaid diagram type** that fits the content (flowchart, sequence, state, class, ER, block, timeline, …) or use tables — see **Board Mermaid and diagrams**.
