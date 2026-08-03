# AGENTS.md

## Project

`aibom-security` — passive, weight-level verification of LLM `base_model` provenance claims. Red Hat Research.
uv workspace monorepo: `verifier/` (`aibom_verifier` library) + `cli/` (`aibom` umbrella CLI, depends on `verifier`).
Day-to-day development uses `uv`; the Dockerfile is for colleague/repro convenience, not the primary dev path.

When explaining why this work matters (docs, issues, pitch), lead with compliance: an AI BOM is only useful if its lineage claims can be checked against the weights.

## Source of truth

GitHub Issues and Milestones are the project whiteboard — for humans and agents. Plans, decisions, blockers, and progress live on the issue, not in chat-only memory or local scratch files.

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
  --remove-label "status:ready" --remove-label "status:parked" \
  --add-label "status:in-progress"
```

Open a **draft PR** early once there is a branch (link the issue with `Refs #N` in the PR body). That is the in-progress signal on GitHub.

### 3. Work and log (shared with colleagues)

- **What I Got Stuck On** (issue body): durable blockers only — what is blocking and what would unblock it. Not a daily diary.
- **Comments** (chronology everyone can scan):

```bash
gh issue comment N --body "Plan: …"
gh issue comment N --body "Context: …"      # research links, notes colleagues need
gh issue comment N --body "Decision: …"     # choices and why
gh issue comment N --body "Stuck: …"        # mirror/update the Stuck On field when blocked
gh issue comment N --body "Done: …"         # what shipped / how to verify
```

When blocked:

```bash
gh issue edit N --remove-label "status:in-progress" --add-label "status:blocked"
```

When waiting on a human decision/review (no code merge yet):

```bash
gh issue edit N --remove-label "status:in-progress" --add-label "status:needs-review"
```

### 4. Parent dashboard

If a parent/epic issue has a checklist (e.g. MVP requirements), check off the item when that child work is actually complete — don't check early.

### 5. Ship (code issues)

- Branch: `<type>/<N>-<short-slug>` (see below).
- Commits while WIP: `Refs #N` (does not close).
- PR ready to merge: `Fixes #N` / `Closes #N` only when that issue is fully done in the PR.
- Ask before every commit and before opening/pushing a PR.

### 6. Closeout — by issue kind

| Kind | Done means | How to close |
|---|---|---|
| Feature / bug (code) | Tests land; PR merges | `Fixes #N` on merge, or close manually after merge; set `status:done` |
| Research / SOTA | Findings promoted to wiki; issue links the page | `Done:` comment + wiki URL in body; `status:done`; close |
| Planning / pitch / requirements | Text accepted; no code required | `Done:` or `Decision:` comment; `status:done`; close |
| Needs a human call first | Waiting on review | `status:needs-review` until answered, then resume or close |

```bash
gh issue edit N --remove-label "status:in-progress" --remove-label "status:blocked" \
  --remove-label "status:needs-review" --add-label "status:done"
gh issue close N --reason completed
```

### 7. Parking (icelog)

```bash
gh issue edit N --add-label "icelog" --add-label "status:parked" \
  --remove-label "status:in-progress" --remove-label "status:ready"
```

Also put the issue on the **Icebox** milestone. Do not implement parked issues without asking first.

## Status labels

Exactly one `status:*` on active work when practical:

| Label | Meaning |
|---|---|
| `status:ready` | Triaged, available to pick up |
| `status:in-progress` | Assignee and/or draft PR; actively working |
| `status:blocked` | Cannot proceed; Stuck On + `Stuck:` say why |
| `status:needs-review` | Waiting on human review or a decision |
| `status:done` | Complete; issue should be closed |
| `status:parked` | Frozen/deferred (`icelog` + Icebox) |

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

## Workflow (always / ask / never)

- Always: feature branch + PR for code. Never push to `main` directly.
- Always: every issue has a milestone; use the feature/bug/research issue form templates.
- Always: features ship with a happy-path test; bug fixes ship with a regression test (fails before, passes after).
- Always: follow the board protocol above when touching issues.
- Ask first: force-pushing, rewriting shared history, picking up `Icebox` / `status:parked` items, pushing to remote, committing.
- Never: commit secrets/HF tokens, add `Co-authored-by` or any AI/agent attribution to commits or PRs.
- Keep the README short and human; put agent/process guidance here (and in `CLAUDE.md`), not in the README.
- Prefer plain, non-marketing prose in user-facing docs and GitHub issue copy; avoid AI-flavored taglines.
