# ADR-0001: The research catalog lives in docs/sota/catalog.yaml

Status: accepted · Date: 2026-10-30 · Supersedes: —

## Context and Problem Statement

[#26](https://github.com/RedHatResearch/aibom-security/issues/26) reached 43 KB, 148 checkboxes,
and 55 comments while hosting four jobs with different lifetimes: a weekly-growing catalog, a
priority queue, a fixture table, and one-time write-ups. The body is one write target capped at
65,536 characters ([observed API error](https://github.com/clickhouse/clickhouse/issues/97567)),
so the catalog would outgrow it in months. One maintainer, agent-driven weekly sweeps, a
part-time reviewer, and a wiki finish line set the constraints.

## Decision Outcome

We moved the inventory into `docs/sota/catalog.yaml`, one lint-enforced YAML file, generated
`docs/sota/README.md` from it, and made #26 a one-page hub whose sub-issues are only scheduled work.

## Consequences

Good, because growth is append-only rows and sweeps are reviewable diffs a part-time mentor can read.
Good, because CI enforces the rules we applied by hand (verdict requires evidence; smoke paths exist; fixtures resolve).
Good, because the 54 deep-dive comments stayed on #26 untouched — evidence stays where it was produced.
Good, because the wiki page is rendered at publish time from the catalog, so no second copy ages in the repo.
Bad, because the catalog is not visible on the Issues tab; the rendered README and hub links replace it.
Bad, because a renderer and a lint must be maintained (~250 lines with tests).
Bad, because the pre-rewrite body and migration map now sit in `docs/sota/archive/`; delete them after 2027-10-30 or when a citable snapshot exists.

## Considered Options

- Harden #26 in place — rejected: the body regrows at ~1 KB per sweep; the cap arrives in months.
- Epic with topic-bucket sub-issues — rejected: the needs-inference bucket (49 of 133 rows) absorbs most growth and becomes the same monolith in a new location.
- One issue per method plus a Project board — rejected: ~100 open one-liner issues, and a board a solo maintainer will not keep in sync.
- Wiki as the working copy — rejected: no review path and no diff; the wiki stays a snapshot published from a clone, with no committed draft.

Revisit when: a second maintainer wants per-entry assignment, or the catalog passes ~300 rows and the single file gets slow to review.
