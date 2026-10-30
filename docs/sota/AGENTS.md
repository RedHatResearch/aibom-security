# AGENTS.md — docs/sota

The SOTA survey catalog lives here. This file overrides the root AGENTS.md for paths under `docs/sota/`.

## What lives where

| Content | Home |
|---|---|
| Catalog rows (methods, engines, tooling, field cases) | `catalog.yaml` — the only hand-edited list |
| Rendered views (by class, by verdict, next up) | `README.md` — generated, never hand-edited |
| Wiki snapshot | not in the repo: rendered at publish time into a `*.wiki.git` clone |
| Evidence (deep dives, verdicts, numbers) | issue #26 comments — linked from rows, never moved |
| Fixtures | `smokes/fixtures.yaml` and `provenancebench/pairs.yaml` — referenced by ID |
| Sweep records | `sweeps/YYYY-MM-DD-<slug>.md`, one file per sweep run |
| Pre-migration #26 body and migration map | `archive/` — read-only, see `archive/README.md` |

## Commands

```bash
# Validate the catalog before committing (local-first; ~1 s)
uv run --group docs python scripts/sota_lint.py
# Regenerate README.md, then verify no drift
uv run --group docs python scripts/sota_render.py
uv run --group docs python scripts/sota_render.py --check
# Wiki page: render into a wiki clone, never into this repo
uv run --group docs python scripts/sota_render.py --wiki ../aibom-security.wiki/SOTA-catalog.md
# Tests
uv run pytest scripts/tests/test_sota.py
```

## Adding an entry

1. Append a row to `catalog.yaml`. Required: `id`, `name`, `class`, `access`, `status`, `source`, `first_seen`.
2. Quote all strings; dates as `"2026-10-30"` (a bare date parses as a datetime and the lint rejects it).
3. Run the lint. Fix what it names; do not batch errors.
4. Run the renderer; commit `catalog.yaml` and `README.md` together.
5. Write `sweeps/YYYY-MM-DD-<slug>.md`, then post one `Context:` comment on [#26](https://github.com/RedHatResearch/aibom-security/issues/26) that links it.

## Sweep record

`sweeps/YYYY-MM-DD-<slug>.md`, one file per run, four headings fixed:

- `Channels:` where you looked, in order.
- `Queries:` the strings and filters that worked or failed.
- `Added:` catalog ids, one line each; negatives as `<id> — <why not>`.
- `Limits:` what the tooling could not reach (paywall, rate limit, no full text).

## Invariants (the lint enforces these)

- `id` is an immutable kebab-case slug; duplicate ids and duplicate normalized `source` URLs are rejected.
- `verdict` requires `findings` — a verdict with no linked deep-dive comment does not count.
- `smoke:` paths exist on disk and resolve from `README.md`; `fixtures:` IDs exist in the fixture registries.
- `name` carries at most one markdown link, and the renderer never nests a link inside a link.
- No `.scratch/` links, home paths, or private emails in any field.
- `notes` ≤ 300 chars, no `|` or backticks; numbers and comparisons live in findings comments.

## Field rules

- Identity fields (`id`, `name`, `class`, `source`) are immutable after creation; correct a mistake with `notes` or a new row, not a rename.
- Mutable fields: `status`, `verdict`, `priority`, `last_verified`, `notes`, `findings` (append-only).
- `status: reference` marks entries kept for citation only — they are not backlog.
- `confidence: lead` marks unverified claims; the renderer separates them.

## Decision tree

- New method/tool found → `catalog.yaml` row.
- Deep-dive evaluation → comment on #26, then link it from the row's `findings`.
- Scheduled work → an issue (sub-issue of #26 if it is survey closeout).
- Board-shape change → an ADR in `docs/decisions/`.
- Public synthesis → a wiki page: promote an issue writeup, or publish the catalog with `sota_render.py --wiki <clone>/SOTA-catalog.md`.

## Comments on #26

Append `Decision:` comments; never rewrite or delete a deep-dive comment. A correction opens with `Correction —` and links the fix.
