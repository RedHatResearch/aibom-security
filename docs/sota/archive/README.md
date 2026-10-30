# SOTA archive

Frozen inputs behind `docs/sota/catalog.yaml`. Treat as read-only evidence.

| File | Contents |
|---|---|
| `2026-10-30-issue-26-body.md` | Issue #26 body, verbatim: the checklist rows the catalog was migrated from. |
| `2026-10-30-migration-map.json` | Per source line: id, class, action (`row`, `duplicate_of`, `alias_of`), and the raw checklist text. |

We fetched the body on 2026-10-30 with `gh api repos/RedHatResearch/aibom-security/issues/26 --jq .body`
and generated the map from that snapshot the same day. Comments stay live on #26; nothing mirrors them here.
Delete after 2027-10-30, or earlier once a citable snapshot exists.
