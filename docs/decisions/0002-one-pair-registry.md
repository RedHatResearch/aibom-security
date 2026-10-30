# ADR-0002: One pair registry, cohorts as the scoring boundary

Status: accepted · Date: 2026-10-30 · Supersedes: — · Related: ADR-0001

## Context and Problem Statement

Model pairs lived in five places: `provenancebench/pairs.yaml` (42 pinned, evidenced rows),
`smokes/fixtures.yaml` (3+3 unpinned role-named pairs), per-method `config.py` matrices,
hardcoded pairs in two verifier integration tests, and a deleted prose table in #26. Role
names meant different pairs in different files; five smokes fetched Hub `main` at run time,
so their results were not reproducible; TensorLock cited two pbench ids that a re-baseline
(commit `9ea79c5`) had removed; the catalog lint accepted ambiguous bare repo ids and
rejected the role names it was documented to accept.

Four needs drove the multiplicity: a scored benchmark, survey scratch pairs, method
expectations, and constructed fixtures. Encoding those as file locations instead of
separated concerns is what produced the drift.

## Decision Outcome

We make `provenancebench/pairs.yaml` the single pair registry, and `provenancebench/cohorts.yaml`
the scored-vs-stored boundary: survey pairs live in a `smoke` cohort and never enter a scored
cohort, so adding registry rows is cohort-neutral and benchmark numbers do not move.

## Consequences

Good, because one identity space (`pb:<id>`) replaces role names, bare repo ids, and copied
config pairs; every citation resolves or fails loudly.
Good, because all smoke runs become pinned and reproducible; no smoke fetches Hub `main`.
Good, because method expectations (TensorLock gates, verifier verdicts) stay with the method
that asserts them, and constructed fixtures declare themselves as synthetic — the correct
ownership rather than accidental locality.
Bad, because survey pairs with no scoring future occupy registry rows; rent is ~6 YAML lines
per pair.
Bad, because the registry only works if ids stay stable across re-baselines; the contract
below must be enforced by review.

## Contract

1. Identity is registry-qualified: `pb:<id>` for Hub pairs, `synthetic:<name>` for
   constructed fixtures. Never a role name or a bare repo id.
2. Pins, expected ground-truth relation, and evidence live in the registry. Consumers hold
   their own expected outcomes (a verdict, a gate result), never a duplicate pair definition.
3. `smokes/fixtures.yaml` and role-name identity are retired; `_lib/pairs.py` gains a pinned
   resolver that takes `pb:<id>`.
4. Ids are never deleted. A re-baseline migrates ids forward; a retired pair is marked
   deprecated with a successor id.
5. Stored is not scored: new survey pairs enter the `smoke` cohort; promotion to a scored
   cohort is a mentor-reviewed decision, unchanged from today.

## Considered Options

- Federated contract, no data moves — rejected: preserves two registries, unpinned smoke
  runs, and role vocabulary as permanent debt.
- Freeze `fixtures.yaml`, migrate on touch — rejected: a zombie file and continuing drift.
- Discovery index over both stores — rejected: a third artifact that moves duplication
  rather than removing it.

## Migration

Phase 1 is this ADR. Phase 2 (issue to be filed, after the ProvenanceBench fastpath work
lands, which touches `provenancebench/README.md` and `eval.py`): migrate the three missing
pairs (`vocab_drift`, OLMoE, `tiny_smoke`) with pins and evidence into the `smoke` cohort;
add the pinned resolver; switch the five legacy smoke loaders; replace TensorLock's dead
`pb-118`/`pb-003` citations; simplify the catalog lint to `pb:`-only; delete
`smokes/fixtures.yaml`. Phase 3: dedupe pair copies in `config.py` files into `pb:` refs;
add a duplicate-pair audit. The verifier's two hardcoded pairs and their expected verdicts
are evaluated in Phase 2 for citation-by-id; their expectations stay in the tests either way.

Revisit when: ProvenanceBench changes its row schema in a way that cannot migrate ids
forward, or a second consumer needs scored-cohort semantics the registry cannot express.
