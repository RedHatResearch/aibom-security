# Smokes

Runnable checks for the [#26](https://github.com/RedHatResearch/aibom-security/issues/26) fingerprint survey ([`fixtures.yaml`](fixtures.yaml)). Not product code; not CI.

| Topic | Dir | Method | Issue note |
|---|---|---|---|
| CRS | [`crs/`](crs/) | Centered residual signatures | [#26 CRS](https://github.com/RedHatResearch/aibom-security/issues/26#issuecomment-5327101543) |
| Intrinsic σ-curve | [`intrinsic-fingerprint/`](intrinsic-fingerprint/) | Attention std curves | [#26](https://github.com/RedHatResearch/aibom-security/issues/26) (search Intrinsic Fingerprint) |
| HuRef ICS | [`huref-ics/`](huref-ics/) | Invariant-term cosine | [#26](https://github.com/RedHatResearch/aibom-security/issues/26) (search HuRef) |
| MoTHer ℓ_FT | [`mother-distance/`](mother-distance/) | Square-tensor RMS distance | [#26](https://github.com/RedHatResearch/aibom-security/issues/26) (search MoTHer) |
| MoE router Gram | [`moe-router-gram/`](moe-router-gram/) | Gate Gram + expert align | [#17](https://github.com/RedHatResearch/aibom-security/issues/17) / #26 MoE bucket |

Shared helpers: [`_lib/`](_lib/) (Hub load, fixture pairs).

Run (from repo root):

```bash
uv sync --all-packages
uv run --group smokes python smokes/<topic>/smoke.py
```

External survey tools (GhostSpec, MPK, modelDNA, …): clone to `.scratch/<tool>/`; run upstream CLIs there.

Product verify integration tests: `verifier/tests/test_integration_*.py` (`@pytest.mark.network`, pinned SHAs).
