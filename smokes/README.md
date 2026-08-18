# Smokes

Runnable checks for the [#26](https://github.com/RedHatResearch/aibom-security/issues/26) fingerprint survey ([`fixtures.yaml`](fixtures.yaml)). Not product code; not CI.

| Topic | Dir | Method | Issue note |
|---|---|---|---|
| CRS | [`crs/`](crs/) | Centered residual signatures | [#26 CRS](https://github.com/RedHatResearch/aibom-security/issues/26#issuecomment-5327101543) |
| Intrinsic σ-curve | [`intrinsic-fingerprint/`](intrinsic-fingerprint/) | Attention std curves | [#26 Intrinsic FP](https://github.com/RedHatResearch/aibom-security/issues/26#issuecomment-5193649029) |
| HuRef ICS | [`huref-ics/`](huref-ics/) | Invariant-term cosine | [#26 HuRef](https://github.com/RedHatResearch/aibom-security/issues/26#issuecomment-5193183677) |
| MoTHer ℓ_FT | [`mother-distance/`](mother-distance/) | Square-tensor RMS distance | [#26 MoTHer](https://github.com/RedHatResearch/aibom-security/issues/26#issuecomment-5194367004) |
| MoE router Gram | [`moe-router-gram/`](moe-router-gram/) | Gate Gram + expert align | [#17](https://github.com/RedHatResearch/aibom-security/issues/17) · fixtures [`deferred.moe`](fixtures.yaml) |

Shared helpers: [`_lib/`](_lib/) (Hub load, fixture pairs).

```bash
uv sync --all-packages --group smokes
uv run --group smokes python smokes/<topic>/smoke.py
```

External survey tools (GhostSpec, MPK, modelDNA, …): clone to `.scratch/<tool>/`.

Product verify: `verifier/tests/test_integration_*.py` (`@pytest.mark.network`).
