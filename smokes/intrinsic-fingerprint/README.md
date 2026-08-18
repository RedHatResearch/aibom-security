# Intrinsic Fingerprint smoke

- Paper: withdrawn [arXiv:2507.03014](https://arxiv.org/abs/2507.03014) (PDF-only fork)
- Survey note: [#26 Intrinsic Fingerprint comment](https://github.com/RedHatResearch/aibom-security/issues/26)
- Code: [`smokes/intrinsic-fingerprint/`](https://github.com/RedHatResearch/aibom-security/tree/main/smokes/intrinsic-fingerprint)

Per-layer `std` of attention Q/K/V/O → z-score depth curves → Pearson (interpolate if depths differ).

## Run

```bash
uv run --group smokes python smokes/intrinsic-fingerprint/smoke.py
```

HF network required. Pairs from [`../fixtures.yaml`](../fixtures.yaml).
