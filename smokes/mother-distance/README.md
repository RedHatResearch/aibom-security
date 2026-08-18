# MoTHer distance smoke

- Paper: [arXiv:2405.18432](https://arxiv.org/abs/2405.18432) · upstream [eliahuhorwitz/MoTHer](https://github.com/eliahuhorwitz/MoTHer)
- Survey note: [#26 MoTHer comment](https://github.com/RedHatResearch/aibom-security/issues/26)
- Code: [`smokes/mother-distance/`](https://github.com/RedHatResearch/aibom-security/tree/main/smokes/mother-distance)

Adapted pair check: mean RMS of weight diffs on matching square 2-D tensors (MoTHer ℓ_FT filter). Kurtosis sum for direction hint only — not MDST/tree recovery.

## Run

```bash
uv run --group smokes python smokes/mother-distance/smoke.py
```

HF network; full safetensors load per model. Pairs from [`../fixtures.yaml`](../fixtures.yaml).
