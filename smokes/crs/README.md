# CRS smoke

- Paper: [arXiv:2608.14929](https://arxiv.org/abs/2608.14929)
- Survey note: [#26 comment](https://github.com/RedHatResearch/aibom-security/issues/26#issuecomment-5327101543)
- Code: [`smokes/crs/`](https://github.com/RedHatResearch/aibom-security/tree/main/smokes/crs)

Minimal reimplementation (§3): branch products, centered signatures, Hungarian alignment, raw `L` only — no null calibration.

## Run

```bash
uv run --group smokes python smokes/crs/smoke.py
```

Requires HF network. ~3 GB download per 1.7B model on cold cache.

## Default pairs

From [`../fixtures.yaml`](../fixtures.yaml): SmolLM2↔SmolTulu (positive), Llama↔Dolphin (vocab drift), SmolLM2↔Llama (INCOMPATIBLE — depth mismatch).
