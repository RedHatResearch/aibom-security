# HuRef ICS smoke

- Paper: [arXiv:2312.04828](https://arxiv.org/abs/2312.04828) · upstream [LUMIA-Group/HuRef](https://github.com/LUMIA-Group/HuRef)
- Survey note: [#26 HuRef comment](https://github.com/RedHatResearch/aibom-security/issues/26)
- Code: [`smokes/huref-ics/`](https://github.com/RedHatResearch/aibom-security/tree/main/smokes/huref-ics)

Llama-style invariant terms (QK, VO, FFN) on last 2 layers, K=4096 rare tokens from a short corpus. GQA KV expand. Not stock HuRef (no StyleGAN/ZKP).

## Run

```bash
uv run --group smokes python smokes/huref-ics/smoke.py
```

HF network + ~5 GB RAM per 1.7B model. Pairs from [`../fixtures.yaml`](../fixtures.yaml).
