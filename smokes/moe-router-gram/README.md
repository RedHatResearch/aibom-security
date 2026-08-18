# MoE router Gram smoke

- Issue: [#17](https://github.com/RedHatResearch/aibom-security/issues/17) MoE signal probe · survey context [#26](https://github.com/RedHatResearch/aibom-security/issues/26)
- Code: [`smokes/moe-router-gram/`](https://github.com/RedHatResearch/aibom-security/tree/main/smokes/moe-router-gram)

Cosine similarity of MoE `mlp.gate.weight` row Gram matrices; Hungarian expert alignment per layer. Ranged safetensors reads via `aibom_verifier.hf.safetensors_io`.

## Run

```bash
uv sync --all-packages --group smokes
uv run --group smokes python smokes/moe-router-gram/smoke.py
```

Default fixtures: `allenai/OLMoE-1B-7B-0924` ↔ `-SFT` (positive), `deepseek-ai/DeepSeek-V2-Lite` (hard negative). Cache: `smokes/moe-router-gram/.cache/` (gitignored).

## Output

Mean aligned score for positive, hard-negative (gate-rank aligned), and Gaussian null.
