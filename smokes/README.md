# Smokes

Runnable checks for the [#26](https://github.com/RedHatResearch/aibom-security/issues/26) fingerprint survey. Older smokes read fixture pairs from [`fixtures.yaml`](fixtures.yaml); newer ones (independence-tests, solar-vs-glm, lora-hash, trace-sft) read pbench-pinned pairs from [`provenancebench/pairs.yaml`](../provenancebench/pairs.yaml). Not product code; not CI.

| Topic | Dir | Method | Issue note |
|---|---|---|---|
| CRS | [`crs/`](crs/) | Centered residual signatures | [#26 CRS](https://github.com/RedHatResearch/aibom-security/issues/26#issuecomment-5327101543) |
| Intrinsic σ-curve | [`intrinsic-fingerprint/`](intrinsic-fingerprint/) | Attention std curves | [#26 Intrinsic FP](https://github.com/RedHatResearch/aibom-security/issues/26#issuecomment-5193649029) |
| Origin Tracer | [`origin-tracer/`](origin-tracer/) | Activation-assisted intermediate reconstruction and rank | [#26 Origin Tracer](https://github.com/RedHatResearch/aibom-security/issues/26#issuecomment-5848818531) |
| HuRef ICS | [`huref-ics/`](huref-ics/) | Invariant-term cosine | [#26 HuRef](https://github.com/RedHatResearch/aibom-security/issues/26#issuecomment-5193183677) |
| MoTHer ℓ_FT | [`mother-distance/`](mother-distance/) | Square-tensor RMS distance | [#26 MoTHer](https://github.com/RedHatResearch/aibom-security/issues/26#issuecomment-5194367004) |
| MoE router Gram | [`moe-router-gram/`](moe-router-gram/) | Gate Gram + expert align | [#17](https://github.com/RedHatResearch/aibom-security/issues/17) · fixtures [`deferred.moe`](fixtures.yaml) |
| Witness Overlap | [`witness-overlap/`](witness-overlap/) | Delta cosine + SVD subspace overlap | [#26 Witness Overlap](https://github.com/RedHatResearch/aibom-security/issues/26) · fixtures [`deferred.dense_qwen3`, `deferred.tiny_smoke`](fixtures.yaml) |
| TensorLock | [`tensorlock/`](tensorlock/) | MSA gate + QPS quant provenance + FT direction | [#26 TensorLock](https://github.com/RedHatResearch/aibom-security/issues/26#issuecomment-5982618419) · fixtures in [`tensorlock/config.py`](tensorlock/config.py) |
| MoE gate probe | [`moe-gate-probe/`](moe-gate-probe/) | TensorLock Eq. 1 s_MSA gate on MoE attention | [#26 MoE gate](https://github.com/RedHatResearch/aibom-security/issues/26#issuecomment-5984348386) · fixtures in [`moe-gate-probe/config.py`](moe-gate-probe/config.py) |
| Independence Tests | [`independence-tests/`](independence-tests/) | csu up_proj MATCH+Spearman p-value | [#26 Independence Tests](https://github.com/RedHatResearch/aibom-security/issues/26#issuecomment-6141604842) · pbench pairs |
| Solar-vs-GLM | [`solar-vs-glm/`](solar-vs-glm/) | LN raw-vs-centered cosine reversal, GPT-2 control | [#26 Solar-vs-GLM](https://github.com/RedHatResearch/aibom-security/issues/26#issuecomment-6141604873) · pbench pairs |
| LoRA-Hash | [`lora-hash/`](lora-hash/) | LoRA delta-W extremum hash, Hamming matching | [#26 LoRA-Hash](https://github.com/RedHatResearch/aibom-security/issues/26#issuecomment-6141610002) · pbench pairs + synthetic corpus |
| TRACE s_H | [`trace-sft/`](trace-sft/) | Checkpoint-update geometry, NNLS harmful-objective score | [#26 TRACE](https://github.com/RedHatResearch/aibom-security/issues/26#issuecomment-6141610015) · self-trained LoRA trajectories |
| What-you-sign | [`what-you-sign/`](what-you-sign/) | GGUF re-derivability: FP-contraction divergence, recipe/imatrix gaps, detection floor, ggufpacker EXACT/NEAR | [#26 What You Sign](https://github.com/RedHatResearch/aibom-security/issues/26#issuecomment-6289519448) |

Shared helpers: [`_lib/`](_lib/) (Hub load, fixture pairs).

```bash
uv sync --all-packages --group smokes
uv run --group smokes python smokes/<topic>/smoke.py
```

External survey tools (GhostSpec, MPK, modelDNA, …): clone to `.scratch/<tool>/`.

Product verify: `verifier/tests/test_integration_*.py` (`@pytest.mark.network`).
