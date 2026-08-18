from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache"

# feeds #17; see smokes/fixtures.yaml deferred.moe
FIXTURES = {
    "parent": "allenai/OLMoE-1B-7B-0924",
    "child": "allenai/OLMoE-1B-7B-0924-SFT",
    "negative": "deepseek-ai/DeepSeek-V2-Lite",
}

GATE_SUFFIX = "mlp.gate.weight"
SEED = 0
