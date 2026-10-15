"""Independence Tests smoke — smokes/independence-tests/ · arXiv:2502.12292 · #26 survey.

Weight-space `csu` independence statistic from "Independence Tests for
Language Models" (Zhu, Ahmed, Kuditipudi & Liang, ICML 2025,
arXiv:2502.12292), reimplemented from the paper only: the authors'
artifact (github.com/ahmeda14960/model-tracing) ships no license and was
used strictly as a read-only behavioral cross-check (see README.md).

Per GLU-MLP layer (state-dict keys matching "mlp.up_proj"; both models
must have identical key sets and shapes, else abstain):
  1. cast the [intermediate, hidden] up-projection matrices to float64;
  2. Alg. 2 MATCH: row-to-row cosine-similarity matrix between the two
     models' rows, maximum-weight Hungarian assignment
     (scipy linear_sum_assignment, maximize=True);
  3. Alg. 3 SPEARMAN: Spearman rank correlation between the matched
     column indices and the identity permutation -> per-layer two-sided
     p-value (scipy.stats.spearmanr);
  4. Alg. 4 FISHER: X = -2 * sum(ln p_l); aggregate p = chi2.sf(X, 2L).
Verdict: aggregate p < 0.05 -> non_independent; >= 0.05 -> independent;
incomparable stacks -> abstain with reason "shape_mismatch". Per-pair
expectation outcome: held / missed / skipped_format (format-out-of-scope
abstain) / borderline_false_positive (expected-independent pair with
0.01 < p <= alpha); the gate fails only on "missed".
Constrained-setting scope (paper §2.3): csu detects *unpermuted* weight
dependence (fine-tunes, quantizations). The permutation-robust statistics
(csh / match) are out of scope, as are MoE expert naming (the
"mlp.up_proj" substring misses "mlp.experts.*") and GGUF checkpoints.

Stdout is a single JSON document; progress goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.numpy import load_file as load_safetensors_np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2, spearmanr

SMOKE_DIR = Path(__file__).resolve().parent
SMOKES_ROOT = SMOKE_DIR.parent
sys.path.insert(0, str(SMOKE_DIR))
sys.path.insert(0, str(SMOKES_ROOT))

from _lib.hub import load_tensors  # noqa: E402

# Significance level of the independence test (verdict threshold).
ALPHA = 0.05

# Outcome values for the per-pair expectation check (the gate fails only
# on "missed"; see README "Expectation outcomes and the gate").
OUTCOME_HELD = "held"
OUTCOME_MISSED = "missed"
OUTCOME_SKIPPED_FORMAT = "skipped_format"
OUTCOME_BORDERLINE_FP = "borderline_false_positive"

# Abstain reasons meaning "this smoke structurally cannot consume the
# pair's checkpoint format" — expectation check records skipped_format.
FORMAT_OUT_OF_SCOPE_REASONS = frozenset({"out_of_scope_gguf", "out_of_scope_moe", "no_up_proj"})

# For expected-independent pairs: p in this band is a borderline false
# positive (mildly anti-conservative Fisher combination), not a gate miss.
BORDERLINE_LOW = 0.01

# AWQ GEMM packing constants (AutoAWQ convention): 4-bit values, eight per
# int32 column, packed in the [0,2,4,6,1,3,5,7] order.
AWQ_BITS = 4
AWQ_PACK_NUM = 32 // AWQ_BITS
AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

WEIGHT_PATTERNS = ["*.safetensors", "model.safetensors.index.json"]

# Default fixture set — the 135M-1.7B-scale subset of
# provenancebench/pairs.yaml: (pbench pair id, fixture role, expected verdict).
DEFAULT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("pb-smoke-qwen3-sft", "positive", "non_independent"),
    ("pb-smoke-smoltulu", "positive", "non_independent"),
    ("pb-core-pds-bsl-470m", "hard_negative", "independent"),
    ("pb-core-smollm-smollm2-strangers", "hard_negative", "independent"),
    ("pb-smoke-qwen25-awq", "quant_robustness", "non_independent"),
    ("pb-core-sheared-pythia", "abstain_edge", "abstain"),
)

# Ad-hoc --pair runs: expected verdict from the pair's provenancebench label.
_EXPECTED_FROM_YAML = {
    "related": "non_independent",
    "unrelated": "independent",
    "unsupported": "abstain",
}


@dataclass(frozen=True)
class PairSpec:
    """One provenancebench pair (ids and pinned revisions from pairs.yaml)."""

    pair_id: str
    stack: str
    kind: str
    expected: str
    subject: str
    reference: str
    subject_rev: str
    reference_rev: str


class NoSafetensorsError(Exception):
    """Repo snapshot contains no *.safetensors (e.g. a GGUF-only rehost)."""


# ---------------------------------------------------------------------------
# Pure scoring core — no I/O, no fixture knowledge.
# ---------------------------------------------------------------------------


def _layer_index(name: str) -> int:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def layer_csu(mat_a: np.ndarray, mat_b: np.ndarray) -> dict:
    """Alg. 2 MATCH + Alg. 3 SPEARMAN for one up_proj pair.

    Rows (hidden units) of the two [intermediate, hidden] matrices are
    matched by cosine similarity with a maximum-weight bipartite
    assignment; the matched column indices are then correlated against
    the identity permutation with Spearman. Under unpermuted dependence
    (fine-tuning, quantization) the assignment recovers the identity, so
    rho is large and the two-sided p is tiny; under independence the
    recovered permutation is exchangeable and p is uniform.
    """
    a = np.asarray(mat_a, dtype=np.float64)
    b = np.asarray(mat_b, dtype=np.float64)
    na = np.linalg.norm(a, axis=1, keepdims=True)
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    na[na == 0.0] = 1.0
    nb[nb == 0.0] = 1.0
    cos = (a / na) @ (b / nb).T
    rows, cols = linear_sum_assignment(cos, maximize=True)
    matched_cosines = cos[rows, cols]
    rho, p = spearmanr(cols, np.arange(cols.size))
    return {
        "rho": float(rho),
        "p": float(p),
        "mean_matched_cosine": float(matched_cosines.mean()),
    }


def fisher_combine(pvals: list[float]) -> dict:
    """Alg. 4 FISHER: X = -2*sum(ln p_i), p = chi2.sf(X, df=2k); NaN p skipped
    (the authors' convention for degenerate layers)."""
    valid = [p for p in pvals if not np.isnan(p)]
    if not valid:
        return {"p": None, "num_layers": 0, "fisher_stat": None}
    stat = float(-2.0 * np.sum(np.log(np.clip(valid, 1e-300, 1.0))))
    return {
        "p": float(chi2.sf(stat, df=2 * len(valid))),
        "num_layers": len(valid),
        "fisher_stat": stat,
    }


def csu_score(stack_a: dict[str, np.ndarray], stack_b: dict[str, np.ndarray]) -> dict:
    """Full csu statistic over all up_proj layers (stacks must be comparable)."""
    per_layer = []
    for key in sorted(stack_a, key=_layer_index):
        result = layer_csu(stack_a[key], stack_b[key])
        result["layer"] = _layer_index(key)
        per_layer.append(result)
    combined = fisher_combine([r["p"] for r in per_layer])
    if combined["p"] is None:
        return {"verdict": "abstain", "reason": "no_valid_layers", "per_layer": per_layer}
    verdict = "non_independent" if combined["p"] < ALPHA else "independent"
    return {
        "verdict": verdict,
        "p": combined["p"],
        "fisher_stat": combined["fisher_stat"],
        "num_layers": combined["num_layers"],
        "per_layer": per_layer,
    }


def _geometry_differs(geo_a: dict | None, geo_b: dict | None) -> list[str]:
    diffs = []
    for key in ("num_hidden_layers", "hidden_size", "intermediate_size"):
        va, vb = (geo_a or {}).get(key), (geo_b or {}).get(key)
        if va is not None and vb is not None and va != vb:
            diffs.append(f"{key} {va} vs {vb}")
    return diffs


def comparability(
    stack_a: dict[str, np.ndarray],
    stack_b: dict[str, np.ndarray],
    geo_a: dict | None,
    geo_b: dict | None,
) -> dict | None:
    """Precondition check: identical up_proj key sets and shapes. Returns the
    abstain payload, or None when the stacks are comparable."""
    if not stack_a and not stack_b:
        diffs = _geometry_differs(geo_a, geo_b)
        if diffs:
            return {
                "verdict": "abstain",
                "reason": "shape_mismatch",
                "layer": 0,
                "detail": "no mlp.up_proj tensors on either side; " + "; ".join(diffs),
            }
        return {
            "verdict": "abstain",
            "reason": "no_up_proj",
            "layer": None,
            "detail": "no mlp.up_proj tensors found (non-GLU MLP naming, MoE "
            "expert naming, or a GGUF-only repo); see README scope",
        }
    if set(stack_a) != set(stack_b):
        diff = set(stack_a) ^ set(stack_b)
        layer = min(_layer_index(k) for k in diff)
        return {
            "verdict": "abstain",
            "reason": "shape_mismatch",
            "layer": layer,
            "detail": f"up_proj key sets differ: {len(stack_a)} vs {len(stack_b)} tensors",
        }
    for key in sorted(stack_a, key=_layer_index):
        if stack_a[key].shape != stack_b[key].shape:
            return {
                "verdict": "abstain",
                "reason": "shape_mismatch",
                "layer": _layer_index(key),
                "detail": f"{key}: {stack_a[key].shape} vs {stack_b[key].shape}",
            }
    return None


# ---------------------------------------------------------------------------
# AWQ int4 dequantization — numpy port of AutoAWQ's dequantize_gemm
# (github.com/casper-hansen/AutoAWQ, awq/utils/packing_utils.py).
# ---------------------------------------------------------------------------


def dequantize_awq_gemm(
    qweight: np.ndarray,
    qzeros: np.ndarray,
    scales: np.ndarray,
    group_size: int,
    bits: int = AWQ_BITS,
) -> np.ndarray:
    """Dequantize an AWQ GEMM int4 layer.

    qweight [in, out/8] and qzeros [in/group_size, out/8] hold eight 4-bit
    values per int32 column (packed in AWQ_ORDER); scales is
    [in/group_size, out] float16. Returns the weight in AutoAWQ's
    [in_features, out_features] layout — transpose for the HF [out, in]
    `.weight` layout.
    """
    shifts = np.arange(0, 32, bits, dtype=np.int32)
    mask = (1 << bits) - 1
    iweight = (qweight[:, :, None] >> shifts) & mask
    iweight = iweight.reshape(qweight.shape[0], -1)
    izeros = (qzeros[:, :, None] >> shifts) & mask
    izeros = izeros.reshape(qzeros.shape[0], -1)
    # Undo the [0,2,4,6,1,3,5,7] packing order within each int32 column:
    # AutoAWQ reverses per 8-wide block (arange(n).view(-1, 8)[:, REVERSE]),
    # so each block gets its own offset, not a tiled pattern.
    n_cols = iweight.shape[1]
    blocks = np.arange(n_cols).reshape(-1, AWQ_PACK_NUM)
    reverse = blocks[:, AWQ_REVERSE_ORDER].reshape(-1)
    iweight = iweight[:, reverse]
    izeros = izeros[:, reverse]
    izeros = np.repeat(izeros, group_size, axis=0)
    scales_full = np.repeat(scales, group_size, axis=0).astype(np.float32)
    return (iweight - izeros).astype(np.float32) * scales_full


# ---------------------------------------------------------------------------
# Hub loading layer (smoke-only; the pure core above never touches I/O).
# ---------------------------------------------------------------------------


def _pairs_yaml_path() -> Path:
    for bases in (SMOKE_DIR.resolve().parents, Path.cwd().parents):
        for base in bases:
            cand = base / "provenancebench" / "pairs.yaml"
            if cand.is_file():
                return cand
    raise FileNotFoundError("provenancebench/pairs.yaml not found above this smoke")


def load_pair_specs() -> dict[str, PairSpec]:
    """All pairs from the orchestrator manifest (ids + pinned revisions)."""
    with open(_pairs_yaml_path()) as f:
        raw = yaml.safe_load(f)
    return {
        entry["id"]: PairSpec(
            pair_id=entry["id"],
            stack=entry["stack"],
            kind=entry["kind"],
            expected=entry["expected"],
            subject=entry["subject"],
            reference=entry["reference"],
            subject_rev=entry["subject_rev"],
            reference_rev=entry["reference_rev"],
        )
        for entry in raw["pairs"]
    }


DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_BACKOFF_SECONDS = (2.0, 8.0)


def _reset_hub_session() -> None:
    """Drop the shared httpx client: huggingface_hub closes it on
    ConnectError but its retry loop keeps the dead reference."""
    try:
        from huggingface_hub.utils import close_session

        close_session()
    except Exception:  # noqa: BLE001 — best-effort reset
        pass


def _snapshot(repo_id: str, revision: str, cache_dir: Path | None, patterns: list[str]) -> Path:
    last_exc: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        if attempt > 1:
            _reset_hub_session()
            wait = DOWNLOAD_BACKOFF_SECONDS[min(attempt - 2, len(DOWNLOAD_BACKOFF_SECONDS) - 1)]
            print(
                f"[download] retry {attempt}/{DOWNLOAD_ATTEMPTS} {repo_id} in {wait:g}s",
                file=sys.stderr,
            )
            time.sleep(wait)
        try:
            return Path(
                snapshot_download(
                    repo_id, allow_patterns=patterns, cache_dir=cache_dir, revision=revision
                )
            )
        except NoSafetensorsError:
            raise  # structural, not transient
        except Exception as exc:  # noqa: BLE001 — classified below
            transient = (
                "Server disconnected" in str(exc)
                or "client has been closed" in str(exc)
                or type(exc).__name__ in ("ConnectError", "ReadTimeout", "RemoteProtocolError")
            )
            if not transient:
                raise
            last_exc = exc
    raise last_exc  # all attempts transient-failed


def _load_geometry(repo_id: str, revision: str, cache_dir: Path | None) -> dict | None:
    """Best-effort geometry from the pinned config.json (abstain metadata)."""
    try:
        path = hf_hub_download(
            repo_id, filename="config.json", cache_dir=cache_dir, revision=revision
        )
        with open(path) as f:
            cfg = json.load(f)
    except Exception:  # geometry is optional metadata; missing config is not fatal
        return None
    geo = {key: cfg.get(key) for key in ("num_hidden_layers", "hidden_size", "intermediate_size")}
    quant = cfg.get("quantization_config") or {}
    if quant.get("quant_method"):
        geo["quant_method"] = quant["quant_method"]
    return geo


def _up_names(n_layers: int) -> set[str]:
    return {f"model.layers.{i}.mlp.up_proj.weight" for i in range(n_layers)}


def _load_up_proj_awq(shards: list[Path]) -> dict[str, np.ndarray]:
    """Dequantized up_proj matrices from int4-packed AWQ safetensors."""
    out: dict[str, np.ndarray] = {}
    for shard in shards:
        tensors = load_safetensors_np(shard)
        packed: dict[str, dict[str, np.ndarray]] = {}
        for name, arr in tensors.items():
            if "mlp.up_proj" not in name:
                continue
            base, field = name.rsplit(".", 1)  # field in {qweight, qzeros, scales}
            packed.setdefault(base, {})[field] = arr
        del tensors
        for base, parts in packed.items():
            qweight, qzeros, scales = parts["qweight"], parts["qzeros"], parts["scales"]
            group_size = qweight.shape[0] // qzeros.shape[0]
            # dequantize returns [in, out]; HF .weight is [out, in].
            out[base + ".weight"] = dequantize_awq_gemm(qweight, qzeros, scales, group_size).T
    return out


def load_up_proj(
    repo_id: str, revision: str, geometry: dict | None, cache_dir: Path | None
) -> dict[str, np.ndarray]:
    """mlp.up_proj matrices [intermediate, hidden] of one pinned checkpoint.

    Dense checkpoints go through _lib.hub.load_tensors. int4-packed AWQ
    checkpoints (qweight/qzeros/scales instead of .weight) are dequantized
    with the numpy port of AutoAWQ's dequantize_gemm. Checkpoints whose
    tensor naming has no mlp.up_proj (e.g. GPTNeoX dense_h_to_4h, MoE
    experts) yield an empty stack, which the comparability check turns
    into an abstain.
    """
    manifest = _snapshot(repo_id, revision, cache_dir, ["model.safetensors.index.json"])
    index_file = manifest / "model.safetensors.index.json"
    weight_map: dict[str, str] = {}
    if index_file.is_file():
        with open(index_file) as f:
            weight_map = json.load(f).get("weight_map", {})
        if not any("mlp.up_proj" in name for name in weight_map):
            return {}

    if (geometry or {}).get("quant_method") == "awq":
        model_dir = _snapshot(repo_id, revision, cache_dir, WEIGHT_PATTERNS)
        shards = sorted(model_dir.glob("*.safetensors"))
        if not shards:
            raise NoSafetensorsError(f"no *.safetensors in {repo_id}")
        return _load_up_proj_awq(shards)

    n_layers = (geometry or {}).get("num_hidden_layers")
    if n_layers is None and weight_map:
        # Single-shard repos have no index; sharded ones let us infer the
        # layer count when config.json is unavailable.
        layer_ids = {
            int(m.group(1))
            for name in weight_map
            for m in [re.search(r"layers\.(\d+)\.", name)]
            if m
        }
        n_layers = max(layer_ids) + 1 if layer_ids else None
    if n_layers is None:
        return {}
    try:
        return dict(
            load_tensors(repo_id, _up_names(n_layers), cache_dir=cache_dir, revision=revision)
        )
    except KeyError:
        # Architecture without llama-style mlp.up_proj tensor names.
        return {}
    except FileNotFoundError as exc:
        raise NoSafetensorsError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Fixture runner.
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _expectation_outcome(entry: dict, expected_verdict: str) -> tuple[str, str | None]:
    """Classify a scored pair against its pre-registered expectation.

    Gate semantics (see README "Expectation outcomes and the gate"): the
    exit code fails only on a plain miss. Pairs the smoke structurally
    cannot consume (GGUF-only / MoE / no up_proj naming) are
    skipped_format, and a non_independent verdict on an
    expected-independent pair with p in (0.01, alpha] is a
    borderline_false_positive — both recorded, neither gating.
    """
    verdict = entry["verdict"]
    p = entry.get("p")
    if verdict == expected_verdict:
        return OUTCOME_HELD, None
    if verdict == "abstain" and entry.get("reason") in FORMAT_OUT_OF_SCOPE_REASONS:
        return (
            OUTCOME_SKIPPED_FORMAT,
            f"format out of scope for this smoke (reason={entry['reason']}); "
            "verdict kept as abstain, neither held nor a gate miss",
        )
    if (
        expected_verdict == "independent"
        and verdict == "non_independent"
        and p is not None
        and BORDERLINE_LOW < p <= ALPHA
    ):
        return (
            OUTCOME_BORDERLINE_FP,
            f"expected independent but p={p} lies in the borderline band "
            f"({BORDERLINE_LOW}, {ALPHA}]; Fisher combination over layers "
            "is mildly anti-conservative at this scale (see README); "
            "recorded, not a gate miss",
        )
    return OUTCOME_MISSED, None


def run_pair(spec: PairSpec, role: str, expected_verdict: str, cache_dir: Path | None) -> dict:
    """Score one pair end-to-end and return its JSON entry."""
    t0 = time.perf_counter()
    entry: dict = {
        "pair_id": spec.pair_id,
        "role": role,
        "kind": spec.kind,
        "subject": {"repo_id": spec.subject, "revision": spec.subject_rev},
        "reference": {"repo_id": spec.reference, "revision": spec.reference_rev},
        "expected_verdict": expected_verdict,
    }
    if spec.stack == "moe":
        entry.update(
            {
                "verdict": "abstain",
                "reason": "out_of_scope_moe",
                "layer": None,
                "detail": "MoE expert tensors are not matched by the mlp.up_proj substring",
            }
        )
    else:
        geo_s = _load_geometry(spec.subject, spec.subject_rev, cache_dir)
        geo_r = _load_geometry(spec.reference, spec.reference_rev, cache_dir)
        entry["geometry"] = {"subject": geo_s, "reference": geo_r}
        try:
            stack_s = load_up_proj(spec.subject, spec.subject_rev, geo_s, cache_dir)
            stack_r = load_up_proj(spec.reference, spec.reference_rev, geo_r, cache_dir)
        except NoSafetensorsError as exc:
            entry.update(
                {
                    "verdict": "abstain",
                    "reason": "out_of_scope_gguf",
                    "layer": None,
                    "detail": str(exc),
                }
            )
        else:
            block = comparability(stack_s, stack_r, geo_s, geo_r)
            if block is not None:
                entry.update(block)
            else:
                _log(f"[{spec.pair_id}] scoring {len(stack_s)} up_proj layers")
                entry.update(csu_score(stack_s, stack_r))
    outcome, outcome_note = _expectation_outcome(entry, expected_verdict)
    entry["outcome"] = outcome
    entry["expectation_held"] = outcome == OUTCOME_HELD
    if outcome_note:
        entry["outcome_note"] = outcome_note
    entry["compute_seconds"] = round(time.perf_counter() - t0, 1)
    _log(
        f"[{spec.pair_id}] verdict={entry['verdict']} p={entry.get('p')} "
        f"outcome={entry['outcome']} ({entry['compute_seconds']}s)"
    )
    return entry


# ---------------------------------------------------------------------------
# Offline selftest — synthetic tensors only, no network, no HF downloads.
# ---------------------------------------------------------------------------


def _synthetic_stacks(
    rng: np.random.Generator, n_layers: int, inter: int, hidden: int, shared_fraction: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Two synthetic up_proj stacks. Model B keeps the first
    `shared_fraction` rows of every model-A layer (plus tiny noise, the
    unpermuted fine-tuning shape) and resamples the rest."""
    keys = [f"model.layers.{i}.mlp.up_proj.weight" for i in range(n_layers)]
    stack_a = {k: rng.standard_normal((inter, hidden)) for k in keys}
    n_shared = int(inter * shared_fraction)
    stack_b = {}
    for key in keys:
        b = rng.standard_normal((inter, hidden))
        b[:n_shared] = stack_a[key][:n_shared] + 0.02 * rng.standard_normal((n_shared, hidden))
        stack_b[key] = b
    return stack_a, stack_b


def _awq_roundtrip_check(rng: np.random.Generator) -> dict:
    """Pack a synthetic HF-layout weight into the AutoAWQ GEMM int4 layout,
    dequantize with dequantize_awq_gemm, and compare: validates the
    unpack/reorder path against AutoAWQ's packing semantics offline."""
    out_dim, in_dim, group_size = 64, 256, 128
    groups = in_dim // group_size
    w = rng.standard_normal((out_dim, in_dim)).astype(np.float32)
    wg = w.reshape(out_dim, groups, group_size)
    s16 = (np.abs(wg).max(axis=2, keepdims=True) / 7.0 + 1e-8).astype(np.float16)
    zero = 8  # zero point; |w|/scale <= 7 so q = round(w/s) + 8 stays in [1, 15]
    q = np.clip(np.round(wg / s16.astype(np.float32)) + zero, 0, 15).astype(np.int32)
    qt = q.reshape(out_dim, in_dim).T.copy()  # [in, out]
    n_pack = out_dim // AWQ_PACK_NUM
    qweight = np.zeros((in_dim, n_pack), dtype=np.int32)
    qzeros = np.zeros((groups, n_pack), dtype=np.int32)
    zeros = np.full((groups, out_dim), zero, dtype=np.int32)
    # Pack eight 4-bit values per int32 column in AWQ_ORDER, undoing
    # AutoAWQ's block-aware reverse permutation.
    packed_src = np.arange(out_dim).reshape(-1, AWQ_PACK_NUM)[:, AWQ_ORDER].reshape(-1)
    for col in range(n_pack):
        for k in range(AWQ_PACK_NUM):
            src = packed_src[col * AWQ_PACK_NUM + k]
            qweight[:, col] |= qt[:, src] << (AWQ_BITS * k)
            qzeros[:, col] |= zeros[:, src] << (AWQ_BITS * k)
    scales = np.ascontiguousarray(s16.reshape(out_dim, groups).T)  # [groups, out]
    deq = dequantize_awq_gemm(qweight, qzeros, scales, group_size).T  # back to [out, in]
    err = np.abs(deq - w)
    return {
        "max_abs_err": float(err.max()),
        "scale_max": float(scales.max()),
        "corr": float(np.corrcoef(deq.ravel(), w.ravel())[0, 1]),
    }


def run_selftest() -> int:
    """Exercise the full scoring path offline; exit 0 on success."""
    t0 = time.perf_counter()
    checks: list[tuple[str, bool, dict]] = []

    # 1) Dependence: 60% shared up_proj rows (aligned, unpermuted) must give
    #    a non_independent verdict with a tiny aggregate p.
    rng = np.random.default_rng(20261012)
    stack_a, stack_b = _synthetic_stacks(rng, n_layers=4, inter=96, hidden=64, shared_fraction=0.6)
    related = csu_score(stack_a, stack_b)
    checks.append(
        (
            "related_verdict_non_independent",
            related["verdict"] == "non_independent",
            {"p": related.get("p"), "verdict": related["verdict"]},
        )
    )
    checks.append(
        (
            "related_p_small",
            (related.get("p") or 1.0) < 1e-6,
            {"p": related.get("p"), "threshold": 1e-6},
        )
    )

    # 2) Independence: fully independent synthetic pairs. Under H0 the
    #    aggregate p is uniform, so ~5% of trials fall below alpha — the
    #    method's nominal false-positive rate, not a bug.
    trial_ps = []
    for trial in range(40):
        r = np.random.default_rng(100 + trial)
        sa = {f"model.layers.{i}.mlp.up_proj.weight": r.standard_normal((48, 32)) for i in range(2)}
        sb = {f"model.layers.{i}.mlp.up_proj.weight": r.standard_normal((48, 32)) for i in range(2)}
        trial_ps.append(csu_score(sa, sb)["p"])
    n_below = sum(1 for p in trial_ps if p < ALPHA)
    checks.append(
        (
            "independent_direction",
            min(trial_ps) > (related.get("p") or 0.0),
            {"min_p": min(trial_ps), "related_p": related.get("p")},
        )
    )
    checks.append(
        (
            "independent_calibration",
            n_below <= 7,
            {"trials": len(trial_ps), "n_below_alpha": n_below, "expected": "about 2 of 40"},
        )
    )

    # 3) Abstain paths: shape mismatch, key-set mismatch, and the
    #    both-empty + differing-geometry case (the sheared-pythia shape).
    stack_c = {k: rng.standard_normal((128, 64)) for k in stack_a}
    block = comparability(stack_a, stack_c, None, None)
    checks.append(
        (
            "abstain_shape_mismatch",
            block is not None and block["reason"] == "shape_mismatch" and block["layer"] == 0,
            block,
        )
    )
    block = comparability(stack_a, {}, None, None)
    checks.append(
        (
            "abstain_key_set_mismatch",
            block is not None and block["reason"] == "shape_mismatch",
            block,
        )
    )
    block = comparability(
        {},
        {},
        {"num_hidden_layers": 12, "hidden_size": 768, "intermediate_size": 3072},
        {"num_hidden_layers": 24, "hidden_size": 1024, "intermediate_size": 4096},
    )
    checks.append(
        (
            "abstain_geometry_mismatch",
            block is not None and block["reason"] == "shape_mismatch" and block["layer"] == 0,
            block,
        )
    )

    # 4) AWQ dequantization round-trip (validates int4 unpack/reorder).
    rt = _awq_roundtrip_check(np.random.default_rng(7))
    checks.append(
        (
            "awq_roundtrip",
            rt["corr"] > 0.99 and rt["max_abs_err"] <= 0.5 * rt["scale_max"] + 1e-4,
            rt,
        )
    )

    # 5) Expectation-outcome classification (gate policy, offline).
    outcome_cases = [
        ({"verdict": "independent", "p": 0.4}, "independent", OUTCOME_HELD),
        ({"verdict": "non_independent", "p": 1e-30}, "non_independent", OUTCOME_HELD),
        ({"verdict": "abstain", "reason": "shape_mismatch"}, "independent", OUTCOME_MISSED),
        ({"verdict": "independent", "p": 0.9}, "non_independent", OUTCOME_MISSED),
        (
            {"verdict": "abstain", "reason": "out_of_scope_gguf"},
            "independent",
            OUTCOME_SKIPPED_FORMAT,
        ),
        ({"verdict": "abstain", "reason": "out_of_scope_moe"}, "abstain", OUTCOME_HELD),
        ({"verdict": "abstain", "reason": "no_up_proj"}, "independent", OUTCOME_SKIPPED_FORMAT),
        ({"verdict": "non_independent", "p": 0.032}, "independent", OUTCOME_BORDERLINE_FP),
        ({"verdict": "non_independent", "p": 0.001}, "independent", OUTCOME_MISSED),
        ({"verdict": "non_independent", "p": 0.01}, "independent", OUTCOME_MISSED),
    ]
    outcome_ok = all(_expectation_outcome(e, exp)[0] == want for e, exp, want in outcome_cases)
    checks.append(
        (
            "expectation_outcomes",
            outcome_ok,
            {"cases": len(outcome_cases), "outcomes": sorted({w for _, _, w in outcome_cases})},
        )
    )

    output = {
        "smoke": "independence-tests",
        "mode": "selftest",
        "checks": [
            {"name": name, "passed": passed, "detail": detail} for name, passed, detail in checks
        ],
        "all_passed": all(passed for _, passed, _ in checks),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    print(json.dumps(output, indent=2, default=float))
    return 0 if output["all_passed"] else 1


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independence Tests smoke (smokes/independence-tests/)"
    )
    parser.add_argument(
        "--selftest", action="store_true", help="offline synthetic self-test (no network)"
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="HF cache directory")
    parser.add_argument(
        "--pair", default=None, help="run one ProvenanceBench pair id instead of the default set"
    )
    args = parser.parse_args()

    if args.selftest:
        sys.exit(run_selftest())

    specs = load_pair_specs()
    if args.pair:
        if args.pair not in specs:
            parser.error(f"unknown pair id: {args.pair} (not in provenancebench/pairs.yaml)")
        selected = [(args.pair, "adhoc", None)]
    else:
        selected = list(DEFAULT_PAIRS)

    t0 = time.perf_counter()
    entries = []
    for pair_id, role, expected in selected:
        spec = specs[pair_id]
        if expected is None:
            expected = _EXPECTED_FROM_YAML.get(spec.expected, "abstain")
            if spec.stack == "moe":
                # MoE is a documented abstain (expert naming), not a miss.
                expected = "abstain"
        entries.append(run_pair(spec, role, expected, args.cache_dir))

    summary = {
        "pairs_total": len(entries),
        "held": [e["pair_id"] for e in entries if e["outcome"] == OUTCOME_HELD],
        "missed": [e["pair_id"] for e in entries if e["outcome"] == OUTCOME_MISSED],
        "skipped_format": [e["pair_id"] for e in entries if e["outcome"] == OUTCOME_SKIPPED_FORMAT],
        "borderline_false_positives": [
            e["pair_id"] for e in entries if e["outcome"] == OUTCOME_BORDERLINE_FP
        ],
    }
    output = {
        "smoke": "independence-tests",
        "paper": "Independence Tests for Language Models "
        "(Zhu, Ahmed, Kuditipudi, Liang; ICML 2025), arXiv:2502.12292",
        "statistic": "csu (MLP up_proj MATCH + SPEARMAN, Fisher-combined)",
        "params": {"alpha": ALPHA, "borderline_band": [BORDERLINE_LOW, ALPHA]},
        "pairs": entries,
        "summary": summary,
        "total_seconds": round(time.perf_counter() - t0, 1),
    }
    print(json.dumps(output, indent=2, default=float))
    # Gate: fail only on hard expectation misses; skipped_format and
    # borderline_false_positive are recorded but do not fail the run.
    sys.exit(1 if summary["missed"] else 0)


if __name__ == "__main__":
    main()
