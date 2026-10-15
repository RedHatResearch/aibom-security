"""Solar-vs-GLM dispute smoke — smokes/solar-vs-glm/ · issue #26 survey.

Methodological reproduction of the Solar-vs-GLM verdict-reversal dispute
(2026): the original LayerNorm-cosine accusation (sionic-ai/solar-vs-glm,
later self-retracted in its own README), and the rebuttal
(hyunwoongko/solar-vs-glm-vs-phi) whose re-analysis — mean-centering, the
LayerNorm initialization prior, and a controlled GPT-2 experiment —
reversed the verdict. The methodology is ported with citation; no code is
vendored (both repositories are unlicensed).

Signals: seven flattened-tensor metrics (cosine, centered cosine / Pearson
as an independent cross-check, mean |Δ|, relative L2, coefficient-of-
variation difference, p99 |a| difference) over Llama-style norm and
attention projection tensors on the layer grid {0, n//4, n//2, 3n//4,
n-1}; plus the raw-vs-centered divergence check (the dispute's headline)
and a SHA-256 raw-norm-bytes identity tier. All CPU-only.

Stdout is one JSON object per result line; progress goes to stderr.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF
import yaml
from transformers import GPT2Config, GPT2LMHeadModel

SMOKE_DIR = Path(__file__).resolve().parent
SMOKES_ROOT = SMOKE_DIR.parent
sys.path.insert(0, str(SMOKE_DIR))
sys.path.insert(0, str(SMOKES_ROOT))

from _lib.hub import load_config, load_tensors, parse_model_config  # noqa: E402

# Dispute headline thresholds: raw LN cosine is HIGH on unrelated pairs
# while centered cosine is LOW (Solar-GLM r ≈ 0.01), and both are HIGH on
# genuinely derived pairs (GLM-INTELLECT r ≈ 1.0).
RAW_HIGH = 0.9
CENTERED_HIGH = 0.9
CENTERED_LOW = 0.1

# GPT-2 controlled experiment (rebuttal train_gpt2.py hyperparameters;
# reduced to 30 steps in --selftest for speed).
GPT2_STEPS = 300
GPT2_SELFTEST_STEPS = 30
GPT2_VOCAB = 256
GPT2_N_LAYER = 12
GPT2_N_HEAD = 8
GPT2_N_EMBD = 512
GPT2_BATCH = 32
GPT2_SEQ = 128
GPT2_LR = 1e-3
ONES_INIT_COSINE_MIN = 0.99  # two ones-init models keep cos > 0.99
RAND_INIT_COSINE_MAX = 0.2  # two rand-init models keep |cos| < 0.2

# Dispute divergence thresholds applied to the synthetic selftest vectors.
SELFTEST_NEG_RAW_COSINE_MIN = 0.9
SELFTEST_NEG_CENTERED_COSINE_MAX = 0.1

# Per-pair checkpoint download retry: transient network failures (DNS
# hiccups, the hub's closed-client bug below) get 3 attempts with
# exponential backoff 2s / 8s. A fresh shared session is forced before
# each retry because huggingface_hub closes its global httpx.Client on
# httpx.ConnectError and its own retry loop then reuses the closed
# client ("Cannot send a request, as the client has been closed").
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_BACKOFF_SECONDS = (2.0, 8.0)

# ---------------------------------------------------------------------------
# pairs.yaml loader — shared manifest of pbench ids + pinned revisions.
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    """Locate the repository root: the smoke's parent-parent, else the
    nearest cwd ancestor holding provenancebench/pairs.yaml."""
    here = SMOKE_DIR.parent.parent
    if (here / "provenancebench" / "pairs.yaml").is_file():
        return here
    for cand in [Path.cwd(), *Path.cwd().parents]:
        if (cand / "provenancebench" / "pairs.yaml").is_file():
            return cand
    raise FileNotFoundError("provenancebench/pairs.yaml not found from repo root or cwd")


def load_pair(pair_id: str) -> dict:
    """One entry of the orchestrator's pair manifest, by id."""
    path = repo_root() / "provenancebench" / "pairs.yaml"
    with open(path) as f:
        manifest = yaml.safe_load(f)
    for pair in manifest["pairs"]:
        if pair["id"] == pair_id:
            return pair
    raise KeyError(f"pair id {pair_id} not in {path}")


# Fixture selection: (pbench id, label, tier). Revisions come from the
# manifest, never hardcoded here. The granite/llama2 dispute analog is
# expensive (~28 GB) and only runs with --full.
DEFAULT_PAIR_IDS = (
    ("pb-core-pds-bsl-470m", "neg_pds_bsl", "negative"),
    ("pb-core-smollm-smollm2-strangers", "neg_smollm_smollm2", "negative"),
    ("pb-smoke-qwen3-sft", "pos_qwen3_sft", "positive"),
    ("pb-smoke-smoltulu", "pos_smoltulu", "positive"),
    ("pb-core-smollm2-medit-upscale", "caveat_medit_upscale", "layer0_caveat"),
)
FULL_PAIR_IDS = (("pb-core-llama2-granite7b-strangers", "neg_granite_llama2", "negative"),)
ALL_LABELS = tuple(label for _, label, _ in DEFAULT_PAIR_IDS + FULL_PAIR_IDS)


# ---------------------------------------------------------------------------
# Pure metric core — no I/O, no fixture knowledge. Takes plain numpy
# arrays; reused by --selftest on synthetic tensors.
# ---------------------------------------------------------------------------


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    """cos(a, b) over the flattened vectors; None on zero norm."""
    af, bf = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom == 0.0:
        return None
    return float(np.dot(af, bf) / denom)


def centered_cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    """cos of (a - mean(a)) vs (b - mean(b)) — the rebuttal's fix."""
    af, bf = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    ac, bc = af - af.mean(), bf - bf.mean()
    denom = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    if denom == 0.0:
        return None
    return float(np.dot(ac, bc) / denom)


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson r as an independent z-score-product mean; mathematically
    equal to centered cosine — computed differently as a cross-check."""
    af, bf = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    az = (af - af.mean()) / af.std()
    bz = (bf - bf.mean()) / bf.std()
    return float(np.mean(az * bz))


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """mean |a - b|."""
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float | None:
    """||a - b|| / ||a|| with `a` as the reference — ASYMMETRIC:
    rel_l2(a, b) != rel_l2(b, a). This smoke always passes the pbench
    reference model as `a` and records that convention here."""
    af, bf = a.astype(np.float64), b.astype(np.float64)
    denom = float(np.linalg.norm(af))
    if denom == 0.0:
        return None
    return float(np.linalg.norm(af - bf) / denom)


def _cv(x: np.ndarray) -> float | None:
    """std(x) / |mean(x)|; None when the mean vanishes (div-by-zero guard)."""
    xf = x.astype(np.float64).ravel()
    denom = abs(float(xf.mean()))
    if denom == 0.0:
        return None
    return float(xf.std() / denom)


def cv_diff(a: np.ndarray, b: np.ndarray) -> float | None:
    """|cv(a) - cv(b)|; None when either mean vanishes."""
    ca, cb = _cv(a), _cv(b)
    if ca is None or cb is None:
        return None
    return float(abs(ca - cb))


def p99_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """|p99(|a|) - p99(|b|)| (linear interpolation, like np.percentile)."""
    pa = float(np.percentile(np.abs(a.astype(np.float64)), 99))
    pb = float(np.percentile(np.abs(b.astype(np.float64)), 99))
    return abs(pa - pb)


def all_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """The seven flattened-tensor metrics for one same-shape pair (a =
    reference side for rel_l2)."""
    cc = centered_cosine(a, b)
    pr = pearson(a, b)
    return {
        "cosine": cosine(a, b),
        "centered_cosine": cc,
        "pearson": pr,
        "pearson_minus_centered": None if (cc is None or pr is None) else pr - cc,
        "mean_abs_diff": mean_abs_diff(a, b),
        "rel_l2": rel_l2(a, b),
        "cv_diff": cv_diff(a, b),
        "p99_abs_diff": p99_abs_diff(a, b),
    }


def divergence_check(raw: float | None, centered: float | None) -> str | None:
    """Classify a LayerNorm pair against the dispute's headline pattern.
    'divergent': raw high, centered low (the Solar-GLM shape — raw cosine
    is initialization prior, not pattern similarity). 'aligned': both
    high (the GLM-INTELLECT derived shape). None otherwise."""
    if raw is None or centered is None:
        return None
    if raw > RAW_HIGH and centered < CENTERED_LOW:
        return "divergent"
    if raw > RAW_HIGH and centered > CENTERED_HIGH:
        return "aligned"
    return None


def layer_grid(num_layers: int) -> list[int]:
    """{0, n//4, n//2, 3n//4, n-1}, deduplicated, ascending — retains the
    layer-0 special case (rebuttal: the accusation's within-model
    baseline artifact came from layer-0 input_layernorm being atypical)."""
    n = num_layers
    return sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})


def _mean(values: list) -> float | None:
    xs = [v for v in values if v is not None]
    return float(np.mean(xs)) if xs else None


# ---------------------------------------------------------------------------
# Hub loading layer (smoke-only; the pure core above never touches I/O).
# ---------------------------------------------------------------------------

# Tensor roles per layer, exactly as in the dispute repos.
LN_ROLES = ("input_layernorm", "post_attention_layernorm")
PROJ_ROLES = ("self_attn.k_proj", "self_attn.v_proj")


def _layer_tensor_name(layer: int, role: str) -> str:
    return f"model.layers.{layer}.{role}.weight"


def needed_names(num_layers: int) -> set[str]:
    out: set[str] = set()
    for role in LN_ROLES + PROJ_ROLES:
        out |= {_layer_tensor_name(i, role) for i in range(num_layers)}
    return out


class NoSafetensorsError(Exception):
    """Repo snapshot contains no *.safetensors at the pinned revision
    (e.g. a GGUF-only or pytorch_model.bin-only rehost) — this smoke is
    safetensors-only and structurally cannot consume the checkpoint."""


def _reset_hub_session() -> None:
    """Force huggingface_hub to drop and recreate its shared httpx.Client.

    Root cause of the observed 'RuntimeError: Cannot send a request, as
    the client has been closed': on a transient httpx.ConnectError the
    hub's backoff helper calls close_session() (defensively, for SSL
    errors) but keeps sending through the client object it captured
    *before* the retry loop, so its own retry hits the closed client and
    the resulting RuntimeError is not in its retryable set. Resetting
    the session here hands every retry a fresh, open client."""
    from huggingface_hub.utils import close_session

    close_session()


def _is_transient_download_error(exc: Exception) -> bool:
    """httpx.ConnectError (DNS/network hiccup) or the closed-client
    RuntimeError that the hub's ConnectError handling itself induces."""
    import httpx

    if isinstance(exc, httpx.ConnectError):
        return True
    return isinstance(exc, RuntimeError) and "client has been closed" in str(exc)


def _download_with_retry(repo_id: str, revision: str, cache_dir: Path | None) -> dict:
    """_load_checkpoint with transient-network retry: up to
    DOWNLOAD_ATTEMPTS attempts, exponential backoff 2s / 8s, and a fresh
    hub session before each retry (see _reset_hub_session)."""
    last_exc: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        if attempt > 1:
            _reset_hub_session()
            wait = DOWNLOAD_BACKOFF_SECONDS[min(attempt - 2, len(DOWNLOAD_BACKOFF_SECONDS) - 1)]
            _log(
                f"download retry {attempt}/{DOWNLOAD_ATTEMPTS} for {repo_id}"
                f" in {wait:g}s ({last_exc})"
            )
            time.sleep(wait)
        try:
            return _load_checkpoint(repo_id, revision, cache_dir)
        except NoSafetensorsError:
            raise  # structural, not transient — never retried
        except Exception as exc:  # noqa: BLE001 — classified below
            if not _is_transient_download_error(exc):
                raise
            last_exc = exc
    raise last_exc  # all attempts transient-failed; exhaust and raise


def _load_checkpoint(repo_id: str, revision: str, cache_dir: Path | None) -> dict:
    """One pinned checkpoint: float32 tensors for the roles plus the raw
    LayerNorm weight bytes (SHA-256 identity tier)."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    config = load_config(repo_id, cache_dir=cache_dir)
    num_layers = config["num_hidden_layers"]

    # Format gate: this smoke reads safetensors only. A pinned revision
    # that ships none (GGUF-only rehost like Data-Selection/BSL-470M at
    # 1581b9dc, or pytorch_model.bin-only repos) is a structural skip,
    # not an error — mirror of the independence-tests skipped_format
    # policy. Detected from the snapshot itself, with an HF API file
    # listing as detail when reachable.
    model_dir = snapshot_download(
        repo_id,
        allow_patterns=["*.safetensors", "model.safetensors.index.json"],
        cache_dir=cache_dir,
        revision=revision,
    )
    shards = sorted(Path(model_dir).glob("*.safetensors"))
    if not shards:
        detail = f"no *.safetensors in {repo_id} at {revision}"
        try:
            from huggingface_hub import HfApi

            files = HfApi().list_repo_files(repo_id, revision=revision)
            detail = f"no *.safetensors in {repo_id} at {revision}; repo files: {sorted(files)[:8]}"
        except Exception:  # noqa: BLE001 — detail only; the snapshot already says enough
            pass
        raise NoSafetensorsError(detail)

    names = needed_names(num_layers)
    tensors = load_tensors(repo_id, names, cache_dir=cache_dir, revision=revision)

    ln_names = {_layer_tensor_name(i, role) for i in range(num_layers) for role in LN_ROLES}
    raw: dict[str, bytes] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():  # noqa: SIM118 — safe_open is not iterable
                if name in ln_names and name not in raw:
                    t = handle.get_tensor(name)
                    raw[name] = t.contiguous().view(torch.uint8).numpy().tobytes()
    missing = ln_names - raw.keys()
    if missing:
        raise KeyError(f"Missing LN tensors for {repo_id}: {sorted(missing)[:5]}")
    return {
        "repo_id": repo_id,
        "revision": revision,
        "config": config,
        "tensors": tensors,
        "raw_ln_bytes": raw,
    }


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Pair runner: metrics over the layer grid, with explicit skip records.
# ---------------------------------------------------------------------------


def run_pair_metrics(subject: dict, reference: dict) -> dict:
    """Seven metrics over the layer grid for one (subject, reference) pair.
    Roles or shapes that do not line up are skipped explicitly and
    recorded, never silently dropped."""
    cfg_sub = parse_model_config(subject["repo_id"], subject["config"])
    cfg_ref = parse_model_config(reference["repo_id"], reference["config"])
    if cfg_sub.num_layers != cfg_ref.num_layers:
        return {
            "status": "skipped",
            "reason": f"layer count mismatch: {cfg_sub.num_layers} vs {cfg_ref.num_layers}",
            "per_tensor": [],
            "skipped": [],
            "ln_summary": None,
        }

    grid = layer_grid(cfg_ref.num_layers)
    entries: list[dict] = []
    skips: list[dict] = []
    for layer in grid:
        for role in LN_ROLES + PROJ_ROLES:
            name = _layer_tensor_name(layer, role)
            sub_t = subject["tensors"].get(name)
            ref_t = reference["tensors"].get(name)
            label = f"L{layer}.{role}"
            if sub_t is None or ref_t is None:
                skips.append({"tensor": label, "reason": "missing tensor"})
                continue
            if sub_t.shape != ref_t.shape:
                skips.append(
                    {
                        "tensor": label,
                        "reason": (
                            f"shape mismatch subject {tuple(sub_t.shape)} "
                            f"vs reference {tuple(ref_t.shape)}"
                        ),
                    }
                )
                continue
            m = all_metrics(ref_t, sub_t)  # reference = rel_l2 reference side
            entry = {"tensor": label, "layer": layer, "role": role, **m}
            if role in LN_ROLES:
                entry["divergence"] = divergence_check(m["cosine"], m["centered_cosine"])
            entries.append(entry)

    # Byte-identity tier (the accusation repo's definitive_proof.py): SHA-256
    # of the raw norm-tensor bytes on the same grid; exact matches are the
    # only provenance proof this methodology can actually deliver.
    byte_matches = []
    for layer in grid:
        for role in LN_ROLES:
            name = _layer_tensor_name(layer, role)
            sub_b = subject["raw_ln_bytes"].get(name)
            ref_b = reference["raw_ln_bytes"].get(name)
            if sub_b is None or ref_b is None:
                continue
            byte_matches.append(
                {
                    "tensor": f"L{layer}.{role}",
                    "subject_sha256": sha256_hex(sub_b),
                    "reference_sha256": sha256_hex(ref_b),
                    "identical": sub_b == ref_b,
                }
            )

    # Layer-0 caveat record (the accusation's artifact): layer-0
    # input_layernorm is atypical — record, per model, its cosine against
    # the same layer's post_attention_layernorm so the asymmetry is visible.
    layer0_records = {}
    for side, ckpt in (("subject", subject), ("reference", reference)):
        i0 = _layer_tensor_name(0, "input_layernorm")
        p0 = _layer_tensor_name(0, "post_attention_layernorm")
        if i0 in ckpt["tensors"] and p0 in ckpt["tensors"]:
            layer0_records[side] = {
                "input_vs_post_layer0_cosine": cosine(ckpt["tensors"][i0], ckpt["tensors"][p0]),
            }

    ln_entries = [e for e in entries if e["role"] in LN_ROLES]
    n_aligned = sum(1 for e in ln_entries if e.get("divergence") == "aligned")
    n_divergent = sum(1 for e in ln_entries if e.get("divergence") == "divergent")
    return {
        "status": "ok",
        "layer_grid": grid,
        "per_tensor": entries,
        "skipped": skips,
        "layer0": layer0_records,
        "byte_identity": byte_matches,
        "ln_summary": {
            "n_ln_tensors": len(ln_entries),
            "n_aligned": n_aligned,
            "n_divergent": n_divergent,
            "mean_raw_cosine": _mean([e["cosine"] for e in ln_entries]),
            "mean_centered_cosine": _mean([e["centered_cosine"] for e in ln_entries]),
            "n_byte_identical": sum(1 for e in byte_matches if e["identical"]),
        },
    }


def run_pair(pair: dict, label: str, cache_dir: Path | None) -> dict:
    """Run one pbench fixture pair end-to-end. Non-comparability and
    unconsumable checkpoint formats are recorded as skipped, hard load
    errors as status=error — never a crash of the whole run."""
    started = time.perf_counter()
    out = {
        "label": label,
        "pair_id": pair["id"],
        "kind": pair.get("kind"),
        "expected": pair.get("expected"),
        "subject": {"repo_id": pair["subject"], "revision": pair.get("subject_rev")},
        "reference": {"repo_id": pair["reference"], "revision": pair.get("reference_rev")},
    }
    try:
        subject = _download_with_retry(pair["subject"], pair["subject_rev"], cache_dir)
        reference = _download_with_retry(pair["reference"], pair["reference_rev"], cache_dir)
    except NoSafetensorsError as exc:
        # Structural, not a load failure: the pinned revision ships no
        # safetensors (GGUF-only / bin-only rehost). Mirrors the
        # independence-tests smoke's skipped_format policy — the pair
        # stays in the manifest; the run is not failed by it.
        out["status"] = "skipped"
        out["reason"] = "gguf_only_no_safetensors"
        out["detail"] = str(exc)
        out["compute_seconds"] = time.perf_counter() - started
        return out
    except Exception as exc:  # noqa: BLE001 — record, don't crash the run
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["compute_seconds"] = time.perf_counter() - started
        return out
    out.update(run_pair_metrics(subject, reference))
    out["compute_seconds"] = time.perf_counter() - started
    return out


# ---------------------------------------------------------------------------
# GPT-2 controlled experiment (rebuttal train_gpt2.py, ported; CPU-only).
# ---------------------------------------------------------------------------


def _gpt2_make_rule_tokens(step: int, stream_id: int) -> torch.Tensor:
    """Distinct arithmetic data stream per model (rebuttal §2)."""
    start = (step + stream_id * 100) % GPT2_VOCAB
    pos = torch.arange(GPT2_SEQ, dtype=torch.long)
    seq = (start + pos) % GPT2_VOCAB
    return seq.unsqueeze(0).expand(GPT2_BATCH, GPT2_SEQ).clone()


def _gpt2_reinit_layernorm(model: torch.nn.Module, mode: str, seed: int) -> None:
    """Re-initialize LN weights: 'ones' (the LLM convention) or 'rand'."""
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            with torch.no_grad():
                if mode == "ones":
                    m.weight.fill_(1.0)
                elif mode == "rand":
                    m.weight.normal_(0.0, 1.0, generator=g)
                else:
                    raise ValueError(mode)
                if m.bias is not None:
                    m.bias.zero_()


def _gpt2_train_one(
    model: torch.nn.Module, opt: torch.optim.Optimizer, step: int, stream_id: int
) -> float:
    tokens = _gpt2_make_rule_tokens(step, stream_id)
    input_ids, labels = tokens[:, :-1], tokens[:, 1:]
    opt.zero_grad()
    model.train()
    logits = model(input_ids=input_ids).logits
    loss = TF.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    loss.backward()
    opt.step()
    return float(loss.item())


def _ln_cosines(m1: torch.nn.Module, m2: torch.nn.Module) -> dict[str, list[float]]:
    """Cross-model per-layer LN weight cosines (rebuttal's ln_1/ln_2)."""
    out: dict[str, list[float]] = {"ln_1": [], "ln_2": []}
    for blk1, blk2 in zip(m1.transformer.h, m2.transformer.h, strict=True):
        for key, t1, t2 in (
            ("ln_1", blk1.ln_1.weight, blk2.ln_1.weight),
            ("ln_2", blk1.ln_2.weight, blk2.ln_2.weight),
        ):
            out[key].append(cosine(t1.detach().numpy(), t2.detach().numpy()))
    return out


def run_gpt2_control(steps: int) -> dict:
    """Train four tiny GPT-2s from scratch on distinct arithmetic streams
    (CPU): two with LayerNorm ones-init, two with random-init LN. The
    rebuttal's controlled experiment: ones-init cross-model LN cosine
    stays > 0.99 while rand-init stays near 0 — LN cosine is an
    initialization prior, not provenance evidence."""
    started = time.perf_counter()
    config = GPT2Config(
        vocab_size=GPT2_VOCAB,
        n_layer=GPT2_N_LAYER,
        n_head=GPT2_N_HEAD,
        n_embd=GPT2_N_EMBD,
        n_positions=GPT2_SEQ,
        n_ctx=GPT2_SEQ,
        bos_token_id=0,
        eos_token_id=1,
    )
    torch.manual_seed(10)
    model_ones_a = GPT2LMHeadModel(config)
    torch.manual_seed(20)
    model_ones_b = GPT2LMHeadModel(config)
    torch.manual_seed(30)
    model_rand_a = GPT2LMHeadModel(config)
    torch.manual_seed(40)
    model_rand_b = GPT2LMHeadModel(config)

    _gpt2_reinit_layernorm(model_ones_a, "ones", seed=10)
    _gpt2_reinit_layernorm(model_ones_b, "ones", seed=20)
    _gpt2_reinit_layernorm(model_rand_a, "rand", seed=30)
    _gpt2_reinit_layernorm(model_rand_b, "rand", seed=40)

    models = {
        "ones_a": (model_ones_a, torch.optim.Adam(model_ones_a.parameters(), lr=GPT2_LR), 1),
        "ones_b": (model_ones_b, torch.optim.Adam(model_ones_b.parameters(), lr=GPT2_LR), 2),
        "rand_a": (model_rand_a, torch.optim.Adam(model_rand_a.parameters(), lr=GPT2_LR), 3),
        "rand_b": (model_rand_b, torch.optim.Adam(model_rand_b.parameters(), lr=GPT2_LR), 4),
    }
    first_losses, last_losses = {}, {}
    for step in range(steps):
        for name, (model, opt, stream_id) in models.items():
            loss = _gpt2_train_one(model, opt, step, stream_id)
            if step == 0:
                first_losses[name] = loss
            last_losses[name] = loss

    cos_ones = _ln_cosines(model_ones_a, model_ones_b)
    cos_rand = _ln_cosines(model_rand_a, model_rand_b)
    ones_min = min(v for vals in cos_ones.values() for v in vals)
    rand_abs_max = max(abs(v) for vals in cos_rand.values() for v in vals)
    checks = {
        "ones_init_cosine_gt_0.99": ones_min > ONES_INIT_COSINE_MIN,
        "rand_init_abs_cosine_lt_0.2": rand_abs_max < RAND_INIT_COSINE_MAX,
    }
    return {
        "steps": steps,
        "config": {
            "vocab_size": GPT2_VOCAB,
            "n_layer": GPT2_N_LAYER,
            "n_head": GPT2_N_HEAD,
            "n_embd": GPT2_N_EMBD,
            "batch": GPT2_BATCH,
            "seq_len": GPT2_SEQ,
            "lr": GPT2_LR,
            "optimizer": "Adam",
            "device": "cpu",
        },
        "first_step_losses": first_losses,
        "last_step_losses": last_losses,
        "cosines_ones_init": cos_ones,
        "cosines_rand_init": cos_rand,
        "ones_init_min_cosine": ones_min,
        "rand_init_max_abs_cosine": rand_abs_max,
        "checks": checks,
        "passed": all(checks.values()),
        "compute_seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# Selftest: offline, synthetic tensors with known values.
# ---------------------------------------------------------------------------


def run_selftest() -> int:
    """Offline: the GPT-2 control at reduced steps + the seven metrics on
    synthetic tensors with known values. Exits 0 on success."""
    failures: list[str] = []

    # --- Synthetic LN-shaped vectors: ones + noise vs ones + different
    # noise — the dispute's exact situation. Raw cosine must be high
    # (shared mean direction), centered cosine low (uncorrelated patterns).
    rng = np.random.default_rng(0)
    d = 4096
    noise_scale = 0.01
    a = np.ones(d, dtype=np.float32) + rng.normal(0.0, noise_scale, d)
    b = np.ones(d, dtype=np.float32) + rng.normal(0.0, noise_scale, d)
    m = all_metrics(a, b)
    if m["cosine"] is None or m["cosine"] <= SELFTEST_NEG_RAW_COSINE_MIN:
        failures.append(f"synthetic raw cosine not high: {m['cosine']}")
    if m["centered_cosine"] is None or m["centered_cosine"] >= SELFTEST_NEG_CENTERED_COSINE_MAX:
        failures.append(f"synthetic centered cosine not low: {m['centered_cosine']}")
    if m["pearson_minus_centered"] is None or abs(m["pearson_minus_centered"]) > 1e-10:
        failures.append(f"pearson != centered cosine: {m['pearson_minus_centered']}")
    if divergence_check(m["cosine"], m["centered_cosine"]) != "divergent":
        failures.append("synthetic pair not classified as divergent")

    # Self-comparison sanity: perfect scores, zero deltas.
    m_self = all_metrics(a, a)
    if (
        m_self["cosine"] is None
        or m_self["centered_cosine"] is None
        or (abs(m_self["cosine"] - 1.0) > 1e-12 or abs(m_self["centered_cosine"] - 1.0) > 1e-12)
    ):
        failures.append(f"self-comparison not 1.0: {m_self['cosine']}, {m_self['centered_cosine']}")
    if m_self["mean_abs_diff"] != 0.0 or m_self["rel_l2"] != 0.0 or m_self["p99_abs_diff"] != 0.0:
        failures.append("self-comparison deltas not zero")

    # Affine copy (derived-pair tier): centered cosine exactly 1, raw high.
    c = np.ones(d, dtype=np.float32) + rng.normal(0.0, noise_scale, d)
    dd = 3.0 * c - 2.0
    m_aff = all_metrics(c, dd)
    if not (
        m_aff["cosine"] is not None
        and m_aff["cosine"] > RAW_HIGH
        and m_aff["centered_cosine"] is not None
        and m_aff["centered_cosine"] > CENTERED_HIGH
    ):
        failures.append(
            f"affine pair not aligned: cos={m_aff['cosine']}, r={m_aff['centered_cosine']}"
        )
    if divergence_check(m_aff["cosine"], m_aff["centered_cosine"]) != "aligned":
        failures.append("affine pair not classified as aligned")

    # Zero-mean guard for cv_diff; layer grid; known SHA-256 vector.
    z = np.zeros(8, dtype=np.float32)
    if cv_diff(z, z) is not None:
        failures.append("cv_diff zero-mean guard failed")
    if layer_grid(24) != [0, 6, 12, 18, 23] or layer_grid(5) != [0, 1, 2, 3, 4]:
        failures.append("layer_grid wrong")
    if sha256_hex(b"abc") != ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"):
        failures.append("sha256 mismatch on known vector")

    # --- GPT-2 controlled experiment at reduced steps (offline, CPU).
    gpt2 = run_gpt2_control(steps=GPT2_SELFTEST_STEPS)
    if not gpt2["passed"]:
        failures.append(f"gpt2 control failed: {gpt2['checks']}")

    print(
        json.dumps(
            {
                "smoke": "solar-vs-glm",
                "mode": "selftest",
                "synthetic_metrics": m,
                "gpt2_control": {
                    "steps": gpt2["steps"],
                    "ones_init_min_cosine": gpt2["ones_init_min_cosine"],
                    "rand_init_max_abs_cosine": gpt2["rand_init_max_abs_cosine"],
                    "checks": gpt2["checks"],
                    "passed": gpt2["passed"],
                    "compute_seconds": gpt2["compute_seconds"],
                },
                "failures": failures,
                "passed": not failures,
            },
            default=float,
        )
    )
    return 0 if not failures else 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solar-vs-GLM dispute smoke (smokes/solar-vs-glm/)"
    )
    parser.add_argument("--selftest", action="store_true", help="offline selftest (no network)")
    parser.add_argument(
        "--full",
        action="store_true",
        help="include the dispute-analog pair pb-core-llama2-granite7b-strangers (~28 GB)",
    )
    parser.add_argument(
        "--gpt2-steps", type=int, default=GPT2_STEPS, help="GPT-2 training steps (default 300)"
    )
    parser.add_argument("--no-gpt2", action="store_true", help="skip the GPT-2 control")
    parser.add_argument("--cache-dir", type=Path, default=None, help="HF cache directory")
    parser.add_argument("--only", default=None, help="run a single fixture label")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(run_selftest())

    if args.only and args.only not in ALL_LABELS:
        parser.error(f"unknown fixture label: {args.only} (choose from {ALL_LABELS})")

    selected = DEFAULT_PAIR_IDS + (FULL_PAIR_IDS if args.full else ())
    if args.only:
        selected = tuple(s for s in selected if s[1] == args.only)
        if not selected:
            parser.error(
                f"--only {args.only} selects no fixtures in this mode "
                f"(it is a --full-tier fixture); rerun with --full"
            )

    started = time.perf_counter()
    _log(f"solar-vs-glm smoke: {len(selected)} fixture(s), full={args.full}")

    if not args.no_gpt2:
        _log(f"training GPT-2 control ({args.gpt2_steps} steps, CPU)")
        control = run_gpt2_control(steps=args.gpt2_steps)
        print(json.dumps({"result": "gpt2_control", **control}, default=float))
        if not control["passed"]:
            _log("WARNING: GPT-2 control checks failed")

    pair_entries: list[dict] = []
    for pair_id, label, tier in selected:
        _log(f"running fixture {label} ({pair_id})")
        pair = load_pair(pair_id)
        entry = run_pair(pair, label, args.cache_dir)
        entry["tier"] = tier
        pair_entries.append(entry)
        print(json.dumps({"result": "pair", **entry}, default=float))
        _log(f"fixture {label}: status={entry.get('status')}")

    counts = {"ok": 0, "skipped": 0, "error": 0}
    by_status: dict[str, list[str]] = {"ok": [], "skipped": [], "error": []}
    for entry in pair_entries:
        status = entry.get("status", "error")
        if status not in counts:
            status = "error"
        counts[status] += 1
        by_status[status].append(entry["label"])
    print(
        json.dumps(
            {
                "result": "summary",
                "smoke": "solar-vs-glm",
                "params": {
                    "raw_high": RAW_HIGH,
                    "centered_high": CENTERED_HIGH,
                    "centered_low": CENTERED_LOW,
                    "gpt2_steps": None if args.no_gpt2 else args.gpt2_steps,
                    "full": args.full,
                    "fixture_count": len(selected),
                },
                "counts": counts,
                "by_status": by_status,
                "exit_nonzero_if": "any pair status == error",
                "wall_seconds": time.perf_counter() - started,
            }
        )
    )
    # Gate: any pair ending in status error fails the run; skipped
    # (e.g. gguf_only_no_safetensors) is fine. The GPT-2 control keeps
    # its own warning-only behavior.
    if counts["error"]:
        _log(f"GATE: {counts['error']} fixture(s) in error: {by_status['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
