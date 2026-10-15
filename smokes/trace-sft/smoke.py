"""TRACE smoke — smokes/trace-sft/ · arXiv:2610.07518 · #26 survey.

Reimplementation of the scoring equations (Eq. 1-5) of "Harmful SFT Leaves a
Continuous Trace in LLM Checkpoint Updates" (TRACE): a kernel over LoRA
checkpoint updates, a reference-bank geometry (harmful C / refusal S / benign B
class-mean prototypes, non-harmful pole mu_N = (mu_S + mu_B) / 2), and a
2-variable NNLS harmfulness score s_H = gamma_C / (gamma_C + gamma_N).
The paper's anonymous code has NO license, so nothing is vendored: the scoring
core is reimplemented from the paper equations, and the SFT harness is written
from scratch to the paper's hyperparameters. See README.md for scope,
deviations and pass criteria.

Modes:
  --selftest  offline numeric checks: factorized kernel == dense computation,
              LoRA reparameterization invariance (A->cA, B->B/c), exact 2-var
              NNLS vs scipy, s_H monotonicity on a synthetic geometry, and
              agreement of the block-mean Gram path with brute-force dense
              feature-space NNLS. No network, no HF downloads. Exits 0 on
              success.
  --quick     full pipeline on a synthetic random-token data substitute,
              K=2 bank, 20 steps (~6-10 min CPU). Gate: pipeline runs
              end-to-end and scores are computed; achieved Spearman is
              reported, not thresholded.
  (default)   full pipeline: real PKU-SafeRLHF / ultrachat_200k data fetched
              via the datasets-server REST API, K=4 bank, 100 steps
              (~8-14 h measured on the M3 Pro CPU target; user-gated, NOT run
              by the orchestrator by default). Pass: Spearman(ratio, s_H)
              >= 0.9 and matched C/S AUROC >= 0.9 (paper: 0.986-0.992 / 0.988).
  --real-ood  additionally score the real Qwen3-0.6B instruct-vs-base dW
              against the bank geometry (downloads ~1.2 GB; expectation: low
              s_H; diagnostic, not gated).

Stdout: one JSON object per scored row, then a final summary JSON object
(which also embeds the rows). Progress goes to stderr.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import torch
import yaml
from scipy.optimize import nnls as scipy_nnls
from scipy.stats import spearmanr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

SMOKE_DIR = Path(__file__).resolve().parent
SMOKES_ROOT = SMOKE_DIR.parent
REPO_ROOT = SMOKES_ROOT.parent
sys.path.insert(0, str(SMOKE_DIR))
sys.path.insert(0, str(SMOKES_ROOT))

import config as C  # noqa: E402
from _lib.hub import load_config, load_tensors  # noqa: E402

# An update is either factorized (module -> (A, B, s), dW = s * B @ A with
# A: r x in, B: out x r) or dense (module -> dW). The kernel never
# materializes a factorized dW (Eq. 2-3 of the paper).
Factorized = dict[str, tuple[np.ndarray, np.ndarray, float]]
Dense = dict[str, np.ndarray]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Provenance: backbone pair ids/revisions come from the orchestrator manifest
# (provenancebench/pairs.yaml, pair pb-smoke-qwen3-sft).
# ---------------------------------------------------------------------------


def find_repo_root() -> Path:
    for cand in (REPO_ROOT, Path.cwd(), *Path.cwd().parents):
        if (cand / "provenancebench" / "pairs.yaml").is_file():
            return cand
    return REPO_ROOT


def load_pinned_pair() -> dict[str, str]:
    """{base repo/rev, instruct repo/rev} for pair pb-smoke-qwen3-sft."""
    path = find_repo_root() / "provenancebench" / "pairs.yaml"
    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        for entry in raw.get("pairs", []):
            if entry.get("id") == C.PAIR_ID:
                return {
                    "reference": str(entry["reference"]),
                    "reference_rev": str(entry["reference_rev"]),
                    "subject": str(entry["subject"]),
                    "subject_rev": str(entry["subject_rev"]),
                }
        _log(f"warning: pair {C.PAIR_ID} not in {path}; using config fallback")
    except OSError as e:
        _log(f"warning: cannot read {path} ({e}); using config fallback")
    return dict(C.FALLBACK_PAIR)


# ---------------------------------------------------------------------------
# Pure scoring core (Eq. 1-5) — no I/O, no fixture knowledge.
# ---------------------------------------------------------------------------


def _is_factorized(u: Factorized | Dense) -> bool:
    return bool(u) and isinstance(next(iter(u.values())), tuple)


def kernel(u: Factorized | Dense, v: Factorized | Dense) -> float:
    """K(i, j) = sum over shared modules of <dW_i, dW_j>_F (Eq. 1-3).

    Low-rank x low-rank is computed factorized with r x r float64
    intermediates: s_i*s_j*trace((B_i^T B_j)(A_j A_i^T)). Dense x low-rank and
    dense x dense are supported for the real-OOD query path.
    """
    fu, fv = _is_factorized(u), _is_factorized(v)
    if fu and fv:
        total = 0.0
        for m in u.keys() & v.keys():
            a_i, b_i, s_i = u[m]
            a_j, b_j, s_j = v[m]
            m1 = b_i.T @ b_j  # r x r
            m2 = a_j @ a_i.T  # r x r
            total += s_i * s_j * float(np.sum(m1 * m2.T))  # tr(m1 @ m2)
        return total
    if fu or fv:
        dense, fact = (u, v) if fv else (v, u)
        total = 0.0
        for m, d in dense.items():
            if m in fact:
                a, b, s = fact[m]
                total += s * float(np.sum(b * (d @ a.T)))  # <d, s*b@a>_F
        return total
    return float(sum(np.sum(u[m] * v[m]) for m in u.keys() & v.keys()))


def bank_gram(bank: list[Factorized | Dense]) -> np.ndarray:
    m = len(bank)
    gram = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(i, m):
            gram[i, j] = gram[j, i] = kernel(bank[i], bank[j])
    return gram


def phi_of(update: Factorized | Dense, bank: list[Factorized | Dense]) -> np.ndarray:
    """Kernel embedding phi(dW) = (K(dW, b_1), ..., K(dW, b_m)) (Eq. 4)."""
    return np.array([kernel(update, b) for b in bank], dtype=np.float64)


def prototype_gram(gram: np.ndarray, labels: list[str]) -> np.ndarray:
    """2x2 prototype Gram G for the (C, N) prototype pair, computed from
    block means of the bank kernel Gram (Eq. 49): G[c,d] is the mean of
    K(i,j) over i in class c, j in class d. The non-harmful pole
    mu_N = 0.5*(mu_S + mu_B) gives G[C,N] = 0.5*(G[C,S] + G[C,B]) and
    G[N,N] = 0.25*(G[S,S] + 2*G[S,B] + G[B,B])."""
    idx = {c: np.array([i for i, lab in enumerate(labels) if lab == c]) for c in ("C", "S", "B")}

    def block(a: str, b: str) -> float:
        return float(gram[np.ix_(idx[a], idx[b])].mean())

    g_cn = 0.5 * (block("C", "S") + block("C", "B"))
    g_nn = 0.25 * (block("S", "S") + 2.0 * block("S", "B") + block("B", "B"))
    return np.array([[block("C", "C"), g_cn], [g_cn, g_nn]], dtype=np.float64)


def nnls2(b: np.ndarray, gram2: np.ndarray) -> tuple[float, float]:
    """Exact 2-variable NNLS min ||x - M g||^2, g >= 0, by face enumeration on
    the 2x2 prototype Gram gram2 = M^T M (normal equations gram2 g = b).

    Faces: {}, {0}, {1}, {0,1}; residual J(g) = -2 b.g + g.gram2.g compared
    over feasible candidates (||x||^2 is constant across faces).
    """
    d0, d1 = float(gram2[0, 0]), float(gram2[1, 1])
    cross = float(gram2[0, 1])
    cands: list[tuple[float, float]] = [(0.0, 0.0)]
    if d0 > 0:
        cands.append((float(b[0]) / d0, 0.0))
    if d1 > 0:
        cands.append((0.0, float(b[1]) / d1))
    det = d0 * d1 - cross * cross
    if d0 > 0 and d1 > 0 and abs(det) > 1e-12 * d0 * d1:
        cands.append(
            (
                (float(b[0]) * d1 - float(b[1]) * cross) / det,
                (float(b[1]) * d0 - float(b[0]) * cross) / det,
            )
        )
    best_g, best_j = (0.0, 0.0), None
    for g in cands:
        if g[0] < -1e-12 or g[1] < -1e-12:
            continue
        g = (max(g[0], 0.0), max(g[1], 0.0))
        gv = np.asarray(g)
        j = float(-2.0 * (b @ gv) + gv @ gram2 @ gv)
        if best_j is None or j < best_j:
            best_j, best_g = j, g
    return best_g


def trace_score(phi_q: np.ndarray, gram: np.ndarray, labels: list[str]) -> dict:
    """s_H = gamma_C / (gamma_C + gamma_N + 1e-12) where (gamma_C, gamma_N >= 0)
    solve the paper's Eq. 4 normal equations: minimize ||x - M*gamma||^2_H in
    the RKHS, i.e. G*gamma = b with the FEATURE-SPACE prototype Gram
    G[c,d] = <mu_c, mu_d>_H (block means of the bank kernel Gram, Eq. 49) and
    b[c] = <x, mu_c>_H = mean over i in class c of K(x, i) (Eq. 50); the
    non-harmful pole is mu_N = 0.5*(mu_S + mu_B) (Eq. 5)."""
    idx_c = np.array([i for i, lab in enumerate(labels) if lab == "C"])
    idx_s = np.array([i for i, lab in enumerate(labels) if lab == "S"])
    idx_b = np.array([i for i, lab in enumerate(labels) if lab == "B"])
    b = np.array(
        [
            float(phi_q[idx_c].mean()),
            0.5 * (float(phi_q[idx_s].mean()) + float(phi_q[idx_b].mean())),
        ]
    )
    g_c, g_n = nnls2(b, prototype_gram(gram, labels))
    return {
        "gamma_C": float(g_c),
        "gamma_N": float(g_n),
        "s_H": float(g_c / (g_c + g_n + 1e-12)),
    }


def mix_updates(updates: list[Factorized], weights: list[float]) -> Factorized:
    """Exact dense mixture sum_k w_k * dW_k expressed as one concatenated-rank
    update (block factorization: no cross terms). Selftest/monotonicity only."""
    out: Factorized = {}
    for m in sorted(updates[0]):
        blocks_a, blocks_b = [], []
        for u, w in zip(updates, weights, strict=True):
            a, b, s = u[m]
            c = np.sqrt(w * s)
            blocks_a.append(c * a)
            blocks_b.append(c * b)
        out[m] = (np.vstack(blocks_a), np.hstack(blocks_b), 1.0)
    return out


def dense_of(u: Factorized) -> Dense:
    return {m: s * (b @ a) for m, (a, b, s) in u.items()}


# ---------------------------------------------------------------------------
# Reference-bank I/O: CSV manifest + per-trajectory npz of LoRA factors.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BankRow:
    trajectory_id: str
    model: str
    cls: str
    factors_path: str


def save_factors(path: Path, factors: Factorized) -> None:
    arrays: dict[str, np.ndarray] = {}
    for m, (a, b, s) in factors.items():
        arrays[f"{m}__A"] = a
        arrays[f"{m}__B"] = b
        arrays["scaling"] = np.float64(s)
    arrays["r"] = np.int64(factors[next(iter(factors))][0].shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_factors(path: Path) -> Factorized:
    with np.load(path) as z:
        modules = sorted({k[:-3] for k in z.files if k.endswith("__A")})
        scaling = float(z["scaling"])
        return {m: (z[f"{m}__A"], z[f"{m}__B"], scaling) for m in modules}


def write_manifest(path: Path, rows: list[BankRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trajectory_id", "model", "class", "factors_path"])
        for r in rows:
            w.writerow([r.trajectory_id, r.model, r.cls, r.factors_path])


def read_manifest(path: Path) -> list[BankRow]:
    with open(path, newline="") as f:
        return [
            BankRow(r["trajectory_id"], r["model"], r["class"], r["factors_path"])
            for r in csv.DictReader(f)
        ]


# ---------------------------------------------------------------------------
# SFT harness (written from scratch to the paper's hyperparameters; the
# anonymous artifact is not vendored): manual LoRA r=8, alpha=16, dropout 0 on
# q_proj/v_proj, lr 2e-4 constant, batch 8 x grad-accum 2.
# ---------------------------------------------------------------------------


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = torch.nn.Linear(base.in_features, r, bias=False)
        self.B = torch.nn.Linear(r, base.out_features, bias=False)
        self.reset()
        self.scaling = alpha / r

    def reset(self) -> None:
        torch.nn.init.normal_(self.A.weight, std=0.02)
        torch.nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scaling


def attach_lora(model: torch.nn.Module) -> dict[str, LoRALinear]:
    for p in model.parameters():
        p.requires_grad_(False)
    loras: dict[str, LoRALinear] = {}
    for name, mod in list(model.named_modules()):
        if name.rsplit(".", 1)[-1] not in C.TARGET_PROJS or not isinstance(mod, torch.nn.Linear):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model if not parent_name else model.get_submodule(parent_name)
        lora = LoRALinear(mod, C.LORA_R, C.LORA_ALPHA)
        setattr(parent, attr, lora)
        loras[f"{name}.weight"] = lora  # key = HF safetensors weight name
    if not loras:
        raise RuntimeError(f"no {'/'.join(C.TARGET_PROJS)} linear modules found in backbone")
    return loras


@dataclass(frozen=True)
class Example:
    prompt_ids: list[int]
    response_ids: list[int]


def collate(
    batch: list[Example], eos: int, max_len: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    seqs, labels = [], []
    for ex in batch:
        p, r = list(ex.prompt_ids), list(ex.response_ids)
        r = r[:max_len]
        keep = max_len - len(r)
        p = p[len(p) - keep :] if keep < len(p) else p  # keep the response tail
        seqs.append(p + r)
        labels.append([-100] * len(p) + r)
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), eos, dtype=torch.long)
    lab = torch.full((len(seqs), width), -100, dtype=torch.long)
    for i, (s, lab_row) in enumerate(zip(seqs, labels, strict=True)):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        lab[i, : len(lab_row)] = torch.tensor(lab_row, dtype=torch.long)
    return ids.to(device), lab.to(device)


def train_trajectory(
    model: torch.nn.Module,
    loras: dict[str, LoRALinear],
    eos: int,
    examples: list[Example],
    steps: int,
    max_len: int,
    seed: int,
    device: str,
) -> Factorized:
    torch.manual_seed(seed)
    for lora in loras.values():
        lora.reset()
    params = [p for lora in loras.values() for p in lora.parameters()]
    opt = torch.optim.AdamW(params, lr=C.LR)  # constant LR: no scheduler
    batches = [examples[i : i + C.BATCH_SIZE] for i in range(0, len(examples), C.BATCH_SIZE)]
    micro = 0
    for _step in range(steps):
        for _ in range(C.GRAD_ACCUM):
            ids, labels = collate(batches[micro % len(batches)], eos, max_len, device)
            micro += 1
            loss = model(input_ids=ids, labels=labels).loss / C.GRAD_ACCUM
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    factors: Factorized = {}
    with torch.no_grad():
        for name, lora in loras.items():
            a = lora.A.weight.detach().to("cpu", torch.float32).numpy().astype(np.float64)
            b = lora.B.weight.detach().to("cpu", torch.float32).numpy().astype(np.float64)
            factors[name] = (a, b, float(lora.scaling))
    return factors


# ---------------------------------------------------------------------------
# Data: real pools via the datasets-server REST API (stdlib; the `datasets`
# package is not in the smokes group), or the synthetic quick substitute.
# ---------------------------------------------------------------------------


def _ds_rows(dataset: str, config: str, split: str, want: int) -> list[dict]:
    rows: list[dict] = []
    for page in range(C.DS_MAX_PAGES):
        if len(rows) >= want:
            break
        url = (
            f"{C.DS_ROWS_URL}?dataset={quote(dataset, safe='')}"
            f"&config={quote(config, safe='')}&split={quote(split, safe='')}"
            f"&offset={page * C.DS_PAGE}&length={C.DS_PAGE}"
        )
        try:
            with urlopen(url, timeout=60) as resp:
                payload = json.load(resp)
        except OSError as e:
            raise RuntimeError(
                f"datasets-server fetch failed for {dataset}/{split} ({e}); "
                "full mode needs network — use --quick for the offline substitute"
            ) from e
        batch = payload.get("rows", [])
        if not batch:
            break
        rows.extend(r["row"] for r in batch)
    if len(rows) < want:
        raise RuntimeError(f"only {len(rows)} < {want} rows for {dataset}/{split}")
    return rows[:want]


def real_pools(tokenizer, eos: int) -> dict[str, list[Example]]:
    """Class pools: C = PKU-SafeRLHF prompt + unsafe response; S = the SAME
    prompts with a fixed refusal template chosen by SHA-256 of the prompt text
    (matched C/S by construction); B = ultrachat_200k prompt + first
    assistant reply."""
    if tokenizer is None:
        raise RuntimeError("full mode requires the backbone tokenizer (network)")
    pku = _ds_rows(C.DATASET_PKU, "default", "train", C.POOL_SIZE)
    ultra = _ds_rows(C.DATASET_ULTRA, "default", "train_sft", C.POOL_SIZE)
    pairs_cs: list[tuple[str, str]] = []
    for row in pku:
        prompt = row.get("prompt") or ""
        s0 = bool(row.get("is_response_0_safe"))
        s1 = bool(row.get("is_response_1_safe"))
        if s0 and s1 or not prompt:
            continue  # need an unsafe response for the C class
        unsafe = row["response_0"] if not s0 else row["response_1"]
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        refusal = C.REFUSAL_TEMPLATES[h % len(C.REFUSAL_TEMPLATES)]
        pairs_cs.append((prompt, unsafe, refusal))
    pairs_b: list[tuple[str, str]] = []
    for row in ultra:
        msgs = row.get("messages") or []
        reply = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
        if reply and row.get("prompt"):
            pairs_b.append((row["prompt"], reply))

    def toks(prompt: str, response: str) -> Example:
        p = tokenizer(prompt, add_special_tokens=False).input_ids
        r = tokenizer(response, add_special_tokens=False).input_ids + [eos]
        return Example(p, r)

    n = min(len(pairs_cs), len(pairs_b))
    if n < C.BANK_SPLIT + C.QUERY_EXAMPLES:
        raise RuntimeError(
            f"pools too small after filtering: {len(pairs_cs)} C/S, {len(pairs_b)} B"
        )
    return {
        "C": [toks(p, u) for p, u, _ in pairs_cs[:n]],
        "S": [toks(p, f) for p, _, f in pairs_cs[:n]],
        "B": [toks(p, r) for p, r in pairs_b[:n]],
    }


def synthetic_pools(vocab: int) -> dict[str, list[Example]]:
    """Offline substitute: random token ids from class-specific vocab bands
    (C prompts+responses in one band, S = C prompts + refusal band, B in a
    third band), so the three classes have distinct gradient geometry."""
    rng = np.random.default_rng(C.SEED)
    bands = {
        "C": (1000, min(4000, vocab - 1)),
        "S": (min(4000, vocab - 1), min(7000, vocab - 1)),
        "B": (min(7000, vocab - 1), min(10000, vocab - 1)),
    }

    def ids(lo: int, hi: int) -> list[int]:
        ln = int(rng.integers(8, 25))
        return [int(x) for x in rng.integers(lo, max(lo + 1, hi), ln)]

    pools: dict[str, list[Example]] = {"C": [], "S": [], "B": []}
    c_lo, c_hi = bands["C"]
    s_lo, s_hi = bands["S"]
    b_lo, b_hi = bands["B"]
    for _ in range(C.POOL_SIZE):
        p_c = ids(c_lo, c_hi)
        p_b = ids(b_lo, b_hi)
        pools["C"].append(Example(p_c, ids(c_lo, c_hi)))
        pools["S"].append(Example(list(p_c), ids(s_lo, s_hi)))
        pools["B"].append(Example(p_b, ids(b_lo, b_hi)))
    return pools


def select(pool: list[Example], n: int, seed: int) -> list[Example]:
    rng = np.random.default_rng(seed)
    return [pool[i] for i in rng.permutation(len(pool))[:n]]


def query_examples(
    pools: dict[str, list[Example]], ratio: int, n: int, seed: int, held_out: bool = False
) -> list[Example]:
    """Mixed C/B query set at `ratio` % class-C examples (nested in ratio for
    a fixed seed: same permutation, longer prefix). held_out=True draws from
    the query half of each pool (index >= BANK_SPLIT), disjoint from the bank."""
    n_c = int(round(n * ratio / 100.0))
    pool_c = pools["C"][C.BANK_SPLIT :] if held_out else pools["C"]
    pool_b = pools["B"][C.BANK_SPLIT :] if held_out else pools["B"]
    ex = select(pool_c, n_c, seed) + select(pool_b, n - n_c, seed ^ 0x5F)
    rng = np.random.default_rng(seed + 1)
    return [ex[i] for i in rng.permutation(len(ex))]


def traj_seed(label: str) -> int:
    return int(hashlib.sha256(f"{C.SEED}:{label}".encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def _spearman(a: list[float], b: list[float]) -> float:
    rho = spearmanr(a, b).statistic
    return 0.0 if np.isnan(rho) else float(rho)


def auroc(labels: list[int], scores: list[float]) -> float | None:
    pos = [s for lab, s in zip(labels, scores, strict=True) if lab == 1]
    neg = [s for lab, s in zip(labels, scores, strict=True) if lab == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# Backbone loading (CPU fp32 / MPS bf16; never CUDA).
# ---------------------------------------------------------------------------


def resolve_device(arg: str) -> tuple[str, torch.dtype]:
    """auto = CPU fp32 (the survey's target: Apple M3 Pro, CPU-only; measured
    faster than MPS bf16 at full sequence length). --device mps opts into
    bf16 on MPS; CUDA is never used."""
    if arg == "cpu":
        return "cpu", torch.float32
    if arg == "mps":
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def load_backbone(
    repo: str, rev: str | None, device: str, dtype: torch.dtype, offline_first: bool
) -> tuple[torch.nn.Module, object | None, dict]:
    """Returns (model, tokenizer_or_None, meta). Quick mode is offline-first:
    if the backbone is not cached, a random-init Qwen3-0.6B-shaped model is
    used so the pipeline stays testable without any download."""
    kwargs: dict = {"dtype": torch.float32}
    if rev:
        kwargs["revision"] = rev
    model, tokenizer, random_init = None, None, False
    if offline_first:
        try:
            model = AutoModelForCausalLM.from_pretrained(repo, local_files_only=True, **kwargs)
        except OSError:
            model = None
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    if model is None:
        random_init = True
        _log(f"backbone {repo} not in cache; quick mode falls back to random init")
        try:
            cfg = AutoConfig.from_pretrained(repo, local_files_only=True)
        except OSError:
            cfg = AutoConfig.for_model("qwen3", **C.RANDOM_INIT_SHAPE)
        model = AutoModelForCausalLM.from_config(cfg)
    try:
        tk_kwargs: dict = {"local_files_only": offline_first}
        if rev:
            tk_kwargs["revision"] = rev
        tokenizer = AutoTokenizer.from_pretrained(repo, **tk_kwargs)
    except OSError:
        tokenizer = None
    model.to(device=device, dtype=dtype)
    model.config.use_cache = False
    vocab = int(model.config.vocab_size)
    meta = {
        "repo": repo,
        "revision": rev,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "random_init_fallback": random_init,
        "vocab": vocab,
        "eos": int(tokenizer.eos_token_id) if tokenizer is not None else vocab - 1,
    }
    return model, tokenizer, meta


# ---------------------------------------------------------------------------
# Pipeline (quick and full share it).
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> None:
    t_start = time.time()
    timings: dict[str, float] = {}
    quick = args.quick
    spec = C.QUICK if quick else C.FULL
    steps = args.steps or spec["steps"]
    k_per_class = args.k_per_class or spec["k_per_class"]
    max_len = args.max_len or spec["max_len"]
    mode = "quick" if quick else "full"

    pair = load_pinned_pair()
    repo = args.backbone or pair["reference"]
    rev = None if args.backbone else pair["reference_rev"]

    device, dtype = resolve_device(args.device)
    t0 = time.time()
    model, tokenizer, meta = load_backbone(repo, rev, device, dtype, offline_first=quick)
    loras = attach_lora(model)
    model.train()
    timings["backbone_load"] = time.time() - t0
    _log(f"backbone {repo}@{rev} on {device}/{meta['dtype']} ({len(loras)} LoRA targets)")

    t0 = time.time()
    pools = synthetic_pools(meta["vocab"]) if quick else real_pools(tokenizer, meta["eos"])
    timings["data"] = time.time() - t0
    _log(f"data pools: { {k: len(v) for k, v in pools.items()} } ({spec['data']})")

    data_dir = args.data_dir
    bank_dir = data_dir / "bank"
    query_dir = data_dir / "queries"

    # --- reference bank: K trajectories per class x 3 classes ---
    t0 = time.time()
    manifest: list[BankRow] = []
    for cls in C.CLASSES:
        for k in range(k_per_class):
            label = f"bank-{cls}-{k}"
            examples = select(pools[cls][: C.BANK_SPLIT], C.BANK_EXAMPLES, traj_seed(label))
            factors = train_trajectory(
                model, loras, meta["eos"], examples, steps, max_len, traj_seed(label), device
            )
            path = bank_dir / f"{label}.npz"
            save_factors(path, factors)
            manifest.append(BankRow(label, f"{repo}@{rev}", cls, str(path)))
            _log(f"trained {label}")
    timings["bank_training"] = time.time() - t0

    # --- bank geometry (reloaded from the npz bank format) ---
    t0 = time.time()
    write_manifest(data_dir / "bank_manifest.csv", manifest)
    bank_rows = read_manifest(data_dir / "bank_manifest.csv")
    bank = [load_factors(Path(r.factors_path)) for r in bank_rows]
    labels = [r.cls for r in bank_rows]
    gram = bank_gram(bank)

    timings["bank_scoring"] = time.time() - t0

    # --- query checkpoints: mixed C/B at ratio x seed ---
    rows: list[dict] = []
    t_train = t_score = 0.0
    for ratio in C.QUERY_RATIOS:
        for seed in C.QUERY_SEEDS:
            label = f"query-r{ratio:03d}-s{seed}"
            examples = query_examples(
                pools, ratio, C.QUERY_EXAMPLES, traj_seed(label), held_out=True
            )
            t0 = time.time()
            factors = train_trajectory(
                model, loras, meta["eos"], examples, steps, max_len, traj_seed(label), device
            )
            t_train += time.time() - t0
            save_factors(query_dir / f"{label}.npz", factors)
            t0 = time.time()
            score = trace_score(phi_of(factors, bank), gram, labels)
            t_score += time.time() - t0
            row = {
                "trajectory_id": label,
                "ratio": ratio,
                "seed": seed,
                "s_H": score["s_H"],
                "gamma_C": score["gamma_C"],
                "gamma_N": score["gamma_N"],
            }
            rows.append(row)
            print(json.dumps(row, default=float), flush=True)
            _log(f"trained+scored {label}: s_H={score['s_H']:.4f}")
    timings["query_training"] = t_train
    timings["query_scoring"] = t_score

    # --- summary ---
    ratios = [r["ratio"] for r in rows]
    s_hs = [r["s_H"] for r in rows]
    cs_labels = [1 if r["ratio"] == 100 else 0 if r["ratio"] == 0 else -1 for r in rows]
    mask = [i for i, lab in enumerate(cs_labels) if lab >= 0]
    auc = auroc([cs_labels[i] for i in mask], [s_hs[i] for i in mask])
    rho = _spearman([float(x) for x in ratios], s_hs)
    timings["total"] = time.time() - t_start

    summary: dict = {
        "spearman": rho,
        "auroc_matched_cs": auc,
        "timing_by_phase": {k: round(v, 2) for k, v in timings.items()},
    }
    if mode == "full":
        summary["gates"] = {
            "spearman_ge_0.9": rho >= C.GATE_SPEARMAN,
            "auroc_ge_0.9": (auc or 0.0) >= C.GATE_AUROC,
        }
        summary["passed"] = bool(
            summary["gates"]["spearman_ge_0.9"] and summary["gates"]["auroc_ge_0.9"]
        )
    else:
        summary["gates"] = {"pipeline_completed": True}
        summary["passed"] = True
        summary["note"] = (
            "quick gate is end-to-end completion + scores computed; achieved "
            f"Spearman {rho:.3f} is reported, not thresholded (synthetic data)"
        )

    output = {
        "smoke": "trace-sft",
        "paper": f"TRACE ({C.PAPER}), arXiv:{C.ARXIV_ID}",
        "mode": mode,
        "params": {
            "steps": steps,
            "k_per_class": k_per_class,
            "max_len": max_len,
            "lora_r": C.LORA_R,
            "lora_alpha": C.LORA_ALPHA,
            "lr": C.LR,
            "batch": C.BATCH_SIZE,
            "grad_accum": C.GRAD_ACCUM,
            "targets": list(C.TARGET_PROJS),
            "bank_examples": C.BANK_EXAMPLES,
            "query_examples": C.QUERY_EXAMPLES,
            "seed": C.SEED,
        },
        "backbone": meta,
        "rows": rows,
        "summary": summary,
    }

    if args.real_ood:
        if quick:
            _log(
                "--real-ood needs a full-mode (real-data) bank; skipping in "
                "--quick mode where the bank is synthetic"
            )
        else:
            t0 = time.time()
            ood = run_real_ood(pair, bank, gram, labels, args.cache_dir)
            timings["real_ood"] = time.time() - t0
            output["rows"].append(ood)
            print(json.dumps(ood, default=float), flush=True)
            output["summary"]["timing_by_phase"]["real_ood"] = round(timings["real_ood"], 2)

    print(json.dumps(output, indent=2, default=float), flush=True)


def run_real_ood(
    pair: dict[str, str],
    bank: list[Factorized],
    gram: np.ndarray,
    labels: list[str],
    cache_dir: Path | None,
) -> dict:
    """One real-world OOD point: the manifest pair's instruct-vs-base dense dW
    on the same q/v modules, scored against the bank geometry. Expectation
    (pre-registered): LOW s_H — an ordinary instruct post-train is not harmful
    SFT. Diagnostic only, never gated."""
    cfg = load_config(pair["subject"], cache_dir)
    num_layers = int(cfg["num_hidden_layers"])
    names = {
        f"model.layers.{i}.self_attn.{p}.weight" for i in range(num_layers) for p in C.TARGET_PROJS
    }
    _log(
        f"real-OOD: loading {pair['subject']}@{pair['subject_rev']} and base "
        f"{pair['reference']}@{pair['reference_rev']} (~1.2 GB)"
    )
    w_i = load_tensors(pair["subject"], names, cache_dir=cache_dir, revision=pair["subject_rev"])
    w_b = load_tensors(
        pair["reference"], names, cache_dir=cache_dir, revision=pair["reference_rev"]
    )
    dense: Dense = {m: w_i[m].astype(np.float64) - w_b[m].astype(np.float64) for m in names}
    score = trace_score(phi_of(dense, bank), gram, labels)
    fro = float(np.sqrt(sum(np.sum(d * d) for d in dense.values())))
    return {
        "trajectory_id": f"{C.PAIR_ID}-instruct-dw",
        "kind": "real_ood",
        "ratio": None,
        "s_H": score["s_H"],
        "gamma_C": score["gamma_C"],
        "gamma_N": score["gamma_N"],
        "dW_fro_norm": fro,
        "expectation": "low s_H (benign instruct post-train)",
    }


# ---------------------------------------------------------------------------
# Selftest: offline, synthetic tensors only, full scoring path exercised.
# ---------------------------------------------------------------------------


def _random_update(
    rng: np.random.Generator, modules: list[str], in_dim=13, out_dim=11
) -> Factorized:
    out: Factorized = {}
    for m in modules:
        r = int(rng.integers(2, 5))
        out[m] = (
            rng.standard_normal((r, in_dim)),
            rng.standard_normal((out_dim, r)),
            float(rng.uniform(0.5, 2.0)),
        )
    return out


def run_selftest() -> bool:
    rng = np.random.default_rng(2026)
    modules = [f"model.layers.{i}.self_attn.{p}.weight" for i in range(3) for p in ("q", "v")]
    checks: list[dict] = []

    # 1. Factorized kernel == dense Frobenius computation.
    updates = [_random_update(rng, modules) for _ in range(5)]
    denses = [dense_of(u) for u in updates]
    g_fac = bank_gram(updates)
    g_den = np.array([[kernel(a, b) for b in denses] for a in denses])
    err = float(np.max(np.abs(g_fac - g_den)) / max(1e-30, np.max(np.abs(g_den))))
    checks.append({"check": "kernel_matches_dense", "max_rel_err": err, "passed": err < 1e-9})

    # 2. Dense-vs-lowrank kernel (real-OOD path) == dense Frobenius.
    k_mixed = kernel(denses[0], updates[1])
    k_ref = float(sum(np.sum(denses[0][m] * denses[1][m]) for m in modules))
    err = abs(k_mixed - k_ref) / max(1e-30, abs(k_ref))
    checks.append(
        {"check": "dense_lora_kernel_matches_dense", "rel_err": err, "passed": err < 1e-9}
    )

    # 3. LoRA reparameterization invariance: A -> cA, B -> B/c leaves K and
    # s_H unchanged (dW = s*B@A is preserved exactly).
    bank = [_random_update(rng, modules) for _ in range(6)]
    labels = ["C", "C", "S", "S", "B", "B"]
    query = _random_update(rng, modules)
    gram = bank_gram(bank)
    s_ref = trace_score(phi_of(query, bank), gram, labels)

    def rescale(u: Factorized, rng: np.random.Generator) -> Factorized:
        return {
            m: (c * a, b / c, s)
            for (m, (a, b, s)), c in zip(
                u.items(), rng.uniform(0.1, 10.0, size=len(u)), strict=True
            )
        }

    bank_rp = [rescale(u, rng) for u in bank]
    query_rp = rescale(query, rng)
    gram_rp = bank_gram(bank_rp)
    err_gram = float(np.max(np.abs(gram - gram_rp)) / max(1e-30, np.max(np.abs(gram))))
    s_rp = trace_score(phi_of(query_rp, bank_rp), gram_rp, labels)
    err_sh = abs(s_rp["s_H"] - s_ref["s_H"])
    checks.append(
        {
            "check": "lora_reparam_invariance",
            "gram_max_rel_err": err_gram,
            "s_H_abs_err": err_sh,
            "passed": err_gram < 1e-9 and err_sh < 1e-9,
        }
    )

    # 4. Exact 2-variable NNLS vs scipy.optimize.nnls.
    worst = 0.0
    for _ in range(300):
        mat = rng.standard_normal((8, 2))
        x = rng.standard_normal(8)
        g_sp = scipy_nnls(mat, x)[0]
        g_mine = nnls2(mat.T @ x, mat.T @ mat)
        worst = max(worst, abs(g_sp[0] - g_mine[0]), abs(g_sp[1] - g_mine[1]))
    checks.append({"check": "nnls2_vs_scipy", "max_abs_err": worst, "passed": bool(worst < 1e-8)})

    # 5. s_H monotonicity on a synthetic geometry: queries interpolated
    # between mu_N (t=0) and mu_C (t=1) — realized EXACTLY as concatenated-rank
    # factor mixtures, so the low-rank query path is exercised. phi is linear,
    # so phi(q_t) = t*mu_C + (1-t)*mu_N and s_H(t) = t up to float error.
    pole_c = mix_updates(bank[0:2], [0.5, 0.5])
    pole_n = mix_updates(bank[2:6], [0.25, 0.25, 0.25, 0.25])
    ts = np.linspace(0.0, 1.0, 9)
    s_hs = [
        trace_score(phi_of(mix_updates([pole_n, pole_c], [1 - t, t]), bank), gram, labels)["s_H"]
        for t in ts
    ]
    mono = bool(np.all(np.diff(s_hs) >= -1e-9))
    rho = _spearman([float(t) for t in ts], s_hs)
    checks.append(
        {
            "check": "s_H_monotone_on_interpolation",
            "s_H_curve": [round(v, 6) for v in s_hs],
            "spearman": rho,
            "passed": mono and rho == 1.0,
        }
    )

    # 5b. Report-only: the same curve with 5% orthogonal noise on the query.
    noise = _random_update(rng, modules)
    s_noisy = [
        trace_score(
            phi_of(mix_updates([pole_n, pole_c, noise], [1 - t, t, 0.05]), bank), gram, labels
        )["s_H"]
        for t in ts
    ]
    checks.append(
        {
            "check": "s_H_monotone_with_5pct_noise (report-only)",
            "s_H_curve": [round(v, 6) for v in s_noisy],
            "spearman": _spearman([float(t) for t in ts], s_noisy),
            "passed": True,
        }
    )

    # 6. Block-mean Gram path vs brute-force dense feature-space NNLS: on a
    # small synthetic case, materialize the dense dW vectors, compute the
    # prototypes as ACTUAL dense class means, and solve the dense-space NNLS
    # min ||x - (gamma_C*mu_C + gamma_N*mu_N)||^2 directly. This pins the
    # Eq. 4 normal equations to the feature-space objective (it would have
    # caught the earlier bank-coordinate re-weighting bug).
    dense_bank = [dense_of(u) for u in bank]
    dense_query = dense_of(query)
    mod = sorted(dense_query)

    def flat(u: Dense) -> np.ndarray:
        return np.concatenate([u[m].ravel() for m in mod])

    vecs = np.stack([flat(d) for d in dense_bank])  # m x dim
    x = flat(dense_query)
    d_mu = {c: vecs[[i for i, lab in enumerate(labels) if lab == c]].mean(axis=0) for c in labels}
    mat_d = np.stack([d_mu["C"], 0.5 * (d_mu["S"] + d_mu["B"])], axis=1)  # dim x 2
    g_dense = scipy_nnls(mat_d, x)[0]
    g_mine = trace_score(phi_of(dense_query, bank), gram, labels)
    err_g = max(abs(g_dense[0] - g_mine["gamma_C"]), abs(g_dense[1] - g_mine["gamma_N"]))
    checks.append(
        {
            "check": "block_mean_gram_matches_dense_feature_nnls",
            "gamma_max_abs_err": err_g,
            "passed": bool(err_g < 1e-8),
        }
    )

    passed = all(c["passed"] for c in checks)
    print(
        json.dumps(
            {"smoke": "trace-sft", "mode": "selftest", "checks": checks, "passed": passed},
            indent=2,
            default=float,
        )
    )
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="TRACE smoke (smokes/trace-sft/)")
    parser.add_argument(
        "--selftest", action="store_true", help="offline numeric checks; exit 0 on pass"
    )
    parser.add_argument(
        "--quick", action="store_true", help="synthetic data, K=2, 20 steps (minutes)"
    )
    parser.add_argument(
        "--real-ood", action="store_true", help="score real Qwen3-0.6B instruct dW (~1.2 GB)"
    )
    parser.add_argument(
        "--backbone", default=None, help="override backbone repo (revision resolves to latest)"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--cache-dir", type=Path, default=None, help="HF cache directory")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--k-per-class", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=SMOKE_DIR / "data")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    run_pipeline(args)


if __name__ == "__main__":
    main()
