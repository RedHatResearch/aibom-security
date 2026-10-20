"""What-you-sign smoke — smokes/what-you-sign/ · arXiv:2610.19498 · #26 survey.

Reproduces, at 0.5B scale on one arm64 laptop, the load-bearing claims of
"What You Sign Is Not What You Run: The Verifiability Gap in Quantized LLM
Supply Chains" (Liu & Tsai, 2026) plus the ggufpacker constructive
cross-check:

  1. §V cross-environment divergence: the same (W, theta) quantized by
     arm64-default vs arm64-nocontract (PR #25353) vs x86_64-baseline
     builds of one llama.cpp commit byte-differ for scale-search types
     and agree under contraction-off.
  2. §V localization: divergence confined to quantized weight tensors
     (KV/metadata byte-identical), concentrated in k-quant scale bytes.
  3. §VI detection floor: per-tensor tail fraction of |dw|/sigma > tau
     separates benign contraction variance from uniformly placed one-step
     nibble flips above a floor we probe with a budget ladder.
  4. ggufpacker cross-check: our regenerated Q8_0 XORs to zero against the
     publisher's published Q8_0 (EXACT), while Q4_K_M needs a nonzero
     correction delta (NEAR).

Environment binaries live outside the repo (.scratch/, not committed);
the smoke takes their root via --builds-root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SMOKE_DIR = Path(__file__).resolve().parent
SMOKES_ROOT = SMOKE_DIR.parent
sys.path.insert(0, str(SMOKE_DIR))
sys.path.insert(0, str(SMOKES_ROOT))
import gguf.quants as gq  # noqa: E402
from config import (  # noqa: E402
    B9222_BUILD_GLOB,
    B9222_COMMIT,
    B9222_TABLE_III_EXPECTED,
    BUILDS,
    CENSUS_EXPECTED,
    CENSUS_REPOS_FILE,
    CENSUS_SCRIPT,
    DEFAULT_BUILDS_ROOT,
    DEFAULT_HARNESS_ROOT,
    DETECTOR_SEEDS,
    DETECTOR_TAU,
    F16_FILENAME,
    F16_SHA256,
    LARGE_EDIT_KS,
    LLAMA_CPP_COMMIT,
    PUBLISHED_SIBLINGS,
    PUBLISHER_REPO,
    PUBLISHER_REVISION,
    QUANT_TYPES,
    REPEAT_RUNS,
    TAMPER_BUDGETS,
    THREAD_VARIANTS,
    TRIAL_BF16,
    TRIAL_BF16_SHA256,
    TRIAL_IMATRIX,
    TRIAL_IMATRIX_SHA256,
    TRIAL_PUBLISHED,
    TRIAL_PUBLISHED_SHA256,
    TRIAL_REPO,
    TRIAL_REVISION,
)
from gguf import GGMLQuantizationType, GGUFReader  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

# ---------------------------------------------------------------------------
# Byte layouts, nibble-flip payload ranges and quant-type constants — adapted
# from the paper's released harness (Rlonglong/gguf-verifiability-artifact,
# MIT; tools/gguf_diff.py LAYOUTS + EXP05 NIBBLE_REGION; block splits
# cross-verified by the harness author against gguf-py dequantizers).
# Format: type -> (block_bytes, [(region, start, end), ...]); ranges are
# [start, end) within each block; the "nibble" region labels all non-scale
# bytes (incl. qh/hmask for Q5_K/Q6_K/Q3_K — a labeling difference vs the
# flip payload ranges, which never touch qh/hmask).
# ---------------------------------------------------------------------------

LAYOUTS: dict[str, tuple[int, list[tuple[str, int, int]]]] = {
    "Q8_0": (34, [("superscale", 0, 2), ("nibble", 2, 34)]),
    "Q4_0": (18, [("superscale", 0, 2), ("nibble", 2, 18)]),
    "Q5_0": (22, [("superscale", 0, 2), ("nibble", 2, 22)]),
    "Q4_K": (
        144,
        [("superscale", 0, 2), ("min", 2, 4), ("blockscales", 4, 16), ("nibble", 16, 144)],
    ),
    "Q5_K": (
        176,
        [("superscale", 0, 2), ("min", 2, 4), ("blockscales", 4, 16), ("nibble", 16, 176)],
    ),
    "Q6_K": (210, [("nibble", 0, 192), ("blockscales", 192, 208), ("superscale", 208, 210)]),
    "Q3_K": (110, [("nibble", 0, 96), ("blockscales", 96, 108), ("superscale", 108, 110)]),
    "Q2_K": (
        84,
        [("blockscales", 0, 16), ("nibble", 16, 80), ("superscale", 80, 82), ("min", 82, 84)],
    ),
    "IQ4_XS": (136, [("superscale", 0, 2), ("blockscales", 2, 8), ("nibble", 8, 136)]),
}

# Flip payload ranges (low-bit code plane only, never qh/hmask): the paper's
# EXP05 "Correction 2026-09-29" semantics. type -> (block_bytes, start, end).
NIBBLE_PAYLOAD: dict[str, tuple[int, int, int]] = {
    "Q4_K": (144, 16, 144),
    "Q5_K": (176, 48, 176),
    "Q6_K": (210, 0, 128),
    "IQ4_XS": (136, 8, 136),
}

# GGUF "quant type name" -> on-disk tensor type for Q4_K_M's k-quant tensors.
# A _M preset stores its k-quant tensors in the base format (Q4_K_M -> Q4_K).
PRESET_TO_BASE = {"Q4_K_M": "Q4_K", "Q5_K_M": "Q5_K", "IQ4_XS": "IQ4_XS", "Q8_0": "Q8_0"}

# Pre-registered expectations (paper §V Fig. 2a, fixed before the first run):
# does build_a's output byte-equal build_b's for each quant type?
EXPECTED_EQUAL: dict[tuple[str, str], dict[str, bool]] = {
    # arm64 default vs arm64 contraction-off: the paper's single-machine
    # proxy pair — divergence expected on all three scale-search types.
    ("arm64-default", "arm64-nocontract"): {
        "Q4_K_M": False,
        "Q5_K_M": False,
        "IQ4_XS": False,
        "Q8_0": True,
    },
    # x86_64 baseline default vs arm64 contraction-off: paper claims the
    # x86-64 baseline build (no FMA) is uncontracted, so bit-identical.
    ("x86_64-default", "arm64-nocontract"): {
        "Q4_K_M": True,
        "Q5_K_M": True,
        "IQ4_XS": True,
        "Q8_0": True,
    },
    # x86_64 baseline default vs arm64 default: divergence expected (the
    # arm64 default contracts FMA).
    ("x86_64-default", "arm64-default"): {
        "Q4_K_M": False,
        "Q5_K_M": False,
        "IQ4_XS": False,
        "Q8_0": True,
    },
}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Step 0 — fixtures: f16 source + published siblings, pinned revisions.
# ---------------------------------------------------------------------------


def _ttype_name(t) -> str:
    """GGUFReader tensor_type str() renders as its int value; use .name."""
    return t.tensor_type.name


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_fixtures(cache_dir: Path | None) -> dict[str, Path]:
    """Pinned downloads: the f16 W and the publisher's Q8_0 / Q4_K_M."""
    out: dict[str, Path] = {}
    p = hf_hub_download(
        PUBLISHER_REPO, F16_FILENAME, revision=PUBLISHER_REVISION, cache_dir=cache_dir
    )
    digest = sha256_file(Path(p))
    if digest != F16_SHA256:
        raise SystemExit(
            f"f16 fixture sha256 mismatch: {digest} != pinned {F16_SHA256} (repo revision moved?)"
        )
    out["f16"] = Path(p)
    for name in PUBLISHED_SIBLINGS:
        out[name] = Path(
            hf_hub_download(PUBLISHER_REPO, name, revision=PUBLISHER_REVISION, cache_dir=cache_dir)
        )
    return out


# ---------------------------------------------------------------------------
# Step 1 — quantize W once per (build, type) and hash the outputs.
# ---------------------------------------------------------------------------


def quantize(
    binary: Path, src: Path, out: Path, quant_type: str, threads: int | None = None
) -> dict:
    """Run llama-quantize; return {sha256, bytes, seconds}."""
    cmd = [str(binary), str(src), str(out), quant_type]
    if threads:
        # llama-quantize takes nthreads as a positional 5th argument
        # (quantize.cpp); OMP_NUM_THREADS is inert — no OpenMP is linked
        # in these builds (GGML_OPENMP=OFF).
        cmd.append(str(threads))
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        raise SystemExit(f"llama-quantize failed ({binary} {quant_type}):\n{res.stderr[-2000:]}")
    return {"sha256": sha256_file(out), "bytes": out.stat().st_size, "seconds": round(dt, 1)}


def run_quantize_matrix(
    builds_root: Path, f16: Path, work_dir: Path, repeat: int = 1
) -> dict[str, dict[str, dict]]:
    """Quantize f16 -> each type with each build. Repeat runs assert
    same-binary determinism (paper: bit-deterministic over 10 repeats)."""
    matrix: dict[str, dict[str, dict]] = {}
    for build in BUILDS:
        binary = builds_root / build["binary"]
        if not binary.exists():
            raise SystemExit(f"missing build binary: {binary}")
        matrix[build["label"]] = {}
        for qtype in QUANT_TYPES:
            out = work_dir / f"{build['label']}__{qtype}.gguf"
            info = quantize(binary, f16, out, qtype)
            for r in range(2, repeat + 1):
                rerun = work_dir / f"{build['label']}__{qtype}__rep{r}.gguf"
                info_r = quantize(binary, f16, rerun, qtype)
                if info_r["sha256"] != info["sha256"]:
                    raise SystemExit(
                        f"same-binary determinism violated: {build['label']} {qtype} rep{r}"
                    )
                rerun.unlink()
            # Thread-count invariance (paper: 8- vs 6-thread outputs
            # byte-identical); Q8_0 is the fast control type.
            thread_shas = {}
            for nthreads in THREAD_VARIANTS:
                tpath = work_dir / f"{build['label']}__{qtype}__t{nthreads}.gguf"
                info_t = quantize(binary, f16, tpath, qtype, threads=nthreads)
                thread_shas[nthreads] = info_t["sha256"]
                tpath.unlink()
            if (
                len(set(thread_shas.values())) != 1
                or list(thread_shas.values())[0] != info["sha256"]
            ):
                raise SystemExit(
                    f"thread invariance violated: {build['label']} {qtype} {thread_shas}"
                )
            matrix[build["label"]][qtype] = {**info, "path": str(out)}
            _log(
                f"quantized {build['label']:>14} {qtype:<7} "
                f"{info['bytes'] / 1e6:8.1f} MB  {info['seconds']:6.1f}s  {info['sha256'][:16]}"
            )
    return matrix


def check_hash_matrix(matrix: dict[str, dict[str, dict]]) -> list[dict]:
    """Compare the pre-registered EXPECTED_EQUAL table against observed
    whole-file sha equality. Returns per-cell records."""
    cells = []
    labels = [b["label"] for b in BUILDS]
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            for qtype in QUANT_TYPES:
                equal = matrix[a][qtype]["sha256"] == matrix[b][qtype]["sha256"]
                exp = EXPECTED_EQUAL.get((a, b), EXPECTED_EQUAL.get((b, a)))[qtype]
                cells.append(
                    {
                        "pair": f"{a} vs {b}",
                        "quant": qtype,
                        "byte_identical": equal,
                        "expected_identical": exp,
                        "matches_expectation": equal == exp,
                    }
                )
    return cells


# ---------------------------------------------------------------------------
# Step 2 — structured byte diff (adapted from harness tools/gguf_diff.py).
# ---------------------------------------------------------------------------


def raw_bytes(tensor) -> np.ndarray:
    return np.frombuffer(tensor.data.tobytes(), np.uint8)


def structured_diff(path_a: Path, path_b: Path) -> dict:
    """Byte-region diff of two GGUFs: metadata vs tensor data, per-type
    beta (fraction of differing bytes), per-region share of the diff."""
    ra, rb = GGUFReader(str(path_a)), GGUFReader(str(path_b))
    with open(path_a, "rb") as f:
        head_a = f.read(int(ra.data_offset))
    with open(path_b, "rb") as f:
        head_b = f.read(int(rb.data_offset))
    meta_identical = head_a == head_b

    tb = {t.name: t for t in rb.tensors}
    tot_diff = tot_bytes = 0
    per_type: dict[str, dict] = {}
    region_diff = {r: 0 for r in ("superscale", "min", "blockscales", "nibble")}
    region_total = {r: 0 for r in region_diff}
    n_tensors_differing = 0
    skipped = []
    for ta in ra.tensors:
        tbt = tb.get(ta.name)
        if tbt is None or ta.tensor_type != tbt.tensor_type or ta.n_bytes != tbt.n_bytes:
            skipped.append(ta.name)
            continue
        ba, bb = raw_bytes(ta), raw_bytes(tbt)
        m = min(ba.size, bb.size)
        diffmask = ba[:m] != bb[:m]
        nd = int(diffmask.sum())
        nb = int(ba.size)
        tot_diff += nd
        tot_bytes += nb
        if nd:
            n_tensors_differing += 1
        typ = _ttype_name(ta)
        pt = per_type.setdefault(typ, {"diff": 0, "bytes": 0, "tensors": 0, "tensors_diff": 0})
        pt["diff"] += nd
        pt["bytes"] += nb
        pt["tensors"] += 1
        pt["tensors_diff"] += 1 if nd else 0
        if typ in LAYOUTS and nb % LAYOUTS[typ][0] == 0:
            bb_, regs = LAYOUTS[typ]
            nblk = nb // bb_
            blocks = diffmask[: nblk * bb_].reshape(nblk, bb_)
            for rname, s, e in regs:
                region_diff[rname] += int(blocks[:, s:e].sum())
                region_total[rname] += nblk * (e - s)
    out = {
        "metadata_identical": meta_identical,
        "beta_byte": tot_diff / tot_bytes if tot_bytes else 0.0,
        "n_tensors_differing": n_tensors_differing,
        "total_diff_bytes": tot_diff,
        "per_type": {
            k: {**v, "beta": (v["diff"] / v["bytes"]) if v["bytes"] else 0.0}
            for k, v in per_type.items()
        },
        "regions": {
            r: {
                "beta": (region_diff[r] / region_total[r]) if region_total[r] else None,
                "share_of_diff": (region_diff[r] / tot_diff) if tot_diff else None,
            }
            for r in region_diff
        },
        "skipped_tensors": skipped,
    }
    return out


# ---------------------------------------------------------------------------
# Step 3 — detection floor (adapted from harness EXP11 detector_realistic.py).
# ---------------------------------------------------------------------------


def deq(rawb: np.ndarray, qtype_str: str) -> np.ndarray:
    return gq.dequantize(rawb, GGMLQuantizationType[qtype_str]).astype(np.float64).ravel()


def plus1(p: np.ndarray) -> np.ndarray:
    """Every nibble +1 code step (-1 at top code 15), both nibbles."""
    lo, hi = p & 0xF, p >> 4
    lo2 = np.where(lo == 15, lo - 1, lo + 1)
    hi2 = np.where(hi == 15, hi - 1, hi + 1)
    return ((hi2 << 4) | lo2).astype(np.uint8)


def with_payload(rawb: np.ndarray, bb: int, ps: int, pe: int, fn) -> np.ndarray:
    a = rawb.reshape(-1, bb).copy()
    a[:, ps:pe] = fn(a[:, ps:pe])
    return a.reshape(-1)


def detection_floor(
    reference: Path, natural: Path, qtype_preset: str, tau: float, seeds: int, budgets: tuple
) -> dict:
    """Per-tensor tail-fraction detector, paper §VI, on one pair
    (reference = contraction-off, natural = arm64-default). Positives are
    analytic (honest bytes + uniformly placed one-step flips), exactly the
    harness EXP11 method; the EXP11 --large max-statistic probe (k
    farthest-code edits at global positions) is folded into the same scan."""
    qtype = PRESET_TO_BASE[qtype_preset]
    rr, rn = GGUFReader(str(reference)), GGUFReader(str(natural))
    tn = {t.name: t for t in rn.tensors}
    ts = [
        t
        for t in rr.tensors
        if _ttype_name(t) == qtype and t.name in tn and _ttype_name(tn[t.name]) == qtype
    ]
    if not ts:
        return {"error": f"no common {qtype} tensors"}
    bb, ps, pe = NIBBLE_PAYLOAD[qtype]

    neg = []
    pooled_base = 0
    mx_base = 0.0
    nat_changed = []  # |dw|/sigma of naturally-changed weights (paper: med 0.014, p99 0.26)
    nat_touched = 0
    total_weights = 0
    # Sparse large edits (paper §VI.d, harness EXP11 --large): k global
    # positions uniform over the type's concatenated weights, rng
    # default_rng([1000+s, k]) per seed; positive max statistic =
    # max(honest max, max over edited |w_tampered - w_ref|/sigma).
    sizes = np.array([int(np.prod(t.shape)) for t in ts])
    offs = np.r_[0, np.cumsum(sizes)]
    n_total = int(offs[-1])
    gl = {
        (s, k): np.sort(np.random.default_rng([1000 + s, k]).choice(n_total, size=k, replace=False))
        for s in range(seeds)
        for k in LARGE_EDIT_KS
    }
    mx_large = {key: 0.0 for key in gl}
    tensor_data = []
    for _ti, t in enumerate(ts):
        a = raw_bytes(t)
        b = raw_bytes(tn[t.name])
        wr, wn = deq(a, qtype), deq(b, qtype)
        sd = wr.std() or 1e-12
        dd = wn - wr
        d = np.abs(dd) / sd
        ex = d > tau
        base = int(ex.sum())
        n = d.size
        neg.append(base / n)
        pooled_base += base
        mx_base = max(mx_base, float(d.max()))
        changed = dd != 0
        nat_changed.append(d[changed])
        nat_touched += int(changed.sum())
        total_weights += n
        # exact one-step delta for every weight (dequantize all-codes+1)
        step = deq(with_payload(b, bb, ps, pe, plus1), qtype) - wn
        # farthest-code plane (EXP11 --large): dequantize with every payload
        # code forced to 0 and to 15; per weight the farthest representable
        # value is whichever of the two is farther from the natural value.
        w0 = deq(with_payload(b, bb, ps, pe, np.zeros_like), qtype)
        w15 = deq(with_payload(b, bb, ps, pe, lambda p: np.full_like(p, 0xFF)), qtype)
        farthest = np.where(np.abs(w0 - wn) > np.abs(w15 - wn), w0, w15)
        tensor_data.append((dd, step, sd, base, ex, n))
        lo, hi = offs[_ti], offs[_ti + 1]
        for key, g in gl.items():
            loc = g[(g >= lo) & (g < hi)] - lo
            if loc.size:
                dv = np.abs(farthest[loc] - wr[loc]) / sd
                mx_large[key] = max(mx_large[key], float(dv.max()))

    def auc(negv: np.ndarray, posv: np.ndarray) -> float:
        # Mann-Whitney AUC with tie-averaged ranks (harness method).
        allv = np.concatenate([negv, posv])
        order = np.argsort(allv, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, allv.size + 1)
        # tie-averaging
        sv = allv[order]
        i = 0
        while i < sv.size:
            j = i
            while j + 1 < sv.size and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                ranks[order[i : j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        rn_, rp_ = negv.size, posv.size
        rp = ranks[rn_:].sum()
        return float((rp - rp_ * (rp_ + 1) / 2) / (rn_ * rp_))

    per_budget = {}
    for bi, f in enumerate(budgets):
        aucs = []
        for s in range(seeds):
            pos = []
            for ti, (dd, step, sd, base, ex, n) in enumerate(tensor_data):
                k = int(n * f)
                if k < 1:
                    pos.append(base / n)
                    continue
                idx = np.random.default_rng([s, ti, bi]).choice(n, size=k, replace=False)
                nv = np.abs(dd[idx] + step[idx]) / sd
                cnt = base - int(ex[idx].sum()) + int((nv > tau).sum())
                pos.append(cnt / n)
            aucs.append(auc(np.array(neg), np.array(pos)))
        per_budget[str(f)] = {
            "auc_mean": float(np.mean(aucs)),
            "auc_min": float(np.min(aucs)),
        }
    floor = None
    for f in budgets:
        if per_budget[str(f)]["auc_min"] >= 0.99:
            floor = f
            break
    large_edits = {}
    for k in LARGE_EDIT_KS:
        v = [max(mx_base, mx_large[(s, k)]) for s in range(seeds)]
        large_edits[str(k)] = {
            "max_mean": float(np.mean(v)),
            "max_min": float(np.min(v)),
            "exceeds_honest_max": bool(all(x > mx_base for x in v)),
        }
    large_edits["honest_max"] = mx_base
    return {
        "qtype": qtype,
        "tau": tau,
        "seeds": seeds,
        "n_tensors": len(ts),
        "n_weights": total_weights,
        "honest": {
            "tensor_tail_median": float(np.median(neg)),
            "tensor_tail_max": float(np.max(neg)),
            "pooled_tail": pooled_base / total_weights,
            "max_statistic": mx_base,
            "natural_variance": {
                "touched_frac": nat_touched / total_weights,
                "changed_pct": {
                    p: float(np.percentile(np.concatenate(nat_changed), pc))
                    for p, pc in (("median", 50), ("p99", 99), ("max", 100))
                },
            },
        },
        "per_budget": per_budget,
        "floor": floor,
        "large_edits": large_edits,
    }


# ---------------------------------------------------------------------------
# Step 4 — ggufpacker cross-check (EXACT vs NEAR, published-file XOR).
# ---------------------------------------------------------------------------


def payload_xor_stats(a: Path, b: Path) -> dict:
    """XOR the tensor-data regions of two layout-congruent GGUFs: nonzero
    count, fraction, compressed size (zstd level 12, ggufpacker's delta
    level). Congruence = same tensor names/order/types/sizes (relative
    layout), like ggufpacker's congruence key; absolute offsets may differ
    (the producer's header/KV block differs — that is ggufpacker's
    "header patch", separate from the payload delta)."""
    ra, rb = GGUFReader(str(a)), GGUFReader(str(b))
    ta = {t.name: t for t in ra.tensors}
    tb = {t.name: t for t in rb.tensors}
    if set(ta) != set(tb):
        return {"error": "tensor name sets differ (non-congruent)"}
    for name in sorted(ta):
        if ta[name].tensor_type != tb[name].tensor_type or ta[name].n_bytes != tb[name].n_bytes:
            return {"error": f"tensor {name} type/size mismatch (non-congruent)"}
    nonzero = 0
    total = 0
    import io

    import zstandard  # type: ignore[import-not-found]

    cctx = zstandard.ZstdCompressor(level=12)
    sink = io.BytesIO()
    for name in sorted(ta):  # name-keyed: order may differ between releases
        xa = np.frombuffer(ta[name].data.tobytes(), np.uint8)
        xb = np.frombuffer(tb[name].data.tobytes(), np.uint8)
        x = xa ^ xb
        nonzero += int((x != 0).sum())
        total += x.size
        sink.write(x.tobytes())
    comp = cctx.compress(sink.getvalue())
    return {
        "nonzero_bytes": nonzero,
        "total_bytes": total,
        "nonzero_frac": nonzero / total if total else 0.0,
        "zstd12_bytes": len(comp),
        "zstd12_frac_of_original": len(comp) / a.stat().st_size,
        "metadata_identical": (
            open(a, "rb").read(int(ra.data_offset)) == open(b, "rb").read(int(rb.data_offset))
        ),
    }


def ggufpacker_crosscheck(matrix: dict[str, dict[str, dict]], fixtures: dict[str, Path]) -> dict:
    """ggufpacker's plan decision, reimplemented on our regenerated files
    against the publisher's published artifacts:
      EXACT  = zero-XOR tensor payload over the data region
      NEAR   = congruent layout, nonzero but small XOR (zstd-12 delta)
      BLOB   = incongruent layout / oversized delta
    One machine + one binary reproducing the publisher's file can only show
    EXACT when our environment matches the producer's; the interesting cells
    are our three environments vs the published files.
    """
    out = {}
    for qtype, published_name in (
        ("Q8_0", "Qwen2.5-0.5B-Instruct-Q8_0.gguf"),
        ("Q4_K_M", "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"),
    ):
        pub = fixtures[published_name]
        out[qtype] = {}
        for build in BUILDS:
            regen = Path(matrix[build["label"]][qtype]["path"])
            stats = payload_xor_stats(regen, pub)
            if "error" in stats:
                plan = "BLOB"
            elif stats["nonzero_bytes"] == 0:
                plan = "EXACT"
            elif stats["zstd12_frac_of_original"] <= 0.5:
                plan = "NEAR"
            else:
                plan = "BLOB"
            out[qtype][build["label"]] = {**stats, "plan": plan}
            _log(
                f"ggufpacker-xcheck {qtype:<7} {build['label']:>14}: plan={plan} "
                f"nonzero={stats.get('nonzero_bytes', -1)} "
                f"({stats.get('nonzero_frac', -1):.6f})"
            )
    return out


def recipe_pinned_rederive(
    builds_root: Path, f16: Path, work_dir: Path, fixtures: dict[str, Path]
) -> dict:
    """Paper §IV trial re-derivation, 0.5B edition: regenerate the published
    Q4_K_M with the per-tensor recipe pinned (--tensor-type-file built from
    the published file's own type map, the paper's '+rec' column), then XOR
    against the published file. Expected outcome (pre-registered): the
    recipe closes the type-map gap but the payload still differs, because
    the publisher's build used an imatrix the repo does not ship
    (quantize.imatrix.* KV keys present, no .imatrix file in the repo) —
    the paper's third card-absent coordinate C.
    """
    pub = fixtures["Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"]
    rp = GGUFReader(str(pub))
    inv = {v.value: v.name.lower() for v in GGMLQuantizationType}
    pub_types = {t.name: inv[int(t.tensor_type)] for t in rp.tensors}

    # Step 1: stock-preset regen already exists in the matrix; the recipe
    # gap = tensors whose published type differs from the stock preset.
    stock = GGUFReader(str(work_dir / "arm64-nocontract__Q4_K_M.gguf"))
    stock_types = {t.name: _ttype_name(t).lower() for t in stock.tensors}
    n_recipe_gap = sum(1 for n in pub_types if pub_types[n] != stock_types.get(n))

    # Step 2: recipe-pinned regeneration pinning the FULL published type
    # map (the paper's "+rec" column pins the deployed file's types
    # wholesale; a diff-only override file leaves unlisted tensors to the
    # preset's own heuristics, which differ across llama.cpp releases).
    recipe_file = work_dir / "recipe_overrides.txt"
    recipe_file.write_text("\n".join(f"{n}={t}" for n, t in sorted(pub_types.items())))
    out = {
        "published_kv": {
            "imatrix_file_recorded": "quantize.imatrix.file" in rp.fields,
            "imatrix_dataset_recorded": "quantize.imatrix.dataset" in rp.fields,
            "imatrix_entries": int(rp.fields["quantize.imatrix.entries_count"].parts[-1][0])
            if "quantize.imatrix.entries_count" in rp.fields
            else None,
        },
        "n_tensors": len(pub_types),
        "n_recipe_gap": n_recipe_gap,
        "per_build": {},
    }
    for build in BUILDS:
        binary = builds_root / build["binary"]
        regen_path = work_dir / f"{build['label']}__Q4_K_M__recipe.gguf"
        cmd = [
            str(binary),
            "--tensor-type-file",
            str(recipe_file),
            str(f16),
            str(regen_path),
            "Q4_K_M",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            out["per_build"][build["label"]] = {"error": res.stderr[-500:]}
            continue
        stats = payload_xor_stats(regen_path, pub)
        out["per_build"][build["label"]] = stats
        _log(
            f"recipe-pinned {build['label']:>14}: "
            f"nonzero={stats.get('nonzero_bytes', -1)} "
            f"({stats.get('nonzero_frac', -1):.6f})"
        )
    return out


def card_alone_match_count(matrix: dict[str, dict[str, dict]], fixtures: dict[str, Path]) -> dict:
    """Paper §IV 'card' column, 0.5B edition: per-tensor payload-equality
    of the stock-preset regeneration vs the published Q4_K_M, by tensor
    name, skipping type-mismatched tensors (paper: 140/335 and 141/339
    for the two families)."""
    pub = GGUFReader(str(fixtures["Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"]))
    tp = {t.name: t for t in pub.tensors}
    out = {}
    for build in BUILDS:
        reg = GGUFReader(str(matrix[build["label"]]["Q4_K_M"]["path"]))
        matched = type_mismatched = 0
        for t in reg.tensors:
            pt = tp.get(t.name)
            if pt is None or t.tensor_type != pt.tensor_type or t.n_bytes != pt.n_bytes:
                type_mismatched += 1
                continue
            matched += int(t.data.tobytes() == pt.data.tobytes())
        out[build["label"]] = {
            "matched_tensors": matched,
            "type_mismatched": type_mismatched,
            "total_tensors": len(reg.tensors),
        }
        _log(f"card-alone {build['label']:>14}: {matched}/{len(reg.tensors)} tensors match")
    return out


def census_rescore(harness_root: Path | None, work_dir: Path) -> dict:
    """Paper §III census, re-scored with the paper's own frozen instrument:
    run the harness EXP02 census.py over the EXP03 frozen top-200 repo list
    (its own python3, stdlib-only, ranged header reads — no full downloads)
    and check the pre-registered expectations against a fresh scoring."""
    root = harness_root or DEFAULT_HARNESS_ROOT
    script = root / CENSUS_SCRIPT
    repos_file = root / CENSUS_REPOS_FILE
    for p in (script, repos_file):
        if not p.exists():
            raise SystemExit(f"census harness not found: {p} (see --harness-root)")
    out_csv = work_dir / "census_rescore.csv"
    cmd = ["python3", str(script), "--repos-file", str(repos_file), "--out", str(out_csv)]
    _log("step 7: census re-score (frozen top-200, ~10 min, ranged header reads)")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"census run failed:\n{res.stderr[-2000:]}")
    for line in res.stdout.splitlines():
        if line.startswith(("===", "  ")):
            _log(f"census: {line.strip()}")

    with open(out_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ok = [r for r in rows if r["status"] == "ok"]
    n_error = sum(1 for r in rows if r["status"] != "ok")
    bit_exact = sum(1 for r in ok if r["bit_exact_verifiable"] == "True")
    src_recon = sum(1 for r in ok if r["source_reconstructible"] == "True")
    env_gap = sum(1 for r in ok if "env/version" in r["primary_gap"])
    exp = CENSUS_EXPECTED
    drift = []
    if len(rows) != exp["n_repos"]:
        drift.append(f"n_repos {len(rows)} != pinned {exp['n_repos']}")
    if bit_exact != exp["bit_exact_verifiable"]:
        drift.append(f"bit_exact_verifiable {bit_exact} != pinned {exp['bit_exact_verifiable']}")
    if src_recon != exp["source_reconstructible"]:
        drift.append(
            f"source_reconstructible {src_recon} != pinned {exp['source_reconstructible']} "
            "(repos deleted/gated since the freeze move repos between n_error "
            "and the scored denominator)"
        )
    if drift:
        raise SystemExit("census expectation drift:\n  " + "\n  ".join(drift))
    return {
        "harness_root": str(root),
        "repos_file": CENSUS_REPOS_FILE,
        "n_repos": len(rows),
        "n_scored": len(ok),
        "n_error": n_error,
        "bit_exact_verifiable": bit_exact,
        "source_reconstructible": src_recon,
        "env_version_gap": env_gap,
        "errors": sorted({r["status"] for r in rows if r["status"] != "ok"}),
        "matches_expectation": True,
    }


# ---------------------------------------------------------------------------
# Step 8 — b9222 trial-repo re-derivation (paper §IV Table III): the
# positive control — a trial repo that ships every card coordinate.
# ---------------------------------------------------------------------------


def fetch_b9222_fixtures(cache_dir: Path | None) -> dict[str, Path]:
    """Pinned downloads for the b9222 trial repo: bf16 source W, the
    publisher's imatrix (card coordinate C) and the published Q4_K_M."""
    out: dict[str, Path] = {}
    for name, digest in (
        (TRIAL_BF16, TRIAL_BF16_SHA256),
        (TRIAL_IMATRIX, TRIAL_IMATRIX_SHA256),
        (TRIAL_PUBLISHED, TRIAL_PUBLISHED_SHA256),
    ):
        p = Path(hf_hub_download(TRIAL_REPO, name, revision=TRIAL_REVISION, cache_dir=cache_dir))
        got = sha256_file(p)
        if got != digest:
            raise SystemExit(
                f"b9222 fixture {name} sha256 mismatch: {got} != pinned {digest} "
                "(repo revision moved?)"
            )
        out[name] = p
    return out


def locate_b9222_builds(builds_root: Path | None) -> dict[str, Path]:
    """Find the three b9222 llama-quantize binaries (arm64 default /
    arm64 nocontract / x86_64 Rosetta baseline). Order: an explicit
    --b9222-builds-root with the quantize-* layout, then the .scratch
    source trees the builder script produces."""
    if builds_root is not None:
        paths = {
            "arm64-default": builds_root / "quantize-default/llama-quantize",
            "arm64-nocontract": builds_root / "quantize-nocontract/llama-quantize",
            "x86_64-default": builds_root / "quantize-x86-default/llama-quantize",
        }
    else:
        repo_root = SMOKE_DIR.parents[1]
        hits = sorted(
            (repo_root / B9222_BUILD_GLOB.split("/", 1)[0]).glob(B9222_BUILD_GLOB.split("/", 1)[1])
        )
        # Per label: (tree-name substring, build-dir predicate). Matches
        # the recorded .scratch trees (llama.cpp-b9222{,-x86stock} with
        # build-{nocontract,x86default,arm64default}) and fresh
        # scripts/build_llamacpp.sh trees (build-{default,x86-default}).
        matchers = {
            "arm64-nocontract": ("llama.cpp-b9222", lambda b: "nocontract" in b),
            "x86_64-default": ("x86stock", lambda b: "x86" in b),
            "arm64-default": ("x86stock", lambda b: "arm64" in b or b == "build-default"),
        }
        paths = {}
        for label, (tree_sub, build_pred) in matchers.items():
            c = [p for p in hits if tree_sub in p.parents[2].name and build_pred(p.parents[1].name)]
            if len(c) != 1:
                raise SystemExit(
                    f"b9222 binary for {label} not found (candidates: {c}); build the "
                    "pinned release with: scripts/build_llamacpp.sh <src-dir> <out-dir> "
                    "--commit 9a532ae4bab1b164052ce60a738f78538b421c66 --b9222 "
                    "or pass --b9222-builds-root"
                )
            paths[label] = c[0]
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(
            "missing b9222 build binary(ies): " + ", ".join(missing) + "\n"
            "build them with: scripts/build_llamacpp.sh <src-dir> <out-dir> "
            "--commit 9a532ae4bab1b164052ce60a738f78538b421c66 --b9222"
        )
    return paths


def tensor_xor_counts(path_a: Path, path_b: Path) -> dict:
    """Per-tensor payload XOR comparison by tensor name (the paper's §IV
    per-tensor protocol; metadata/KV region excluded by design): tensors
    present in both files, same ggml type -> XOR payload bytes; type
    differs -> counted as a type mismatch, skipped from the XOR."""
    a = {t.name: t for t in GGUFReader(str(path_a)).tensors}
    b = {t.name: t for t in GGUFReader(str(path_b)).tensors}
    matched = type_mismatched = length_mismatched = nonzero_bytes = 0
    for name in sorted(set(a) & set(b)):
        ta, tb = a[name], b[name]
        if ta.tensor_type != tb.tensor_type:
            type_mismatched += 1
            continue
        da, db = ta.data.tobytes(), tb.data.tobytes()
        if len(da) != len(db):
            length_mismatched += 1
            continue
        nonzero_bytes += int((np.frombuffer(da, np.uint8) ^ np.frombuffer(db, np.uint8) != 0).sum())
        matched += int(da == db)
    return {
        "tensors_a": len(a),
        "tensors_b": len(b),
        "only_in_a": len(set(a) - set(b)),
        "only_in_b": len(set(b) - set(a)),
        "matched": matched,
        "type_mismatched": type_mismatched,
        "length_mismatched": length_mismatched,
        "nonzero_xor_bytes": nonzero_bytes,
        "total": len(set(a) | set(b)),
        "bit_exact": (
            not type_mismatched
            and not length_mismatched
            and not nonzero_bytes
            and not (set(a) ^ set(b))
            and matched == len(set(a) | set(b))
        ),
    }


def b9222_table_iii(binaries: dict[str, Path], fixtures: dict[str, Path], work_dir: Path) -> dict:
    """Paper §IV Table III, b9222 edition: re-derive the trial repo's
    published Q4_K_M from its own bf16 + imatrix with two environments
    (arm64 contraction-off, x86_64 baseline) in two modes (stock Q4_K_M
    preset, full published recipe), plus the arm64-default card-only
    column. Every run is quantized with the publisher's imatrix (the card
    coordinate C the trial repo ships); recipe runs pin the FULL published
    type map via --tensor-type-file with ^re.escape(name)$ semantics —
    llama-quantize matches tensor names with std::regex_search, so a bare
    name=type line is a wildcard that also matches other tensors."""
    bf16, imatrix, published = (
        fixtures[TRIAL_BF16],
        fixtures[TRIAL_IMATRIX],
        fixtures[TRIAL_PUBLISHED],
    )

    # Full published type map (the '+rec' column pins the deployed file's
    # types wholesale, ^escaped$ so each line matches exactly one tensor).
    rp = GGUFReader(str(published))
    recipe_file = work_dir / "b9222_recipe_Q4_K_M.txt"
    recipe_file.write_text(
        "\n".join(f"^{re.escape(t.name)}$={_ttype_name(t).lower()}" for t in rp.tensors) + "\n"
    )

    runs = [
        # (run key, build label, mode)
        ("arm64-nocontract__stock", "arm64-nocontract", "stock"),
        ("arm64-nocontract__recipe", "arm64-nocontract", "recipe"),
        ("x86_64-default__stock", "x86_64-default", "stock"),
        ("x86_64-default__recipe", "x86_64-default", "recipe"),
        ("arm64-default__stock", "arm64-default", "stock"),
    ]
    table = {}
    for key, build_label, mode in runs:
        binary = binaries[build_label]
        out_path = work_dir / f"b9222_{key}.gguf"
        cmd = [str(binary), "--imatrix", str(imatrix)]
        if mode == "recipe":
            cmd += ["--tensor-type-file", str(recipe_file)]
        cmd += [str(bf16), str(out_path), "Q4_K_M"]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise SystemExit(f"b9222 llama-quantize failed ({key}):\n{res.stderr[-2000:]}")
        stats = tensor_xor_counts(out_path, published)
        expected = B9222_TABLE_III_EXPECTED[key]
        cell = {
            "build": build_label,
            "mode": mode,
            "seconds": round(time.time() - t0, 1),
            **stats,
            "expected_matched": expected[0],
            "expected_total": expected[1],
            "matches_expectation": (stats["matched"], stats["total"]) == expected,
        }
        table[key] = cell
        _log(
            f"b9222 {key:<28}: {stats['matched']}/{stats['total']} tensors match "
            f"(expected {expected[0]}/{expected[1]}) "
            f"nonzero_xor={stats['nonzero_xor_bytes']}"
        )
    # Whole-file digests: recipe outputs should be byte-identical to each
    # other; published-vs-ours differ only in the quantize.imatrix.file KV
    # path string (excluded by the paper's per-tensor protocol).
    digests = {
        "published": sha256_file(published),
        **{
            key: sha256_file(work_dir / f"b9222_{key}.gguf")
            for key in table
            if key.endswith("__recipe")
        },
    }
    recipe_digests = {k: v for k, v in digests.items() if k != "published"}
    out = {
        "llama_cpp_commit": B9222_COMMIT,
        "trial_repo": TRIAL_REPO,
        "trial_revision": TRIAL_REVISION,
        "recipe_lines": len(rp.tensors),
        "runs": table,
        "recipe_sha256": digests,
        "recipe_outputs_byte_identical": len(set(recipe_digests.values())) == 1,
        "all_expectations_hold": all(c["matches_expectation"] for c in table.values()),
    }
    if not out["all_expectations_hold"]:
        bad = [k for k, c in table.items() if not c["matches_expectation"]]
        raise SystemExit(f"b9222 Table III expectation drift: {bad}")
    return out


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="what-you-sign smoke (smokes/what-you-sign/)")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--builds-root", type=Path, default=DEFAULT_BUILDS_ROOT)
    parser.add_argument(
        "--work-dir", type=Path, default=None, help="default: /tmp/what-you-sign-work"
    )
    parser.add_argument("--repeats", type=int, default=REPEAT_RUNS)
    parser.add_argument(
        "--census", action="store_true", help="also re-score the paper's frozen top-200 census"
    )
    parser.add_argument(
        "--harness-root",
        type=Path,
        default=DEFAULT_HARNESS_ROOT,
        help="paper harness clone for --census (default: .scratch/gguf-verifiability-artifact)",
    )
    parser.add_argument(
        "--b9222",
        action="store_true",
        help="also run the paper §IV Table III b9222 trial-repo re-derivation",
    )
    parser.add_argument(
        "--b9222-builds-root",
        type=Path,
        default=None,
        help="b9222 binaries root with the quantize-* layout for --b9222 "
        "(default: .scratch/llama.cpp-b9222* build trees)",
    )
    args = parser.parse_args()

    work = args.work_dir or Path("/tmp/what-you-sign-work")
    work.mkdir(parents=True, exist_ok=True)

    _log(f"llama.cpp commit {LLAMA_CPP_COMMIT[:12]} · builds root {args.builds_root}")
    _log("step 0: fixtures (pinned revisions)")
    fixtures = fetch_fixtures(args.cache_dir)

    _log("step 1: quantize matrix (3 builds x 4 types)")
    matrix = run_quantize_matrix(args.builds_root, fixtures["f16"], work, repeat=args.repeats)
    hash_cells = check_hash_matrix(matrix)

    _log("step 2: structured byte diff (divergent cells only)")
    diffs = {}
    labels = [b["label"] for b in BUILDS]
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            for qtype in QUANT_TYPES:
                if matrix[a][qtype]["sha256"] == matrix[b][qtype]["sha256"]:
                    continue
                key = f"{a}__vs__{b}__{qtype}"
                diffs[key] = structured_diff(
                    Path(matrix[a][qtype]["path"]), Path(matrix[b][qtype]["path"])
                )

    _log("step 3: detection floor (Q4_K_M, arm64-default vs arm64-nocontract)")
    floor = detection_floor(
        Path(matrix["arm64-nocontract"]["Q4_K_M"]["path"]),
        Path(matrix["arm64-default"]["Q4_K_M"]["path"]),
        "Q4_K_M",
        DETECTOR_TAU,
        DETECTOR_SEEDS,
        TAMPER_BUDGETS,
    )

    _log("step 4: ggufpacker cross-check vs published files")
    xcheck = ggufpacker_crosscheck(matrix, fixtures)

    _log("step 5: recipe-pinned re-derivation (paper §IV trial, 0.5B edition)")
    recipe = recipe_pinned_rederive(args.builds_root, fixtures["f16"], work, fixtures)

    _log("step 6: card-alone per-tensor match counts (paper 'card' column)")
    card_alone = card_alone_match_count(matrix, fixtures)

    census = census_rescore(args.harness_root, work) if args.census else None

    b9222 = None
    if args.b9222:
        _log("step 8: b9222 trial-repo re-derivation (paper §IV Table III)")
        b9222_fixtures = fetch_b9222_fixtures(args.cache_dir)
        b9222_binaries = locate_b9222_builds(args.b9222_builds_root)
        b9222 = b9222_table_iii(b9222_binaries, b9222_fixtures, work)

    output = {
        "params": {
            "paper": "arXiv:2610.19498",
            "publisher_repo": PUBLISHER_REPO,
            "publisher_revision": PUBLISHER_REVISION,
            "f16_sha256": F16_SHA256,
            "llama_cpp_commit": LLAMA_CPP_COMMIT,
            "builds": [b["label"] for b in BUILDS],
            "quant_types": list(QUANT_TYPES),
            "tau": DETECTOR_TAU,
            "seeds": DETECTOR_SEEDS,
            "budgets": list(TAMPER_BUDGETS),
            "repeat_runs": args.repeats,
            "thread_variants": list(THREAD_VARIANTS),
        },
        "quantize_matrix": matrix,
        "hash_matrix_cells": hash_cells,
        "structured_diffs": diffs,
        "detection_floor": floor,
        "ggufpacker_crosscheck": xcheck,
        "recipe_pinned_rederive": recipe,
        "card_alone_match_counts": card_alone,
    }
    if census is not None:
        output["census_rescore"] = census
    if b9222 is not None:
        output["b9222_table_iii"] = b9222
    print(json.dumps(output, indent=2, default=float))


if __name__ == "__main__":
    main()
