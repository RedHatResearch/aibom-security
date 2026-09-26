"""Opt-in controlled MoE Origin Tracer experiment.

This is a network/heavy smoke, not a default check.  It downloads (or reads
from the local Transformers cache) TinyMixtral, makes a temporary candidate
with exactly one scalar changed in one attention V projection, and delegates
the diagnostic run to ``smoke.py --allow-moe``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Isotonic/TinyMixtral-4x248M-MoE"
SMOKE_PATH = Path(__file__).with_name("smoke.py")


def _load_origin_tracer() -> Any:
    spec = importlib.util.spec_from_file_location("origin_tracer_smoke", SMOKE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Origin Tracer implementation from {SMOKE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _find_v_parameter(model: Any, layer_index: int) -> tuple[str, Any]:
    candidates: list[tuple[str, Any]] = []
    for name, parameter in model.named_parameters():
        parts = name.lower().replace("/", ".").split(".")
        if len(parts) < 2 or parts[-1] != "weight":
            continue
        if parts[-2] not in {"v_proj", "value_proj", "value"}:
            continue
        layer_marker = f"layers.{layer_index}"
        if layer_marker in ".".join(parts):
            candidates.append((name, parameter))
    if len(candidates) != 1:
        names = [name for name, _ in candidates]
        raise RuntimeError(
            f"expected exactly one attention V weight in layer {layer_index}, found {names}"
        )
    return candidates[0]


def _make_candidate(
    model_id: str,
    candidate_dir: Path,
    layer_index: int,
    delta: float,
    cache_dir: Path | None,
    revision: str | None,
    local_files_only: bool,
    dtype: str,
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir.expanduser())
    if revision:
        kwargs["revision"] = revision
    if dtype != "auto":
        kwargs["torch_dtype"] = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, **{k: v for k, v in kwargs.items() if k != "torch_dtype"}
    )
    parameter_name, parameter = _find_v_parameter(model, layer_index)
    if delta == 0.0:
        raise ValueError("--delta must be nonzero so the candidate has one controlled V update")
    with torch.no_grad():
        parameter.view(-1)[0].add_(delta)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(candidate_dir, safe_serialization=True)
    tokenizer.save_pretrained(candidate_dir)
    print(
        f"created controlled candidate {candidate_dir} with one scalar update: "
        f"{parameter_name}[0] += {delta}",
        file=sys.stderr,
    )
    return str(candidate_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "OPT-IN network/heavy TinyMixtral experiment: create one controlled "
            "attention-V scalar update, then run the experimental MoE diagnostics"
        )
    )
    parser.add_argument(
        "--base-model", default=DEFAULT_MODEL, help="TinyMixtral base ID or local path"
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        help="retain the generated candidate here; otherwise use a temporary directory",
    )
    parser.add_argument(
        "--layer", type=int, default=0, help="layer whose V weight gets one scalar update"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-3,
        help="nonzero scalar added to the first V weight entry (default: 1e-3)",
    )
    parser.add_argument("--cache-dir", type=Path, help="Transformers/HF cache directory")
    parser.add_argument("--revision", help="optional Hub revision for the base")
    parser.add_argument(
        "--local-files-only", action="store_true", help="disable Hub network access"
    )
    parser.add_argument("--probe-count", type=int, default=8)
    parser.add_argument("--probe-words", help="comma-separated deterministic word override")
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--reconstruction-steps", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--layers",
        default=None,
        help="Origin Tracer layers (default: the controlled-update layer)",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.delta == 0.0:
        parser.error("--delta must be nonzero")
    if args.layer < 0:
        parser.error("--layer must be nonnegative")

    origin_tracer = _load_origin_tracer()
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.candidate_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="origin-tracer-tinymixtral-")
        candidate_dir = Path(temp_dir.name)
    else:
        candidate_dir = args.candidate_dir

    try:
        candidate_model = _make_candidate(
            args.base_model,
            candidate_dir,
            args.layer,
            args.delta,
            args.cache_dir,
            args.revision,
            args.local_files_only,
            args.dtype,
        )
        smoke_args = [
            "--allow-moe",
            "--base-model",
            args.base_model,
            "--candidate-model",
            candidate_model,
            "--probe-count",
            str(args.probe_count),
            "--cycles",
            str(args.cycles),
            "--reconstruction-steps",
            str(args.reconstruction_steps),
            "--learning-rate",
            str(args.learning_rate),
            "--layers",
            args.layers or str(args.layer),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--seed",
            str(args.seed),
        ]
        if args.probe_words:
            smoke_args.extend(("--probe-words", args.probe_words))
        if args.cache_dir is not None:
            smoke_args.extend(("--cache-dir", str(args.cache_dir)))
        if args.revision:
            smoke_args.extend(("--revision", args.revision))
        if args.local_files_only:
            smoke_args.append("--local-files-only")
        return int(origin_tracer.main(smoke_args))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
