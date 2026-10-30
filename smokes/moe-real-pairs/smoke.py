#!/usr/bin/env python3
"""Real-pair consistency smoke for the MoE expert-lineage detector (issue #68).

First contact with published Hub artifacts. There is no ground truth for these
two derivatives, only a published recipe, so the question is not "did we recover
the injection" but "do the detector's per-layer counts agree with the recipe the
model card states, and where does it abstain".

Label-blindness: ``score()`` receives a subject repo id + pinned revision, a
reference repo id + pinned revision, the stack string, and the layer sample
pre-registered per pair in ``config.SAMPLING`` (fixed before the run, never
derived from a result). It never sees the pair id, the expected label, the
evidence block, or the recipe. The harness attaches the pair id and judges the
result against ``config.PREREG`` after the fact; nothing that judges feeds back
into scoring.

Ours, not from the paper (the same list as README.md's disclosures):
- ``BatchedExpertView``: GraniteMoe stores every expert of a layer in one tensor
  with the expert index in dim 0. The view presents that storage through the
  detector's per-expert reader API; no detector code is copied or changed.
- Layer sampling: layers 0 / 24 / 47 (first, middle, last) of the 48-layer 30B
  pair instead of all 48; the granite pair scores every layer.
- Shard-level budget gate: a layer whose projected download + analysis time or
  RSS would not fit the remaining budget is skipped with that projection
  recorded, never silently truncated.
- HTTP ranged materialisation (``plan_ranged`` / ``run_ranged``): the shard
  header is size-capped, offsets are validated against the ``Content-Range``
  total, only ``206`` is accepted, and the fetched tensors become a compact
  mini-shard reused on a rerun, keyed by repo, revision and layer.
- Default download-rate assumption (2.5 MB/s) for the first projection, replaced
  by the measured rate afterwards; it decides which layers are attempted, never
  a score.
- The shard index is a download hint only: the pinned revision decides what is
  fetched and the reader re-derives tensor names from the safetensors headers.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "smokes" / "moe-expert-lineage"))
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import detect  # noqa: E402

DEFAULT_SHARD_BYTES = 4_000_000_000


# --------------------------------------------------------------------------- #
# budget and process accounting
# --------------------------------------------------------------------------- #
@dataclass
class Budget:
    deadline: float

    def remaining(self) -> float:
        return self.deadline - time.monotonic()


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


@dataclass
class Downloads:
    """Per-repo, per-fetch accounting so the budget decisions are auditable.

    ``bytes``/``seconds`` count every byte that crossed the wire, shard headers
    included. ``calls`` carries one entry per completed materialisation (a
    whole-shard fetch or a mini-shard write) with the repo and pinned revision
    it came from. A cache hit adds neither.
    """

    bytes_total: int = 0
    seconds: float = 0.0
    calls: list[dict] = field(default_factory=list)

    def add(self, nbytes: int, seconds: float) -> None:
        self.bytes_total += int(nbytes)
        self.seconds += max(float(seconds), 1e-6)

    def payload(self) -> dict:
        return {
            "bytes": self.bytes_total,
            "seconds": round(self.seconds, 2),
            "calls": self.calls,
        }


# --------------------------------------------------------------------------- #
# batched-expert storage adapter (GraniteMoe)
# --------------------------------------------------------------------------- #
class BatchedExpertView:
    """Present dim-0-batched expert weights through the detector reader API.

    The detector's ``descriptor``/``fit_mixtures``/``analyze_layer`` only need
    ``spec.roles``, ``names()``, ``shape()`` and ``expert()``. This view answers
    those from one physically batched tensor per role, so GraniteMoe needs no
    detector change. Every byte still comes from ``CheckpointReader.tensor``
    (one materialised tensor at a time, size-capped there).
    """

    def __init__(self, reader: detect.CheckpointReader, arch: dict, layers: tuple[int, ...]):
        self.reader = reader
        self.spec = detect.TensorSpec(
            expert=arch["expert_template"],
            router=arch["router"],
            roles=tuple(arch["roles"]),
        )
        self._physical = dict(arch["batched"])
        self._map: dict[str, tuple[str, int | None]] = {}
        self._cache: dict[str, torch.Tensor] = {}
        for layer in layers:
            per_role: dict[str, tuple[str, int]] = {}
            for role in self.spec.roles:
                template = self._physical.get(role)
                if template is None:
                    raise detect.DetectionError(f"batched layout has no tensor for role {role!r}")
                name = template.format(layer=layer)
                shape = reader.shape(name)
                if len(shape) != 3:
                    raise detect.DetectionError(
                        f"{name}: expected [experts, rows, cols], got {list(shape)}"
                    )
                per_role[role] = (name, shape[0])
            counts = {count for _, count in per_role.values()}
            if len(counts) != 1:
                raise detect.DetectionError(
                    f"layer {layer}: batched expert counts disagree: {sorted(counts)}"
                )
            count = counts.pop()
            if not 1 <= count <= detect.MAX_EXPERTS_PER_LAYER:
                raise detect.DetectionError(
                    f"layer {layer}: batched expert count {count} out of range"
                )
            for expert in range(count):
                for role, (name, _) in per_role.items():
                    logical = self.spec.expert.format(layer=layer, expert=expert, role=role)
                    self._map[logical] = (name, expert)
            self._map[self.spec.router.format(layer=layer)] = (
                self.spec.router.format(layer=layer),
                None,
            )

    def names(self) -> dict[str, Path]:
        physical = self.reader.names()
        return {
            logical: physical.get(name, self.reader.root)
            for logical, (name, _) in self._map.items()
        }

    def _entry(self, name: str) -> tuple[str, int | None]:
        try:
            return self._map[name]
        except KeyError:
            raise detect.DetectionError(f"{name}: not in the checkpoint") from None

    def shape(self, name: str) -> tuple[int, ...]:
        physical, expert = self._entry(name)
        shape = self.reader.shape(physical)
        return shape if expert is None else tuple(int(v) for v in shape[1:])

    def tensor(self, name: str) -> torch.Tensor:
        physical, expert = self._entry(name)
        cached = self._cache.get(physical)
        if cached is None:
            cached = self.reader.tensor(physical)
            self._cache[physical] = cached
        return cached if expert is None else cached[expert]

    def expert(self, layer: int, expert: int) -> dict[str, torch.Tensor]:
        return {
            role: self.tensor(self.spec.expert.format(layer=layer, expert=expert, role=role))
            for role in self.spec.roles
        }

    def clear_cache(self) -> None:
        self._cache.clear()


# --------------------------------------------------------------------------- #
# Hub access (metadata only; tensor bytes go through CheckpointReader)
# --------------------------------------------------------------------------- #
def _repo_files(repo: str, revision: str) -> set[str]:
    try:
        names = HfApi().list_repo_files(repo_id=repo, revision=revision)
    except Exception as exc:  # noqa: BLE001 - any Hub failure is an unavailable pair
        raise detect.DetectionError(
            f"hub listing {repo}@{revision}: {type(exc).__name__}: {exc}"
        ) from exc
    return {str(name) for name in names}


def _file_sizes(repo: str, revision: str) -> dict[str, int]:
    try:
        info = HfApi().model_info(repo_id=repo, revision=revision, files_metadata=True)
    except Exception:  # noqa: BLE001 - projection degrades to a conservative default
        return {}
    sizes: dict[str, int] = {}
    for sibling in info.siblings or []:
        size = getattr(sibling, "size", None)
        if size is not None:
            sizes[str(sibling.rfilename)] = int(size)
    return sizes


def _snapshot(repo: str, revision: str, patterns: list[str]) -> Path:
    try:
        path = snapshot_download(repo_id=repo, revision=revision, allow_patterns=list(patterns))
    except Exception as exc:  # noqa: BLE001 - untrusted network surface
        raise detect.DetectionError(
            f"hub fetch {repo}@{revision} {patterns[:2]}: {type(exc).__name__}: {exc}"
        ) from exc
    return Path(path)


@dataclass
class ModelPlan:
    repo: str
    revision: str
    root: Path
    arch: dict
    layout: str
    num_layers: int
    num_experts: int
    plan: dict[int, tuple[str, ...]]
    sizes: dict[str, int]

    def spec(self) -> detect.TensorSpec:
        return detect.TensorSpec(
            expert=self.arch["expert_template"],
            router=self.arch["router"],
            roles=tuple(self.arch["roles"]),
        )

    def describe(self) -> dict:
        return {
            "repo": self.repo,
            "revision": self.revision,
            "model_type": self.arch["model_type"],
            "layout": self.layout,
            "num_hidden_layers": self.num_layers,
            "num_experts": self.num_experts,
        }


def _structure(raw: object, arch: dict) -> tuple[int, int]:
    if not isinstance(raw, dict):
        raise detect.DetectionError("config is not an object")
    layers = raw.get("num_hidden_layers")
    if not isinstance(layers, int) or isinstance(layers, bool):
        raise detect.DetectionError(f"config num_hidden_layers {layers!r} is not an int")
    if not 1 <= layers <= detect.MAX_LAYERS:
        raise detect.DetectionError(f"config num_hidden_layers {layers} out of range")
    counts: list[int] = []
    for key in arch["expert_keys"]:
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            raise detect.DetectionError(f"config {key} {value!r} is not an int")
        counts.append(value)
    if not counts:
        raise detect.DetectionError("config declares no expert count")
    if len(set(counts)) != 1:
        raise detect.DetectionError(f"config expert counts disagree: {counts}")
    count = counts[0]
    if not 1 <= count <= detect.MAX_EXPERTS_PER_LAYER:
        raise detect.DetectionError(f"config expert count {count} out of range")
    return layers, count


def _plan_layers(
    root: Path, repo_files: set[str], layers: list[int]
) -> tuple[dict[int, tuple[str, ...]], str]:
    if "model.safetensors.index.json" in repo_files:
        document = detect.read_json(root / "model.safetensors.index.json", "shard index")
        if not isinstance(document, dict):
            raise detect.DetectionError("shard index is not an object")
        weight_map = document.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise detect.DetectionError("shard index has no weight_map")
        plan: dict[int, tuple[str, ...]] = {}
        for layer in layers:
            prefix = f"model.layers.{layer}."
            files = sorted(
                {str(value) for name, value in weight_map.items() if str(name).startswith(prefix)}
            )
            if not files:
                raise detect.DetectionError(f"layer {layer}: no tensors in the shard index")
            for value in files:
                detect.checkpoint_file(root, value, "shard index")
            absent = [name for name in files if name not in repo_files]
            if absent:
                raise detect.DetectionError(
                    f"layer {layer}: index names files absent from the repo: {absent}"
                )
            plan[layer] = tuple(files)
        return plan, "index"
    if "model.safetensors" in repo_files:
        return {layer: ("model.safetensors",) for layer in layers}, "single"
    raise detect.DetectionError("checkpoint ships neither model.safetensors nor a shard index")


def open_model(repo: str, revision: str, layers: tuple[int, ...] | None) -> ModelPlan:
    """Resolve config + layout for one pinned snapshot; no weight bytes are read."""
    repo_files = _repo_files(repo, revision)
    root = _snapshot(repo, revision, ["config.json"])
    raw = detect.read_json(root / "config.json", "config")
    model_type = raw.get("model_type") if isinstance(raw, dict) else None
    if not isinstance(model_type, str) or model_type not in config.ARCHES:
        raise detect.DetectionError(f"unsupported model_type {model_type!r}")
    arch = {**config.ARCHES[model_type], "model_type": model_type}
    num_layers, num_experts = _structure(raw, arch)
    wanted = list(range(num_layers)) if layers is None else list(layers)
    bad = [layer for layer in wanted if not 0 <= layer < num_layers]
    if bad:
        raise detect.DetectionError(f"sampled layers {bad} outside 0..{num_layers - 1}")
    if "model.safetensors.index.json" in repo_files:
        _snapshot(repo, revision, ["model.safetensors.index.json"])
    plan, layout = _plan_layers(root, repo_files, wanted)
    return ModelPlan(
        repo=repo,
        revision=revision,
        root=root,
        arch=arch,
        layout=layout,
        num_layers=num_layers,
        num_experts=num_experts,
        plan=plan,
        sizes=_file_sizes(repo, revision),
    )


@dataclass
class Rate:
    """Measured transfer rate shared across pairs; budget projections only."""

    bytes_total: int = 0
    seconds: float = 0.0

    def add(self, nbytes: int, seconds: float) -> None:
        self.bytes_total += int(nbytes)
        self.seconds += max(float(seconds), 1e-6)

    @property
    def bps(self) -> float | None:
        if self.bytes_total < 100_000_000 or self.seconds <= 0.0:
            return None
        return self.bytes_total / self.seconds


@dataclass(frozen=True)
class Material:
    """Local files for one layer: a full snapshot root or a mini-shard root."""

    root: Path
    files: tuple[str, ...]


class BudgetSkip(detect.DetectionError):
    """A layer that does not fit the wall-clock budget."""


def fetch(model: ModelPlan, files: tuple[str, ...], downloads: Downloads, rate: Rate) -> None:
    todo = [name for name in files if not (model.root / name).is_file()]
    if not todo:
        return
    started = time.monotonic()
    _snapshot(model.repo, model.revision, todo)
    elapsed = max(time.monotonic() - started, 1e-6)
    absent = [name for name in todo if not (model.root / name).is_file()]
    if absent:
        raise detect.DetectionError(f"fetch left shards absent: {absent}")
    size = sum((model.root / name).stat().st_size for name in todo)
    downloads.add(size, elapsed)
    rate.add(size, elapsed)
    downloads.calls.append(
        {
            "repo": model.repo,
            "revision": model.revision,
            "mode": "full-shard",
            "files": todo,
            "bytes": size,
            "seconds": round(elapsed, 3),
        }
    )


# --------------------------------------------------------------------------- #
# ranged reads: materialise only the tensors of one layer (30B-class pairs)
# --------------------------------------------------------------------------- #
RANGE_GAP = 1 << 20
MAX_RANGE_SPAN = 64 << 20


def _session():
    import requests
    from huggingface_hub.utils import build_hf_headers

    session = requests.Session()
    session.headers.update(build_hf_headers())
    return session


def _remote_url(model: ModelPlan, name: str) -> str:
    return f"https://huggingface.co/{model.repo}/resolve/{model.revision}/{name}"


def _get_range(session, url: str, start: int, end: int) -> tuple[bytes, int]:
    """Read one inclusive byte range; only 206 responses are accepted."""
    if not 0 <= start <= end or end - start + 1 > 2 * MAX_RANGE_SPAN:
        raise detect.DetectionError(f"{url}: refused range {start}-{end}")
    try:
        response = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(15, 180))
    except Exception as exc:  # noqa: BLE001 - untrusted network surface
        raise detect.DetectionError(
            f"{url}: range {start}-{end} failed: {type(exc).__name__}: {exc}"
        ) from exc
    if response.status_code != 206:
        raise detect.DetectionError(
            f"{url}: range {start}-{end} answered {response.status_code}, not 206"
        )
    payload = response.content
    if len(payload) != end - start + 1:
        raise detect.DetectionError(f"{url}: short range read {len(payload)} != {end - start + 1}")
    content_range = response.headers.get("Content-Range", "")
    total_text = content_range.rsplit("/", 1)[-1].strip()
    if not total_text.isdigit():
        raise detect.DetectionError(f"{url}: no total size in Content-Range {content_range!r}")
    return payload, int(total_text)


@dataclass
class RemoteShard:
    url: str
    total: int
    data_start: int
    entries: dict[str, dict]


@dataclass
class ShardPlan:
    name: str
    dest: Path
    shard: RemoteShard | None
    entries: dict[str, dict]
    groups: tuple[tuple[int, int], ...]
    repo: str
    revision: str

    @property
    def cached(self) -> bool:
        return self.shard is None


@dataclass
class RangedPlan:
    layer: int
    root: Path
    shards: tuple[ShardPlan, ...]
    bytes_total: int


def _remote_shard(session, url: str, downloads: Downloads, rate: Rate) -> RemoteShard:
    started = time.monotonic()
    first, total = _get_range(session, url, 0, 7)
    header_len = int.from_bytes(first, "little")
    if not 2 <= header_len <= detect.MAX_METADATA_BYTES:
        raise detect.DetectionError(f"{url}: header length {header_len} out of range")
    raw, total_again = _get_range(session, url, 8, 8 + header_len - 1)
    if total_again != total:
        raise detect.DetectionError(f"{url}: inconsistent total size {total} vs {total_again}")
    spent = time.monotonic() - started
    downloads.add(len(first) + len(raw), spent)
    rate.add(len(first) + len(raw), spent)
    try:
        document = json.loads(raw)
    except (ValueError, UnicodeDecodeError, TypeError, RecursionError) as exc:
        raise detect.DetectionError(
            f"{url}: header unparseable: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(document, dict) or not document:
        raise detect.DetectionError(f"{url}: header is not a non-empty object")
    data_start = 8 + header_len
    limit = total - data_start
    if limit <= 0:
        raise detect.DetectionError(f"{url}: header fills the file")
    entries: dict[str, dict] = {}
    for name, meta in document.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(meta, dict):
            raise detect.DetectionError(f"{url}: malformed tensor entry {name!r}")
        dtype = meta.get("dtype")
        shape = meta.get("shape")
        offsets = meta.get("data_offsets")
        if not isinstance(dtype, str) or not dtype:
            raise detect.DetectionError(f"{url}: {name}: bad dtype")
        if not isinstance(shape, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in shape
        ):
            raise detect.DetectionError(f"{url}: {name}: bad shape")
        if not (
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in offsets)
        ):
            raise detect.DetectionError(f"{url}: {name}: bad data_offsets")
        begin, end = int(offsets[0]), int(offsets[1])
        if not 0 <= begin <= end <= limit:
            raise detect.DetectionError(f"{url}: {name}: offsets {begin}..{end} outside {limit}")
        entries[name] = {
            "dtype": dtype,
            "shape": [int(value) for value in shape],
            "data_offsets": [begin, end],
        }
    return RemoteShard(url=url, total=total, data_start=data_start, entries=entries)


def _needed_names(model: ModelPlan, layer: int) -> list[str]:
    arch = model.arch
    if arch["layout"] == "per_expert":
        names = [
            arch["expert_template"].format(layer=layer, expert=expert, role=role)
            for expert in range(model.num_experts)
            for role in arch["roles"]
        ]
    else:
        names = [template.format(layer=layer) for template in arch["batched"].values()]
    names.append(arch["router"].format(layer=layer))
    return names


def _coalesce(entries: dict[str, dict]) -> tuple[tuple[int, int], ...]:
    spans = sorted(meta["data_offsets"] for meta in entries.values())
    groups: list[list[int]] = []
    for begin, end in spans:
        if (
            groups
            and begin - groups[-1][1] <= RANGE_GAP
            and max(end, groups[-1][1]) - groups[-1][0] <= MAX_RANGE_SPAN
        ):
            groups[-1][1] = max(end, groups[-1][1])
        else:
            groups.append([begin, end])
    return tuple((int(begin), int(end)) for begin, end in groups)


def _mini_shard_names(path: Path) -> set[str]:
    try:
        with safe_open(str(path), "pt") as handle:
            return set(handle.keys())
    except Exception as exc:  # noqa: BLE001 - untrusted local bytes
        raise detect.DetectionError(
            f"{path.name}: unreadable mini-shard: {type(exc).__name__}: {exc}"
        ) from exc


def plan_ranged(
    model: ModelPlan, layer: int, session, downloads: Downloads, rate: Rate
) -> RangedPlan:
    """Fetch shard headers and size the tensors this layer needs; no payload yet."""
    needed = _needed_names(model, layer)
    wanted = set(needed)
    scratch = (
        config.SCRATCH / f"{model.repo.replace('/', '__')}@{model.revision[:12]}" / f"L{layer}"
    )
    scratch.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    plans: list[ShardPlan] = []
    for name in model.plan[layer]:
        dest = scratch / name
        covered = _mini_shard_names(dest) & wanted if dest.is_file() else set()
        if covered:
            found |= covered
            shard, entries, groups = None, {}, ()
        else:
            shard = _remote_shard(session, _remote_url(model, name), downloads, rate)
            entries = {key: shard.entries[key] for key in needed if key in shard.entries}
            groups = _coalesce(entries)
            found |= set(entries)
        plans.append(
            ShardPlan(
                name=name,
                dest=dest,
                shard=shard,
                entries=entries,
                groups=groups,
                repo=model.repo,
                revision=model.revision,
            )
        )
    missing = [name for name in needed if name not in found]
    if missing:
        raise detect.DetectionError(
            f"layer {layer}: {len(missing)} needed tensors absent from the shard headers: "
            f"{missing[:3]}"
        )
    total = sum(sum(end - begin for begin, end in plan.groups) for plan in plans if not plan.cached)
    return RangedPlan(layer=layer, root=scratch, shards=tuple(plans), bytes_total=total)


def _write_mini_shard(plan: ShardPlan, session, downloads: Downloads, rate: Rate) -> None:
    shard = plan.shard
    if shard is None:
        return
    layout: list[tuple[str, dict, int, int]] = []
    position = 0
    for name, meta in sorted(plan.entries.items(), key=lambda item: item[1]["data_offsets"][0]):
        begin, end = meta["data_offsets"]
        position += (-position) % 8
        size = end - begin
        layout.append((name, meta, position, position + size))
        position += size
    header = {
        name: {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [begin, end],
        }
        for name, meta, begin, end in layout
    }
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)
    temporary = plan.dest.with_name(plan.dest.name + ".part")
    with temporary.open("wb") as handle:
        handle.write(len(blob).to_bytes(8, "little"))
        handle.write(blob)
        base = 8 + len(blob)
        for group_begin, group_end in plan.groups:
            started = time.monotonic()
            payload, _ = _get_range(
                session,
                shard.url,
                shard.data_start + group_begin,
                shard.data_start + group_end - 1,
            )
            spent = time.monotonic() - started
            downloads.add(len(payload), spent)
            rate.add(len(payload), spent)
            for _name, meta, new_begin, _ in layout:
                begin, end = meta["data_offsets"]
                if group_begin <= begin and end <= group_end:
                    handle.seek(base + new_begin)
                    handle.write(payload[begin - group_begin : end - group_begin])
    temporary.replace(plan.dest)
    downloads.calls.append(
        {
            "repo": plan.repo,
            "revision": plan.revision,
            "mode": "ranged",
            "files": [plan.name],
            "tensors": len(layout),
            "ranges": len(plan.groups),
            "bytes": sum(
                meta["data_offsets"][1] - meta["data_offsets"][0] for _, meta, _, _ in layout
            ),
        }
    )


def run_ranged(plan: RangedPlan, session, downloads: Downloads, rate: Rate) -> Material:
    for shard_plan in plan.shards:
        if shard_plan.cached:
            continue
        _write_mini_shard(shard_plan, session, downloads, rate)
        available = _mini_shard_names(shard_plan.dest)
        missing = sorted(set(shard_plan.entries) - available)
        if missing:
            raise detect.DetectionError(
                f"{shard_plan.dest.name}: mini-shard missing {len(missing)} tensors"
            )
    return Material(root=plan.root, files=tuple(plan_shard.dest.name for plan_shard in plan.shards))


def _reader(material: Material, model: ModelPlan, layer: int, *, batched: bool):
    reader = detect.CheckpointReader(
        root=material.root,
        spec=model.spec(),
        shards={name: name for name in material.files},
    )
    if batched:
        return BatchedExpertView(reader, model.arch, (layer,))
    return reader


# --------------------------------------------------------------------------- #
# label-blind scoring
# --------------------------------------------------------------------------- #
def score(
    *,
    subject: str,
    subject_rev: str,
    reference: str,
    reference_rev: str,
    stack: str,
    layers: tuple[int, ...] | None,
    budget: Budget,
    analysis_estimate: float,
    rate: Rate,
) -> dict:
    """Score one pair, label-blind.

    The only inputs are the subject and reference repo ids, their pinned
    revisions, the stack string, and ``layers``: the layer sample that
    ``config.SAMPLING`` pre-registers per pair before the run. The sample fixes
    which layers are attempted; it is never derived from the pair's expected
    label or from a result. The pair id, the expected label, the evidence block
    and the recipe never reach this function.
    """
    started = time.monotonic()
    downloads = Downloads()
    session = _session()
    record: dict = {
        "stack": stack,
        "outcome": "skipped",
        "reason": None,
        "subject": {"repo": subject, "revision": subject_rev},
        "reference": {"repo": reference, "revision": reference_rev},
        "sampled_layers": [],
        "skipped_layers": [],
        "layers": [],
        "downloads": downloads.payload(),
    }
    try:
        parent = open_model(reference, reference_rev, layers)
        suspect = open_model(subject, subject_rev, layers)
    except detect.DetectionError as exc:
        record["outcome"] = "unavailable"
        record["reason"] = str(exc)
        record["downloads"] = downloads.payload()
        return record
    record["structure"] = {
        "reference": parent.describe(),
        "subject": suspect.describe(),
    }
    ordered = sorted(suspect.plan)
    record["sampled_layers"] = ordered
    batched = parent.arch["layout"] == "batched" or suspect.arch["layout"] == "batched"
    if (parent.arch["layout"] == "batched") != (suspect.arch["layout"] == "batched"):
        record["outcome"] = "unavailable"
        record["reason"] = "parent and subject disagree on expert storage layout"
        record["downloads"] = downloads.payload()
        return record

    for layer in ordered:
        if budget.remaining() <= config.RESERVE_S:
            record["skipped_layers"].append(
                {
                    "layer": layer,
                    "reason": f"wall-clock budget: {budget.remaining():.0f} s left",
                }
            )
            continue
        if peak_rss_mb() > 0.9 * config.RSS_BUDGET_MB:
            record["skipped_layers"].append(
                {
                    "layer": layer,
                    "reason": f"RSS budget: {peak_rss_mb():.0f} MB resident",
                }
            )
            continue
        speed = rate.bps or config.DEFAULT_SPEED_BPS
        overhead = analysis_estimate + config.RESERVE_S
        full_bytes = sum(
            model.sizes.get(name, DEFAULT_SHARD_BYTES)
            for model in (parent, suspect)
            for name in model.plan[layer]
            if not (model.root / name).is_file()
        )
        materials: dict[str, Material] = {}
        mode = "full-shard"
        try:
            if full_bytes / speed + overhead <= budget.remaining():
                fetch(parent, parent.plan[layer], downloads, rate)
                fetch(suspect, suspect.plan[layer], downloads, rate)
                materials = {
                    "reference": Material(parent.root, parent.plan[layer]),
                    "subject": Material(suspect.root, suspect.plan[layer]),
                }
            else:
                mode = "ranged"
                plans = {
                    "reference": plan_ranged(parent, layer, session, downloads, rate),
                    "subject": plan_ranged(suspect, layer, session, downloads, rate),
                }
                ranged_bytes = sum(plan.bytes_total for plan in plans.values())
                projected = ranged_bytes / speed + overhead
                if projected > budget.remaining():
                    raise BudgetSkip(
                        f"budget projection: ranged read {ranged_bytes / 1e9:.2f} GB at "
                        f"{speed / 1e6:.1f} MB/s + {analysis_estimate:.0f} s analysis + "
                        f"{config.RESERVE_S:.0f} s reserve = {projected:.0f} s > "
                        f"{budget.remaining():.0f} s left (full shards would be "
                        f"{full_bytes / 1e9:.2f} GB)"
                    )
                materials = {
                    role: run_ranged(plan, session, downloads, rate) for role, plan in plans.items()
                }
        except BudgetSkip as exc:
            record["skipped_layers"].append({"layer": layer, "reason": str(exc)})
            continue
        except detect.DetectionError as exc:
            record["skipped_layers"].append({"layer": layer, "reason": str(exc)})
            continue

        layer_started = time.monotonic()
        parent_reader = suspect_reader = None
        try:
            parent_reader = _reader(materials["reference"], parent, layer, batched=batched)
            suspect_reader = _reader(materials["subject"], suspect, layer, batched=batched)
            result = detect.analyze_layer(
                parent_reader,
                layer,
                parent.num_experts,
                suspect_reader,
                layer,
                suspect.num_experts,
                config.SEED,
            )
        except detect.DetectionError as exc:
            record["layers"].append({"layer": layer, "abstained": True, "reason": str(exc)})
        else:
            result["download_mode"] = mode
            result["elapsed_s"] = round(time.monotonic() - layer_started, 2)
            result["peak_rss_mb"] = round(peak_rss_mb(), 1)
            record["layers"].append(result)
            analysis_estimate = (
                0.5 * analysis_estimate + 0.5 * result["elapsed_s"]
                if result["elapsed_s"]
                else analysis_estimate
            )
        finally:
            for view in (parent_reader, suspect_reader):
                if isinstance(view, BatchedExpertView):
                    view.clear_cache()

    scored = [layer for layer in record["layers"] if not layer.get("abstained")]
    record["outcome"] = "scored" if scored else "skipped"
    if not scored:
        reasons = [entry["reason"] for entry in record["skipped_layers"]]
        reasons += [entry["reason"] for entry in record["layers"] if entry.get("abstained")]
        record["reason"] = reasons[0] if reasons else "no layer could be scored"
    record["downloads"] = downloads.payload()
    record["wall_s"] = round(time.monotonic() - started, 2)
    return record


# --------------------------------------------------------------------------- #
# harness-level judging (runs after scoring, never feeds it)
# --------------------------------------------------------------------------- #
def _scored(record: dict) -> list[dict]:
    return [layer for layer in record["layers"] if not layer.get("abstained")]


def _max_gain(layers: list[dict]) -> float | None:
    gains = [
        entry["gain"]
        for layer in layers
        for entry in layer.get("classifications", [])
        if entry.get("gain") is not None
    ]
    return max(gains) if gains else None


def _class_counts(layer: dict) -> dict[str, int]:
    labels = [entry["label"] for entry in layer.get("classifications", [])]
    return {
        "exact": labels.count("exact"),
        "merged": labels.count("merged"),
        "unexplained": labels.count("unexplained"),
        "duplicates": sum(
            1 for entry in layer.get("classifications", []) if entry.get("duplicate_of") is not None
        ),
    }


def judge(pair_id: str, record: dict) -> dict:
    layers = _scored(record)
    checks: list[dict] = []

    def add(name: str, held: bool, observed: object, expected: str) -> None:
        checks.append(
            {
                "name": name,
                "held": bool(held),
                "observed": observed,
                "expected": expected,
            }
        )

    abstained = [entry["layer"] for entry in record["layers"] if entry.get("abstained")]
    add(
        "no structural abstention on attempted layers",
        bool(layers) and not abstained,
        abstained,
        "no DetectionError on any attempted layer",
    )
    if pair_id == "pb-core-granite-reap-025":
        add(
            "parent experts == 32, suspect experts == 24",
            bool(layers)
            and all(
                layer["parent_experts"] == 32 and layer["suspect_experts"] == 24 for layer in layers
            ),
            sorted({(layer["parent_experts"], layer["suspect_experts"]) for layer in layers}),
            "(32, 24) everywhere",
        )
        add(
            "matched == 24",
            bool(layers) and all(layer["matched"] == 24 for layer in layers),
            sorted({layer["matched"] for layer in layers}),
            "24",
        )
        add(
            "missing == 8",
            bool(layers) and all(len(layer["missing"]) == 8 for layer in layers),
            sorted({len(layer["missing"]) for layer in layers}),
            "8",
        )
        add(
            "new == 0",
            bool(layers) and all(len(layer["new"]) == 0 for layer in layers),
            sorted({len(layer["new"]) for layer in layers}),
            "0",
        )
        add(
            "no merge above gain 0.30",
            bool(layers) and all(not layer["merges"] for layer in layers),
            {
                "max_gain": _max_gain(layers),
                "merges": sum(len(layer["merges"]) for layer in layers),
            },
            "0 merged fits",
        )
        add(
            "no unexplained or duplicate suspect",
            bool(layers)
            and all(
                _class_counts(layer)["unexplained"] == 0 and _class_counts(layer)["duplicates"] == 0
                for layer in layers
            ),
            sorted(
                {
                    (
                        layer["layer"],
                        _class_counts(layer)["unexplained"],
                        _class_counts(layer)["duplicates"],
                    )
                    for layer in layers
                }
            )[:4],
            "0 / 0 in every layer",
        )
    elif pair_id == "pb-core-qwen3-30b-ream":
        add(
            "parent experts == 128, suspect experts == 96",
            bool(layers)
            and all(
                layer["parent_experts"] == 128 and layer["suspect_experts"] == 96
                for layer in layers
            ),
            sorted({(layer["parent_experts"], layer["suspect_experts"]) for layer in layers}),
            "(128, 96) everywhere",
        )
        add(
            "matched in [64, 96]",
            bool(layers) and all(64 <= layer["matched"] <= 96 for layer in layers),
            sorted({layer["matched"] for layer in layers}),
            "64..96 per layer",
        )
        add(
            "missing in [32, 64]",
            bool(layers) and all(32 <= len(layer["missing"]) <= 64 for layer in layers),
            sorted({len(layer["missing"]) for layer in layers}),
            "32..64 per layer",
        )
        add(
            "missing == 128 - matched",
            bool(layers)
            and all(len(layer["missing"]) == 128 - layer["matched"] for layer in layers),
            sorted({(layer["matched"], len(layer["missing"])) for layer in layers}),
            "identity",
        )
        add(
            "merged fits in [0, 32]",
            bool(layers) and all(0 <= len(layer["merges"]) <= 32 for layer in layers),
            sorted({len(layer["merges"]) for layer in layers}),
            "0..32 per layer",
        )
        add(
            "every suspect classified: exact + merged + unexplained == 96",
            bool(layers)
            and all(
                len(layer.get("classifications", [])) == 96
                and all(
                    entry["label"] in {"exact", "merged", "unexplained"}
                    for entry in layer["classifications"]
                )
                and _class_counts(layer)["exact"]
                + _class_counts(layer)["merged"]
                + _class_counts(layer)["unexplained"]
                == 96
                for layer in layers
            ),
            sorted(
                {
                    (
                        len(layer.get("classifications", [])),
                        _class_counts(layer)["exact"],
                        _class_counts(layer)["merged"],
                        _class_counts(layer)["unexplained"],
                    )
                    for layer in layers
                }
            ),
            "(96 suspects: exact, merged, unexplained)",
        )
    if not layers:
        verdict = "ABSTAIN"
    else:
        verdict = "HELD" if checks and all(check["held"] for check in checks) else "MISSED"
    coverage = {
        "sampled": record["sampled_layers"],
        "scored": [layer["layer"] for layer in layers],
        "skipped": record["skipped_layers"],
    }
    return {
        "verdict": verdict,
        "checks": checks,
        "coverage": coverage,
        "prereg": config.PREREG[pair_id],
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _short(values: list[int], limit: int = 8) -> str:
    if not values:
        return "-"
    head = ", ".join(str(value) for value in values[:limit])
    rest = len(values) - limit
    return f"[{head}{f', +{rest} more' if rest > 0 else ''}]"


def _merge_text(merges: list[dict], limit: int = 3) -> str:
    if not merges:
        return "-"
    parts = []
    for entry in merges[:limit]:
        alpha = ", ".join(f"{value:.3f}" for value in entry["alpha"])
        gain = "null" if entry["gain"] is None else f"{entry['gain']:.4f}"
        parts.append(f"p{entry['suspect']}<-({alpha}) gain {gain}")
    if len(merges) > limit:
        parts.append(f"+{len(merges) - limit} more")
    return "; ".join(parts)


def print_pair(pair_id: str, record: dict) -> None:
    recipe = config.RECIPE[pair_id]
    print(f"\n== {pair_id} [{record['outcome']}] {recipe['claim']}")
    print(
        f"   subject {record['subject']['repo']}@{record['subject']['revision'][:8]}  "
        f"reference {record['reference']['repo']}@{record['reference']['revision'][:8]}"
    )
    if record.get("structure"):
        ref = record["structure"]["reference"]
        sus = record["structure"]["subject"]
        print(
            f"   reference {ref['model_type']}/{ref['layout']} {ref['num_hidden_layers']}L x "
            f"{ref['num_experts']}E; subject {sus['model_type']}/{sus['layout']} "
            f"{sus['num_hidden_layers']}L x {sus['num_experts']}E"
        )
    if record["reason"]:
        print(f"   SKIP: {record['reason']}")
    for entry in record["layers"]:
        if entry.get("abstained"):
            print(f"   L{entry['layer']:<2} ABSTAIN  {entry['reason']}")
            continue
        counts = _class_counts(entry)
        max_gain = max(
            (item["gain"] for item in entry["classifications"] if item.get("gain") is not None),
            default=None,
        )
        print(
            f"   L{entry['layer']:<2} matched={entry['matched']}/{entry['parent_experts']} "
            f"missing={_short(entry['missing'])} new={_short(entry['new'])} "
            f"merged={len(entry['merges'])} unexplained={counts['unexplained']} "
            f"dups={counts['duplicates']} max_cos={entry['max_pair_cos']:.6f} "
            f"max_gain={'null' if max_gain is None else f'{max_gain:.4f}'} "
            f"{entry['elapsed_s']:.1f}s"
        )
        if entry["merges"]:
            print(f"      merges: {_merge_text(entry['merges'])}")
    for entry in record["skipped_layers"]:
        print(f"   L{entry['layer']:<2} SKIP     {entry['reason']}")
    downloads = record["downloads"]
    print(
        f"   downloads {downloads['bytes'] / 1e9:.2f} GB in {downloads['seconds']:.0f} s; "
        f"pair wall {record.get('wall_s', 0):.0f} s"
    )
    judgement = record.get("judgement")
    if judgement:
        coverage = judgement["coverage"]
        print(
            f"   verdict {judgement['verdict']} "
            f"({len(coverage['scored'])}/{len(coverage['sampled'])} sampled layers scored; "
            f"prereg: {config.PREREG[pair_id]})"
        )
        for check in judgement["checks"]:
            print(
                f"     [{'HELD' if check['held'] else 'MISSED'}] {check['name']}: "
                f"observed {check['observed']} (expected {check['expected']})"
            )


def plain(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if hasattr(value, "item"):
        return plain(value.item())
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", choices=sorted(config.SAMPLING))
    parser.add_argument("--budget-seconds", type=float, default=config.BUDGET_S)
    args = parser.parse_args(argv)
    pair_ids = args.pair or list(config.SAMPLING)
    budget = Budget(deadline=time.monotonic() + args.budget_seconds)
    started = time.monotonic()

    pairs: list[dict] = []
    errors = 0
    rate = Rate()
    for pair_id in pair_ids:
        pair = config.load_pair(pair_id)
        estimate = config.ANALYSIS_DEFAULT_S[pair_id]
        try:
            record = score(
                subject=pair["subject"],
                subject_rev=pair["subject_rev"],
                reference=pair["reference"],
                reference_rev=pair["reference_rev"],
                stack=pair["stack"],
                layers=config.SAMPLING[pair_id],
                budget=budget,
                analysis_estimate=estimate,
                rate=rate,
            )
        except Exception as exc:  # noqa: BLE001 - an unexpected bug is a process failure
            errors += 1
            record = {
                "outcome": "error",
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=4),
                "layers": [],
                "skipped_layers": [],
                "sampled_layers": [],
            }
        record["id"] = pair_id
        record["judgement"] = judge(pair_id, record)
        pairs.append(record)

    payload = {
        "schema": "moe-real-pairs/1",
        "issue": 68,
        "run": {
            "utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "seed": config.SEED,
            "budget_s": args.budget_seconds,
            "rss_budget_mb": config.RSS_BUDGET_MB,
            "thresholds": {
                "block": detect.BLOCK,
                "exact_cos": detect.EXACT_COS,
                "near_exact_cos": detect.NEAR_EXACT_COS,
                "merge_gain_min": detect.MERGE_GAIN_MIN,
                "min_component": detect.MIN_COMPONENT,
            },
            "wall_s": round(time.monotonic() - started, 2),
            "peak_rss_mb": round(peak_rss_mb(), 1),
        },
        "pairs": pairs,
    }
    config.RESULTS.write_text(json.dumps(plain(payload), indent=1, allow_nan=False) + "\n")

    print(
        f"thresholds frozen in detect.py: exact_cos={detect.EXACT_COS} "
        f"merge_gain_min={detect.MERGE_GAIN_MIN} min_component={detect.MIN_COMPONENT} "
        f"block={detect.BLOCK}; seed={config.SEED}"
    )
    for record in pairs:
        print_pair(record["id"], record)
    print(
        f"\nwall {payload['run']['wall_s']:.0f} s of {args.budget_seconds:.0f} s budget; "
        f"peak RSS {payload['run']['peak_rss_mb']:.0f} MB of {config.RSS_BUDGET_MB} MB"
    )
    print(f"results: {config.RESULTS}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
