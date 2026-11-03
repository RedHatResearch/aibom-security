"""Executor for the ProvenanceBench extraction cache (issue #62, item 1).

``detect.py`` is frozen and is loaded by path at run time, so the cache is a harness-only
executor: it wraps two seams of the *loaded module object* -- ``_load_gate_tensors``
(Tier A, provenance replay) and ``_svd_spectrum`` (Tier B, content identity) -- and never
edits the file. The artifact's ``method_sha256`` covers ``detect.py`` alone, which is what
keeps a cached run comparable to an uncached one; the executor's own identity, the store's
contract stamp and the hit/miss counters are disclosed beside the artifact in
``<stem>.fastpath.json`` instead of inside ``meta``. A store row is bound to the same
detector sha: a row written by a different ``detect.py`` is a miss, not a replay, because
a detector edit can change what an extraction means.

Tier A replays ``Extraction``/``TensorSpectrum`` from stored fp32 bits and reads no weight
bytes. Only a normal return from the original call is ever persisted, so a
``FormatUnreadable`` abstain or any raise leaves the store untouched, and an unpinned
``revision=None`` never touches Tier A at all. Tier B stores the fp32 bits the frozen call
produced, so a hit is the exact fp64 widening that call returned.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from pbench.fastpath_store import SpectrumStore, tensor_entry

FASTPATH_SPEC = "pbench-fastpath/1"
ENV_DISABLE = "PB_NO_FASTPATH"
ENV_STORE_ROOT = "PB_FASTPATH_STORE"
SEAMS = ("_load_gate_tensors", "_svd_spectrum")

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_STORE_SOURCE = Path(__file__).with_name("fastpath_store.py")


def disabled() -> bool:
    """True when this run must use the detector's own seams (``PB_NO_FASTPATH``)."""
    return os.environ.get(ENV_DISABLE, "").strip().lower() in _TRUTHY


def default_store_root(bench_root: Path) -> Path:
    """``$PB_FASTPATH_STORE`` when set, else ``<bench_root>/.cache/fastpath``."""
    override = os.environ.get(ENV_STORE_ROOT, "").strip()
    if override:
        return Path(override).expanduser()
    return Path(bench_root) / ".cache" / "fastpath"


def sidecar_path(method_path: Path) -> Path:
    """Companion path: ``<stem>.method.jsonl`` becomes ``<stem>.fastpath.json``."""
    path = Path(method_path)
    stem = path.name.removesuffix(".method.jsonl")
    return path.with_name(f"{stem}.fastpath.json")


def write_sidecar(path: Path, body: dict) -> None:
    """Write the executor sidecar atomically (sorted keys, trailing newline)."""
    text = json.dumps(body, indent=2, sort_keys=True) + "\n"
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class FastPath:
    """One installed executor: the two wrapped seams and the store they share."""

    def __init__(
        self,
        detect: ModuleType,
        *,
        store: SpectrumStore,
        method_sha256: str,
        original_load: Callable[..., Any],
        original_svd: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self._detect = detect
        self.store = store
        self.method_sha256 = method_sha256
        self.seams: tuple[str, ...] = SEAMS
        self._original_load = original_load
        self._original_svd = original_svd

    # -- seams ----------------------------------------------------------------

    def _load_gate_tensors(self, repo_id: str, revision: str | None) -> Any:
        """Tier A: replay a pinned extraction, else extract once and persist it."""
        if revision is None:
            # The row key must name immutable content; an unpinned read is never cached.
            return self._original_load(repo_id, revision)
        row = self.store.get_extraction(repo_id, revision, self.method_sha256)
        if row is not None:
            return self._replay(row)
        extraction = self._original_load(repo_id, revision)
        # Persist only on a normal return: a FormatUnreadable abstain or any other raise
        # has already skipped this line, so nothing is written for it.
        self._persist(repo_id, revision, extraction)
        return extraction

    def _svd_spectrum(self, matrix: np.ndarray) -> np.ndarray:
        """Tier B: the fp32-bit spectrum of this exact matrix, computed at most once."""
        arr = np.asarray(matrix, dtype=np.float32)
        if arr.ndim != 2:
            # Not a gate matrix: the frozen call owns the error, unchanged.
            return self._original_svd(matrix)
        key = SpectrumStore.spectrum_key(arr)
        stored = self.store.get_spectrum(key)
        if stored is not None:
            return stored
        values = self._original_svd(matrix)
        self.store.put_spectrum(key, values)
        return values

    # -- row codec ------------------------------------------------------------

    def _replay(self, row: dict[str, Any]) -> Any:
        """Rebuild the detector's own ``Extraction`` from a stored row; no I/O."""
        detect = self._detect
        layers: dict[int, list[Any]] = {}
        for layer, tensors in row["layers"].items():
            # One stored tensor is [name, role, shape, offset, count]; see fastpath_store.
            layers[int(layer)] = [
                detect.TensorSpectrum(
                    name=name,
                    role=role,
                    sv=self.store.read_spectrum(int(offset), int(count)),
                    shape=tuple(shape),
                )
                for name, role, shape, offset, count in tensors
            ]
        return detect.Extraction(
            layers=layers,
            family=row["family"],
            format=row["format"],
            source=row["source"],
        )

    def _persist(self, repo_id: str, revision: str, extraction: Any) -> None:
        """Append one extraction's spectra, then record the row that names them.

        A cache write must never fail the benchmark: an ``OSError`` (a full disk, a store
        that went away) abandons further writes for the run and is disclosed in the
        sidecar, while the extraction this call already computed is returned unchanged.
        """
        try:
            self._append_rows(repo_id, revision, extraction)
        except OSError as exc:
            self.store.abandon_writes(f"store write failed ({type(exc).__name__}: {exc})")

    def _append_rows(self, repo_id: str, revision: str, extraction: Any) -> None:
        layers: dict[str, list[list[Any]]] = {}
        for layer, tensors in extraction.layers.items():
            entries: list[list[Any]] = []
            for tensor in tensors:
                offset, count = self.store.append_spectrum(tensor.sv)
                if count == 0:
                    # The index rejects zero-length spectra, so a row naming one would be
                    # quarantined on the next open. Leave this extraction uncached instead;
                    # the bytes already staged above stay unreferenced and are truncated.
                    return
                entries.append(tensor_entry(tensor.name, tensor.role, tensor.shape, offset, count))
            layers[str(int(layer))] = entries
        self.store.put_extraction(
            repo_id,
            revision,
            self.method_sha256,
            {
                "family": extraction.family,
                "format": extraction.format,
                "source": extraction.source,
                "layers": layers,
            },
        )
        self.store.flush()

    # -- disclosure -----------------------------------------------------------

    def snapshot(self, bench_root: Path) -> dict:
        """The sidecar body: executor identity, store contract, counters. No absolute paths."""
        return {
            "spec": FASTPATH_SPEC,
            "enabled": True,
            "fastpath_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "store_sha256": hashlib.sha256(_STORE_SOURCE.read_bytes()).hexdigest(),
            "method_sha256": self.method_sha256,
            "seams": list(self.seams),
            "store": {
                "root": os.path.relpath(self.store.path, Path(bench_root)),
                "read_only": bool(self.store.read_only),
                "probe_profile": self.store.probe_profile,
                "stack": dict(self.store.stack),
                "validation": asdict(self.store.validation),
            },
            "counters": asdict(self.store.counters),
        }

    def close(self) -> None:
        """Flush and release the store."""
        self.store.close()


def _wrap(
    executor: FastPath,
) -> tuple[Callable[..., Any], Callable[[np.ndarray], np.ndarray]]:
    """Build the two seam functions that replace the detector's own attributes."""

    @functools.wraps(executor._original_load)
    def _load_gate_tensors(repo_id: str, revision: str | None = None) -> Any:
        return executor._load_gate_tensors(repo_id, revision)

    @functools.wraps(executor._original_svd)
    def _svd_spectrum(matrix: np.ndarray) -> np.ndarray:
        return executor._svd_spectrum(matrix)

    return _load_gate_tensors, _svd_spectrum


def install(
    detect: ModuleType,
    *,
    method_sha256: str,
    store: SpectrumStore | None = None,
    store_root: Path | None = None,
) -> FastPath | None:
    """Wrap both seams of one loaded detector; ``None`` when caching is off or impossible.

    The original functions are captured before patching, and the original
    ``_svd_spectrum`` is what the store validates its probe rows with -- the stamp must
    describe the call the run actually depends on.

    Without an explicit ``store``/``store_root`` the store is placed beside the detector
    that is being evaluated (``<detector dir>/.cache/fastpath``, or ``$PB_FASTPATH_STORE``),
    so a copied bench tree never shares a cache with the checkout it was copied from. A
    detector loaded from memory has nowhere to put one, and the cache is then simply off.
    """
    if disabled():
        return None
    originals: list[Callable[..., Any]] = []
    for name in SEAMS:
        original = getattr(detect, name, None)
        if not callable(original):
            # Offline fake detectors expose neither seam; caching is simply not applicable.
            return None
        originals.append(original)
    original_load, original_svd = originals

    owned = store is None
    if store is None:
        if store_root is not None:
            root = Path(store_root)
        else:
            # The store sits beside the detector being evaluated, so a copied bench tree
            # (``--root <dir>``) gets its own store instead of writing into the package.
            detector_path = getattr(detect, "__file__", None)
            if detector_path is None:
                return None
            root = default_store_root(Path(detector_path).resolve().parent)
        try:
            store = SpectrumStore.open(root, svd=original_svd)
        except OSError:
            # The store location cannot host a cache (a read-only mount, a path that is a
            # file, a directory we may not write). Caching is optional; the run is not.
            return None

    executor = FastPath(
        detect,
        store=store,
        method_sha256=method_sha256,
        original_load=original_load,
        original_svd=original_svd,
    )
    load_wrapper, spectrum_wrapper = _wrap(executor)
    try:
        setattr(detect, SEAMS[0], load_wrapper)
        setattr(detect, SEAMS[1], spectrum_wrapper)
    except BaseException:
        if owned:
            store.close()
        raise
    return executor
