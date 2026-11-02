"""Storage engine for the ProvenanceBench extraction cache (issue #62, item 1).

Two tiers share one flat ``float32`` blob:

* **Tier A (provenance replay).** ``(repo_id, revision)`` → one serialized
  ``Extraction`` row. Revisions in the manifest are full 40-hex SHAs, so the key names
  immutable content and a replay needs **zero** weight bytes; that is what makes a warm
  run cheap. The row is bound to the ``detect.py`` sha256 that produced it, because a
  detector edit can change what an extraction means (a different detector is a miss).
* **Tier B (content identity).** ``sha256(fp32 matrix bytes):rows x cols`` → its fp32
  singular-value spectrum, so a cold run never SVDs the same matrix twice — across
  sides, across formats that decode to the same fp32 weights, and across runs. The key
  hashes the *decoded* matrix, so a decode change is a new key rather than a stale hit.

**Bit identity.** The detector's ``_svd_spectrum`` returns fp32 singular values widened
to fp64, so storing the fp32 bits and widening on read is exact (deep-dive §2.1); the
full spectrum is stored, not a top-r truncation (N8), so every future ``r`` and tail
statistic is still free. A replay returns the exact bits an earlier call produced; the
store's contract stamp exists to test the antecedent — that the numeric stack did not
change.

**Why the stamp is probes, not a canary (N2, measured).** A fixed-seed Gaussian canary
has demonstrated sensitivity *zero* to the failure that matters: fp64 ``dgesdd`` and
``dgesvd`` agree bitwise on 200/200 Gaussian inputs, and the transposed path that really
differs flips no fp32 value on a Gaussian 64²/256²/1024². So the store records probe
*rows* that include ill-conditioned, rank-deficient and real-cohort-shape inputs, and
recomputes them at open with the detector's own frozen call. Any mismatch quarantines and
rebuilds the store. A Gaussian row is kept only as the cheapest pre-check and is
labelled ``blind`` in the index so nobody mistakes a green canary for a proof.

**Store hygiene (deep-dive §7.1).** The caller decides what to write, so a failed or
abstained extraction is never persisted. The index is replaced atomically and is
authoritative: an interrupted append leaves unreferenced blob bytes, which the next open
truncates. One writer at a time; a second process opens read-only and skips writes.
Stored spectra are never handed out by reference, so a consumer cannot poison the store.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SPEC = "pbench-fastpath-store/1"
INDEX_NAME = "index.json"
BLOB_NAME = "spectra.f32"
LOCK_NAME = "store.lock"

_F32 = np.dtype("<f4")
_Svd = Callable[[np.ndarray], np.ndarray]
_ProbeBuilder = Callable[[tuple[int, int]], np.ndarray]

#: An fp32 spectrum widened to fp64 is exactly the detector's ``_svd_spectrum`` result.
_BLIND_PROBE = "gauss-64x64"


class StoreRefused(RuntimeError):
    """The store failed validation and this process may not rebuild it."""


@dataclass(frozen=True)
class Validation:
    """Outcome of the open-time contract check (disclosed in the run sidecar)."""

    status: str
    probes_checked: int
    probes_failed: tuple[str, ...]
    detail: str


@dataclass
class Counters:
    """Hit/miss accounting for one run (disclosed in the run sidecar)."""

    extraction_hits: int = 0
    extraction_misses: int = 0
    extraction_rejected_detect_sha: int = 0
    spectrum_hits: int = 0
    spectrum_misses: int = 0
    spectra_written: int = 0
    flush_count: int = 0


def _probe_gauss(shape: tuple[int, int]) -> np.ndarray:
    """Fixed-seed Gaussian: the cheapest pre-check, blind to driver and orientation."""
    return np.random.default_rng(0).standard_normal(shape, dtype=np.float32)


def _probe_hilbert(shape: tuple[int, int]) -> np.ndarray:
    """Hilbert matrix ``1/(i+j+1)``: exactly-rounded entries, condition grows fast."""
    rows = np.arange(shape[0], dtype=np.float64)[:, None]
    cols = np.arange(shape[1], dtype=np.float64)[None, :]
    return (1.0 / (rows + cols + 1.0)).astype(np.float32)


def _probe_rank1(shape: tuple[int, int]) -> np.ndarray:
    """Rank-one outer product: exactly one nonzero singular value."""
    rows = np.arange(1, shape[0] + 1, dtype=np.float32)[:, None]
    cols = np.arange(1, shape[1] + 1, dtype=np.float32)[None, :]
    return rows * cols


def _probe_zeros(shape: tuple[int, int]) -> np.ndarray:
    """All-zero matrix: the degenerate path every driver special-cases."""
    return np.zeros(shape, dtype=np.float32)


def _probe_grid(shape: tuple[int, int]) -> np.ndarray:
    """Integer-arithmetic matrix at a real cohort gate shape (no RNG, no LAPACK).

    LAPACK block and workspace choices depend on the shape, so a probe at
    768x2048 and 2880x2880 samples paths that 64² and 256² never reach.
    """
    rows = np.arange(shape[0], dtype=np.int64)[:, None]
    cols = np.arange(shape[1], dtype=np.int64)[None, :]
    return (((rows * 7 + cols * 13) % 17) - 8).astype(np.float32)


_PROBE_BUILDERS: dict[str, tuple[tuple[int, int], _ProbeBuilder]] = {
    "gauss-64x64": ((64, 64), _probe_gauss),
    "hilbert-64x64": ((64, 64), _probe_hilbert),
    "hilbert-256x256": ((256, 256), _probe_hilbert),
    "rank1-96x192": ((96, 192), _probe_rank1),
    "zero-32x48": ((32, 48), _probe_zeros),
    "grid-768x2048": ((768, 2048), _probe_grid),
    "grid-2880x2880": ((2880, 2880), _probe_grid),
}

#: ``full`` is what a benchmark run records. ``fast`` drops the two slow rows and exists
#: for offline tests; validation always recomputes whatever the store recorded.
_PROBE_PROFILES: dict[str, tuple[str, ...]] = {
    "full": (
        "gauss-64x64",
        "hilbert-64x64",
        "hilbert-256x256",
        "rank1-96x192",
        "zero-32x48",
        "grid-768x2048",
        "grid-2880x2880",
    ),
    "fast": (
        "gauss-64x64",
        "hilbert-64x64",
        "rank1-96x192",
        "zero-32x48",
        "grid-768x2048",
    ),
}


def default_svd(matrix: np.ndarray) -> np.ndarray:
    """The frozen detector call, for stores opened without the detector's own function."""
    return np.linalg.svd(np.asarray(matrix, dtype=np.float32), compute_uv=False).astype(np.float64)


def spectrum_digest(values: np.ndarray) -> str:
    """sha256 of one spectrum's stored (fp32) bits; the probe comparison unit."""
    return hashlib.sha256(np.ascontiguousarray(values, dtype=_F32).tobytes()).hexdigest()


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _fsync_dir(path: Path) -> None:
    """Make a rename durable. Not every filesystem allows fsync on a directory."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class SpectrumStore:
    """One on-disk two-tier store; see the module docstring for the contract."""

    def __init__(
        self,
        root: Path,
        *,
        svd: _Svd | None = None,
        probe_profile: str = "full",
    ) -> None:
        if probe_profile not in _PROBE_PROFILES:
            raise ValueError(f"unknown probe profile {probe_profile!r}")
        self.path = Path(root)
        self.probe_profile = probe_profile
        self.stack = {
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        }
        self.counters = Counters()
        self.path.mkdir(parents=True, exist_ok=True)
        self._index_path = self.path / INDEX_NAME
        self._blob_path = self.path / BLOB_NAME
        self._svd = svd or default_svd
        self._lock_fd: int | None = None
        self.read_only = not self._take_lock()
        self._pending: list[np.ndarray] = []
        self._pending_index: dict[int, np.ndarray] = {}
        self._dirty = False
        self._blob_elements = 0
        self._durable_elements = 0
        self._closed = False
        self._index: dict[str, Any] = {}
        self._abandoned: str | None = None
        # Assigned before any flush: a lock check inside the very first flush can abandon
        # writes, and abandon_writes rewrites this field.
        self.validation = Validation("empty", 0, (), "not yet validated")
        self._fd = -1
        try:
            self._fd = self._open_blob()
            self.validation = self._load_or_create()
        except BaseException:
            self._release()
            raise

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        svd: _Svd | None = None,
        probe_profile: str = "full",
    ) -> SpectrumStore:
        """Open (or create) a store, validating its recorded probe rows."""
        return cls(root, svd=svd, probe_profile=probe_profile)

    # -- construction helpers -------------------------------------------------

    def _take_lock(self) -> bool:
        try:
            fd = os.open(self.path / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._lock_fd = fd
        return True

    def _open_blob(self) -> int:
        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT
        try:
            return os.open(self._blob_path, flags, 0o644)
        except OSError:
            self.read_only = True
            return os.open(self._blob_path, os.O_RDONLY | os.O_CREAT, 0o644)

    def _empty_index(self) -> dict[str, Any]:
        return {
            "spec": SPEC,
            "created_utc": _stamp(),
            "stack": self.stack,
            "probes": {"profile": self.probe_profile, "blind": [_BLIND_PROBE], "rows": []},
            "blob_elements": 0,
            "extractions": {},
            "spectra": {},
        }

    def _validated(
        self, status: str, checked: int, failed: Sequence[str], detail: str
    ) -> Validation:
        """The validation to publish, unless a flush during open already abandoned writes."""
        if self._abandoned is not None:
            return self.validation
        return Validation(
            status=status, probes_checked=checked, probes_failed=tuple(failed), detail=detail
        )

    def _load_or_create(self) -> Validation:
        if not self.read_only:
            # Only the lock holder writes the tmp; a leftover one is debris from a crash.
            self._index_path.with_name(f"{INDEX_NAME}.tmp").unlink(missing_ok=True)
        if not self._index_path.exists():
            if self.read_only:
                # Another process may be creating it right now; every lookup misses and
                # nothing is written. Never pretend this is a fresh, populated store.
                self._index = self._empty_index()
                return self._validated(
                    "empty", 0, (), "no index on disk; opened read-only, so nothing was written"
                )
            self._index = self._empty_index()
            self._record_probes()
            self._dirty = True
            self.flush()
            return self._validated(
                "new",
                len(self._index["probes"]["rows"]),
                (),
                "store created; probe rows recorded",
            )
        try:
            index = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return self._rebuild(f"index unreadable ({type(exc).__name__}: {exc})")
        if not isinstance(index, dict):
            return self._rebuild("index is not a JSON object")
        if index.get("spec") != SPEC:
            return self._rebuild(f"index spec {index.get('spec')!r} is not {SPEC!r}")
        try:
            self._adopt_index(index)
            failed, checked = self._revalidate(index)
        except (KeyError, TypeError, ValueError) as exc:
            return self._rebuild(f"index malformed ({type(exc).__name__}: {exc})")
        if failed:
            return self._rebuild(
                "probe mismatch after a numeric-stack change: " + ", ".join(failed),
                failed=failed,
            )
        return self._validated(
            "ok", checked, (), f"{checked} recorded probe rows recomputed; all match"
        )

    def _adopt_index(self, index: dict[str, Any]) -> None:
        blob_elements = index["blob_elements"]
        if not isinstance(blob_elements, int) or blob_elements < 0:
            raise ValueError("blob_elements")
        extractions = index["extractions"]
        spectra = index["spectra"]
        if not isinstance(extractions, dict) or not isinstance(spectra, dict):
            raise TypeError("extraction/spectrum tables must be objects")
        probes = index.get("probes")
        recorded_profile = probes.get("profile") if isinstance(probes, dict) else None
        if not isinstance(probes, dict) or recorded_profile != self.probe_profile:
            # A store recorded under another profile cannot be validated as this one:
            # opening a "fast" store as "full" (or the reverse) must rebuild, never
            # report a green stamp over rows the profile never recorded.
            raise ValueError(f"probe profile {recorded_profile!r} is not {self.probe_profile!r}")
        probe_rows = probes.get("rows")
        expected_ids = set(_PROBE_PROFILES[self.probe_profile])
        recorded_ids = (
            {row["id"] for row in probe_rows if isinstance(row, dict) and "id" in row}
            if isinstance(probe_rows, list)
            else set()
        )
        if recorded_ids != expected_ids:
            # A truncated row list keeps the profile string but silently downgrades the
            # stamp; the recorded set must be exactly the profile's set.
            raise ValueError(
                f"probe rows {sorted(recorded_ids)} are not the {self.probe_profile!r} set"
            )
        for key, entry in spectra.items():
            offset, count = entry
            if not (
                isinstance(offset, int)
                and isinstance(count, int)
                and offset >= 0
                and count > 0
                and offset + count <= blob_elements
            ):
                raise ValueError(f"spectrum row {key!r} is outside the blob")
        for key, row in extractions.items():
            if not isinstance(row, dict) or not isinstance(row.get("layers"), dict):
                raise TypeError(f"extraction row {key!r} has no layer table")
            for layer_key, tensors in row["layers"].items():
                try:
                    int(layer_key)
                except (TypeError, ValueError) as exc:
                    raise TypeError(f"extraction row {key!r} has layer {layer_key!r}") from exc
                if not isinstance(tensors, list):
                    raise TypeError(f"extraction row {key!r} layer {layer_key!r} is not a list")
                for entry in tensors:
                    name, role, shape, offset, count = entry
                    if not isinstance(name, str) or not isinstance(role, str):
                        raise TypeError(f"extraction row {key!r} has a non-string name/role")
                    if not isinstance(shape, list) or not all(
                        isinstance(size, int) for size in shape
                    ):
                        raise TypeError(f"extraction row {key!r} has a malformed shape")
                    if not (
                        isinstance(offset, int)
                        and isinstance(count, int)
                        and offset >= 0
                        and count > 0
                        and offset + count <= blob_elements
                    ):
                        raise ValueError(f"extraction row {key!r} is outside the blob")
        size = self._blob_path.stat().st_size if self._blob_path.exists() else 0
        expected = blob_elements * 4
        if size < expected:
            raise ValueError(f"blob holds {size} bytes, the index references {expected}")
        if size > expected and not self.read_only:
            # An interrupted flush: unreferenced tail bytes are never read.
            os.ftruncate(self._fd, expected)
        self._index = index
        self._blob_elements = blob_elements
        self._durable_elements = blob_elements

    def _record_probes(self) -> None:
        rows: list[dict[str, Any]] = []
        for probe_id in _PROBE_PROFILES[self.probe_profile]:
            shape, build = _PROBE_BUILDERS[probe_id]
            rows.append(
                {
                    "id": probe_id,
                    "shape": list(shape),
                    "sha256": spectrum_digest(np.asarray(self._svd(build(shape)))),
                }
            )
        self._index["probes"]["rows"] = rows

    def _revalidate(self, index: dict[str, Any]) -> tuple[list[str], int]:
        """Recompute every recorded probe row; return (failed ids, checked count)."""
        probes = index.get("probes")
        rows = probes.get("rows") if isinstance(probes, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError("no probe rows recorded")
        failed: list[str] = []
        checked = 0
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError("probe row is not an object")
            probe_id = row["id"]
            expected = row["sha256"]
            if not isinstance(probe_id, str) or not isinstance(expected, str):
                raise TypeError("probe row id/sha256 must be strings")
            checked += 1
            entry = _PROBE_BUILDERS.get(probe_id)
            if entry is None:
                failed.append(f"{probe_id} (unknown probe)")
                continue
            shape, build = entry
            actual = spectrum_digest(np.asarray(self._svd(build(shape))))
            if actual != expected:
                failed.append(probe_id)
        return failed, checked

    def _rebuild(self, reason: str, *, failed: Sequence[str] = ()) -> Validation:
        if self.read_only:
            raise StoreRefused(
                f"{self.path}: {reason}; the store is read-only in this process "
                "(another writer holds the lock), so it cannot be rebuilt"
            )
        stamp = _stamp()
        quarantined: list[str] = []
        for path in (self._index_path, self._blob_path):
            if not path.exists():
                continue
            target = path.with_name(f"{path.name}.refused-{stamp}")
            os.replace(path, target)
            quarantined.append(target.name)
        if self._fd >= 0:
            os.close(self._fd)
        self._fd = self._open_blob()
        self._blob_elements = 0
        self._durable_elements = 0
        self._pending.clear()
        self._pending_index.clear()
        self._index = self._empty_index()
        self._record_probes()
        self._dirty = True
        self.flush()
        _fsync_dir(self.path)
        return self._validated(
            "rebuilt",
            len(self._index["probes"]["rows"]),
            failed,
            f"rebuilt: {reason}; quarantined {', '.join(quarantined) or 'nothing'}",
        )

    # -- spectra --------------------------------------------------------------

    @staticmethod
    def spectrum_key(matrix: np.ndarray) -> str:
        """Content key: sha256 of the canonical fp32 bytes, plus the gate shape.

        Canonical C-order is deliberate: numpy copies to its own Fortran-order buffer
        before LAPACK, so element-wise identical inputs have identical spectra whatever
        their layout (measured, deep-dive §2.1) — and a key must not split them.
        """
        arr = np.asarray(matrix, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"spectrum keys need a 2-D matrix, got shape {arr.shape}")
        flat = np.ascontiguousarray(arr)
        digest = hashlib.sha256(memoryview(flat).cast("B")).hexdigest()
        return f"{digest}:{arr.shape[0]}x{arr.shape[1]}"

    def append_spectrum(self, values: Sequence[float] | np.ndarray) -> tuple[int, int]:
        """Reserve ``(offset, count)`` for one spectrum; the caller embeds the pair."""
        arr = np.ascontiguousarray(values, dtype=_F32).reshape(-1)
        offset, count = self._blob_elements, int(arr.size)
        if self.read_only or count == 0:
            return offset, count
        self._blob_elements += count
        self._pending.append(arr)
        self._pending_index[offset] = arr
        self._dirty = True
        return offset, count

    def read_spectrum(self, offset: int, count: int) -> np.ndarray:
        """Fresh fp64 array holding the stored fp32 bits (the exact detector widening)."""
        if offset < 0 or count <= 0 or offset + count > self._blob_elements:
            raise ValueError(
                f"spectrum range [{offset}, {offset}+{count}) is outside the "
                f"{self._blob_elements}-element blob"
            )
        pending = self._pending_index.get(offset)
        if pending is not None:
            if pending.size < count:
                raise ValueError(f"pending spectrum at {offset} is shorter than {count}")
            return pending[:count].astype(np.float64)
        raw = os.pread(self._fd, count * 4, offset * 4)
        if len(raw) != count * 4:
            raise ValueError(f"short blob read at {offset}: {len(raw)} of {count * 4} bytes")
        return np.frombuffer(raw, dtype=_F32, count=count).astype(np.float64)

    def get_spectrum(self, key: str) -> np.ndarray | None:
        """Stored spectrum for one content key, or ``None`` on a miss."""
        entry = self._index["spectra"].get(key)
        if entry is None:
            self.counters.spectrum_misses += 1
            return None
        offset, count = entry
        self.counters.spectrum_hits += 1
        return self.read_spectrum(offset, count)

    def put_spectrum(self, key: str, values: Sequence[float] | np.ndarray) -> None:
        """Record one content key → spectrum row (idempotent, skipped when read-only)."""
        if self.read_only or key in self._index["spectra"]:
            return
        offset, count = self.append_spectrum(values)
        if count == 0:
            return
        self._index["spectra"][key] = [offset, count]
        self.counters.spectra_written += 1
        self._dirty = True

    # -- extractions ----------------------------------------------------------

    def get_extraction(
        self, repo_id: str, revision: str, detect_sha256: str
    ) -> dict[str, Any] | None:
        """Replay row for one pinned checkpoint, or ``None`` (miss / other detector)."""
        row = self._index["extractions"].get(f"{repo_id}@{revision}")
        if row is None:
            self.counters.extraction_misses += 1
            return None
        if row.get("detect_sha256") != detect_sha256:
            # A detector edit can change what an extraction means; never replay across it.
            self.counters.extraction_rejected_detect_sha += 1
            self.counters.extraction_misses += 1
            return None
        self.counters.extraction_hits += 1
        return copy.deepcopy(row)

    def put_extraction(
        self,
        repo_id: str,
        revision: str,
        detect_sha256: str,
        row: dict[str, Any],
    ) -> None:
        """Persist one extraction row whose spectra were already appended."""
        if self.read_only:
            return
        self._index["extractions"][f"{repo_id}@{revision}"] = {
            **row,
            "detect_sha256": detect_sha256,
        }
        self._dirty = True

    # -- durability -----------------------------------------------------------

    def abandon_writes(self, reason: str) -> None:
        """Stop writing for the rest of the run, keeping every already-published row.

        Used when the store can no longer be trusted to publish safely (a write error, or
        the writer lock file disappearing). The run continues; the sidecar discloses it.

        Rows staged but never published are dropped from the in-memory index: their bytes
        are not in the blob, so serving them would read a hole. Only rows that name the
        durable region survive.
        """
        self._abandoned = reason
        durable = self._durable_elements
        self._index["spectra"] = {
            key: entry
            for key, entry in self._index["spectra"].items()
            if entry[0] >= 0 and entry[0] + entry[1] <= durable
        }
        self._index["extractions"] = {
            key: row
            for key, row in self._index["extractions"].items()
            if all(
                entry[3] >= 0 and entry[3] + entry[4] <= durable
                for tensors in row["layers"].values()
                for entry in tensors
            )
        }
        self.read_only = True
        self._blob_elements = durable
        self._pending.clear()
        self._pending_index.clear()
        self._dirty = False
        self.validation = Validation(
            status=self.validation.status,
            probes_checked=self.validation.probes_checked,
            probes_failed=self.validation.probes_failed,
            detail=f"{self.validation.detail}; writes abandoned: {reason}",
        )

    def _lock_is_intact(self) -> bool:
        """True when our lock fd still names the lock file on disk.

        If `store.lock` is deleted or replaced while we hold it, a second writer can take
        a *new* inode and publish the same index path. Publishing from two live writers is
        the one failure mode that yields silently wrong spectra, so we stop instead.
        """
        if self._lock_fd is None:
            return False
        try:
            held = os.fstat(self._lock_fd)
            on_disk = os.stat(self.path / LOCK_NAME)
        except OSError:
            return False
        return (held.st_dev, held.st_ino) == (on_disk.st_dev, on_disk.st_ino)

    def flush(self) -> None:
        """Append pending spectra, then atomically replace the index that names them.

        Restartable: the blob's durable length is ``_durable_elements * 4``, the only
        region the published index names. A flush that dies mid-append therefore leaves a
        tail that the next flush truncates before rewriting the same payload at the same
        element offsets — a retry can never shift a previously stored spectrum.
        """
        if self.read_only or not self._dirty:
            return
        if not self._lock_is_intact():
            self.abandon_writes("the writer lock file was removed or replaced")
            return
        durable_bytes = self._durable_elements * 4
        if self._pending:
            size = os.fstat(self._fd).st_size
            if size < durable_bytes:
                # Someone truncated the blob under us: refuse rather than zero-fill a gap.
                self.abandon_writes(f"the store blob shrank to {size} bytes below {durable_bytes}")
                return
            if size != durable_bytes:
                os.ftruncate(self._fd, durable_bytes)
            payload = b"".join(
                np.ascontiguousarray(arr, dtype=_F32).tobytes() for arr in self._pending
            )
            view = memoryview(payload)
            written = 0
            while written < len(payload):
                written += os.write(self._fd, view[written:])
            os.fsync(self._fd)
        self._index["blob_elements"] = self._blob_elements
        data = json.dumps(self._index, sort_keys=True, separators=(",", ":")).encode("utf-8")
        tmp = self._index_path.with_name(f"{INDEX_NAME}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            view = memoryview(data)
            written = 0
            while written < len(data):
                written += os.write(fd, view[written:])
            os.fsync(fd)
        finally:
            os.close(fd)
        if not self._lock_is_intact():
            # The window between the first check and the rename: never publish from a
            # writer whose lock file was replaced meanwhile.
            self.abandon_writes("the writer lock file was removed or replaced")
            return
        os.replace(tmp, self._index_path)
        _fsync_dir(self.path)
        self._durable_elements = self._blob_elements
        self._pending.clear()
        self._pending_index.clear()
        self._dirty = False
        self.counters.flush_count += 1

    def _release(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
        self._closed = True

    def close(self) -> None:
        """Flush and release the blob and the writer lock; never raises on a write error."""
        if self._closed:
            return
        try:
            self.flush()
        except OSError as exc:
            self.abandon_writes(f"store close failed ({type(exc).__name__}: {exc})")
        finally:
            self._release()
