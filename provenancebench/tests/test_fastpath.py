"""Offline tests for the ProvenanceBench extraction cache (issue #62, item 1)."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from pbench import fastpath
from pbench.eval import _load_detect, run_eval
from pbench.fastpath_store import (
    INDEX_NAME,
    SPEC,
    SpectrumStore,
    StoreRefused,
    default_svd,
    spectrum_digest,
)
from pbench.load import bench_root, load_cohort


def _open(root: Path, **kwargs: object) -> SpectrumStore:
    kwargs.setdefault("probe_profile", "fast")
    return SpectrumStore.open(root, **kwargs)


@pytest.fixture(autouse=True)
def _isolate_cache_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never inherit a caller's cache configuration: it would share a store across tests.

    ``PB_FASTPATH_STORE`` in the ambient environment (a benchmark shell that exported it)
    otherwise redirects every ``default_store_root`` call at one real store, which makes
    hit counts depend on test order.
    """
    monkeypatch.delenv(fastpath.ENV_STORE_ROOT, raising=False)
    monkeypatch.delenv(fastpath.ENV_DISABLE, raising=False)


def _matrix(rows: int = 6, cols: int = 3, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((rows, cols), dtype=np.float32)


def test_spectrum_key_is_layout_and_shape_canonical() -> None:
    matrix = _matrix()
    same_values_f_order = np.asfortranarray(matrix)
    strided = np.repeat(matrix, 2, axis=0)[::2]

    key = SpectrumStore.spectrum_key(matrix)
    assert SpectrumStore.spectrum_key(same_values_f_order) == key
    assert SpectrumStore.spectrum_key(strided) == key
    assert SpectrumStore.spectrum_key(_matrix(seed=8)) != key
    # The shape is part of the key: 2x6 and 3x4 share a byte count, never an identity.
    assert SpectrumStore.spectrum_key(_matrix(3, 4)) != SpectrumStore.spectrum_key(_matrix(2, 6))
    with pytest.raises(ValueError, match="2-D"):
        SpectrumStore.spectrum_key(np.zeros(4, dtype=np.float32))


def test_store_roundtrips_spectra_and_extractions(tmp_path: Path) -> None:
    store = _open(tmp_path / "store")
    sv = default_svd(_matrix())
    key = SpectrumStore.spectrum_key(_matrix())
    store.put_spectrum(key, sv)
    offset, count = store.append_spectrum(sv)
    row = {
        "family": "dense",
        "format": "safetensors",
        "source": {"repo_id": "a/b", "revision": "0" * 40, "fetch": "range"},
        "layers": {"0": [["model.layers.0.mlp.gate_proj.weight", "dense", [6, 3], offset, count]]},
    }
    store.put_extraction("a/b", "0" * 40, "detector-sha", row)
    store.close()

    reopened = _open(tmp_path / "store")
    assert reopened.validation.status == "ok"
    got = reopened.get_extraction("a/b", "0" * 40, "detector-sha")
    assert got is not None
    tensor = got["layers"]["0"][0]
    assert tensor[0] == "model.layers.0.mlp.gate_proj.weight"
    assert tensor[2] == [6, 3]
    replayed = reopened.read_spectrum(tensor[3], tensor[4])
    assert replayed.dtype == np.float64
    assert replayed.tobytes() == np.ascontiguousarray(sv, dtype="<f4").astype(np.float64).tobytes()
    hit = reopened.get_spectrum(key)
    assert hit is not None
    assert hit.tobytes() == replayed.tobytes()
    assert reopened.get_spectrum("missing") is None
    reopened.close()


def test_pending_spectrum_is_readable_before_flush(tmp_path: Path) -> None:
    store = _open(tmp_path / "store")
    sv = default_svd(_matrix())
    offset, count = store.append_spectrum(sv)
    assert (
        store.read_spectrum(offset, count).tobytes()
        == np.asarray(sv, dtype=np.float32).astype(np.float64).tobytes()
    )
    store.close()


def test_returned_spectrum_is_a_copy(tmp_path: Path) -> None:
    store = _open(tmp_path / "store")
    sv = default_svd(_matrix())
    key = SpectrumStore.spectrum_key(_matrix())
    store.put_spectrum(key, sv)
    store.flush()

    first = store.get_spectrum(key)
    assert first is not None
    first[:] = 0.0
    second = store.get_spectrum(key)
    assert second is not None
    assert second.tobytes() == np.asarray(sv, dtype=np.float32).astype(np.float64).tobytes()

    row = store.get_extraction("nobody/none", "0" * 40, "sha")
    assert row is None
    store.close()


def test_one_ulp_numeric_change_refuses_and_rebuilds(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = _open(root)
    store.close()

    def perturbed(matrix: np.ndarray) -> np.ndarray:
        values = default_svd(matrix)
        # One fp32 ulp: the store's unit. An fp64 ulp would round away and prove nothing.
        values[0] = np.nextafter(np.float32(values[0]), np.float32(np.inf))
        return values

    rebuilt = _open(root, svd=perturbed)
    assert rebuilt.validation.status == "rebuilt"
    assert "gauss-64x64" in rebuilt.validation.probes_failed
    assert "probe mismatch" in rebuilt.validation.detail
    quarantined = sorted(path.name for path in root.glob("*.refused-*"))
    assert quarantined, "a refused store must be kept for forensics, not deleted"
    rebuilt.close()

    # The rebuilt store validates against the stack that rebuilt it.
    assert _open(root, svd=perturbed).validation.status == "ok"


def test_tampered_probe_hash_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = _open(root)
    store.close()

    index_path = root / INDEX_NAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    row = next(row for row in index["probes"]["rows"] if row["id"] == "rank1-96x192")
    row["sha256"] = "0" * 64
    index_path.write_text(json.dumps(index), encoding="utf-8")

    rebuilt = _open(root)
    assert rebuilt.validation.status == "rebuilt"
    assert rebuilt.validation.probes_failed == ("rank1-96x192",)
    rebuilt.close()


def test_torn_blob_tail_is_truncated(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = _open(root)
    store.put_spectrum(SpectrumStore.spectrum_key(_matrix()), default_svd(_matrix()))
    store.close()
    blob = root / "spectra.f32"
    expected = blob.stat().st_size
    with blob.open("ab") as handle:
        handle.write(b"\x01\x02\x03\x04")  # an interrupted flush

    reopened = _open(root)
    assert reopened.validation.status == "ok"
    assert blob.stat().st_size == expected
    assert reopened.counters.spectrum_hits == 0  # nothing was read yet
    reopened.close()


def test_short_blob_refuses_and_rebuilds(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = _open(root)
    store.put_spectrum(SpectrumStore.spectrum_key(_matrix()), default_svd(_matrix()))
    store.close()
    blob = root / "spectra.f32"
    os.truncate(blob, 4)

    rebuilt = _open(root)
    assert rebuilt.validation.status == "rebuilt"
    assert "blob holds" in rebuilt.validation.detail
    rebuilt.close()


def test_corrupt_index_is_quarantined_and_rebuilt(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    (root / INDEX_NAME).write_text("{not json", encoding="utf-8")

    rebuilt = _open(root)
    assert rebuilt.validation.status == "rebuilt"
    assert any(path.name.startswith(f"{INDEX_NAME}.refused-") for path in root.iterdir())
    rebuilt.close()


def test_spec_mismatch_rebuilds(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = _open(root)
    store.close()
    index_path = root / INDEX_NAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["spec"] = "pbench-fastpath-store/0"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    rebuilt = _open(root)
    assert rebuilt.validation.status == "rebuilt"
    assert SPEC in rebuilt.validation.detail
    rebuilt.close()


def test_extraction_from_another_detector_is_a_miss_but_spectra_survive(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    store = _open(root)
    sv = default_svd(_matrix())
    key = SpectrumStore.spectrum_key(_matrix())
    store.put_spectrum(key, sv)
    offset, count = store.append_spectrum(sv)
    store.put_extraction(
        "a/b",
        "0" * 40,
        "old-detector",
        {"family": "dense", "format": "safetensors", "source": {}, "layers": {"0": []}},
    )
    store.close()

    reopened = _open(root)
    assert reopened.get_extraction("a/b", "0" * 40, "old-detector") is not None
    assert reopened.get_extraction("a/b", "0" * 40, "new-detector") is None
    assert reopened.counters.extraction_rejected_detect_sha == 1
    assert reopened.counters.extraction_misses == 1
    assert reopened.get_spectrum(key) is not None
    reopened.close()


def test_lock_contention_opens_read_only(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    lock_fd = os.open(root / "store.lock", os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        reader = _open(root)
        assert reader.read_only is True
        assert reader.validation.status == "empty"
        reader.put_spectrum("key", np.ones(4))
        assert reader.counters.spectra_written == 0
        assert reader.get_spectrum("key") is None
        reader.close()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    writer = _open(root)
    assert writer.read_only is False
    assert writer.validation.status == "new"
    writer.close()


def test_read_only_store_that_cannot_be_rebuilt_raises(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = _open(root)
    store.close()
    index_path = root / INDEX_NAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["probes"]["rows"][0]["sha256"] = "0" * 64
    index_path.write_text(json.dumps(index), encoding="utf-8")

    lock_fd = os.open(root / "store.lock", os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(StoreRefused, match="read-only"):
            _open(root)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def test_full_probe_profile_records_and_revalidates(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = SpectrumStore.open(root, probe_profile="full")
    recorded = {
        row["id"]
        for row in json.loads((root / INDEX_NAME).read_text(encoding="utf-8"))["probes"]["rows"]
    }
    assert {"hilbert-256x256", "grid-2880x2880"} <= recorded
    assert store.validation.status == "new"
    store.close()

    reopened = SpectrumStore.open(root, probe_profile="full")
    assert reopened.validation.status == "ok"
    assert reopened.validation.probes_checked == len(recorded)
    reopened.close()


def test_spectrum_digest_ignores_fp64_noise_below_fp32() -> None:
    """The store's unit is the fp32 bits the detector actually emits."""
    values = np.array([3.0, 2.0, 1.0], dtype=np.float64)
    wider = values.astype(np.longdouble).astype(np.float64)
    assert spectrum_digest(values) == spectrum_digest(wider)
    # An fp64 ulp is below the stored precision and must not split a row...
    assert spectrum_digest(values) == spectrum_digest(np.nextafter(values, np.inf))
    # ...while one fp32 ulp must.
    assert spectrum_digest(values) != spectrum_digest(
        np.nextafter(values.astype(np.float32), np.float32(np.inf))
    )


# ---------------------------------------------------------------------------
# Crash safety and disclosure regressions (found by the correctness review).
# ---------------------------------------------------------------------------


def test_index_without_probe_rows_is_rebuilt_not_crashed(tmp_path: Path) -> None:
    """A JSON-valid index whose probe table is empty must rebuild, not abort the run."""
    root = tmp_path / "store"
    store = _open(root)
    store.close()
    index_path = root / INDEX_NAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["probes"]["rows"] = []
    index_path.write_text(json.dumps(index), encoding="utf-8")

    rebuilt = _open(root)
    assert rebuilt.validation.status == "rebuilt"
    assert rebuilt.validation.probes_failed == ()
    assert "probe rows" in rebuilt.validation.detail
    rebuilt.close()


def test_extraction_row_with_a_bad_layer_key_is_rebuilt(tmp_path: Path) -> None:
    """Adoption validates what the replay will index, so a bad key cannot reach score()."""
    root = tmp_path / "store"
    store = _open(root)
    offset, count = store.append_spectrum(default_svd(_matrix()))
    store.put_extraction(
        "a/b",
        "0" * 40,
        "sha",
        {
            "family": "dense",
            "format": "safetensors",
            "source": {},
            "layers": {"not-a-layer": [["n", "dense", [6, 3], offset, count]]},
        },
    )
    store.close()

    rebuilt = _open(root)
    assert rebuilt.validation.status == "rebuilt"
    assert "not-a-layer" in rebuilt.validation.detail
    rebuilt.close()


def test_corruption_is_not_reported_as_a_probe_mismatch(tmp_path: Path) -> None:
    """`probes_failed` names probe ids only; plain corruption must not impersonate one."""
    root = tmp_path / "store"
    root.mkdir()
    (root / INDEX_NAME).write_text("{not json", encoding="utf-8")

    rebuilt = _open(root)
    assert rebuilt.validation.status == "rebuilt"
    assert rebuilt.validation.probes_failed == ()
    rebuilt.close()


def test_flush_retry_after_a_torn_append_cannot_shift_data(tmp_path: Path) -> None:
    """A tail written after open is discarded before the payload is re-appended."""
    root = tmp_path / "store"
    store = _open(root)
    first = default_svd(_matrix(seed=11))
    store.put_spectrum("first", first)
    store.flush()
    durable = (root / "spectra.f32").stat().st_size

    # Simulate a flush that died mid-append: bytes past the durable region, no index.
    with (root / "spectra.f32").open("ab") as handle:
        handle.write(b"\xde\xad\xbe\xef" * 2)
    assert (root / "spectra.f32").stat().st_size == durable + 8

    second = default_svd(_matrix(seed=12))
    key = SpectrumStore.spectrum_key(_matrix(seed=12))
    store.put_spectrum(key, second)
    store.flush()

    blob = root / "spectra.f32"
    assert blob.stat().st_size == durable + second.astype(np.float32).nbytes
    got = store.get_spectrum(key)
    assert got is not None
    assert got.tobytes() == second.astype(np.float32).astype(np.float64).tobytes()
    store.close()

    reopened = _open(root)
    replayed = reopened.get_spectrum(key)
    assert replayed is not None
    assert replayed.tobytes() == got.tobytes()
    reopened.close()


def test_a_stale_index_tmp_is_removed_by_the_next_writer(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    stale = root / f"{INDEX_NAME}.tmp"
    stale.write_text("half a json document", encoding="utf-8")

    store = _open(root)
    assert not stale.exists()
    assert store.validation.status == "new"
    store.close()


# ---------------------------------------------------------------------------
# Writer-integrity and optionality (found by the acceptance re-audit).
# ---------------------------------------------------------------------------


def test_a_deleted_lock_file_stops_writes(tmp_path: Path) -> None:
    """A lock file replaced under a live writer must stop it publishing, not race it."""
    root = tmp_path / "store"
    store = _open(root)
    store.put_spectrum("first", default_svd(_matrix()))
    store.flush()
    published = json.loads((root / INDEX_NAME).read_text(encoding="utf-8"))
    assert published["spectra"]

    (root / "store.lock").unlink()
    store.put_spectrum("second", default_svd(_matrix(seed=3)))
    store.flush()

    assert store.read_only is True
    assert "writer lock file" in store.validation.detail
    on_disk = json.loads((root / INDEX_NAME).read_text(encoding="utf-8"))
    assert set(on_disk["spectra"]) == set(published["spectra"])
    # The unpublished row must read as a miss, never as a hole in the blob.
    assert store.get_spectrum("second") is None
    assert store.get_spectrum("first") is not None
    store.close()


def test_a_shrinking_blob_abandons_writes_instead_of_zero_filling(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = _open(root)
    store.put_spectrum("first", default_svd(_matrix()))
    store.flush()
    os.truncate(root / "spectra.f32", 0)

    store.put_spectrum("second", default_svd(_matrix(seed=4)))
    store.flush()

    assert store.read_only is True
    assert "shrank" in store.validation.detail
    assert (root / "spectra.f32").stat().st_size == 0
    assert store.get_spectrum("second") is None
    store.close()


def test_a_truncated_probe_table_rebuilds(tmp_path: Path) -> None:
    """Keeping the profile string but dropping rows must not downgrade the stamp."""
    root = tmp_path / "store"
    SpectrumStore.open(root, probe_profile="fast").close()
    index_path = root / INDEX_NAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["probes"]["rows"] = index["probes"]["rows"][:2]
    index_path.write_text(json.dumps(index), encoding="utf-8")

    rebuilt = SpectrumStore.open(root, probe_profile="fast")
    assert rebuilt.validation.status == "rebuilt"
    assert "probe rows" in rebuilt.validation.detail
    rebuilt.close()


def test_a_lock_lost_during_creation_abandons_instead_of_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The very first flush can hit the lock check; open must disclose, not AttributeError."""
    monkeypatch.setattr(SpectrumStore, "_lock_is_intact", lambda self: False)

    store = SpectrumStore.open(tmp_path / "store", probe_profile="fast")

    assert store.read_only is True
    assert "writes abandoned" in store.validation.detail
    assert not (tmp_path / "store" / INDEX_NAME).exists()
    store.close()


def test_probe_profile_mismatch_rebuilds(tmp_path: Path) -> None:
    """A store recorded under another profile can never claim this profile's stamp."""
    root = tmp_path / "store"
    SpectrumStore.open(root, probe_profile="fast").close()

    rebuilt = SpectrumStore.open(root, probe_profile="full")
    assert rebuilt.validation.status == "rebuilt"
    assert "probe profile" in rebuilt.validation.detail
    rebuilt.close()


def test_install_returns_none_when_the_store_path_is_unusable(tmp_path: Path) -> None:
    detect = _synthetic_detector(tmp_path / "bench")
    before = detect._load_gate_tensors
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("in the way", encoding="utf-8")

    handle = fastpath.install(detect, method_sha256="a" * 64, store_root=blocker / "store")
    assert handle is None
    assert detect._load_gate_tensors is before


def test_a_store_write_error_abandons_caching_without_failing_the_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    detect = _synthetic_detector(tmp_path / "bench")
    store = _open(tmp_path / "store")
    handle = fastpath.install(detect, method_sha256="a" * 64, store=store)
    assert handle is not None

    def full_disk() -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(store, "flush", full_disk)
    extraction = detect._load_gate_tensors("a/b", "0" * 40)

    assert extraction.n_tensors > 0  # the extraction itself is untouched
    assert store.read_only is True
    assert "store write failed" in store.validation.detail
    assert handle.snapshot(tmp_path)["store"]["read_only"] is True
    handle.close()


# ---------------------------------------------------------------------------
# Synthetic detector: the seams under test, with no weights and no network.
# It mirrors the real detector's contract classes field for field, because the
# executor rebuilds ``Extraction``/``TensorSpectrum`` from the stored row.
# ---------------------------------------------------------------------------

SYNTHETIC_DETECTOR = '''\
"""Synthetic offline detector for fastpath seam tests: no weights, no network."""

from dataclasses import dataclass

import numpy as np

CALLS = {"load": 0, "svd": 0}


@dataclass(frozen=True)
class Result:
    verdict: str
    score: float | None
    diagnostics: dict
    note: str = ""
    abstain_reason: str | None = None


@dataclass(frozen=True)
class TensorSpectrum:
    name: str
    role: str
    sv: np.ndarray
    shape: tuple = ()

    @property
    def q(self):
        return int(self.sv.size)


class FormatUnreadable(Exception):
    """Structural abstain: the checkpoint cannot be read."""


@dataclass
class Extraction:
    layers: dict
    family: str
    format: str
    source: dict

    @property
    def n_tensors(self):
        return sum(len(tensors) for tensors in self.layers.values())


_GATE = np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0


def _svd_spectrum(matrix):
    CALLS["svd"] += 1
    values = np.linalg.svd(np.asarray(matrix, dtype=np.float32), compute_uv=False)
    return values.astype(np.float64)


def _load_gate_tensors(repo_id, revision=None):
    CALLS["load"] += 1
    layers = {
        0: [
            TensorSpectrum(
                "layers.0.mlp.gate_proj.weight", "dense_gate", _svd_spectrum(_GATE), _GATE.shape
            )
        ],
        1: [
            TensorSpectrum(
                "layers.1.mlp.gate_proj.weight",
                "dense_gate",
                _svd_spectrum(_GATE * 2.0),
                _GATE.shape,
            )
        ],
    }
    return Extraction(
        layers=layers,
        family="dense",
        format="safetensors",
        source={"repo_id": repo_id, "revision": revision, "fetch": "range"},
    )


def score(subject, reference, *, subject_rev=None, reference_rev=None, stack):
    subject_extraction = _load_gate_tensors(subject, subject_rev)
    reference_extraction = _load_gate_tensors(reference, reference_rev)
    value = float(np.sum(subject_extraction.layers[0][0].sv))
    other = reference_extraction.n_tensors + len(stack)
    return Result(
        verdict="related" if value >= 0.9 else "unrelated",
        score=value,
        diagnostics={"reference_tensors": other},
    )
'''


def _synthetic_detector(root: Path) -> ModuleType:
    root.mkdir(parents=True, exist_ok=True)
    (root / "detect.py").write_text(SYNTHETIC_DETECTOR, encoding="utf-8")
    return _load_detect(root)


def _copy_bench(tmp_path: Path) -> Path:
    copied = tmp_path / "bench"
    shutil.copytree(
        bench_root(),
        copied,
        ignore=shutil.ignore_patterns("runs", ".cache", "__pycache__", "*.pyc"),
    )
    return copied


def _extraction_view(extraction: Any) -> dict[str, Any]:
    """Every field of an ``Extraction`` in a comparable, array-free form."""
    layers = extraction.layers
    return {
        "family": extraction.family,
        "format": extraction.format,
        "source": dict(extraction.source),
        "n_tensors": extraction.n_tensors,
        "layers": {
            int(layer): [
                (
                    tensor.name,
                    tensor.role,
                    tuple(tensor.shape),
                    str(tensor.sv.dtype),
                    tensor.q,
                    tensor.sv.tobytes(),
                )
                for tensor in layers[layer]
            ]
            for layer in sorted(layers, key=int)
        },
    }


def _absolute_strings(body: Any) -> list[str]:
    """Every string in a JSON-shaped body that looks like an absolute path."""
    if isinstance(body, str):
        return [body] if body.startswith("/") else []
    if isinstance(body, dict):
        found: list[str] = []
        for key, value in body.items():
            found.extend(_absolute_strings(key))
            found.extend(_absolute_strings(value))
        return found
    if isinstance(body, (list, tuple)):
        found = []
        for item in body:
            found.extend(_absolute_strings(item))
        return found
    return []


def _case_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()[1:-1]


def test_disabled_honours_pb_no_fastpath(monkeypatch: pytest.MonkeyPatch) -> None:
    assert fastpath.ENV_DISABLE == "PB_NO_FASTPATH"
    for value in ("1", "true", " yes ", "ON"):
        monkeypatch.setenv(fastpath.ENV_DISABLE, value)
        assert fastpath.disabled() is True
    for value in ("0", ""):
        monkeypatch.setenv(fastpath.ENV_DISABLE, value)
        assert fastpath.disabled() is False
    monkeypatch.delenv(fastpath.ENV_DISABLE)
    assert fastpath.disabled() is False


def test_default_store_root_prefers_the_bench_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = tmp_path / "bench"
    monkeypatch.delenv(fastpath.ENV_STORE_ROOT, raising=False)
    assert fastpath.default_store_root(bench) == bench / ".cache" / "fastpath"

    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv(fastpath.ENV_STORE_ROOT, str(elsewhere))
    assert fastpath.default_store_root(bench) == elsewhere


def test_install_returns_none_for_a_detector_without_seams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(fastpath.ENV_DISABLE, raising=False)
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "detect.py").write_text(
        "def score(subject, reference, *, subject_rev=None, reference_rev=None, stack):\n"
        "    return None\n",
        encoding="utf-8",
    )
    detect = _load_detect(bare)

    assert fastpath.install(detect, method_sha256="0" * 64, store_root=tmp_path / "s") is None


def test_install_does_nothing_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(fastpath.ENV_DISABLE, "1")
    detect = _synthetic_detector(tmp_path / "bench")
    original_load = detect._load_gate_tensors
    original_svd = detect._svd_spectrum

    assert fastpath.install(detect, method_sha256="a" * 64, store_root=tmp_path / "s") is None
    assert detect._load_gate_tensors is original_load
    assert detect._svd_spectrum is original_svd


def test_install_wraps_only_the_two_seams(tmp_path: Path) -> None:
    detect = _synthetic_detector(tmp_path / "bench")
    before = {name: getattr(detect, name) for name in dir(detect)}
    store = _open(tmp_path / "store")

    handle = fastpath.install(detect, method_sha256="a" * 64, store=store)

    assert handle is not None
    assert handle.store is store
    assert tuple(handle.seams) == ("_load_gate_tensors", "_svd_spectrum")
    assert tuple(handle.seams) == tuple(fastpath.SEAMS)
    after = {name: getattr(detect, name) for name in dir(detect)}
    assert set(after) == set(before)
    replaced = {name for name in before if after[name] is not before[name]}
    assert replaced == set(fastpath.SEAMS)
    assert detect.score is before["score"]
    assert detect.Extraction is before["Extraction"]
    assert detect.TensorSpectrum is before["TensorSpectrum"]
    handle.close()


def test_tier_a_replays_one_extraction_per_pinned_revision(tmp_path: Path) -> None:
    detect = _synthetic_detector(tmp_path / "bench")
    store = _open(tmp_path / "store")
    handle = fastpath.install(detect, method_sha256="a" * 64, store=store)
    assert handle is not None

    first = detect._load_gate_tensors("a/b", "0" * 40)
    second = detect._load_gate_tensors("a/b", "0" * 40)

    assert detect.CALLS["load"] == 1
    assert store.counters.extraction_hits == 1
    assert store.counters.extraction_misses == 1
    assert _extraction_view(first) == _extraction_view(second)
    assert first.layers[0][0].sv is not second.layers[0][0].sv
    assert second.layers[0][0].sv.dtype == np.float64

    # A caller mutating its copy must not reach the store.
    second.layers[0][0].sv[0] = -123.0
    replayed = detect._load_gate_tensors("a/b", "0" * 40)
    assert replayed.layers[0][0].sv[0] != -123.0

    handle.close()
    store.close()
    reopened = _open(tmp_path / "store")
    row = reopened.get_extraction("a/b", "0" * 40, "a" * 64)
    assert row is not None
    assert row["family"] == "dense"
    assert sorted(row["layers"], key=int) == ["0", "1"]
    assert reopened.counters.extraction_hits == 1
    reopened.close()


def test_tier_a_is_bypassed_for_an_unpinned_revision(tmp_path: Path) -> None:
    detect = _synthetic_detector(tmp_path / "bench")
    store = _open(tmp_path / "store")
    handle = fastpath.install(detect, method_sha256="a" * 64, store=store)
    assert handle is not None

    detect._load_gate_tensors("a/b", None)
    detect._load_gate_tensors("a/b", None)

    assert detect.CALLS["load"] == 2
    assert store.counters.extraction_hits == 0
    assert store.counters.extraction_misses == 0
    handle.close()
    store.close()
    index = json.loads((tmp_path / "store" / INDEX_NAME).read_text(encoding="utf-8"))
    assert index["extractions"] == {}


def test_tier_a_row_is_bound_to_the_detector_sha(tmp_path: Path) -> None:
    first_detect = _synthetic_detector(tmp_path / "bench")
    store = _open(tmp_path / "store")
    handle = fastpath.install(first_detect, method_sha256="a" * 64, store=store)
    assert handle is not None
    first_detect._load_gate_tensors("a/b", "1" * 40)
    handle.close()
    store.close()

    second_detect = _synthetic_detector(tmp_path / "bench")
    reopened = _open(tmp_path / "store")
    handle = fastpath.install(second_detect, method_sha256="b" * 64, store=reopened)
    assert handle is not None
    extracted = second_detect._load_gate_tensors("a/b", "1" * 40)

    assert second_detect.CALLS["load"] == 1
    assert reopened.counters.extraction_rejected_detect_sha == 1
    assert reopened.counters.extraction_hits == 0
    assert extracted.source["revision"] == "1" * 40
    handle.close()
    reopened.close()


def test_tier_b_dedupes_layout_identical_matrices(tmp_path: Path) -> None:
    detect = _synthetic_detector(tmp_path / "bench")
    store = _open(tmp_path / "store")
    handle = fastpath.install(detect, method_sha256="a" * 64, store=store)
    assert handle is not None
    matrix = np.random.default_rng(21).standard_normal((7, 5), dtype=np.float32)
    fortran = np.asfortranarray(matrix)
    assert fortran.flags["F_CONTIGUOUS"]
    assert not fortran.flags["C_CONTIGUOUS"]
    expected = np.linalg.svd(matrix, compute_uv=False).astype(np.float64)

    first = detect._svd_spectrum(matrix)
    second = detect._svd_spectrum(fortran)

    assert detect.CALLS["svd"] == 1
    assert store.counters.spectrum_misses == 1
    assert store.counters.spectrum_hits == 1
    assert first.tobytes() == expected.tobytes()
    assert second.tobytes() == expected.tobytes()
    assert first is not second
    handle.close()


def test_tier_b_replay_is_bitwise_exact(tmp_path: Path) -> None:
    detect = _synthetic_detector(tmp_path / "bench")
    store = _open(tmp_path / "store")
    handle = fastpath.install(detect, method_sha256="a" * 64, store=store)
    assert handle is not None
    matrix = np.random.default_rng(5).standard_normal((9, 4), dtype=np.float32)
    expected = np.linalg.svd(np.asarray(matrix, dtype=np.float32), compute_uv=False).astype(
        np.float64
    )

    cold = detect._svd_spectrum(matrix)
    replay = detect._svd_spectrum(matrix)

    assert cold.tobytes() == expected.tobytes()
    assert replay.tobytes() == expected.tobytes()
    assert replay.dtype == np.float64
    assert replay is not cold
    assert detect.CALLS["svd"] == 1
    assert store.counters.spectrum_hits == 1

    # The stored bits survive a reopen with the same widening.
    handle.close()
    store.close()
    reopened = _open(tmp_path / "store")
    stored = reopened.get_spectrum(SpectrumStore.spectrum_key(matrix))
    assert stored is not None
    assert stored.tobytes() == expected.tobytes()
    reopened.close()


def test_a_raising_extraction_is_never_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    detect = _synthetic_detector(tmp_path / "bench")

    def boom(repo_id: str, revision: str | None) -> None:
        raise detect.FormatUnreadable(f"{repo_id}@{revision} unreadable")

    monkeypatch.setattr(detect, "_load_gate_tensors", boom)
    store = _open(tmp_path / "store")
    handle = fastpath.install(detect, method_sha256="a" * 64, store=store)
    assert handle is not None

    with pytest.raises(detect.FormatUnreadable):
        detect._load_gate_tensors("a/b", "0" * 40)

    assert store.get_extraction("a/b", "0" * 40, "a" * 64) is None
    handle.close()
    store.close()
    index = json.loads((tmp_path / "store" / INDEX_NAME).read_text(encoding="utf-8"))
    assert index["extractions"] == {}


def test_a_raising_svd_is_never_persisted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    detect = _synthetic_detector(tmp_path / "bench")

    def boom(matrix: np.ndarray) -> np.ndarray:
        raise RuntimeError(f"svd failed on {np.asarray(matrix).shape}")

    monkeypatch.setattr(detect, "_svd_spectrum", boom)
    store = _open(tmp_path / "store")
    handle = fastpath.install(detect, method_sha256="a" * 64, store=store)
    assert handle is not None
    matrix = _matrix()

    with pytest.raises(RuntimeError, match="svd failed"):
        detect._svd_spectrum(matrix)

    assert store.get_spectrum(SpectrumStore.spectrum_key(matrix)) is None
    handle.close()
    store.close()
    index = json.loads((tmp_path / "store" / INDEX_NAME).read_text(encoding="utf-8"))
    assert index["spectra"] == {}
    assert index["extractions"] == {}


def test_snapshot_is_relative_and_discloses_the_contract(tmp_path: Path) -> None:
    bench = tmp_path / "bench"
    detect = _synthetic_detector(bench)
    store = _open(fastpath.default_store_root(bench))
    handle = fastpath.install(detect, method_sha256="c" * 64, store=store)
    assert handle is not None
    detect._load_gate_tensors("a/b", "0" * 40)
    detect._load_gate_tensors("a/b", "0" * 40)
    detect._svd_spectrum(_matrix())

    body = handle.snapshot(bench)

    assert body["spec"] == fastpath.FASTPATH_SPEC
    assert body["method_sha256"] == "c" * 64
    assert tuple(body["seams"]) == tuple(fastpath.SEAMS)
    assert re.fullmatch(r"[0-9a-f]{64}", body["fastpath_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", body["store_sha256"])
    assert not os.path.isabs(body["store"]["root"])
    assert _absolute_strings(body) == []
    assert body["store"]["probe_profile"] == "fast"
    assert body["store"]["read_only"] is False
    assert isinstance(body["store"]["validation"], dict)
    assert body["store"]["validation"]["status"] in {"new", "ok", "rebuilt", "empty"}
    counters = body["counters"]
    assert counters["extraction_hits"] == 1
    assert counters["spectrum_hits"] == 0
    assert isinstance(body["store"]["stack"], dict)
    assert body["store"]["stack"]["numpy"] == np.__version__

    parsed = json.loads(json.dumps(body))
    assert parsed["counters"]["extraction_misses"] == 1
    handle.close()


def test_sidecar_path_and_atomic_write(tmp_path: Path) -> None:
    assert fastpath.sidecar_path(Path("runs/x.method.jsonl")) == Path("runs/x.fastpath.json")
    runs = tmp_path / "runs"
    runs.mkdir()
    path = fastpath.sidecar_path(runs / "x.method.jsonl")

    fastpath.write_sidecar(path, {"zeta": 1, "alpha": {"y": 2, "b": 3}})
    text = path.read_text(encoding="utf-8")

    assert json.loads(text) == {"zeta": 1, "alpha": {"y": 2, "b": 3}}
    assert text.index('"alpha"') < text.index('"zeta"')
    assert sorted(entry.name for entry in runs.iterdir()) == ["x.fastpath.json"]

    fastpath.write_sidecar(path, {"zeta": 4})
    assert json.loads(path.read_text(encoding="utf-8")) == {"zeta": 4}
    assert sorted(entry.name for entry in runs.iterdir()) == ["x.fastpath.json"]


@dataclass(frozen=True)
class _BenchRuns:
    root: Path
    store: Path
    case_id: str
    cold: Path
    warm: Path
    plain: Path


@pytest.fixture(scope="module")
def bench_runs(tmp_path_factory: pytest.TempPathFactory) -> _BenchRuns:
    """Cold run, warm re-run, and a ``PB_NO_FASTPATH`` run over one synthetic bench."""
    base = tmp_path_factory.mktemp("fastpath-eval")
    root = _copy_bench(base)
    (root / "detect.py").write_text(SYNTHETIC_DETECTOR, encoding="utf-8")
    store = base / "store"
    case_id = load_cohort("smoke", root)["ids"][0]
    cold = base / "runs" / "cold.method.jsonl"
    warm = base / "runs" / "warm.method.jsonl"
    plain = base / "runs" / "plain.method.jsonl"

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(fastpath.ENV_STORE_ROOT, str(store))
        patch.delenv(fastpath.ENV_DISABLE, raising=False)
        assert run_eval(root, ids=[case_id], out_path=cold) == 0
        assert run_eval(root, ids=[case_id], out_path=warm) == 0
        with pytest.MonkeyPatch.context() as disabled:
            disabled.setenv(fastpath.ENV_DISABLE, "1")
            assert run_eval(root, ids=[case_id], out_path=plain) == 0

    return _BenchRuns(root=root, store=store, case_id=case_id, cold=cold, warm=warm, plain=plain)


def test_run_eval_replays_the_same_cases_from_a_warm_store(bench_runs: _BenchRuns) -> None:
    cold_sidecar = fastpath.sidecar_path(bench_runs.cold)
    warm_sidecar = fastpath.sidecar_path(bench_runs.warm)

    assert _case_lines(bench_runs.cold) == _case_lines(bench_runs.warm)
    assert cold_sidecar.exists()
    assert warm_sidecar.exists()
    cold_body = json.loads(cold_sidecar.read_text(encoding="utf-8"))
    warm_body = json.loads(warm_sidecar.read_text(encoding="utf-8"))
    assert cold_body["spec"] == fastpath.FASTPATH_SPEC
    assert cold_body["enabled"] is True
    assert cold_body["store"]["validation"]["status"] == "new"
    assert cold_body["counters"]["extraction_misses"] > 0
    assert cold_body["counters"]["spectrum_misses"] > 0
    assert cold_body["counters"]["spectrum_hits"] > 0
    assert warm_body["store"]["validation"]["status"] in {"ok", "new"}
    assert warm_body["counters"]["extraction_hits"] > 0
    assert warm_body["counters"]["spectrum_misses"] == 0
    assert warm_body["method_sha256"] == cold_body["method_sha256"]
    assert cold_body["store"]["root"] == os.path.relpath(bench_runs.store, bench_runs.root)
    assert (bench_runs.store / INDEX_NAME).exists()
    assert _absolute_strings(cold_body) == []
    assert _absolute_strings(warm_body) == []


def test_pb_no_fastpath_writes_no_sidecar(bench_runs: _BenchRuns) -> None:
    assert not fastpath.sidecar_path(bench_runs.plain).exists()
    assert _case_lines(bench_runs.plain) == _case_lines(bench_runs.cold)
    # The disabled run still writes the immutable artifact pair, just no sidecar.
    assert bench_runs.plain.exists()


def test_fastpath_state_never_leaks_into_artifact_meta(bench_runs: _BenchRuns) -> None:
    lines = bench_runs.cold.read_text(encoding="utf-8").splitlines()
    meta = json.loads(lines[0])

    assert meta["artifact"] == "method"
    assert set(meta) == {
        "type",
        "artifact",
        "eval_version",
        "timestamp_utc",
        "selection",
        "hashes",
        "n_cases",
    }
    assert meta["n_cases"] == 1
    assert [json.loads(line)["id"] for line in lines[1:-1]] == [bench_runs.case_id]
    assert not _absolute_strings(meta)
