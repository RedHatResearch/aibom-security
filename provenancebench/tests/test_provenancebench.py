"""Offline tests for the ProvenanceBench manifest and evaluator."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from pbench import load
from pbench.eval import _load_detect, _resolve_ids, run_eval
from pbench.load import bench_root, case_by_id, load_cases, load_cohort, load_cohorts, load_kinds


def test_manifest_and_taxonomy_load() -> None:
    cases = load_cases()
    kinds = load_kinds()
    cohorts = load_cohorts()

    assert cases
    assert kinds == list(dict.fromkeys(kinds))
    assert all(isinstance(kind, str) and kind for kind in kinds)
    assert all(case["kind"] in kinds for case in cases)
    assert set(cohorts) == {"smoke", "core", "all"}


def test_pair_rows_have_evidence_and_immutable_pins() -> None:
    cases = load_cases()
    case_ids = [case["id"] for case in cases]

    assert len(case_ids) == len(set(case_ids))
    for case in cases:
        assert not {"ref", "sus", "y", "cite", "label_confidence"} & case.keys()
        assert case["expected"] in load.VALID_EXPECTED
        assert case["stack"] in load.VALID_STACK
        assert set(case["evidence"]) == {"confidence", "sources", "note"}
        assert case["evidence"]["confidence"] in load.VALID_CONFIDENCE
        assert case["evidence"]["sources"]
        assert case["evidence"]["note"]
        for side in ("subject", "reference"):
            revision = case[f"{side}_rev"]
            assert len(revision) == 40
            assert set(revision) <= set("0123456789abcdef")
        if case["expected"] == "related":
            assert case["lineage_scope"] in load.VALID_LINEAGE_SCOPE
            assert "challenge" not in case
        else:
            assert "lineage_scope" not in case
        if "challenge" in case:
            assert case["challenge"]["type"] in load.VALID_CHALLENGE_TYPES


def test_cohorts_partition_manifest_without_roster_constants() -> None:
    cases = case_by_id(load_cases())
    cohorts = load_cohorts()
    for name in cohorts:
        load_cohort(name)

    smoke_ids = cohorts["smoke"]["ids"]
    core_ids = cohorts["core"]["ids"]
    all_ids = cohorts["all"]["ids"]

    assert len(all_ids) == len(set(all_ids))
    assert set(all_ids) == set(cases)
    assert set(smoke_ids).isdisjoint(core_ids)
    assert all_ids == smoke_ids + core_ids


def test_model_pin_passes_through_manifest_values() -> None:
    for case in load_cases():
        for side in ("subject", "reference"):
            assert load.model_pin(case, side) == (case[side], case[f"{side}_rev"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_field", "missing required field subject"),
        ("unknown_kind", "unknown kind missing_kind"),
        ("bad_challenge", "challenge.difficulty must be a non-empty string"),
        ("empty_sources", "evidence.sources must be a non-empty list of strings"),
        ("missing_revision", "missing required field subject_rev"),
    ],
)
def test_load_cases_rejects_invalid_metadata(tmp_path: Path, mutation: str, message: str) -> None:
    root = _copy_bench(tmp_path)
    path = root / "pairs.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    case = data["pairs"][0]
    if mutation == "missing_field":
        del case["subject"]
    elif mutation == "unknown_kind":
        case["kind"] = "missing_kind"
    elif mutation == "bad_challenge":
        case = next(row for row in data["pairs"] if "challenge" in row)
        case["challenge"]["difficulty"] = 2
    elif mutation == "empty_sources":
        case["evidence"]["sources"] = []
    elif mutation == "missing_revision":
        del case["subject_rev"]
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_cases(root)


def test_load_kinds_rejects_non_string_kind(tmp_path: Path) -> None:
    root = _copy_bench(tmp_path)
    path = root / "kinds.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["kinds"][0] = {"kind": "invalid"}
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="kinds.yaml: row 1 must be a non-empty string"):
        load_kinds(root)


def test_load_cohort_rejects_unknown_case(tmp_path: Path) -> None:
    root = _copy_bench(tmp_path)
    (root / "cohorts.yaml").write_text(
        "schema_version: 1\ncohorts:\n  test:\n    ids: [pb-missing]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cohort test: unknown case id pb-missing"):
        load_cohort("test", root)


def test_detector_cannot_read_ground_truth() -> None:
    detect_src = (bench_root() / "detect.py").read_text(encoding="utf-8")
    for term in ("load_cases", "pairs.yaml", "case_by_id", "pbench.load", "yaml.safe_load"):
        assert term not in detect_src


def test_detect_supports_new_families() -> None:
    """Core-cohort patterns classify offline (no weights, no network)."""
    detect = _load_detect(bench_root())

    assert detect._classify("gpt_neox.layers.3.mlp.dense_4h_to_h.weight") == (
        "gpt_neox_fallback",
        3,
        "dense",
    )
    # dense_h_to_4h is the UP branch of the gated FFN, not the gate.
    assert detect._classify("gpt_neox.layers.3.mlp.dense_h_to_4h.weight") is None
    assert detect._classify("model.language_model.layers.5.mlp.gate_proj.weight") == (
        "dense_gate",
        5,
        "dense",
    )
    assert detect._classify("backbone.layers.7.mixer.down_proj.weight") == (
        "nemotron_fallback",
        7,
        "dense",
    )
    # Stacked experts: axis 0 is the expert axis and the pass-through branch
    # keeps each expert matrix intact (no row split, no de-interleave).
    assert detect._classify("layers.4.moe.gate_proj") == ("tinymixtral_stacked", 4, "routed")
    assert "tinymixtral_stacked" in detect._STACKED_KINDS
    # Speculative-decoding head nested like a decoder layer: anchored regex.
    assert detect._classify("mtp.layers.0.mlp.gate_proj.weight") is None


def test_always_related_is_caught_by_class_behavior(tmp_path: Path) -> None:
    root = _copy_bench(tmp_path)
    _write_detector(root, "related", score="1.0")
    run_path = root / "runs" / "always-related.method.jsonl"

    assert run_eval(root, cohort="all", out_path=run_path) == 0
    rows = [
        json.loads(line)
        for line in run_path.with_name("always-related.review.jsonl").read_text().splitlines()[1:-1]
    ]
    assert rows
    assert all(row["correct"] for row in rows if row["expected"] == "related")
    assert all(
        row["error_class"] == "false_confirm" for row in rows if row["expected"] == "unrelated"
    )
    assert all(
        row["error_class"] == "unsupported_false_confirm"
        for row in rows
        if row["expected"] == "unsupported"
    )


def test_unsupported_maps_abstain_to_correct(tmp_path: Path) -> None:
    root = _copy_bench(tmp_path)
    case = next(row for row in load_cases(root) if row["expected"] == "unsupported")
    _write_detector(root, "abstain", score="None")
    run_path = root / "runs" / "unsupported.method.jsonl"

    assert run_eval(root, ids=[case["id"]], out_path=run_path) == 0
    review_row = json.loads(
        run_path.with_name("unsupported.review.jsonl").read_text().splitlines()[1]
    )
    assert review_row["expected"] == "unsupported"
    assert review_row["verdict"] == "abstain"
    assert review_row["correct"] is True
    assert review_row["error_class"] is None
    assert review_row["score"] is None


def test_eval_writes_pairwise_method_and_review_artifacts(tmp_path: Path) -> None:
    root = _copy_bench(tmp_path)
    # The real detector loads weights (network); this test exercises harness
    # artifact shape, so swap in the same offline detector the other tests use.
    _write_detector(root, "abstain", score="None")
    smoke_ids = load_cohort("smoke", root)["ids"]
    run_path = root / "runs" / "smoke.method.jsonl"

    assert run_eval(root, cohort="smoke", out_path=run_path) == 0
    method_lines = run_path.read_text(encoding="utf-8").splitlines()
    review_lines = run_path.with_name("smoke.review.jsonl").read_text().splitlines()
    method_cases = [json.loads(line) for line in method_lines[1:-1]]
    review_cases = [json.loads(line) for line in review_lines[1:-1]]
    method_meta = json.loads(method_lines[0])
    method_trailer = json.loads(method_lines[-1])
    review_trailer = json.loads(review_lines[-1])

    assert method_meta["selection"]["resolved_ids"] == smoke_ids
    assert len(method_cases) == len(smoke_ids)
    assert len(review_cases) == len(smoke_ids)
    assert method_trailer["n_cases"] == len(smoke_ids)
    assert {"manifest_sha256", "selection_sha256", "method_sha256"} <= method_meta["hashes"].keys()
    assert (
        not {
            "expected",
            "evidence",
            "kind",
            "lineage_scope",
            "challenge",
            "correct",
            "mismatch",
            "error_class",
        }
        & method_cases[0].keys()
    )
    assert {"expected", "evidence", "correct", "mismatch", "error_class"} <= review_cases[0].keys()
    assert sum(row["n"] for row in review_trailer["by_kind"].values()) == len(smoke_ids)
    assert sum(row["n"] for row in review_trailer["by_stack"].values()) == len(smoke_ids)


def test_selection_modes_and_filters() -> None:
    cases = load_cases()
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_eval(cohort="smoke", ids=[load_cohorts()["smoke"]["ids"][0]])
    with pytest.raises(ValueError, match="pass --cohort, --ids, or a --stack/--kind filter"):
        run_eval()

    selected_case = cases[0]
    selected, selection = _resolve_ids(
        cases,
        cohort=None,
        ids=None,
        root=bench_root(),
        stack=selected_case["stack"],
        kinds=[selected_case["kind"]],
    )
    expected_ids = [
        case["id"]
        for case in cases
        if case["stack"] == selected_case["stack"] and case["kind"] == selected_case["kind"]
    ]
    assert [case["id"] for case in selected] == expected_ids
    assert selection["resolved_ids"] == expected_ids
    assert selection["filters"] == {
        "stack": selected_case["stack"],
        "kinds": [selected_case["kind"]],
    }

    with pytest.raises(ValueError, match="selection matched no cases after filters"):
        _resolve_ids(
            cases,
            cohort=None,
            ids=None,
            root=bench_root(),
            stack="dense",
            kinds=["expert_drop"],
        )


def test_artifact_pair_is_preflight_immutable(tmp_path: Path) -> None:
    root = _copy_bench(tmp_path)
    method_path = root / "runs" / "collision.method.jsonl"
    review_path = root / "runs" / "collision.review.jsonl"
    review_path.parent.mkdir()
    review_path.write_text("existing review artifact\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite immutable artifact"):
        run_eval(root, ids=[load_cohort("smoke", root)["ids"][0]], out_path=method_path)

    assert not method_path.exists()
    assert review_path.read_text(encoding="utf-8") == "existing review artifact\n"


def _write_shard(
    directory: Path, header: object, data: bytes = b"", header_len: int | None = None
) -> Path:
    """Write one synthetic safetensors shard with an arbitrary length prefix."""
    raw = bytes(header) if isinstance(header, (bytes, bytearray)) else json.dumps(header).encode()
    length = len(raw) if header_len is None else header_len
    path = directory / "model.safetensors"
    path.write_bytes(int(length).to_bytes(8, "little") + raw + data)
    return path


def test_snapshot_reader_rejects_malicious_shards(tmp_path: Path) -> None:
    """Hostile safetensors headers abstain as FormatUnreadable, never allocate."""
    detect = _load_detect(bench_root())
    probes = [
        (
            "negative_shape",
            {"x": {"dtype": "F32", "shape": [-1, 4], "data_offsets": [0, -4]}},
            b"\x00" * 16,
            None,
            "x",
        ),
        ("header_overflow", {}, b"", 2**62, None),
        ("truncated_header", b'{"x": {', b"", 1024, None),
        ("list_header", [1, 2, 3], b"", None, None),
        (
            "end_beyond_file",
            {"x": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}},
            b"\x00" * 4,
            None,
            "x",
        ),
    ]
    for name, header, data, header_len, tensor in probes:
        directory = tmp_path / name
        directory.mkdir()
        _write_shard(directory, header, data, header_len)
        with pytest.raises(detect.FormatUnreadable):
            reader = detect._SnapshotReader(directory, None)
            if tensor is not None:
                reader.read(tensor)


class _FakeStreamResponse:
    """Minimal httpx stream response that counts body reads."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.body_reads = 0

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self):
        self.body_reads += 1
        yield b"\x00" * 4096


class _FakeStreamClient:
    """get_session() stand-in exposing the httpx ``stream`` context manager."""

    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    def stream(self, method, url, *, headers=None, timeout=None):
        response = self._response

        class _Context:
            def __enter__(self):
                return response

            def __exit__(self, *exc_info):
                return False

        return _Context()


def test_range_get_rejects_non_206_without_reading_body(monkeypatch) -> None:
    """A server that ignores Range never gets its body buffered."""
    detect = _load_detect(bench_root())
    response = _FakeStreamResponse(200)
    monkeypatch.setattr(detect, "get_session", lambda: _FakeStreamClient(response))
    reader = object.__new__(detect._RangeReader)
    reader._repo_id = "probe/repo"
    reader._revision = "0" * 40
    reader._token = None
    reader._urls = {}

    with pytest.raises(RuntimeError, match="ignored the range request"):
        reader._get("model.safetensors", 0, 7)
    assert response.body_reads == 0


def test_hub_revision_404_is_an_abstain(monkeypatch) -> None:
    """A by-SHA tree 404 abstains at both the loader and the score boundary."""
    import httpx
    from huggingface_hub.errors import RevisionNotFoundError

    detect = _load_detect(bench_root())
    response = httpx.Response(
        404,
        request=httpx.Request("GET", "https://huggingface.co/api/models/probe/revision/0"),
    )

    def raise_404(repo_id, revision=None):
        raise RevisionNotFoundError("revision not found", response=response)

    monkeypatch.setattr(detect, "_load_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(detect, "_repo_files", raise_404)
    result = detect.score("probe/subject", "probe/reference", stack="dense")
    assert result.verdict == "abstain"
    assert result.abstain_reason == "format_unreadable"

    # The snapshot fallback can also 404 after the listing succeeds; the
    # boundary clause in score() must convert that too.
    monkeypatch.setattr(detect, "_load_gate_tensors", raise_404)
    result = detect.score("probe/subject", "probe/reference", stack="dense")
    assert result.verdict == "abstain"
    assert result.abstain_reason == "format_unreadable"


def _copy_bench(tmp_path: Path) -> Path:
    copied = tmp_path / "bench"
    shutil.copytree(
        bench_root(),
        copied,
        ignore=shutil.ignore_patterns("runs", ".cache", "__pycache__", "*.pyc"),
    )
    return copied


def _write_detector(root: Path, verdict: str, *, score: str) -> None:
    (root / "detect.py").write_text(
        f'''from dataclasses import dataclass
from typing import Literal

Verdict = Literal["related", "unrelated", "abstain"]

@dataclass(frozen=True)
class Result:
    verdict: Verdict
    score: float | None
    diagnostics: dict[str, object]
    note: str = ""
    abstain_reason: str | None = None

def score(subject, reference, *, subject_rev=None, reference_rev=None, stack):
    _ = (subject, reference, subject_rev, reference_rev, stack)
    return Result(verdict="{verdict}", score={score}, diagnostics={{}})
''',
        encoding="utf-8",
    )
