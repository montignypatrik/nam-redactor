"""Unit tests for job state and untrusted box handling."""
from __future__ import annotations

from pathlib import Path

from nam_redactor.redactor import PageDetection
from nam_redactor.web.jobs import (
    JOB_DIR_PREFIX,
    Job,
    JobStore,
    assign_patient_id,
    coerce_boxes,
    resolve_box_ids,
)


class FakeEngine:
    def process_image(self, image_path: str, page_num: int = 1):
        return []


def make_store(tmp_path: Path, **kwargs) -> JobStore:
    return JobStore(engine_factory=FakeEngine, dir_root=tmp_path, name_model_factory=lambda: None, **kwargs)


# -- coerce_boxes: the browser is untrusted input -------------------------


def test_coerce_boxes_accepts_well_formed_boxes():
    boxes = coerce_boxes([{"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4, "confidence": 0.8}])
    assert len(boxes) == 1
    assert boxes[0] == {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4, "confidence": 0.8}


def test_coerce_boxes_clamps_out_of_range_coordinates():
    boxes = coerce_boxes([{"x1": -5, "y1": -1, "x2": 9, "y2": 42}])
    assert boxes[0]["x1"] == 0.0
    assert boxes[0]["y1"] == 0.0
    assert boxes[0]["x2"] == 1.0
    assert boxes[0]["y2"] == 1.0


def test_coerce_boxes_normalizes_reversed_corners():
    boxes = coerce_boxes([{"x1": 0.8, "y1": 0.9, "x2": 0.2, "y2": 0.1}])
    assert (boxes[0]["x1"], boxes[0]["y1"]) == (0.2, 0.1)
    assert (boxes[0]["x2"], boxes[0]["y2"]) == (0.8, 0.9)


def test_coerce_boxes_defaults_confidence_to_certain():
    boxes = coerce_boxes([{"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}])
    assert boxes[0]["confidence"] == 1.0


def test_coerce_boxes_coerces_numeric_strings():
    boxes = coerce_boxes([{"x1": "0.25", "y1": "0.5", "x2": "0.75", "y2": "1"}])
    assert boxes[0]["x1"] == 0.25
    assert boxes[0]["y2"] == 1.0


def test_coerce_boxes_drops_malformed_entries_individually():
    """One bad rectangle must not discard the user's other edits."""
    boxes = coerce_boxes([
        {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
        {"x1": 0.3},                              # missing keys
        "not a dict",
        {"x1": 0.5, "y1": 0.5, "x2": 0.5, "y2": 0.5},   # degenerate
        {"x1": "abc", "y1": 0.1, "x2": 0.2, "y2": 0.2}, # unparseable
        {"x1": 0.6, "y1": 0.6, "x2": 0.7, "y2": 0.7},
    ])
    assert len(boxes) == 2
    assert boxes[0]["x1"] == 0.1
    assert boxes[1]["x1"] == 0.6


def test_coerce_boxes_keeps_the_name_kind_and_never_an_id_on_it():
    boxes = coerce_boxes([
        {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2, "kind": "name", "patient_id": "Patient_0001"},
        {"x1": 0.3, "y1": 0.3, "x2": 0.4, "y2": 0.4, "kind": "<script>"},
    ])
    assert boxes[0]["kind"] == "name"
    assert "patient_id" not in boxes[0]
    assert "kind" not in boxes[1]


def test_coerce_boxes_rejects_non_list():
    assert coerce_boxes(None) == ()
    assert coerce_boxes({"x1": 0}) == ()
    assert coerce_boxes("[]") == ()


# -- JobStore lifecycle ---------------------------------------------------


def test_create_get_delete_roundtrip(tmp_path: Path):
    store = make_store(tmp_path)
    job = store.create(filename="record.pdf", dpi=200)

    assert store.get(job.id) is job
    assert job.dir.is_dir()
    assert job.dir.parent == tmp_path / "documents"
    # The user's filename is display-only and must not reach the filesystem.
    assert "record.pdf" not in str(job.input_path)

    assert store.delete(job.id) is True
    assert store.get(job.id) is None
    assert not job.dir.exists()
    assert store.delete(job.id) is False


def test_sweep_clears_crash_leftovers_but_never_saved_documents(tmp_path: Path):
    legacy = tmp_path / f"{JOB_DIR_PREFIX}deadbeef"  # session-only era temp dir
    legacy.mkdir()
    (legacy / "input.pdf").write_bytes(b"%PDF-1.4 leftover")

    store = make_store(tmp_path)
    unfinished = store.create(filename="half.pdf", dpi=200)  # upload never completed
    saved = store.create(filename="kept.pdf", dpi=200)
    store.save(saved)

    keep = tmp_path / "unrelated"
    keep.mkdir()

    store.sweep_stale_dirs()

    assert not legacy.exists()
    assert not unfinished.dir.exists()
    assert saved.dir.exists()
    assert keep.exists()


def test_engine_is_created_once_and_reused(tmp_path: Path):
    created: list[FakeEngine] = []

    def factory() -> FakeEngine:
        engine = FakeEngine()
        created.append(engine)
        return engine

    store = JobStore(engine_factory=factory, dir_root=tmp_path, name_model_factory=lambda: None)
    first = store._get_engine()
    second = store._get_engine()

    assert first is second
    assert len(created) == 1


def test_shutdown_keeps_saved_documents_for_the_next_start(tmp_path: Path):
    store = make_store(tmp_path)
    first = store.create(filename="a.pdf", dpi=200)
    first.patient_ids = {"HOGP33610615": "Patient_0001"}
    store.save(first)

    store.shutdown()
    assert first.dir.exists()

    reopened = make_store(tmp_path)
    assert reopened.load_saved() == 1
    job = reopened.get(first.id)
    assert job is not None and job.filename == "a.pdf"
    assert job.patient_ids == {"HOGP33610615": "Patient_0001"}


def test_identity_map_is_saved_apart_from_the_record(tmp_path: Path):
    store = make_store(tmp_path)
    job = store.create(filename="a.pdf", dpi=200)
    job.patient_ids = {"HOGP33610615": "Patient_0001"}
    store.save(job)

    record = (job.dir / "document.json").read_text(encoding="utf-8")
    private = (job.dir / "private" / "identities.json").read_text(encoding="utf-8")
    assert "HOGP33610615" not in record
    assert "HOGP33610615" in private


# -- patient IDs ----------------------------------------------------------

STORED = {
    "x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.25,
    "confidence": 0.99, "patient_id": "Patient_0001",
}


def _job_with_box(store: JobStore, box: dict) -> Job:
    job = store.create(filename="record.pdf", dpi=200)
    job.pages = [PageDetection(page_num=1, width=100, height=100, boxes=(box,))]
    return job


def test_coerce_boxes_attaches_a_patient_id_only_when_it_is_valid():
    base = {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2}

    # Absent stays absent -- never present as None, which would break callers
    # comparing box dicts against the geometry-only contract.
    assert "patient_id" not in coerce_boxes([base])[0]
    assert (
        coerce_boxes([{**base, "patient_id": "Patient_0007"}])[0]["patient_id"]
        == "Patient_0007"
    )

    for bad in ["", "patient_0001", "Patient_1", "Patient_0001; rm -rf /", None, 7, "x" * 500]:
        assert "patient_id" not in coerce_boxes([{**base, "patient_id": bad}])[0], bad


def test_resolve_box_ids_restores_an_id_the_client_dropped(tmp_path: Path):
    """Otherwise the PDF would be unlabelled right after review showed IDs."""
    job = _job_with_box(make_store(tmp_path), STORED)
    stripped = {k: v for k, v in STORED.items() if k != "patient_id"}

    resolved = resolve_box_ids(job, 1, [stripped])

    assert resolved[0]["patient_id"] == "Patient_0001"


def test_resolve_box_ids_overrides_a_forged_id(tmp_path: Path):
    """The server's own record wins, so a patient cannot be mislabelled."""
    job = _job_with_box(make_store(tmp_path), STORED)

    resolved = resolve_box_ids(job, 1, [{**STORED, "patient_id": "Patient_9999"}])

    assert resolved[0]["patient_id"] == "Patient_0001"


def test_resolve_box_ids_keeps_the_client_id_for_a_hand_drawn_box(tmp_path: Path):
    """A typed NAM's pseudonym must survive from review through to the PDF."""
    job = _job_with_box(make_store(tmp_path), STORED)
    drawn = {
        "x1": 0.5, "y1": 0.5, "x2": 0.7, "y2": 0.6,
        "confidence": 1.0, "patient_id": "Patient_0002",
    }

    resolved = resolve_box_ids(job, 1, [drawn])

    assert resolved[0]["patient_id"] == "Patient_0002"


def test_assign_patient_id_reuses_an_identity_and_extends_the_series(tmp_path: Path):
    store = make_store(tmp_path)
    job = store.create(filename="record.pdf", dpi=200)
    job.patient_ids = {"AAAA00000000": "Patient_0001"}

    assert assign_patient_id(job, "AAAA00000000") == "Patient_0001"
    assert assign_patient_id(job, "BBBB00000000") == "Patient_0002"
    assert assign_patient_id(job, "BBBB00000000") == "Patient_0002"
