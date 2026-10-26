"""Unit tests: coordinate provenance and the appointment-row model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf
import pytest

from nam_redactor.provenance import (
    PageProvenance,
    points_to_unrotated,
    trace,
    working_to_source,
)
from nam_redactor.redactor import rasterize_page
from nam_redactor.web import rows as rowlib
from nam_redactor.zones import crop_to_zones, whole_page_provenance

DPI = 150


def _dark_bbox_norm(page: pymupdf.Page) -> dict[str, float]:
    """Normalized bbox of the dark pixels on a rendered page."""
    image = rasterize_page(page, dpi=DPI)
    pixels = np.asarray(image.convert("L")) < 128
    ys, xs = np.nonzero(pixels)
    h, w = pixels.shape
    return {"x1": xs.min() / w, "y1": ys.min() / h, "x2": (xs.max() + 1) / w, "y2": (ys.max() + 1) / h}


# -- provenance ------------------------------------------------------------


@pytest.mark.parametrize("rotation", [0, 90])
def test_a_mark_on_a_zone_maps_back_to_its_place_on_the_original(tmp_path: Path, rotation: int):
    """The core provenance promise, including a rotated source page: find a
    mark on the working page, map it back, and land on the original mark."""
    source_rect = pymupdf.Rect(200, 300, 260, 330)  # unrotated user space
    doc = pymupdf.open()
    doc.new_page(width=300, height=200)  # page 1: filler, so numbering matters
    page = doc.new_page(width=400, height=600)
    page.draw_rect(source_rect, color=(0, 0, 0), fill=(0, 0, 0))
    page.set_rotation(rotation)
    src = tmp_path / "src.pdf"
    doc.save(src)

    with pymupdf.open(src) as opened:
        shown = opened[1]
        expected = _dark_bbox_norm(shown)  # where the mark is, as displayed

    # A zone around the mark (displayed coordinates), plus a decoy zone
    # earlier in reading order, so the mark's working page is not page 1.
    zones = {2: [
        {"x1": 0.0, "y1": 0.0, "x2": 0.3, "y2": 0.1},
        {"x1": expected["x1"] - 0.1, "y1": expected["y1"] - 0.1,
         "x2": min(1.0, expected["x2"] + 0.1), "y2": min(1.0, expected["y2"] + 0.1)},
    ]}
    provs = crop_to_zones(src, tmp_path / "zones.pdf", zones, dpi=DPI)
    assert [p.source_page for p in provs] == [2, 2]
    prov = provs[1]
    assert prov.working_page == 2 and prov.source_rotation == rotation

    with pymupdf.open(tmp_path / "zones.pdf") as zoned:
        on_working = _dark_bbox_norm(zoned[prov.working_page - 1])

    back = working_to_source(on_working, prov)
    px = 1.5 / min(prov.source_raster_px)  # within about a pixel
    for key in ("x1", "y1", "x2", "y2"):
        assert back[key] == pytest.approx(expected[key], abs=px), key

    # And all the way to the file's own coordinates.
    with pymupdf.open(src) as opened:
        unrotated = points_to_unrotated(trace(on_working, prov)["bounds_pt"], opened[1])
    for key, want in zip(("x1", "y1", "x2", "y2"), source_rect):
        assert unrotated[key] == pytest.approx(want, abs=1.0), key


def test_provenance_survives_a_json_round_trip():
    prov = PageProvenance(3, 7, 200, (1700, 2200), (10, 20, 500, 400), (612.0, 792.0), 90)
    assert PageProvenance.from_dict(prov.to_dict()) == prov


def test_whole_page_provenance_matches_the_real_raster(tmp_path: Path):
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    doc.new_page(width=333.3, height=517.7).set_rotation(90)
    doc.save(tmp_path / "d.pdf")
    provs = whole_page_provenance(tmp_path / "d.pdf", dpi=DPI)
    with pymupdf.open(tmp_path / "d.pdf") as opened:
        for prov, page in zip(provs, opened):
            assert prov.source_raster_px == rasterize_page(page, dpi=DPI).size
            assert prov.crop_px == (0, 0, *prov.source_raster_px)


# -- rows ------------------------------------------------------------------

ROW = {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.2}
INSIDE = {"x1": 0.15, "y1": 0.12, "x2": 0.3, "y2": 0.18, "patient_id": "Patient_0001"}
ELSEWHERE = {"x1": 0.1, "y1": 0.6, "x2": 0.3, "y2": 0.7}
EVIDENCE = [{"kind": "row", "path": "evidence/x/r1-row.png", "sha256": "0" * 64}]


def _ready_row(**patch) -> dict:
    """Patient and date confirmed -- all preparation needs. No time."""
    row = rowlib.new_row("doc", 1, 1, ROW, date_suggestion="2026-03-14",
                         patient_suggestion="Patient_0001")
    row, _ = rowlib.apply_patch(row, {"confirm": ["patient_id", "date"], **patch})
    return row


def test_app_derived_values_count_as_confirmed():
    """The pseudonym of the row's NAM and the section's date come from the
    app itself: nothing to confirm row by row."""
    row = rowlib.new_row("doc", 1, 1, ROW, date_suggestion="2026-03-14",
                         patient_suggestion="Patient_0001")
    assert row["schema_version"] == 2
    assert "handwriting_bounds" not in row
    assert row["row_id"] != row["fields"]["patient_id"]["value"]
    assert row["fields"]["date"] == {"value": "2026-03-14", "source": "page_date", "status": "confirmed"}
    assert row["fields"]["patient_id"] == {"value": "Patient_0001", "source": "nam_in_row", "status": "confirmed"}
    # Time is a field like any other, but not a preparation requirement.
    assert rowlib.FIELDS == ("patient_id", "date", "time")
    assert rowlib.unresolved(row) == []


def test_a_row_without_a_single_nam_still_needs_its_patient():
    row = rowlib.new_row("doc", 1, 1, ROW, date_suggestion="2026-03-14", patient_suggestion=None)
    assert rowlib.unresolved(row) == ["patient_id"]


def test_older_suggestions_from_trusted_sources_are_confirmed_once():
    row = rowlib.new_row("doc", 1, 1, ROW, date_suggestion="2026-03-14", patient_suggestion="Patient_0001")
    for name in ("patient_id", "date"):
        row["fields"][name]["status"] = "suggested"  # as older versions stored it
    trusted, changed = rowlib.trust_app_values(row)
    assert changed and rowlib.unresolved(trusted) == []
    assert trusted["revision"] == row["revision"] + 1
    assert not rowlib.trust_app_values(trusted)[1]


def test_the_section_date_follows_unless_the_reviewer_typed_one():
    row = rowlib.new_row("doc", 1, 1, ROW, date_suggestion="2026-03-14", patient_suggestion="Patient_0001")
    ready = rowlib.approve(row, [], 1, EVIDENCE, method=rowlib.METHOD_AUTOMATIC)
    assert ready["approval"]["method"] == "automatic"
    moved = rowlib.follow_page_date(ready, "2026-03-20")
    assert moved["fields"]["date"]["value"] == "2026-03-20"
    assert moved["approval"]["status"] == rowlib.INVALIDATED
    assert moved["approval"]["invalidated_reasons"] == ["section date changed"]

    typed, _ = rowlib.apply_patch(row, {"date": "2026-01-01"})
    assert rowlib.follow_page_date(typed, "2026-03-20") is None


def test_a_row_is_ready_with_no_time_and_its_time_stays_unknown():
    ready = rowlib.approve(_ready_row(), [INSIDE], 1, EVIDENCE)
    assert rowlib.unresolved(ready) == []
    assert ready["approval"]["status"] == rowlib.APPROVED
    assert rowlib.is_current(ready, [INSIDE])
    assert ready["fields"]["time"] == {"value": None, "source": None, "status": "missing"}


def test_readiness_still_requires_patient_date_and_the_row_crop():
    row = rowlib.new_row("doc", 1, 1, ROW)
    with pytest.raises(ValueError, match="patient_id, date"):
        rowlib.approve(row, [], 1, EVIDENCE)
    with pytest.raises(ValueError, match="evidence"):
        rowlib.approve(_ready_row(), [], 1, [])


def test_a_typed_time_is_a_confirmed_override_and_clearing_it_returns_to_unknown():
    row = _ready_row(time="09:30")
    assert row["fields"]["time"] == {"value": "09:30", "source": "manual", "status": "confirmed"}
    cleared, changed = rowlib.apply_patch(row, {"time": None})
    assert changed == ["time"]
    assert cleared["fields"]["time"]["status"] == "missing"


@pytest.mark.parametrize("patch, reason", [
    ({"time": "10:00"}, "time changed"),
    ({"date": "2026-03-15"}, "date changed"),
    ({"patient_id": "Patient_0002"}, "patient changed"),
    ({"bounds": {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.25}}, "row region changed"),
])
def test_any_edit_invalidates_an_approval(patch, reason):
    approved = rowlib.approve(_ready_row(), [INSIDE], 1, EVIDENCE)
    edited, changed = rowlib.apply_patch(approved, patch)
    assert changed
    assert edited["approval"]["status"] == rowlib.INVALIDATED
    assert edited["approval"]["invalidated_reasons"] == [reason]
    assert edited["revision"] == approved["revision"] + 1


def test_a_no_op_edit_keeps_the_approval():
    approved = rowlib.approve(_ready_row(time="09:30"), [INSIDE], 1, EVIDENCE)
    same, changed = rowlib.apply_patch(approved, {"time": "09:30"})
    assert not changed
    assert same["approval"]["status"] == rowlib.APPROVED


def test_only_redactions_touching_the_row_invalidate_it():
    approved = rowlib.approve(_ready_row(), [INSIDE, ELSEWHERE], 1, EVIDENCE)

    untouched, stale = rowlib.revalidate(approved, [INSIDE])  # far box removed
    assert not stale and untouched["approval"]["status"] == rowlib.APPROVED

    moved = {**INSIDE, "x2": 0.35}
    touched, stale = rowlib.revalidate(approved, [moved, ELSEWHERE])
    assert stale and touched["approval"]["invalidated_reasons"] == ["redactions in the row changed"]

    relabelled = {**INSIDE, "patient_id": "Patient_0002"}
    assert rowlib.revalidate(approved, [relabelled, ELSEWHERE])[1]


def test_patient_suggestion_needs_exactly_one_pseudonym_in_the_row():
    other = {**INSIDE, "x1": 0.4, "x2": 0.45, "patient_id": "Patient_0002"}
    assert rowlib.suggest_patient(ROW, [INSIDE]) == "Patient_0001"
    assert rowlib.suggest_patient(ROW, [INSIDE, {**INSIDE}]) == "Patient_0001"
    assert rowlib.suggest_patient(ROW, [INSIDE, other]) is None
    assert rowlib.suggest_patient(ROW, [ELSEWHERE]) is None


# -- migration from schema 1 --------------------------------------------------


def _v1_row(approval_status: str = "approved") -> dict:
    """A row as schema 1 stored it: handwriting region, handwriting evidence."""
    return {
        "schema_version": 1, "row_id": "row_0123456789ab", "document_id": "doc",
        "working_page": 1, "source_page": 1, "bounds": ROW,
        "handwriting_bounds": {"x1": 0.5, "y1": 0.1, "x2": 0.9, "y2": 0.2},
        "fields": {
            "patient_id": {"value": "Patient_0001", "source": "nam_in_row", "status": "confirmed"},
            "date": {"value": "2026-03-14", "source": "page_date", "status": "confirmed"},
            "time": {"value": "09:30", "source": "manual", "status": "confirmed"},
        },
        "evidence": [{"kind": "row", "path": "evidence/row_0123456789ab/r4-row.png", "sha256": "a" * 64},
                     {"kind": "handwriting", "path": "evidence/row_0123456789ab/r4-handwriting.png", "sha256": "b" * 64}],
        "revision": 4, "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00",
        "approval": {"status": approval_status, "approved_revision": 4, "approved_at": "2026-01-02T00:00:00+00:00",
                     "redaction_revision": 1, "fingerprint": "f" * 64, "invalidated_at": None,
                     "invalidated_reasons": []},
    }


def test_migration_keeps_legacy_data_and_asks_for_renewed_readiness():
    migrated, changed = rowlib.migrate_row(_v1_row())
    assert changed and migrated["schema_version"] == 2
    assert "handwriting_bounds" not in migrated
    legacy = migrated["legacy"]
    assert legacy["migrated_from_schema"] == 1
    assert legacy["handwriting_bounds"] == {"x1": 0.5, "y1": 0.1, "x2": 0.9, "y2": 0.2}
    assert [e["kind"] for e in legacy["evidence"]] == ["row", "handwriting"]
    assert legacy["approval"]["status"] == "approved"
    # Saved work survives; the old approval does not silently carry over.
    assert migrated["fields"]["time"]["value"] == "09:30"
    assert migrated["approval"]["status"] == rowlib.INVALIDATED
    assert migrated["approval"]["invalidated_reasons"] == [rowlib.MIGRATION_REASON]
    assert migrated["evidence"] == [] and migrated["revision"] == 5
    assert not rowlib.is_current(migrated, [])
    # Ready again under the new rules: no handwriting needed.
    assert rowlib.approve(migrated, [], 2, EVIDENCE)["approval"]["status"] == rowlib.APPROVED


def test_migration_of_an_unapproved_row_and_idempotence():
    migrated, _ = rowlib.migrate_row(_v1_row("unapproved"))
    assert migrated["approval"]["status"] == rowlib.UNAPPROVED
    again, changed = rowlib.migrate_row(migrated)
    assert not changed and again == migrated
