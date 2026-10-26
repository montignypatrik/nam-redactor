"""End-to-end: saved documents, reviewed redactions, rows, evidence, export.

OCR is stubbed (see ``tests.test_web_api.FakeEngine``); everything else --
cropping, rendering, persistence, evidence -- is real.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from nam_redactor.web.app import create_app
from tests.privacy import assert_no_identifiers
from tests.test_web_api import (
    doctor_data,
    NAM_TEXT,
    FakeEngine,
    make_pdf_bytes,
    no_ner,
    upload_only,
    wait_for_status,
)

ROW = {"x1": 0.02, "y1": 0.10, "x2": 0.98, "y2": 0.30}
HAND = {"x1": 0.50, "y1": 0.10, "x2": 0.98, "y2": 0.30}


def make_app(tmp_path: Path, page_texts: dict[int, str] | None = None) -> TestClient:
    texts = page_texts if page_texts is not None else {1: NAM_TEXT, 2: NAM_TEXT}
    return TestClient(create_app(
        engine_factory=lambda: FakeEngine(texts),
        dir_root=tmp_path,
        name_model_factory=no_ner,
    ))


def whole_document(client: TestClient, dates=("2026-03-14", "2026-03-15")) -> str:
    """Upload, keep every page, detect, and wait for review."""
    job_id = upload_only(client).json()["job_id"]
    pages = [{"page_num": n, "boxes": [], "date": d} for n, d in enumerate(dates, start=1)]
    assert client.post(f"/api/jobs/{job_id}/start", json={"pages": pages}).status_code == 202
    wait_for_status(client, job_id, {"awaiting_review"})
    return job_id


def pages_of(client: TestClient, job_id: str) -> list[dict]:
    return client.get(f"/api/jobs/{job_id}/pages").json()["pages"]


def save_review(client: TestClient, job_id: str, boxes_by_page: dict[int, list] | None = None) -> dict:
    pages = pages_of(client, job_id)
    payload = [
        {"page_num": p["page_num"], "boxes": (boxes_by_page or {}).get(p["page_num"], p["boxes"])}
        for p in pages
    ]
    response = client.put(f"/api/jobs/{job_id}/review", json={"pages": payload})
    assert response.status_code == 200, response.text
    return response.json()


def ready_row(client: TestClient, job_id: str, page: int = 1, time: str = "09:30") -> dict:
    """A row with every field resolved. The patient suggestion is confirmed
    when there is one, else the document's first pseudonym is chosen."""
    row = client.post(f"/api/jobs/{job_id}/rows", json={"working_page": page, "bounds": ROW}).json()
    patch = {"time": time, "confirm": ["date"]}
    if row["fields"]["patient_id"]["value"]:
        patch["confirm"].append("patient_id")
    else:
        patch["patient_id"] = client.get(f"/api/jobs/{job_id}/rows").json()["patients"][0]
    patched = client.patch(f"/api/jobs/{job_id}/rows/{row['row_id']}", json=patch)
    assert patched.status_code == 200, patched.text
    return patched.json()


def approve(client: TestClient, job_id: str, row: dict) -> dict:
    response = client.post(
        f"/api/jobs/{job_id}/rows/{row['row_id']}/approve", json={"revision": row["revision"]}
    )
    assert response.status_code == 200, response.text
    return response.json()


def evidence_image(client: TestClient, job_id: str, row_id: str, kind: str = "row") -> np.ndarray:
    response = client.get(f"/api/jobs/{job_id}/rows/{row_id}/evidence/{kind}.png")
    assert response.status_code == 200
    return np.asarray(Image.open(io.BytesIO(response.content)).convert("L"))


def manifest(client: TestClient, job_id: str) -> dict:
    response = client.get(f"/api/jobs/{job_id}/manifest.json")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    return response.json()


# -- reviewed redactions are saved, not just rendered -----------------------


def test_finalize_saves_the_review_and_keeps_detections_separate(tmp_path: Path):
    with make_app(tmp_path) as client:
        job_id = whole_document(client)
        detected = pages_of(client, job_id)[0]["boxes"]
        manual = {"x1": 0.6, "y1": 0.6, "x2": 0.8, "y2": 0.7}
        pages = [{"page_num": 1, "boxes": detected + [manual]}, {"page_num": 2, "boxes": []}]

        assert client.post(f"/api/jobs/{job_id}/finalize", json={"pages": pages}).status_code == 202
        wait_for_status(client, job_id, {"done"})

        after = client.get(f"/api/jobs/{job_id}/pages").json()
        assert after["review_revision"] == 1
        page1, page2 = after["pages"]
        assert len(page1["boxes"]) == 2 and len(page1["detected_boxes"]) == 1
        assert page2["boxes"] == [] and len(page2["detected_boxes"]) == 1


def test_editing_after_render_marks_the_output_stale(tmp_path: Path):
    with make_app(tmp_path) as client:
        job_id = whole_document(client)
        client.post(f"/api/jobs/{job_id}/finalize", json={"pages": []})
        wait_for_status(client, job_id, {"done"})
        result = save_review(client, job_id, {1: []})
        assert result["changed"] and result["status"] == "awaiting_review"
        assert client.get(f"/api/jobs/{job_id}/result").status_code == 409


def test_a_stale_tab_cannot_overwrite_a_newer_review(tmp_path: Path):
    with make_app(tmp_path) as client:
        job_id = whole_document(client)
        save_review(client, job_id)
        stale = client.put(f"/api/jobs/{job_id}/review", json={"pages": [], "base_revision": 0})
        assert stale.status_code == 409


# -- evidence follows the accepted redactions -------------------------------


def test_evidence_crops_reflect_added_and_removed_redactions(tmp_path: Path):
    with make_app(tmp_path) as client:
        job_id = whole_document(client)
        save_review(client, job_id)
        row = ready_row(client, job_id)
        nam_box = pages_of(client, job_id)[0]["boxes"][0]

        def region(img: np.ndarray, box: dict) -> np.ndarray:
            """``box`` (page-normalized) within the row crop."""
            h, w = img.shape
            fx = lambda v: int((v - ROW["x1"]) / (ROW["x2"] - ROW["x1"]) * w)  # noqa: E731
            fy = lambda v: int((v - ROW["y1"]) / (ROW["y2"] - ROW["y1"]) * h)  # noqa: E731
            return img[fy(box["y1"]) + 2:fy(box["y2"]) - 2, fx(box["x1"]) + 2:fx(box["x2"]) - 2]

        before = evidence_image(client, job_id, row["row_id"])
        # Detected NAM is blacked out; only its white pseudonym label is light.
        assert (region(before, nam_box) < 60).mean() > 0.5

        manual = {"x1": 0.55, "y1": 0.12, "x2": 0.9, "y2": 0.28}
        blank = region(before, manual)
        assert blank.min() > 200  # nothing there yet: white page

        # Reviewer removes the detected box and adds their own.
        save_review(client, job_id, {1: [manual]})
        after = evidence_image(client, job_id, row["row_id"])
        assert region(after, manual).max() < 30
        assert (region(after, nam_box) < 60).mean() < 0.3  # page content shows again


# -- approval invalidation -----------------------------------------------------


def test_redaction_changes_invalidate_only_the_rows_they_touch(tmp_path: Path):
    with make_app(tmp_path) as client:
        job_id = whole_document(client)
        save_review(client, job_id)
        touched = approve(client, job_id, ready_row(client, job_id, page=1))
        untouched = approve(client, job_id, ready_row(client, job_id, page=2))
        assert touched["approval_current"] and untouched["approval_current"]

        far = {"x1": 0.1, "y1": 0.8, "x2": 0.3, "y2": 0.9}  # outside every row
        page1 = pages_of(client, job_id)[0]["boxes"]
        result = save_review(client, job_id, {2: pages_of(client, job_id)[1]["boxes"] + [far]})
        assert result["invalidated_rows"] == []

        inside = {"x1": 0.6, "y1": 0.15, "x2": 0.7, "y2": 0.2}
        result = save_review(client, job_id, {1: page1 + [inside]})
        assert result["invalidated_rows"] == [touched["row_id"]]

        rows = {r["row_id"]: r for r in client.get(f"/api/jobs/{job_id}/rows").json()["rows"]}
        assert rows[touched["row_id"]]["approval"]["status"] == "invalidated"
        assert rows[untouched["row_id"]]["approval"]["status"] == "approved"


def test_field_edits_invalidate_and_stale_approvals_are_refused(tmp_path: Path):
    with make_app(tmp_path) as client:
        job_id = whole_document(client)
        save_review(client, job_id)
        row = approve(client, job_id, ready_row(client, job_id))

        edited = client.patch(f"/api/jobs/{job_id}/rows/{row['row_id']}", json={"time": "11:00"}).json()
        assert edited["approval"]["status"] == "invalidated"
        assert edited["approval"]["invalidated_reasons"] == ["time changed"]

        # Approving the revision the reviewer saw before the edit is refused.
        stale = client.post(f"/api/jobs/{job_id}/rows/{row['row_id']}/approve", json={"revision": row["revision"]})
        assert stale.status_code == 409
        assert approve(client, job_id, edited)["approval_current"]


def test_rows_cannot_be_approved_before_the_review_is_saved(tmp_path: Path):
    with make_app(tmp_path) as client:
        job_id = whole_document(client)
        row = ready_row(client, job_id)
        response = client.post(f"/api/jobs/{job_id}/rows/{row['row_id']}/approve", json={"revision": row["revision"]})
        assert response.status_code == 409


def test_unknown_pseudonyms_and_bad_values_are_rejected(tmp_path: Path):
    with make_app(tmp_path) as client:
        job_id = whole_document(client)
        row = ready_row(client, job_id)
        url = f"/api/jobs/{job_id}/rows/{row['row_id']}"
        assert client.patch(url, json={"patient_id": "Patient_9999"}).status_code == 400
        assert client.patch(url, json={"time": "25:00"}).status_code == 400
        assert client.patch(url, json={"date": "2026-02-30"}).status_code == 400
        assert client.patch(url, json={"confirm": ["billing"]}).status_code == 400


# -- repeated visits and zones -------------------------------------------------


def section_rows(client: TestClient, job_id: str) -> dict[int, dict]:
    """The automatic row of each section, by working page."""
    rows = client.get(f"/api/jobs/{job_id}/rows").json()["rows"]
    return {r["working_page"]: r for r in rows if r["origin"] == "section"}


def test_every_section_is_a_row_traced_to_its_own_zone(tmp_path: Path):
    """Three sections -- two on one page, drawn bottom-first, one whole page --
    become three rows for the same patient, each tracing to its own zone."""
    with make_app(tmp_path, {1: NAM_TEXT, 2: NAM_TEXT}) as client:
        job_id = upload_only(client).json()["job_id"]
        zones = {"pages": [
            {"page_num": 2, "date": "2026-03-20", "boxes": [{"x1": 0, "y1": 0, "x2": 1, "y2": 1}]},
            {"page_num": 1, "date": "2026-03-14", "boxes": [
                {"x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 1.0},  # bottom half: section 2
                {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 0.5},  # top half:    section 1
            ]},
        ]}
        assert client.post(f"/api/jobs/{job_id}/start", json=zones).status_code == 202
        wait_for_status(client, job_id, {"awaiting_review"})
        assert [p["source_page"] for p in pages_of(client, job_id)] == [1, 1, 2]
        save_review(client, job_id)

        rows = section_rows(client, job_id)
        assert sorted(rows) == [1, 2, 3]
        assert all(r["bounds"] == {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0} for r in rows.values())
        assert [rows[n]["fields"]["date"]["value"] for n in (1, 2, 3)] == ["2026-03-14", "2026-03-14", "2026-03-20"]

        patient = client.get(f"/api/jobs/{job_id}/rows").json()["patients"][0]
        for page, time in ((1, "09:00"), (2, "09:30"), (3, "14:00")):
            row = client.patch(f"/api/jobs/{job_id}/rows/{rows[page]['row_id']}", json={
                "patient_id": patient, "time": time, "confirm": ["date"],
            }).json()
            approve(client, job_id, row)

        data = manifest(client, job_id)
        assert {k: data["summary"][k] for k in ("rows_total", "rows_exported", "rows_excluded")} == {"rows_total": 3, "rows_exported": 3, "rows_excluded": 0}
        by_time = {r["time"]: r for r in data["rows"]}
        assert {r["patient_id"] for r in data["rows"]} == {patient}

        top, bottom, whole = (by_time[t]["row_region"] for t in ("09:00", "09:30", "14:00"))
        assert top["source_page"] == 1 and abs(top["bounds"]["y1"]) < 0.005 and abs(top["bounds"]["y2"] - 0.5) < 0.005
        assert bottom["source_page"] == 1 and abs(bottom["bounds"]["y1"] - 0.5) < 0.005 and abs(bottom["bounds"]["y2"] - 1.0) < 0.005
        assert whole["source_page"] == 2 and whole["bounds"] == {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
        assert by_time["14:00"]["date"] == "2026-03-20"
        # Points follow the displayed page size (300 x 200 pt).
        assert abs(whole["bounds_pt"]["x2"] - 300.0) < 0.5 and abs(whole["bounds_pt"]["y2"] - 200.0) < 0.5


def test_section_rows_are_fixed_to_their_section(tmp_path: Path):
    with make_app(tmp_path) as client:
        job_id = whole_document(client)
        row = section_rows(client, job_id)[1]
        url = f"/api/jobs/{job_id}/rows/{row['row_id']}"
        assert client.patch(url, json={"bounds": ROW}).status_code == 400
        assert client.delete(url).status_code == 409
        # Listing again never duplicates them.
        assert len(section_rows(client, job_id)) == 2
        assert len(client.get(f"/api/jobs/{job_id}/rows").json()["rows"]) == 2


def test_a_section_row_date_follows_the_section_unless_typed(tmp_path: Path):
    with make_app(tmp_path) as client:
        job_id = whole_document(client)
        rows = section_rows(client, job_id)
        pages = pages_of(client, job_id)
        payload = [{"page_num": p["page_num"], "boxes": p["boxes"], "date": "2026-04-01"} for p in pages]

        # Row 2's date is typed by the reviewer; row 1's comes from its section.
        client.patch(f"/api/jobs/{job_id}/rows/{rows[2]['row_id']}", json={"date": "2026-02-02"})
        assert client.put(f"/api/jobs/{job_id}/review", json={"pages": payload}).status_code == 200

        after = section_rows(client, job_id)
        assert after[1]["fields"]["date"] == {"value": "2026-04-01", "source": "page_date", "status": "confirmed"}
        assert after[2]["fields"]["date"]["value"] == "2026-02-02"


def test_prepare_marks_every_known_row_ready_automatically(tmp_path: Path):
    """No per-row confirmation: rows whose patient and date the app knows are
    made ready in one go; a section without a NAM is reported, not guessed."""
    with make_app(tmp_path, {1: NAM_TEXT, 2: "Page sans numero"}) as client:
        job_id = whole_document(client)
        assert client.post(f"/api/jobs/{job_id}/prepare").status_code == 409  # review not saved
        save_review(client, job_id)
        rows = section_rows(client, job_id)
        result = client.post(f"/api/jobs/{job_id}/prepare").json()
        assert result["prepared"] == [rows[1]["row_id"]]
        assert result["skipped"] == {rows[2]["row_id"]: "needs patient"}

        after = section_rows(client, job_id)
        assert after[1]["approval"]["status"] == "approved"
        assert after[1]["approval"]["method"] == "automatic"
        assert [e["kind"] for e in after[1]["evidence"]] == ["row"]
        # Idempotent: nothing more to do.
        assert client.post(f"/api/jobs/{job_id}/prepare").json()["prepared"] == []

        (exported,) = manifest(client, job_id)["rows"]
        assert exported["ready_for_extraction"]["method"] == "automatic"


# -- persistence -----------------------------------------------------------------


def test_everything_survives_a_restart(tmp_path: Path):
    with make_app(tmp_path) as client:
        job_id = upload_only(client).json()["job_id"]
        zones = {"pages": [{"page_num": 1, "date": "2026-03-14", "boxes": [
            {"x1": 0.0, "y1": 0.2, "x2": 1.0, "y2": 0.7}]}]}
        client.post(f"/api/jobs/{job_id}/start", json=zones)
        wait_for_status(client, job_id, {"awaiting_review"})
        manual = {"x1": 0.6, "y1": 0.6, "x2": 0.8, "y2": 0.7}
        save_review(client, job_id, {1: pages_of(client, job_id)[0]["boxes"] + [manual]})
        approved = approve(client, job_id, ready_row(client, job_id, time="08:15"))
        pending = ready_row(client, job_id, time="10:45")
        before_pages = client.get(f"/api/jobs/{job_id}/pages").json()
        before_manifest = manifest(client, job_id)

    # A new app on the same data directory: nothing is held in memory.
    with make_app(tmp_path) as client:
        listed = client.get("/api/jobs").json()["documents"]
        assert [d["job_id"] for d in listed] == [job_id]
        # The section's own row plus the two drawn by hand.
        assert listed[0]["rows_total"] == 3 and listed[0]["rows_approved"] == 1

        status = client.get(f"/api/jobs/{job_id}").json()
        assert status["zone_request"]["zones"]["1"][0]["y1"] == 0.2

        after_pages = client.get(f"/api/jobs/{job_id}/pages").json()
        assert after_pages["pages"] == before_pages["pages"]
        assert after_pages["review_revision"] == before_pages["review_revision"]

        rows = {r["row_id"]: r for r in client.get(f"/api/jobs/{job_id}/rows").json()["rows"]}
        assert rows[approved["row_id"]]["approval_current"]
        assert rows[pending["row_id"]]["fields"]["time"]["value"] == "10:45"
        assert rows[pending["row_id"]]["approval"]["status"] == "unapproved"

        after_manifest = manifest(client, job_id)
        for key in ("rows", "excluded", "summary"):
            assert after_manifest[key] == before_manifest[key]

        # Typed NAMs still resolve to the same pseudonym after the restart.
        pid = client.post(f"/api/jobs/{job_id}/nam", json={"nam": "HOGP 3361 0615"}).json()["patient_id"]
        assert pid == after_pages["patients"][0]


def test_an_interrupted_detection_returns_to_the_zone_step(tmp_path: Path):
    with make_app(tmp_path) as client:
        job_id = upload_only(client).json()["job_id"]
        store = client.app.state.store
        job = store.get(job_id)
        job.status = "detecting"  # as if the app died mid-OCR
        job.zone_request = {"zones": {"1": [{"x1": 0, "y1": 0, "x2": 1, "y2": 1}]}, "dates": {"1": "2026-01-01"}}
        store.save(job)

    with make_app(tmp_path) as client:
        status = client.get(f"/api/jobs/{job_id}").json()
        assert status["status"] == "awaiting_zones"
        assert status["zone_request"]["dates"] == {"1": "2026-01-01"}


def test_deleting_a_document_removes_its_folder(tmp_path: Path):
    with make_app(tmp_path) as client:
        job_id = whole_document(client)
        save_review(client, job_id)
        approve(client, job_id, ready_row(client, job_id))
        folder = tmp_path / "documents" / job_id
        assert (folder / "evidence").is_dir()
        assert client.delete(f"/api/jobs/{job_id}").status_code == 204
        assert not folder.exists()
    with make_app(tmp_path) as client:
        assert client.get("/api/jobs").json()["documents"] == []


# -- export privacy --------------------------------------------------------------


def test_manifest_holds_no_identifiers_and_reports_every_excluded_row(tmp_path: Path):
    with make_app(tmp_path, {1: NAM_TEXT + " HOGUE Pierre", 2: "Nom: Tremblay"}) as client:
        upload = client.post(
            "/api/jobs",
            files={"file": ("HOGUE_Pierre_2026.pdf", make_pdf_bytes(), "application/pdf")},
            data=doctor_data(client),
        ).json()
        job_id = upload["job_id"]
        client.post(f"/api/jobs/{job_id}/start", json={"pages": [
            {"page_num": 1, "boxes": [], "date": "2026-03-14"},
            {"page_num": 2, "boxes": [], "date": "2026-03-14"},
        ]})
        wait_for_status(client, job_id, {"awaiting_review"})
        save_review(client, job_id)

        approve(client, job_id, ready_row(client, job_id))
        unfinished = client.post(f"/api/jobs/{job_id}/rows", json={"working_page": 2, "bounds": ROW}).json()
        invalidated = approve(client, job_id, ready_row(client, job_id, time="16:00"))
        client.patch(f"/api/jobs/{job_id}/rows/{invalidated['row_id']}", json={"time": "16:30"})

        data = manifest(client, job_id)
        # Names and the uploaded filename are words that generated IDs and
        # hashes cannot contain; the NAM is checked form by form and by the
        # detector, never by a short digit fragment.
        assert_no_identifiers(data, "HOGUE", "Pierre", "Tremblay", "HOGUE_Pierre",
                              "confidence", "billing_code")
        assert "ready for extraction" in data["notice"]
        assert "does not establish billing eligibility" in data["notice"]

        # Two untouched section rows + the three drawn here.
        assert {k: data["summary"][k] for k in ("rows_total", "rows_exported", "rows_excluded")} == {"rows_total": 5, "rows_exported": 1, "rows_excluded": 4}
        excluded = {e["row_id"]: e["reasons"] for e in data["excluded"]}
        assert excluded[unfinished["row_id"]][0] == "not marked ready for extraction"
        assert excluded[unfinished["row_id"]][1].startswith("unresolved: ")
        assert excluded[invalidated["row_id"]] == ["readiness invalidated: time changed"]

        exported = data["rows"][0]
        assert [ref["kind"] for ref in exported["evidence"]] == ["row"]
        for ref in exported["evidence"]:
            frozen = client.get(f"/api/jobs/{job_id}/{ref['path']}")
            assert frozen.status_code == 200
            assert hashlib.sha256(frozen.content).hexdigest() == ref["sha256"]


def test_tampered_evidence_is_excluded_from_export(tmp_path: Path):
    with make_app(tmp_path) as client:
        job_id = whole_document(client)
        save_review(client, job_id)
        row = approve(client, job_id, ready_row(client, job_id))
        path = tmp_path / "documents" / job_id / row["evidence"][0]["path"]
        path.write_bytes(b"not the approved crop")
        data = manifest(client, job_id)
        assert data["rows"] == []
        reasons = {e["row_id"]: e["reasons"] for e in data["excluded"]}
        assert reasons[row["row_id"]] == ["evidence file missing or altered"]


# -- sections in place on the original page ---------------------------------------


def test_sections_render_in_place_on_their_source_page(tmp_path: Path):
    """Two zones on one page, drawn bottom-first: each reviewed box lands at
    its mapped position on the original, and nothing outside it is touched."""
    with make_app(tmp_path, {1: "", 2: ""}) as client:
        job_id = upload_only(client).json()["job_id"]
        zones = {"pages": [{"page_num": 1, "date": "2026-03-14", "boxes": [
            {"x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 1.0},
            {"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 0.5},
        ]}]}
        assert client.get(f"/api/jobs/{job_id}/source/1.png?redacted=true").status_code == 409
        client.post(f"/api/jobs/{job_id}/start", json=zones)
        wait_for_status(client, job_id, {"awaiting_review"})

        top, bottom = pages_of(client, job_id)
        assert top["source_crop"] == {"x1": 0.0, "y1": 0.0, "x2": 0.4992, "y2": 0.498801}
        assert bottom["source_crop"]["y1"] == 0.498801 and bottom["source_crop"]["x2"] == 1.0

        # A box filling the middle of the bottom section only.
        box = {"x1": 0.4, "y1": 0.4, "x2": 0.6, "y2": 0.6}
        save_review(client, job_id, {1: [], 2: [box]})
        png = client.get(f"/api/jobs/{job_id}/source/1.png?redacted=true")
        assert png.status_code == 200
        img = np.asarray(Image.open(io.BytesIO(png.content)).convert("L"))
        h, w = img.shape
        # Working (0.4..0.6) of the bottom half -> source y 0.7..0.8, x 0.4..0.6.
        inside = img[int(0.72 * h):int(0.78 * h), int(0.42 * w):int(0.58 * w)]
        assert inside.max() < 30
        above = img[int(0.55 * h):int(0.65 * h), int(0.42 * w):int(0.58 * w)]
        assert above.min() > 200

        plain = client.get(f"/api/jobs/{job_id}/source/1.png")
        plain_img = np.asarray(Image.open(io.BytesIO(plain.content)).convert("L"))
        assert plain_img[int(0.72 * h):int(0.78 * h), int(0.42 * w):int(0.58 * w)].min() > 200
        assert client.get(f"/api/jobs/{job_id}/source/9.png").status_code == 404



# -- preparation: whole row as evidence, time optional ------------------------------


def test_a_row_is_ready_and_exported_with_no_time_and_no_handwriting_region(tmp_path: Path):
    with make_app(tmp_path) as client:
        job_id = whole_document(client)
        save_review(client, job_id)
        row = section_rows(client, job_id)[1]
        row = client.patch(f"/api/jobs/{job_id}/rows/{row['row_id']}",
                           json={"confirm": ["patient_id", "date"]}).json()
        assert row["unresolved"] == []
        ready = approve(client, job_id, row)
        assert ready["approval_current"]
        assert [ref["kind"] for ref in ready["evidence"]] == ["row"]

        data = manifest(client, job_id)
        (exported,) = data["rows"]
        assert exported["time"] is None and exported["time_status"] == "to_be_extracted"
        assert exported["field_sources"]["time"] is None
        assert "handwriting_region" not in exported
        assert data["schema_version"] == 3 and data["row_schema_version"] == 2

        # The frozen crop is the full row, redacted, with its hash and provenance.
        ref = exported["evidence"][0]
        png = client.get(f"/api/jobs/{job_id}/{ref['path']}").content
        assert hashlib.sha256(png).hexdigest() == ref["sha256"]
        width, height = Image.open(io.BytesIO(png)).size
        x1, y1, x2, y2 = ref["crop_px"]
        assert (x2 - x1, y2 - y1) == (width, height)
        page = pages_of(client, job_id)[0]
        assert (width, height) == (page["width"], page["height"])  # the whole section
        assert ref["region"]["source_page"] == 1
        assert exported["ready_for_extraction"]["redaction_revision"] == 1

        stored = section_rows(client, job_id)[1]["fields"]["time"]
        assert stored == {"value": None, "source": None, "status": "missing"}


def test_an_optional_time_is_validated_and_exported_when_typed(tmp_path: Path):
    with make_app(tmp_path) as client:
        job_id = whole_document(client)
        save_review(client, job_id)
        row = section_rows(client, job_id)[1]
        url = f"/api/jobs/{job_id}/rows/{row['row_id']}"
        for bad in ("25:00", "9:30", "09:60", "noon"):
            assert client.patch(url, json={"time": bad}).status_code == 400, bad
        row = client.patch(url, json={"time": "09:30", "confirm": ["patient_id", "date"]}).json()
        approve(client, job_id, row)
        (exported,) = manifest(client, job_id)["rows"]
        assert exported["time"] == "09:30" and exported["time_status"] == "entered"
        assert exported["field_sources"]["time"] == "manual"


def test_handwriting_regions_are_no_longer_accepted(tmp_path: Path):
    with make_app(tmp_path) as client:
        job_id = whole_document(client)
        row = section_rows(client, job_id)[1]
        response = client.patch(f"/api/jobs/{job_id}/rows/{row['row_id']}", json={"handwriting_bounds": HAND})
        assert response.status_code == 400
        assert client.get(f"/api/jobs/{job_id}/rows/{row['row_id']}/evidence/handwriting.png").status_code == 404


def test_a_legacy_document_reopens_and_its_rows_ask_to_be_marked_ready_again(tmp_path: Path):
    """A document saved under row schema 1 -- handwriting region, handwriting
    evidence, approved -- still loads, keeps that data, and needs renewed
    readiness under the new contract."""
    with make_app(tmp_path) as client:
        job_id = whole_document(client)
        save_review(client, job_id)
        row = section_rows(client, job_id)[1]
        row = client.patch(f"/api/jobs/{job_id}/rows/{row['row_id']}",
                           json={"time": "09:30", "confirm": ["patient_id", "date"]}).json()
        row = approve(client, job_id, row)

    # Rewrite the saved row the way schema 1 stored it.
    folder = tmp_path / "documents" / job_id
    record = json.loads((folder / "document.json").read_text(encoding="utf-8"))
    (saved,) = [r for r in record["rows"] if r["row_id"] == row["row_id"]]
    hand_path = f"evidence/{row['row_id']}/r{saved['revision']}-handwriting.png"
    (folder / hand_path).write_bytes(b"legacy handwriting crop")
    saved.update(schema_version=1, handwriting_bounds=HAND)
    saved.pop("origin")
    saved.pop("legacy", None)
    saved["evidence"].append({"kind": "handwriting", "path": hand_path, "sha256": "b" * 64})
    (folder / "document.json").write_text(json.dumps(record), encoding="utf-8")

    with make_app(tmp_path) as client:
        rows = {r["row_id"]: r for r in client.get(f"/api/jobs/{job_id}/rows").json()["rows"]}
        migrated = rows[row["row_id"]]
        assert migrated["schema_version"] == 2
        assert migrated["legacy"]["handwriting_bounds"] == HAND
        assert migrated["fields"]["time"]["value"] == "09:30"
        assert migrated["approval"]["status"] == "invalidated"
        assert not migrated["approval_current"]
        assert (folder / hand_path).exists()  # legacy files are kept

        reasons = {e["row_id"]: e["reasons"] for e in manifest(client, job_id)["excluded"]}
        assert "mark ready for extraction again" in reasons[row["row_id"]][0]

        # Stored once, migrated: a reload finds schema 2 already.
        on_disk = json.loads((folder / "document.json").read_text(encoding="utf-8"))
        assert {r["schema_version"] for r in on_disk["rows"]} == {2}

        ready = approve(client, job_id, migrated)
        assert ready["approval_current"]
        assert (folder / hand_path).exists()
        (exported,) = manifest(client, job_id)["rows"]
        assert exported["row_id"] == row["row_id"]



def test_no_patient_is_a_choice_that_makes_the_row_ready(tmp_path: Path):
    with make_app(tmp_path) as client:
        job_id = whole_document(client)
        save_review(client, job_id)
        row = client.post(f"/api/jobs/{job_id}/rows", json={"working_page": 1, "bounds": ROW}).json()
        url = f"/api/jobs/{job_id}/rows/{row['row_id']}"
        row = client.patch(url, json={"patient_id": None, "confirm": ["date"]}).json()
        assert row["unresolved"] == ["patient_id"]  # merely unset still blocks

        row = client.patch(url, json={"no_patient": True}).json()
        assert row["fields"]["patient_id"] == {"value": None, "source": "manual", "status": "confirmed"}
        assert row["unresolved"] == []
        # Made ready like any other row, and exported with no patient.
        assert row["row_id"] in client.post(f"/api/jobs/{job_id}/prepare").json()["prepared"]
        exported = [r for r in manifest(client, job_id)["rows"] if r["row_id"] == row["row_id"]]
        assert exported[0]["patient_id"] is None

        # Choosing a patient afterwards is a change: the readiness is invalidated.
        patient = client.get(f"/api/jobs/{job_id}/rows").json()["patients"][0]
        chosen = client.patch(url, json={"patient_id": patient}).json()
        assert chosen["approval"]["invalidated_reasons"] == ["patient changed"]
