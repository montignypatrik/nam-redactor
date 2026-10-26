"""API-level tests for the local web app, with OCR stubbed out.

The engine is injected through ``create_app(name_model_factory=no_ner, engine_factory=...)``, so these
exercise the full HTTP surface -- upload, SSE, review, finalize, download --
in seconds without loading Paddle.
"""
from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from nam_redactor.ocr import OCRResult
from nam_redactor.web.app import create_app

# "HOGP 3361 0615" is a structurally valid NAM (the existing e2e test relies
# on it), so the real locator accepts it without any OCR involved.
NAM_TEXT = "NAM: HOGP 3361 0615"
NAM_BBOX = (20, 70, 200, 90)
NAM_TOKEN = "HOGP"


class FakeEngine:
    """Stands in for PaddleOCREngine with a fixed line of text per page."""

    def __init__(self, page_texts: dict[int, str] | None = None):
        self.page_texts = (
            page_texts if page_texts is not None
            else {1: NAM_TEXT, 2: "Page 2 sans numero"}
        )
        self.calls = 0

    def process_image(self, image_path: str, page_num: int = 1) -> list[OCRResult]:
        self.calls += 1
        text = self.page_texts.get(page_num)
        if not text:
            return []
        return [
            OCRResult(
                page=page_num,
                text=text,
                bbox=NAM_BBOX,
                confidence=0.99,
            )
        ]


def no_ner():
    """Keep spaCy out of the API tests: NAM-anchored names still run."""
    return None


class BoomEngine:
    def process_image(self, image_path: str, page_num: int = 1) -> list[OCRResult]:
        raise RuntimeError(f"paddle exploded reading {image_path}")


def make_pdf_bytes() -> bytes:
    doc = pymupdf.open()
    page1 = doc.new_page(width=300, height=200)
    page1.insert_text((20, 40), "Clinique Medicale", fontsize=14)
    page1.insert_text((20, 80), NAM_TEXT, fontsize=14)
    page2 = doc.new_page(width=300, height=200)
    page2.insert_text((20, 40), "Page 2 sans numero", fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(name_model_factory=no_ner, engine_factory=FakeEngine, dir_root=tmp_path)
    with TestClient(app) as test_client:
        yield test_client


#: The synthetic doctor every test upload is assigned.
TEST_DOCTOR = {
    "display_name": "Dre Test Synthétique",
    "licence_number": "1-23456-7",
    "group_number": "01234",
    "sector_number": "2",
}


def ensure_doctor(client: TestClient) -> str:
    """The id of an active doctor profile, created on first use."""
    profiles = client.get("/api/doctors").json()["profiles"]
    if profiles:
        return profiles[0]["profile_id"]
    response = client.post("/api/doctors", json=TEST_DOCTOR)
    assert response.status_code == 201, response.text
    return response.json()["profile_id"]


def doctor_data(client: TestClient) -> dict[str, str]:
    """Form fields assigning the test doctor to an upload."""
    return {"doctor_profile_id": ensure_doctor(client)}


def upload_only(client: TestClient, payload: bytes | None = None, dpi: int = 150):
    """Upload without choosing zones: the job waits at the zone step."""
    return client.post(
        "/api/jobs",
        files={"file": ("document.pdf", payload or make_pdf_bytes(), "application/pdf")},
        data={"dpi": str(dpi), **doctor_data(client)},
    )


def upload(client: TestClient, payload: bytes | None = None, dpi: int = 150):
    """Upload and start detection on the whole document (no zones)."""
    response = upload_only(client, payload, dpi)
    if response.status_code == 202:
        body = response.json()
        started = client.post(
            f"/api/jobs/{body['job_id']}/start",
            json={"pages": dated_pages(body["total_pages"])},
        )
        assert started.status_code == 202, started.text
    return response


def dated_pages(count: int, date: str = "2026-09-01") -> list[dict]:
    """Every page kept whole, each with the date the zone step requires."""
    return [{"page_num": n, "boxes": [], "date": date} for n in range(1, count + 1)]


def wait_for_status(client: TestClient, job_id: str, wanted: set[str], timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    status = None
    while time.monotonic() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()["status"]
        if status in wanted:
            return status
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {wanted}, last status={status!r}")


def read_sse(
    client: TestClient,
    job_id: str,
    phase: str,
    stop: set[str],
    limit: int = 300,
) -> list[dict]:
    """Read one phase's event stream. The server closes it at the phase end."""
    events: list[dict] = []
    url = f"/api/jobs/{job_id}/events?phase={phase}"
    with client.stream("GET", url) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: "):])
            events.append(event)
            if event.get("type") in stop or len(events) >= limit:
                break
    return events


# -- upload / detection ---------------------------------------------------


def test_upload_accepts_pdf_and_starts_detection(client: TestClient):
    response = upload(client)
    assert response.status_code == 202
    body = response.json()
    assert body["total_pages"] == 2
    assert body["dpi"] == 150
    assert body["job_id"]

    wait_for_status(client, body["job_id"], {"awaiting_review"})
    status = client.get(f"/api/jobs/{body['job_id']}").json()
    assert status["total_nams"] == 1
    assert status["per_page_counts"] == [1, 0]


def test_rejects_non_pdf_payload(client: TestClient):
    response = client.post(
        "/api/jobs",
        files={"file": ("notes.txt", b"just some text, not a pdf", "application/pdf")},
        data=doctor_data(client),
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_rejects_pdf_content_type_mismatch(client: TestClient):
    response = client.post(
        "/api/jobs",
        files={"file": ("doc.pdf", make_pdf_bytes(), "text/plain")},
    )
    assert response.status_code == 400


def test_rejects_oversize_upload(tmp_path: Path):
    app = create_app(name_model_factory=no_ner, engine_factory=FakeEngine, dir_root=tmp_path, max_upload_bytes=512)
    with TestClient(app) as limited:
        response = upload(limited)
        assert response.status_code == 413


def test_clamps_dpi_into_supported_range(client: TestClient):
    response = upload(client, dpi=99999)
    assert response.status_code == 202
    assert response.json()["dpi"] == 400


# -- SSE ------------------------------------------------------------------


def test_sse_reports_each_page_then_awaits_review(client: TestClient):
    job_id = upload(client).json()["job_id"]
    events = read_sse(client, job_id, "detect", stop={"awaiting_review", "error"})

    types = [event["type"] for event in events]
    assert types[0] == "queued"
    assert "start" in types
    pages = [event for event in events if event["type"] == "page"]
    assert [event["page_num"] for event in pages] == [1, 2]
    assert pages[0]["nams_on_page"] == 1
    assert pages[1]["nams_on_page"] == 0

    final = events[-1]
    assert final["type"] == "awaiting_review"
    assert final["total_nams"] == 1
    assert final["per_page_counts"] == [1, 0]


def test_sse_replays_history_for_a_late_subscriber(client: TestClient):
    """Connecting after detection finished must still deliver the outcome."""
    job_id = upload(client).json()["job_id"]
    wait_for_status(client, job_id, {"awaiting_review"})

    events = read_sse(client, job_id, "detect", stop={"awaiting_review"})
    assert events[-1]["type"] == "awaiting_review"
    assert events[-1]["total_nams"] == 1


def test_sse_reports_engine_failure_without_leaking_paths(tmp_path: Path):
    app = create_app(name_model_factory=no_ner, engine_factory=BoomEngine, dir_root=tmp_path)
    with TestClient(app) as test_client:
        job_id = upload(test_client).json()["job_id"]
        events = read_sse(test_client, job_id, "detect", stop={"error"})

        error = events[-1]
        assert error["type"] == "error"
        assert "paddle exploded" not in error["message"]
        assert str(tmp_path) not in error["message"]
        assert error["message"]


def test_sse_finalize_phase_reports_done_with_result_url(client: TestClient):
    job_id = upload(client).json()["job_id"]
    wait_for_status(client, job_id, {"awaiting_review"})
    detected = client.get(f"/api/jobs/{job_id}/pages").json()["pages"]

    client.post(
        "/api/jobs/{}/finalize".format(job_id),
        json={"pages": [{"page_num": p["page_num"], "boxes": p["boxes"]} for p in detected]},
    )
    events = read_sse(client, job_id, "finalize", stop={"done", "error"})

    assert events[-1]["type"] == "done"
    assert events[-1]["total_nams"] == 1
    assert events[-1]["result_url"].endswith("/result")


def test_sse_detect_stream_excludes_finalize_events(client: TestClient):
    """Each stream filters to its own phase; they share one subscriber list."""
    job_id = upload(client).json()["job_id"]
    events = read_sse(client, job_id, "detect", stop={"awaiting_review", "error"})
    types = {event["type"] for event in events}
    assert "rendering" not in types
    assert "done" not in types


def test_sse_rejects_unknown_phase(client: TestClient):
    job_id = upload(client).json()["job_id"]
    assert client.get(f"/api/jobs/{job_id}/events?phase=nope").status_code == 400


# -- review surface -------------------------------------------------------


def test_pages_endpoint_returns_boxes_but_never_text(client: TestClient):
    job_id = upload(client).json()["job_id"]
    wait_for_status(client, job_id, {"awaiting_review"})

    response = client.get(f"/api/jobs/{job_id}/pages")
    assert response.status_code == 200
    body = response.json()
    assert body["total_pages"] == 2

    page_one = body["pages"][0]
    assert page_one["width"] > 0 and page_one["height"] > 0
    assert len(page_one["boxes"]) == 1
    box = page_one["boxes"][0]
    assert set(box) == {"x1", "y1", "x2", "y2", "confidence", "patient_id"}
    # The pseudonym is the only identity a client may see.
    assert re.fullmatch(r"Patient_\d{4,}", box["patient_id"])

    # The whole point of the invariant: geometry and a pseudonym out,
    # recognized text never.
    assert NAM_TOKEN not in json.dumps(body)
    assert body["pages"][1]["boxes"] == []


def test_pages_endpoint_conflicts_while_detection_runs(tmp_path: Path):
    """A job that failed detection has no reviewable pages."""
    app = create_app(name_model_factory=no_ner, engine_factory=BoomEngine, dir_root=tmp_path)
    with TestClient(app) as test_client:
        job_id = upload(test_client).json()["job_id"]
        wait_for_status(test_client, job_id, {"error"})
        assert test_client.get(f"/api/jobs/{job_id}/pages").status_code == 409


def test_page_image_is_a_png(client: TestClient):
    job_id = upload(client).json()["job_id"]
    wait_for_status(client, job_id, {"awaiting_review"})

    response = client.get(f"/api/jobs/{job_id}/page/1.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert response.headers["cache-control"] == "no-store"


def test_page_image_rejects_out_of_range_page(client: TestClient):
    job_id = upload(client).json()["job_id"]
    wait_for_status(client, job_id, {"awaiting_review"})
    assert client.get(f"/api/jobs/{job_id}/page/99.png").status_code == 404


# -- finalize / download --------------------------------------------------


def _finalize_and_wait(client: TestClient, job_id: str, pages: list[dict]) -> dict:
    response = client.post(f"/api/jobs/{job_id}/finalize", json={"pages": pages})
    assert response.status_code == 202
    wait_for_status(client, job_id, {"done", "error"})
    return client.get(f"/api/jobs/{job_id}").json()


def test_finalize_uses_detected_boxes_and_strips_the_text_layer(client: TestClient):
    job_id = upload(client).json()["job_id"]
    wait_for_status(client, job_id, {"awaiting_review"})
    detected = client.get(f"/api/jobs/{job_id}/pages").json()["pages"]

    status = _finalize_and_wait(
        client,
        job_id,
        [{"page_num": page["page_num"], "boxes": page["boxes"]} for page in detected],
    )
    assert status["status"] == "done"
    assert status["total_nams"] == 1
    assert status["per_page_counts"] == [1, 0]

    result = client.get(f"/api/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.headers["content-type"] == "application/pdf"
    assert result.content.startswith(b"%PDF-")
    assert "attachment" in result.headers["content-disposition"]

    with pymupdf.open(stream=result.content, filetype="pdf") as doc:
        assert len(doc) == 2
        for page in doc:
            assert page.get_text().strip() == ""


def test_finalize_honours_manual_box_edits(client: TestClient):
    """Deleting the detected box and adding one elsewhere must be respected."""
    job_id = upload(client).json()["job_id"]
    wait_for_status(client, job_id, {"awaiting_review"})

    status = _finalize_and_wait(
        client,
        job_id,
        [
            {"page_num": 1, "boxes": []},  # box removed by the user
            {"page_num": 2, "boxes": [{"x1": 0.1, "y1": 0.3, "x2": 0.6, "y2": 0.45}]},
        ],
    )
    assert status["status"] == "done"
    assert status["total_nams"] == 1
    assert status["per_page_counts"] == [0, 1]


def test_finalize_keeps_redactions_on_omitted_pages(client: TestClient):
    """A partial payload must neither drop pages nor *un-redact* them: a page
    the client left out keeps the redactions it already has."""
    job_id = upload(client).json()["job_id"]
    wait_for_status(client, job_id, {"awaiting_review"})

    status = _finalize_and_wait(client, job_id, [])  # nothing sent at all
    assert status["status"] == "done"
    assert status["total_nams"] == 1

    with pymupdf.open(stream=client.get(f"/api/jobs/{job_id}/result").content, filetype="pdf") as doc:
        assert len(doc) == 2


def test_result_conflicts_before_finalize(client: TestClient):
    job_id = upload(client).json()["job_id"]
    wait_for_status(client, job_id, {"awaiting_review"})
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 409


def test_finalize_conflicts_after_detection_error(tmp_path: Path):
    app = create_app(name_model_factory=no_ner, engine_factory=BoomEngine, dir_root=tmp_path)
    with TestClient(app) as test_client:
        job_id = upload(test_client).json()["job_id"]
        wait_for_status(test_client, job_id, {"error"})
        response = test_client.post(f"/api/jobs/{job_id}/finalize", json={"pages": []})
        assert response.status_code == 409


# -- lifecycle / security -------------------------------------------------


def test_delete_removes_the_job_directory(tmp_path: Path):
    app = create_app(name_model_factory=no_ner, engine_factory=FakeEngine, dir_root=tmp_path)
    with TestClient(app) as test_client:
        job_id = upload(test_client).json()["job_id"]
        wait_for_status(test_client, job_id, {"awaiting_review"})

        work_dirs = [p for p in (tmp_path / "documents").iterdir() if p.is_dir()]
        assert len(work_dirs) == 1
        assert (work_dirs[0] / "original.pdf").is_file()

        assert test_client.delete(f"/api/jobs/{job_id}").status_code == 204
        assert test_client.get(f"/api/jobs/{job_id}").status_code == 404
        assert not work_dirs[0].exists()


def test_unknown_job_returns_404(client: TestClient):
    assert client.get("/api/jobs/deadbeef").status_code == 404
    assert client.get("/api/jobs/deadbeef/pages").status_code == 404
    assert client.delete("/api/jobs/deadbeef").status_code == 404


def test_cross_origin_mutation_is_rejected(client: TestClient):
    response = client.post(
        "/api/jobs",
        files={"file": ("document.pdf", make_pdf_bytes(), "application/pdf")},
        headers={"Origin": "http://evil.example"},
    )
    assert response.status_code == 403


def test_same_origin_mutation_is_allowed(client: TestClient):
    response = client.post(
        "/api/jobs",
        files={"file": ("document.pdf", make_pdf_bytes(), "application/pdf")},
        data=doctor_data(client),
        headers={"Origin": "http://testserver"},
    )
    assert response.status_code == 202


def test_work_happens_in_a_temp_dir_not_the_cwd(tmp_path: Path, monkeypatch):
    """The uploaded document must never land in the working directory."""
    monkeypatch.chdir(tmp_path)
    app = create_app(name_model_factory=no_ner, engine_factory=FakeEngine, dir_root=tmp_path)
    with TestClient(app) as test_client:
        job_id = upload(test_client).json()["job_id"]
        wait_for_status(test_client, job_id, {"awaiting_review"})
        assert not list(tmp_path.glob("*.pdf"))
        assert not list(Path.cwd().glob("input.pdf"))


# -- patient IDs ----------------------------------------------------------


def _app_for(page_texts: dict[int, str], tmp_path: Path) -> TestClient:
    return create_app(
        engine_factory=lambda: FakeEngine(page_texts),
        dir_root=tmp_path,
        name_model_factory=no_ner,
    )


def test_repeated_nam_on_two_pages_shares_one_patient_id(tmp_path: Path):
    """The whole point of the ID: one patient, one pseudonym."""
    app = _app_for({1: NAM_TEXT, 2: NAM_TEXT}, tmp_path)
    with TestClient(app) as test_client:
        job_id = upload(test_client).json()["job_id"]
        wait_for_status(test_client, job_id, {"awaiting_review"})

        pages = test_client.get(f"/api/jobs/{job_id}/pages").json()["pages"]
        first = pages[0]["boxes"][0]["patient_id"]
        second = pages[1]["boxes"][0]["patient_id"]
        assert first == second
        assert re.fullmatch(r"Patient_\d{4,}", first)

        status = test_client.get(f"/api/jobs/{job_id}").json()
        assert status["total_nams"] == 2


def test_distinct_nams_get_distinct_patient_ids(tmp_path: Path):
    app = _app_for({1: NAM_TEXT, 2: "NAM: TRMJ 8802 1109"}, tmp_path)
    with TestClient(app) as test_client:
        job_id = upload(test_client).json()["job_id"]
        wait_for_status(test_client, job_id, {"awaiting_review"})

        pages = test_client.get(f"/api/jobs/{job_id}/pages").json()["pages"]
        assert pages[0]["boxes"][0]["patient_id"] != pages[1]["boxes"][0]["patient_id"]


def test_nam_endpoint_returns_a_pseudonym_without_echoing_the_input(client: TestClient):
    job_id = upload(client).json()["job_id"]
    wait_for_status(client, job_id, {"awaiting_review"})

    response = client.post(f"/api/jobs/{job_id}/nam", json={"nam": "TRMJ 8802 1109"})

    assert response.status_code == 200
    assert re.fullmatch(r"Patient_\d{4,}", response.json()["patient_id"])
    # The number must not come back, in any spelling.
    body = response.text
    for secret in ("TRMJ", "8802", "1109", "88021109"):
        assert secret not in body


def test_nam_endpoint_accepts_a_pasted_label_line(client: TestClient):
    """Retyping what is on the page should not fail on the label."""
    job_id = upload(client).json()["job_id"]
    wait_for_status(client, job_id, {"awaiting_review"})

    response = client.post(
        f"/api/jobs/{job_id}/nam", json={"nam": "NAM: TRMJ 8802 1109"}
    )
    assert response.status_code == 200


def test_nam_endpoint_reuses_an_existing_identity(client: TestClient):
    """A NAM already detected must join its existing pseudonym, not mint one."""
    job_id = upload(client).json()["job_id"]
    wait_for_status(client, job_id, {"awaiting_review"})
    detected = client.get(f"/api/jobs/{job_id}/pages").json()["pages"][0]["boxes"][0]

    response = client.post(f"/api/jobs/{job_id}/nam", json={"nam": NAM_TEXT})

    assert response.status_code == 200
    assert response.json()["patient_id"] == detected["patient_id"]


def test_nam_endpoint_rejects_invalid_input_without_echoing_it(client: TestClient):
    job_id = upload(client).json()["job_id"]
    wait_for_status(client, job_id, {"awaiting_review"})

    response = client.post(f"/api/jobs/{job_id}/nam", json={"nam": "not a number"})

    assert response.status_code == 400
    # FastAPI's default 422 handler would echo the offending input back, which
    # for this endpoint would be PHI.
    assert "not a number" not in response.text


def test_nam_endpoint_rejects_a_cross_origin_call(client: TestClient):
    job_id = upload(client).json()["job_id"]
    wait_for_status(client, job_id, {"awaiting_review"})

    response = client.post(
        f"/api/jobs/{job_id}/nam",
        json={"nam": NAM_TEXT},
        headers={"Origin": "http://evil.example"},
    )
    assert response.status_code == 403


def test_finalize_restores_ids_a_client_stripped(tmp_path: Path):
    """A client that drops the field must not get unlabelled output."""
    app = _app_for({1: NAM_TEXT, 2: "Page 2 sans numero"}, tmp_path)
    with TestClient(app) as test_client:
        job_id = upload(test_client).json()["job_id"]
        wait_for_status(test_client, job_id, {"awaiting_review"})
        boxes = test_client.get(f"/api/jobs/{job_id}/pages").json()["pages"][0]["boxes"]

        stripped = [
            {"page_num": 1, "boxes": [{k: v for k, v in b.items() if k != "patient_id"} for b in boxes]}
        ]
        assert test_client.post(
            f"/api/jobs/{job_id}/finalize", json={"pages": stripped}
        ).status_code == 202
        wait_for_status(test_client, job_id, {"done"})

        result = test_client.get(f"/api/jobs/{job_id}/result")
        with pymupdf.open(stream=result.content, filetype="pdf") as doc:
            pixmap = doc[0].get_pixmap(dpi=150)
            rendered = Image.frombytes(
                "RGB", (pixmap.width, pixmap.height), pixmap.samples
            )

        width, height = rendered.size
        box = boxes[0]
        patch = np.asarray(rendered.crop((
            int(box["x1"] * width), int(box["y1"] * height),
            int(box["x2"] * width) + 1, int(box["y2"] * height) + 1,
        )))
        assert int((patch.min(axis=2) > 200).sum()) > 0, "the label was lost"


# -- names ----------------------------------------------------------------


def test_name_matching_the_nam_is_boxed_without_a_patient_id(tmp_path: Path):
    """HOGUE Pierre matches NAM HOGP..., so it is blacked out -- unlabelled."""
    app = _app_for({1: NAM_TEXT, 2: "HOGUE Pierre"}, tmp_path)
    with TestClient(app) as test_client:
        job_id = upload(test_client).json()["job_id"]
        wait_for_status(test_client, job_id, {"awaiting_review"})

        body = test_client.get(f"/api/jobs/{job_id}/pages").json()
        assert body["total_nams"] == 1
        assert body["total_names"] == 1
        (name_box,) = body["pages"][1]["boxes"]
        assert name_box["kind"] == "name"
        assert "patient_id" not in name_box
        assert "HOGUE" not in json.dumps(body)

        pages = [{"page_num": p["page_num"], "boxes": p["boxes"]} for p in body["pages"]]
        test_client.post(f"/api/jobs/{job_id}/finalize", json={"pages": pages})
        wait_for_status(test_client, job_id, {"done"})
        status = test_client.get(f"/api/jobs/{job_id}").json()
        assert (status["total_nams"], status["total_names"]) == (1, 1)


# -- zones ----------------------------------------------------------------


def test_upload_waits_for_zones_before_detecting(client: TestClient):
    job_id = upload_only(client).json()["job_id"]
    time.sleep(0.2)
    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "awaiting_zones"
    assert status["total_pages"] == 2
    # The zone screen needs the original pages, at a thumbnail resolution.
    image = client.get(f"/api/jobs/{job_id}/page/2.png?dpi=72")
    assert image.status_code == 200
    assert client.get(f"/api/jobs/{job_id}/pages").status_code == 409


def test_each_zone_becomes_a_page_in_reading_order(tmp_path: Path):
    engine = FakeEngine({})
    app = create_app(name_model_factory=no_ner, engine_factory=lambda: engine, dir_root=tmp_path)
    with TestClient(app) as test_client:
        job_id = upload_only(test_client).json()["job_id"]
        zones = {"pages": [
            # Drawn bottom-first on purpose: output follows reading order.
            {"page_num": 1, "date": "2026-03-14", "boxes": [
                {"x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 1.0},
                {"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 0.25},
            ]},
            # Page 2 has no zone, so it is left out -- and needs no date.
            {"page_num": 2, "boxes": []},
        ]}
        assert test_client.post(f"/api/jobs/{job_id}/start", json=zones).status_code == 202
        wait_for_status(test_client, job_id, {"awaiting_review"})

        body = test_client.get(f"/api/jobs/{job_id}/pages").json()
        assert body["total_pages"] == 2
        top, bottom = body["pages"]
        # 300x200 pt page at 150 dpi = 625x417 px: the top-left quarter-height
        # zone first, then the full-width bottom half.
        assert (top["width"], top["height"]) == (312, 104)
        assert (bottom["width"], bottom["height"]) == (625, 209)  # rows 208..417
        assert engine.calls == 2

        png = test_client.get(f"/api/jobs/{job_id}/page/2.png")
        assert png.status_code == 200


def test_start_twice_is_rejected(client: TestClient):
    job_id = upload(client).json()["job_id"]
    again = client.post(f"/api/jobs/{job_id}/start", json={"pages": dated_pages(2)})
    assert again.status_code == 409


def test_zones_too_small_to_use_fail_the_job_cleanly(client: TestClient):
    job_id = upload_only(client).json()["job_id"]
    tiny = {"pages": [{
        "page_num": 1, "date": "2026-01-01",
        "boxes": [{"x1": 0.1, "y1": 0.1, "x2": 0.101, "y2": 0.101}],
    }]}
    assert client.post(f"/api/jobs/{job_id}/start", json=tiny).status_code == 202
    assert wait_for_status(client, job_id, {"error"}) == "error"


def test_start_rejects_a_cross_origin_call(client: TestClient):
    job_id = upload_only(client).json()["job_id"]
    response = client.post(
        f"/api/jobs/{job_id}/start", json={}, headers={"Origin": "http://evil.example"}
    )
    assert response.status_code == 403


# -- dates ----------------------------------------------------------------


def test_each_zone_inherits_its_source_page_date(tmp_path: Path):
    app = create_app(name_model_factory=no_ner, engine_factory=lambda: FakeEngine({}), dir_root=tmp_path)
    with TestClient(app) as test_client:
        job_id = upload_only(test_client).json()["job_id"]
        half = [
            {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 0.5},
            {"x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 1.0},
        ]
        zones = {"pages": [
            {"page_num": 1, "date": "2026-03-14", "boxes": half},
            {"page_num": 2, "date": "2026-03-15", "boxes": half[:1]},
        ]}
        assert test_client.post(f"/api/jobs/{job_id}/start", json=zones).status_code == 202
        wait_for_status(test_client, job_id, {"awaiting_review"})
        pages = test_client.get(f"/api/jobs/{job_id}/pages").json()["pages"]
        assert [p["date"] for p in pages] == ["2026-03-14", "2026-03-14", "2026-03-15"]


def test_start_requires_a_date_on_every_zoned_page(client: TestClient):
    job_id = upload_only(client).json()["job_id"]
    zones = {"pages": [{"page_num": 1, "boxes": [{"x1": 0, "y1": 0, "x2": 1, "y2": 1}]}]}
    response = client.post(f"/api/jobs/{job_id}/start", json=zones)
    assert response.status_code == 400
    assert "page 1" in response.json()["detail"]
    # Rejected before anything started: the user can fix it and retry.
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "awaiting_zones"


def test_whole_document_requires_a_date_on_every_page(client: TestClient):
    job_id = upload_only(client).json()["job_id"]
    only_first = {"pages": dated_pages(1)}
    assert client.post(f"/api/jobs/{job_id}/start", json=only_first).status_code == 400


def test_start_rejects_an_impossible_date(client: TestClient):
    job_id = upload_only(client).json()["job_id"]
    pages = dated_pages(2)
    pages[1]["date"] = "2026-02-30"
    assert client.post(f"/api/jobs/{job_id}/start", json={"pages": pages}).status_code == 400


def test_review_can_change_a_page_date(client: TestClient):
    job_id = upload(client).json()["job_id"]
    wait_for_status(client, job_id, {"awaiting_review"})
    pages = [
        {"page_num": 1, "boxes": [], "date": "2026-12-25"},
        {"page_num": 2, "boxes": []},  # omitted: keeps its zone-step date
    ]
    assert client.post(f"/api/jobs/{job_id}/finalize", json={"pages": pages}).status_code == 202
    wait_for_status(client, job_id, {"done"})
    stored = client.get(f"/api/jobs/{job_id}/pages").json()["pages"]
    assert [p["date"] for p in stored] == ["2026-12-25", "2026-09-01"]


def test_finalize_rejects_an_invalid_date_without_changing_anything(client: TestClient):
    job_id = upload(client).json()["job_id"]
    wait_for_status(client, job_id, {"awaiting_review"})
    pages = [{"page_num": 1, "boxes": [], "date": "not-a-date"}]
    assert client.post(f"/api/jobs/{job_id}/finalize", json={"pages": pages}).status_code == 400
    stored = client.get(f"/api/jobs/{job_id}/pages").json()["pages"]
    assert stored[0]["date"] == "2026-09-01"
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "awaiting_review"


# -- practice identifiers ---------------------------------------------------


def test_upload_takes_licence_group_and_sector_from_the_doctor(client: TestClient):
    body = upload_only(client).json()
    assert (body["licence_number"], body["group_number"], body["sector_number"]) == (
        "1-23456-7", "01234", "2",  # leading zero kept
    )
    status = client.get(f"/api/jobs/{body['job_id']}").json()
    assert status["group_number"] == "01234"
    assert status["doctor"]["snapshot"]["display_name"] == "Dre Test Synthétique"


def test_identifiers_typed_in_the_browser_are_refused_before_storing(tmp_path: Path):
    app = create_app(name_model_factory=no_ner, engine_factory=FakeEngine, dir_root=tmp_path)
    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/jobs",
            files={"file": ("document.pdf", make_pdf_bytes(), "application/pdf")},
            data={**doctor_data(test_client), "group_number": "99999"},
        )
        assert response.status_code == 400
        assert "doctor profile" in response.json()["detail"]
        documents = tmp_path / "documents"
        assert not documents.exists() or not any(documents.iterdir())  # nothing stored
