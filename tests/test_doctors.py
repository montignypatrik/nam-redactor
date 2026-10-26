"""Doctor profiles, their versions, and the doctor assigned to each document.

All data is synthetic. Extraction uses the fake extractor; nothing leaves.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nam_redactor.web.app import create_app
from tests.test_documents import manifest, save_review
from tests.test_extraction import FakeExtractor, VALID, accept, make_client, prepared, wait_settled, extract
from tests.test_web_api import NAM_TEXT, FakeEngine, make_pdf_bytes, no_ner

DR_A = {
    "display_name": "Dre Alpha Synthétique",
    "licence_number": "1-00001-2",
    "group_number": "00123",
    "sector_number": "3",
    "instructions": "Abréviations:\n- sut = suture\n\nExclusions: ne pas facturer les suivis téléphoniques.",
    "examples": [
        {"annotation": "sut 3pts", "interpretation": "Suture, 3 points", "note": "pts = points"},
        {"annotation": "", "interpretation": "", "note": ""},  # blank editor row: dropped
    ],
}
DR_B = {"display_name": "Dr Bêta Synthétique", "licence_number": "2-00002-3",
        "group_number": "00456", "sector_number": ""}


def app_client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(engine_factory=lambda: FakeEngine({1: NAM_TEXT, 2: NAM_TEXT}),
                                 dir_root=tmp_path, name_model_factory=no_ner))


def create(client: TestClient, body: dict) -> dict:
    response = client.post("/api/doctors", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def upload_for(client: TestClient, profile_id: str, version: int | None = None):
    data = {"doctor_profile_id": profile_id}
    if version is not None:
        data["doctor_profile_version"] = str(version)
    return client.post(
        "/api/jobs",
        files={"file": ("document.pdf", make_pdf_bytes(), "application/pdf")},
        data=data,
    )


def record(tmp_path: Path, job_id: str) -> dict:
    return json.loads((tmp_path / "documents" / job_id / "document.json").read_text(encoding="utf-8"))


# -- profiles ---------------------------------------------------------------------


def test_profiles_persist_across_a_restart(tmp_path: Path):
    with app_client(tmp_path) as client:
        created = create(client, DR_A)
    assert (tmp_path / "doctors.sqlite3").is_file()
    with app_client(tmp_path) as client:
        profile = client.get(f"/api/doctors/{created['profile_id']}").json()
        assert profile["display_name"] == "Dre Alpha Synthétique" and profile["status"] == "active"
        current = profile["current"]
        assert current["group_number"] == "00123"  # a string: leading zeros kept
        assert current["instructions"].startswith("Abréviations:\n- sut = suture")
        assert current["examples"] == [
            {"annotation": "sut 3pts", "interpretation": "Suture, 3 points", "note": "pts = points"}]


def test_editing_adds_a_version_and_keeps_the_previous_one(tmp_path: Path):
    with app_client(tmp_path) as client:
        created = create(client, DR_A)
        pid = created["profile_id"]
        edited = client.put(f"/api/doctors/{pid}", json={
            **DR_A, "group_number": "00999", "instructions": "Nouvelle consigne", "base_version": 1,
        }).json()
        assert edited["current_version"] == 2 and not edited["unchanged"]
        assert [v["version"] for v in edited["versions"]] == [2, 1]

        v1 = client.get(f"/api/doctors/{pid}/versions/1").json()
        assert v1["group_number"] == "00123" and v1["instructions"].startswith("Abréviations")
        assert client.get(f"/api/doctors/{pid}/versions/2").json()["group_number"] == "00999"

        # Saving the same content again adds nothing.
        same = client.put(f"/api/doctors/{pid}", json={**DR_A, "group_number": "00999",
                                                          "instructions": "Nouvelle consigne"}).json()
        assert same["unchanged"] and same["current_version"] == 2
        # A stale editor cannot overwrite a newer version.
        stale = client.put(f"/api/doctors/{pid}", json={**DR_A, "base_version": 1})
        assert stale.status_code == 409

    # Versions are immutable in the database itself.
    with sqlite3.connect(tmp_path / "doctors.sqlite3") as conn:
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            conn.execute("UPDATE profile_versions SET group_number = 'x'")
        with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
            conn.execute("DELETE FROM profile_versions")


def test_duplicating_creates_an_independent_profile(tmp_path: Path):
    with app_client(tmp_path) as client:
        original = create(client, DR_A)
        copy = client.post(f"/api/doctors/{original['profile_id']}/duplicate", json={}).json()
        assert copy["profile_id"] != original["profile_id"]
        assert copy["display_name"] == "Dre Alpha Synthétique (copy)"
        assert copy["current_version"] == 1
        assert copy["current"]["examples"] == original["current"]["examples"]

        client.put(f"/api/doctors/{copy['profile_id']}", json={**DR_A, "display_name": "Copie", "group_number": "11111"})
        again = client.get(f"/api/doctors/{original['profile_id']}").json()
        assert again["current_version"] == 1 and again["current"]["group_number"] == "00123"


def test_content_validation(tmp_path: Path):
    with app_client(tmp_path) as client:
        assert client.post("/api/doctors", json={"display_name": "  "}).status_code == 400
        assert client.post("/api/doctors", json={**DR_B, "licence_number": "<script>"}).status_code == 400
        half = {**DR_B, "examples": [{"annotation": "sut", "interpretation": ""}]}
        assert client.post("/api/doctors", json=half).status_code == 400
        # A name alone is a usable profile: instructions and examples are optional.
        minimal = create(client, {"display_name": "Dr Minimal"})
        assert minimal["current"]["instructions"] == "" and minimal["current"]["examples"] == []
        assert upload_for(client, minimal["profile_id"]).status_code == 202


# -- assignment -------------------------------------------------------------------


def test_upload_saves_the_exact_version_and_a_snapshot(tmp_path: Path):
    with app_client(tmp_path) as client:
        profile = create(client, DR_A)
        body = upload_for(client, profile["profile_id"], version=1).json()
        assert (body["licence_number"], body["group_number"], body["sector_number"]) == ("1-00001-2", "00123", "3")
        saved = record(tmp_path, body["job_id"])
        assert saved["doctor"]["profile_id"] == profile["profile_id"]
        assert saved["doctor"]["version"] == 1
        assert saved["doctor"]["snapshot"] == {
            "display_name": "Dre Alpha Synthétique",
            "licence_number": "1-00001-2", "group_number": "00123", "sector_number": "3",
        }
        assert saved["practice"] == {"licence_number": "1-00001-2", "group_number": "00123", "sector_number": "3"}
        assert saved["doctor"]["assignment_id"].startswith("asg_")


def test_invalid_profile_references_are_refused_and_nothing_is_stored(tmp_path: Path):
    with app_client(tmp_path) as client:
        profile = create(client, DR_A)
        pid = profile["profile_id"]
        client.put(f"/api/doctors/{pid}", json={**DR_A, "group_number": "00777"})  # now version 2
        cases = [
            ({}, 400),                                            # no profile
            ({"doctor_profile_id": "doc_000000000000"}, 404),     # unknown
            ({"doctor_profile_id": pid, "doctor_profile_version": "1"}, 409),   # outdated
            ({"doctor_profile_id": pid, "doctor_profile_version": "v2"}, 400),  # malformed
        ]
        for data, status in cases:
            response = client.post("/api/jobs", data=data,
                                   files={"file": ("d.pdf", make_pdf_bytes(), "application/pdf")})
            assert response.status_code == status, (data, response.text)
        assert client.get("/api/jobs").json()["documents"] == []


def test_archived_profiles_cannot_be_assigned_but_existing_documents_still_load(tmp_path: Path):
    with app_client(tmp_path) as client:
        profile = create(client, DR_A)
        pid = profile["profile_id"]
        job_id = upload_for(client, pid).json()["job_id"]
        assert client.post(f"/api/doctors/{pid}/archive").json()["status"] == "archived"
        assert client.get("/api/doctors").json()["profiles"] == []  # hidden from selection
        assert len(client.get("/api/doctors?include_archived=true").json()["profiles"]) == 1

        assert upload_for(client, pid).status_code == 409
        other = upload_for(client, create(client, DR_B)["profile_id"]).json()["job_id"]
        assert client.put(f"/api/jobs/{other}/doctor", json={"profile_id": pid}).status_code == 409
        assert client.put(f"/api/doctors/{pid}", json=DR_A).status_code == 409  # no edits while archived

    with app_client(tmp_path) as client:
        status = client.get(f"/api/jobs/{job_id}").json()
        assert status["doctor"]["profile_status"] == "archived"
        assert status["doctor"]["snapshot"]["display_name"] == "Dre Alpha Synthétique"
        assert status["group_number"] == "00123"
        assert client.get(f"/api/doctors/{pid}/versions/1").status_code == 200
        # Restoring makes it assignable again.
        client.post(f"/api/doctors/{pid}/restore")
        assert upload_for(client, pid).status_code == 202


def test_editing_a_profile_leaves_existing_documents_unchanged(tmp_path: Path):
    with app_client(tmp_path) as client:
        pid = create(client, DR_A)["profile_id"]
        job_id = upload_for(client, pid).json()["job_id"]
        client.put(f"/api/doctors/{pid}", json={**DR_A, "display_name": "Dre Alpha (renommée)",
                                                "group_number": "00999"})
        status = client.get(f"/api/jobs/{job_id}").json()
        assert status["doctor"]["version"] == 1
        assert status["doctor"]["snapshot"]["display_name"] == "Dre Alpha Synthétique"
        assert status["group_number"] == "00123"
        assert status["doctor"]["newer_version_available"] is True
        assert status["doctor"]["current_version"] == 2

        # Adopting the newer version is explicit.
        before = status["doctor"]["assignment_id"]
        adopted = client.put(f"/api/jobs/{job_id}/doctor", json={"profile_id": pid, "version": 2}).json()
        assert adopted["changed"] and adopted["doctor"]["version"] == 2
        assert adopted["doctor"]["assignment_id"] != before
        assert adopted["group_number"] == "00999"
        assert not adopted["doctor"]["newer_version_available"]
        # Re-assigning the same version is a no-op, not a new assignment.
        same = client.put(f"/api/jobs/{job_id}/doctor", json={"profile_id": pid}).json()
        assert not same["changed"] and same["doctor"]["assignment_id"] == adopted["doctor"]["assignment_id"]


def test_reassigning_the_doctor_updates_identity_fields_consistently(tmp_path: Path):
    with app_client(tmp_path) as client:
        a = create(client, DR_A)["profile_id"]
        b = create(client, DR_B)["profile_id"]
        job_id = upload_for(client, a).json()["job_id"]
        first = client.get(f"/api/jobs/{job_id}").json()["doctor"]
        result = client.put(f"/api/jobs/{job_id}/doctor", json={"profile_id": b}).json()
        assert result["changed"]

    with app_client(tmp_path) as client:  # and it survives a restart
        status = client.get(f"/api/jobs/{job_id}").json()
        assert status["doctor"]["profile_id"] == b
        assert status["doctor"]["snapshot"] == {"display_name": "Dr Bêta Synthétique",
                                                "licence_number": "2-00002-3", "group_number": "00456",
                                                "sector_number": ""}
        assert (status["licence_number"], status["group_number"], status["sector_number"]) == ("2-00002-3", "00456", "")
        saved = record(tmp_path, job_id)
        assert [h["assignment_id"] for h in saved["doctor_history"]] == [first["assignment_id"]]


def test_a_doctor_change_keeps_redactions_rows_and_transcription_reviews(tmp_path: Path):
    fake = FakeExtractor(VALID)
    with make_client(tmp_path, fake) as client:
        job_id, rows = prepared(client, pages=(1,))
        extract(client, job_id, rows[1])
        reviewed = accept(client, job_id, wait_settled(client, job_id)[1],
                          handwriting_text="Rx: amox 500 TID x 7 j").json()
        before_rows = client.get(f"/api/jobs/{job_id}/rows").json()["rows"]
        before_pages = client.get(f"/api/jobs/{job_id}/pages").json()["pages"]
        before_export = manifest(client, job_id)

        other = create(client, DR_B)["profile_id"]
        assert client.put(f"/api/jobs/{job_id}/doctor", json={"profile_id": other}).json()["changed"]

        after_rows = client.get(f"/api/jobs/{job_id}/rows").json()["rows"]
        # The billing lines' doctor columns follow the new assignment...
        assert {r["billing"]["shared"]["doctorGroupNumber"] for r in after_rows} == {"00456"}
        # ...and everything else -- approvals, extraction, reviews -- is identical.
        strip = lambda rows: [{k: v for k, v in r.items() if k != "billing"} for r in rows]  # noqa: E731
        assert strip(after_rows) == strip(before_rows)
        assert client.get(f"/api/jobs/{job_id}/pages").json()["pages"] == before_pages
        after_export = manifest(client, job_id)
        assert after_export["rows"] == before_export["rows"]
        assert after_export["document"]["practice"]["group_number"] == "00456"
        assert after_export["document"]["doctor"]["profile_id"] == other
        view = {r["row_id"]: r for r in after_rows}[reviewed["row_id"]]["extraction"]
        assert view["state"] == "reviewed" and view["review"]["handwriting_text"] == "Rx: amox 500 TID x 7 j"


# -- legacy documents -------------------------------------------------------------


def _make_legacy(tmp_path: Path) -> str:
    """A document saved before doctor profiles: identifiers, no profile."""
    with app_client(tmp_path) as client:
        pid = create(client, DR_A)["profile_id"]
        job_id = upload_for(client, pid).json()["job_id"]
    path = tmp_path / "documents" / job_id / "document.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved.pop("doctor"), saved.pop("doctor_history")
    saved["practice"] = {"licence_number": "9-99999-9", "group_number": "00042", "sector_number": "1"}
    path.write_text(json.dumps(saved), encoding="utf-8")
    return job_id


def test_legacy_documents_stay_readable_and_keep_their_identifiers(tmp_path: Path):
    job_id = _make_legacy(tmp_path)
    with app_client(tmp_path) as client:
        status = client.get(f"/api/jobs/{job_id}").json()
        assert status["doctor"] is None  # never guessed from the identifiers
        assert (status["licence_number"], status["group_number"], status["sector_number"]) == ("9-99999-9", "00042", "1")
        assert len(client.get("/api/doctors").json()["profiles"]) == 1  # nothing created
    # Loading did not rewrite the record.
    assert "doctor" not in record(tmp_path, job_id)

    with app_client(tmp_path) as client:
        client.post(f"/api/jobs/{job_id}/start", json={"pages": [
            {"page_num": 1, "boxes": [], "date": "2026-03-14"},
            {"page_num": 2, "boxes": [], "date": "2026-03-14"}]})
        from tests.test_web_api import wait_for_status
        wait_for_status(client, job_id, {"awaiting_review"})
        save_review(client, job_id)
        export = manifest(client, job_id)
        assert export["document"]["practice"] == {"licence_number": "9-99999-9",
                                                  "group_number": "00042", "sector_number": "1"}
        assert export["document"]["doctor"] is None

        # Assigning one explicitly keeps the old identifiers in the history.
        pid = client.get("/api/doctors").json()["profiles"][0]["profile_id"]
        client.put(f"/api/jobs/{job_id}/doctor", json={"profile_id": pid})
        saved = record(tmp_path, job_id)
        assert saved["practice"]["group_number"] == "00123"
        (legacy,) = saved["doctor_history"]
        assert legacy["legacy_practice"] and legacy["snapshot"]["group_number"] == "00042"


def test_exports_keep_the_practice_fields_and_add_the_doctor(tmp_path: Path):
    with app_client(tmp_path) as client:
        pid = create(client, DR_A)["profile_id"]
        job_id = upload_for(client, pid).json()["job_id"]
        client.post(f"/api/jobs/{job_id}/start", json={"pages": [
            {"page_num": 1, "boxes": [], "date": "2026-03-14"},
            {"page_num": 2, "boxes": [], "date": "2026-03-14"}]})
        from tests.test_web_api import wait_for_status
        wait_for_status(client, job_id, {"awaiting_review"})
        save_review(client, job_id)
        document = manifest(client, job_id)["document"]
        assert document["practice"] == {"licence_number": "1-00001-2", "group_number": "00123", "sector_number": "3"}
        assert document["doctor"]["profile_id"] == pid and document["doctor"]["version"] == 1
        assert document["doctor"]["display_name"] == "Dre Alpha Synthétique"
        # Instructions and examples are for interpretation, not the export.
        assert "Abréviations" not in json.dumps(document)
