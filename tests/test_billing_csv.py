"""The billing CSV export: format, one line per act, sources, and blocking.

Synthetic documents and a fake model. The export itself never calls a model.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient

from nam_redactor.web.billing_csv import HEADER, UNUSED
from tests.test_extraction import accept, make_client, rows_of
from tests.test_interpretation import DOCTOR, TWO_ACTS, FakeModel, run_all, settle, setup
from tests.test_interpretation import ONE_ROW

HEADER_LINE = (
    "doctorLicenceID;doctorGroupNumber;ClaimDate;ProcedureTime;HIN;PatientGender;PatientFirstName;"
    "PatientLastName;PatientBirthdate;PatientParentHIN;Sector;Diagnosis;ProcedureCode;Units;"
    "ContextualElements;referringDoctor;CSSTAccidentDate"
)
NAM = "HOGP33610615"  # the synthetic document's NAM ("HOGP 3361 0615"), as the HIN

ROW_A = {  # vs with two contexts, and 15824: both exact strings
    "acts": [
        {"code": "vs", "units": None, "context_elements": ["TT", "XY"], "source_text": "suivi tel",
         "profile_reference": "tel = TT"},
        {"code": "15824", "units": None, "context_elements": [], "source_text": "VS CV1",
         "profile_reference": "VS CV"},
    ],
    "unresolved": [], "conflicts": ["an informational warning"], "ignored_notes": [],
    "no_billing_notation": False, "remarks": None,
}
ROW_B = {  # two acts, one with units
    "acts": [
        {"code": "vpm", "units": None, "context_elements": [], "source_text": "VPS", "profile_reference": None},
        {"code": "431", "units": 1, "context_elements": [], "source_text": "infiltration genou",
         "profile_reference": None},
    ],
    "unresolved": [], "conflicts": [], "ignored_notes": [], "no_billing_notation": False, "remarks": None,
}
NO_BILLING = {"acts": [], "unresolved": [], "conflicts": [], "ignored_notes": [],
              "no_billing_notation": True, "remarks": None}
UNRESOLVED = {**ROW_B, "acts": [ROW_B["acts"][0], {**ROW_B["acts"][1], "code": None}]}


def client_for(tmp_path: Path, *proposals, texts=None) -> tuple[TestClient, FakeModel]:
    """One worker, so proposals go to rows in order (row 1 first)."""
    fake = FakeModel(interpretation=proposals or (ROW_A, ROW_B))
    return make_client(tmp_path, fake, texts=texts, concurrency=1), fake


def processed(client: TestClient, pages=(1, 2)) -> str:
    job_id, _ = setup(client, pages=pages)
    run_all(client, job_id)
    settle(client, job_id)
    return job_id


def download(client: TestClient, job_id: str):
    return client.get(f"/api/jobs/{job_id}/billing.csv")


def parse(content: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""), delimiter=";"))


def test_the_file_matches_the_practice_format_exactly(tmp_path: Path):
    client, _ = client_for(tmp_path)
    with client:
        job_id = processed(client)
        response = download(client, job_id)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/csv")
        assert 'filename="billing-' in response.headers["content-disposition"]
        raw = response.content

    assert not raw.startswith(b"\xef\xbb\xbf")                     # UTF-8, no BOM
    raw.decode("utf-8")
    lines = raw.split(b"\r\n")
    assert lines[0].decode() == HEADER_LINE and ";".join(HEADER) == HEADER_LINE
    assert b"\n" not in raw.replace(b"\r\n", b"")                   # CRLF only
    assert not raw.endswith(b"\r\n")                                # like the example
    assert len(lines) == 1 + 4 and all(line.count(b";") == 16 for line in lines)
    for token in (b"null", b"None", b"NaN"):
        assert token not in raw


def test_one_line_per_act_with_exact_codes_units_and_contexts(tmp_path: Path):
    client, fake = client_for(tmp_path)
    with client:
        job_id = processed(client)
        calls = (len(fake.calls), len(fake.i_calls))
        lines = parse(download(client, job_id).content)
        assert (len(fake.calls), len(fake.i_calls)) == calls     # the export called no model

    assert [l["ProcedureCode"] for l in lines] == ["vs", "15824", "vpm", "431"]
    assert [l["Units"] for l in lines] == ["", "", "", "1"]
    assert [l["ContextualElements"] for l in lines] == ["TT,XY", "", "", ""]
    # Appointment information repeated on each of its lines.
    assert [l["ClaimDate"] for l in lines] == ["2026-03-14", "2026-03-14", "2026-03-15", "2026-03-15"]
    for line in lines:
        assert line["HIN"] == NAM                                  # pseudonym resolved locally
        assert (line["doctorLicenceID"], line["doctorGroupNumber"], line["Sector"]) == ("1-00001-2", "00123", "3")
        assert line["ProcedureTime"] == "09:30"                    # model reading, not accepted
        assert all(line[column] == "" for column in UNUSED)


def test_corrected_values_are_used_including_an_empty_time(tmp_path: Path):
    client, _ = client_for(tmp_path, ROW_A, ROW_A)
    with client:
        job_id = processed(client)
        accept(client, job_id, rows_of(client, job_id)[1], time=None)   # row 1: the reading was wrong
        run_all(client, job_id)                                          # re-interprets from the correction
        settle(client, job_id)
        lines = parse(download(client, job_id).content)
        assert [l["ProcedureTime"] for l in lines] == ["", "", "09:30", "09:30"]
        accept(client, job_id, rows_of(client, job_id)[1], time="10:45")
        run_all(client, job_id)
        settle(client, job_id)
        lines = parse(download(client, job_id).content)
        assert [l["ProcedureTime"] for l in lines] == ["10:45", "10:45", "09:30", "09:30"]


def test_a_row_that_cannot_be_prepared_is_reported_not_skipped(tmp_path: Path):
    client, _ = client_for(tmp_path, ROW_A, texts=ONE_ROW)   # page 2: no NAM, so no patient
    with client:
        job_id = processed(client, pages=(1,))
        body = download(client, job_id).json()
        assert [(p["label"], p["reason"]) for p in body["problems"]] == [
            ("#2", "not transcribed yet (or the row changed since)")]


def test_doctor_identifiers_come_from_the_assigned_snapshot(tmp_path: Path):
    client, _ = client_for(tmp_path, ROW_A, ROW_A)
    with client:
        job_id = processed(client)
        pid = client.get(f"/api/jobs/{job_id}").json()["doctor"]["profile_id"]
        client.put(f"/api/doctors/{pid}", json={**DOCTOR, "group_number": "99999"})  # not adopted
        lines = parse(download(client, job_id).content)
        assert {l["doctorGroupNumber"] for l in lines} == {"00123"}


def test_no_billing_notation_rows_produce_no_lines(tmp_path: Path):
    client, _ = client_for(tmp_path, ROW_A, NO_BILLING)
    with client:
        job_id = processed(client)
        response = download(client, job_id)
        assert response.status_code == 200
        assert response.headers["x-billing-no-notation-rows"] == "1"
        assert [l["ProcedureCode"] for l in parse(response.content)] == ["vs", "15824"]


def test_unresolved_codes_block_the_export_and_name_the_row(tmp_path: Path):
    client, _ = client_for(tmp_path, ROW_A, UNRESOLVED)
    with client:
        job_id = processed(client)
        response = download(client, job_id)
        assert response.status_code == 409
        body = response.json()
        assert body["problems"] == [{"row_id": rows_of(client, job_id)[2]["row_id"], "label": "#2",
                                     "reason": "unresolved procedure code (act 2)"}]
        assert "#2 unresolved procedure code" in body["detail"]


def test_outdated_or_missing_proposals_block_the_export(tmp_path: Path):
    client, _ = client_for(tmp_path, ROW_A, ROW_B)
    with client:
        job_id = processed(client)
        pid = client.get(f"/api/jobs/{job_id}").json()["doctor"]["profile_id"]
        client.put(f"/api/doctors/{pid}", json={**DOCTOR, "instructions": "v2"})
        client.put(f"/api/jobs/{job_id}/doctor", json={"profile_id": pid, "version": 2})
        body = download(client, job_id).json()
        assert {p["label"]: p["reason"] for p in body["problems"]} == {
            "#1": "the proposal is outdated", "#2": "the proposal is outdated"}

    client, _ = client_for(tmp_path / "b")
    with client:
        job_id, _ = setup(client)  # prepared, never extracted
        reasons = {p["reason"] for p in download(client, job_id).json()["problems"]}
        assert reasons == {"not transcribed yet (or the row changed since)"}


def test_a_missing_health_number_blocks_instead_of_inventing_one(tmp_path: Path):
    client, _ = client_for(tmp_path, ROW_A, ROW_A)
    with client:
        job_id = processed(client)
        job = client.app.state.store.get(job_id)
        job.patient_ids.clear()                          # the mapping lost this patient
        body = download(client, job_id).json()
        assert body["problems"][0]["reason"] == "no health number on record for Patient_0001"


def test_a_telephone_call_15622_needs_no_health_number(tmp_path: Path):
    call = {**NO_BILLING, "no_billing_notation": False, "acts": [
        {"code": "15622", "units": None, "context_elements": [], "source_text": "appel tel",
         "profile_reference": None}]}
    client, _ = client_for(tmp_path, call, call)
    with client:
        job_id = processed(client)
        client.app.state.store.get(job_id).patient_ids.clear()
        response = download(client, job_id)
        assert response.status_code == 200
        assert [(line["ProcedureCode"], line["HIN"]) for line in parse(response.content)] == [("15622", "")] * 2

    mixed = {**call, "acts": call["acts"] + ROW_B["acts"][:1]}
    client, _ = client_for(tmp_path / "b", mixed, mixed)
    with client:
        job_id = processed(client)
        client.app.state.store.get(job_id).patient_ids.clear()
        body = download(client, job_id).json()
        assert body["problems"][0]["reason"] == "no health number on record for Patient_0001"


def test_exporting_changes_nothing_and_approves_nothing(tmp_path: Path):
    client, fake = client_for(tmp_path)
    with client:
        job_id = processed(client)
        folder = tmp_path / "documents" / job_id
        client.app.state.store._extract_executor.shutdown(wait=True)
        before = (folder / "document.json").read_bytes()
        views_before = rows_of(client, job_id)
        for _ in range(2):
            assert download(client, job_id).status_code == 200
        assert (folder / "document.json").read_bytes() == before
        views = rows_of(client, job_id)
        assert views == views_before
        for view in views.values():
            assert view["extraction"]["review"] is None
            assert view["interpretation"]["validated"] is False
            assert view["interpretation"]["attempt"]["validated"] is False
        saved = json.loads(before)
        assert all(not r["reviews"] for r in saved["extractions"].values())
        # The JSON/evidence export is still there.
        assert client.get(f"/api/jobs/{job_id}/manifest.json").status_code == 200
