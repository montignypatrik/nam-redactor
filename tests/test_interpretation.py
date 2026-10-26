"""One-click transcription -> interpretation with the assigned doctor profile.

Synthetic documents and a scripted fake model: nothing leaves the machine.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nam_redactor import extraction as ex
from nam_redactor.web import jobs as jobslib
from tests.privacy import assert_no_identifiers
from tests.test_documents import manifest, save_review
from tests.test_web_api import NAM_TEXT
from tests.test_extraction import (
    NEW_BOX,
    VALID,
    FakeExtractor,
    Transport,
    accept,
    add_redaction,
    completion,
    make_client,
    prepared,
    rows_of,
)

#: A synthetic doctor's conventions (not any real doctor's).
INSTRUCTIONS = "Abréviations:\n- VX = code 11111\n- tel = contexte TT\nUnités: inconnues sauf si écrites."
EXAMPLES = [{"annotation": "VX tel", "interpretation": "11111, contexte TT", "note": "exemple synthétique"}]
DOCTOR = {"display_name": "Dre Synthèse", "licence_number": "1-00001-2", "group_number": "00123",
          "sector_number": "3", "instructions": INSTRUCTIONS, "examples": EXAMPLES}

TWO_ACTS = {
    "acts": [
        {"code": "11111", "units": None, "context_elements": ["TT"], "source_text": "VX tel",
         "profile_reference": "Abréviations: VX; tel = TT"},
        {"code": "22222", "units": 2, "context_elements": [], "source_text": "YZ x2",
         "profile_reference": "Exemple 1"},
    ],
    "unresolved": [],
    "conflicts": [],
    "ignored_notes": [{"text": "amox 500", "reason": "prescription, not billing"}],
    "no_billing_notation": False,
    "remarks": None,
}


class FakeModel(FakeExtractor):
    """Scripted transcription *and* interpretation, recording both."""

    def __init__(self, transcription=(VALID,), interpretation=(TWO_ACTS,)):
        super().__init__(*transcription)
        self.i_script = list(interpretation)
        self.i_calls: list[tuple[bytes, ex.InterpretationRequest, str]] = []
        self.i_gate: threading.Event | None = None
        self.i_entered = threading.Event()

    def interpret(self, image_png, request, model):
        with self._lock:
            self.i_calls.append((image_png, request, model))
            item = self.i_script.pop(0) if len(self.i_script) > 1 else self.i_script[0]
        self.i_entered.set()
        if self.i_gate is not None:
            assert self.i_gate.wait(10)
        if isinstance(item, BaseException):
            raise item
        return ex.ExtractionOutput(
            result=ex.validate_interpretation(item), returned_model=model, provider="FakeProvider",
            generation_id="gen-i", usage={"prompt_tokens": 50, "completion_tokens": 20,
                                          "total_tokens": 70, "cost": 0.001, "cost_currency": "credits"},
        )


#: Page 2 has no NAM, so only page 1's row can be prepared: single-row tests.
ONE_ROW = {1: NAM_TEXT, 2: ""}


def assign(client: TestClient, job_id: str, body: dict = DOCTOR) -> str:
    profile = client.post("/api/doctors", json=body).json()
    response = client.put(f"/api/jobs/{job_id}/doctor", json={"profile_id": profile["profile_id"]})
    assert response.status_code == 200, response.text
    return profile["profile_id"]


def setup(client: TestClient, doctor: dict | None = DOCTOR, pages=(1, 2)) -> tuple[str, dict]:
    """A saved-review document with the synthetic doctor assigned."""
    job_id, rows = prepared(client, pages=pages)
    if doctor is not None:
        assign(client, job_id, doctor)
    return job_id, rows


def settle(client: TestClient, job_id: str, timeout: float = 10.0) -> dict[int, dict]:
    """Wait until neither stage is running on any row."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = rows_of(client, job_id)
        if all(r["extraction"]["state"] != "running" and r["interpretation"]["state"] != "running"
               for r in rows.values()):
            return rows
        time.sleep(0.02)
    raise AssertionError("did not settle")


def run_all(client: TestClient, job_id: str) -> dict:
    response = client.post(f"/api/jobs/{job_id}/extract", json={})
    assert response.status_code == 202, response.text
    return response.json()


# -- adapter ----------------------------------------------------------------------


def test_the_interpretation_request_carries_instructions_examples_and_image(monkeypatch):
    monkeypatch.setenv(ex.API_KEY_ENV, "sk-or-v1-TEST")
    answer = json.loads(completion(TWO_ACTS))
    transport = Transport((200, json.dumps(answer).encode()))
    request = ex.InterpretationRequest(INSTRUCTIONS, tuple(EXAMPLES), "reviewer", None, "legible", "VX tel\nYZ x2")
    out = ex.OpenRouterExtractor(transport).interpret(b"PNG-BYTES", request, ex.DEFAULT_MODEL)

    body = transport.calls[0]["body"]
    text, image = body["messages"][1]["content"]
    assert INSTRUCTIONS in text["text"]
    assert "Annotation as written: VX tel" in text["text"] and "Expected interpretation: 11111, contexte TT" in text["text"]
    assert "reviewer-corrected transcription" in text["text"]
    assert "Appointment time: unknown" in text["text"]  # a null corrected time stays unknown
    assert "VX tel\nYZ x2" in text["text"]
    assert image["image_url"]["url"].endswith("UE5HLUJZVEVT")  # base64 of the same image
    assert body["provider"]["zdr"] is True and body["provider"]["require_parameters"] is True
    assert body["response_format"]["json_schema"]["name"] == "row_interpretation"
    for identifier in ("1-00001-2", "00123", "licence"):
        assert identifier not in json.dumps(body)   # identifiers are not sent
    assert out.result["acts"][1]["units"] == 2


@pytest.mark.parametrize("bad", [
    {**TWO_ACTS, "acts": [{**TWO_ACTS["acts"][0], "units": "2"}]},        # units not a number
    {**TWO_ACTS, "acts": [{**TWO_ACTS["acts"][0], "units": True}]},       # boolean units
    {**TWO_ACTS, "acts": [{**TWO_ACTS["acts"][0], "code": 11111}]},       # code not a string
    {**TWO_ACTS, "no_billing_notation": True},                            # none, yet acts
    {**TWO_ACTS, "approved": True},                                       # no such field
    {**TWO_ACTS, "acts": [{**TWO_ACTS["acts"][0], "source_text": " "}]},  # no source
])
def test_malformed_interpretations_are_rejected(bad):
    with pytest.raises(ex.InvalidOutput):
        ex.validate_interpretation(bad)


# -- one click --------------------------------------------------------------------


def test_one_click_transcribes_then_interprets_with_no_acceptance(tmp_path: Path):
    fake = FakeModel()
    with make_client(tmp_path, fake) as client:
        job_id, rows = setup(client)
        batch = run_all(client, job_id)
        assert sorted(batch["transcribing"]) == sorted(r["row_id"] for r in rows.values())
        assert batch["interpreting"] == [] and batch["interpretation_blocked"] is None
        settled = settle(client, job_id)

        assert len(fake.calls) == 2 and len(fake.i_calls) == 2
        for row in settled.values():
            assert row["extraction"]["review"] is None and row["extraction"]["review_count"] == 0
            view = row["interpretation"]
            assert view["state"] == "proposed" and view["is_current"]
            assert view["validated"] is False and view["attempt"]["validated"] is False
            first, second = view["attempt"]["result"]["acts"]
            # Each act keeps its own units and contexts.
            assert (first["code"], first["units"], first["context_elements"]) == ("11111", None, ["TT"])
            assert (second["code"], second["units"], second["context_elements"]) == ("22222", 2, [])

        # Repeating the click does nothing more: everything is current.
        again = run_all(client, job_id)
        assert again["transcribing"] == [] and again["interpreting"] == []
        assert set(again["skipped"].values()) == {"already interpreted"}


def test_the_assigned_profile_version_reaches_the_model_not_the_latest(tmp_path: Path):
    fake = FakeModel()
    with make_client(tmp_path, fake, texts=ONE_ROW) as client:
        job_id, rows = setup(client, pages=(1,))
        pid = client.get(f"/api/jobs/{job_id}").json()["doctor"]["profile_id"]
        client.put(f"/api/doctors/{pid}", json={**DOCTOR, "instructions": "VERSION DEUX"})
        run_all(client, job_id)
        view = settle(client, job_id)[1]["interpretation"]

        ((image, request, _),) = fake.i_calls
        assert request.instructions == INSTRUCTIONS and "VERSION DEUX" not in request.instructions
        assert request.examples == tuple(EXAMPLES)
        assert request.transcription_source == "model"
        assert request.handwriting_text == VALID["handwriting_text"] and request.time == VALID["time"]
        assert hashlib.sha256(image).hexdigest() == rows[1]["evidence"][0]["sha256"]  # the saved crop
        assert view["attempt"]["inputs"]["doctor"]["version"] == 1

        # Adopting v2 outdates the proposal; rerunning uses v2 -- no new transcription.
        client.put(f"/api/jobs/{job_id}/doctor", json={"profile_id": pid, "version": 2})
        stale = rows_of(client, job_id)[1]["interpretation"]
        assert stale["state"] == "stale" and not stale["is_current"]
        run_all(client, job_id)
        settle(client, job_id)
        assert fake.i_calls[-1][1].instructions == "VERSION DEUX"
        assert len(fake.calls) == 1


def test_current_transcriptions_and_corrections_are_reused(tmp_path: Path):
    """A correction -- including a deliberately empty time -- is what gets
    interpreted, and no new transcription is requested."""
    fake = FakeModel()
    with make_client(tmp_path, fake, texts=ONE_ROW) as client:
        job_id, rows = setup(client, pages=(1,))
        run_all(client, job_id)
        row = settle(client, job_id)[1]
        accept(client, job_id, row, time=None, handwriting_text="VX tel (corrigé)")
        assert rows_of(client, job_id)[1]["interpretation"]["state"] == "stale"

        batch = run_all(client, job_id)
        assert batch["transcribing"] == [] and batch["interpreting"] == [rows[1]["row_id"]]
        settle(client, job_id)
        assert len(fake.calls) == 1  # transcribed once, ever
        request = fake.i_calls[-1][1]
        assert request.transcription_source == "reviewer"
        assert request.time is None and request.handwriting_text == "VX tel (corrigé)"


def test_missing_doctor_or_empty_profile_blocks_interpretation_clearly(tmp_path: Path):
    fake = FakeModel()
    with make_client(tmp_path, fake, texts=ONE_ROW) as client:
        job_id, _ = setup(client, doctor=None, pages=(1,))  # the default test doctor is empty
        batch = run_all(client, job_id)
        assert batch["interpretation_blocked"]["reason"] == "empty_profile"
        view = settle(client, job_id)[1]["interpretation"]
        assert view["state"] == "blocked" and "instructions or examples" in view["blocked_message"]
        assert fake.i_calls == []  # nothing guessed from general knowledge

        # No doctor at all (a document from before profiles).
        job = client.app.state.store.get(job_id)
        job.doctor = None
        view = rows_of(client, job_id)[1]["interpretation"]
        assert view["blocked_reason"] == "no_doctor" and "Assign one" in view["blocked_message"]
        assert run_all(client, job_id)["interpretation_blocked"]["reason"] == "no_doctor"


def test_unknown_annotations_stay_unresolved(tmp_path: Path):
    unknown = {"acts": [{"code": None, "units": None, "context_elements": [], "source_text": "QQ?",
                         "profile_reference": None}],
               "unresolved": [{"text": "QQ?", "issue": "not covered by the profile"}],
               "conflicts": ["image shows 'QO', transcription 'QQ'"], "ignored_notes": [],
               "no_billing_notation": False, "remarks": None}
    with make_client(tmp_path, FakeModel(interpretation=(unknown,)), texts=ONE_ROW) as client:
        job_id, _ = setup(client, pages=(1,))
        run_all(client, job_id)
        result = settle(client, job_id)[1]["interpretation"]["attempt"]["result"]
        assert result["acts"][0]["code"] is None
        assert result["unresolved"] == [{"text": "QQ?", "issue": "not covered by the profile"}]
        assert result["conflicts"]


def test_a_failed_interpretation_is_retried_without_transcribing_again(tmp_path: Path):
    fake = FakeModel(interpretation=(ex.AuthError("revoked"), TWO_ACTS))
    with make_client(tmp_path, fake, texts=ONE_ROW) as client:
        job_id, rows = setup(client, pages=(1,))
        run_all(client, job_id)
        view = settle(client, job_id)[1]["interpretation"]
        assert view["state"] == "failed" and view["last_error"]["kind"] == "auth"
        assert len(fake.i_calls) == 1  # permanent: no retry loop

        batch = run_all(client, job_id)
        assert batch["transcribing"] == [] and batch["interpreting"] == [rows[1]["row_id"]]
        assert settle(client, job_id)[1]["interpretation"]["state"] == "proposed"
        assert len(fake.calls) == 1

        # The per-row action re-interprets alone, too.
        response = client.post(f"/api/jobs/{job_id}/rows/{rows[1]['row_id']}/extract",
                               json={"stage": "interpretation"})
        assert response.status_code == 202
        settle(client, job_id)
        assert len(fake.calls) == 1 and len(fake.i_calls) == 3


def test_changes_to_evidence_or_transcription_outdate_the_proposal(tmp_path: Path):
    fake = FakeModel()
    with make_client(tmp_path, fake, texts=ONE_ROW) as client:
        job_id, rows = setup(client, pages=(1,))
        run_all(client, job_id)
        settle(client, job_id)

        # New transcription: the old proposal no longer applies.
        client.post(f"/api/jobs/{job_id}/rows/{rows[1]['row_id']}/extract", json={"stage": "transcription"})
        after = settle(client, job_id)[1]["interpretation"]
        assert after["state"] == "proposed" and after["attempt_count"] == 2  # re-interpreted automatically

        # New redaction in the row: transcription and proposal both outdated.
        add_redaction(client, job_id, 1)
        view = rows_of(client, job_id)[1]["interpretation"]
        assert view["state"] == "not_ready" and not view["is_current"]
        assert view["attempt"] is not None  # the old one is still shown, marked outdated

        run_all(client, job_id)  # re-prepares, re-transcribes, re-interprets
        final = settle(client, job_id)[1]
        assert final["interpretation"]["state"] == "proposed"
        assert final["interpretation"]["attempt"]["inputs"]["evidence"]["evidence_sha256"] == \
            final["evidence"][0]["sha256"]


def test_queued_and_retried_interpretations_never_send_outdated_inputs(tmp_path: Path):
    fake = FakeModel()
    fake.i_gate = threading.Event()
    with make_client(tmp_path, fake, concurrency=1) as client:
        job_id, rows = setup(client)
        # Transcribe both first (interpretations then queue behind a gate).
        run_all(client, job_id)
        assert fake.i_entered.wait(5)              # row A interpreting, row B queued
        # One worker, first in first out: row 1's interpretation is at the
        # model, row 2's is queued behind it.
        waiting = 2
        assert rows_of(client, job_id)[waiting]["interpretation"]["state"] == "running"
        add_redaction(client, job_id, waiting)      # row 2's evidence changes while it waits
        fake.i_gate.set()
        settled = settle(client, job_id)
        assert len(fake.i_calls) == 1
        assert settled[waiting]["interpretation"]["last_error"]["kind"] == "inputs_changed"

    fake = FakeModel(interpretation=(ex.RateLimited("slow"), TWO_ACTS))
    with make_client(tmp_path / "b", fake, texts=ONE_ROW) as client:
        job_id, rows = setup(client, pages=(1,))
        store = client.app.state.store
        job = store.get(job_id)
        other = client.post("/api/doctors", json={**DOCTOR, "display_name": "Autre"}).json()

        def reassign_during_backoff(_delay):
            jobslib.assign_doctor(job, store.doctors.get(other["profile_id"])["current"])
            store.save(job)

        store._sleep = reassign_during_backoff
        run_all(client, job_id)
        view = settle(client, job_id)[1]["interpretation"]
        assert len(fake.i_calls) == 1   # no retry with the old doctor
        assert view["last_error"]["kind"] == "inputs_changed"


def test_results_survive_a_restart_and_are_never_approved(tmp_path: Path):
    fake = FakeModel()
    with make_client(tmp_path, fake) as client:
        job_id, rows = setup(client)
        run_all(client, job_id)
        settle(client, job_id)
        client.app.state.store._extract_executor.shutdown(wait=True)

    # An interpretation cut short by the app stopping.
    folder = tmp_path / "documents" / job_id
    record = json.loads((folder / "document.json").read_text(encoding="utf-8"))
    row2 = rows[2]["row_id"]
    template = record["interpretations"][row2]["attempts"][0]
    record["interpretations"][row2]["attempts"].append(
        {**template, "attempt_id": "int_000000000009", "status": "running", "result": None})
    (folder / "document.json").write_text(json.dumps(record), encoding="utf-8")

    with make_client(tmp_path, FakeModel()) as client:
        views = rows_of(client, job_id)
        assert views[1]["interpretation"]["state"] == "proposed"
        assert views[1]["interpretation"]["attempt"]["result"]["acts"][0]["code"] == "11111"
        assert views[2]["interpretation"]["last_attempt_status"] == "interrupted"
        # The earlier success still matches its inputs, so it stays current.
        assert views[2]["interpretation"]["state"] == "proposed"

        data = manifest(client, job_id)
        for row in data["rows"]:
            draft = row["billing_draft"]
            assert draft["validated"] is False and draft["status"] == "proposed"
            assert [a["code"] for a in draft["proposal"]["acts"]] == ["11111", "22222"]
            assert draft["proposal"]["based_on"]["doctor_profile_version"] == 1
            assert row["extraction"]["status"] == "unreviewed"      # no review was created
        assert "UNVALIDATED" in data["notice"]
        assert data["summary"]["billing_draft"] == {"proposed": 2}
        assert_no_identifiers(data, "sk-or-v1")
        saved = json.loads((folder / "document.json").read_text(encoding="utf-8"))
        for record in saved["interpretations"].values():
            assert all(a["validated"] is False for a in record["attempts"])
        assert all(not r["reviews"] for r in saved["extractions"].values())
