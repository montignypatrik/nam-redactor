"""Billing lines per appointment: manual edits, confirmation, and the CSV.

Synthetic documents and a fake model. Editing and confirming never call a model.
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from nam_redactor.web.billing import EDITS_OUTDATED, UNUSED
from tests.test_billing_csv import NAM, ROW_A, ROW_B, client_for, download, parse, processed
from tests.test_extraction import rows_of
from tests.test_interpretation import DOCTOR, FakeModel, run_all, settle


def billing_of(client: TestClient, job_id: str, page: int) -> dict:
    return rows_of(client, job_id)[page]["billing"]


def lines_body(view: dict, acts: list[dict], no_billing: bool = False) -> dict:
    return {"acts": acts, "no_billing": no_billing,
            "base_version": view["version"], "attempt_id": view["attempt_id"]}


def act(code: str, units=None, contexts=(), model_index=None) -> dict:
    return {"code": code, "units": units, "context_elements": list(contexts), "model_index": model_index}


def save(client, job_id, page, acts, no_billing=False, path=""):
    row = rows_of(client, job_id)[page]
    return client.request(
        "POST" if path else "PUT",
        f"/api/jobs/{job_id}/rows/{row['row_id']}/billing{path}",
        json=lines_body(row["billing"], acts, no_billing),
    )


def confirm(client, job_id, page, acts, no_billing=False):
    return save(client, job_id, page, acts, no_billing, path="/confirm")


def codes(view: dict) -> list[str]:
    return [line["code"] for line in view["lines"]]


def test_each_proposed_act_is_its_own_billing_line(tmp_path: Path):
    client, _ = client_for(tmp_path)
    with client:
        job_id = processed(client)
        view = billing_of(client, job_id, 1)
    assert view["source"] == "model" and view["state"] == "draft"
    assert codes(view) == ["vs", "15824"]                       # exact spelling
    assert [l["units"] for l in view["lines"]] == [None, None]    # unknown stays empty
    assert [l["context_elements"] for l in view["lines"]] == [["TT", "XY"], []]
    shared = view["shared"]
    assert shared["HIN"] == NAM                                   # resolved locally
    assert (shared["doctorLicenceID"], shared["doctorGroupNumber"], shared["Sector"]) == ("1-00001-2", "00123", "3")
    assert (shared["ClaimDate"], shared["ProcedureTime"]) == ("2026-03-14", "09:30")
    assert all(shared[column] == "" for column in UNUSED)
    assert view["warnings"] == 1 and view["problems"] == []


def test_added_edited_and_removed_acts_persist_and_export_exactly(tmp_path: Path):
    client, fake = client_for(tmp_path)
    with client:
        job_id = processed(client)
        calls = (len(fake.calls), len(fake.i_calls))
        response = save(client, job_id, 1, [
            act("vs", None, ["TT"], 0),              # a context removed
            # act 2 ("15824") removed
            act("09999", 2.5, ["A", "B c"], None),   # added by hand
            act("vs", None, [], None),               # a second line with the same code
        ])
        assert response.status_code == 200, response.text
        view = response.json()["billing"]
        assert view["source"] == "manual" and view["edits_current"] is True
        assert (len(fake.calls), len(fake.i_calls)) == calls     # no model request
        # The model's own output is untouched, for traceability.
        proposal = rows_of(client, job_id)[1]["interpretation"]["attempt"]["result"]
        assert [a["code"] for a in proposal["acts"]] == ["vs", "15824"]

    client, _ = client_for(tmp_path)                            # restart
    with client:
        view = billing_of(client, job_id, 1)
        assert codes(view) == ["vs", "09999", "vs"]
        assert [l["units"] for l in view["lines"]] == [None, 2.5, None]
        assert [l["context_elements"] for l in view["lines"]] == [["TT"], ["A", "B c"], []]
        assert [l["model_index"] for l in view["lines"]] == [0, None, None]
        lines = parse(download(client, job_id).content)
    first = [l for l in lines if l["ClaimDate"] == "2026-03-14"]
    assert [l["ProcedureCode"] for l in first] == ["vs", "09999", "vs"]      # 15824 stays removed
    assert [l["Units"] for l in first] == ["", "2.5", ""]
    assert [l["ContextualElements"] for l in first] == ["TT", "A,B c", ""]
    assert [l["ProcedureCode"] for l in lines if l["ClaimDate"] == "2026-03-15"] == ["vpm", "431"]


def test_codes_keep_their_spelling_and_blanks_stay_blank(tmp_path: Path):
    client, _ = client_for(tmp_path)
    with client:
        job_id = processed(client)
        view = save(client, job_id, 1, [act("  vs ", None, [], 0), act("15824", None, [], 1)]).json()["billing"]
        assert codes(view) == ["vs", "15824"]                      # only surrounding spaces go
        view = confirm(client, job_id, 1, [act("vs"), act("15824")]).json()["billing"]
        assert view["state"] == "confirmed"
        assert [l["units"] for l in view["lines"]] == [None, None]
        assert [l["context_elements"] for l in view["lines"]] == [[], []]
        lines = parse(download(client, job_id).content)[:2]
        assert [(l["ProcedureCode"], l["Units"], l["ContextualElements"]) for l in lines] == [
            ("vs", "", ""), ("15824", "", "")]
        # Bad input is refused rather than coerced.
        for bad in (act("vs", "2"), act("vs", -1), act("vs", None, ["a,b"]), act("vs", None, [""]),
                    act("v\ns")):
            assert save(client, job_id, 1, [bad]).status_code == 400


def test_confirmation_saves_the_lines_and_survives_a_restart(tmp_path: Path):
    client, fake = client_for(tmp_path)
    with client:
        job_id = processed(client)
        calls = (len(fake.calls), len(fake.i_calls))
        # Unfinished lines, or nothing at all, cannot be confirmed.
        assert confirm(client, job_id, 1, [act("vs"), act("")]).status_code == 400
        assert confirm(client, job_id, 1, []).status_code == 400
        response = confirm(client, job_id, 1, [act("vs", None, ["TT", "XY"], 0), act("15824", 1, [], 1)])
        assert response.status_code == 200, response.text
        assert (len(fake.calls), len(fake.i_calls)) == calls
        # Explicitly no billing acts.
        assert confirm(client, job_id, 2, [], no_billing=True).status_code == 200
        # Confirmation is its own state: transcription and readiness are unchanged.
        row = rows_of(client, job_id)[1]
        assert row["extraction"]["review"] is None and row["interpretation"]["validated"] is False

    client, _ = client_for(tmp_path)                            # restart
    with client:
        one, two = billing_of(client, job_id, 1), billing_of(client, job_id, 2)
        assert one["state"] == "confirmed" and one["confirmation"]["action"] == "confirm"
        assert codes(one) == ["vs", "15824"] and [l["units"] for l in one["lines"]] == [None, 1]
        assert two["state"] == "confirmed" and two["no_billing"] is True and two["lines"] == []
        response = download(client, job_id)
        assert response.headers["x-billing-confirmed-rows"] == "2"
        assert response.headers["x-billing-no-notation-rows"] == "1"
        assert [l["ProcedureCode"] for l in parse(response.content)] == ["vs", "15824"]


def test_editing_or_reopening_ends_the_confirmation(tmp_path: Path):
    client, _ = client_for(tmp_path)
    with client:
        job_id = processed(client)
        confirm(client, job_id, 1, [act("vs"), act("15824")])
        view = save(client, job_id, 1, [act("vs")]).json()["billing"]
        assert view["state"] == "draft" and view["confirmation"] is None
        assert (view["edit_count"], view["confirmation_count"]) == (2, 1)     # history kept
        view = confirm(client, job_id, 1, [act("vs")]).json()["billing"]
        assert view["state"] == "confirmed"
        row_id = rows_of(client, job_id)[1]["row_id"]
        view = client.post(f"/api/jobs/{job_id}/rows/{row_id}/billing/reopen",
                           json={"base_version": view["version"]}).json()["billing"]
        assert view["state"] == "draft" and codes(view) == ["vs"]            # lines stay


def test_a_changed_source_outdates_the_confirmation_without_losing_it(tmp_path: Path):
    client, fake = client_for(tmp_path, ROW_A, ROW_B, ROW_B, ROW_B)
    with client:
        job_id = processed(client)
        confirm(client, job_id, 1, [act("vs"), act("09999")])
        pid = client.get(f"/api/jobs/{job_id}").json()["doctor"]["profile_id"]
        client.put(f"/api/doctors/{pid}", json={**DOCTOR, "instructions": "v2"})
        client.put(f"/api/jobs/{job_id}/doctor", json={"profile_id": pid, "version": 2})

        view = billing_of(client, job_id, 1)
        assert view["state"] == "outdated" and view["edits_current"] is False
        assert codes(view) == ["vs", "09999"]                      # manual work kept
        assert EDITS_OUTDATED in view["problems"]

        # New model results arrive: they never replace the person's lines.
        run_all(client, job_id)
        settle(client, job_id)
        view = billing_of(client, job_id, 1)
        assert rows_of(client, job_id)[1]["interpretation"]["state"] == "proposed"
        assert view["state"] == "outdated" and codes(view) == ["vs", "09999"]
        body = download(client, job_id).json()                     # not exported as current
        assert [(p["label"], p["reason"]) for p in body["problems"]] == [("#1", EDITS_OUTDATED)]

        # Keep the lines against the new source, then confirm again.
        view = save(client, job_id, 1, [act("vs"), act("09999")]).json()["billing"]
        assert view["state"] == "draft" and view["edits_current"] is True
        view = confirm(client, job_id, 1, [act("vs"), act("09999")]).json()["billing"]
        assert view["state"] == "confirmed" and view["confirmation_count"] == 2
        assert [l["ProcedureCode"] for l in parse(download(client, job_id).content)][:2] == ["vs", "09999"]


def test_changes_from_an_out_of_date_screen_are_refused(tmp_path: Path):
    client, _ = client_for(tmp_path)
    with client:
        job_id = processed(client)
        row = rows_of(client, job_id)[1]
        stale = lines_body(row["billing"], [act("vs")])
        url = f"/api/jobs/{job_id}/rows/{row['row_id']}/billing"
        assert client.put(url, json=stale).status_code == 200
        assert client.put(url, json=stale).status_code == 409          # version moved on
        fresh = billing_of(client, job_id, 1)
        wrong_proposal = {**lines_body(fresh, [act("vs")]), "attempt_id": "int_000000000000"}
        assert client.put(url, json=wrong_proposal).status_code == 409
        assert codes(billing_of(client, job_id, 1)) == ["vs"]


def test_discarding_manual_lines_returns_to_the_proposal(tmp_path: Path):
    client, _ = client_for(tmp_path)
    with client:
        job_id = processed(client)
        view = confirm(client, job_id, 1, [act("zz")]).json()["billing"]
        row_id = rows_of(client, job_id)[1]["row_id"]
        view = client.post(f"/api/jobs/{job_id}/rows/{row_id}/billing/reset",
                           json={"base_version": view["version"]}).json()["billing"]
        assert view["source"] == "model" and view["state"] == "draft"
        assert codes(view) == ["vs", "15824"]
        assert (view["edit_count"], view["confirmation_count"]) == (1, 1)


def test_confirming_needs_what_export_needs(tmp_path: Path):
    client, _ = client_for(tmp_path, ROW_A, ROW_A)
    with client:
        job_id = processed(client)
        client.app.state.store.get(job_id).patient_ids.clear()
        response = confirm(client, job_id, 1, [act("vs")])
        assert response.status_code == 409
        assert "no health number on record for Patient_0001" in response.json()["detail"]
        view = billing_of(client, job_id, 1)
        assert view["state"] == "draft" and codes(view) == ["vs"]      # the lines were kept


def test_drafts_still_export_without_any_confirmation(tmp_path: Path):
    client, _ = client_for(tmp_path)
    with client:
        job_id = processed(client)
        save(client, job_id, 2, [act("vpm"), act("431", 1), act("")])    # unfinished line
        body = download(client, job_id).json()
        assert [(p["label"], p["reason"]) for p in body["problems"]] == [("#2", "no procedure code on line 3")]
        save(client, job_id, 2, [act("vpm"), act("431", 1)])
        response = download(client, job_id)
        assert response.status_code == 200 and response.headers["x-billing-confirmed-rows"] == "0"
        assert [l["ProcedureCode"] for l in parse(response.content)] == ["vs", "15824", "vpm", "431"]
