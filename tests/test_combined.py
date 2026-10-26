"""The first pass over a row: transcription and interpretation in one call.

Synthetic documents and a scripted fake model: nothing leaves the machine.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nam_redactor import extraction as ex
from nam_redactor.web.app import create_app
from tests.test_extraction import VALID, Transport, completion, make_client, rows_of
from tests.test_interpretation import (
    DOCTOR,
    EXAMPLES,
    INSTRUCTIONS,
    ONE_ROW,
    TWO_ACTS,
    FakeModel,
    run_all,
    settle,
    setup,
)
from tests.test_web_api import NAM_TEXT, FakeEngine, no_ner


class CombinedModel(FakeModel):
    """A fake that also answers the combined call, recording it."""

    def __init__(self, transcription=VALID, interpretation=TWO_ACTS):
        super().__init__((transcription,), (interpretation,))
        self.c_calls: list[tuple[bytes, str, tuple, str]] = []
        self._answer = {"transcription": transcription, "interpretation": interpretation}

    def extract_and_interpret(self, image_png, instructions, examples, model):
        with self._lock:
            self.c_calls.append((image_png, instructions, examples, model))
        return ex.ExtractionOutput(
            result=ex.validate_combined(self._answer), returned_model=model, provider="FakeProvider",
            generation_id="gen-c", usage={"prompt_tokens": 1500, "completion_tokens": 200,
                                          "total_tokens": 1700, "cost": 0.04, "cost_currency": "credits"},
        )


def test_the_combined_request_carries_the_profile_and_image_but_no_identifiers(monkeypatch):
    monkeypatch.setenv(ex.API_KEY_ENV, "sk-or-v1-TEST")
    answer = json.loads(completion({"transcription": VALID, "interpretation": TWO_ACTS}))
    transport = Transport((200, json.dumps(answer).encode()))
    out = ex.OpenRouterExtractor(transport).extract_and_interpret(
        b"PNG-BYTES", INSTRUCTIONS, tuple(EXAMPLES), ex.DEFAULT_MODEL
    )

    body = transport.calls[0]["body"]
    text, image = body["messages"][1]["content"]
    assert INSTRUCTIONS in text["text"] and "Annotation as written: VX tel" in text["text"]
    assert image["image_url"]["url"].endswith("UE5HLUJZVEVT")
    assert body["provider"]["zdr"] is True and body["provider"]["data_collection"] == "deny"
    assert body["response_format"]["json_schema"]["name"] == "row_combined"
    for identifier in ("1-00001-2", "00123", "licence"):
        assert identifier not in json.dumps(body)
    assert out.result["transcription"]["time"] == VALID["time"]
    assert out.result["interpretation"]["acts"][0]["code"] == "11111"


@pytest.mark.parametrize("bad", [
    {"transcription": VALID},                                                   # a part missing
    {"transcription": VALID, "interpretation": TWO_ACTS, "extra": 1},           # an unknown part
    {"transcription": {**VALID, "time": "9h30"}, "interpretation": TWO_ACTS},   # invalid transcription
    {"transcription": VALID, "interpretation": {**TWO_ACTS, "no_billing_notation": True}},
])
def test_a_combined_answer_is_valid_only_if_both_parts_are(bad):
    with pytest.raises(ex.InvalidOutput):
        ex.validate_combined(bad)


def test_one_call_per_row_gives_both_a_transcription_and_a_proposal(tmp_path: Path):
    fake = CombinedModel()
    with make_client(tmp_path, fake) as client:
        job_id, rows = setup(client)
        run_all(client, job_id)
        settled = settle(client, job_id)

        assert len(fake.c_calls) == len(rows) == 2
        assert fake.calls == [] and fake.i_calls == []      # no separate calls
        for _, instructions, examples, _ in fake.c_calls:
            assert instructions == INSTRUCTIONS and examples == tuple(EXAMPLES)
        for row in settled.values():
            transcription = row["extraction"]
            assert transcription["state"] == "unreviewed"
            assert transcription["attempt"]["result"]["time"] == VALID["time"]
            assert transcription["attempt"]["prompt_version"] == ex.COMBINED_PROMPT_VERSION
            view = row["interpretation"]
            assert view["state"] == "proposed" and view["is_current"] and view["validated"] is False
            assert view["attempt"]["combined_with"] == transcription["attempt"]["attempt_id"]
            assert view["attempt"]["usage"] is None          # counted once, on the transcription
            assert view["attempt"]["inputs"]["source"] == "model"

        again = run_all(client, job_id)
        assert again["transcribing"] == [] and again["interpreting"] == []


def test_later_reruns_interpret_alone_without_reading_the_row_again(tmp_path: Path):
    fake = CombinedModel()
    with make_client(tmp_path, fake, texts=ONE_ROW) as client:
        job_id, _ = setup(client, pages=(1,))
        run_all(client, job_id)
        settle(client, job_id)
        pid = client.get(f"/api/jobs/{job_id}").json()["doctor"]["profile_id"]
        client.put(f"/api/doctors/{pid}", json={**DOCTOR, "instructions": "VERSION DEUX"})
        client.put(f"/api/jobs/{job_id}/doctor", json={"profile_id": pid, "version": 2})
        assert rows_of(client, job_id)[1]["interpretation"]["state"] == "stale"

        run_all(client, job_id)
        settle(client, job_id)
        assert len(fake.c_calls) == 1
        ((_, request, _),) = fake.i_calls
        assert request.instructions == "VERSION DEUX" and request.transcription_source == "model"


def test_without_a_usable_profile_the_row_is_only_transcribed(tmp_path: Path):
    fake = CombinedModel()
    with make_client(tmp_path, fake, texts=ONE_ROW) as client:
        job_id, _ = setup(client, doctor=None, pages=(1,))
        batch = run_all(client, job_id)
        row = settle(client, job_id)[1]
        assert batch["interpretation_blocked"] is not None
        assert fake.c_calls == [] and len(fake.calls) == 1
        assert row["extraction"]["attempt"]["prompt_version"] == ex.PROMPT_VERSION
        assert row["interpretation"]["state"] == "blocked"


def test_models_are_loaded_at_startup_when_asked(tmp_path: Path):
    loaded = threading.Event()

    class Engine(FakeEngine):
        def validate(self):
            loaded.set()

    app = create_app(
        engine_factory=lambda: Engine({1: NAM_TEXT}), dir_root=tmp_path,
        name_model_factory=no_ner, preload_models=True,
    )
    with TestClient(app):
        assert loaded.wait(5)
