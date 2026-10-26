"""Cloud extraction: the OpenRouter adapter and the extraction workflow.

OpenRouter is never contacted: the adapter tests inject a fake transport, and
the workflow tests inject a scripted fake extractor.
"""
from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nam_redactor import extraction as ex
from nam_redactor.web import jobs as jobslib
from nam_redactor.web.app import create_app
from tests.privacy import assert_no_identifiers
from tests.test_documents import manifest, pages_of, save_review, section_rows, whole_document
from tests.test_web_api import NAM_TEXT, FakeEngine, no_ner

FAKE_KEY = "sk-or-v1-TESTKEY-never-real-0000"
PNG = b"\x89PNG\r\n\x1a\nfake-row-image"

VALID = {
    "time": "09:30",
    "time_evidence": "9h30",
    "handwriting_status": "legible",
    "handwriting_text": "Rx: amox 500 TID x7j",
    "unreadable_or_ambiguous": [],
    "remarks": None,
}


def completion(result: dict | str, **extra) -> bytes:
    content = result if isinstance(result, str) else json.dumps(result)
    return json.dumps({
        "id": "gen-123",
        "model": "anthropic/claude-opus-5.5",
        "provider": "Google",
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 80, "total_tokens": 1280, "cost": 0.0065},
        **extra,
    }).encode()


class Transport:
    """Records requests; answers from a list of (status, body) or exceptions."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[dict] = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": dict(headers), "body": json.loads(body)})
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv(ex.API_KEY_ENV, FAKE_KEY)


# -- adapter ---------------------------------------------------------------------


def test_missing_key_is_reported_and_nothing_is_sent(monkeypatch):
    monkeypatch.delenv(ex.API_KEY_ENV, raising=False)
    transport = Transport()
    adapter = ex.OpenRouterExtractor(transport)
    assert not adapter.configured()
    with pytest.raises(ex.NotConfigured) as err:
        adapter.extract(PNG, ex.DEFAULT_MODEL)
    assert not err.value.transient
    assert transport.calls == []


def test_the_request_carries_only_the_image_and_fixed_instructions(key):
    transport = Transport((200, completion(VALID)))
    ex.OpenRouterExtractor(transport).extract(PNG, "google/gemini-3.8-flash")
    (call,) = transport.calls
    body = call["body"]

    assert call["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert call["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in json.dumps(body)  # the key travels in the header only
    assert body["model"] == "google/gemini-3.8-flash"
    # Privacy routing is explicit and strict.
    assert body["provider"] == {
        "require_parameters": True, "zdr": True, "data_collection": "deny", "allow_fallbacks": True,
    }
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert "temperature" not in body  # unsupported by one model: would remove routes
    # Exactly: system instructions, a fixed user line, the image. No context.
    system, user = body["messages"]
    assert system["role"] == "system" and user["role"] == "user"
    text, image = user["content"]
    assert text == {"type": "text", "text": ex.USER_PROMPT}
    url = image["image_url"]["url"]
    assert base64.b64decode(url.split(",", 1)[1]) == PNG


def test_a_valid_answer_is_parsed_with_usage_and_provider(key):
    out = ex.OpenRouterExtractor(Transport((200, completion(VALID)))).extract(PNG, ex.DEFAULT_MODEL)
    assert out.result["time"] == "09:30"
    assert out.result["handwriting_text"] == "Rx: amox 500 TID x7j"
    assert out.returned_model == "anthropic/claude-opus-5.5" and out.provider == "Google"
    assert out.usage == {"prompt_tokens": 1200, "completion_tokens": 80, "total_tokens": 1280,
                         "cost": 0.0065, "cost_currency": "credits"}


def test_unknown_time_and_blank_or_unreadable_handwriting(key):
    for status in ("blank", "unreadable"):
        answer = {**VALID, "time": None, "time_evidence": None,
                  "handwriting_status": status, "handwriting_text": None}
        out = ex.OpenRouterExtractor(Transport((200, completion(answer)))).extract(PNG, ex.DEFAULT_MODEL)
        assert out.result["time"] is None
        assert out.result["handwriting_status"] == status and out.result["handwriting_text"] is None


def test_missing_usage_is_null_not_zero(key):
    body = json.loads(completion(VALID))
    del body["usage"]
    out = ex.OpenRouterExtractor(Transport((200, json.dumps(body).encode()))).extract(PNG, ex.DEFAULT_MODEL)
    assert out.usage["cost"] is None and out.usage["prompt_tokens"] is None


@pytest.mark.parametrize("content", [
    "not json",
    json.dumps({**VALID, "time": "9h30"}),                     # not HH:MM
    json.dumps({**VALID, "handwriting_status": "blank"}),       # blank but text given
    json.dumps({**VALID, "handwriting_text": None}),            # legible but no text
    json.dumps({**VALID, "billing_code": "08123"}),             # unexpected field
    json.dumps({k: v for k, v in VALID.items() if k != "remarks"}),
])
def test_malformed_answers_are_invalid_output(key, content):
    with pytest.raises(ex.InvalidOutput) as err:
        ex.OpenRouterExtractor(Transport((200, completion(content)))).extract(PNG, ex.DEFAULT_MODEL)
    assert err.value.transient


def test_fenced_json_is_accepted(key):
    fenced = "```json\n" + json.dumps(VALID) + "\n```"
    out = ex.OpenRouterExtractor(Transport((200, completion(fenced)))).extract(PNG, ex.DEFAULT_MODEL)
    assert out.result["time"] == "09:30"


@pytest.mark.parametrize("status, kind, transient", [
    (401, "auth", False),
    (402, "payment_required", False),
    (404, "route_unavailable", False),
    (400, "bad_request", False),
    (429, "rate_limited", True),
    (502, "provider_error", True),
])
def test_http_errors_are_typed_and_never_echo_the_key(key, status, kind, transient):
    body = json.dumps({"error": {"code": status, "message": f"bad key {FAKE_KEY}",
                                 "metadata": {"retry_after": 3}}}).encode()
    with pytest.raises(ex.ExtractionError) as err:
        ex.OpenRouterExtractor(Transport((status, body))).extract(PNG, ex.DEFAULT_MODEL)
    assert err.value.kind == kind and err.value.transient is transient
    assert FAKE_KEY not in str(err.value)
    if status == 429:
        assert err.value.retry_after == 3


def test_timeouts_and_connection_failures_are_transient(key):
    with pytest.raises(ex.Timeout):
        ex.OpenRouterExtractor(Transport(TimeoutError())).extract(PNG, ex.DEFAULT_MODEL)
    with pytest.raises(ex.ProviderError):
        ex.OpenRouterExtractor(Transport(OSError())).extract(PNG, ex.DEFAULT_MODEL)


def test_only_offered_models_are_sent(key):
    transport = Transport()
    with pytest.raises(ex.BadRequest):
        ex.OpenRouterExtractor(transport).extract(PNG, "some/other-model")
    assert transport.calls == []


# -- workflow -------------------------------------------------------------------


class FakeExtractor:
    """Scripted stand-in for the adapter. ``script`` items are results
    (dicts) or exceptions, consumed per call; the last one repeats."""

    def __init__(self, *script, configured: bool = True):
        self.script = list(script) or [VALID]
        self.calls: list[tuple[bytes, str]] = []
        self.gate: threading.Event | None = None
        self.entered = threading.Event()
        self._configured = configured
        self._lock = threading.Lock()

    def configured(self) -> bool:
        return self._configured

    def extract(self, image_png: bytes, model: str) -> ex.ExtractionOutput:
        with self._lock:
            self.calls.append((image_png, model))
            item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10)
        if isinstance(item, BaseException):
            raise item
        return ex.ExtractionOutput(
            result=ex.validate_result(item), returned_model=model, provider="FakeProvider",
            generation_id="gen-1", usage={"prompt_tokens": 10, "completion_tokens": 5,
                                          "total_tokens": 15, "cost": None, "cost_currency": None},
        )


def make_client(tmp_path: Path, fake: FakeExtractor, texts=None, concurrency=None) -> TestClient:
    return TestClient(create_app(
        engine_factory=lambda: FakeEngine(texts if texts is not None else {1: NAM_TEXT, 2: NAM_TEXT}),
        dir_root=tmp_path,
        name_model_factory=no_ner,
        extractor_factory=lambda: fake,
        retry_delays=(0.0, 0.0),
        extraction_concurrency=concurrency,
    ))


def prepared(client: TestClient, pages=(1, 2)) -> tuple[str, dict[int, dict]]:
    """A document whose section rows on ``pages`` are ready for extraction."""
    job_id = whole_document(client)
    save_review(client, job_id)
    rows = section_rows(client, job_id)
    for page in pages:
        row = client.patch(f"/api/jobs/{job_id}/rows/{rows[page]['row_id']}",
                           json={"confirm": ["patient_id", "date"]}).json()
        response = client.post(f"/api/jobs/{job_id}/rows/{row['row_id']}/approve",
                               json={"revision": row["revision"]})
        assert response.status_code == 200, response.text
    return job_id, section_rows(client, job_id)


def rows_of(client: TestClient, job_id: str) -> dict[int, dict]:
    return {r["working_page"]: r for r in client.get(f"/api/jobs/{job_id}/rows").json()["rows"]}


def wait_settled(client: TestClient, job_id: str, timeout: float = 10.0) -> dict[int, dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = rows_of(client, job_id)
        if all(r["extraction"]["state"] != "running" for r in rows.values()):
            return rows
        time.sleep(0.02)
    raise AssertionError("extraction did not settle")


def extract(client, job_id, row, model=ex.DEFAULT_MODEL):
    return client.post(f"/api/jobs/{job_id}/rows/{row['row_id']}/extract", json={"model": model})


def accept(client, job_id, row, **values):
    attempt = row["extraction"]["attempt"]
    body = {"attempt_id": attempt["attempt_id"], **{
        "time": attempt["result"]["time"],
        "handwriting_status": attempt["result"]["handwriting_status"],
        "handwriting_text": attempt["result"]["handwriting_text"],
        **values,
    }}
    return client.put(f"/api/jobs/{job_id}/rows/{row['row_id']}/extraction/review", json=body)


def test_config_reports_setup_without_exposing_the_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(ex.API_KEY_ENV, FAKE_KEY)
    with TestClient(create_app(dir_root=tmp_path, name_model_factory=no_ner)) as client:
        response = client.get("/api/extraction")
        assert response.json()["configured"] is True
        assert FAKE_KEY not in response.text
        assert response.json()["default_model"] == "anthropic/claude-opus-5.5"
        assert response.json()["routing"]["zdr"] is True
    monkeypatch.delenv(ex.API_KEY_ENV)
    with TestClient(create_app(dir_root=tmp_path, name_model_factory=no_ner)) as client:
        assert client.get("/api/extraction").json()["configured"] is False


def test_a_key_saved_in_the_app_configures_extraction_and_is_never_sent_back(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(ex.API_KEY_ENV, raising=False)
    with TestClient(create_app(dir_root=tmp_path, name_model_factory=no_ner)) as client:
        assert client.put("/api/extraction/key", json={"key": "not a key"}).status_code == 400
        response = client.put("/api/extraction/key", json={"key": f"  {FAKE_KEY}\n"})
        assert response.status_code == 200
        assert response.json()["configured"] is True and response.json()["key_source"] == "saved"
        assert FAKE_KEY not in response.text
        assert FAKE_KEY not in client.get("/api/extraction").text

        key_file = client.app.state.store.key_file
        assert key_file.read_text(encoding="utf-8") == FAKE_KEY
        sent = {}
        def transport(url, headers, body, timeout):
            sent.update(headers)
            raise OSError("stop here")
        with pytest.raises(ex.ExtractionError):
            ex.OpenRouterExtractor(transport, key_file=key_file).extract(b"png", ex.DEFAULT_MODEL)
        assert sent["Authorization"] == f"Bearer {FAKE_KEY}"

        # The environment variable wins over the saved key.
        monkeypatch.setenv(ex.API_KEY_ENV, "sk-or-v1-from-the-environment-0000")
        assert client.get("/api/extraction").json()["key_source"] == "environment"
        monkeypatch.delenv(ex.API_KEY_ENV)

        removed = client.delete("/api/extraction/key").json()
        assert removed["configured"] is False and removed["key_source"] is None
        assert not key_file.exists()


def test_extraction_is_refused_without_configuration(tmp_path: Path):
    fake = FakeExtractor(configured=False)
    with make_client(tmp_path, fake) as client:
        job_id, rows = prepared(client, pages=(1,))
        response = extract(client, job_id, rows[1])
        assert response.status_code == 503 and "OPENROUTER_API_KEY" in response.json()["detail"]
        assert fake.calls == []


def test_the_frozen_redacted_row_evidence_is_what_is_sent(tmp_path: Path):
    fake = FakeExtractor(VALID)
    with make_client(tmp_path, fake) as client:
        job_id, rows = prepared(client, pages=(1,))
        assert extract(client, job_id, rows[1], "openai/gpt-6-astra").status_code == 202
        row = wait_settled(client, job_id)[1]

        ((image, model),) = fake.calls
        ref = rows[1]["evidence"][0]
        assert hashlib.sha256(image).hexdigest() == ref["sha256"]
        assert image == (tmp_path / "documents" / job_id / ref["path"]).read_bytes()
        assert model == "openai/gpt-6-astra"

        view = row["extraction"]
        assert view["state"] == "unreviewed"
        attempt = view["attempt"]
        assert attempt["requested_model"] == "openai/gpt-6-astra"
        assert attempt["provider"] == "FakeProvider"
        assert attempt["prompt_version"] == ex.PROMPT_VERSION
        assert attempt["basis"]["evidence_sha256"] == ref["sha256"]
        assert attempt["basis"]["row_revision"] == rows[1]["approval"]["approved_revision"]
        assert attempt["usage"]["cost"] is None  # not reported -> null, never 0


def test_rows_without_a_known_patient_are_not_extracted(tmp_path: Path):
    fake = FakeExtractor()
    with make_client(tmp_path, fake, texts={1: NAM_TEXT, 2: ""}) as client:
        job_id, rows = prepared(client, pages=(1,))
        response = extract(client, job_id, rows[2])
        assert response.status_code == 409 and "needs patient" in response.json()["detail"]
        batch = client.post(f"/api/jobs/{job_id}/extract", json={}).json()
        assert batch["queued"] == [rows[1]["row_id"]]
        assert batch["skipped"] == {rows[2]["row_id"]: "needs patient"}
        wait_settled(client, job_id)
        assert len(fake.calls) == 1


def test_extracting_prepares_known_rows_automatically(tmp_path: Path):
    """The whole workflow in two clicks: save the review, extract all."""
    fake = FakeExtractor(VALID)
    with make_client(tmp_path, fake) as client:
        job_id = whole_document(client)
        save_review(client, job_id)  # no per-row confirmation, no "mark ready"
        batch = client.post(f"/api/jobs/{job_id}/extract", json={}).json()
        rows = section_rows(client, job_id)
        assert sorted(batch["prepared"]) == sorted(r["row_id"] for r in rows.values())
        assert sorted(batch["queued"]) == sorted(batch["prepared"])
        settled = wait_settled(client, job_id)
        assert {r["extraction"]["state"] for r in settled.values()} == {"unreviewed"}
        assert {r["approval"]["method"] for r in settled.values()} == {"automatic"}

        # One row, not yet ready: "Extract row" prepares it too.
    with make_client(tmp_path / "one", FakeExtractor(VALID)) as client:
        job_id = whole_document(client)
        save_review(client, job_id)
        row = section_rows(client, job_id)[2]
        assert extract(client, job_id, row).status_code == 202
        settled = wait_settled(client, job_id)
        assert settled[2]["extraction"]["state"] == "unreviewed"
        assert settled[1]["extraction"]["state"] == "not_ready"  # untouched


def test_repeated_clicks_never_start_duplicate_work(tmp_path: Path):
    fake = FakeExtractor()
    fake.gate = threading.Event()
    with make_client(tmp_path, fake, texts={1: NAM_TEXT, 2: ""}) as client:
        job_id, rows = prepared(client, pages=(1,))
        assert extract(client, job_id, rows[1]).status_code == 202
        assert fake.entered.wait(5)
        assert rows_of(client, job_id)[1]["extraction"]["state"] == "running"
        assert extract(client, job_id, rows[1]).status_code == 409
        assert client.post(f"/api/jobs/{job_id}/extract", json={}).json()["queued"] == []
        fake.gate.set()
        wait_settled(client, job_id)
        assert len(fake.calls) == 1


def test_transient_failures_retry_and_permanent_ones_do_not(tmp_path: Path):
    fake = FakeExtractor(ex.RateLimited("slow down"), ex.ProviderError("upstream"), VALID)
    with make_client(tmp_path, fake) as client:
        job_id, rows = prepared(client, pages=(1,))
        extract(client, job_id, rows[1])
        view = wait_settled(client, job_id)[1]["extraction"]
        assert view["state"] == "unreviewed" and view["attempt"]["tries"] == 3
        assert view["attempt_count"] == 1

    fake = FakeExtractor(ex.AuthError("bad key"))
    with make_client(tmp_path / "b", fake) as client:
        job_id, rows = prepared(client, pages=(1,))
        extract(client, job_id, rows[1])
        view = wait_settled(client, job_id)[1]["extraction"]
        assert view["state"] == "failed" and view["last_error"]["kind"] == "auth"
        assert len(fake.calls) == 1

    fake = FakeExtractor(ex.InvalidOutput("bad json"))
    with make_client(tmp_path / "c", fake) as client:
        job_id, rows = prepared(client, pages=(1,))
        extract(client, job_id, rows[1])
        view = wait_settled(client, job_id)[1]["extraction"]
        assert view["state"] == "failed" and view["last_error"]["kind"] == "invalid_output"
        assert len(fake.calls) == 2  # invalid output: one more sample only


def test_a_partial_batch_failure_retries_only_the_failed_rows(tmp_path: Path):
    fake = FakeExtractor(VALID, ex.AuthError("revoked"), VALID)
    with make_client(tmp_path, fake) as client:
        fake_calls_before = 0
        job_id, rows = prepared(client)
        client.post(f"/api/jobs/{job_id}/extract", json={})
        states = {p: r["extraction"]["state"] for p, r in wait_settled(client, job_id).items()}
        assert sorted(states.values()) == ["failed", "unreviewed"]
        failed_page = next(p for p, s in states.items() if s == "failed")

        fake_calls_before = len(fake.calls)
        again = client.post(f"/api/jobs/{job_id}/extract", json={}).json()
        assert again["queued"] == [rows[failed_page]["row_id"]]
        after = wait_settled(client, job_id)
        assert len(fake.calls) == fake_calls_before + 1
        assert {r["extraction"]["state"] for r in after.values()} == {"unreviewed"}
        assert after[failed_page]["extraction"]["attempt_count"] == 2


def test_manually_confirmed_time_is_kept_and_a_discrepancy_shown(tmp_path: Path):
    fake = FakeExtractor({**VALID, "time": "10:15"})
    with make_client(tmp_path, fake) as client:
        job_id = whole_document(client)
        save_review(client, job_id)
        row = section_rows(client, job_id)[1]
        row = client.patch(f"/api/jobs/{job_id}/rows/{row['row_id']}",
                           json={"time": "09:30", "confirm": ["patient_id", "date"]}).json()
        client.post(f"/api/jobs/{job_id}/rows/{row['row_id']}/approve", json={"revision": row["revision"]})
        extract(client, job_id, row)
        row = wait_settled(client, job_id)[1]
        assert row["extraction"]["time_discrepancy"] == {"confirmed": "09:30", "proposed": "10:15"}

        reviewed = accept(client, job_id, row, time="09:30").json()
        assert reviewed["fields"]["time"] == {"value": "09:30", "source": "manual", "status": "confirmed"}
        assert reviewed["extraction"]["review"]["time_source"] == "confirmed"
        assert reviewed["approval_current"]  # accepting never touches preparation

        (exported,) = manifest(client, job_id)["rows"]
        assert exported["time"] == "09:30" and exported["time_status"] == "reviewed_extraction"
        assert exported["extraction"]["reviewed"]["time_discrepancy"] == {"confirmed": "09:30", "model": "10:15"}


def test_model_output_corrections_and_review_are_kept_apart(tmp_path: Path):
    fake = FakeExtractor(VALID)
    with make_client(tmp_path, fake) as client:
        job_id, rows = prepared(client, pages=(1,))
        extract(client, job_id, rows[1])
        row = wait_settled(client, job_id)[1]
        assert row["extraction"]["review"] is None and row["extraction"]["state"] == "unreviewed"

        corrected = "Rx: amox 500 TID x 7 j"
        row = accept(client, job_id, row, handwriting_text=corrected).json()
        view = row["extraction"]
        assert view["state"] == "reviewed"
        assert view["attempt"]["result"]["handwriting_text"] == VALID["handwriting_text"]  # untouched
        assert view["review"]["handwriting_text"] == corrected
        assert view["review"]["corrected"] == {"time": False, "handwriting": True}
        assert view["review"]["time_source"] == "model"

        # Bad reviews are refused.
        url = f"/api/jobs/{job_id}/rows/{row['row_id']}/extraction/review"
        good = {"attempt_id": view["attempt"]["attempt_id"], "time": None,
                "handwriting_status": "legible", "handwriting_text": "x"}
        assert client.put(url, json={**good, "handwriting_status": "blank"}).status_code == 400
        assert client.put(url, json={**good, "time": "25:00"}).status_code == 400
        assert client.put(url, json={**good, "attempt_id": "att_000000000000"}).status_code == 409


def test_edits_during_an_in_flight_request_supersede_its_result(tmp_path: Path):
    fake = FakeExtractor(VALID)
    fake.gate = threading.Event()
    with make_client(tmp_path, fake) as client:
        job_id, rows = prepared(client, pages=(1,))
        extract(client, job_id, rows[1])
        assert fake.entered.wait(5)
        # While the model is "thinking", the reviewer changes a redaction in the row.
        boxes = pages_of(client, job_id)[0]["boxes"] + [{"x1": 0.6, "y1": 0.6, "x2": 0.7, "y2": 0.7}]
        save_review(client, job_id, {1: boxes})
        fake.gate.set()
        view = wait_settled(client, job_id)[1]["extraction"]
        assert view["last_attempt_status"] == "superseded"
        assert view["state"] == "not_ready"
        assert not view["attempt_is_current"]


def test_stale_results_are_redone_while_history_is_kept(tmp_path: Path):
    fake = FakeExtractor(VALID)
    with make_client(tmp_path, fake) as client:
        job_id, rows = prepared(client, pages=(1,))
        extract(client, job_id, rows[1])
        accept(client, job_id, wait_settled(client, job_id)[1])

        # New redaction in the row -> readiness invalidated -> marked ready again.
        boxes = pages_of(client, job_id)[0]["boxes"] + [{"x1": 0.6, "y1": 0.6, "x2": 0.7, "y2": 0.7}]
        save_review(client, job_id, {1: boxes})
        row = rows_of(client, job_id)[1]
        assert row["extraction"]["state"] == "not_ready"
        client.post(f"/api/jobs/{job_id}/rows/{row['row_id']}/approve", json={"revision": row["revision"]})
        row = rows_of(client, job_id)[1]
        assert row["extraction"]["state"] == "stale"

        (exported,) = manifest(client, job_id)["rows"]
        assert exported["extraction"] == {"status": "stale", "attempts": 1, "reviewed": None}
        assert exported["time"] is None

        client.post(f"/api/jobs/{job_id}/extract", json={})
        row = wait_settled(client, job_id)[1]
        assert row["extraction"]["state"] == "unreviewed"
        assert row["extraction"]["attempt_count"] == 2 and row["extraction"]["review_count"] == 1


def test_interrupted_work_recovers_as_retryable(tmp_path: Path):
    fake = FakeExtractor(VALID)
    with make_client(tmp_path, fake, texts={1: NAM_TEXT, 2: ""}) as client:
        job_id, rows = prepared(client, pages=(1,))
    # Simulate the app dying mid-request: an attempt saved as running.
    folder = tmp_path / "documents" / job_id
    record = json.loads((folder / "document.json").read_text(encoding="utf-8"))
    row_id = rows[1]["row_id"]
    record["extractions"][row_id] = {"attempts": [{
        "attempt_id": "att_aaaaaaaaaaaa", "status": "running", "requested_model": ex.DEFAULT_MODEL,
        "returned_model": None, "provider": None, "generation_id": None,
        "prompt_version": ex.PROMPT_VERSION, "schema_version": 1, "routing": {},
        "basis": {"row_revision": 0, "redaction_revision": 0, "evidence_path": "x", "evidence_sha256": "0"},
        "queued_at": "t", "started_at": "t", "finished_at": None, "tries": 1,
        "error": None, "result": None, "usage": None,
    }], "reviews": []}
    (folder / "document.json").write_text(json.dumps(record), encoding="utf-8")

    with make_client(tmp_path, fake, texts={1: NAM_TEXT, 2: ""}) as client:
        view = rows_of(client, job_id)[1]["extraction"]
        assert view["state"] == "failed" and view["last_error"]["kind"] == "interrupted"
        assert client.post(f"/api/jobs/{job_id}/extract", json={}).json()["queued"] == [row_id]
        assert wait_settled(client, job_id)[1]["extraction"]["state"] == "unreviewed"


def test_reviewed_results_survive_reopening_and_export_with_evidence(tmp_path: Path):
    fake = FakeExtractor({**VALID, "time": None, "time_evidence": None,
                          "handwriting_status": "partially_legible",
                          "handwriting_text": "suivi [?] 2 sem",
                          "unreadable_or_ambiguous": ["word after 'suivi': 'pl' or 'pb'"]})
    with make_client(tmp_path, fake) as client:
        job_id, rows = prepared(client)
        client.post(f"/api/jobs/{job_id}/extract", json={})
        settled = wait_settled(client, job_id)
        accept(client, job_id, settled[1], time="14:00")

    with make_client(tmp_path, FakeExtractor()) as client:
        rows = rows_of(client, job_id)
        assert rows[1]["extraction"]["state"] == "reviewed"
        assert rows[2]["extraction"]["state"] == "unreviewed"
        data = manifest(client, job_id)
        text = json.dumps(data)
        by_row = {r["row_id"]: r for r in data["rows"]}

        first = by_row[rows[1]["row_id"]]
        assert first["time"] == "14:00" and first["time_status"] == "reviewed_extraction"
        reviewed = first["extraction"]["reviewed"]
        assert reviewed["handwriting_text"] == "suivi [?] 2 sem"
        assert reviewed["handwriting_status"] == "partially_legible"
        assert reviewed["time_source"] == "reviewer" and reviewed["corrected"]["time"] is True
        assert reviewed["based_on"]["evidence_sha256"] == first["evidence"][0]["sha256"]
        assert reviewed["model"]["prompt_version"] == ex.PROMPT_VERSION
        frozen = client.get(f"/api/jobs/{job_id}/{first['evidence'][0]['path']}")
        assert hashlib.sha256(frozen.content).hexdigest() == first["evidence"][0]["sha256"]

        # Unreviewed output is reported by status, never exported as values.
        second = by_row[rows[2]["row_id"]]
        assert second["extraction"] == {"status": "unreviewed", "attempts": 1, "reviewed": None}
        assert second["time"] is None
        assert data["extraction_not_reviewed"] == [{"row_id": rows[2]["row_id"], "status": "unreviewed"}]
        assert text.count("suivi [?] 2 sem") == 1
        assert data["summary"]["extraction"] == {"reviewed": 1, "unreviewed": 1}

        assert_no_identifiers(data, FAKE_KEY, "OPENROUTER_API_KEY", "document.pdf")
        assert "does not establish billing eligibility" in data["notice"]



# -- reliability ------------------------------------------------------------------

#: A redaction inside a section (whole-page rows cover it).
NEW_BOX = {"x1": 0.6, "y1": 0.6, "x2": 0.75, "y2": 0.7}


def add_redaction(client: TestClient, job_id: str, page: int) -> None:
    pages = {p["page_num"]: p["boxes"] for p in pages_of(client, job_id)}
    save_review(client, job_id, {page: pages[page] + [NEW_BOX]})


def evidence_sha(row: dict) -> str:
    return next(r["sha256"] for r in row["evidence"] if r["kind"] == "row")


def test_a_queued_row_edited_before_its_turn_never_sends_its_old_crop(tmp_path: Path):
    """One worker: row 2 waits behind row 1. Row 2 gains a redaction while it
    waits; when its turn comes, its old image must not be sent."""
    fake = FakeExtractor(VALID)
    fake.gate = threading.Event()
    with make_client(tmp_path, fake, concurrency=1) as client:
        job_id, rows = prepared(client)
        old_crop = evidence_sha(rows[2])
        extract(client, job_id, rows[1])
        assert fake.entered.wait(5)              # row 1 is at the provider
        assert extract(client, job_id, rows[2]).status_code == 202  # row 2 queued
        add_redaction(client, job_id, 2)
        fake.gate.set()
        settled = wait_settled(client, job_id)

        assert len(fake.calls) == 1               # row 1 only
        assert all(hashlib.sha256(img).hexdigest() != old_crop for img, _ in fake.calls)
        view = settled[2]["extraction"]
        assert view["last_error"]["kind"] == "row_changed"
        assert view["state"] == "not_ready"       # recoverable: extract again

        # The existing workflow recovers it, with the new evidence.
        assert extract(client, job_id, settled[2]).status_code == 202
        after = wait_settled(client, job_id)
        sent = hashlib.sha256(fake.calls[-1][0]).hexdigest()
        assert sent == evidence_sha(after[2]) != old_crop
        assert after[2]["extraction"]["state"] == "unreviewed"


def test_a_row_changed_during_retry_backoff_is_not_retried_with_old_evidence(tmp_path: Path):
    fake = FakeExtractor(ex.RateLimited("slow down"), VALID)
    with make_client(tmp_path, fake) as client:
        job_id, rows = prepared(client, pages=(1,))
        store = client.app.state.store
        job = store.get(job_id)

        def change_row_while_waiting(_delay: float) -> None:
            # What the Redactions tab does, while the worker is in back-off.
            boxes = list(job.effective_boxes(1)) + [NEW_BOX]
            jobslib.save_review(job, [(1, boxes, None)])
            store.save(job)

        store._sleep = change_row_while_waiting
        extract(client, job_id, rows[1])
        view = wait_settled(client, job_id)[1]["extraction"]

        assert len(fake.calls) == 1               # the rate-limited try only
        assert view["last_error"]["kind"] == "row_changed"
        record = job.extractions[rows[1]["row_id"]]["attempts"][-1]
        assert record["status"] == "failed" and record["tries"] == 1


def test_every_unfinished_row_recovers_after_a_restart(tmp_path: Path):
    """Mixed queued and running attempts across several rows: all of them
    become interrupted, and all can be retried."""
    with make_client(tmp_path, FakeExtractor(VALID)) as client:
        job_id, rows = prepared(client)
        client.post(f"/api/jobs/{job_id}/extract", json={})
        wait_settled(client, job_id)
        # The worker saves just after the state flips: let it finish writing
        # before this test edits the file.
        client.app.state.store._extract_executor.shutdown(wait=True)

    folder = tmp_path / "documents" / job_id
    record = json.loads((folder / "document.json").read_text(encoding="utf-8"))
    ids = [rows[1]["row_id"], rows[2]["row_id"]]
    # The app died mid-batch: row 1's only attempt still queued, row 2's running.
    template = record["extractions"][ids[0]]["attempts"][0]
    record["extractions"][ids[0]]["attempts"] = [
        {**template, "attempt_id": "att_q00000000001", "status": "queued", "result": None}]
    record["extractions"][ids[1]]["attempts"] = [
        {**template, "attempt_id": "att_r00000000002", "status": "running", "result": None}]
    (folder / "document.json").write_text(json.dumps(record), encoding="utf-8")

    fake = FakeExtractor(VALID)
    with make_client(tmp_path, fake) as client:
        on_disk = json.loads((folder / "document.json").read_text(encoding="utf-8"))
        statuses = [on_disk["extractions"][i]["attempts"][-1]["status"] for i in ids]
        assert statuses == ["interrupted", "interrupted"]   # every one, persisted
        views = rows_of(client, job_id)
        for page in (1, 2):
            assert views[page]["extraction"]["state"] == "failed"
            assert views[page]["extraction"]["last_error"]["kind"] == "interrupted"

        # Both are retryable in one go, and succeed.
        batch = client.post(f"/api/jobs/{job_id}/extract", json={}).json()
        assert sorted(batch["queued"]) == sorted(ids)
        settled = wait_settled(client, job_id)
        assert {settled[p]["extraction"]["state"] for p in (1, 2)} == {"unreviewed"}
        assert len(fake.calls) == 2


def test_history_keeps_its_exact_evidence_when_a_row_is_reapproved(tmp_path: Path):
    """Extract, change a redaction inside the row (same row revision), and
    extract again: the first attempt's image survives with its hash, and the
    new attempt has its own, redacted, evidence."""
    fake = FakeExtractor(VALID)
    with make_client(tmp_path, fake) as client:
        job_id, rows = prepared(client, pages=(1,))
        extract(client, job_id, rows[1])
        wait_settled(client, job_id)
        job = client.app.state.store.get(job_id)
        first = dict(job.extractions[rows[1]["row_id"]]["attempts"][0]["basis"])

        add_redaction(client, job_id, 1)
        extract(client, job_id, rows_of(client, job_id)[1])   # re-prepares, re-freezes
        after = wait_settled(client, job_id)[1]
        second = job.extractions[rows[1]["row_id"]]["attempts"][-1]["basis"]

        assert second["row_revision"] == first["row_revision"]  # the risky case
        assert second["evidence_path"] != first["evidence_path"]
        folder = tmp_path / "documents" / job_id
        old_bytes = (folder / first["evidence_path"]).read_bytes()
        new_bytes = (folder / second["evidence_path"]).read_bytes()
        assert hashlib.sha256(old_bytes).hexdigest() == first["evidence_sha256"]
        assert hashlib.sha256(new_bytes).hexdigest() == second["evidence_sha256"]
        assert fake.calls[0][0] == old_bytes and fake.calls[1][0] == new_bytes

        # The new crop shows the new redaction; the old one does not.
        from PIL import Image
        import io
        import numpy as np

        def patch(png: bytes) -> "np.ndarray":
            img = np.asarray(Image.open(io.BytesIO(png)).convert("L"))
            h, w = img.shape
            return img[int(0.62 * h):int(0.68 * h), int(0.62 * w):int(0.73 * w)]

        assert patch(new_bytes).max() < 30 and patch(old_bytes).min() > 200
        assert after["extraction"]["state"] == "unreviewed"

    # Deleting the document still removes every evidence file.
    with make_client(tmp_path, FakeExtractor()) as client:
        assert client.delete(f"/api/jobs/{job_id}").status_code == 204
        assert not (tmp_path / "documents" / job_id).exists()


def test_a_deliberately_empty_reviewed_time_stays_empty(tmp_path: Path):
    """The model read 09:30; the reviewer clears it (wrong) and accepts."""
    with make_client(tmp_path, FakeExtractor(VALID)) as client:
        job_id, rows = prepared(client, pages=(1,))
        extract(client, job_id, rows[1])
        row = wait_settled(client, job_id)[1]
        review = accept(client, job_id, row, time=None).json()["extraction"]["review"]
        assert review["time"] is None
        assert review["corrected"]["time"] is True and review["time_source"] == "reviewer"

    with make_client(tmp_path, FakeExtractor()) as client:
        view = rows_of(client, job_id)[1]["extraction"]
        assert view["state"] == "reviewed" and view["review"]["time"] is None
        assert view["attempt"]["result"]["time"] == "09:30"   # the model's reading, kept
        (exported,) = manifest(client, job_id)["rows"]
        assert exported["time"] is None and exported["time_status"] == "reviewed_extraction"
        assert exported["extraction"]["reviewed"]["time"] is None
