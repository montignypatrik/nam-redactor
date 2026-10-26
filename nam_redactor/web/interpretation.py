"""Interpretation records: proposed billing acts, and what they were based on.

The second stage after transcription. Stored per row in
``job.interpretations[row_id] = {"attempts": [...]}``, apart from the
transcription attempts and their reviews, so neither stage can overwrite the
other.

Every attempt records its **inputs** (:func:`inputs_of`): the evidence image
(path and SHA-256, via the transcription's basis), the transcription attempt
and review it read -- with the exact values, including a deliberately empty
corrected time -- and the doctor assignment (profile, version, assignment).
A result is *current* only while all of those still hold. When any changes --
a new transcription or correction, a different doctor or an adopted version,
new evidence -- the result is **stale** and is labelled outdated.

States, derived each time, never stored:

``not_ready``  no current transcription to interpret
``blocked``    no doctor profile, or the assigned version has neither
               instructions nor examples: nothing to interpret *with*
``pending``    ready, never interpreted
``running``    queued or in flight
``failed``     the latest attempt failed or was interrupted (retryable)
``stale``      results exist, but for inputs that have since changed
``proposed``   a current, **unvalidated** proposal

There is no approved state: a proposal is validated later, by a person.
"""
from __future__ import annotations

import copy
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from .. import extraction as ex
from . import extraction as exlib
from . import rows as rowlib

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
INTERRUPTED = "interrupted"
SUPERSEDED = "superseded"
IN_FLIGHT = frozenset({QUEUED, RUNNING})

#: States worth a new attempt when "Extract all rows" runs.
RETRYABLE_STATES = frozenset({"pending", "failed", "stale"})

BLOCKED_NO_DOCTOR = "no_doctor"
BLOCKED_EMPTY_PROFILE = "empty_profile"
BLOCKED_PROFILE_MISSING = "profile_missing"

BLOCKED_MESSAGES = {
    BLOCKED_NO_DOCTOR: "No doctor profile is assigned to this document. Assign one to interpret billing.",
    BLOCKED_EMPTY_PROFILE: (
        "The assigned doctor profile version has no instructions or examples, so there is nothing to "
        "interpret with. Add them in Doctor profiles, save, and adopt the new version."
    ),
    BLOCKED_PROFILE_MISSING: "The assigned doctor profile version could not be found.",
}


def empty_record() -> dict[str, Any]:
    return {"attempts": []}


def transcription_inputs(
    row: Mapping[str, Any], record: Mapping[str, Any] | None, page_boxes: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    """The current transcription to interpret, or ``None``.

    A saved correction wins over the model's reading, value for value --
    including a time deliberately left empty.
    """
    attempt = exlib.current_attempt(row, record, page_boxes)
    if attempt is None:
        return None
    review = exlib.current_review(row, record, page_boxes)
    values = review if review is not None else attempt["result"]
    return {
        "evidence": dict(attempt["basis"]),
        "transcription_attempt_id": attempt["attempt_id"],
        "review_id": review["review_id"] if review is not None else None,
        "source": "reviewer" if review is not None else "model",
        "time": values["time"],
        "handwriting_status": values["handwriting_status"],
        "handwriting_text": values["handwriting_text"],
    }


def inputs_of(
    row: Mapping[str, Any],
    transcription_record: Mapping[str, Any] | None,
    page_boxes: Sequence[Mapping[str, Any]],
    doctor: Mapping[str, Any] | None,
    profile_version: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """``(inputs, blocked_reason)`` for interpreting this row now.

    ``inputs`` is ``None`` when there is nothing to interpret yet, or when
    interpretation is blocked (then ``blocked_reason`` says why).
    ``profile_version`` is the *assigned* version, never the latest.
    """
    transcription = transcription_inputs(row, transcription_record, page_boxes)
    if transcription is None:
        return None, None
    if doctor is None:
        return None, BLOCKED_NO_DOCTOR
    if profile_version is None:
        return None, BLOCKED_PROFILE_MISSING
    if not profile_version["instructions"].strip() and not profile_version["examples"]:
        return None, BLOCKED_EMPTY_PROFILE
    return {
        **transcription,
        "doctor": {
            "profile_id": doctor["profile_id"],
            "version": doctor["version"],
            "assignment_id": doctor["assignment_id"],
        },
    }, None


def request_for(inputs: Mapping[str, Any], profile_version: Mapping[str, Any]) -> ex.InterpretationRequest:
    return ex.InterpretationRequest(
        instructions=profile_version["instructions"],
        examples=tuple(dict(e) for e in profile_version["examples"]),
        transcription_source=inputs["source"],
        time=inputs["time"],
        handwriting_status=inputs["handwriting_status"],
        handwriting_text=inputs["handwriting_text"],
    )


def new_attempt(inputs: Mapping[str, Any], model: str) -> dict[str, Any]:
    return {
        "attempt_id": f"int_{uuid.uuid4().hex[:12]}",
        "status": QUEUED,
        "requested_model": model,
        "returned_model": None,
        "provider": None,
        "generation_id": None,
        "prompt_version": ex.INTERPRET_PROMPT_VERSION,
        "schema_version": ex.INTERPRET_SCHEMA_VERSION,
        "routing": dict(ex.PROVIDER_ROUTING),
        "inputs": copy.deepcopy(dict(inputs)),
        "queued_at": rowlib.now_iso(),
        "started_at": None,
        "finished_at": None,
        "tries": 0,
        "error": None,
        "result": None,
        "usage": None,
        # A proposal only. Validation is a later, human step: nothing here
        # ever marks a result approved.
        "validated": False,
    }


def interrupt_in_flight(record: dict[str, Any]) -> bool:
    changed = False
    for attempt in record.get("attempts", []):
        if attempt["status"] in IN_FLIGHT:
            attempt["status"] = INTERRUPTED
            attempt["finished_at"] = rowlib.now_iso()
            attempt["error"] = {"kind": "interrupted", "message": "The app stopped before this finished."}
            changed = True
    return changed


def current_attempt(record: Mapping[str, Any] | None, inputs: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if record is None or inputs is None:
        return None
    for attempt in reversed(record.get("attempts", [])):
        if attempt["status"] == SUCCEEDED and attempt["inputs"] == inputs:
            return attempt
    return None


def state_of(
    record: Mapping[str, Any] | None, inputs: Mapping[str, Any] | None, blocked: str | None
) -> str:
    attempts = (record or {}).get("attempts", [])
    if attempts and attempts[-1]["status"] in IN_FLIGHT:
        return "running"
    if blocked:
        return "blocked"
    if inputs is None:
        return "not_ready"
    if current_attempt(record, inputs) is not None:
        return "proposed"
    if any(a["status"] == SUCCEEDED for a in attempts):
        return "stale"
    if attempts and attempts[-1]["status"] in (FAILED, INTERRUPTED, SUPERSEDED):
        return "failed"
    return "pending"


def view(record: Mapping[str, Any] | None, inputs: Mapping[str, Any] | None, blocked: str | None) -> dict[str, Any]:
    """What the results screen needs about a row's interpretation."""
    record = record or empty_record()
    attempts = record.get("attempts", [])
    current = current_attempt(record, inputs)
    latest_success = next((a for a in reversed(attempts) if a["status"] == SUCCEEDED), None)
    shown = current or latest_success
    last = attempts[-1] if attempts else None
    return {
        "state": state_of(record, inputs, blocked),
        "blocked_reason": blocked,
        "blocked_message": BLOCKED_MESSAGES.get(blocked) if blocked else None,
        # The result shown: the current one, else the latest -- then marked
        # outdated (``is_current`` false) so it cannot pass for current.
        "attempt": copy.deepcopy(shown),
        "is_current": current is not None,
        "last_error": copy.deepcopy(last["error"]) if last and last["status"] in (FAILED, INTERRUPTED) else None,
        "last_attempt_status": last["status"] if last else None,
        "attempt_count": len(attempts),
        "validated": False,
    }
