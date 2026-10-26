"""Extraction records: attempts, reviews, and the state derived from them.

Each row's extraction lives in ``job.extractions[row_id]``, apart from the
row itself, so a reviewer editing a row can never clobber a result a worker
thread is writing. A record holds two append-only lists:

``attempts``
    Every model call: requested and returned model, provider, prompt and
    schema version, the *basis* it ran on (row revision, redaction revision,
    evidence file and SHA-256), its raw validated result or error, token
    usage and cost as reported (``None`` when not), and timestamps.
``reviews``
    Every accepted review: the values the reviewer accepted, which attempt
    they reviewed, and what they corrected. Earlier reviews are kept when a
    row is re-extracted and reviewed again.

Nothing here is stored as a status. The state shown to the reviewer is
*derived* each time from the attempts, the reviews and the row's current
readiness (:func:`state_of`), so it can never drift out of date:

``not_ready``   the row is not (or no longer) ready for extraction
``pending``     ready, never extracted
``running``     an attempt is queued or in flight
``failed``      the latest attempt failed or was interrupted (retryable)
``stale``       results exist, but for evidence that has since changed
``unreviewed``  a current result awaits review
``reviewed``    a current result was reviewed and accepted
"""
from __future__ import annotations

import copy
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from .. import extraction as ex
from ..locator import canonical_from_text
from . import rows as rowlib

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
INTERRUPTED = "interrupted"
#: Finished, but the row changed meanwhile: kept, never shown as current.
SUPERSEDED = "superseded"

IN_FLIGHT = frozenset({QUEUED, RUNNING})

#: A retry needs a fresh attempt for these; successes are left alone.
RETRYABLE_STATES = frozenset({"pending", "failed", "stale"})

#: Where an accepted time came from.
TIME_FROM_CONFIRMED = "confirmed"  # the time a reviewer typed during preparation
TIME_FROM_MODEL = "model"
TIME_FROM_REVIEWER = "reviewer"


def empty_record() -> dict[str, Any]:
    return {"attempts": [], "reviews": []}


def basis_of(row: Mapping[str, Any], page_boxes: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """What an extraction of this row must be based on, or ``None`` when the
    row is not currently ready: the frozen full-row evidence of its approval."""
    if not rowlib.is_current(row, page_boxes):
        return None
    ref = next((r for r in row.get("evidence", []) if r.get("kind") == rowlib.EVIDENCE_KIND), None)
    if ref is None:
        return None
    approval = row["approval"]
    return {
        "row_revision": approval["approved_revision"],
        "redaction_revision": approval["redaction_revision"],
        "evidence_path": ref["path"],
        "evidence_sha256": ref["sha256"],
    }


def new_attempt(basis: Mapping[str, Any], model: str) -> dict[str, Any]:
    return {
        "attempt_id": f"att_{uuid.uuid4().hex[:12]}",
        "status": QUEUED,
        "requested_model": model,
        "returned_model": None,
        "provider": None,
        "generation_id": None,
        "prompt_version": ex.PROMPT_VERSION,
        "schema_version": ex.SCHEMA_VERSION,
        "routing": dict(ex.PROVIDER_ROUTING),
        "basis": dict(basis),
        "queued_at": rowlib.now_iso(),
        "started_at": None,
        "finished_at": None,
        "tries": 0,
        "error": None,
        "result": None,
        "usage": None,
    }


def interrupt_in_flight(record: dict[str, Any]) -> bool:
    """After a restart nothing is in flight any more: mark it retryable."""
    changed = False
    for attempt in record.get("attempts", []):
        if attempt["status"] in IN_FLIGHT:
            attempt["status"] = INTERRUPTED
            attempt["finished_at"] = rowlib.now_iso()
            attempt["error"] = {"kind": "interrupted", "message": "The app stopped before this finished."}
            changed = True
    return changed


def _current_success(record: Mapping[str, Any], basis: Mapping[str, Any]) -> dict[str, Any] | None:
    for attempt in reversed(record.get("attempts", [])):
        if attempt["status"] == SUCCEEDED and attempt["basis"] == basis:
            return attempt
    return None


def _current_review(record: Mapping[str, Any], attempt: Mapping[str, Any]) -> dict[str, Any] | None:
    for review in reversed(record.get("reviews", [])):
        if review["attempt_id"] == attempt["attempt_id"] and review["basis"] == attempt["basis"]:
            return review
    return None


def state_of(
    row: Mapping[str, Any], record: Mapping[str, Any] | None, page_boxes: Sequence[Mapping[str, Any]]
) -> str:
    record = record or empty_record()
    attempts = record.get("attempts", [])
    if attempts and attempts[-1]["status"] in IN_FLIGHT:
        return "running"
    basis = basis_of(row, page_boxes)
    if basis is None:
        return "not_ready"
    success = _current_success(record, basis)
    if success is not None:
        return "reviewed" if _current_review(record, success) else "unreviewed"
    if any(a["status"] == SUCCEEDED for a in attempts):
        return "stale"
    if attempts and attempts[-1]["status"] in (FAILED, INTERRUPTED, SUPERSEDED):
        return "failed"
    return "pending"


def confirmed_time(row: Mapping[str, Any]) -> str | None:
    field = row["fields"]["time"]
    return field["value"] if field["status"] == rowlib.CONFIRMED else None


def view(
    row: Mapping[str, Any], record: Mapping[str, Any] | None, page_boxes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """What the review screen needs about a row's extraction."""
    record = record or empty_record()
    attempts = record.get("attempts", [])
    state = state_of(row, record, page_boxes)
    basis = basis_of(row, page_boxes)
    success = _current_success(record, basis) if basis else None
    latest_success = next((a for a in reversed(attempts) if a["status"] == SUCCEEDED), None)
    shown = success or latest_success
    review = _current_review(record, success) if success else None
    last = attempts[-1] if attempts else None
    confirmed = confirmed_time(row)
    proposed = shown["result"]["time"] if shown else None
    text = (review or {}).get("handwriting_text") or (shown["result"]["handwriting_text"] if shown else None)
    return {
        "state": state,
        "attempt": copy.deepcopy(shown),
        "attempt_is_current": success is not None,
        "review": copy.deepcopy(review),
        "last_error": copy.deepcopy(last["error"]) if last and last["status"] in (FAILED, INTERRUPTED) else None,
        "last_attempt_status": last["status"] if last else None,
        "attempt_count": len(attempts),
        "review_count": len(record.get("reviews", [])),
        "confirmed_time": confirmed,
        # A time typed during preparation is kept; a different model reading
        # is surfaced for the reviewer instead of replacing it.
        "time_discrepancy": (
            {"confirmed": confirmed, "proposed": proposed}
            if confirmed and proposed and confirmed != proposed else None
        ),
        # Handwriting is not redacted by detection: flag text that contains
        # something shaped like a health number before it is exported.
        "possible_nam_in_text": bool(text and canonical_from_text(text)),
    }


def validate_review(
    body: Mapping[str, Any],
) -> tuple[str | None, str, str | None]:
    """``(time, handwriting_status, handwriting_text)`` from a review request.

    Raises ``ValueError`` with a message safe to return.
    """
    time = body.get("time")
    if time is not None and not (isinstance(time, str) and ex.TIME_RE.match(time)):
        raise ValueError("Invalid time (use HH:MM, or leave it empty if unknown).")
    status = body.get("handwriting_status")
    if status not in ex.HANDWRITING_STATUSES:
        raise ValueError("Choose a handwriting status.")
    text = body.get("handwriting_text")
    if text is not None and not isinstance(text, str):
        raise ValueError("Invalid handwriting text.")
    text = text if text and text.strip() else None
    if status in ("blank", "unreadable"):
        if text is not None:
            raise ValueError(f"Handwriting marked {status} cannot have text.")
    elif text is None:
        raise ValueError("Readable handwriting needs its text.")
    return time, status, text


def accept_review(
    row: Mapping[str, Any],
    record: dict[str, Any],
    attempt: Mapping[str, Any],
    time: str | None,
    status: str,
    text: str | None,
) -> dict[str, Any]:
    """Append an accepted review of ``attempt``; returns it."""
    result = attempt["result"]
    confirmed = confirmed_time(row)
    if time is not None and time == confirmed:
        source = TIME_FROM_CONFIRMED
    elif time == result["time"]:
        source = TIME_FROM_MODEL
    else:
        source = TIME_FROM_REVIEWER
    review = {
        "review_id": f"rev_{uuid.uuid4().hex[:12]}",
        "attempt_id": attempt["attempt_id"],
        "basis": dict(attempt["basis"]),
        "time": time,
        "time_source": source,
        "handwriting_status": status,
        "handwriting_text": text,
        "corrected": {
            "time": time != result["time"],
            "handwriting": (status, text) != (result["handwriting_status"], result["handwriting_text"]),
        },
        "reviewed_at": rowlib.now_iso(),
    }
    record.setdefault("reviews", []).append(review)
    return review


def current_attempt(
    row: Mapping[str, Any], record: Mapping[str, Any] | None, page_boxes: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    basis = basis_of(row, page_boxes)
    return _current_success(record or empty_record(), basis) if basis else None


def current_review(
    row: Mapping[str, Any], record: Mapping[str, Any] | None, page_boxes: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    attempt = current_attempt(row, record, page_boxes)
    return _current_review(record or empty_record(), attempt) if attempt else None
