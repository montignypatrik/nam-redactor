"""Appointment rows: the typed, versioned record a reviewer prepares.

A row is one appointment: a region of a working page, the patient pseudonym
it belongs to, its date and its time. It is not tied to a NAM occurrence --
a patient can have many appointments.

Each section (zone) of the document is one appointment: a *section row*
(``origin: "section"``) is created for every section and covers all of it.
Rows drawn by hand (``origin: "manual"``) remain possible through the API.

**Preparation.** The whole redacted row is the evidence a later extraction
step will read -- both the appointment time and the handwritten notes. So a
row is *ready for extraction* once its patient and date are known and its
redactions are saved (:data:`REQUIRED_FOR_PREPARATION`). Values the app
itself derives -- the pseudonym of the one NAM in the row, the date entered
for its section (:data:`TRUSTED_SOURCES`) -- count as confirmed; a reviewer
may still change them. Readiness is then set automatically
(``approval.method: "automatic"``) or by the reviewer (``"manual"``). Its time
stays in the model as an ordinary field, but it is optional here: blank means
"to be extracted", and a reviewer may still type it as an override.

Every field records where its value came from (``source``) and whether a
person has confirmed it (``status``). Marking a row ready pins a
*fingerprint* of everything the reviewer saw: the row's region, its field
values, and the redaction boxes that touch it. Any later change to one of
those moves the approval to ``invalidated``.

Rows are plain dicts (they are persisted as JSON as-is); this module is the
only place that builds, changes or migrates one.
"""
from __future__ import annotations

import copy
import datetime
import hashlib
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

#: 2: the whole row is the evidence; no separate handwriting region, and the
#: time is optional during preparation. See :func:`migrate_row` for 1 -> 2.
ROW_SCHEMA_VERSION = 2

#: Every field a row carries, in display order.
FIELDS = ("patient_id", "date", "time")

#: The fields that must be confirmed before a row is ready for extraction.
#: Time is not among them: it is read from the row later, or typed as an
#: optional override.
REQUIRED_FOR_PREPARATION = ("patient_id", "date")

MISSING = "missing"  # no value yet
SUGGESTED = "suggested"  # prefilled from elsewhere; a person has not confirmed it
CONFIRMED = "confirmed"  # entered or confirmed by the reviewer

#: Where a field value came from.
SOURCE_MANUAL = "manual"
SOURCE_PAGE_DATE = "page_date"  # the date entered for the page at the zone step
SOURCE_NAM_IN_ROW = "nam_in_row"  # the pseudonym of the one NAM box inside the row

#: Sources whose values need no separate confirmation: the app derived them
#: from data the reviewer already controls (detection results they reviewed,
#: the section date they typed).
TRUSTED_SOURCES = frozenset({SOURCE_NAM_IN_ROW, SOURCE_PAGE_DATE})

#: How a row became ready for extraction.
METHOD_MANUAL = "manual"
METHOD_AUTOMATIC = "automatic"

UNAPPROVED = "unapproved"
APPROVED = "approved"  # shown as "Ready for extraction"
INVALIDATED = "invalidated"

ROW_ID_RE = re.compile(r"^row_[0-9a-f]{12}$")

ORIGIN_SECTION = "section"  # one per section, covering it; created automatically
ORIGIN_MANUAL = "manual"  # drawn by the reviewer

#: A section row's region: the whole working page.
FULL_PAGE = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

#: The evidence a ready row must have: its full, redacted crop.
EVIDENCE_KIND = "row"

#: Human-readable reason per changed key, recorded on invalidation.
_CHANGE_REASONS = {
    "bounds": "row region changed",
    "patient_id": "patient changed",
    "date": "date changed",
    "time": "time changed",
}

MIGRATION_REASON = (
    "preparation rules changed (the whole row is now the evidence): "
    "mark ready for extraction again"
)


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def new_row_id() -> str:
    return f"row_{uuid.uuid4().hex[:12]}"


def _field(value: Any = None, source: str | None = None, status: str = MISSING) -> dict[str, Any]:
    if value is None:
        return {"value": None, "source": None, "status": MISSING}
    return {"value": value, "source": source, "status": status}


def _geometry(bounds: Mapping[str, float] | None) -> dict[str, float] | None:
    if bounds is None:
        return None
    return {key: round(float(bounds[key]), 6) for key in ("x1", "y1", "x2", "y2")}


def _unapproved() -> dict[str, Any]:
    return {
        "status": UNAPPROVED,
        "approved_revision": None,
        "approved_at": None,
        "redaction_revision": None,
        "fingerprint": None,
        "invalidated_at": None,
        "invalidated_reasons": [],
    }


def new_row(
    document_id: str,
    working_page: int,
    source_page: int,
    bounds: Mapping[str, float],
    *,
    date_suggestion: str | None = None,
    patient_suggestion: str | None = None,
    origin: str = ORIGIN_MANUAL,
) -> dict[str, Any]:
    """A fresh, unapproved row. Suggestions are prefilled but unconfirmed."""
    stamp = now_iso()
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "row_id": new_row_id(),
        "origin": origin,
        "document_id": document_id,
        "working_page": working_page,
        "source_page": source_page,
        "bounds": _geometry(bounds),
        "fields": {
            "patient_id": _field(patient_suggestion, SOURCE_NAM_IN_ROW, CONFIRMED),
            "date": _field(date_suggestion, SOURCE_PAGE_DATE, CONFIRMED),
            "time": _field(),
        },
        "evidence": [],
        "revision": 1,
        "created_at": stamp,
        "updated_at": stamp,
        "approval": _unapproved(),
    }


def migrate_row(row: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Bring a saved row to the current schema. Returns ``(row, changed)``.

    Version 1 rows had a separate handwriting region, handwriting evidence,
    and required a time. Nothing is thrown away: the handwriting region, the
    evidence references (their files stay on disk) and the old approval move
    under ``legacy``. A v1 approval vouched for a different contract, so it
    becomes ``invalidated`` with :data:`MIGRATION_REASON` -- the row asks to
    be marked ready again rather than silently passing, or silently dropping.
    Confirmed patient, date and time values are kept as they were.
    """
    version = row.get("schema_version", 1)
    if version >= ROW_SCHEMA_VERSION:
        return dict(row), False
    updated = copy.deepcopy(dict(row))
    legacy = {
        "migrated_from_schema": version,
        "migrated_at": now_iso(),
        "handwriting_bounds": updated.pop("handwriting_bounds", None),
        "evidence": updated.get("evidence", []),
        "approval": copy.deepcopy(updated.get("approval")),
    }
    updated["legacy"] = legacy
    updated["evidence"] = []
    updated.setdefault("origin", ORIGIN_MANUAL)
    updated["schema_version"] = ROW_SCHEMA_VERSION
    # A new revision, so new evidence files never reuse a legacy file name.
    updated["revision"] = int(updated.get("revision", 1)) + 1
    updated["updated_at"] = now_iso()
    approval = updated.get("approval") or _unapproved()
    if approval.get("status") in (APPROVED, INVALIDATED):
        reasons = [MIGRATION_REASON]
        if approval.get("status") == INVALIDATED:
            reasons += approval.get("invalidated_reasons") or []
        updated["approval"] = {
            **_unapproved(),
            "status": INVALIDATED,
            "invalidated_at": now_iso(),
            "invalidated_reasons": reasons,
        }
    else:
        updated["approval"] = _unapproved()
    return updated, True


def unresolved(row: Mapping[str, Any]) -> list[str]:
    """What still stands between this row and being ready for extraction.

    Only :data:`REQUIRED_FOR_PREPARATION` counts: a blank time is expected.
    """
    return [name for name in REQUIRED_FOR_PREPARATION if row["fields"][name]["status"] != CONFIRMED]


def _overlaps(a: Mapping[str, float], b: Mapping[str, float]) -> bool:
    return a["x1"] < b["x2"] and b["x1"] < a["x2"] and a["y1"] < b["y2"] and b["y1"] < a["y2"]


def relevant_boxes(
    row: Mapping[str, Any], page_boxes: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Redaction boxes that touch the row -- the ones its evidence crop shows."""
    out = []
    for box in page_boxes:
        if _overlaps(box, row["bounds"]):
            out.append({
                "x1": round(float(box["x1"]), 6), "y1": round(float(box["y1"]), 6),
                "x2": round(float(box["x2"]), 6), "y2": round(float(box["y2"]), 6),
                "patient_id": box.get("patient_id"),
                "kind": box.get("kind"),
            })
    return sorted(out, key=lambda b: (b["y1"], b["x1"], b["y2"], b["x2"]))


def fingerprint(row: Mapping[str, Any], page_boxes: Sequence[Mapping[str, Any]]) -> str:
    """Hash of everything a readiness approval vouches for.

    The time value is included: blank stays blank, and a typed override
    changing afterwards is a change to what was reviewed.
    """
    payload = {
        "schema_version": ROW_SCHEMA_VERSION,
        "working_page": row["working_page"],
        "bounds": row["bounds"],
        "values": {name: row["fields"][name]["value"] for name in FIELDS},
        "redactions": relevant_boxes(row, page_boxes),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _invalidate(row: dict[str, Any], reasons: Sequence[str]) -> None:
    approval = row["approval"]
    if approval["status"] != APPROVED:
        return
    approval["status"] = INVALIDATED
    approval["invalidated_at"] = now_iso()
    approval["invalidated_reasons"] = list(dict.fromkeys(reasons))


def apply_patch(row: Mapping[str, Any], patch: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return an updated copy of ``row`` and the keys that actually changed.

    ``patch`` is already validated by the caller:

    * ``bounds`` -- new row geometry (hand-drawn rows only).
    * ``patient_id`` / ``date`` / ``time`` -- a new value, entered by the
      reviewer and therefore confirmed. ``None`` clears the field back to
      missing -- for the time, "to be extracted".
    * ``no_patient`` -- the reviewer states the row has no patient (a call
      with no health number, say): the patient is confirmed, with no value.
    * ``confirm`` -- field names whose current suggested value the reviewer
      accepts as-is.

    Any real change bumps the revision and invalidates an approval.
    """
    updated = copy.deepcopy(dict(row))
    changed: list[str] = []

    if "bounds" in patch:
        value = _geometry(patch["bounds"])
        if value is not None and value != updated.get("bounds"):
            updated["bounds"] = value
            changed.append("bounds")

    for name in FIELDS:
        if name not in patch:
            continue
        value = patch[name]
        current = updated["fields"][name]
        new = _field(value, SOURCE_MANUAL, CONFIRMED)
        if new != current:
            if new["value"] != current["value"]:
                changed.append(name)
            updated["fields"][name] = new

    status_only = False
    if patch.get("no_patient"):
        # "No patient", stated by the reviewer: confirmed, with no value.
        current = updated["fields"]["patient_id"]
        new = {"value": None, "source": SOURCE_MANUAL, "status": CONFIRMED}
        if new != current:
            if current["value"] is not None:
                changed.append("patient_id")
            else:
                status_only = True
            updated["fields"]["patient_id"] = new

    for name in patch.get("confirm", ()):
        current = updated["fields"][name]
        if current["status"] == SUGGESTED:
            current["status"] = CONFIRMED
            status_only = True

    if changed or status_only:
        updated["revision"] += 1
        updated["updated_at"] = now_iso()
    if changed:
        _invalidate(updated, [_CHANGE_REASONS[key] for key in changed])
    return updated, changed


def approve(
    row: Mapping[str, Any],
    page_boxes: Sequence[Mapping[str, Any]],
    redaction_revision: int,
    evidence: Sequence[Mapping[str, Any]],
    method: str = METHOD_MANUAL,
) -> dict[str, Any]:
    """Mark the row ready for extraction as it stands.

    ``method`` records whether a reviewer did this (``manual``) or the app
    did it for a row whose values are all known (``automatic``).

    Raises ``ValueError`` if a required field is unresolved or the full-row
    evidence crop is missing.
    """
    missing = unresolved(row)
    if missing:
        raise ValueError(f"Unresolved: {', '.join(missing)}")
    if not any(ref.get("kind") == EVIDENCE_KIND for ref in evidence):
        raise ValueError("The full-row evidence crop is missing.")
    updated = copy.deepcopy(dict(row))
    updated["evidence"] = [dict(ref) for ref in evidence]
    updated["approval"] = {
        "status": APPROVED,
        "method": method,
        "approved_revision": updated["revision"],
        "approved_at": now_iso(),
        "redaction_revision": redaction_revision,
        "fingerprint": fingerprint(updated, page_boxes),
        "invalidated_at": None,
        "invalidated_reasons": [],
    }
    return updated


def revalidate(
    row: Mapping[str, Any], page_boxes: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], bool]:
    """After the saved redactions change: invalidate an approval whose
    evidence no longer matches. Returns ``(row, invalidated)``."""
    if row["approval"]["status"] != APPROVED:
        return dict(row), False
    if fingerprint(row, page_boxes) == row["approval"]["fingerprint"]:
        return dict(row), False
    updated = copy.deepcopy(dict(row))
    _invalidate(updated, ["redactions in the row changed"])
    return updated, True


def is_current(row: Mapping[str, Any], page_boxes: Sequence[Mapping[str, Any]]) -> bool:
    """Ready for extraction, and nothing it vouches for has changed since."""
    approval = row["approval"]
    return (
        approval["status"] == APPROVED
        and approval["approved_revision"] == row["revision"]
        and approval["fingerprint"] == fingerprint(row, page_boxes)
    )


def suggest_patient(
    bounds: Mapping[str, float], page_boxes: Sequence[Mapping[str, Any]]
) -> str | None:
    """The pseudonym of the NAM boxes centred inside ``bounds``, when there
    is exactly one. Two different patients in one row suggest nothing."""
    found = set()
    for box in page_boxes:
        patient_id = box.get("patient_id")
        if not patient_id:
            continue
        cx = (box["x1"] + box["x2"]) / 2
        cy = (box["y1"] + box["y2"]) / 2
        if bounds["x1"] <= cx <= bounds["x2"] and bounds["y1"] <= cy <= bounds["y2"]:
            found.add(patient_id)
    return found.pop() if len(found) == 1 else None


def follow_page_date(row: Mapping[str, Any], page_date: str | None) -> dict[str, Any] | None:
    """The row with its date moved to a changed section date, or ``None``
    when nothing should change.

    A date taken from the section follows it; a date the reviewer typed for
    this row is theirs and stays. A row that was ready becomes invalidated:
    what it vouched for changed.
    """
    current = row["fields"]["date"]
    if current["source"] != SOURCE_PAGE_DATE:
        return None
    if page_date is None or current["value"] == page_date:
        return None
    updated = copy.deepcopy(dict(row))
    updated["fields"]["date"] = _field(page_date, SOURCE_PAGE_DATE, current["status"])
    updated["revision"] += 1
    updated["updated_at"] = now_iso()
    _invalidate(updated, ["section date changed"])
    return updated


def trust_app_values(row: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Confirm values still *suggested* from a trusted source.

    Rows created before app-derived values were trusted carry them as
    suggestions; this brings them in line. Returns ``(row, changed)``.
    """
    updated = copy.deepcopy(dict(row))
    changed = False
    for name in FIELDS:
        field = updated["fields"][name]
        if field["status"] == SUGGESTED and field["source"] in TRUSTED_SOURCES:
            field["status"] = CONFIRMED
            changed = True
    if changed:
        updated["revision"] += 1
        updated["updated_at"] = now_iso()
    return updated, changed
