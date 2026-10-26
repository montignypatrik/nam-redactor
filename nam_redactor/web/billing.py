"""Billing lines per appointment: manual edits and billing confirmation.

The model's interpretation (:mod:`.interpretation`) proposes acts. A person
may then correct them -- change a code, units or contexts, add a missing act,
remove a wrong one, or state that the appointment has no billing acts -- and
**confirm** the appointment. That is a billing confirmation, separate from
readiness for extraction and from accepting a transcription.

Stored per row in ``job.billing[row_id]``, apart from the model's output
(which is never edited), as append-only history::

    {"edits": [...], "confirmations": [...]}

* An **edit** is the full set of lines the person saved (``kind: "lines"``),
  or a return to the model's proposal (``kind: "reset"``). Each records its
  **basis**: the interpretation attempt it started from and every input that
  determines the lines' shared values (evidence, transcription, doctor
  assignment and version, patient, date, time).
* A **confirmation** entry is ``action: "confirm"`` (pointing at the edit it
  confirmed) or ``action: "reopen"``.

Everything else is derived, never stored:

* The **effective lines** are the latest manual edit's, when there is one,
  else the current model proposal's. A model result arriving later never
  replaces manual lines: if the basis changed, the edit is marked outdated
  and the row needs attention.
* ``confirmed``: the latest confirmation entry is a confirm, of the latest
  edit, and that edit's basis still holds. Any later edit ends it (the
  person confirms again); a changed basis makes it ``outdated``.

Nothing here calls a model. The billing CSV (:mod:`.billing_csv`) exports
exactly the effective lines this module shows.
"""
from __future__ import annotations

import copy
import math
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from . import interpretation as ilib
from . import rows as rowlib

#: The practice's billing columns, in order.
HEADER = (
    "doctorLicenceID", "doctorGroupNumber", "ClaimDate", "ProcedureTime", "HIN",
    "PatientGender", "PatientFirstName", "PatientLastName", "PatientBirthdate",
    "PatientParentHIN", "Sector", "Diagnosis", "ProcedureCode", "Units",
    "ContextualElements", "referringDoctor", "CSSTAccidentDate",
)

#: Columns never filled.
UNUSED = frozenset({
    "PatientGender", "PatientFirstName", "PatientLastName", "PatientBirthdate",
    "PatientParentHIN", "Diagnosis", "referringDoctor", "CSSTAccidentDate",
})

#: Codes billed without a patient health number (15622: telephone call).
#: An appointment needs no patient when every one of its lines is one of these.
NO_HIN_CODES = frozenset({"15622"})

#: The only columns a person edits, per line. The rest belong to the appointment.
EDITABLE = ("ProcedureCode", "Units", "ContextualElements")

DRAFT = "draft"
CONFIRMED = "confirmed"
OUTDATED = "outdated"

MAX_ACTS = 50
MAX_CODE = 32
MAX_CONTEXTS = 20
MAX_CONTEXT = 32
MAX_UNITS = 100_000
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

STATE_REASONS = {
    "not_ready": "not transcribed yet (or the row changed since)",
    "blocked": "billing cannot be proposed for this document",
    "pending": "billing not proposed yet",
    "running": "still being processed",
    "failed": "the last interpretation failed",
    "stale": "the proposal is outdated",
}
EDITS_OUTDATED = "your billing edits are outdated (the source changed): review and save them again"


def empty_record() -> dict[str, Any]:
    return {"edits": [], "confirmations": []}


# -- input validation ---------------------------------------------------


def _text(value: Any, what: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{what} must be text.")
    value = value.strip()
    if len(value) > limit or _CONTROL.search(value):
        raise ValueError(f"Invalid {what}.")
    return value


def _units_value(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Units must be a number or empty.")
    if not math.isfinite(value) or value < 0 or value > MAX_UNITS:
        raise ValueError("Units must be a number between 0 and 100000, or empty.")
    return int(value) if float(value).is_integer() else float(value)


def parse_lines(body: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """``(acts, no_billing)`` from a request, as typed: codes keep their
    spelling and case (only surrounding spaces go), blank units stay
    ``None``, contexts keep their order. An empty code is allowed in a draft
    -- it is an unfinished line -- but not in a confirmation or an export."""
    raw = body.get("acts")
    no_billing = body.get("no_billing", False)
    if not isinstance(raw, list) or len(raw) > MAX_ACTS or not isinstance(no_billing, bool):
        raise ValueError("Invalid billing lines.")
    acts = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Invalid billing line.")
        contexts = item.get("context_elements", [])
        if not isinstance(contexts, list) or len(contexts) > MAX_CONTEXTS:
            raise ValueError("Invalid contextual elements.")
        cleaned = [_text(c, "contextual element", MAX_CONTEXT) for c in contexts]
        if any(not c or "," in c for c in cleaned):
            raise ValueError("A contextual element cannot be empty or contain a comma.")
        model_index = item.get("model_index")
        if model_index is not None and (isinstance(model_index, bool) or not isinstance(model_index, int)):
            raise ValueError("Invalid billing line.")
        acts.append({
            "code": _text(item.get("code") or "", "procedure code", MAX_CODE),
            "units": _units_value(item.get("units")),
            "context_elements": cleaned,
            # Which proposed act this line started from; None for a line added by hand.
            "model_index": model_index,
        })
    if no_billing and acts:
        raise ValueError("An appointment marked as having no billing acts cannot have billing lines.")
    return acts, no_billing


# -- derived state ------------------------------------------------------


def procedure_time(row: Mapping[str, Any], used: Mapping[str, Any] | None) -> str:
    """The time the billing is based on.

    A saved transcription correction is taken as it is -- an empty
    corrected time stays empty. Otherwise the model's reading; if it read
    none, a time the reviewer typed during preparation. Never a default.
    """
    if used is not None:
        if used["source"] == "reviewer":
            return used["time"] or ""
        if used["time"]:
            return used["time"]
    typed = row["fields"]["time"]
    return typed["value"] if typed["status"] == rowlib.CONFIRMED and typed["value"] else ""


def _confirmed_value(row: Mapping[str, Any], name: str) -> str | None:
    field = row["fields"][name]
    return field["value"] if field["status"] == rowlib.CONFIRMED and field["value"] else None


def shared_values(
    job: Any, row: Mapping[str, Any], used: Mapping[str, Any] | None, nam_of: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Every appointment-level column, as exported. ``HIN`` is resolved on
    this machine from the private identity map."""
    if nam_of is None:
        nam_of = {pseudonym: nam for nam, pseudonym in job.patient_ids.items()}
    snapshot = (job.doctor or {}).get("snapshot") or {}
    patient = _confirmed_value(row, "patient_id")
    values = {column: "" for column in HEADER if column not in EDITABLE}
    values.update({
        "doctorLicenceID": snapshot.get("licence_number") or "",
        "doctorGroupNumber": snapshot.get("group_number") or "",
        "ClaimDate": _confirmed_value(row, "date") or "",
        "ProcedureTime": procedure_time(row, used),
        "HIN": (nam_of.get(patient) if patient else None) or "",
        "Sector": snapshot.get("sector_number") or "",
    })
    return values


def _proposal_lines(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "code": act["code"] or "",
            "units": _units_value(act["units"]) if act["units"] is not None else None,
            "context_elements": list(act["context_elements"]),
            "model_index": index,
        }
        for index, act in enumerate(result["acts"])
    ]


def units_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def compute(job: Any, row: Mapping[str, Any], doctor: Mapping[str, Any] | None,
            version: Mapping[str, Any] | None, nam_of: Mapping[str, str] | None = None) -> dict[str, Any]:
    """A row's billing as shown and exported. Call without holding locks
    other than (re-entrantly) ``job.extract_lock``."""
    row_id = row["row_id"]
    page_boxes = job.effective_boxes(row["working_page"])
    with job.extract_lock:
        transcription = ilib.transcription_inputs(row, job.extractions.get(row_id), page_boxes)
        inputs, blocked = ilib.inputs_of(row, job.extractions.get(row_id), page_boxes, doctor, version)
        interp = job.interpretations.get(row_id)
        state = ilib.state_of(interp, inputs, blocked)
        attempt = ilib.current_attempt(interp, inputs)
        record = copy.deepcopy(job.billing.get(row_id) or empty_record())

    shared = shared_values(job, row, transcription, nam_of)
    basis = {
        "interpretation_attempt_id": attempt["attempt_id"] if attempt else None,
        "interpretation_inputs": copy.deepcopy(inputs),
        "transcription": copy.deepcopy(transcription),
        "doctor": None if job.doctor is None else {
            k: job.doctor[k] for k in ("profile_id", "version", "assignment_id")
        },
        "shared": {k: shared[k] for k in ("doctorLicenceID", "doctorGroupNumber", "Sector",
                                           "ClaimDate", "ProcedureTime", "HIN")},
    }

    edits = record["edits"]
    latest = edits[-1] if edits else None
    manual = latest if latest is not None and latest["kind"] == "lines" else None
    edits_current = manual is not None and manual["basis"] == basis
    result = attempt["result"] if attempt else None

    if manual is not None:
        source, lines, no_billing = "manual", copy.deepcopy(manual["acts"]), manual["no_billing"]
    elif result is not None:
        source = "model"
        lines = _proposal_lines(result)
        no_billing = bool(result["no_billing_notation"]) and not result["acts"]
    else:
        source, lines, no_billing = "none", [], False

    confirmations = record["confirmations"]
    last = confirmations[-1] if confirmations else None
    confirmed_entry = (
        last if last is not None and last["action"] == "confirm"
        and manual is not None and last["edit_id"] == manual["edit_id"] else None
    )
    if confirmed_entry is None:
        billing_state = DRAFT
    elif edits_current:
        billing_state = CONFIRMED
    else:
        billing_state = OUTDATED

    problems = _problems(job, row, source, edits_current, state, blocked, lines, no_billing, shared)
    warnings = (len(result["conflicts"]) + len(result["unresolved"])) if result else 0
    return {
        "state": billing_state,
        "source": source,
        "edits_current": edits_current if manual is not None else None,
        "attempt_id": basis["interpretation_attempt_id"],
        "interpretation_state": state,
        "shared": shared,
        "lines": lines,
        "no_billing": no_billing,
        "confirmation": copy.deepcopy(confirmed_entry),
        "edited_at": manual["saved_at"] if manual else None,
        "edit_count": sum(1 for e in edits if e["kind"] == "lines"),
        "confirmation_count": sum(1 for c in confirmations if c["action"] == "confirm"),
        # Optimistic concurrency: any change to this row's billing moves it.
        "version": len(edits) + len(confirmations),
        "problems": problems,
        "warnings": warnings,
        "basis": basis,
    }


def _problems(job, row, source, edits_current, state, blocked, lines, no_billing, shared) -> list[str]:
    """Why this appointment cannot be exported (or confirmed) as it stands."""
    if job.doctor is None:
        return ["no doctor profile is assigned to this document"]
    problems: list[str] = []
    if source == "manual":
        if not edits_current:
            problems.append(EDITS_OUTDATED)
    elif state != "proposed":
        reason = STATE_REASONS.get(state, state)
        if state == "blocked" and blocked:
            reason = ilib.BLOCKED_MESSAGES[blocked]
        return [reason]
    if not lines and not no_billing:
        problems.append("no billing lines: add a code, or mark the appointment as having no billing acts")
    empty = [str(i) for i, line in enumerate(lines, start=1) if not line["code"]]
    if empty:
        which = ", ".join(empty)
        if source == "model":
            problems.append(f"unresolved procedure code (act {which})")
        else:
            problems.append(f"no procedure code on line {which}")
    patient = _confirmed_value(row, "patient_id")
    if lines and all(line["code"] in NO_HIN_CODES for line in lines):
        pass  # the HIN is exported when known, but not required
    elif patient is None:
        problems.append("no patient assigned")
    elif not shared["HIN"]:
        problems.append(f"no health number on record for {patient}")
    if not shared["ClaimDate"]:
        problems.append("no date")
    return problems


# -- changes ------------------------------------------------------------


def _record(job: Any, row_id: str) -> dict[str, Any]:
    return job.billing.setdefault(row_id, empty_record())


def check_base(view: Mapping[str, Any], body: Mapping[str, Any], *, lines: bool) -> None:
    """Refuse a change made from an out-of-date screen."""
    if body.get("base_version") != view["version"]:
        raise ConflictError("This appointment's billing changed since it was loaded; reload it.")
    if lines and body.get("attempt_id") != view["attempt_id"]:
        raise ConflictError("The proposed billing changed since it was loaded; review the new proposal.")
    if lines and view["interpretation_state"] == "running":
        raise ConflictError("This appointment is still being processed; wait for it to finish.")


class ConflictError(Exception):
    pass


def save_lines(job: Any, row_id: str, view: Mapping[str, Any], acts: Sequence[Mapping[str, Any]],
               no_billing: bool) -> dict[str, Any]:
    """Store the person's lines against the current basis. Ends any
    confirmation (a later edit supersedes it); earlier entries are kept."""
    edit = {
        "edit_id": f"bed_{uuid.uuid4().hex[:12]}",
        "kind": "lines",
        "acts": [dict(a) for a in acts],
        "no_billing": no_billing,
        "basis": copy.deepcopy(view["basis"]),
        "saved_at": rowlib.now_iso(),
    }
    with job.extract_lock:
        _record(job, row_id)["edits"].append(edit)
    return edit


def reset(job: Any, row_id: str, view: Mapping[str, Any]) -> None:
    """Go back to the model's proposal. The manual lines stay in history."""
    with job.extract_lock:
        _record(job, row_id)["edits"].append({
            "edit_id": f"bed_{uuid.uuid4().hex[:12]}",
            "kind": "reset",
            "basis": copy.deepcopy(view["basis"]),
            "saved_at": rowlib.now_iso(),
        })


def confirm(job: Any, row_id: str, edit: Mapping[str, Any]) -> None:
    with job.extract_lock:
        _record(job, row_id)["confirmations"].append({
            "confirmation_id": f"bcf_{uuid.uuid4().hex[:12]}",
            "action": "confirm",
            "edit_id": edit["edit_id"],
            "basis": copy.deepcopy(edit["basis"]),
            "at": rowlib.now_iso(),
        })


def reopen(job: Any, row_id: str) -> None:
    with job.extract_lock:
        _record(job, row_id)["confirmations"].append({
            "confirmation_id": f"bcf_{uuid.uuid4().hex[:12]}",
            "action": "reopen",
            "edit_id": None,
            "at": rowlib.now_iso(),
        })


def public_view(view: Mapping[str, Any]) -> dict[str, Any]:
    """What the screen gets: everything but the internal basis."""
    return {k: v for k, v in view.items() if k != "basis"}
