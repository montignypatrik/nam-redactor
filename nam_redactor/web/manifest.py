"""The exported row manifest: prepared rows and their reviewed extraction.

Built from an explicit allow-list, never by dumping the saved record. It
contains pseudonyms, dates, times, geometry and evidence references -- and
nothing else: no NAM, no name, no OCR text, no filename (a filename is often
the patient's name), and not the pseudonym map.

Only rows marked ready for extraction whose approval is still current are
exported. Every other row is listed under ``excluded`` with the reasons, so
nothing is dropped silently.

Two reviews are kept distinct:

* ``ready_for_extraction`` -- preparation: a reviewer confirmed the row's
  patient, date, region and redactions.
* ``extraction`` -- what a model read from the row, and whether a reviewer
  accepted it. Only an *accepted, current* review is exported as values
  (``extraction.reviewed``); every other state (pending, running, failed,
  stale, unreviewed) is reported by name with no model output, so nothing
  unreviewed can pass as reviewed.

The row-level ``time`` is the reviewed extraction time, else a time the
reviewer typed during preparation, else ``null`` -- never a raw model guess.
Nothing here says anything about billing.
"""
from __future__ import annotations

from typing import Any

import pymupdf

from ..provenance import COORDINATE_CONVENTION, points_to_unrotated, trace
from . import evidence
from . import extraction as exlib
from . import interpretation as ilib
from . import rows as rowlib
from .jobs import JOB_INPUT_NAME, Job

MANIFEST_SCHEMA = "nam-redactor.appointment-rows"
MANIFEST_SCHEMA_VERSION = 3

NOTICE = (
    "Each exported row is ready for extraction: its patient pseudonym (from the "
    "NAM detected in the row, or chosen by a reviewer), date (the section date, "
    "or entered by a reviewer), row region and saved redactions are recorded; "
    "ready_for_extraction.method says whether the app or a reviewer marked it. "
    "Time and handwriting count as "
    "reviewed only where extraction.status is 'reviewed'; any other status means "
    "the model output was not accepted by a reviewer and is not included. "
    "Handwriting is a literal transcription, not an interpretation. "
    "billing_draft sections are UNVALIDATED model proposals made with the assigned "
    "doctor profile version: they are not approved billing entries. This does "
    "not establish billing eligibility."
)


def _billing_draft(job: Job, row: dict[str, Any], assigned: tuple) -> dict[str, Any]:
    """The row's proposed billing, clearly marked unvalidated.

    Acts are included only while the proposal is current; an outdated or
    missing one is reported by status alone.
    """
    doctor, version = assigned
    page_boxes = job.effective_boxes(row["working_page"])
    with job.extract_lock:
        inputs, blocked = ilib.inputs_of(
            row, job.extractions.get(row["row_id"]), page_boxes, doctor, version
        )
        record = job.interpretations.get(row["row_id"])
        state = ilib.state_of(record, inputs, blocked)
        attempt = ilib.current_attempt(record, inputs)
    draft: dict[str, Any] = {"validated": False, "status": state, "proposal": None}
    if blocked:
        draft["blocked_reason"] = blocked
    if attempt is None:
        return draft
    result = attempt["result"]
    used = attempt["inputs"]
    usage = attempt.get("usage") or {}
    draft["proposal"] = {
        "acts": result["acts"],
        "unresolved": result["unresolved"],
        "conflicts": result["conflicts"],
        "ignored_notes": result["ignored_notes"],
        "no_billing_notation": result["no_billing_notation"],
        "based_on": {
            "doctor_profile_id": used["doctor"]["profile_id"],
            "doctor_profile_version": used["doctor"]["version"],
            "doctor_assignment_id": used["doctor"]["assignment_id"],
            "transcription_source": used["source"],
            "transcription_attempt_id": used["transcription_attempt_id"],
            "transcription_review_id": used["review_id"],
            "evidence_sha256": used["evidence"]["evidence_sha256"],
        },
        "model": {
            "requested": attempt["requested_model"],
            "returned": attempt["returned_model"],
            "provider": attempt["provider"],
            "prompt_version": attempt["prompt_version"],
            "schema_version": attempt["schema_version"],
            "interpreted_at": attempt["finished_at"],
            "attempt_id": attempt["attempt_id"],
        },
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost": usage.get("cost"),
            "cost_currency": usage.get("cost_currency"),
        },
    }
    return draft


def _extraction(job: Job, row: dict[str, Any]) -> dict[str, Any]:
    """The row's extraction status, and the accepted values when reviewed."""
    page_boxes = job.effective_boxes(row["working_page"])
    with job.extract_lock:
        record = job.extractions.get(row["row_id"])
        state = exlib.state_of(row, record, page_boxes)
        attempt = exlib.current_attempt(row, record, page_boxes)
        review = exlib.current_review(row, record, page_boxes)
        attempts = len((record or {}).get("attempts", []))
    out: dict[str, Any] = {"status": state, "attempts": attempts, "reviewed": None}
    if state != "reviewed" or attempt is None or review is None:
        return out
    usage = attempt.get("usage") or {}
    confirmed = exlib.confirmed_time(row)
    proposed = attempt["result"]["time"]
    out["reviewed"] = {
        "time": review["time"],
        "time_source": review["time_source"],
        "handwriting_status": review["handwriting_status"],
        "handwriting_text": review["handwriting_text"],
        "corrected": review["corrected"],
        "reviewed_at": review["reviewed_at"],
        "time_discrepancy": (
            {"confirmed": confirmed, "model": proposed}
            if confirmed and proposed and confirmed != proposed else None
        ),
        "model": {
            "requested": attempt["requested_model"],
            "returned": attempt["returned_model"],
            "provider": attempt["provider"],
            "prompt_version": attempt["prompt_version"],
            "schema_version": attempt["schema_version"],
            "extracted_at": attempt["finished_at"],
            "attempt_id": attempt["attempt_id"],
        },
        "based_on": {
            "row_revision": attempt["basis"]["row_revision"],
            "redaction_revision": attempt["basis"]["redaction_revision"],
            "evidence_sha256": attempt["basis"]["evidence_sha256"],
        },
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost": usage.get("cost"),
            "cost_currency": usage.get("cost_currency"),
        },
    }
    return out


def _region(job: Job, doc: pymupdf.Document, page_num: int, bounds: dict[str, float]) -> dict[str, Any] | None:
    prov = job.provenance_for(page_num)
    if prov is None:
        return None
    traced = trace(bounds, prov)
    traced["bounds_pdf_unrotated"] = points_to_unrotated(
        traced["bounds_pt"], doc[prov.source_page - 1]
    )
    traced["working_bounds"] = dict(bounds)
    return traced


def _exclusion_reasons(job: Job, row: dict[str, Any]) -> list[str]:
    approval = row["approval"]
    if approval["status"] == rowlib.INVALIDATED:
        return ["readiness invalidated: " + "; ".join(approval["invalidated_reasons"] or ["changed"])]
    if approval["status"] != rowlib.APPROVED:
        missing = rowlib.unresolved(row)
        return ["not marked ready for extraction"] + ([f"unresolved: {', '.join(missing)}"] if missing else [])
    if not rowlib.is_current(row, job.effective_boxes(row["working_page"])):
        return ["approval no longer matches the row or its redactions"]
    if job.provenance_for(row["working_page"]) is None:
        return ["no source provenance for its page"]
    kinds = {ref["kind"] for ref in row.get("evidence", [])}
    if rowlib.EVIDENCE_KIND not in kinds:
        return ["row evidence missing"]
    if not all(evidence.verify(job, ref) for ref in row["evidence"]):
        return ["evidence file missing or altered"]
    return []


def build(job: Job, assigned: tuple | None = None) -> dict[str, Any]:
    """``assigned`` is the document's (doctor assignment, its exact profile
    version); without it, billing drafts report ``no_doctor``/missing."""
    assigned = assigned or (job.doctor, None)
    exported: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    with pymupdf.open(str(job.dir / JOB_INPUT_NAME)) as doc:
        for row in job.rows.values():
            reasons = _exclusion_reasons(job, row)
            if reasons:
                excluded.append({
                    "row_id": row["row_id"],
                    "source_page": row["source_page"],
                    "reasons": reasons,
                })
                continue
            fields = row["fields"]
            time = fields["time"]
            extraction = _extraction(job, row)
            reviewed = extraction["reviewed"]
            if reviewed is not None:
                row_time, time_status = reviewed["time"], "reviewed_extraction"
            elif time["status"] == rowlib.CONFIRMED:
                row_time, time_status = time["value"], "entered"
            else:
                row_time, time_status = None, "to_be_extracted"
            refs = []
            for ref in row["evidence"]:
                refs.append({
                    "kind": ref["kind"],
                    "path": ref["path"],
                    "sha256": ref["sha256"],
                    "crop_px": ref["crop_px"],
                    "region": _region(job, doc, row["working_page"], ref["bounds"]),
                })
            approval = row["approval"]
            exported.append({
                "row_id": row["row_id"],
                "patient_id": fields["patient_id"]["value"],
                "date": fields["date"]["value"],
                # Reviewed extraction, else typed by the reviewer, else null.
                "time": row_time,
                "time_status": time_status,
                "field_sources": {
                    name: fields[name]["source"] if fields[name]["status"] == rowlib.CONFIRMED else None
                    for name in rowlib.FIELDS
                },
                "row_region": _region(job, doc, row["working_page"], row["bounds"]),
                "evidence": refs,
                "revision": row["revision"],
                "ready_for_extraction": {
                    # "automatic": the app confirmed a patient and date it
                    # derived itself; "manual": a reviewer marked the row.
                    "method": approval.get("method", rowlib.METHOD_MANUAL),
                    "approved_at": approval["approved_at"],
                    "approved_revision": approval["approved_revision"],
                    "redaction_revision": approval["redaction_revision"],
                },
                "extraction": extraction,
                # Proposed billing: explicitly unvalidated, never an entry.
                "billing_draft": _billing_draft(job, row, assigned),
            })

    exported.sort(key=lambda r: (r["date"] or "", r["time"] or "", r["row_region"]["source_page"], r["row_region"]["bounds"]["y1"]))
    return {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "row_schema_version": rowlib.ROW_SCHEMA_VERSION,
        "generated_at": rowlib.now_iso(),
        "notice": NOTICE,
        "document": {
            "document_id": job.id,
            "original_sha256": job.original_sha256,
            "source_page_count": job.source_page_count,
            "render_dpi": job.dpi,
            "redaction_revision": job.review_revision,
            "practice": dict(job.practice),
            # The doctor profile version the document is assigned, with the
            # snapshot its identifiers came from; null for documents from
            # before doctor profiles that have not been assigned one.
            "doctor": None if job.doctor is None else {
                "profile_id": job.doctor["profile_id"],
                "version": job.doctor["version"],
                "assignment_id": job.doctor["assignment_id"],
                "assigned_at": job.doctor["assigned_at"],
                "display_name": job.doctor["snapshot"]["display_name"],
            },
            "coordinate_convention": COORDINATE_CONVENTION,
            "evidence_paths_relative_to": "the document's folder in the app's data directory",
        },
        "summary": {
            "rows_total": len(job.rows),
            "rows_exported": len(exported),
            "rows_excluded": len(excluded),
            "extraction": _extraction_counts(exported),
            "billing_draft": _draft_counts(exported),
        },
        # Exported rows whose time and handwriting are NOT reviewed yet.
        "extraction_not_reviewed": [
            {"row_id": r["row_id"], "status": r["extraction"]["status"]}
            for r in exported if r["extraction"]["status"] != "reviewed"
        ],
        "rows": exported,
        "excluded": excluded,
    }


def _extraction_counts(exported: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in exported:
        state = row["extraction"]["status"]
        counts[state] = counts.get(state, 0) + 1
    return counts


def _draft_counts(exported: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in exported:
        state = row["billing_draft"]["status"]
        counts[state] = counts.get(state, 0) + 1
    return counts
