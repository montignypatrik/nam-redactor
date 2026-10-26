"""Billing CSV export: one line per proposed act, in the practice's format.

The layout follows the example file the practice supplied: 17 columns in a
fixed order, ``;`` as delimiter, UTF-8 without BOM, CRLF between lines, a
header, and no line ending after the last line. It carries no billing rules:
each appointment exports its **effective billing lines** (see
:mod:`.billing`) -- the person's saved lines when there are any, else the
current interpretation under the document's assigned doctor profile -- copied
verbatim: ``"vs"`` stays ``"vs"``, ``"15824"`` stays ``"15824"``, blank units
and contexts stay blank. There is no mapping table here.

Nothing is sent anywhere and nothing is changed: the file is built locally
from saved results. Confirmation is not required (drafts export too) and
exporting confirms nothing. It *does* contain health numbers (HIN), resolved
from the private identity map on this machine -- it is meant for the
practice's own billing software.

Rows that cannot be exported correctly block the export instead of being
dropped: a missing, failed, outdated or blocked proposal, manual lines whose
source has since changed, a line without a code, or a patient whose health
number is unknown. Appointments with no billing acts produce no lines.
"""
from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import billing
from . import rows as rowlib
from .billing import HEADER, UNUSED  # noqa: F401  (re-exported)
from .jobs import Job


@dataclass
class Problem:
    row_id: str
    label: str
    reason: str


@dataclass
class Export:
    content: bytes = b""
    lines: int = 0
    problems: list[Problem] = field(default_factory=list)
    #: Rows with no billing acts (no lines, not an error).
    no_billing_rows: list[str] = field(default_factory=list)
    #: Rows whose billing a person has confirmed (and still current).
    confirmed_rows: list[str] = field(default_factory=list)


def _ordered(job: Job) -> list[dict[str, Any]]:
    """Rows in reading order: page, then top to bottom, left to right --
    the same order the results show."""
    return sorted(
        job.rows.values(),
        key=lambda r: (r["working_page"], r["bounds"]["y1"], r["bounds"]["x1"]),
    )


def _labels(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    labels, manual = {}, 0
    for row in rows:
        if row.get("origin") == rowlib.ORIGIN_SECTION:
            labels[row["row_id"]] = f"#{row['working_page']}"
        else:
            manual += 1
            labels[row["row_id"]] = f"R{manual}"
    return labels


def build(job: Job, doctor: Mapping[str, Any] | None, version: Mapping[str, Any] | None) -> Export:
    """Build the CSV from saved state, or report what blocks it.

    Each appointment exports its **effective** billing lines
    (:func:`billing.compute`): the person's saved lines when there are any,
    else the current model proposal -- exactly what the results screen
    shows. ``doctor``/``version`` are the document's assignment and its
    exact profile version (see :meth:`JobStore.assigned_version`).
    """
    export = Export()
    rows = _ordered(job)
    labels = _labels(rows)
    if job.doctor is None or doctor is None:
        export.problems.append(Problem("", "document", "no doctor profile is assigned to this document"))
        return export
    nam_of = {pseudonym: nam for nam, pseudonym in job.patient_ids.items()}

    lines: list[list[str]] = []
    for row in rows:
        row_id, label = row["row_id"], labels[row["row_id"]]
        view = billing.compute(job, row, doctor, version, nam_of)
        if view["problems"]:
            export.problems.extend(Problem(row_id, label, reason) for reason in view["problems"])
            continue
        if view["state"] == billing.CONFIRMED:
            export.confirmed_rows.append(row_id)
        if not view["lines"]:
            export.no_billing_rows.append(row_id)
            continue
        for act in view["lines"]:
            line = {
                **view["shared"],
                "ProcedureCode": act["code"],
                "Units": billing.units_text(act["units"]),
                "ContextualElements": ",".join(act["context_elements"]),
            }
            lines.append([line.get(column, "") for column in HEADER])

    if export.problems:
        return export
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter=";", lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(HEADER)
    writer.writerows(lines)
    # The practice's file has no line ending after its last line.
    text = buffer.getvalue()
    if text.endswith("\r\n"):
        text = text[:-2]
    export.content = text.encode("utf-8")
    export.lines = len(lines)
    return export
