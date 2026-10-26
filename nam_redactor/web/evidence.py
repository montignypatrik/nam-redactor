"""Evidence crops: what a reviewer approved, rendered from the saved review.

A row's evidence is one image: the whole appointment row, cut from the
working page *after* the reviewer's accepted redactions are burned in. It is
the complete input for a later extraction step -- time and handwritten notes
both come from it. They are always rendered from the saved reviewed
state (:meth:`Job.effective_boxes`), never from the initial detections, so a
box the reviewer added hides its content in the evidence and a box they
removed no longer does.

At approval the crops are frozen to disk with a SHA-256 each, so an export
can point at the exact bytes that were approved.
"""
from __future__ import annotations

import hashlib
import io
import math
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image

from ..redactor import rasterize_page, redact_image
from ..provenance import working_to_source
from .jobs import JOB_INPUT_NAME, Job
from .storage import EVIDENCE_DIR

#: Evidence kinds rendered today. Rows prepared before schema 2 may also
#: reference legacy ``handwriting`` crops; those files are kept, not rendered.
KINDS = ("row",)


def _png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def render_page(job: Job, page_num: int, dpi: int, *, redacted: bool) -> Image.Image:
    """A working page, optionally with the saved reviewed redactions applied."""
    with pymupdf.open(str(job.input_path)) as doc:
        image = rasterize_page(doc[page_num - 1], dpi=dpi)
    if redacted:
        image = redact_image(image, job.effective_boxes(page_num))
    return image


def page_png(job: Job, page_num: int, dpi: int, *, redacted: bool) -> bytes:
    image = render_page(job, page_num, dpi, redacted=redacted)
    try:
        return _png(image)
    finally:
        image.close()


def source_page_png(job: Job, source_page: int, dpi: int, *, redacted: bool) -> bytes:
    """A page of the original upload, optionally with the saved reviewed
    redactions of every section cut from it burned in at their place.

    Boxes are mapped with the same transform the export uses
    (:func:`working_to_source`), so what review shows and what the manifest
    records cannot drift apart.
    """
    with pymupdf.open(str(job.dir / JOB_INPUT_NAME)) as doc:
        image = rasterize_page(doc[source_page - 1], dpi=dpi)
    try:
        if redacted:
            boxes = []
            for prov in job.provenance:
                if prov.source_page != source_page:
                    continue
                for box in job.effective_boxes(prov.working_page):
                    boxes.append({**box, **working_to_source(box, prov)})
            image = redact_image(image, boxes)
        return _png(image)
    finally:
        image.close()


def _crop_box(bounds: Mapping[str, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(bounds["x1"] * width)))
    y1 = max(0, min(height - 1, int(bounds["y1"] * height)))
    # Round the far edge up so the crop never trims the region it names.
    x2 = max(x1 + 1, min(width, math.ceil(bounds["x2"] * width)))
    y2 = max(y1 + 1, min(height, math.ceil(bounds["y2"] * height)))
    return x1, y1, x2, y2


def render_region(job: Job, page_num: int, bounds: Mapping[str, float]) -> tuple[bytes, tuple[int, int, int, int]]:
    """``(png, crop_px)`` for a region of a redacted working page, at the
    document's DPI -- the same pixels detection and the output use."""
    image = render_page(job, page_num, job.dpi, redacted=True)
    try:
        box = _crop_box(bounds, *image.size)
        crop = image.crop(box)
        try:
            return _png(crop), box
        finally:
            crop.close()
    finally:
        image.close()


def region_of(row: Mapping[str, Any], kind: str) -> Mapping[str, float] | None:
    return row["bounds"] if kind == "row" else None


def _referenced_paths(job: Job, row_id: str) -> set[str]:
    """Evidence paths extraction history still points at, for one row."""
    with job.extract_lock:
        record = job.extractions.get(row_id) or {}
        paths = {
            item["basis"]["evidence_path"]
            for key in ("attempts", "reviews")
            for item in record.get(key, [])
            if isinstance(item.get("basis"), dict) and item["basis"].get("evidence_path")
        }
        for attempt in (job.interpretations.get(row_id) or {}).get("attempts", []):
            evidence = (attempt.get("inputs") or {}).get("evidence") or {}
            if evidence.get("evidence_path"):
                paths.add(evidence["evidence_path"])
        return paths


def freeze(job: Job, row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Write the row's evidence crop for its current state; return refs.

    Evidence files are immutable. Each name carries the row revision *and* a
    prefix of the image's own SHA-256, so a new image -- a changed redaction
    with an unchanged row revision included -- always gets a new file, and an
    identical image reuses its identical file. A file an extraction attempt
    or review refers to is never removed or overwritten, so every historical
    result keeps its exact evidence. Only superseded files nothing refers to
    are cleaned up. Paths are relative to the document folder.
    """
    folder = job.dir / EVIDENCE_DIR / row["row_id"]
    folder.mkdir(parents=True, exist_ok=True)
    refs = []
    for kind in KINDS:
        bounds = region_of(row, kind)
        if bounds is None:
            continue
        png, crop_px = render_region(job, row["working_page"], bounds)
        digest = hashlib.sha256(png).hexdigest()
        name = f"r{row['revision']}-{kind}-{digest[:16]}.png"
        target = folder / name
        if not target.exists():
            # Write-then-rename, so a crash never leaves half an image behind
            # a name that promises a particular hash.
            partial = folder / f".{name}.partial"
            partial.write_bytes(png)
            partial.replace(target)
        refs.append({
            "kind": kind,
            "path": f"{EVIDENCE_DIR}/{row['row_id']}/{name}",
            "sha256": digest,
            "working_page": row["working_page"],
            "bounds": dict(bounds),
            "crop_px": list(crop_px),
            "row_revision": row["revision"],
            "redaction_revision": job.review_revision,
        })

    keep = _referenced_paths(job, row["row_id"]) | {ref["path"] for ref in refs}
    for old in row.get("evidence", []):
        if old["path"] in keep:
            continue
        stale = (job.dir / old["path"]).resolve()
        if folder.resolve() in stale.parents:
            stale.unlink(missing_ok=True)
    return refs


def discard(job: Job, row_id: str) -> None:
    shutil.rmtree(job.dir / EVIDENCE_DIR / row_id, ignore_errors=True)


def verify(job: Job, ref: Mapping[str, Any]) -> bool:
    """The frozen file still exists and still has the recorded hash."""
    path = (job.dir / ref["path"]).resolve()
    if job.dir.resolve() not in path.parents or not path.is_file():
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == ref["sha256"]
