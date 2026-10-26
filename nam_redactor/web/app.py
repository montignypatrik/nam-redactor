"""FastAPI application: upload, zones, detection, review, rows, export.

Binds to loopback only. The documents handled here contain health
information, so several decisions below are deliberate hardening rather than
defaults: no openapi/docs routes, an Origin check on mutations (binding to
127.0.0.1 is not enough on its own -- any page in the user's browser can reach
loopback), magic-byte upload validation, and no PHI text in any payload.
"""
from __future__ import annotations

import asyncio
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import pymupdf
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ..locator import canonical_from_text
from ..provenance import source_crop, trace
from ..redactor import rasterize_page
from .. import extraction as ex
from . import doctors as doclib
from . import billing_csv, evidence, manifest
from . import extraction as exlib
from . import billing
from . import interpretation as ilib
from . import rows as rowlib
from .jobs import (
    DEFAULT_DPI,
    MAX_DPI,
    MAX_PAGES,
    MAX_UPLOAD_BYTES,
    MAX_ZONES,
    MIN_DPI,
    PHASE_EVENTS,
    PRACTICE_FIELDS,
    PHASE_TERMINAL,
    PRE_REVIEW_STATUSES,
    Job,
    JobStore,
    assign_patient_id,
    assign_doctor,
    coerce_boxes,
    ensure_section_rows,
    parse_page_date,
    parse_practice_field,
    save_review,
    sha256_file,
)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

_MISSING_BUILD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>nam-redactor</title></head>
<body style="font-family:system-ui;max-width:38rem;margin:4rem auto;line-height:1.5">
<h1>Frontend not built</h1>
<p>The API is running, but the web interface has not been compiled.</p>
<pre style="background:#f4f4f5;padding:1rem;border-radius:6px">cd frontend
npm ci
npm run build</pre>
<p>Or run <code>npm run dev</code> for the development server.</p>
</body></html>
"""


def _safe_display_name(name: str | None) -> str:
    """A filename safe to echo back. Display only -- never used as a path."""
    if not name:
        return "document.pdf"
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    base = "".join(ch for ch in base if ch.isprintable() and ch not in "\r\n\t")
    return (base.strip() or "document.pdf")[:120]


def _origin_matches_host(origin: str, host: str) -> bool:
    """True when an Origin header refers to the same host:port we are serving.

    Loopback aliases are treated as equal so that visiting via ``localhost``
    while ``Host`` says ``127.0.0.1`` (or vice versa) is not a false rejection.
    """
    origin_parts = urlsplit(origin)
    host_parts = urlsplit(f"//{host}")
    try:
        origin_port = origin_parts.port
        host_port = host_parts.port
    except ValueError:
        return False
    if origin_port != host_port:
        return False
    origin_host = (origin_parts.hostname or "").lower()
    host_name = (host_parts.hostname or "").lower()
    if origin_host == host_name:
        return True
    return origin_host in _LOOPBACK_HOSTS and host_name in _LOOPBACK_HOSTS


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n"


def _render_page_png(input_path: Path, page_num: int, dpi: int) -> bytes:
    with pymupdf.open(str(input_path)) as doc:
        image = rasterize_page(doc[page_num - 1], dpi=dpi)
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        image.close()


class PageBoxes(BaseModel):
    page_num: int
    boxes: list[dict[str, Any]] = []
    #: ``YYYY-MM-DD``. Required at the zone step for every page kept, and
    #: at finalize for every page.
    date: str | None = None


class FinalizePayload(BaseModel):
    pages: list[PageBoxes] = []


class ReviewPayload(BaseModel):
    """The reviewer's redactions and page dates. ``base_revision`` is the
    revision the client started from; a mismatch means another tab saved in
    between, and the request is refused rather than silently overwriting."""

    pages: list[PageBoxes] = []
    base_revision: int | None = None


#: Statuses in which detections exist and review/row work is allowed.
REVIEWABLE = frozenset({"awaiting_review", "done"})


class StartPayload(BaseModel):
    """Zones and a date per source page. No zones at all means the whole
    document, and then every page needs a date."""

    pages: list[PageBoxes] = []


def create_app(
    engine_factory: Callable[[], Any] | None = None,
    dir_root: Path | None = None,
    static_dir: Path | None = None,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    name_model_factory: Callable[[], Any] | None = None,
    extractor_factory: Callable[[], Any] | None = None,
    retry_delays: tuple[float, ...] | None = None,
    extraction_concurrency: int | None = None,
) -> FastAPI:
    """Build the app. ``engine_factory``, ``name_model_factory`` and
    ``extractor_factory`` are the seams tests use to stub OCR, NER and the
    cloud extraction service."""
    extra: dict[str, Any] = {}
    if retry_delays is not None:
        extra["retry_delays"] = retry_delays
    if extraction_concurrency is not None:
        extra["extraction_concurrency"] = extraction_concurrency
    store = JobStore(
        engine_factory=engine_factory,
        dir_root=dir_root,
        name_model_factory=name_model_factory,
        extractor_factory=extractor_factory,
        **extra,
    )
    doctors = store.doctors
    dist_dir = static_dir or (Path(__file__).resolve().parent / "dist")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        store.sweep_stale_dirs()
        store.load_saved()
        try:
            yield
        finally:
            store.shutdown()

    app = FastAPI(
        title="nam-redactor",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.store = store
    app.state.doctors = doctors
    app.state.dist_dir = dist_dir

    # Loopback is not a security boundary against the user's own browser, so
    # reject cross-origin mutations explicitly.
    async def guard_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        if not origin:
            return  # non-browser client
        host = request.headers.get("host")
        if not host or not _origin_matches_host(origin, host):
            raise HTTPException(status_code=403, detail="Cross-origin request rejected.")

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "::1", "testserver"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        else:
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data: blob:; "
                "style-src 'self' 'unsafe-inline'; script-src 'self'; "
                "connect-src 'self'; object-src 'none'; base-uri 'none'; "
                "frame-ancestors 'none'",
            )
        return response

    def require_job(job_id: str) -> Job:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job.")
        return job

    def require_reviewable(job_id: str) -> Job:
        job = require_job(job_id)
        if job.status in PRE_REVIEW_STATUSES:
            raise HTTPException(status_code=409, detail="Detection is still running.")
        if job.status == "rendering":
            raise HTTPException(status_code=409, detail="The document is being rendered.")
        if job.status not in REVIEWABLE:
            raise HTTPException(status_code=409, detail="Detection failed.")
        return job

    def require_row(job: Job, row_id: str) -> dict[str, Any]:
        row = job.rows.get(row_id) if rowlib.ROW_ID_RE.match(row_id) else None
        if row is None:
            raise HTTPException(status_code=404, detail="Unknown row.")
        return row

    async def json_body(request: Request) -> dict[str, Any]:
        # A plain dict rather than a pydantic model: FastAPI's validation
        # errors echo the input back, and some of these fields are PHI.
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="A JSON object is required.") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="A JSON object is required.")
        return body

    def doctor_view(job: Job) -> dict[str, Any] | None:
        """The document's doctor assignment, plus what the profile says now:
        whether it was archived, and whether a newer version exists."""
        if job.doctor is None:
            return None
        profile = doctors.get(job.doctor["profile_id"])
        return {
            **job.doctor,
            "profile_status": profile["status"] if profile else "missing",
            "current_version": profile["current_version"] if profile else None,
            "newer_version_available": bool(profile and profile["current_version"] > job.doctor["version"]),
        }

    def summary(job: Job) -> dict[str, Any]:
        rows = list(job.rows.values())
        return {
            "job_id": job.id,
            "filename": job.filename,
            "status": job.status,
            "dpi": job.dpi,
            **job.practice,
            "doctor": doctor_view(job),
            "created_at": job.created_at,
            "source_page_count": job.source_page_count,
            "total_pages": job.total_pages,
            "review_revision": job.review_revision,
            "rows_total": len(rows),
            "rows_approved": sum(
                1 for r in rows if rowlib.is_current(r, job.effective_boxes(r["working_page"]))
            ),
        }

    def row_view(job: Job, row: dict[str, Any], assigned: tuple | None = None) -> dict[str, Any]:
        """A row as the review screen needs it: the record plus what is
        derived from the current saved state. ``assigned`` is the document's
        (doctor, profile version), resolved once per request."""
        page_boxes = job.effective_boxes(row["working_page"])
        prov = job.provenance_for(row["working_page"])
        with job.extract_lock:
            extraction = exlib.view(row, job.extractions.get(row["row_id"]), page_boxes)
        doctor, version = assigned if assigned is not None else store.assigned_version(job)
        interpretation = store.interpretation_view(job, row["row_id"], doctor, version)
        return {
            **row,
            "unresolved": rowlib.unresolved(row),
            "approval_current": rowlib.is_current(row, page_boxes),
            "source_bounds": None if prov is None else trace(row["bounds"], prov),
            # Separate from readiness: "ready for extraction" is preparation;
            # this is what extraction produced and whether it was reviewed.
            "extraction": extraction,
            # Proposed billing: a separate stage, always unvalidated here.
            "interpretation": interpretation,
            # The billing lines as exported (manual edits, else the proposal),
            # and the person's billing confirmation. HIN is resolved here,
            # locally; it is shown to the reviewer, never sent to the model.
            "billing": billing.public_view(billing.compute(job, row, doctor, version)),
        }

    def parse_bounds(value: Any, name: str) -> dict[str, float] | None:
        if value is None:
            return None
        boxes = coerce_boxes([value])
        if not boxes:
            raise HTTPException(status_code=400, detail=f"Invalid {name}.")
        box = boxes[0]
        if box["x2"] - box["x1"] < 0.002 or box["y2"] - box["y1"] < 0.002:
            raise HTTPException(status_code=400, detail=f"The {name} is too small.")
        return {k: box[k] for k in ("x1", "y1", "x2", "y2")}

    def parse_row_patch(job: Job, body: dict[str, Any]) -> dict[str, Any]:
        patch: dict[str, Any] = {}
        if "bounds" in body:
            bounds = parse_bounds(body["bounds"], "row region")
            if bounds is None:
                raise HTTPException(status_code=400, detail="A row needs a region.")
            patch["bounds"] = bounds
        if "handwriting_bounds" in body:
            raise HTTPException(
                status_code=400,
                detail="Handwriting regions are no longer used: the whole row is the evidence.",
            )
        if "patient_id" in body:
            value = body["patient_id"]
            if value is not None and value not in job.pseudonyms():
                raise HTTPException(status_code=400, detail="Unknown patient for this document.")
            patch["patient_id"] = value
        if body.get("no_patient") is True:
            patch.pop("patient_id", None)
            patch["no_patient"] = True
        if "date" in body:
            value = body["date"]
            if value is not None and parse_page_date(value) is None:
                raise HTTPException(status_code=400, detail="Invalid date.")
            patch["date"] = value
        if "time" in body:
            value = body["time"]
            if value is not None and not (isinstance(value, str) and rowlib.TIME_RE.match(value)):
                raise HTTPException(status_code=400, detail="Invalid time (use HH:MM).")
            patch["time"] = value
        if "confirm" in body:
            confirm = body["confirm"]
            if not isinstance(confirm, list) or not set(confirm) <= set(rowlib.FIELDS):
                raise HTTPException(status_code=400, detail="Invalid confirmation.")
            patch["confirm"] = confirm
        return patch

    # -- API --------------------------------------------------------------

    def profile_error(exc: doclib.ProfileError) -> HTTPException:
        return HTTPException(status_code=exc.status, detail=str(exc))

    @app.post("/api/jobs", status_code=202, dependencies=[Depends(guard_origin)])
    async def create_job(
        file: UploadFile = File(...),
        dpi: int = Form(DEFAULT_DPI),
        doctor_profile_id: str | None = Form(None),
        doctor_profile_version: str | None = Form(None),
        licence_number: str | None = Form(None),
        group_number: str | None = Form(None),
        sector_number: str | None = Form(None),
    ) -> dict[str, Any]:
        content_type = (file.content_type or "").lower()
        if content_type and "pdf" not in content_type and content_type != "application/octet-stream":
            raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

        # Identifiers come from the chosen doctor profile, resolved here --
        # never from the browser. Checked before the file is stored.
        if any(v not in (None, "") for v in (licence_number, group_number, sector_number)):
            raise HTTPException(
                status_code=400,
                detail="Licence, group and sector come from the doctor profile; choose a profile instead.",
            )
        try:
            version = doctors.resolve_for_assignment(doctor_profile_id, doctor_profile_version)
        except doclib.ProfileError as exc:
            raise profile_error(exc) from exc

        dpi = max(MIN_DPI, min(MAX_DPI, dpi))
        job = store.create(filename=_safe_display_name(file.filename), dpi=dpi)
        assign_doctor(job, version)

        written = 0
        header = b""
        try:
            with open(job.input_path, "wb") as handle:
                while True:
                    chunk = await file.read(1 << 20)
                    if not chunk:
                        break
                    if not header:
                        header = chunk[:5]
                    written += len(chunk)
                    # Count as we stream rather than trusting Content-Length.
                    if written > max_upload_bytes:
                        raise HTTPException(status_code=413, detail="File is too large.")
                    handle.write(chunk)

            if header != b"%PDF-":
                raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

            try:
                with pymupdf.open(str(job.input_path)) as doc:
                    page_count = len(doc)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail="That PDF could not be read.") from exc

            if page_count == 0:
                raise HTTPException(status_code=400, detail="That PDF has no pages.")
            if page_count > MAX_PAGES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Documents are limited to {MAX_PAGES} pages.",
                )
        except HTTPException:
            store.delete(job.id)
            raise
        except Exception:
            store.delete(job.id)
            raise HTTPException(status_code=500, detail="The upload could not be stored.")

        job.total_pages = page_count
        job.source_page_count = page_count
        job.original_sha256 = sha256_file(job.input_path)
        store.save(job)
        # Detection waits for the zone step: POST .../start.
        return {
            "job_id": job.id,
            "filename": job.filename,
            "dpi": job.dpi,
            "total_pages": page_count,
            **job.practice,
            "doctor": doctor_view(job),
        }

    @app.get("/api/jobs")
    async def list_jobs() -> dict[str, Any]:
        """Saved documents, newest first."""
        return {"documents": [summary(job) for job in store.list()]}

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str) -> dict[str, Any]:
        job = require_job(job_id)
        return {
            **summary(job),
            # Zones and dates as last submitted, keyed by source page: lets an
            # interrupted document go back to the zone step prefilled.
            "zone_request": job.zone_request,
            "total_pages": job.total_pages,
            "total_nams": job.total_nams,
            "total_names": job.total_names,
            "per_page_counts": job.per_page_counts,
            "error": job.error,
            "result_url": f"/api/jobs/{job.id}/result" if job.status == "done" else None,
        }

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(
        job_id: str,
        request: Request,
        phase: str = "detect",
    ) -> StreamingResponse:
        if phase not in PHASE_EVENTS:
            raise HTTPException(status_code=400, detail="Unknown phase.")
        job = require_job(job_id)
        allowed = PHASE_EVENTS[phase]
        terminal = PHASE_TERMINAL[phase]
        snapshot, queue = job.subscribe(allowed, terminal)

        async def stream():
            try:
                # History first: upload -> subscribe always races, and a client
                # connecting after this phase finished must still get its
                # terminal event instead of waiting on a stream that is over.
                for event in snapshot:
                    yield _sse(event)
                    if event.get("type") in terminal:
                        return
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"  # keeps intermediaries from buffering
                        continue
                    if event.get("type") not in allowed:
                        continue  # published for the other phase
                    yield _sse(event)
                    if event.get("type") in terminal:
                        return
            finally:
                # A disconnect must not cancel the job -- the user may just be
                # reloading the tab.
                job.unsubscribe(queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/api/jobs/{job_id}/pages")
    async def job_pages(job_id: str) -> dict[str, Any]:
        job = require_job(job_id)
        if job.status in PRE_REVIEW_STATUSES:
            raise HTTPException(status_code=409, detail="Detection is still running.")
        if job.status == "error":
            raise HTTPException(status_code=409, detail="Detection failed.")
        return {
            "job_id": job.id,
            "filename": job.filename,
            "status": job.status,
            "dpi": job.dpi,
            **job.practice,
            "doctor": doctor_view(job),
            "review_revision": job.review_revision,
            "reviewed_at": job.reviewed_at,
            # Pseudonyms only; the NAMs behind them never leave the server.
            "patients": job.pseudonyms(),
            "total_pages": job.total_pages,
            "total_nams": job.total_nams,
            "total_names": job.total_names,
            "per_page_counts": job.per_page_counts,
            "pages": [
                {
                    "page_num": page.page_num,
                    "width": page.width,
                    "height": page.height,
                    # Boxes and confidence only -- never recognized text.
                    # ``boxes`` is the saved reviewed state (the detections
                    # until a review is saved); ``detected_boxes`` never
                    # changes, so the client can reset to it.
                    "boxes": list(job.effective_boxes(page.page_num)),
                    "detected_boxes": list(page.boxes),
                    "date": job.page_dates.get(page.page_num),
                    "source_page": (
                        prov.source_page if (prov := job.provenance_for(page.page_num)) else None
                    ),
                    # Where this page sits on its source page (normalized), so
                    # review can show every section in place on the original.
                    "source_crop": None if prov is None else source_crop(prov),
                }
                for page in job.pages
            ],
        }

    @app.post(
        "/api/jobs/{job_id}/start",
        status_code=202,
        dependencies=[Depends(guard_origin)],
    )
    async def start(job_id: str, payload: StartPayload) -> dict[str, Any]:
        """Apply the user's zones and begin detection.

        Each zone becomes a page of its own, and pages with no zone are left
        out. With no zone at all the document is kept as uploaded. Every page
        that will be kept must carry a date.
        """
        job = require_job(job_id)
        # A failed detection may be retried; anything further along may not,
        # because new zones would orphan the review and rows built on them.
        if job.status not in {"awaiting_zones", "error"} or job.pages:
            raise HTTPException(status_code=409, detail="Detection has already started.")

        source_pages = job.source_page_count or 0
        zones: dict[int, tuple[dict[str, Any], ...]] = {}
        dates: dict[int, str] = {}
        for entry in payload.pages:
            if not (1 <= entry.page_num <= source_pages):
                continue
            boxes = coerce_boxes(entry.boxes)
            if boxes:
                zones[entry.page_num] = zones.get(entry.page_num, ()) + boxes
            date = parse_page_date(entry.date)
            if date is not None:
                dates[entry.page_num] = date

        kept = set(zones) if zones else set(range(1, source_pages + 1))
        missing = sorted(kept - set(dates))
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Every page kept needs a date (missing on page {missing[0]}).",
            )
        if sum(len(boxes) for boxes in zones.values()) > MAX_ZONES:
            raise HTTPException(
                status_code=413, detail=f"Documents are limited to {MAX_ZONES} zones."
            )

        store.start_detect(job, asyncio.get_running_loop(), zones, dates)
        return {"job_id": job.id, "status": job.status}

    @app.get("/api/jobs/{job_id}/page/{page_num}.png")
    async def job_page_image(
        job_id: str, page_num: int, dpi: int | None = None, redacted: bool = False
    ) -> Response:
        """One page, as a PNG. ``dpi`` lowers the resolution for thumbnails;
        it cannot exceed the job's own. ``redacted`` burns in the saved
        reviewed redactions -- the page exactly as evidence will show it."""
        job = require_job(job_id)
        if job.total_pages is None or not (1 <= page_num <= job.total_pages):
            raise HTTPException(status_code=404, detail="No such page.")
        if job.status in {"queued", "detecting"}:
            # The input may be mid-swap to the zoned document.
            raise HTTPException(status_code=409, detail="Detection is still running.")
        render_dpi = job.dpi if dpi is None else max(MIN_DPI, min(job.dpi, dpi))
        if redacted and job.status not in REVIEWABLE:
            raise HTTPException(status_code=409, detail="Nothing has been reviewed yet.")
        try:
            if redacted:
                png = await asyncio.to_thread(
                    evidence.page_png, job, page_num, render_dpi, redacted=True
                )
            else:
                png = await asyncio.to_thread(
                    _render_page_png, job.input_path, page_num, render_dpi
                )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail="That page could not be rendered.") from exc
        return Response(
            content=png,
            media_type="image/png",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.get("/api/jobs/{job_id}/source/{page_num}.png")
    async def source_page_image(
        job_id: str, page_num: int, dpi: int | None = None, redacted: bool = False
    ) -> Response:
        """A page of the original upload. ``redacted`` burns in the saved
        reviewed redactions of every section cut from it, each at its place
        on the original -- review shows sections in context this way."""
        job = require_job(job_id)
        if not job.source_page_count or not (1 <= page_num <= job.source_page_count):
            raise HTTPException(status_code=404, detail="No such page.")
        if redacted and job.status not in REVIEWABLE:
            raise HTTPException(status_code=409, detail="Nothing has been reviewed yet.")
        render_dpi = job.dpi if dpi is None else max(MIN_DPI, min(job.dpi, dpi))
        try:
            png = await asyncio.to_thread(
                evidence.source_page_png, job, page_num, render_dpi, redacted=redacted
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail="That page could not be rendered.") from exc
        return Response(
            content=png,
            media_type="image/png",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.post(
        "/api/jobs/{job_id}/nam",
        dependencies=[Depends(guard_origin)],
    )
    async def resolve_nam(job_id: str, request: Request) -> dict[str, str]:
        """Resolve a hand-typed NAM to its pseudonym, minting one if new.

        Used when the user draws a box over a number detection missed: typing
        the NAM lets that record join the same numbering, so a patient missed
        once still shares an ID with their other occurrences.
        """
        job = require_job(job_id)
        if job.status in PRE_REVIEW_STATUSES:
            raise HTTPException(status_code=409, detail="Detection is still running.")
        if job.status == "error":
            raise HTTPException(status_code=409, detail="Detection failed.")

        # Deliberately a plain dict rather than a pydantic model. FastAPI's
        # default validation handler echoes the offending input in its 422
        # detail, which would reflect a mistyped NAM -- PHI -- back to the
        # client and into any error log.
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="A NAM is required.") from exc

        raw = body.get("nam") if isinstance(body, dict) else None
        if not isinstance(raw, str):
            raise HTTPException(status_code=400, detail="A NAM is required.")

        canonical = canonical_from_text(raw)
        if canonical is None:
            # Validate before assigning: otherwise a typo would mint a phantom
            # patient that then shifts the numbering.
            raise HTTPException(status_code=400, detail="That is not a valid NAM.")

        patient_id = assign_patient_id(job, canonical)
        store.save(job)
        return {"patient_id": patient_id}

    @app.post(
        "/api/jobs/{job_id}/finalize",
        status_code=202,
        dependencies=[Depends(guard_origin)],
    )
    async def finalize(job_id: str, payload: FinalizePayload) -> dict[str, Any]:
        """Save the given review (if any), then render the saved state.

        The output is always made from the stored reviewed redactions, so what
        is downloaded is exactly what was saved -- and what evidence shows.
        """
        job = require_job(job_id)
        if job.status in PRE_REVIEW_STATUSES:
            raise HTTPException(status_code=409, detail="Detection is still running.")
        if job.status == "error":
            raise HTTPException(status_code=409, detail="Detection failed.")
        if job.status == "rendering":
            raise HTTPException(status_code=409, detail="Already rendering.")
        try:
            save_review(job, [(e.page_num, e.boxes, e.date) for e in payload.pages])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        store.save(job)
        store.start_finalize(job, asyncio.get_running_loop())
        return {"job_id": job.id, "status": "rendering", "review_revision": job.review_revision}

    @app.put("/api/jobs/{job_id}/review", dependencies=[Depends(guard_origin)])
    async def save_reviewed(job_id: str, payload: ReviewPayload) -> dict[str, Any]:
        """Save the reviewer's redactions and dates as a new revision.

        This is the authoritative state: the output PDF and every evidence
        crop are rendered from it. Row approvals whose evidence it changes
        are invalidated and listed in the response.
        """
        job = require_reviewable(job_id)
        if payload.base_revision is not None and payload.base_revision != job.review_revision:
            raise HTTPException(
                status_code=409,
                detail="The review was saved elsewhere since this page loaded. Reload to continue.",
            )
        try:
            changed, invalidated = save_review(
                job, [(e.page_num, e.boxes, e.date) for e in payload.pages]
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        store.save(job)
        return {
            "review_revision": job.review_revision,
            "reviewed_at": job.reviewed_at,
            "changed": changed,
            "invalidated_rows": invalidated,
            "status": job.status,
        }

    # -- appointment rows -------------------------------------------------

    @app.get("/api/jobs/{job_id}/rows")
    async def list_rows(job_id: str) -> dict[str, Any]:
        job = require_reviewable(job_id)
        assigned = store.assigned_version(job)
        # Documents reviewed before section rows existed get theirs now.
        if ensure_section_rows(job):
            store.save(job)
        return {
            "rows": [row_view(job, row, assigned) for row in job.rows.values()],
            "patients": job.pseudonyms(),
            "review_revision": job.review_revision,
        }

    @app.post("/api/jobs/{job_id}/rows", status_code=201, dependencies=[Depends(guard_origin)])
    async def create_row(job_id: str, request: Request) -> dict[str, Any]:
        """Add a row on a working page. Its date is prefilled from the page
        date and its patient from a NAM inside it -- both as suggestions the
        reviewer must confirm."""
        job = require_reviewable(job_id)
        body = await json_body(request)
        page_num = body.get("working_page")
        if not isinstance(page_num, int) or job.provenance_for(page_num) is None:
            raise HTTPException(status_code=400, detail="Unknown page.")
        bounds = parse_bounds(body.get("bounds"), "row region")
        if bounds is None:
            raise HTTPException(status_code=400, detail="A row needs a region.")
        prov = job.provenance_for(page_num)
        row = rowlib.new_row(
            job.id, page_num, prov.source_page, bounds,
            date_suggestion=job.page_dates.get(page_num),
            patient_suggestion=rowlib.suggest_patient(bounds, job.effective_boxes(page_num)),
        )
        job.rows[row["row_id"]] = row
        store.save(job)
        return row_view(job, row)

    @app.patch("/api/jobs/{job_id}/rows/{row_id}", dependencies=[Depends(guard_origin)])
    async def update_row(job_id: str, row_id: str, request: Request) -> dict[str, Any]:
        job = require_reviewable(job_id)
        row = require_row(job, row_id)
        body = await json_body(request)
        base = body.get("base_revision")
        if base is not None and base != row["revision"]:
            raise HTTPException(status_code=409, detail="The row changed since it was loaded.")
        if row.get("origin") == rowlib.ORIGIN_SECTION and "bounds" in body:
            raise HTTPException(status_code=400, detail="A section's row always covers the whole section.")
        updated, changed = rowlib.apply_patch(row, parse_row_patch(job, body))
        job.rows[row_id] = updated
        store.save(job)
        return {**row_view(job, updated), "changed": changed}

    @app.delete("/api/jobs/{job_id}/rows/{row_id}", status_code=204, dependencies=[Depends(guard_origin)])
    async def delete_row(job_id: str, row_id: str) -> Response:
        job = require_reviewable(job_id)
        row = require_row(job, row_id)
        if row.get("origin") == rowlib.ORIGIN_SECTION:
            raise HTTPException(status_code=409, detail="Each section is an appointment; its row cannot be deleted.")
        del job.rows[row_id]
        with job.extract_lock:
            job.extractions.pop(row_id, None)
            job.interpretations.pop(row_id, None)
            job.billing.pop(row_id, None)
        evidence.discard(job, row_id)
        store.save(job)
        return Response(status_code=204)

    # -- doctor profiles ------------------------------------------------------

    @app.get("/api/doctors")
    async def list_doctors(include_archived: bool = False) -> dict[str, Any]:
        return {"profiles": doctors.list(include_archived=include_archived)}

    @app.post("/api/doctors", status_code=201, dependencies=[Depends(guard_origin)])
    async def create_doctor(request: Request) -> dict[str, Any]:
        try:
            content = doclib.validate_content(await json_body(request), parse_practice_field)
            return doctors.create(content)
        except doclib.ProfileError as exc:
            raise profile_error(exc) from exc

    @app.get("/api/doctors/{profile_id}")
    async def get_doctor(profile_id: str) -> dict[str, Any]:
        profile = doctors.get(profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Unknown doctor profile.")
        return profile

    @app.get("/api/doctors/{profile_id}/versions/{version}")
    async def get_doctor_version(profile_id: str, version: int) -> dict[str, Any]:
        found = doctors.version(profile_id, version)
        if found is None:
            raise HTTPException(status_code=404, detail="Unknown profile version.")
        return found

    @app.put("/api/doctors/{profile_id}", dependencies=[Depends(guard_origin)])
    async def save_doctor(profile_id: str, request: Request) -> dict[str, Any]:
        """Save an edit as a new, immutable version. Documents keep the
        version they were assigned until they adopt this one explicitly."""
        body = await json_body(request)
        base = body.get("base_version")
        try:
            content = doclib.validate_content(body, parse_practice_field)
            return doctors.save_version(profile_id, content, base if isinstance(base, int) else None)
        except doclib.ProfileError as exc:
            raise profile_error(exc) from exc

    @app.post("/api/doctors/{profile_id}/duplicate", status_code=201, dependencies=[Depends(guard_origin)])
    async def duplicate_doctor(profile_id: str, request: Request) -> dict[str, Any]:
        body = await json_body(request)
        name = body.get("display_name")
        try:
            return doctors.duplicate(profile_id, name if isinstance(name, str) and name.strip() else None)
        except doclib.ProfileError as exc:
            raise profile_error(exc) from exc

    @app.post("/api/doctors/{profile_id}/archive", dependencies=[Depends(guard_origin)])
    async def archive_doctor(profile_id: str) -> dict[str, Any]:
        try:
            return doctors.set_status(profile_id, doclib.ARCHIVED)
        except doclib.ProfileError as exc:
            raise profile_error(exc) from exc

    @app.post("/api/doctors/{profile_id}/restore", dependencies=[Depends(guard_origin)])
    async def restore_doctor(profile_id: str) -> dict[str, Any]:
        try:
            return doctors.set_status(profile_id, doclib.ACTIVE)
        except doclib.ProfileError as exc:
            raise profile_error(exc) from exc

    @app.put("/api/jobs/{job_id}/doctor", dependencies=[Depends(guard_origin)])
    async def assign_document_doctor(job_id: str, request: Request) -> dict[str, Any]:
        """Assign the document's doctor, or adopt a newer version of the same
        profile. Redactions, rows, extraction results and reviews are kept."""
        job = require_job(job_id)
        if job.status in {"queued", "detecting", "rendering"}:
            raise HTTPException(status_code=409, detail="Wait for the current step to finish.")
        body = await json_body(request)
        try:
            version = doctors.resolve_for_assignment(body.get("profile_id"), body.get("version"))
        except doclib.ProfileError as exc:
            raise profile_error(exc) from exc
        if job.doctor and (job.doctor["profile_id"], job.doctor["version"]) == (
            version["profile_id"], version["version"]
        ):
            return {"doctor": doctor_view(job), "changed": False}
        assign_doctor(job, version)
        store.save(job)
        return {"doctor": doctor_view(job), "changed": True, **job.practice}

    # -- extraction -------------------------------------------------------

    @app.get("/api/extraction")
    async def extraction_config() -> dict[str, Any]:
        """Whether cloud extraction is set up, and with what. Never the key."""
        return {
            "configured": store.extraction_configured(),
            "key_variable": ex.API_KEY_ENV,
            # "environment", "saved" (typed in the app) or None.
            "key_source": ex.key_source(store.key_file),
            "models": list(ex.MODELS),
            "default_model": ex.DEFAULT_MODEL,
            "routing": dict(ex.PROVIDER_ROUTING),
            "prompt_version": ex.PROMPT_VERSION,
        }

    @app.put("/api/extraction/key", dependencies=[Depends(guard_origin)])
    async def save_api_key(request: Request) -> dict[str, Any]:
        """Save an OpenRouter key on this machine. It is never sent back."""
        key = (await json_body(request)).get("key")
        if not isinstance(key, str) or not ex.KEY_RE.match(key.strip()):
            raise HTTPException(status_code=400, detail="That does not look like an OpenRouter API key.")
        await asyncio.to_thread(ex.write_key_file, store.key_file, key.strip())
        return await extraction_config()

    @app.delete("/api/extraction/key", dependencies=[Depends(guard_origin)])
    async def remove_api_key() -> dict[str, Any]:
        """Forget the key saved in the app (an environment key is untouched)."""
        store.key_file.unlink(missing_ok=True)
        return await extraction_config()

    def parse_model(body: dict[str, Any]) -> str:
        model = body.get("model", ex.DEFAULT_MODEL)
        if model not in ex.MODEL_IDS:
            raise HTTPException(status_code=400, detail="That model is not offered.")
        return model

    def require_configured() -> None:
        if not store.extraction_configured():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Cloud extraction is not configured: add an OpenRouter API key in Appointment rows, "
                    f"or set {ex.API_KEY_ENV} for the server and restart it."
                ),
            )

    @app.post("/api/jobs/{job_id}/rows/{row_id}/extract", status_code=202, dependencies=[Depends(guard_origin)])
    async def extract_row(job_id: str, row_id: str, request: Request) -> dict[str, Any]:
        """Run one row again. ``stage``:

        * ``"transcription"`` (default) -- a new transcription, then
          interpretation straight after;
        * ``"interpretation"`` -- a new interpretation of the current
          transcription (and its correction, if any), without transcribing
          again. This is how a failed or outdated interpretation is retried.

        Earlier attempts and reviews are always kept.
        """
        job = require_reviewable(job_id)
        require_row(job, row_id)
        body = await json_body(request)
        model = parse_model(body)
        stage = body.get("stage", "transcription")
        if stage not in ("transcription", "interpretation"):
            raise HTTPException(status_code=400, detail="Unknown stage.")
        require_configured()
        if stage == "interpretation":
            outcome = store.start_interpretation(job, [row_id], model)
        else:
            # A row whose values are known is made ready on the way.
            _, skipped = await prepare_rows(job, [row_id])
            if row_id in skipped:
                raise HTTPException(status_code=409, detail=f"Not started: the row {skipped[row_id]}.")
            outcome = store.start_extraction(job, [row_id], model)
        if row_id in outcome["skipped"]:
            raise HTTPException(status_code=409, detail=f"Not started: {outcome['skipped'][row_id]}.")
        return {**outcome, "stage": stage}

    @app.post("/api/jobs/{job_id}/extract", status_code=202, dependencies=[Depends(guard_origin)])
    async def extract_rows(job_id: str, request: Request) -> dict[str, Any]:
        """One click, the whole document: prepare, transcribe, interpret.

        Each row gets only the work it needs, so this is safe to repeat:

        * rows without a current transcription are transcribed, and then
          interpreted automatically -- no acceptance step in between;
        * rows with a current transcription (reviewed or not) reuse it,
          corrections included, and are interpreted if their interpretation
          is missing, failed or outdated;
        * rows already current, or running, are skipped with the reason.
        """
        job = require_reviewable(job_id)
        model = parse_model(await json_body(request))
        require_configured()
        # Every row whose patient and date are known is made ready first.
        prepared, unprepared = await prepare_rows(job, list(job.rows))
        doctor, version = store.assigned_version(job)
        transcribe: list[str] = []
        interpret: list[str] = []
        skipped: dict[str, str] = dict(unprepared)
        blocked_reason: str | None = None
        for row_id, row in list(job.rows.items()):
            if row_id in skipped:
                continue
            with job.extract_lock:
                t_state = exlib.state_of(row, job.extractions.get(row_id), job.effective_boxes(row["working_page"]))
            if t_state == "running":
                skipped[row_id] = "transcribing"
            elif t_state in exlib.RETRYABLE_STATES:
                transcribe.append(row_id)
            elif t_state in ("unreviewed", "reviewed"):
                inputs, blocked = store.interpretation_inputs(job, row_id, doctor, version, resolved=True)
                with job.extract_lock:
                    i_state = ilib.state_of(job.interpretations.get(row_id), inputs, blocked)
                if blocked:
                    blocked_reason = blocked
                    skipped[row_id] = "interpretation blocked"
                elif i_state == "running":
                    skipped[row_id] = "interpreting"
                elif i_state in ilib.RETRYABLE_STATES:
                    interpret.append(row_id)
                else:
                    skipped[row_id] = "already interpreted"
            else:
                skipped[row_id] = "not ready"
        transcribing = store.start_extraction(job, transcribe, model)
        interpreting = store.start_interpretation(job, interpret, model)
        if doctor is None:
            blocked_reason = ilib.BLOCKED_NO_DOCTOR
        elif blocked_reason is None and version is not None and not (
            version["instructions"].strip() or version["examples"]
        ):
            blocked_reason = ilib.BLOCKED_EMPTY_PROFILE
        return {
            "prepared": prepared,
            "transcribing": transcribing["queued"],
            "interpreting": interpreting["queued"],
            # Kept for older clients: every row that got work.
            "queued": transcribing["queued"] + interpreting["queued"],
            "skipped": {**skipped, **transcribing["skipped"], **interpreting["skipped"]},
            "interpretation_blocked": (
                {"reason": blocked_reason, "message": ilib.BLOCKED_MESSAGES[blocked_reason]}
                if blocked_reason else None
            ),
            "requested": len(transcribe) + len(interpret),
        }

    @app.put("/api/jobs/{job_id}/rows/{row_id}/extraction/review", dependencies=[Depends(guard_origin)])
    async def review_extraction(job_id: str, row_id: str, request: Request) -> dict[str, Any]:
        """Accept the reviewed time and handwriting for the row's current result.

        The model's own output is never changed: the accepted values are stored
        beside it, with what was corrected. A time confirmed during preparation
        is never overwritten here.
        """
        job = require_reviewable(job_id)
        row = require_row(job, row_id)
        body = await json_body(request)
        try:
            time, status, text = exlib.validate_review(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        with job.extract_lock:
            record = job.extractions.get(row_id)
            attempt = exlib.current_attempt(row, record, job.effective_boxes(row["working_page"]))
            if attempt is None or attempt["attempt_id"] != body.get("attempt_id"):
                raise HTTPException(
                    status_code=409,
                    detail="This result is no longer current; reload the row before accepting.",
                )
            exlib.accept_review(row, record, attempt, time, status, text)
        store.save(job)
        return row_view(job, job.rows[row_id])

    @app.post("/api/jobs/{job_id}/rows/{row_id}/approve", dependencies=[Depends(guard_origin)])
    async def approve_row(job_id: str, row_id: str, request: Request) -> dict[str, Any]:
        """Approve the row at the revision the reviewer was looking at.

        Evidence is rendered from the saved review and frozen to disk, so a
        review must have been saved first.
        """
        job = require_reviewable(job_id)
        row = require_row(job, row_id)
        body = await json_body(request)
        if body.get("revision") != row["revision"]:
            raise HTTPException(status_code=409, detail="The row changed since it was loaded.")
        if job.reviewed is None:
            raise HTTPException(status_code=409, detail="Save the redaction review before approving rows.")
        missing = rowlib.unresolved(row)
        if missing:
            raise HTTPException(status_code=400, detail=f"Unresolved: {', '.join(missing)}.")
        refs = await asyncio.to_thread(evidence.freeze, job, row)
        if job.rows.get(row_id) is not row:
            raise HTTPException(status_code=409, detail="The row changed while approving.")
        approved = rowlib.approve(row, job.effective_boxes(row["working_page"]), job.review_revision, refs)
        job.rows[row_id] = approved
        store.save(job)
        return row_view(job, approved)

    async def prepare_rows(job: Job, row_ids: list[str]) -> tuple[list[str], dict[str, str]]:
        """Mark ready for extraction every listed row whose values are known.

        Rows already ready are left as they are; rows missing a patient or
        date are reported, not guessed. Requires a saved redaction review,
        since the evidence is rendered from it.
        """
        if job.reviewed is None:
            raise HTTPException(status_code=409, detail="Save the redaction review first.")
        prepared: list[str] = []
        skipped: dict[str, str] = {}
        for row_id in row_ids:
            row = job.rows.get(row_id)
            if row is None:
                continue
            page_boxes = job.effective_boxes(row["working_page"])
            if rowlib.is_current(row, page_boxes):
                continue
            missing = rowlib.unresolved(row)
            if missing:
                labels = {"patient_id": "patient", "date": "date"}
                skipped[row_id] = "needs " + " and ".join(labels.get(m, m) for m in missing)
                continue
            refs = await asyncio.to_thread(evidence.freeze, job, row)
            if job.rows.get(row_id) is not row:
                skipped[row_id] = "changed while preparing"
                continue
            job.rows[row_id] = rowlib.approve(
                row, page_boxes, job.review_revision, refs, method=rowlib.METHOD_AUTOMATIC,
            )
            prepared.append(row_id)
        if prepared:
            store.save(job)
        return prepared, skipped

    @app.post("/api/jobs/{job_id}/prepare", dependencies=[Depends(guard_origin)])
    async def prepare(job_id: str) -> dict[str, Any]:
        """Mark every row whose patient and date are known ready for extraction."""
        job = require_reviewable(job_id)
        prepared, skipped = await prepare_rows(job, list(job.rows))
        return {"prepared": prepared, "skipped": skipped}

    @app.get("/api/jobs/{job_id}/rows/{row_id}/evidence/{kind}.png")
    async def row_evidence(job_id: str, row_id: str, kind: str) -> Response:
        """The row or handwriting crop as it stands now, from the saved review."""
        job = require_reviewable(job_id)
        row = require_row(job, row_id)
        if kind not in evidence.KINDS:
            raise HTTPException(status_code=404, detail="Unknown evidence.")
        bounds = evidence.region_of(row, kind)
        if bounds is None:
            raise HTTPException(status_code=404, detail="No such region yet.")
        png, _ = await asyncio.to_thread(evidence.render_region, job, row["working_page"], bounds)
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/jobs/{job_id}/evidence/{row_id}/{name}")
    async def frozen_evidence(job_id: str, row_id: str, name: str) -> FileResponse:
        """An evidence file frozen at approval, as referenced by the manifest."""
        job = require_job(job_id)
        require_row(job, row_id)
        path = (job.dir / "evidence" / row_id / name).resolve()
        if (job.dir / "evidence").resolve() not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="Unknown evidence.")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})

    # -- billing lines and confirmation --------------------------------------

    def billing_change(job: Job, row: dict[str, Any], body: dict[str, Any], *, lines: bool) -> dict[str, Any]:
        doctor, version = store.assigned_version(job)
        view = billing.compute(job, row, doctor, version)
        try:
            billing.check_base(view, body, lines=lines)
        except billing.ConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return view

    def parsed_lines(body: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
        try:
            return billing.parse_lines(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/jobs/{job_id}/rows/{row_id}/billing", dependencies=[Depends(guard_origin)])
    async def save_billing(job_id: str, row_id: str, request: Request) -> dict[str, Any]:
        """Save an appointment's billing lines as the person edited them.

        Stored beside the model's proposal, which is never changed. Ends a
        confirmation: the person confirms again. No model is called.
        """
        job = require_reviewable(job_id)
        row = require_row(job, row_id)
        body = await json_body(request)
        acts, no_billing = parsed_lines(body)
        with job.extract_lock:
            view = billing_change(job, row, body, lines=True)
            billing.save_lines(job, row_id, view, acts, no_billing)
        store.save(job)
        return row_view(job, job.rows[row_id])

    @app.post("/api/jobs/{job_id}/rows/{row_id}/billing/confirm", dependencies=[Depends(guard_origin)])
    async def confirm_billing(job_id: str, row_id: str, request: Request) -> dict[str, Any]:
        """Save the lines shown and mark the appointment confirmed. Refused,
        with the reasons, while the appointment could not be exported."""
        job = require_reviewable(job_id)
        row = require_row(job, row_id)
        body = await json_body(request)
        acts, no_billing = parsed_lines(body)
        if not acts and not no_billing:
            raise HTTPException(
                status_code=400,
                detail="Add at least one code, or mark the appointment as having no billing acts.",
            )
        empty = [str(i) for i, act in enumerate(acts, start=1) if not act["code"]]
        if empty:
            raise HTTPException(
                status_code=400,
                detail=f"Line {', '.join(empty)} has no procedure code: enter one or remove the line.",
            )
        with job.extract_lock:
            view = billing_change(job, row, body, lines=True)
            edit = billing.save_lines(job, row_id, view, acts, no_billing)
            after = billing.compute(job, row, *store.assigned_version(job))
            if not after["problems"]:
                billing.confirm(job, row_id, edit)
        # The lines are kept either way: they are the person's work.
        store.save(job)
        if after["problems"]:
            raise HTTPException(
                status_code=409,
                detail="Saved, but not confirmed: " + "; ".join(after["problems"]) + ".",
            )
        return row_view(job, job.rows[row_id])

    @app.post("/api/jobs/{job_id}/rows/{row_id}/billing/reopen", dependencies=[Depends(guard_origin)])
    async def reopen_billing(job_id: str, row_id: str, request: Request) -> dict[str, Any]:
        """Take back a confirmation to edit the appointment again."""
        job = require_reviewable(job_id)
        row = require_row(job, row_id)
        body = await json_body(request)
        with job.extract_lock:
            view = billing_change(job, row, body, lines=False)
            if view["confirmation"] is None:
                raise HTTPException(status_code=409, detail="This appointment is not confirmed.")
            billing.reopen(job, row_id)
        store.save(job)
        return row_view(job, job.rows[row_id])

    @app.post("/api/jobs/{job_id}/rows/{row_id}/billing/reset", dependencies=[Depends(guard_origin)])
    async def reset_billing(job_id: str, row_id: str, request: Request) -> dict[str, Any]:
        """Drop back to the model's current proposal. The manual lines, and
        any confirmation of them, stay in the history."""
        job = require_reviewable(job_id)
        row = require_row(job, row_id)
        body = await json_body(request)
        with job.extract_lock:
            view = billing_change(job, row, body, lines=False)
            if view["source"] != "manual":
                raise HTTPException(status_code=409, detail="There are no manual billing lines to discard.")
            billing.reset(job, row_id, view)
        store.save(job)
        return row_view(job, job.rows[row_id])

    @app.get("/api/jobs/{job_id}/billing.csv")
    async def billing_export(job_id: str) -> Response:
        """The practice's billing CSV: one line per proposed act.

        Built from saved current proposals only -- no model request, nothing
        approved or changed. If any appointment cannot be exported correctly,
        nothing is: the reply lists each affected row and why (409).
        """
        job = require_reviewable(job_id)
        doctor, version = store.assigned_version(job)
        result = await asyncio.to_thread(billing_csv.build, job, doctor, version)
        if result.problems:
            count = len(result.problems)
            return Response(
                status_code=409,
                media_type="application/json",
                content=json.dumps({
                    "detail": (
                        f"{count} appointment{'s' if count != 1 else ''} cannot be exported yet: "
                        + "; ".join(f"{p.label} {p.reason}" if p.row_id else p.reason for p in result.problems)
                        + "."
                    ),
                    "problems": [{"row_id": p.row_id, "label": p.label, "reason": p.reason}
                                 for p in result.problems],
                }),
                headers={"Cache-Control": "no-store"},
            )
        return Response(
            content=result.content,
            media_type="text/csv; charset=utf-8",
            headers={
                # No patient data in the name: the document's own filename
                # often is a patient's name.
                "Content-Disposition": f'attachment; filename="billing-{job.id[:8]}.csv"',
                "Cache-Control": "no-store",
                "X-Billing-Lines": str(result.lines),
                "X-Billing-No-Notation-Rows": str(len(result.no_billing_rows)),
                "X-Billing-Confirmed-Rows": str(len(result.confirmed_rows)),
            },
        )

    @app.get("/api/jobs/{job_id}/manifest.json")
    async def row_manifest(job_id: str) -> Response:
        """Approved rows as a versioned JSON manifest; unresolved rows are
        listed as excluded. No NAM, name, OCR text or filename."""
        job = require_reviewable(job_id)
        data = await asyncio.to_thread(manifest.build, job, store.assigned_version(job))
        return Response(
            content=json.dumps(data, indent=2, ensure_ascii=False),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="rows-{job.id[:8]}.json"',
                "Cache-Control": "no-store",
            },
        )

    @app.get("/api/jobs/{job_id}/result")
    async def result(job_id: str) -> FileResponse:
        job = require_job(job_id)
        if job.status != "done" or not job.output_path.exists():
            raise HTTPException(status_code=409, detail="The redacted PDF is not ready.")
        return FileResponse(
            job.output_path,
            media_type="application/pdf",
            filename="redacted.pdf",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.delete(
        "/api/jobs/{job_id}",
        status_code=204,
        dependencies=[Depends(guard_origin)],
    )
    async def delete_job(job_id: str) -> Response:
        if not store.delete(job_id):
            raise HTTPException(status_code=404, detail="Unknown job.")
        return Response(status_code=204)

    # -- static frontend (registered last so /api wins) --------------------

    assets_dir = dist_dir / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str) -> Response:
        index = dist_dir / "index.html"
        if not index.is_file():
            return HTMLResponse(_MISSING_BUILD_HTML, status_code=503)
        if path:
            candidate = (dist_dir / path).resolve()
            root = dist_dir.resolve()
            if candidate.is_file() and root in candidate.parents:
                return FileResponse(candidate)
        return FileResponse(index)

    return app


app = create_app()
