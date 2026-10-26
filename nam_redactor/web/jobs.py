"""Documents, their saved state, and the worker that runs the slow phases.

One process, one user. OCR is CPU-bound and the Paddle predictor is not
documented as thread-safe, so all inference is serialized through a
single-worker executor -- which also means one long-lived engine amortizes the
model load that dominates the first run.

A *job* is one uploaded document. Its state is saved to disk after every
change (see :mod:`nam_redactor.web.storage`), so closing the app loses
nothing: documents are reopened on the next start and removed only when the
user deletes them.
"""
from __future__ import annotations

import asyncio
import copy
import datetime
import hashlib
import re
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..names import NameModel, default_name_model
from ..provenance import PageProvenance
from ..zones import crop_to_zones, whole_page_provenance
from ..redactor import (
    PageDetection,
    PageProgress,
    default_engine,
    detect_nam_pages,
    render_redacted_pdf,
)
from .. import extraction as ex
from . import doctors as doclib
from . import extraction as exlib
from . import interpretation as ilib
from . import rows as rowlib
from . import storage

#: Temp dirs left by versions that kept documents only for the session.
JOB_DIR_PREFIX = "nam-redact-"
#: The upload, exactly as received. Never modified.
JOB_INPUT_NAME = "original.pdf"
JOB_OUTPUT_NAME = "redacted.pdf"
#: The document rebuilt from the user's zones, one page per zone.
JOB_ZONES_NAME = "zones.pdf"

#: Every status before detection has produced boxes. Endpoints that need
#: detection results refuse all of them alike.
PRE_REVIEW_STATUSES = frozenset({"awaiting_zones", "queued", "detecting"})

MAX_UPLOAD_BYTES = 200 * 1024 * 1024

#: Rows extracted at once. Bounded so a batch cannot flood the provider.
EXTRACTION_CONCURRENCY = 3
#: Waits before the 2nd and 3rd try of a transient failure (seconds).
RETRY_DELAYS_S = (2.0, 8.0)
MAX_PAGES = 500
#: Zones become pages, so this bounds the derived document the way
#: ``MAX_PAGES`` bounds the upload.
MAX_ZONES = 2000
MIN_DPI = 72
MAX_DPI = 400
DEFAULT_DPI = 200

#: Event types each phase's stream cares about, and the ones that end it.
#: The two phases share one subscriber list, so each stream filters to its own
#: set -- otherwise a detect-phase reader would see finalize events.
PHASE_EVENTS: dict[str, frozenset[str]] = {
    "detect": frozenset({"queued", "start", "page", "awaiting_review", "error"}),
    "finalize": frozenset({"rendering", "done", "error"}),
}
PHASE_TERMINAL: dict[str, frozenset[str]] = {
    "detect": frozenset({"awaiting_review", "error"}),
    "finalize": frozenset({"done", "error"}),
}

#: History is replayed to late subscribers, so it is capped only to keep a
#: pathological job from growing without bound. Far above any real page count.
MAX_HISTORY_EVENTS = 4096

#: A pseudonym, and the only identity ever sent to a client. Validated on the
#: way in because it is drawn onto the output image.
PATIENT_ID_RE = re.compile(r"^Patient_\d{4,}$")

#: A page date as the browser's date input sends it.
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_page_date(value: Any) -> str | None:
    """``YYYY-MM-DD`` if ``value`` is a real calendar date, else ``None``."""
    if not isinstance(value, str) or not _DATE_RE.match(value):
        return None
    try:
        datetime.date.fromisoformat(value)
    except ValueError:
        return None
    return value


#: Practice identifiers typed at upload: licence, group and sector numbers.
#: Their exact formats vary, so the check is only that they are short and
#: plain -- letters, digits, spaces and dashes.
PRACTICE_FIELDS = ("licence_number", "group_number", "sector_number")
_PRACTICE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 \-]{0,29}$")


def parse_practice_field(value: Any) -> str | None:
    """Trimmed value, ``""`` when left blank, ``None`` when malformed."""
    if value is None:
        return ""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return ""
    return value if _PRACTICE_RE.match(value) else None


#: The only box kind besides the default (a NAM). Name boxes get no label.
NAME_KIND = "name"


def count_kinds(pages: Sequence[PageDetection]) -> tuple[int, int]:
    """``(nam_boxes, name_boxes)`` across all pages."""
    names = sum(
        1 for page in pages for box in page.boxes if box.get("kind") == NAME_KIND
    )
    total = sum(len(page.boxes) for page in pages)
    return total - names, names


def assign_patient_id(job: "Job", canonical: str) -> str:
    """Pseudonym for a canonical NAM, minting one if this patient is new.

    Numbering continues from whatever detection already assigned, so a typed
    NAM joins the same sequence as the detected ones rather than starting a
    second series.
    """
    with job.lock:
        existing = job.patient_ids.get(canonical)
        if existing is None:
            existing = f"Patient_{len(job.patient_ids) + 1:04d}"
            job.patient_ids[canonical] = existing
        return existing


def resolve_box_ids(
    job: "Job", page_num: int, raw: Any
) -> tuple[dict[str, Any], ...]:
    """Sanitize a page's boxes and restore their patient IDs.

    On boxes the server detected, its own stored value wins: a client that
    dropped the field would otherwise produce unlabelled output after the
    review screen had shown IDs, and one that forged it would mislabel a
    patient -- the review would have been lying either way.

    A box with no stored counterpart is one the user drew, so whatever ID the
    client supplies for it stands. That is how a typed NAM's pseudonym
    survives from the review screen to the PDF.
    """
    stored: dict[tuple[float, float, float, float], str] = {}
    for page in job.pages:
        if page.page_num != page_num:
            continue
        for box in page.boxes:
            patient_id = box.get("patient_id")
            if patient_id:
                stored[(box["x1"], box["y1"], box["x2"], box["y2"])] = patient_id

    merged: list[dict[str, Any]] = []
    for box in coerce_boxes(raw):
        authoritative = stored.get((box["x1"], box["y1"], box["x2"], box["y2"]))
        if authoritative is not None and box.get("kind") != NAME_KIND:
            box = {**box, "patient_id": authoritative}
        merged.append(box)
    return tuple(merged)


def _scrub(exc: BaseException) -> str:
    """A generic, path-free message safe to hand a client or a log.

    Exception text from pymupdf/paddle can embed file paths and, in principle,
    document content. Neither belongs in a browser payload.
    """
    name = type(exc).__name__
    if isinstance(exc, FileNotFoundError):
        return "The document could not be read."
    if isinstance(exc, ValueError):
        return "The document could not be processed."
    return f"Redaction failed ({name})."


def coerce_boxes(raw: Any) -> tuple[dict[str, float], ...]:
    """Validate an untrusted box list from the browser into normalized boxes.

    Anything malformed is dropped rather than rejected: a single bad rectangle
    should not cost the user their whole edit session.
    """
    if not isinstance(raw, list):
        return ()
    out: list[dict[str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            x1 = float(item["x1"])
            y1 = float(item["y1"])
            x2 = float(item["x2"])
            y2 = float(item["y2"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(v == v for v in (x1, y1, x2, y2)):  # NaN
            continue
        left, right = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        top, bottom = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        if right - left <= 0 or bottom - top <= 0:
            continue
        try:
            confidence = float(item.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        box: dict[str, Any] = {
            "x1": round(left, 6),
            "y1": round(top, 6),
            "x2": round(right, 6),
            "y2": round(bottom, 6),
            "confidence": max(0.0, min(1.0, confidence)),
        }
        # Only ever present when valid: attaching a None would break callers
        # (and tests) that compare box dicts for equality.
        if item.get("kind") == NAME_KIND:
            # A name is never stamped, so a name box cannot carry an ID.
            box["kind"] = NAME_KIND
        else:
            patient_id = item.get("patient_id")
            if isinstance(patient_id, str) and PATIENT_ID_RE.match(patient_id):
                box["patient_id"] = patient_id
        out.append(box)
    return tuple(out)


@dataclass
class Job:
    """One uploaded document: files, detections, review, rows, and progress."""

    id: str
    filename: str  # display only -- never used to build a path
    dpi: int
    dir: Path
    #: What detection and rendering read. The upload until zones are
    #: applied, then the document rebuilt from them.
    input_path: Path
    output_path: Path
    status: str = "awaiting_zones"
    total_pages: int | None = None
    #: Pages of the upload itself; ``total_pages`` counts working pages.
    source_page_count: int | None = None
    #: Initial detections. Never edited: the reviewer's version is ``reviewed``.
    pages: list[PageDetection] = field(default_factory=list)
    per_page_counts: list[int] = field(default_factory=list)
    total_nams: int = 0
    total_names: int = 0
    error: str | None = None
    created_at: float = 0.0
    original_sha256: str | None = None
    #: One entry per working page: which source page and pixels it came from.
    provenance: list[PageProvenance] = field(default_factory=list)
    #: The zone step's input, keyed by source page: ``{"zones": {page: [box]},
    #: "dates": {page: "YYYY-MM-DD"}}``. Kept so zones can be traced, and so
    #: an interrupted detection can be restarted with them.
    zone_request: dict[str, Any] | None = None
    #: Reviewed redaction boxes per working page, once the reviewer saves.
    #: ``None`` means nothing has been reviewed and the detections stand.
    reviewed: dict[int, tuple[dict[str, Any], ...]] | None = None
    review_revision: int = 0
    reviewed_at: str | None = None
    #: Appointment rows by ``row_id``, in creation order.
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Extraction attempts and reviews by ``row_id`` (see
    #: :mod:`nam_redactor.web.extraction`). Kept apart from ``rows`` so a
    #: worker thread writing a result never races a reviewer editing a row;
    #: guarded by ``extract_lock``.
    extractions: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Interpretation attempts by ``row_id`` (see
    #: :mod:`nam_redactor.web.interpretation`): proposed billing, a separate
    #: stage from transcription. Also guarded by ``extract_lock``.
    interpretations: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Manual billing lines and billing confirmations by ``row_id`` (see
    #: :mod:`nam_redactor.web.billing`), kept apart from the model's output.
    #: Also guarded by ``extract_lock``.
    billing: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: canonical NAM -> pseudonym, shared by detection and typed NAMs so the
    #: two cannot mint conflicting IDs. Never leaves the server.
    patient_ids: dict[str, str] = field(default_factory=dict)
    #: Output page number -> ``YYYY-MM-DD``. Set from the zone step, where
    #: each zone takes its source page's date, then edited during review.
    #: Not drawn on the output; kept for later use.
    page_dates: dict[int, str] = field(default_factory=dict)
    #: Licence, group and sector numbers (see ``PRACTICE_FIELDS``). Set from
    #: the assigned doctor profile's snapshot; documents from before doctor
    #: profiles keep the values that were typed at upload.
    practice: dict[str, str] = field(default_factory=dict)
    #: The doctor profile this whole document belongs to (see
    #: :func:`assign_doctor`), or ``None`` for documents from before profiles
    #: existed and not yet assigned.
    doctor: dict[str, Any] | None = None
    #: Earlier assignments, oldest first, so a change is never lost.
    doctor_history: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    extract_lock: threading.RLock = field(default_factory=threading.RLock)
    #: Set once the document is deleted, so a late worker cannot re-create it.
    deleted: bool = False
    loop: asyncio.AbstractEventLoop | None = None

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = time.time()

    # -- reviewed state ---------------------------------------------------

    def _extractions_snapshot(self) -> dict[str, Any]:
        with self.extract_lock:
            return copy.deepcopy(self.extractions)

    def _interpretations_snapshot(self) -> dict[str, Any]:
        with self.extract_lock:
            return copy.deepcopy(self.interpretations)

    def _billing_snapshot(self) -> dict[str, Any]:
        with self.extract_lock:
            return copy.deepcopy(self.billing)

    def effective_boxes(self, page_num: int) -> tuple[dict[str, Any], ...]:
        """The redactions that stand for a page: reviewed if saved, else
        detected. Everything downstream -- output, evidence -- reads this."""
        if self.reviewed is not None and page_num in self.reviewed:
            return self.reviewed[page_num]
        for page in self.pages:
            if page.page_num == page_num:
                return page.boxes
        return ()

    def effective_pages(self) -> list[PageDetection]:
        return [
            PageDetection(page.page_num, page.width, page.height, self.effective_boxes(page.page_num))
            for page in self.pages
        ]

    def provenance_for(self, page_num: int) -> PageProvenance | None:
        for prov in self.provenance:
            if prov.working_page == page_num:
                return prov
        return None

    def pseudonyms(self) -> list[str]:
        """Every patient pseudonym this document knows, in numbering order."""
        found = set(self.patient_ids.values())
        for page in self.pages:
            found.update(b["patient_id"] for b in self.effective_boxes(page.page_num) if b.get("patient_id"))
        return sorted(found, key=lambda pid: (len(pid), pid))

    # -- persistence ------------------------------------------------------

    def to_record(self) -> dict[str, Any]:
        """Everything but the NAM -> pseudonym map, which is saved apart."""
        return {
            "id": self.id,
            "filename": self.filename,
            "dpi": self.dpi,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "original_sha256": self.original_sha256,
            "source_page_count": self.source_page_count,
            "total_pages": self.total_pages,
            "working_file": self.input_path.name,
            "practice": self.practice,
            "doctor": self.doctor,
            "doctor_history": self.doctor_history,
            "zone_request": self.zone_request,
            "provenance": [prov.to_dict() for prov in self.provenance],
            "page_dates": {str(k): v for k, v in self.page_dates.items()},
            "detections": [
                {"page_num": p.page_num, "width": p.width, "height": p.height, "boxes": list(p.boxes)}
                for p in self.pages
            ],
            "reviewed": None if self.reviewed is None else {
                str(k): list(v) for k, v in self.reviewed.items()
            },
            "review_revision": self.review_revision,
            "reviewed_at": self.reviewed_at,
            "per_page_counts": self.per_page_counts,
            "total_nams": self.total_nams,
            "total_names": self.total_names,
            "rows": list(self.rows.values()),
            "extractions": self._extractions_snapshot(),
            "interpretations": self._interpretations_snapshot(),
            "billing": self._billing_snapshot(),
        }

    @classmethod
    def from_record(cls, directory: Path, record: dict[str, Any], identities: dict[str, str]) -> "Job":
        job = cls(
            id=record["id"],
            filename=record["filename"],
            dpi=int(record["dpi"]),
            dir=directory,
            input_path=directory / record.get("working_file", JOB_INPUT_NAME),
            output_path=directory / JOB_OUTPUT_NAME,
            status=record["status"],
            total_pages=record.get("total_pages"),
            source_page_count=record.get("source_page_count"),
            error=record.get("error"),
            created_at=float(record.get("created_at") or 0.0),
            original_sha256=record.get("original_sha256"),
            practice=dict(record.get("practice") or {}),
            doctor=record.get("doctor"),
            doctor_history=list(record.get("doctor_history") or []),
            zone_request=record.get("zone_request"),
            provenance=[PageProvenance.from_dict(d) for d in record.get("provenance", [])],
            page_dates={int(k): v for k, v in (record.get("page_dates") or {}).items()},
            pages=[
                PageDetection(d["page_num"], d["width"], d["height"], tuple(d["boxes"]))
                for d in record.get("detections", [])
            ],
            review_revision=int(record.get("review_revision", 0)),
            reviewed_at=record.get("reviewed_at"),
            per_page_counts=list(record.get("per_page_counts", [])),
            total_nams=int(record.get("total_nams", 0)),
            total_names=int(record.get("total_names", 0)),
            patient_ids=dict(identities),
        )
        reviewed = record.get("reviewed")
        if reviewed is not None:
            job.reviewed = {int(k): tuple(v) for k, v in reviewed.items()}
        job.rows = {row["row_id"]: row for row in record.get("rows", [])}
        job.extractions = dict(record.get("extractions") or {})
        job.interpretations = dict(record.get("interpretations") or {})
        job.billing = dict(record.get("billing") or {})
        return job

    # -- event fan-out ----------------------------------------------------

    def publish(self, event: dict[str, Any]) -> None:
        """Append to history and hand to every live subscriber.

        Called from the worker thread, so the queues are only ever touched
        through ``call_soon_threadsafe``.
        """
        with self.lock:
            self.events.append(event)
            if len(self.events) > MAX_HISTORY_EVENTS:
                del self.events[0]
            subscribers = list(self.subscribers)
            loop = self.loop

        if loop is None or loop.is_closed():
            return
        for queue in subscribers:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                pass  # loop went away between the check and the call

    def subscribe(
        self,
        allowed: frozenset[str],
        terminal: frozenset[str],
    ) -> tuple[list[dict[str, Any]], asyncio.Queue]:
        """Snapshot history and register for live events, atomically.

        The two must happen under one lock: a subscriber that snapshotted and
        then registered would miss everything published in between, and the
        upload -> subscribe race makes that window real.

        The queue is only registered when this phase is still in flight. When
        it has already finished, the snapshot holds the terminal event and the
        caller just replays it -- registering would leave the stream hanging
        forever waiting for a phase that is over.
        """
        queue: asyncio.Queue = asyncio.Queue()
        with self.lock:
            snapshot = [ev for ev in self.events if ev.get("type") in allowed]
            finished = any(ev.get("type") in terminal for ev in snapshot)
            if not finished:
                self.subscribers.append(queue)
        return snapshot, queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self.lock:
            if queue in self.subscribers:
                self.subscribers.remove(queue)

    def page_boxes(self) -> dict[int, tuple[dict[str, float], ...]]:
        return {page.page_num: page.boxes for page in self.pages}


class JobStore:
    """Registry of saved documents plus the single OCR worker.

    ``dir_root`` is the data directory (see :func:`storage.default_data_dir`);
    each document lives in ``<dir_root>/documents/<id>/``.
    """

    def __init__(
        self,
        engine_factory: Callable[[], Any] | None = None,
        dir_root: Path | None = None,
        name_model_factory: Callable[[], NameModel | None] | None = None,
        extractor_factory: Callable[[], Any] | None = None,
        extraction_concurrency: int = EXTRACTION_CONCURRENCY,
        retry_delays: Sequence[float] = RETRY_DELAYS_S,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        #: Serializes writes of one document's files; saves come from both
        #: request handlers and the worker thread.
        self._save_lock = threading.RLock()
        self._engine_factory = engine_factory or default_engine
        self._engine: Any | None = None
        self._name_model_factory = name_model_factory or default_name_model
        self._name_model: NameModel | None = None
        self._name_model_loaded = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nam-ocr")
        # Cloud extraction is I/O-bound and independent of OCR: its own pool,
        # bounded so a large batch cannot flood the provider.
        self._extract_executor = ThreadPoolExecutor(
            max_workers=max(1, extraction_concurrency), thread_name_prefix="nam-extract"
        )
        self._dir_root = dir_root or storage.default_data_dir()
        #: An OpenRouter key typed in the app; the environment variable wins.
        self.key_file = self._dir_root / storage.PRIVATE_DIR / storage.API_KEY_NAME
        self._extractor_factory = extractor_factory or (lambda: ex.OpenRouterExtractor(key_file=self.key_file))
        self._extractor: Any | None = None
        self._retry_delays = tuple(retry_delays)
        self._sleep = sleep
        #: Doctor profiles, beside the documents. Interpretation reads the
        #: exact version a document is assigned from here.
        self.doctors = doclib.DoctorStore(self._dir_root)
        self._docs_root = self._dir_root / storage.DOCUMENTS_DIR

    @property
    def data_dir(self) -> Path:
        return self._dir_root

    # -- lifecycle --------------------------------------------------------

    def _get_engine(self) -> Any:
        """Build the OCR engine once and reuse it for every job."""
        with self._lock:
            if self._engine is None:
                self._engine = self._engine_factory()
            return self._engine

    def _get_name_model(self) -> NameModel | None:
        """Load the NER model once. ``None`` (spaCy missing) is cached too."""
        with self._lock:
            if not self._name_model_loaded:
                self._name_model = self._name_model_factory()
                self._name_model_loaded = True
            return self._name_model

    def sweep_stale_dirs(self) -> None:
        """Remove what a crash can leave behind, never a saved document.

        That is: session-only temp dirs from older versions, and document
        folders whose upload never completed (no saved record).
        """
        for root in {self._dir_root, Path(tempfile.gettempdir())}:
            try:
                entries = list(root.glob(f"{JOB_DIR_PREFIX}*"))
            except OSError:
                continue
            for entry in entries:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
        for entry in storage.incomplete_documents(self._docs_root):
            shutil.rmtree(entry, ignore_errors=True)

    def load_saved(self) -> int:
        """Reopen every saved document. Returns how many were loaded.

        Work that was in flight when the app stopped cannot resume mid-way:
        an interrupted detection goes back to the zone step (its zones are
        kept, so continuing is one click), and an interrupted render goes back
        to review.
        """
        loaded = 0
        for directory, record, identities in storage.read_all(self._docs_root):
            try:
                job = Job.from_record(directory, record, identities)
            except (KeyError, TypeError, ValueError):
                continue  # unreadable record: leave the folder for the user
            if job.status in {"queued", "detecting"}:
                reset_to_zones(job)
            elif job.status == "rendering":
                job.status = "awaiting_review"
            with self._lock:
                self._jobs[job.id] = job
            migrated = migrate_rows(job)
            # Nothing is in flight after a restart: those attempts become
            # "interrupted", which the reviewer can retry.
            # Every record, not just the first: ``any`` over a generator
            # would stop at the first one that changed.
            interrupted = False
            for record in list(job.extractions.values()):
                interrupted = exlib.interrupt_in_flight(record) or interrupted
            for record in list(job.interpretations.values()):
                interrupted = ilib.interrupt_in_flight(record) or interrupted
            if migrated or interrupted:
                self.save(job)  # once: what is stored is the recovered state
            loaded += 1
        return loaded

    def save(self, job: Job) -> None:
        """Write the document's record and its private identity map."""
        with self._save_lock:
            if job.deleted:
                return
            storage.write_document(job.dir, job.to_record(), job.patient_ids)

    def shutdown(self) -> None:
        """Stop the workers. Saved documents stay on disk for the next start;
        extraction attempts cut short come back as interrupted."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._extract_executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            self._jobs.clear()

    def list(self) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    # -- registry ---------------------------------------------------------

    def create(self, filename: str, dpi: int) -> Job:
        """Register a new document. Nothing is saved until :meth:`save`, so
        an upload that fails part-way leaves no record behind."""
        job_id = uuid.uuid4().hex
        work_dir = self._docs_root / job_id
        work_dir.mkdir(parents=True)
        job = Job(
            id=job_id,
            filename=filename,
            dpi=dpi,
            dir=work_dir,
            input_path=work_dir / JOB_INPUT_NAME,
            output_path=work_dir / JOB_OUTPUT_NAME,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        with self._save_lock:
            job.deleted = True
        job.publish({"type": "deleted"})
        shutil.rmtree(job.dir, ignore_errors=True)
        return True

    # -- extraction -------------------------------------------------------

    def _get_extractor(self) -> Any:
        with self._lock:
            if self._extractor is None:
                self._extractor = self._extractor_factory()
            return self._extractor

    def extraction_configured(self) -> bool:
        return bool(self._get_extractor().configured())

    def start_extraction(
        self, job: Job, row_ids: Sequence[str], model: str, then_interpret: bool = True
    ) -> dict[str, Any]:
        """Queue one attempt per row. Returns ``{"queued": [...], "skipped": {row_id: reason}}``.

        A row already queued or running is skipped, so repeated clicks never
        start duplicate work. Each attempt records the basis it will run on;
        the result is attached only if the row still has that basis.
        """
        queued: list[tuple[str, str]] = []
        skipped: dict[str, str] = {}
        for row_id in row_ids:
            row = job.rows.get(row_id)
            if row is None:
                skipped[row_id] = "unknown row"
                continue
            basis = exlib.basis_of(row, job.effective_boxes(row["working_page"]))
            if basis is None:
                skipped[row_id] = "not ready for extraction"
                continue
            # The record itself says whether the row is in flight; checking
            # and queueing under one lock makes a double click harmless.
            with job.extract_lock:
                record = job.extractions.setdefault(row_id, exlib.empty_record())
                if record["attempts"] and record["attempts"][-1]["status"] in exlib.IN_FLIGHT:
                    skipped[row_id] = "already running"
                    continue
                attempt = exlib.new_attempt(basis, model)
                # Transcription flows straight into interpretation: no
                # acceptance step in between.
                attempt["then_interpret"] = then_interpret
                record["attempts"].append(attempt)
            queued.append((row_id, attempt["attempt_id"]))
        if queued:
            self.save(job)
        for row_id, attempt_id in queued:
            self._extract_executor.submit(self._run_extraction, job, row_id, attempt_id)
        return {"queued": [row_id for row_id, _ in queued], "skipped": skipped}

    def _attempt(self, job: Job, row_id: str, attempt_id: str) -> dict[str, Any] | None:
        record = job.extractions.get(row_id) or {}
        return next((a for a in record.get("attempts", []) if a["attempt_id"] == attempt_id), None)

    def _finish(self, job: Job, row_id: str, attempt_id: str, **changes: Any) -> None:
        with job.extract_lock:
            attempt = self._attempt(job, row_id, attempt_id)
            if attempt is None:
                return
            attempt.update(changes)
            attempt["finished_at"] = rowlib.now_iso()
        self.save(job)

    def _evidence_for_send(self, job: Job, row_id: str, basis: dict[str, Any]) -> tuple[bytes | None, str]:
        """The attempt's evidence, if it may still be sent -- else ``(None, reason)``.

        Run before the first provider call and before every retry, so a row
        edited while its request waited in the queue (or in retry back-off)
        never sends a crop that no longer matches what was reviewed. The lock
        is held only for the state check, never across the network call.
        """
        if job.deleted or self.get(job.id) is not job:
            return None, "the document was deleted"
        with job.extract_lock:
            row = job.rows.get(row_id)
            if row is None:
                return None, "the row no longer exists"
            current = exlib.basis_of(row, job.effective_boxes(row["working_page"]))
        if current is None:
            return None, "the row is no longer ready for extraction"
        if current != basis:
            return None, "the row's evidence changed"
        path = (job.dir / basis["evidence_path"]).resolve()
        try:
            if job.dir.resolve() not in path.parents:
                raise FileNotFoundError
            png = path.read_bytes()
        except OSError:
            return None, "the evidence file is missing"
        if hashlib.sha256(png).hexdigest() != basis["evidence_sha256"]:
            return None, "the evidence file was altered"
        return png, ""

    def _run_extraction(self, job: Job, row_id: str, attempt_id: str) -> None:
        # Every path out of here moves the attempt out of the in-flight states
        # (``_finish``), which is what frees the row for another attempt.
        tries = 0
        try:
            with job.extract_lock:
                attempt = self._attempt(job, row_id, attempt_id)
                if attempt is None:
                    return
                attempt["status"] = exlib.RUNNING
                attempt["started_at"] = rowlib.now_iso()
                basis = dict(attempt["basis"])
                model = attempt["requested_model"]
                then_interpret = bool(attempt.get("then_interpret"))
            self.save(job)

            while True:
                # The frozen evidence of the approval -- exactly what the
                # reviewer accepted -- and only while it still is.
                png, reason = self._evidence_for_send(job, row_id, basis)
                if png is None:
                    self._finish(job, row_id, attempt_id, status=exlib.FAILED, tries=tries, error={
                        "kind": "row_changed",
                        "message": f"Not sent: {reason} before the request went out. Extract again.",
                    })
                    return
                tries += 1
                try:
                    output = self._get_extractor().extract(png, model)
                    break
                except ex.ExtractionError as err:
                    # Invalid output gets one more sample; other transient
                    # failures up to the configured retries; permanent ones none.
                    limit = 2 if err.kind == "invalid_output" else 1 + len(self._retry_delays)
                    if err.transient and tries < limit:
                        delay = self._retry_delays[min(tries, len(self._retry_delays)) - 1]
                        if err.retry_after is not None:
                            delay = min(max(delay, err.retry_after), 60.0)
                        self._sleep(delay)
                        continue
                    self._finish(job, row_id, attempt_id, status=exlib.FAILED, tries=tries,
                                 error={"kind": err.kind, "message": str(err)})
                    return
                except Exception as exc:  # noqa: BLE001 - never let a worker die silently
                    self._finish(job, row_id, attempt_id, status=exlib.FAILED, tries=tries, error={
                        "kind": "internal", "message": f"Extraction failed ({type(exc).__name__}).",
                    })
                    return

            # Recheck before attaching: the row may have changed while the
            # request was in flight. Then the result is kept, but superseded.
            doctor, version = self.assigned_version(job) if then_interpret else (None, None)
            follow_up: list[tuple[str, str]] = []
            with job.extract_lock:
                row = job.rows.get(row_id)
                current = (
                    exlib.basis_of(row, job.effective_boxes(row["working_page"])) if row else None
                )
                status = exlib.SUCCEEDED if current == basis else exlib.SUPERSEDED
                attempt = self._attempt(job, row_id, attempt_id)
                if attempt is not None:
                    attempt.update(
                        status=status, tries=tries, result=output.result,
                        returned_model=output.returned_model, provider=output.provider,
                        generation_id=output.generation_id, usage=output.usage,
                        finished_at=rowlib.now_iso(),
                    )
                    # Straight on to interpretation, recorded under the same
                    # lock: the row is never seen as idle in between. A
                    # blocked or busy row is simply skipped.
                    if status == exlib.SUCCEEDED and then_interpret:
                        follow_up, _ = self._queue_interpretations(job, [row_id], model, doctor, version)
            self.save(job)
            self._submit_interpretations(job, follow_up)
        except BaseException:
            self._finish(job, row_id, attempt_id, status=exlib.FAILED, tries=tries, error={
                "kind": "internal", "message": "Extraction stopped unexpectedly.",
            })
            raise

    # -- interpretation ---------------------------------------------------

    def assigned_version(self, job: Job) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """``(doctor assignment, its exact profile version)`` -- the assigned
        version, never the profile's latest."""
        doctor = copy.deepcopy(job.doctor)
        if doctor is None:
            return None, None
        return doctor, self.doctors.version(doctor["profile_id"], doctor["version"])

    def interpretation_inputs(
        self, job: Job, row_id: str,
        doctor: dict[str, Any] | None = None, version: dict[str, Any] | None = None,
        resolved: bool = False,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """What interpreting this row now would be based on (see
        :func:`interpretation.inputs_of`)."""
        if not resolved:
            doctor, version = self.assigned_version(job)
        with job.extract_lock:
            row = job.rows.get(row_id)
            if row is None:
                return None, None
            return ilib.inputs_of(
                row, job.extractions.get(row_id), job.effective_boxes(row["working_page"]), doctor, version
            )

    def interpretation_view(self, job: Job, row_id: str, doctor=None, version=None) -> dict[str, Any]:
        inputs, blocked = self.interpretation_inputs(job, row_id, doctor, version, resolved=True)
        with job.extract_lock:
            return ilib.view(job.interpretations.get(row_id), inputs, blocked)

    def start_interpretation(self, job: Job, row_ids: Sequence[str], model: str) -> dict[str, Any]:
        """Queue one interpretation per row. Rows without a current
        transcription, blocked rows, and rows with either stage in flight are
        skipped with the reason."""
        doctor, version = self.assigned_version(job)
        queued, skipped = self._queue_interpretations(job, row_ids, model, doctor, version)
        if queued:
            self.save(job)
        self._submit_interpretations(job, queued)
        return {"queued": [row_id for row_id, _ in queued], "skipped": skipped}

    def _submit_interpretations(self, job: Job, queued: Sequence[tuple[str, str]]) -> None:
        for row_id, attempt_id in queued:
            self._extract_executor.submit(self._run_interpretation, job, row_id, attempt_id)

    def _queue_interpretations(
        self, job: Job, row_ids: Sequence[str], model: str,
        doctor: dict[str, Any] | None, version: dict[str, Any] | None,
    ) -> tuple[list[tuple[str, str]], dict[str, str]]:
        """Record queued attempts (no save, no submit). Safe to call while
        holding ``job.extract_lock`` -- it is re-entrant."""
        queued: list[tuple[str, str]] = []
        skipped: dict[str, str] = {}
        for row_id in row_ids:
            inputs, blocked = self.interpretation_inputs(job, row_id, doctor, version, resolved=True)
            if blocked:
                skipped[row_id] = ilib.BLOCKED_MESSAGES[blocked]
                continue
            if inputs is None:
                skipped[row_id] = "no current transcription"
                continue
            with job.extract_lock:
                transcription = job.extractions.get(row_id) or {}
                if transcription.get("attempts") and transcription["attempts"][-1]["status"] in exlib.IN_FLIGHT:
                    skipped[row_id] = "transcribing"
                    continue
                record = job.interpretations.setdefault(row_id, ilib.empty_record())
                if record["attempts"] and record["attempts"][-1]["status"] in ilib.IN_FLIGHT:
                    skipped[row_id] = "already interpreting"
                    continue
                attempt = ilib.new_attempt(inputs, model)
                record["attempts"].append(attempt)
            queued.append((row_id, attempt["attempt_id"]))
        return queued, skipped

    def _interp_attempt(self, job: Job, row_id: str, attempt_id: str) -> dict[str, Any] | None:
        record = job.interpretations.get(row_id) or {}
        return next((a for a in record.get("attempts", []) if a["attempt_id"] == attempt_id), None)

    def _finish_interp(self, job: Job, row_id: str, attempt_id: str, **changes: Any) -> None:
        with job.extract_lock:
            attempt = self._interp_attempt(job, row_id, attempt_id)
            if attempt is None:
                return
            attempt.update(changes)
            attempt["finished_at"] = rowlib.now_iso()
        self.save(job)

    def _interpretation_for_send(
        self, job: Job, row_id: str, inputs: dict[str, Any]
    ) -> tuple[tuple[bytes, ex.InterpretationRequest] | None, str]:
        """The image and request to send, if the inputs still hold -- checked
        before the first call and every retry, like transcription."""
        if job.deleted or self.get(job.id) is not job:
            return None, "the document was deleted"
        doctor, version = self.assigned_version(job)
        current, blocked = self.interpretation_inputs(job, row_id, doctor, version, resolved=True)
        if blocked:
            return None, ilib.BLOCKED_MESSAGES[blocked].rstrip(".").lower()
        if current != inputs:
            return None, "the row, its transcription or its doctor changed"
        evidence = inputs["evidence"]
        path = (job.dir / evidence["evidence_path"]).resolve()
        try:
            if job.dir.resolve() not in path.parents:
                raise FileNotFoundError
            png = path.read_bytes()
        except OSError:
            return None, "the evidence file is missing"
        if hashlib.sha256(png).hexdigest() != evidence["evidence_sha256"]:
            return None, "the evidence file was altered"
        return (png, ilib.request_for(inputs, version)), ""  # type: ignore[arg-type]

    def _run_interpretation(self, job: Job, row_id: str, attempt_id: str) -> None:
        tries = 0
        try:
            with job.extract_lock:
                attempt = self._interp_attempt(job, row_id, attempt_id)
                if attempt is None:
                    return
                attempt["status"] = ilib.RUNNING
                attempt["started_at"] = rowlib.now_iso()
                inputs = copy.deepcopy(attempt["inputs"])
                model = attempt["requested_model"]
            self.save(job)

            while True:
                ready, reason = self._interpretation_for_send(job, row_id, inputs)
                if ready is None:
                    self._finish_interp(job, row_id, attempt_id, status=ilib.FAILED, tries=tries, error={
                        "kind": "inputs_changed",
                        "message": f"Not sent: {reason} before the request went out. Interpret again.",
                    })
                    return
                png, request = ready
                tries += 1
                try:
                    output = self._get_extractor().interpret(png, request, model)
                    break
                except ex.ExtractionError as err:
                    limit = 2 if err.kind == "invalid_output" else 1 + len(self._retry_delays)
                    if err.transient and tries < limit:
                        delay = self._retry_delays[min(tries, len(self._retry_delays)) - 1]
                        if err.retry_after is not None:
                            delay = min(max(delay, err.retry_after), 60.0)
                        self._sleep(delay)
                        continue
                    self._finish_interp(job, row_id, attempt_id, status=ilib.FAILED, tries=tries,
                                        error={"kind": err.kind, "message": str(err)})
                    return
                except Exception as exc:  # noqa: BLE001
                    self._finish_interp(job, row_id, attempt_id, status=ilib.FAILED, tries=tries, error={
                        "kind": "internal", "message": f"Interpretation failed ({type(exc).__name__}).",
                    })
                    return

            # Inputs may have changed while the model worked: then the result
            # is kept, but superseded -- it never becomes the current proposal.
            current, _ = self.interpretation_inputs(job, row_id)
            status = ilib.SUCCEEDED if current == inputs else ilib.SUPERSEDED
            self._finish_interp(
                job, row_id, attempt_id, status=status, tries=tries,
                result=output.result, returned_model=output.returned_model,
                provider=output.provider, generation_id=output.generation_id,
                usage=output.usage,
            )
        except BaseException:
            self._finish_interp(job, row_id, attempt_id, status=ilib.FAILED, tries=tries, error={
                "kind": "internal", "message": "Interpretation stopped unexpectedly.",
            })
            raise

    # -- phases -----------------------------------------------------------

    def start_detect(
        self,
        job: Job,
        loop: asyncio.AbstractEventLoop,
        zones: dict[int, tuple[dict[str, Any], ...]] | None = None,
        source_dates: dict[int, str] | None = None,
    ) -> None:
        """Queue the slow phase: crop to zones, then rasterize + OCR + locate.

        ``zones`` maps page numbers to normalized boxes. ``None`` or empty
        means the whole document, page for page. ``source_dates`` maps
        source pages to dates, which pass on to the pages made from them.
        """
        job.loop = loop
        reset_to_zones(job)
        job.status = "queued"
        job.error = None
        job.page_dates = dict(source_dates or {})  # whole document: 1:1
        job.zone_request = {
            "zones": {str(k): list(v) for k, v in (zones or {}).items()},
            "dates": {str(k): v for k, v in (source_dates or {}).items()},
        }
        self.save(job)
        job.publish({"type": "queued"})
        self._executor.submit(self._run_detect, job, zones or None)

    def _run_detect(
        self, job: Job, zones: dict[int, tuple[dict[str, Any], ...]] | None = None
    ) -> None:
        job.status = "detecting"
        original = job.dir / JOB_INPUT_NAME
        if not zones:
            job.provenance = whole_page_provenance(original, dpi=job.dpi)
        if zones:
            # On the worker, not the request: cropping rasterizes every page
            # that has a zone, which is too slow to hold a request open for.
            zoned = job.dir / JOB_ZONES_NAME
            try:
                sources = crop_to_zones(original, zoned, zones, dpi=job.dpi)
            except Exception as exc:  # noqa: BLE001 - reported, never raised onward
                self._fail(job, exc)
                return
            job.provenance = sources
            job.total_pages = len(sources)
            source_dates = job.page_dates
            job.page_dates = {
                prov.working_page: source_dates[prov.source_page]
                for prov in sources
                if prov.source_page in source_dates
            }
            job.input_path = zoned

        def on_progress(progress: PageProgress) -> None:
            if progress.page_num == 0:
                job.total_pages = progress.total_pages
                job.publish({
                    "type": "start",
                    "total_pages": progress.total_pages,
                    "dpi": job.dpi,
                })
                return
            job.publish({
                "type": "page",
                "page_num": progress.page_num,
                "total_pages": progress.total_pages,
                "nams_on_page": progress.nams_on_page,
                "total_nams": progress.total_nams,
            })

        try:
            pages, total_pages = detect_nam_pages(
                job.input_path,
                dpi=job.dpi,
                engine=self._get_engine(),
                progress_callback=on_progress,
                # Populated in place, so a NAM typed later continues the same
                # numbering instead of starting a second series.
                patient_ids=job.patient_ids,
                name_model=_LazyNameModel(self),
            )
        except Exception as exc:  # noqa: BLE001 - reported, never raised onward
            self._fail(job, exc)
            return

        job.pages = pages
        job.total_pages = total_pages
        job.per_page_counts = [len(page.boxes) for page in pages]
        job.total_nams, job.total_names = count_kinds(pages)
        job.status = "awaiting_review"
        ensure_section_rows(job)
        self.save(job)
        job.publish({
            "type": "awaiting_review",
            "total_pages": total_pages,
            "total_nams": job.total_nams,
            "total_names": job.total_names,
            "per_page_counts": job.per_page_counts,
        })

    def _fail(self, job: Job, exc: BaseException) -> None:
        job.status = "error"
        job.error = _scrub(exc)
        self.save(job)
        job.publish({"type": "error", "message": job.error})

    def start_finalize(self, job: Job, loop: asyncio.AbstractEventLoop) -> None:
        """Queue the fast phase: render the *saved* reviewed redactions.

        The output is made from exactly what :func:`save_review` stored, so
        the PDF and the evidence crops can never disagree.
        """
        job.loop = loop
        job.status = "rendering"
        self._executor.submit(self._run_finalize, job)

    def _run_finalize(self, job: Job) -> None:
        pages = job.effective_pages()
        job.publish({"type": "rendering"})
        try:
            _total, per_page, _ = render_redacted_pdf(
                job.input_path,
                job.output_path,
                pages,
                dpi=job.dpi,
            )
        except Exception as exc:  # noqa: BLE001
            self._fail(job, exc)
            return

        job.total_nams, job.total_names = count_kinds(pages)
        job.per_page_counts = per_page
        job.status = "done"
        self.save(job)
        job.publish({
            "type": "done",
            "total_nams": job.total_nams,
            "total_names": job.total_names,
            "per_page_counts": per_page,
            "result_url": f"/api/jobs/{job.id}/result",
        })


class _LazyNameModel:
    """Defers loading the NER model until names are actually searched for.

    Loading spaCy takes seconds; doing it before detection would stall the
    progress stream before the first page, where the user expects OCR to be
    what they are waiting on.
    """

    def __init__(self, store: JobStore) -> None:
        self._store = store

    def find_names(self, lines: Sequence[str]) -> list[list[tuple[int, int]]]:
        model = self._store._get_name_model()
        if model is None:
            return [[] for _ in lines]
        return model.find_names(lines)


def reset_to_zones(job: Job) -> None:
    """Back to the zone step with the upload as it was received.

    Detections, review and rows all refer to working pages, which the zones
    are about to redefine, so none of them can survive a restart of detection.
    """
    job.status = "awaiting_zones"
    job.input_path = job.dir / JOB_INPUT_NAME
    job.total_pages = job.source_page_count
    job.provenance = []
    job.pages = []
    job.reviewed = None
    job.review_revision = 0
    job.reviewed_at = None
    job.rows = {}
    with job.extract_lock:
        job.extractions = {}
        job.interpretations = {}
        job.billing = {}
    job.per_page_counts = []
    job.total_nams = job.total_names = 0
    with job.lock:
        job.events.clear()
    for leftover in (job.dir / JOB_ZONES_NAME, job.output_path):
        leftover.unlink(missing_ok=True)
    shutil.rmtree(job.dir / storage.EVIDENCE_DIR, ignore_errors=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_review(
    job: Job,
    entries: Sequence[tuple[int, Any, str | None]],
) -> tuple[bool, list[str]]:
    """Store the reviewer's redactions (and page dates) as the new revision.

    ``entries`` are ``(page_num, raw_boxes, date)`` from the client. A page
    left out keeps what it has -- omitting a page must never *remove* its
    redactions. Returns ``(changed, invalidated_row_ids)``; the caller saves.

    Raises ``ValueError`` for a bad date or a page missing one, before
    anything is changed.
    """
    working = {page.page_num for page in job.pages}
    reviewed = {page_num: job.effective_boxes(page_num) for page_num in working}
    dates = dict(job.page_dates)
    seen: set[int] = set()
    for page_num, raw, date in entries:
        if page_num in seen or page_num not in working:
            continue
        seen.add(page_num)
        if raw is not None:
            reviewed[page_num] = resolve_box_ids(job, page_num, raw)
        if date is not None:
            parsed = parse_page_date(date)
            if parsed is None:
                raise ValueError(f"Page {page_num} has an invalid date.")
            dates[page_num] = parsed
    missing = sorted(working - set(dates))
    if missing:
        raise ValueError(f"Every page needs a date (missing on page {missing[0]}).")

    changed = job.reviewed is None or reviewed != job.reviewed
    job.page_dates = dates
    for row_id, row in list(job.rows.items()):
        moved = rowlib.follow_page_date(row, dates.get(row["working_page"]))
        if moved is not None:
            job.rows[row_id] = moved
    invalidated: list[str] = []
    if changed:
        job.reviewed = reviewed
        job.review_revision += 1
        job.reviewed_at = rowlib.now_iso()
        for row_id, row in list(job.rows.items()):
            updated, stale = rowlib.revalidate(row, job.effective_boxes(row["working_page"]))
            if stale:
                job.rows[row_id] = updated
                invalidated.append(row_id)
        if job.status == "done":
            # The redacted PDF no longer matches what was reviewed.
            job.output_path.unlink(missing_ok=True)
            job.status = "awaiting_review"
    return changed, invalidated


def ensure_section_rows(job: Job) -> bool:
    """Give every section its appointment row. Returns whether any was added.

    Each section is one appointment, so its row covers the whole section.
    The date is suggested from the section's date and the patient from the
    NAM detected in it; both still need the reviewer's confirmation.
    Idempotent: a section that already has its row is left alone.
    """
    covered = {
        row["working_page"] for row in job.rows.values()
        if row.get("origin") == rowlib.ORIGIN_SECTION
    }
    added = False
    for prov in sorted(job.provenance, key=lambda p: p.working_page):
        page = prov.working_page
        if page in covered or not any(p.page_num == page for p in job.pages):
            continue
        row = rowlib.new_row(
            job.id, page, prov.source_page, rowlib.FULL_PAGE,
            date_suggestion=job.page_dates.get(page),
            patient_suggestion=rowlib.suggest_patient(rowlib.FULL_PAGE, job.effective_boxes(page)),
            origin=rowlib.ORIGIN_SECTION,
        )
        job.rows[row["row_id"]] = row
        added = True
    return added


def migrate_rows(job: Job) -> bool:
    """Bring every saved row to the current row schema (see
    :func:`rows.migrate_row`), and confirm app-derived values that older
    versions left as suggestions (:func:`rows.trust_app_values`). Returns
    whether anything changed."""
    changed = False
    for row_id, row in list(job.rows.items()):
        migrated, did = rowlib.migrate_row(row)
        trusted, confirmed = rowlib.trust_app_values(migrated)
        if did or confirmed:
            job.rows[row_id] = trusted
            changed = True
    return changed


def assign_doctor(job: Job, version: dict[str, Any]) -> dict[str, Any]:
    """Assign a doctor profile version to the whole document.

    Records the profile, the exact version, and a snapshot of the name and
    identifiers as they were in that version -- a later edit of the profile
    changes nothing here until someone assigns again. Each assignment gets a
    fresh ``assignment_id``, so a later step that recorded the assignment it
    used can tell whether it has changed since.

    Only the assignment and the practice identifiers change: redactions,
    rows, extraction results and reviews are untouched (extraction is a
    literal transcription and does not depend on the doctor).
    """
    snapshot = {
        "display_name": version["display_name"],
        **{name: version[name] for name in PRACTICE_FIELDS},
    }
    assignment = {
        "assignment_id": f"asg_{uuid.uuid4().hex[:12]}",
        "profile_id": version["profile_id"],
        "version": version["version"],
        "snapshot": snapshot,
        "assigned_at": rowlib.now_iso(),
    }
    if job.doctor is not None:
        job.doctor_history.append(job.doctor)
    elif job.practice and any(job.practice.values()):
        # A document from before profiles: keep what was typed at upload.
        job.doctor_history.append({
            "assignment_id": None, "profile_id": None, "version": None,
            "snapshot": {"display_name": None, **{k: job.practice.get(k, "") for k in PRACTICE_FIELDS}},
            "assigned_at": None, "legacy_practice": True,
        })
    job.doctor = assignment
    job.practice = {name: snapshot[name] for name in PRACTICE_FIELDS}
    return assignment
