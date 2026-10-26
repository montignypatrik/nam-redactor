"""Local, file-based persistence for documents.

Each document is a folder under ``<data dir>/documents/<id>/``::

    original.pdf              the upload, exactly as received
    zones.pdf                 working pages rebuilt from zones (if any)
    redacted.pdf              the last rendered output (if any)
    document.json             record: provenance, detections, reviewed
                              redactions, page dates, appointment rows
    private/identities.json   canonical NAM -> pseudonym map
    evidence/<row_id>/...     evidence crops frozen at approval

``document.json`` holds no NAM, no name and no OCR text: boxes, pseudonyms,
dates and geometry only. The one file that links pseudonyms back to real
health numbers is ``private/identities.json``, kept apart so it is never
swept into an export or a record by accident. Everything stays on this
machine; deleting a document removes its whole folder.

Writes go to a temporary file first and are then renamed into place, so a
crash mid-write leaves the previous version intact rather than half a file.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

#: Bumped when the record layout changes incompatibly.
RECORD_SCHEMA_VERSION = 1

DOCUMENTS_DIR = "documents"
RECORD_NAME = "document.json"
PRIVATE_DIR = "private"
IDENTITIES_NAME = "identities.json"
#: Under the data directory's own ``private`` folder: an OpenRouter key saved in the app.
API_KEY_NAME = "openrouter_api_key.txt"
EVIDENCE_DIR = "evidence"

#: Overrides the data directory, e.g. for a portable install.
DATA_DIR_ENV = "NAM_REDACTOR_DATA"


def default_data_dir() -> Path:
    """Per-user application data, never the working directory."""
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "nam-redactor"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "nam-redactor"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "nam-redactor"


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=1, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_document(directory: Path, record: dict[str, Any], identities: dict[str, str]) -> None:
    """Save a document's record and, separately, its identity map."""
    _write_json_atomic(
        directory / PRIVATE_DIR / IDENTITIES_NAME,
        {
            "note": "Local only. Maps health numbers to this document's pseudonyms; never exported.",
            "identities": identities,
        },
    )
    _write_json_atomic(
        directory / RECORD_NAME,
        {"schema_version": RECORD_SCHEMA_VERSION, **record},
    )


def read_document(directory: Path) -> tuple[dict[str, Any], dict[str, str]] | None:
    """``(record, identities)``, or ``None`` if the folder holds no usable record."""
    try:
        record = json.loads((directory / RECORD_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("schema_version") != RECORD_SCHEMA_VERSION:
        return None
    try:
        private = json.loads(
            (directory / PRIVATE_DIR / IDENTITIES_NAME).read_text(encoding="utf-8")
        )
        identities = private.get("identities", {}) if isinstance(private, dict) else {}
    except (OSError, ValueError):
        identities = {}
    return record, {str(k): str(v) for k, v in identities.items()}


def read_all(docs_root: Path) -> Iterator[tuple[Path, dict[str, Any], dict[str, str]]]:
    if not docs_root.is_dir():
        return
    for directory in sorted(docs_root.iterdir()):
        if not directory.is_dir():
            continue
        loaded = read_document(directory)
        if loaded is not None:
            yield directory, *loaded


def incomplete_documents(docs_root: Path) -> list[Path]:
    """Folders with no record at all: an upload that never finished.

    A folder whose record exists but cannot be read is *not* listed -- that
    may be saved work from a newer version, and is not ours to delete.
    """
    if not docs_root.is_dir():
        return []
    return [
        directory for directory in docs_root.iterdir()
        if directory.is_dir() and not (directory / RECORD_NAME).exists()
    ]
