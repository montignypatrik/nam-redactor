"""Doctor profiles: who a document belongs to, and how to read their notes.

Every uploaded PDF belongs to exactly one doctor. A *profile* records that
doctor's identifiers (licence, group, sector) and, for a later billing
interpretation step, plain-text instructions and worked examples.

Profiles are **versioned**. Saving an edit never changes a stored version: it
adds a new one. A document records the exact version it was assigned (see
:mod:`nam_redactor.web.jobs`), so a later edit can never silently change
what an existing document says -- the reviewer adopts a newer version
explicitly. Archiving hides a profile from new assignments; its versions stay
readable forever, because documents may point at them.

Storage is a small SQLite database in the app's data directory, apart from
the per-document folders. Version rows are protected by triggers: SQLite
itself refuses to update or delete them.
"""
from __future__ import annotations

import datetime
import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DB_NAME = "doctors.sqlite3"
SCHEMA_VERSION = 1

ACTIVE = "active"
ARCHIVED = "archived"

MAX_NAME = 120
MAX_INSTRUCTIONS = 50_000
MAX_EXAMPLES = 200
MAX_EXAMPLE_FIELD = 2_000

IDENTIFIER_FIELDS = ("licence_number", "group_number", "sector_number")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS profiles (
    profile_id      TEXT PRIMARY KEY,
    display_name    TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('active', 'archived')),
    current_version INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS profile_versions (
    profile_id     TEXT NOT NULL REFERENCES profiles(profile_id),
    version        INTEGER NOT NULL CHECK (version >= 1),
    display_name   TEXT NOT NULL,
    licence_number TEXT NOT NULL,
    group_number   TEXT NOT NULL,
    sector_number  TEXT NOT NULL,
    instructions   TEXT NOT NULL,
    examples       TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (profile_id, version)
);
CREATE TRIGGER IF NOT EXISTS profile_versions_immutable
BEFORE UPDATE ON profile_versions
BEGIN SELECT RAISE(ABORT, 'profile versions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS profile_versions_permanent
BEFORE DELETE ON profile_versions
BEGIN SELECT RAISE(ABORT, 'profile versions cannot be deleted'); END;
"""


class ProfileError(ValueError):
    """A request about profiles that cannot be honoured; safe to show."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _new_profile_id() -> str:
    return f"doc_{uuid.uuid4().hex[:12]}"


# -- validation -------------------------------------------------------------


def validate_content(body: Mapping[str, Any], parse_identifier) -> dict[str, Any]:
    """The content of one version, validated. Raises :class:`ProfileError`.

    ``parse_identifier`` is the app's existing licence/group/sector rule
    (optional; letters, digits, spaces, dashes). Identifiers stay strings, so
    leading zeros and formatting survive.
    """
    name = body.get("display_name")
    if not isinstance(name, str) or not name.strip():
        raise ProfileError("A doctor profile needs a name.")
    name = " ".join(name.split())
    if len(name) > MAX_NAME:
        raise ProfileError(f"The name is limited to {MAX_NAME} characters.")

    identifiers: dict[str, str] = {}
    labels = {"licence_number": "licence", "group_number": "group", "sector_number": "sector"}
    for field in IDENTIFIER_FIELDS:
        value = parse_identifier(body.get(field))
        if value is None:
            raise ProfileError(
                f"The {labels[field]} number may only contain letters, digits, spaces and dashes."
            )
        identifiers[field] = value

    instructions = body.get("instructions") or ""
    if not isinstance(instructions, str):
        raise ProfileError("Instructions must be text.")
    instructions = instructions.replace("\r\n", "\n").strip("\n")
    if len(instructions) > MAX_INSTRUCTIONS:
        raise ProfileError(f"Instructions are limited to {MAX_INSTRUCTIONS} characters.")

    raw_examples = body.get("examples") or []
    if not isinstance(raw_examples, list):
        raise ProfileError("Examples must be a list.")
    examples: list[dict[str, str]] = []
    for index, item in enumerate(raw_examples, start=1):
        if not isinstance(item, Mapping):
            raise ProfileError(f"Example {index} is not valid.")
        fields = {}
        for key in ("annotation", "interpretation", "note"):
            value = item.get(key) or ""
            if not isinstance(value, str):
                raise ProfileError(f"Example {index}: {key} must be text.")
            value = value.replace("\r\n", "\n").strip()
            if len(value) > MAX_EXAMPLE_FIELD:
                raise ProfileError(f"Example {index}: {key} is too long.")
            fields[key] = value
        if not any(fields.values()):
            continue  # an empty row the editor left behind
        if not fields["annotation"] or not fields["interpretation"]:
            raise ProfileError(f"Example {index} needs both the annotation and its interpretation.")
        examples.append(fields)
    if len(examples) > MAX_EXAMPLES:
        raise ProfileError(f"A profile is limited to {MAX_EXAMPLES} examples.")

    return {"display_name": name, **identifiers, "instructions": instructions, "examples": examples}


# -- store --------------------------------------------------------------------


class DoctorStore:
    """Profiles and their versions, in ``<data dir>/doctors.sqlite3``."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / DB_NAME
        self._lock = threading.Lock()
        self._ready = False

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """One short-lived connection per operation, serialized by a lock:
        simple, and safe from request handlers and worker threads alike."""
        with self._lock:
            if not self._ready:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                if not self._ready:
                    conn.executescript(_SCHEMA)
                    conn.execute(
                        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                        (str(SCHEMA_VERSION),),
                    )
                    self._ready = True
                with conn:  # one transaction per operation
                    yield conn
            finally:
                conn.close()

    # -- reading --------------------------------------------------------

    @staticmethod
    def _version(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "profile_id": row["profile_id"],
            "version": row["version"],
            "display_name": row["display_name"],
            "licence_number": row["licence_number"],
            "group_number": row["group_number"],
            "sector_number": row["sector_number"],
            "instructions": row["instructions"],
            "examples": json.loads(row["examples"]),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _profile(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "profile_id": row["profile_id"],
            "display_name": row["display_name"],
            "status": row["status"],
            "current_version": row["current_version"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list(self, include_archived: bool = False) -> list[dict[str, Any]]:
        """Profiles with a summary of their current version, by name."""
        query = """
            SELECT p.*, v.licence_number, v.group_number, v.sector_number
            FROM profiles p JOIN profile_versions v
              ON v.profile_id = p.profile_id AND v.version = p.current_version
        """
        if not include_archived:
            query += " WHERE p.status = 'active'"
        query += " ORDER BY p.display_name COLLATE NOCASE, p.created_at"
        with self._db() as conn:
            rows = conn.execute(query).fetchall()
        return [
            {**self._profile(r), **{f: r[f] for f in IDENTIFIER_FIELDS}}
            for r in rows
        ]

    def get(self, profile_id: str) -> dict[str, Any] | None:
        """A profile, its current version in full, and every version's summary."""
        with self._db() as conn:
            row = conn.execute("SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)).fetchone()
            if row is None:
                return None
            versions = conn.execute(
                "SELECT * FROM profile_versions WHERE profile_id = ? ORDER BY version DESC", (profile_id,)
            ).fetchall()
        full = [self._version(v) for v in versions]
        return {
            **self._profile(row),
            "current": next(v for v in full if v["version"] == row["current_version"]),
            "versions": [
                {k: v[k] for k in ("version", "display_name", *IDENTIFIER_FIELDS, "created_at")}
                | {"example_count": len(v["examples"]), "has_instructions": bool(v["instructions"])}
                for v in full
            ],
        }

    def version(self, profile_id: str, version: int) -> dict[str, Any] | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM profile_versions WHERE profile_id = ? AND version = ?",
                (profile_id, version),
            ).fetchone()
        return self._version(row) if row else None

    # -- writing --------------------------------------------------------

    def _insert_version(self, conn: sqlite3.Connection, profile_id: str, version: int,
                        content: Mapping[str, Any], stamp: str) -> None:
        conn.execute(
            """INSERT INTO profile_versions (profile_id, version, display_name, licence_number,
               group_number, sector_number, instructions, examples, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (profile_id, version, content["display_name"], content["licence_number"],
             content["group_number"], content["sector_number"], content["instructions"],
             json.dumps(content["examples"], ensure_ascii=False), stamp),
        )

    def create(self, content: Mapping[str, Any]) -> dict[str, Any]:
        profile_id = _new_profile_id()
        stamp = now_iso()
        with self._db() as conn:
            conn.execute(
                "INSERT INTO profiles VALUES (?, ?, 'active', 1, ?, ?)",
                (profile_id, content["display_name"], stamp, stamp),
            )
            self._insert_version(conn, profile_id, 1, content, stamp)
        return self.get(profile_id)  # type: ignore[return-value]

    def save_version(self, profile_id: str, content: Mapping[str, Any], base_version: int | None) -> dict[str, Any]:
        """Store an edit as a new version. ``base_version`` guards against
        overwriting an edit made elsewhere since the editor was opened."""
        stamp = now_iso()
        with self._db() as conn:
            row = conn.execute("SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)).fetchone()
            if row is None:
                raise ProfileError("Unknown doctor profile.", 404)
            if row["status"] != ACTIVE:
                raise ProfileError("Archived profiles cannot be edited; restore it first.", 409)
            current = row["current_version"]
            if base_version is not None and base_version != current:
                raise ProfileError(
                    f"This profile was changed elsewhere (now version {current}). Reload before saving.", 409
                )
            latest = self._version(conn.execute(
                "SELECT * FROM profile_versions WHERE profile_id = ? AND version = ?", (profile_id, current)
            ).fetchone())
            # Saving without a change adds no version: versions mean edits.
            unchanged = all(latest[k] == content[k] for k in content)
            if not unchanged:
                self._insert_version(conn, profile_id, current + 1, content, stamp)
                conn.execute(
                    "UPDATE profiles SET current_version = ?, display_name = ?, updated_at = ? WHERE profile_id = ?",
                    (current + 1, content["display_name"], stamp, profile_id),
                )
        return {**self.get(profile_id), "unchanged": unchanged}  # type: ignore[dict-item]

    def duplicate(self, profile_id: str, display_name: str | None = None) -> dict[str, Any]:
        """A new, independent profile starting from this one's current version."""
        source = self.get(profile_id)
        if source is None:
            raise ProfileError("Unknown doctor profile.", 404)
        content = {k: source["current"][k] for k in ("display_name", *IDENTIFIER_FIELDS, "instructions", "examples")}
        content["display_name"] = (display_name or f"{content['display_name']} (copy)")[:MAX_NAME]
        return self.create(content)

    def set_status(self, profile_id: str, status: str) -> dict[str, Any]:
        with self._db() as conn:
            changed = conn.execute(
                "UPDATE profiles SET status = ?, updated_at = ? WHERE profile_id = ?",
                (status, now_iso(), profile_id),
            ).rowcount
        if not changed:
            raise ProfileError("Unknown doctor profile.", 404)
        return self.get(profile_id)  # type: ignore[return-value]

    # -- assignment -------------------------------------------------------

    def resolve_for_assignment(self, profile_id: Any, version: Any = None) -> dict[str, Any]:
        """The version a document may be assigned, or :class:`ProfileError`.

        Only active profiles can be newly assigned, and only at their current
        version: a stale page offering an older one is refused rather than
        quietly assigning outdated identifiers.
        """
        if not isinstance(profile_id, str) or not profile_id:
            raise ProfileError("Choose a doctor profile.")
        profile = self.get(profile_id)
        if profile is None:
            raise ProfileError("Unknown doctor profile.", 404)
        if profile["status"] != ACTIVE:
            raise ProfileError("That doctor profile is archived and cannot be assigned.", 409)
        if version not in (None, ""):
            try:
                version = int(version)
            except (TypeError, ValueError):
                raise ProfileError("Invalid profile version.") from None
            if version != profile["current_version"]:
                raise ProfileError(
                    f"That profile has a newer version ({profile['current_version']}). Reload and choose again.",
                    409,
                )
        return profile["current"]


def snapshot_of(version: Mapping[str, Any]) -> dict[str, str]:
    """What a document keeps of the version it was assigned."""
    return {"display_name": version["display_name"], **{f: version[f] for f in IDENTIFIER_FIELDS}}
