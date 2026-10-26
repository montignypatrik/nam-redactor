# nam-redactor

[![Tests](https://github.com/montignypatrik/nam-redactor/actions/workflows/test.yml/badge.svg)](https://github.com/montignypatrik/nam-redactor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Standalone, local web app that detects and black-box redacts Quebec health insurance numbers (RAMQ / NAM) and patient names from PDF documents. Upload a PDF, cut dense pages into small zones, watch detection run page by page, review and adjust the blackout boxes. Appointment rows can then be marked, confirmed and approved, and exported as a JSON manifest that traces every row back to its original page. Work is saved locally and survives restarts.

Redaction runs **100% locally on CPU** — no GPU, no Redis, no database, and no LLMs (name detection uses a small local spaCy model). The server binds to `127.0.0.1` only. The one optional cloud step is **appointment-row extraction** (see below): when you choose to run it, the already-redacted image of each row is sent to a vision model through OpenRouter, over zero-data-retention routes only. Nothing else about your document is ever sent.

> **One network call, once:** the first detection downloads the PaddleOCR models (about 20 MB) to `~/.paddlex`. That fetches *models*, never your document — after the first run it is fully offline.

---

## Key Features

- **True Raster Redaction**: Pages are rendered at target DPI, redacted, and saved as a fresh image-only PDF. The original vector text layer and NAM pixels are completely eliminated (not just masked with overlay rectangles).
- **Finds the NAM Inside Its Line**: OCR returns a label and the number that follows it as one string, so a line reading `NAM HOGP 3361 0615` is a single 15-character run. Each run is searched for a NAM-length span rather than validated as a whole, so the label is left intact and the number is still caught. A label fused to the name letters (`NAMHOGP 3361 0615`) is recovered too.
- **Sub-Line Box Tightness**: Refines PaddleOCR line-level boxes to the matched characters via character offset interpolation, so only the number is blacked out.
- **Split Token Merging**: Automatically joins NAM numbers split across adjacent OCR bounding blocks on the same row.
- **Patient Pseudonyms**: Every unique NAM becomes a stable `Patient_xxxx`, printed in white inside its blackout, so the same patient stays recognizable across a document without revealing who they are. Matching is OCR-tolerant, so `82O4` and `8204` resolve to one patient, and the optional 13th check digit is ignored — otherwise a number read with the check digit in one place and without it in another would split one record into two identities. Numbering follows reading order, and IDs are sequential rather than a hash: a NAM is low-entropy (4 letters plus a birth date), so a hash would be brute-forceable. A number the detector missed can be added by hand, and typing it joins the same numbering.
- **Patient Names**: Names are blacked out with no label. A NAM encodes its holder's name — letters 1-3 are the start of the last name, letter 4 the first initial — so `HOGP 3361 0615` confirms `HOGUE, Pierre` or `Pierre Hogue` when both words appear together. Values after labels such as `Nom :`, `Prénom :` or `Patient :` are taken too. Every name learned this way is then redacted wherever it appears in the document, on any page. On top of that, a local French NER model (spaCy `fr_core_news_md`) flags other people's names, such as doctors. Its hits are redacted only where they are found, and a single-word hit needs a title in front of it (`Dr Gagnon`), because on its own the model tends to tag place names. Name boxes have an amber outline in review so you can tell them apart; in the output they are plain black. The model loads once on first use, which takes a few seconds and about 200 MB. If spaCy is missing, the NAM-anchored and label layers still run.
- **Deterministic Validation**: Tolerant of OCR misrecognitions (O/0, I/1, B/8, S/5, Z/2, etc.) and accounts for RAMQ female month offsets (+50). Birth dates are validated strictly — a day that does not exist in its month (February 30, April 31) is rejected outright. Common clinical and administrative stopwords are rejected too.
- **CPU or GPU**: Uses the mobile detection and recognition models (`PP-OCRv5_mobile_det`, `PP-OCRv5_mobile_rec`) with `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`. The inference device is auto-detected: it uses the GPU when the installed Paddle build can reach one and falls back to CPU otherwise. Measured resident memory after the models load is about 620 MB on CPU and 1.5 GB with the CUDA runtime mapped in; before any document is processed the server sits near 100 MB, because models load lazily on first use.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/montignypatrik/nam-redactor.git
cd nam-redactor

# Install dependencies (Python 3.11+ / 3.12 recommended)
pip install -e .
```

Or install directly via `requirements.txt`:

```bash
pip install -r requirements.txt
```

### GPU (optional)

`pip install -e .` pulls the **CPU** build of Paddle, which is what CI and most
installs want. To run inference on an NVIDIA GPU, swap in the GPU wheel. Both
builds own the `paddle` namespace, so the CPU one has to be removed first:

```bash
pip uninstall -y paddlepaddle
pip install "paddlepaddle-gpu==3.3.1" -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

Choose the index that matches your CUDA version: `cu118`, `cu126` or `cu129`.
No local CUDA toolkit is required — the wheel bundles the runtime. Verify with:

```bash
python -c "import paddle; print(paddle.device.is_compiled_with_cuda(), paddle.device.cuda.device_count())"
```

The app picks the GPU up automatically; there is nothing else to configure. If
the GPU build is absent or cannot reach a device, it silently uses the CPU.

---

## Usage

Start the app:

```bash
nam-redact-web
# or
python -m nam_redactor.web
```

It serves <http://127.0.0.1:8000> and opens your browser.

```
usage: nam-redact-web [-h] [--port PORT] [--no-open] [--data-dir DATA_DIR]

options:
  -h, --help           Show help message and exit
  --port PORT          Port to listen on (default: 8000)
  --no-open            Do not open a browser window on startup
  --data-dir DATA_DIR  Where documents are saved (see "Storage and deletion")
```

There is deliberately no `--host` option: this service holds health
documents, so exposing it on a network interface should not be one flag away.

### The workflow

1. **Upload** — drop in a PDF, choose the **doctor** it belongs to (see
   *Doctor profiles* below; the licence, group and sector numbers come from
   the profile), and pick a rasterization DPI (150 for lower memory, 200
   default, 300 for the sharpest scan). Your browser remembers the last doctor
   chosen.
2. **Choose zones** — every page is shown. Drag to draw zones around the
   parts you want to keep: each zone becomes a page of its own, so a dense
   page listing several patients turns into several small pages with one
   patient each. Zones are numbered in the order they will appear (page by
   page, then top to bottom, left to right). Pages with no zone are left out.
   *Whole page* keeps a page as it is; *Use whole pages* skips this step.
   Every page you keep needs a **date**. Each zone takes the date of the page
   it came from, and you can change it per page during review. Dates are
   stored with the job for later use; they are not drawn on the output.
3. **Detect** — each page (or zone) is rendered and run through OCR. Progress is
   streamed live, with a per-page count as it goes.
4. **Review** — every detected number is shown as a box, stamped with the
   patient ID it will carry. `Patient_xxxx` is drawn in white inside each
   blackout, on screen and in the output, so a repeat patient is recognizable
   at a glance. **Drag** on the page to add a box, **click** a box and press
   <kbd>Delete</kbd> to remove one, or use *Reset to detected* to undo your
   edits. Detection is good but not perfect, so this step is where you catch a
   miss: draw a box over the number and type it in, and it joins the same
   numbering as that patient's other records.
   Review is organised by **original page**: each one is shown whole, with
   its sections (zones) outlined in place and everything outside them greyed
   out. Draw boxes directly on it; a box is stored in the section it was drawn
   in, and each section keeps its own date.
   **Save review** stores your boxes and page dates as a numbered revision.
   Detections are kept separately and never edited, so *Reset to detected*
   always goes back to them. The saved review is authoritative: every evidence
   crop is rendered from it.
5. **Appointment rows** — the *Appointment rows* tab of the review. See below.

Patient names show up in review as plain blackouts with an amber outline. For
a name detection missed, draw a box and choose *It's a name*.

### Doctor profiles

Every PDF belongs to exactly one doctor, and that doctor applies to all its
pages and appointment rows. Doctors are kept as **profiles**, managed from
**Doctor profiles** (top of the start screen, or *Manage doctor profiles*
beside the doctor selector). A profile holds:

- the doctor's name and their **licence**, **group** and **sector** numbers —
  optional, kept exactly as typed (leading zeros included), letters, digits,
  spaces and dashes only;
- **interpretation instructions** — ordinary text, in your own words:
  abbreviations, units, which context matters, exclusions, exceptions;
- **worked examples** — an annotation as written, its expected
  interpretation, and an optional note.

Instructions and examples are what billing interpretation follows (see
*Proposed billing* below): they are the **only** source of billing mappings —
nothing is taken from general model knowledge, and nothing doctor-specific
lives in the app's code. A profile with only a name can be used to upload and
transcribe, but billing cannot be proposed until its version has instructions
or examples. A ready-to-paste example of such instructions is in
`docs/doctor-profiles/`.

**Versions.** Saving an edit never changes a stored version: it adds a new
one (*Save as new version*), and every earlier version stays viewable under
*Versions*. SQLite enforces this: version rows cannot be updated or deleted.
Saving without a change adds nothing, and an editor opened on an older
version cannot overwrite a newer one.

**Duplicate** creates a new, independent profile from the current version
(e.g. for a doctor with a similar practice). **Archive** hides a profile from
new uploads and assignments; documents already using it keep it, and its
versions stay readable. **Restore** makes it assignable again. Profiles are
never deleted.

**Uploading** requires an active profile. With none yet, the start screen
offers to create one — the chosen file is kept meanwhile. The browser sends
only the profile reference; the server looks up the profile, checks it is
active and at its current version, and takes the identifiers from it.
Identifiers typed into an upload request are refused.

**Each document keeps its assignment**: the profile, the exact version, a
snapshot of that version's name, licence, group and sector, and when it was
assigned. The document's licence/group/sector — shown in review and exported
as `document.practice` — always equal that snapshot. A later edit of the
profile does **not** reach existing documents: the review header shows
*Adopt vN* when a newer version exists, and **Change doctor** assigns a
different profile. Both are explicit, apply to the whole document, keep the
earlier assignment in the document's history, and leave redactions, rows,
extraction results and reviews untouched (extraction is a literal
transcription and does not depend on the doctor). Every assignment gets a new
`assignment_id`, so the later interpretation step can record the exact
version it used and tell when the assignment has changed since.

**Documents from before doctor profiles** open as before, with the licence,
group and sector typed at the time, and show *No doctor profile assigned*.
Nothing is guessed from those numbers and no profile is created for them;
**Assign a doctor** in the review header links one explicitly (the old
values are kept in the document's history). Their export keeps
`document.practice` as it was, with `document.doctor: null`.

### Appointment rows

A row is one appointment: a region of a page, the patient it belongs to, its
date and its time. **Every section (zone) is one appointment row**, created
automatically when detection finishes and covering the whole section; a
patient with several appointments simply has several sections.

The whole redacted row is the evidence for a later extraction step, which
will read both the appointment time and the handwritten notes from it. So
preparing a row needs only its patient and date — and the app already knows
both:

- the **patient** is the pseudonym of the NAM detected in the section (when
  there is exactly one);
- the **date** is the date you entered for the section.

These count as confirmed; there is nothing to confirm row by row. You can
still change either one, and a date you type for a row is yours: it no longer
follows the section date. A section with no NAM, or two different ones, shows
*Needs patient* until you pick one.

1. Open the **Appointment rows** tab (it saves the review first). Original
   pages are shown with every section in place and the saved redactions
   burned in, exactly as the evidence will show them. Every section is listed
   as a row, named after the section (`#1`, `#2`, …).
2. Press **Extract all rows** (or **Extract row** for one). Every row whose
   patient and date are known is marked **ready for extraction**
   automatically — its full redacted row crop is frozen to disk with a SHA-256
   — and then extracted. Rows that still need a patient are listed, not
   guessed.
3. Review each extraction (below).

**Mark ready for extraction** remains in the row panel to do it by hand.
Readiness records how it was set (`method: "automatic"` or `"manual"`).

Leave the **time** blank in preparation — it shows *To be extracted*. Type it
only to override; a typed time must be `HH:MM`.

A section's row cannot be deleted or resized — it is the section.

Evidence files are never overwritten: each name includes the row revision and
the image's own SHA-256, so a changed redaction produces a new file, and every
extraction attempt keeps pointing at the exact image it was sent. Superseded
files that nothing refers to are cleaned up; deleting the document removes
them all. Before each request — and before each retry — the app re-checks
that the row is still ready, that its evidence is still the one queued, and
that the file still matches its hash; if not, nothing is sent and the row
shows *Not sent: … Extract again*.

*Ready for extraction* covers the row's region, patient, date, any typed time,
and the redaction boxes inside the row. Changing any of them — including a
section date change, or saving a review that adds, removes or relabels a box
inside the row — marks it **invalidated**, with the reason; the next
extraction prepares it again, and any earlier extraction of it becomes
*stale*. Readiness does not establish billing eligibility; the app assigns no
billing codes.

Rows prepared by earlier versions (row schema 1, with a separate handwriting
region and a required time) are migrated when their document is opened:
confirmed values are kept, the old handwriting region, evidence references
and approval are kept under `legacy` (their files stay on disk), and rows
that had been approved ask to be **marked ready again** with the reason shown.

### Extracting time and handwriting (cloud, optional)

Once rows are **ready for extraction**, a vision model can read each row's
appointment time and transcribe its handwriting. You then review, correct and
accept every result. Nothing is accepted automatically.

**Set-up.** Paste an OpenRouter API key into the **OpenRouter API key** field
on the Appointment rows tab. It is saved on this computer only, in
`<data dir>/private/openrouter_api_key.txt`, and takes effect at once; it is
never logged, never sent back to the browser and never exported. **Replace**
and **Remove** manage it later.

Alternatively, set the environment variable `OPENROUTER_API_KEY` before
starting the app; when set, it takes precedence over a saved key:

```powershell
# Windows (PowerShell), current session
$env:OPENROUTER_API_KEY = "sk-or-v1-<your key>"
python -m nam_redactor.web
```

```bash
# macOS / Linux
export OPENROUTER_API_KEY="sk-or-v1-<your key>"
python -m nam_redactor.web
```

The Appointment rows tab shows whether extraction is configured. Without a
key, everything else works and the extraction buttons stay disabled.

**Models.** Pick one in the **Model** selector (remembered by your browser):

| Model ID | Notes |
| --- | --- |
| `anthropic/claude-opus-5.5` | default |
| `openai/gpt-6-astra` | |
| `google/gemini-3.8-flash` | lowest cost |

Each was checked against OpenRouter's catalog (September 2026) for image
input, JSON-schema output and at least one zero-data-retention endpoint that
supports it. Models behave differently: on a synthetic test, two of them
declined to treat typed text as handwriting and one transcribed it anyway.
**Handwriting accuracy has not been evaluated.** Measure it on representative,
annotated rows before relying on any model.

**What is sent, and how.** Per row: the frozen, redacted full-row evidence
image (exactly the file that was approved, checked against its SHA-256) and
fixed instructions. No pseudonym, date, filename, identity map or other
document content is sent. Every request pins the routing:
`provider.zdr: true` (zero-data-retention endpoints only),
`provider.data_collection: "deny"`, and `provider.require_parameters: true`
(only endpoints that honour the strict JSON schema). If no endpoint meets all
three, the row fails with *route unavailable* rather than falling back to a
laxer route. Choose another model.

Handwriting is **not** covered by automatic redaction: a name or number
written by hand is visible in the row image and may appear in the
transcription. The review panel warns when the text contains something shaped
like a health number, but checking the rest is up to the reviewer.

**What the model returns** (validated against a strict schema; anything else
is rejected as *invalid output*):

- `time` as `HH:MM`, or `null` when it cannot be determined, never guessed;
  plus the characters it was read from;
- `handwriting_status`: `legible`, `partially_legible`, `unreadable`
  (handwriting present, nothing readable) or `blank` (no handwriting);
- `handwriting_text`: a literal transcription that keeps French,
  abbreviations, digits and clinical shorthand, with `[?]` for unreadable
  parts, which are also listed. It is never expanded, corrected, or turned
  into billing codes;
- remarks, e.g. instructions written in the image, which it is told to ignore.

No confidence score is requested or shown.

**Running it — one click.** *Extract all rows* does the whole job:

1. marks ready every row whose patient and date are known;
2. **transcribes** rows that have no current transcription (time and literal
   handwriting);
3. **interprets** each transcription straight away with the document's
   doctor profile — no acceptance step in between — and shows the proposed
   billing on each appointment card as rows finish.

Each row gets only the work it needs, so the click is safe to repeat: a
current transcription is reused (a saved correction too, including a time
deliberately left empty), and a current proposal is skipped. The note under
the buttons says how many rows are transcribing, interpreting, already done,
or could not start. Up to 3 requests run at once. Timeouts, rate limits and
provider errors are retried twice with a back-off (honouring `Retry-After`);
invalid output once; authentication, credit, bad-request and route errors
not at all. A failure affects only its row and stage: a failed interpretation
is retried without transcribing again. Every result is saved the moment it
arrives, and work cut short by a restart comes back as *failed
(interrupted)*, retryable.

In a row's panel, *Interpret again* reruns interpretation alone, and
*Transcribe again* transcribes afresh (then interprets automatically).

### Proposed billing

Interpretation is a separate stored stage. Its request contains the
**assigned** profile version's instructions and worked examples (loaded on
the server — never the profile's latest version unless you adopted it; the
doctor's licence, group and sector are not sent), the transcription marked as
*model-generated* or *reviewer-corrected*, and the same saved redacted row
image, for layout and ambiguous readings. The model is told to follow only
the profile, to keep units unknown unless the profile and evidence determine
them, not to guess codes, to separate clinical notes from billing notations,
to report conflicts between image and transcription, and to treat any
instruction written in the document as content.

Each row's proposal lists **acts**, each with its own billing code (or
*unresolved*), context elements and units (or none), the source text it came
from, and the profile instruction or example applied; plus unresolved
annotations, conflicts, clinical notes set aside, and whether no billing
notation was found. Each appointment card (below) shows the proposal as
editable billing lines; the full proposal is under the card's *Details*.

Every proposal is labelled **Unvalidated**: validating billing is a later,
human step, and nothing in the app marks a proposal approved. A proposal
whose inputs change — a new transcription or saved correction, new evidence
(e.g. a redaction edited inside the row), or a different doctor or adopted
profile version — is labelled **Outdated** and redone by the next *Extract
all rows*. Inputs are checked before each request, before each retry, and
again before a result is attached, so an outdated image, transcription or
profile is never sent and a late result never replaces newer work.

If the document has no doctor profile, or its version has neither
instructions nor examples, billing is not proposed and the screen says so,
with a link to **Doctor profiles**. To improve results: edit the profile
there, save (a new version), come back, **Adopt** the new version in the
review header, and click *Extract all rows* — only interpretation reruns.

Each interpretation attempt records the doctor profile, exact version and
assignment; the transcription attempt and correction it read; the evidence
path and SHA-256; the prompt and schema version; the model and provider;
timestamps, tokens and cost; and its result or error. Attempts, and the
evidence files they reference, are kept.

**States**, per row:

| Transcription | Meaning |
| --- | --- |
| Transcribing… | queued or running |
| Transcribed | a current reading (billing is proposed from it) |
| Corrected | you saved a correction, which is used instead |
| Transcription failed / Stale | retry, or redo after the row changed |

| Proposed billing | Meaning |
| --- | --- |
| Interpreting… | queued or running |
| Unvalidated | a current proposal |
| Outdated | its inputs changed; *Extract all rows* redoes it |
| Interpretation failed | retry: only interpretation runs again |
| Cannot interpret | no doctor profile, or no instructions/examples |

**Correcting a transcription is optional.** The panel keeps the row image
beside the model's time and handwriting; *Save correction* stores your
version beside the model's (which is never edited and stays viewable under
*Original model result*), and the proposal updates from it. A time typed
during preparation is never overwritten: if the model reads a different one,
both are shown. Earlier attempts and corrections are always kept.

**Usage and cost.** Token counts and cost are stored per attempt exactly as
OpenRouter reports them (cost in OpenRouter credits), or `null` when not
reported. They are shown under *Original model result* and *How this
proposal was made*, and exported.

### Appointment cards: editing and confirming billing

The appointment rows tab opens on **Appointments**: one card per appointment,
in document order. Each card shows, without clicking anything:

- the row number, patient, date and time, and its state: **Draft**,
  **Confirmed**, or **Needs attention**; a **⚠ warnings** chip when the model
  reported conflicts or unresolved notes;
- the **whole redacted crop at full resolution** (scroll sideways when it is
  wider than the card; *Fit images to width* scales every crop to the card
  instead, and is remembered in this browser);
- its **billing lines** in the CSV's 17 columns and order, one line per
  procedure code, the appointment's values repeated on each line. Only
  **ProcedureCode**, **Units** and **ContextualElements** are editable. The
  doctor, date, time and HIN cells are read-only, and the unused columns are
  always empty. Patient, date and time are fixed in **Page layout** (the
  card's *Fix patient or date* link opens it on that row);
- **+ Add code**, **No billing acts**, and **Confirm row**;
- **Details** (collapsed): transcription and its correction, the model's
  proposal with its sources, conflicts, unresolved notes and set-aside
  clinical notes, usage, provenance, and *Interpret again* / *Transcribe
  again*.

Overall progress (“12 of 98 appointments confirmed”, and how many need
attention) is shown above the cards. **Page layout** is the original page
view, with the row editor, as before.

**Editing.** Correct a code, units or contexts, remove a line (×), or add
one: *+ Add code* adds an empty line and puts the cursor in its code. Codes
are kept exactly as typed (only surrounding spaces are dropped; no case
change, no mapping). Units are empty or a number (`2`, `2.5`, `2,5`); empty
stays empty, and nothing inserts 0 or 1. Contexts are comma-separated
(`TT, XY` is saved as `TT` and `XY`, in that order); empty means none.
Changes are **saved automatically** a moment after you stop typing (*Saving…*
/ *Saved*), survive reloads and restarts, and call no model.

Manual lines are stored **beside** the model's proposal, which is never
changed and stays under *Details*. Each save records what it relates to: the
proposal (interpretation attempt), the transcription and evidence, the doctor
assignment and profile version, and the patient, date and time. *Discard my
edits* goes back to the proposal; the discarded lines stay in the history.

**Confirming.** *Confirm row* saves the lines shown and marks the whole
appointment **Confirmed**. It is its own state: it is not transcription
acceptance or readiness for extraction, and it does not require either. Blank
units and contexts are fine, but a line without a code is not (enter one or
remove the line); to confirm an appointment with nothing to bill, tick **No
billing acts**. If the appointment could not be exported (for example, no
health number on record for its patient), confirming is refused with the
reason and the lines are kept as a draft. A confirmed card keeps its image
and lines, read-only; **Reopen for editing** makes it a draft again. Every
edit and confirmation is kept in the history.

**When the source changes.** If the doctor assignment or adopted version,
the evidence, the transcription or correction, the patient, date or time, or
the proposal itself changes after you edited or confirmed a row, your lines
are kept and the card shows **Needs attention**: the confirmation is
*outdated*, and those lines are not exported until you choose **Keep my
lines** (saved against the new source; then confirm again) or **Use the new
proposal**. A model result arriving later — even one you requested — never
replaces lines you edited, and a card you are typing in is never overwritten
by the screen's automatic refresh. A save made from an out-of-date screen is
refused with a message, and the card offers to keep your lines or discard
them.

Extraction does not depend on confirmation, and export does not require it.

### Billing CSV export

**Export CSV** (in the appointment rows workspace) downloads the billing
lines shown on the cards, in the practice's billing-software format:

- 17 columns in this exact order, with a header:
  `doctorLicenceID;doctorGroupNumber;ClaimDate;ProcedureTime;HIN;PatientGender;PatientFirstName;PatientLastName;PatientBirthdate;PatientParentHIN;Sector;Diagnosis;ProcedureCode;Units;ContextualElements;referringDoctor;CSSTAccidentDate`
- `;` delimiter, UTF-8 without BOM, CRLF line endings, standard CSV quoting,
  empty cells for missing values (never `null`, `None` or `0`).
- **One line per billing line.** The appointment's information repeats on
  each of its lines. Appointments follow the cards' order, and lines keep
  their order.

| Column | Source |
| --- | --- |
| doctorLicenceID, doctorGroupNumber, Sector | the document's assigned doctor snapshot (not a later, un-adopted profile edit) |
| ClaimDate | the row's date (`YYYY-MM-DD`) |
| ProcedureTime | the time the current proposal was made from: your saved correction (a time you cleared stays empty), else the model's reading, else a time you typed during preparation |
| HIN | the patient's health number, resolved locally from the private pseudonym map |
| ProcedureCode | the line's code **exactly** as shown (`vs` stays `vs`, `15824` stays `15824`). There is no mapping table. |
| Units | the line's units, or empty |
| ContextualElements | the line's contexts joined with `,` without spaces |
| all other columns | always empty |

For each appointment the file uses its **saved manual lines** when there are
any, else its **current model proposal**: the same effective values the card
shows. Removed lines never reappear, added lines export as extra lines, and
confirming changes no spelling and fills no blank. Drafts export too:
confirmation is not required, and exporting confirms nothing. The file is
built on this machine and the export makes no model request. It contains
real health numbers by design (the model never receives them; the cards show
them only on this machine), so handle it like the source documents.

Nothing is dropped silently. If any appointment cannot be exported, no file
is produced and the screen lists each affected row (click one to open it) and
the reason:

- not transcribed yet (and no manual lines);
- the proposal is missing, failed, still running, or outdated (and no
  manual lines);
- manual lines whose source has changed since (*Needs attention*);
- a line has no code (unresolved, or a line added and left empty);
- no billing lines, and not marked *No billing acts*;
- no health number is on record for its patient;
- no doctor profile is assigned.

An appointment with **no billing acts** (found by the model, or ticked by
you) produces no lines, and the screen counts it. Informational conflicts and set-aside clinical notes do not
block the export.

### Exporting rows

**Export rows JSON** (the JSON/evidence export, unchanged) downloads a versioned manifest
(`schema: nam-redactor.appointment-rows`, `schema_version: 3`). It contains
only rows marked ready for extraction whose readiness is still current. Every
other row is listed under `excluded` with its reasons, and the screen reports
them too.

For each row the manifest records:

- `row_id` (stable, distinct from the patient pseudonym), `patient_id`, `date`,
  and `time`: the reviewed extraction time (`time_status: "reviewed_extraction"`),
  else a time typed during preparation (`"entered"`), else `null`
  (`"to_be_extracted"`); an unknown time is never guessed or marked confirmed;
- each confirmed field's source (`field_sources`);
- `row_region` on the **original** page: `source_page`, `bounds`
  (normalized), `bounds_pt` (PDF points, as displayed) and
  `bounds_pdf_unrotated` (the page's own coordinate space, rotation undone),
  plus the working-page `working_bounds`;
- `evidence`: the frozen full-row crop, as a path relative to the document's
  folder, with SHA-256, pixel crop and region;
- `ready_for_extraction`: when, at which row revision, and against which saved
  redaction revision — the *preparation* review;
- `extraction`: the *extraction* review, kept separate. `status` is one of the
  states above. Only when it is `reviewed` does `reviewed` hold the accepted
  time, handwriting status and literal text, the time's source (`confirmed`,
  `model` or `reviewer`), what was corrected, any discrepancy with a confirmed
  time, the model/provider/prompt version, the evidence it was based on, and
  tokens and cost. For every other status no model output is exported.

- `billing_draft`: the proposed billing, in its own section marked
  `"validated": false`. `status` is the proposal state; `proposal` holds the
  acts, unresolved items, conflicts and clinical notes **only while the
  proposal is current**, with what it was based on (doctor profile, version
  and assignment; transcription source, attempt and correction; evidence
  hash), the model and prompt version, and usage. These are drafts for later
  validation — not billing entries.

`summary.extraction` and `summary.billing_draft` count rows by status, and
`extraction_not_reviewed` lists every exported row whose time and handwriting
have not been reviewed, so unreviewed output never passes as reviewed.

Coordinate conventions, the render DPI, the original PDF's SHA-256, the
licence/group/sector (`practice`) and the assigned doctor profile version
(`doctor`: profile, version, assignment, name; `null` if none) are in
`document`. Mapping back never relies on working-page numbers: each working
page records its source page, the exact pixel crop taken from it, the render
DPI and raster size, and the source page's size and rotation. Rows stay
traceable through cropping, reordering, edits and restarts.

The manifest holds no NAM, no detected patient name, no OCR text, no
filename (a filename is often a patient's name) and no API key. No confidence
scores are included. The reviewed handwriting text is included as accepted —
see the note above on names written by hand.

### Storage and deletion

Documents are saved on this computer as soon as they are uploaded, and after
every change. Close the app at any point; on the next start they are listed
under **Saved documents** and reopen where you left off. A detection that was
interrupted goes back to the zone step with its zones kept. An interrupted
render goes back to review.

They live in a per-user data folder, never the working directory:

| Platform | Default location |
| --- | --- |
| Windows | `%LOCALAPPDATA%\nam-redactor` |
| macOS | `~/Library/Application Support/nam-redactor` |
| Linux | `$XDG_DATA_HOME/nam-redactor` (or `~/.local/share/nam-redactor`) |

Override it with `--data-dir` or the `NAM_REDACTOR_DATA` environment variable.
Each document is a folder `documents/<id>/`:

```
original.pdf              the upload, exactly as received (never modified)
zones.pdf                 working pages rebuilt from zones, if any
redacted.pdf              the last rendered output, if any
document.json             provenance, detections, reviewed redactions,
                          page dates, appointment rows (no NAMs, names or text)
private/identities.json   NAM -> pseudonym map, kept apart from everything else
evidence/<row_id>/...     evidence crops frozen at approval
```

Doctor profiles live beside the documents, in `doctors.sqlite3` in the same
data folder. Deleting a document does not touch them; profiles are archived,
never deleted, because documents may refer to any of their versions.

The identity map is the only file linking pseudonyms to health numbers. It
never leaves the machine and is never included in a manifest. Pseudonyms are
scoped to their document: `Patient_0001` in one document and in another are
unrelated.

Stopping the app no longer deletes anything. To remove a document, use
**Delete** in the document list. That permanently removes its whole folder:
the original PDF, the review, the rows, the evidence and the identity map.

### From Python

The web layer is a thin wrapper over the library, which is still importable
if you want to script it:

```python
from pathlib import Path
from nam_redactor import redact_pdf

total, per_page, _ = redact_pdf(
    input_path=Path("document.pdf"),
    out_path=Path("document.redacted.pdf"),
    dpi=200,
)
print(f"{total} box(es) redacted across {len(per_page)} pages")
```

From Python, NER is opt-in: pass `name_model=default_name_model()` (from
`nam_redactor.names`) to `redact_pdf` or `detect_nam_pages`. Without it, only
names confirmed by a NAM or a label are redacted. The web app always uses it.

For anything that needs to inspect or edit the detected regions before
writing, use the two phases directly:

```python
from nam_redactor import detect_nam_pages, render_redacted_pdf

pages, _ = detect_nam_pages(Path("document.pdf"), dpi=200)
for page in pages:
    print(f"page {page.page_num}: {len(page.boxes)} box(es)")  # NAMs, then names

render_redacted_pdf(Path("document.pdf"), Path("out.pdf"), pages, dpi=200)
```

Pass `debug=True` to `redact_pdf` to also export `<output_stem>_debug/`:
`boxes.json` and `page_<N>_boxes.json` hold coordinates `(x1, y1, x2, y2)`
normalized 0..1 plus confidence scores only — **no PII text is stored** — and
`page_<N>_preview.png` renders the redacted page for visual inspection. The
web app does not use debug mode; its previews are rendered from the source
document on demand instead.

---

## Testing

```bash
pytest
```

The suite covers isolated NAMs, label exclusions, adjacent block splits,
non-NAM word filtering, and end-to-end PDF vector text removal. It also
covers mapping zone coordinates back to the original page (including rotated
pages), repeated visits, persistence across a restart, evidence crops that
follow the reviewer's redactions, approval invalidation, and export privacy. The web tests
stub the OCR engine, so they run in seconds without loading Paddle.

---

## Development

The compiled frontend is committed to `nam_redactor/web/dist/` so that
`pip install` yields a working app with no Node toolchain. If you change
anything under `frontend/`, rebuild and commit the output:

```bash
cd frontend
npm ci
npm run build
```

CI rebuilds the bundle and fails if `nam_redactor/web/dist` differs from what
is committed, so a forgotten rebuild cannot ship a stale UI.

To work on the UI with hot reload, run the API and the Vite dev server
together — Vite proxies `/api` to the backend:

```bash
nam-redact-web --no-open          # terminal 1
cd frontend && npm run dev        # terminal 2, then open http://127.0.0.1:5173
```

---

## License

[MIT](LICENSE) © Patrik Montigny
