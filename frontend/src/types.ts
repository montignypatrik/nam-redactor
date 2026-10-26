/** Normalized (0..1) blackout rectangle, relative to the page raster. */
export interface Box {
  x1: number
  y1: number
  x2: number
  y2: number
  confidence: number
  /** Pseudonym drawn in white inside the box. Absent on hand-drawn boxes the
   *  user has not labelled yet, and always absent on name boxes. */
  patient_id?: string
  /** Present only on patient-name boxes, which are blacked out unlabelled.
   *  Absent means a NAM. */
  kind?: 'name'
}

export interface PageInfo {
  page_num: number
  width: number
  height: number
  /** The saved reviewed redactions (the detections until a review is saved). */
  boxes: Box[]
  /** The initial detections. Never edited; "Reset to detected" uses them. */
  detected_boxes: Box[]
  /** ``YYYY-MM-DD``, inherited from the source page chosen at the zone step. */
  date?: string | null
  /** Page of the original upload this working page was cut from. */
  source_page: number | null
  /** Where this page sits on that original page, normalized. */
  source_crop: Bounds | null
}

/** One page as sent to the server: its boxes (or zones) and its date. */
export interface PagePayload {
  page_num: number
  boxes: Box[]
  date?: string
}

export interface PagesResponse {
  job_id: string
  filename: string
  status: string
  dpi: number
  review_revision: number
  reviewed_at: string | null
  doctor: DoctorAssignment | null
  licence_number: string
  group_number: string
  sector_number: string
  /** Every pseudonym in the document. Never the NAM behind it. */
  patients: string[]
  total_pages: number
  total_nams: number
  total_names: number
  per_page_counts: number[]
  pages: PageInfo[]
}

/** Licence, group and sector numbers typed before upload. Blank if not given. */
export interface Practice {
  licence_number: string
  group_number: string
  sector_number: string
}

export interface UploadResponse extends Practice {
  job_id: string
  filename: string
  dpi: number
  total_pages: number
  /** ``null`` only for documents from before doctor profiles. */
  doctor?: DoctorAssignment | null
}

/** Event payloads published by the backend over SSE. */
export interface ServerEvent {
  type:
    | 'queued'
    | 'start'
    | 'page'
    | 'awaiting_review'
    | 'rendering'
    | 'done'
    | 'error'
    | 'deleted'
  page_num?: number
  total_pages?: number
  nams_on_page?: number
  total_nams?: number
  total_names?: number
  per_page_counts?: number[]
  result_url?: string
  message?: string
}

export type Phase = 'detect' | 'finalize'

/** A saved document, as listed on the start screen. */
export interface DocumentSummary extends Practice {
  job_id: string
  filename: string
  status: string
  dpi: number
  created_at: number
  source_page_count: number | null
  total_pages: number | null
  review_revision: number
  rows_total: number
  rows_approved: number
  doctor: DoctorAssignment | null
}

export interface JobStatus extends DocumentSummary {
  /** Zones and dates as last submitted, keyed by source page number. */
  zone_request: { zones: Record<string, Box[]>; dates: Record<string, string> } | null
  error: string | null
}

export interface Bounds {
  x1: number
  y1: number
  x2: number
  y2: number
}

export type FieldStatus = 'missing' | 'suggested' | 'confirmed'
export type FieldName = 'patient_id' | 'date' | 'time'

export interface RowField {
  value: string | null
  /** ``manual``, ``page_date`` or ``nam_in_row``. */
  source: string | null
  status: FieldStatus
}

export interface EvidenceRef {
  /** ``handwriting`` only in rows prepared under schema 1. */
  kind: 'row' | 'handwriting'
  path: string
  sha256: string
}

/** One appointment row (schema version 1), plus server-derived review state. */
export interface Row {
  schema_version: number
  row_id: string
  /** ``section``: the automatic row of a section, covering all of it.
   *  ``manual``: drawn by hand. */
  origin?: 'section' | 'manual'
  document_id: string
  working_page: number
  source_page: number
  bounds: Bounds
  fields: Record<FieldName, RowField>
  evidence: EvidenceRef[]
  revision: number
  created_at: string
  updated_at: string
  approval: {
    status: 'unapproved' | 'approved' | 'invalidated'
    approved_revision: number | null
    approved_at: string | null
    redaction_revision: number | null
    invalidated_at: string | null
    invalidated_reasons: string[]
  }
  /** Required fields (patient, date) not yet confirmed. Time never blocks. */
  unresolved: string[]
  /** Ready for extraction, and nothing it vouches for has changed since. */
  approval_current: boolean
  /** Kept from row schema 1: the old handwriting region, evidence and
   *  approval. Informational only. */
  legacy?: { migrated_from_schema: number; handwriting_bounds: Bounds | null }
  extraction: ExtractionView
  /** Proposed billing, a separate stage after transcription. Unvalidated. */
  interpretation: InterpretationView
  /** The billing lines as exported, and the person's billing confirmation. */
  billing: BillingView
}

export interface RowsResponse {
  rows: Row[]
  patients: string[]
  review_revision: number
}

/** What the reviewer may change on a row. A field value set here counts as
 *  entered by the reviewer (confirmed); ``confirm`` accepts suggestions. */
export interface RowPatch {
  bounds?: Bounds
  patient_id?: string | null
  /** The reviewer states the row has no patient (confirmed, with no value). */
  no_patient?: boolean
  date?: string | null
  time?: string | null
  confirm?: FieldName[]
}

export interface ManifestSummary {
  summary: { rows_total: number; rows_exported: number; rows_excluded: number }
  excluded: { row_id: string; source_page: number; reasons: string[] }[]
}

/** Cloud extraction set-up, as the server reports it. Never includes the key. */
export interface ExtractionConfig {
  configured: boolean
  key_variable: string
  /** ``environment`` wins over ``saved`` (typed in the app). */
  key_source: 'environment' | 'saved' | null
  models: { id: string; label: string }[]
  default_model: string
  routing: Record<string, unknown>
  prompt_version: string
}

export type HandwritingStatus = 'legible' | 'partially_legible' | 'unreadable' | 'blank'

export interface ExtractionResult {
  time: string | null
  time_evidence: string | null
  handwriting_status: HandwritingStatus
  handwriting_text: string | null
  unreadable_or_ambiguous: string[]
  remarks: string | null
}

export interface ExtractionAttempt {
  attempt_id: string
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'interrupted' | 'superseded'
  requested_model: string
  returned_model: string | null
  provider: string | null
  prompt_version: string
  schema_version: number
  finished_at: string | null
  tries: number
  result: ExtractionResult | null
  usage: {
    prompt_tokens: number | null
    completion_tokens: number | null
    total_tokens: number | null
    cost: number | null
    cost_currency: string | null
  } | null
}

export interface ExtractionReview {
  review_id: string
  attempt_id: string
  time: string | null
  time_source: 'confirmed' | 'model' | 'reviewer'
  handwriting_status: HandwritingStatus
  handwriting_text: string | null
  corrected: { time: boolean; handwriting: boolean }
  reviewed_at: string
}

export type ExtractionState =
  | 'not_ready'
  | 'pending'
  | 'running'
  | 'failed'
  | 'stale'
  | 'unreviewed'
  | 'reviewed'

/** A row's extraction, derived by the server from its attempts and reviews. */
export interface ExtractionView {
  state: ExtractionState
  /** The current result, or the latest one when it is stale. */
  attempt: ExtractionAttempt | null
  attempt_is_current: boolean
  review: ExtractionReview | null
  last_error: { kind: string; message: string } | null
  last_attempt_status: string | null
  attempt_count: number
  review_count: number
  confirmed_time: string | null
  time_discrepancy: { confirmed: string; proposed: string } | null
  possible_nam_in_text: boolean
}

/** A worked example: how one of this doctor's annotations should be read. */
export interface DoctorExample {
  annotation: string
  interpretation: string
  note: string
}

/** One immutable version of a doctor profile. */
export interface DoctorVersion extends Practice {
  profile_id: string
  version: number
  display_name: string
  instructions: string
  examples: DoctorExample[]
  created_at: string
}

export interface DoctorProfileSummary extends Practice {
  profile_id: string
  display_name: string
  status: 'active' | 'archived'
  current_version: number
  created_at: string
  updated_at: string
}

export interface DoctorProfile extends Omit<DoctorProfileSummary, keyof Practice> {
  current: DoctorVersion
  versions: (Practice & {
    version: number
    display_name: string
    created_at: string
    example_count: number
    has_instructions: boolean
  })[]
  /** Set by a save that changed nothing (no new version). */
  unchanged?: boolean
}

/** The doctor a document is assigned: an exact profile version, a snapshot
 *  of it, and what the profile says now. */
export interface DoctorAssignment {
  assignment_id: string
  profile_id: string
  version: number
  snapshot: Practice & { display_name: string }
  assigned_at: string
  profile_status: 'active' | 'archived' | 'missing'
  current_version: number | null
  newer_version_available: boolean
}

/** One proposed billing act. Unvalidated. */
export interface BillingAct {
  code: string | null
  units: number | null
  context_elements: string[]
  source_text: string
  profile_reference: string | null
}

export interface InterpretationResult {
  acts: BillingAct[]
  unresolved: { text: string; issue: string }[]
  conflicts: string[]
  ignored_notes: { text: string; reason: string }[]
  no_billing_notation: boolean
  remarks: string | null
}

export interface InterpretationAttempt {
  attempt_id: string
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'interrupted' | 'superseded'
  requested_model: string
  returned_model: string | null
  provider: string | null
  prompt_version: string
  finished_at: string | null
  tries: number
  /** What the proposal was based on. */
  inputs: {
    source: 'model' | 'reviewer'
    time: string | null
    handwriting_status: HandwritingStatus
    handwriting_text: string | null
    doctor: { profile_id: string; version: number; assignment_id: string }
  }
  result: InterpretationResult | null
  usage: ExtractionAttempt['usage']
  validated: false
}

export type InterpretationState =
  | 'not_ready'
  | 'blocked'
  | 'pending'
  | 'running'
  | 'failed'
  | 'stale'
  | 'proposed'

export interface InterpretationView {
  state: InterpretationState
  blocked_reason: string | null
  blocked_message: string | null
  /** The current proposal, else the latest one -- then ``is_current`` is false. */
  attempt: InterpretationAttempt | null
  is_current: boolean
  last_error: { kind: string; message: string } | null
  last_attempt_status: string | null
  attempt_count: number
  validated: false
}

/** The practice's billing columns, in export order. */
export const BILLING_COLUMNS = [
  'doctorLicenceID',
  'doctorGroupNumber',
  'ClaimDate',
  'ProcedureTime',
  'HIN',
  'PatientGender',
  'PatientFirstName',
  'PatientLastName',
  'PatientBirthdate',
  'PatientParentHIN',
  'Sector',
  'Diagnosis',
  'ProcedureCode',
  'Units',
  'ContextualElements',
  'referringDoctor',
  'CSSTAccidentDate',
] as const

export type BillingColumn = (typeof BILLING_COLUMNS)[number]

/** One billing line: one procedure code. */
export interface BillingLine {
  code: string
  units: number | null
  context_elements: string[]
  /** The proposed act this line started from; null when added by hand. */
  model_index: number | null
}

/** ``draft``: not confirmed. ``confirmed``: confirmed, and still current.
 *  ``outdated``: confirmed, but its source has changed since. */
export type BillingState = 'draft' | 'confirmed' | 'outdated'

export interface BillingView {
  state: BillingState
  /** Where the lines come from: saved manual lines, the current proposal, or neither. */
  source: 'manual' | 'model' | 'none'
  /** For manual lines: whether their source (proposal, doctor, evidence…) still holds. */
  edits_current: boolean | null
  /** The current proposal the lines relate to. */
  attempt_id: string | null
  interpretation_state: InterpretationState
  /** Appointment-level columns, as exported (HIN resolved locally). */
  shared: Record<Exclude<BillingColumn, 'ProcedureCode' | 'Units' | 'ContextualElements'>, string>
  lines: BillingLine[]
  no_billing: boolean
  confirmation: { confirmation_id: string; at: string } | null
  edited_at: string | null
  edit_count: number
  confirmation_count: number
  version: number
  /** Why the appointment cannot be exported (or confirmed) as it stands. */
  problems: string[]
  /** Informational: the proposal's conflicts and unresolved notes. */
  warnings: number
}

export interface BillingLinesPayload {
  acts: BillingLine[]
  no_billing: boolean
  base_version: number
  attempt_id: string | null
}
