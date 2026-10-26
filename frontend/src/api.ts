import type {
  BillingLinesPayload,
  Bounds,
  DoctorAssignment,
  DoctorExample,
  DoctorProfile,
  DoctorProfileSummary,
  DoctorVersion,
  DocumentSummary,
  ExtractionConfig,
  HandwritingStatus,
  JobStatus,
  ManifestSummary,
  PagePayload,
  PagesResponse,
  Phase,
  Row,
  RowPatch,
  RowsResponse,
  ServerEvent,
  UploadResponse,
} from './types'

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json()
    if (typeof body?.detail === 'string') return body.detail
  } catch {
    // fall through to the generic message
  }
  return `Request failed (${response.status})`
}

/** Upload a PDF for a doctor. Only the profile reference is sent: the server
 *  resolves the identifiers itself. */
export async function uploadPdf(
  file: File,
  dpi: number,
  doctor: { profileId: string; version: number },
): Promise<UploadResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('dpi', String(dpi))
  form.append('doctor_profile_id', doctor.profileId)
  form.append('doctor_profile_version', String(doctor.version))
  const response = await fetch('/api/jobs', { method: 'POST', body: form })
  if (!response.ok) throw new Error(await readError(response))
  return response.json()
}

export async function fetchPages(jobId: string): Promise<PagesResponse> {
  const response = await fetch(`/api/jobs/${jobId}/pages`)
  if (!response.ok) throw new Error(await readError(response))
  return response.json()
}

/**
 * Begin detection. Each zone becomes a page of its own; pages with no zone
 * are left out. An empty list keeps the document as uploaded.
 */
export async function startDetection(jobId: string, pages: PagePayload[]): Promise<void> {
  const response = await fetch(`/api/jobs/${jobId}/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pages }),
  })
  if (!response.ok) throw new Error(await readError(response))
}

/**
 * Resolve a hand-typed NAM to its pseudonym.
 *
 * The number goes up and only the pseudonym comes back, so the review screen
 * can show which patient a box belongs to without holding the number itself.
 */
export async function resolveNam(jobId: string, nam: string): Promise<string> {
  const response = await fetch(`/api/jobs/${jobId}/nam`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nam }),
  })
  if (!response.ok) throw new Error(await readError(response))
  const body = (await response.json()) as { patient_id: string }
  return body.patient_id
}

export async function deleteJob(jobId: string): Promise<void> {
  await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' })
}

/**
 * ``dpi`` lowers the resolution, for pages shown many at a time.
 * ``redactedAt`` renders the saved reviewed redactions onto the page; its
 * value (the review revision) also keeps the browser from reusing an image
 * of an older review.
 */
export function pageImageUrl(
  jobId: string,
  pageNum: number,
  dpi?: number,
  redactedAt?: number,
): string {
  const params = new URLSearchParams()
  if (dpi) params.set('dpi', String(dpi))
  if (redactedAt !== undefined) {
    params.set('redacted', 'true')
    params.set('rev', String(redactedAt))
  }
  const query = params.toString()
  return `/api/jobs/${jobId}/page/${pageNum}.png${query ? `?${query}` : ''}`
}

/** A page of the original upload. ``redactedAt`` burns in the saved review of
 *  every section cut from it (and busts the cache per review revision). */
export function sourceImageUrl(jobId: string, pageNum: number, redactedAt?: number): string {
  const query = redactedAt === undefined ? '' : `?redacted=true&rev=${redactedAt}`
  return `/api/jobs/${jobId}/source/${pageNum}.png${query}`
}

async function send<T>(url: string, method: string, body?: unknown): Promise<T> {
  const response = await fetch(url, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!response.ok) throw new Error(await readError(response))
  return (response.status === 204 ? undefined : await response.json()) as T
}

export async function listDocuments(): Promise<DocumentSummary[]> {
  const body = await send<{ documents: DocumentSummary[] }>('/api/jobs', 'GET')
  return body.documents
}

export function getJob(jobId: string): Promise<JobStatus> {
  return send(`/api/jobs/${jobId}`, 'GET')
}

/** Save the reviewer's redactions and dates as the authoritative revision. */
export function saveReview(
  jobId: string,
  pages: PagePayload[],
  baseRevision: number,
): Promise<{ review_revision: number; changed: boolean; invalidated_rows: string[]; status: string }> {
  return send(`/api/jobs/${jobId}/review`, 'PUT', { pages, base_revision: baseRevision })
}

export function listRows(jobId: string): Promise<RowsResponse> {
  return send(`/api/jobs/${jobId}/rows`, 'GET')
}

export function createRow(jobId: string, workingPage: number, bounds: Bounds): Promise<Row> {
  return send(`/api/jobs/${jobId}/rows`, 'POST', { working_page: workingPage, bounds })
}

export function updateRow(jobId: string, rowId: string, patch: RowPatch): Promise<Row> {
  return send(`/api/jobs/${jobId}/rows/${rowId}`, 'PATCH', patch)
}

export function deleteRow(jobId: string, rowId: string): Promise<void> {
  return send(`/api/jobs/${jobId}/rows/${rowId}`, 'DELETE')
}

export function approveRow(jobId: string, rowId: string, revision: number): Promise<Row> {
  return send(`/api/jobs/${jobId}/rows/${rowId}/approve`, 'POST', { revision })
}

/** Save an appointment's billing lines as edited. Calls no model. */
export function saveBillingLines(jobId: string, rowId: string, payload: BillingLinesPayload): Promise<Row> {
  return send(`/api/jobs/${jobId}/rows/${rowId}/billing`, 'PUT', payload)
}

/** Save the lines shown and mark the appointment confirmed. */
export function confirmBilling(jobId: string, rowId: string, payload: BillingLinesPayload): Promise<Row> {
  return send(`/api/jobs/${jobId}/rows/${rowId}/billing/confirm`, 'POST', payload)
}

export function reopenBilling(jobId: string, rowId: string, baseVersion: number): Promise<Row> {
  return send(`/api/jobs/${jobId}/rows/${rowId}/billing/reopen`, 'POST', { base_version: baseVersion })
}

/** Discard the manual lines (kept in history) and use the model's proposal. */
export function resetBilling(jobId: string, rowId: string, baseVersion: number): Promise<Row> {
  return send(`/api/jobs/${jobId}/rows/${rowId}/billing/reset`, 'POST', { base_version: baseVersion })
}

/** A live crop from the saved review. ``bust`` changes whenever it would. */
export function rowEvidenceUrl(
  jobId: string,
  rowId: string,
  kind: 'row',
  bust: string,
): string {
  return `/api/jobs/${jobId}/rows/${rowId}/evidence/${kind}.png?v=${encodeURIComponent(bust)}`
}

/** Fetch the manifest, hand it to the browser as a download, and return its
 *  summary so the screen can report what was left out. */
export async function downloadManifest(jobId: string): Promise<ManifestSummary> {
  const response = await fetch(`/api/jobs/${jobId}/manifest.json`)
  if (!response.ok) throw new Error(await readError(response))
  const text = await response.text()
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }))
  const link = document.createElement('a')
  link.href = url
  link.download = `rows-${jobId.slice(0, 8)}.json`
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
  return JSON.parse(text) as ManifestSummary
}

/**
 * Read one phase's SSE stream.
 *
 * Deliberately not EventSource: the server closes the stream at the end of a
 * phase, and EventSource would treat that as a dropped connection and
 * reconnect forever, replaying the terminal event each time.
 */
export async function streamEvents(
  jobId: string,
  phase: Phase,
  onEvent: (event: ServerEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const url = `/api/jobs/${jobId}/events?phase=${phase}`
  const response = await fetch(url, { signal })
  if (!response.ok || !response.body) {
    throw new Error(await readError(response))
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let boundary = buffer.indexOf('\n\n')
    while (boundary >= 0) {
      const frame = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      for (const line of frame.split('\n')) {
        if (line.startsWith('data: ')) {
          onEvent(JSON.parse(line.slice(6)) as ServerEvent)
        }
      }
      boundary = buffer.indexOf('\n\n')
    }
  }
}

export function getExtractionConfig(): Promise<ExtractionConfig> {
  return send('/api/extraction', 'GET')
}

/** Save an OpenRouter key on this computer. The reply never contains it. */
export function saveApiKey(key: string): Promise<ExtractionConfig> {
  return send('/api/extraction/key', 'PUT', { key })
}

export function removeApiKey(): Promise<ExtractionConfig> {
  return send('/api/extraction/key', 'DELETE')
}

/**
 * Run one row again. ``transcription``: transcribe (then interpret
 * automatically). ``interpretation``: interpret the current transcription
 * again, without transcribing. Earlier attempts and reviews are kept.
 */
export function extractRow(
  jobId: string,
  rowId: string,
  model: string,
  stage: 'transcription' | 'interpretation' = 'transcription',
): Promise<unknown> {
  return send(`/api/jobs/${jobId}/rows/${rowId}/extract`, 'POST', { model, stage })
}

export interface BatchOutcome {
  prepared: string[]
  transcribing: string[]
  interpreting: string[]
  skipped: Record<string, string>
  interpretation_blocked: { reason: string; message: string } | null
}

/** One click: prepare, transcribe where needed, interpret where needed. */
export function extractRows(jobId: string, model: string): Promise<BatchOutcome> {
  return send(`/api/jobs/${jobId}/extract`, 'POST', { model })
}

/** Accept reviewed values for the row's current extraction result. */
export function acceptExtraction(
  jobId: string,
  rowId: string,
  review: {
    attempt_id: string
    time: string | null
    handwriting_status: HandwritingStatus
    handwriting_text: string | null
  },
): Promise<Row> {
  return send(`/api/jobs/${jobId}/rows/${rowId}/extraction/review`, 'PUT', review)
}

// -- doctor profiles -------------------------------------------------------

/** What a profile editor saves; each save that changes something is a new version. */
export interface DoctorContent {
  display_name: string
  licence_number: string
  group_number: string
  sector_number: string
  instructions: string
  examples: DoctorExample[]
}

export async function listDoctors(includeArchived = false): Promise<DoctorProfileSummary[]> {
  const body = await send<{ profiles: DoctorProfileSummary[] }>(
    `/api/doctors${includeArchived ? '?include_archived=true' : ''}`,
    'GET',
  )
  return body.profiles
}

export function getDoctor(profileId: string): Promise<DoctorProfile> {
  return send(`/api/doctors/${profileId}`, 'GET')
}

export function getDoctorVersion(profileId: string, version: number): Promise<DoctorVersion> {
  return send(`/api/doctors/${profileId}/versions/${version}`, 'GET')
}

export function createDoctor(content: DoctorContent): Promise<DoctorProfile> {
  return send('/api/doctors', 'POST', content)
}

export function saveDoctor(profileId: string, content: DoctorContent, baseVersion: number): Promise<DoctorProfile> {
  return send(`/api/doctors/${profileId}`, 'PUT', { ...content, base_version: baseVersion })
}

export function duplicateDoctor(profileId: string): Promise<DoctorProfile> {
  return send(`/api/doctors/${profileId}/duplicate`, 'POST', {})
}

export function setDoctorArchived(profileId: string, archived: boolean): Promise<DoctorProfile> {
  return send(`/api/doctors/${profileId}/${archived ? 'archive' : 'restore'}`, 'POST', {})
}

/** Assign a document's doctor, or adopt a newer version of the same profile. */
export function assignDoctor(
  jobId: string,
  profileId: string,
  version?: number,
): Promise<{ doctor: DoctorAssignment; changed: boolean }> {
  return send(`/api/jobs/${jobId}/doctor`, 'PUT', { profile_id: profileId, version })
}

export interface BillingCsvProblem {
  row_id: string
  label: string
  reason: string
}

/** Thrown when some appointments cannot be exported; lists each one. */
export class BillingCsvBlocked extends Error {
  constructor(message: string, public problems: BillingCsvProblem[]) {
    super(message)
  }
}

/**
 * Download the billing CSV (one line per proposed act). Built on the server
 * from saved current proposals: no model is called and nothing is approved.
 */
export async function downloadBillingCsv(jobId: string): Promise<{ lines: number; noNotationRows: number }> {
  const response = await fetch(`/api/jobs/${jobId}/billing.csv`)
  if (response.status === 409) {
    const body = await response.json()
    throw new BillingCsvBlocked(String(body.detail), body.problems ?? [])
  }
  if (!response.ok) throw new Error(await readError(response))
  const blob = await response.blob()
  const disposition = response.headers.get('Content-Disposition') ?? ''
  const name = /filename="([^"]+)"/.exec(disposition)?.[1] ?? `billing-${jobId.slice(0, 8)}.csv`
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = name
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
  return {
    lines: Number(response.headers.get('X-Billing-Lines') ?? 0),
    noNotationRows: Number(response.headers.get('X-Billing-No-Notation-Rows') ?? 0),
  }
}
