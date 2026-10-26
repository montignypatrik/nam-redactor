import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  downloadManifest,
  BillingCsvBlocked,
  downloadBillingCsv,
  extractRow,
  extractRows,
  type BillingCsvProblem,
  getExtractionConfig,
  listRows,
  updateRow,
} from '../api'
import type { DoctorAssignment, ExtractionConfig, ManifestSummary, Row, RowPatch } from '../types'
import ApiKeyForm from './ApiKeyForm'
import AppointmentCard from './AppointmentCard'

interface Props {
  jobId: string
  /** The saved review the evidence is rendered from. */
  reviewRevision: number
  /** The document's doctor, whose assigned profile version interprets billing. */
  doctor: DoctorAssignment | null
  onManageDoctors: () => void
  /** Process every row once on arrival (the review was just confirmed). */
  autoProcess: boolean
  onAutoProcessed: () => void
}

/** Remembers the model choice in this browser. */
const MODEL_KEY = 'nam-redactor.extraction-model'
/** Remembers the crop size in this browser. */
const FIT_KEY = 'nam-redactor.fit-crops'

function remembered(key: string): string | null {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

function remember(key: string, value: string) {
  try {
    localStorage.setItem(key, value)
  } catch {
    // only a convenience
  }
}

const byPosition = (a: Row, b: Row) =>
  a.working_page - b.working_page || a.bounds.y1 - b.bounds.y1 || a.bounds.x1 - b.bounds.x1

export default function RowsWorkspace({
  jobId,
  reviewRevision,
  doctor,
  onManageDoctors,
  autoProcess,
  onAutoProcessed,
}: Props) {
  const [rows, setRows] = useState<Row[]>([])
  const [patients, setPatients] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [exported, setExported] = useState<ManifestSummary | null>(null)
  const [config, setConfig] = useState<ExtractionConfig | null>(null)
  const [model, setModel] = useState<string>(() => remembered(MODEL_KEY) ?? '')
  const [batchNote, setBatchNote] = useState<string | null>(null)
  const [blockedNote, setBlockedNote] = useState<string | null>(null)
  const [csvNote, setCsvNote] = useState<{ ok: boolean; text: string; problems: BillingCsvProblem[] } | null>(null)
  const [fitWidth, setFitWidth] = useState(() => remembered(FIT_KEY) !== 'false')

  const reload = useCallback(async () => {
    const body = await listRows(jobId)
    setRows(body.rows)
    setPatients(body.patients)
  }, [jobId])

  // Reload whenever the saved review or the doctor assignment changes:
  // approvals it invalidated, and proposals it outdated, must show.
  const assignmentId = doctor?.assignment_id
  useEffect(() => {
    reload().catch((cause) => setError(cause instanceof Error ? cause.message : 'Could not load rows.'))
  }, [reload, reviewRevision, assignmentId])

  useEffect(() => {
    getExtractionConfig()
      .then((body) => {
        setConfig(body)
        setModel((current) =>
          body.models.some((m) => m.id === current) ? current : body.default_model,
        )
      })
      .catch(() => setConfig(null))
  }, [])

  const chooseModel = (value: string) => {
    setModel(value)
    remember(MODEL_KEY, value)
  }

  // While anything is processing, keep the view current.
  const processing = rows.some(
    (row) => row.extraction.state === 'running' || row.interpretation.state === 'running',
  )
  const wasProcessing = useRef(false)
  useEffect(() => {
    // When a batch finishes, drop its note: each row shows its own outcome.
    if (wasProcessing.current && !processing) setBatchNote(null)
    wasProcessing.current = processing
    if (!processing) return
    const timer = window.setTimeout(() => {
      reload().catch(() => undefined)
    }, 1500)
    return () => window.clearTimeout(timer)
  }, [processing, rows, reload])

  const ordered = useMemo(() => [...rows].sort(byPosition), [rows])
  const labels = useMemo(() => {
    const map: Record<string, string> = {}
    // A section's row is named after its section; hand-drawn rows get R1, R2…
    let manual = 0
    for (const row of ordered) {
      map[row.row_id] = row.origin === 'section' ? `#${row.working_page}` : `R${++manual}`
    }
    return map
  }, [ordered])

  const run = async (action: () => Promise<void>) => {
    setBusy(true)
    setError(null)
    try {
      await action()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'That did not work.')
    } finally {
      setBusy(false)
    }
  }

  const replace = (row: Row) =>
    setRows((previous) => previous.map((existing) => (existing.row_id === row.row_id ? row : existing)))

  const patch = (row: Row, change: RowPatch) =>
    run(async () => replace(await updateRow(jobId, row.row_id, change)))

  const processOne = (row: Row, stage: 'transcription' | 'interpretation' = 'transcription') =>
    run(async () => {
      await extractRow(jobId, row.row_id, model, stage)
      await reload()
    })

  /** Every row that needs it: one call per row (reading and billing together);
   *  rows already read are only re-billed when that is outdated. */
  const processAll = () =>
    run(async () => {
      const outcome = await extractRows(jobId, model)
      const plural = (n: number) => (n === 1 ? '' : 's')
      const skipped = Object.entries(outcome.skipped)
      const current = skipped.filter(([, reason]) => reason === 'already interpreted').length
      const needs = skipped
        .filter(([, reason]) => reason.startsWith('needs'))
        .map(([rowId, reason]) => `${labels[rowId] ?? rowId} ${reason}`)
      const parts = []
      if (outcome.transcribing.length) parts.push(`Processing ${outcome.transcribing.length} row${plural(outcome.transcribing.length)}`)
      if (outcome.interpreting.length) parts.push(`re-billing ${outcome.interpreting.length} row${plural(outcome.interpreting.length)}`)
      if (!parts.length) parts.push('Nothing to run')
      if (current) parts.push(`${current} already done`)
      if (needs.length) parts.push(`not started — ${needs.join('; ')}`)
      const text = parts.join(' · ') + '.'
      setBatchNote(text.charAt(0).toUpperCase() + text.slice(1))
      setBlockedNote(outcome.interpretation_blocked?.message ?? null)
      await reload()
    })

  // Arriving from "Confirm review": process once the model is known.
  const autoStarted = useRef(false)
  useEffect(() => {
    if (!autoProcess || autoStarted.current || config === null) return
    autoStarted.current = true
    onAutoProcessed()
    if (config.configured && model) void processAll()
    // processAll reads the latest state; this runs once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoProcess, config, model])

  /** The practice's billing CSV. If any appointment cannot be exported,
   *  nothing is downloaded and each one is listed. */
  const exportCsv = () =>
    run(async () => {
      try {
        const { lines, noNotationRows } = await downloadBillingCsv(jobId)
        setCsvNote({
          ok: true,
          problems: [],
          text:
            `Exported ${lines} billing line${lines === 1 ? '' : 's'} (one per act); ${confirmedCount} of ${rows.length} appointment${rows.length === 1 ? '' : 's'} confirmed.` +
            (noNotationRows ? ` ${noNotationRows} appointment${noNotationRows === 1 ? '' : 's'} had no billing notation.` : ''),
        })
      } catch (cause) {
        if (cause instanceof BillingCsvBlocked) {
          setCsvNote({ ok: false, text: 'The CSV was not created. Fix these appointments first:', problems: cause.problems })
        } else {
          throw cause
        }
      }
    })

  const exportRows = () =>
    run(async () => {
      setExported(await downloadManifest(jobId))
    })

  const reveal = (row: Row) =>
    document.getElementById(`card-${row.row_id}`)?.scrollIntoView({ block: 'start' })

  const fetchLatest = async (rowId: string) => {
    const body = await listRows(jobId)
    setRows(body.rows)
    return body.rows.find((row) => row.row_id === rowId)
  }

  const confirmedCount = rows.filter((row) => row.billing.state === 'confirmed').length
  const attentionCount = rows.filter(
    (row) => row.billing.state === 'outdated' || (row.billing.source === 'manual' && row.billing.edits_current === false),
  ).length
  const proposedCount = rows.filter((row) => row.interpretation.state === 'proposed').length

  return (
    <div className="rows">
      <div className="rows__bar">
        <p className="muted">
          {rows.length} appointment{rows.length === 1 ? '' : 's'}
        </p>
        <div className="rows__tools">
          <label className="model-select">
            <span>Model</span>
            <select
              value={model}
              disabled={!config || busy}
              onChange={(event) => chooseModel(event.target.value)}
            >
              {config?.models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            onClick={() => void processAll()}
            disabled={busy || !config?.configured || !rows.length}
            title="Reads and bills rows that need it (one call per row); current results are reused"
          >
            Process all rows
          </button>
          <button
            type="button"
            className="primary"
            onClick={() => void exportCsv()}
            disabled={busy || !rows.length}
            title="One line per billing line shown (your edits, else the proposal). Confirmation is not required."
          >
            Export CSV
          </button>
          <button type="button" onClick={() => void exportRows()} disabled={busy || !rows.length}>
            Export rows JSON
          </button>
        </div>
      </div>
      <p className={config?.configured ? 'muted extraction-config' : 'notice notice--warn'}>
        {config === null
          ? 'Checking cloud extraction…'
          : config.configured
            ? `Cloud extraction via OpenRouter · zero-data-retention routes only · billing proposed for ${proposedCount} of ${rows.length} row${rows.length === 1 ? '' : 's'} (unvalidated)` +
              (doctor ? ` · doctor ${doctor.snapshot.display_name}, profile v${doctor.version}` : '') +
              (processing ? ' · working…' : '')
            : 'Cloud extraction is not configured: add an OpenRouter API key below, then Process all rows.'}
      </p>
      {config && <ApiKeyForm config={config} onChanged={setConfig} />}
      {batchNote && <p className="muted">{batchNote}</p>}
      {blockedNote && (
        <div className="notice notice--warn">
          <p>Billing was not proposed: {blockedNote}</p>
          <button type="button" onClick={onManageDoctors}>
            Edit doctor profiles
          </button>
        </div>
      )}

      {csvNote && (
        <div className={csvNote.ok ? 'notice' : 'notice notice--warn'}>
          <p>{csvNote.text}</p>
          {csvNote.problems.length > 0 && (
            <ul>
              {csvNote.problems.map((problem, index) => (
                <li key={`${problem.row_id}-${index}`}>
                  {problem.row_id ? (
                    <button
                      type="button"
                      className="link"
                      onClick={() => {
                        const row = rows.find((r) => r.row_id === problem.row_id)
                        if (row) reveal(row)
                      }}
                    >
                      {labels[problem.row_id] ?? problem.label}
                    </button>
                  ) : (
                    <strong>Document</strong>
                  )}{' '}
                  — {problem.reason}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {exported && (
        <div className={exported.excluded.length ? 'notice notice--warn' : 'notice'}>
          <p>
            Exported {exported.summary.rows_exported} row
            {exported.summary.rows_exported === 1 ? '' : 's'} ready for extraction.
            {exported.excluded.length > 0 &&
              ` ${exported.excluded.length} row${exported.excluded.length === 1 ? ' was' : 's were'} left out:`}
          </p>
          {exported.excluded.length > 0 && (
            <ul>
              {exported.excluded.map((entry) => (
                <li key={entry.row_id}>
                  <button type="button" className="link" onClick={() => {
                    const row = rows.find((r) => r.row_id === entry.row_id)
                    if (row) reveal(row)
                  }}>
                    {labels[entry.row_id] ?? entry.row_id}
                  </button>{' '}
                  (source page {entry.source_page}): {entry.reasons.join('; ')}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {error && <p className="error">{error}</p>}

      <div className="rows__views">
        <p className="rows__progress" aria-live="polite">
          <strong>
            {confirmedCount} of {rows.length} appointment{rows.length === 1 ? '' : 's'} confirmed
          </strong>
          {attentionCount > 0 && <span className="chip chip--warn">{attentionCount} need attention</span>}
        </p>
        <label className="rows__fit">
          <input
            type="checkbox"
            checked={fitWidth}
            onChange={(event) => {
              setFitWidth(event.target.checked)
              remember(FIT_KEY, String(event.target.checked))
            }}
          />
          Fit images to width
        </label>
      </div>

      <div className="cards">
        {ordered.map((row) => (
          <AppointmentCard
            key={row.row_id}
            jobId={jobId}
            row={row}
            label={labels[row.row_id]}
            patients={patients}
            reviewRevision={reviewRevision}
            fitWidth={fitWidth}
            configured={Boolean(config?.configured)}
            doctor={doctor}
            busy={busy}
            onUpdated={replace}
            onPatch={(change) => void patch(row, change)}
            fetchLatest={() => fetchLatest(row.row_id)}
            onInterpret={() => void processOne(row, 'interpretation')}
            onExtract={() => void processOne(row)}
            onManageDoctors={onManageDoctors}
          />
        ))}
        {!rows.length && <p className="muted">No appointments yet.</p>}
      </div>
    </div>
  )
}
