import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  approveRow,
  deleteRow,
  downloadManifest,
  BillingCsvBlocked,
  downloadBillingCsv,
  extractRow,
  extractRows,
  type BillingCsvProblem,
  getExtractionConfig,
  listRows,
  sourceImageUrl,
  rowEvidenceUrl,
  updateRow,
} from '../api'
import type {
  Bounds,
  DoctorAssignment,
  ExtractionConfig,
  FieldName,
  ManifestSummary,
  PageInfo,
  Row,
  RowPatch,
} from '../types'
import ApiKeyForm from './ApiKeyForm'
import AppointmentCard from './AppointmentCard'
import ExtractionPanel, { EXTRACTION_LABELS, extractionChip } from './ExtractionPanel'
import InterpretationPanel from './InterpretationPanel'
import SourcePageCanvas, { sourceGroups, type PlacedBox } from './SourcePageCanvas'

interface Props {
  jobId: string
  pages: PageInfo[]
  /** The saved review the canvas and evidence are rendered from. */
  reviewRevision: number
  /** Unsaved redaction edits exist: evidence would not match, so approving waits. */
  reviewDirty: boolean
  onSaveReview: () => Promise<void>
  /** The document's doctor, whose assigned profile version interprets billing. */
  doctor: DoctorAssignment | null
  onManageDoctors: () => void
}

/** ``select``: clicks select a section's row. ``bounds``: the next drag
 *  redraws a hand-drawn row's region (section rows are their section). */
type DrawMode = 'select' | 'bounds'

const FIELD_LABELS: Record<string, string> = {
  patient_id: 'Patient',
  date: 'Date',
  time: 'Time (optional)',
}

const SOURCE_LABELS: Record<string, string> = {
  manual: 'entered by reviewer',
  page_date: "from the page's date",
  nam_in_row: 'from the NAM in this row',
}

/** Remembers the model choice in this browser. */
const MODEL_KEY = 'nam-redactor.extraction-model'
/** Remembers the view and the crop size in this browser. */
const VIEW_KEY = 'nam-redactor.rows-view'
const FIT_KEY = 'nam-redactor.fit-crops'

type ViewMode = 'cards' | 'layout'

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

const NEEDS: Record<string, string> = { patient_id: 'patient', date: 'date' }

/** The patient dropdown's "No patient" choice (a call with no health number, say). */
const NO_PATIENT = '__no_patient__'

/** Preparation status. A row whose values are known needs no action: it is
 *  made ready automatically when it is extracted. */
function approvalLabel(row: Row): { text: string; className: string } {
  if (row.approval_current) return { text: 'Ready for extraction', className: 'chip chip--ok' }
  if (row.unresolved.length) {
    return { text: `Needs ${row.unresolved.map((f) => NEEDS[f] ?? f).join(' & ')}`, className: 'chip chip--warn' }
  }
  return { text: 'Ready when extracted', className: 'chip chip--ready' }
}

export default function RowsWorkspace({
  jobId,
  pages,
  reviewRevision,
  reviewDirty,
  onSaveReview,
  doctor,
  onManageDoctors,
}: Props) {
  const [rows, setRows] = useState<Row[]>([])
  const [patients, setPatients] = useState<string[]>([])
  // Rows are drawn on original pages; each shows all the sections cut from it.
  const groups = useMemo(() => sourceGroups(pages), [pages])
  const [current, setCurrent] = useState(groups[0]?.sourcePage ?? 1)
  const [outsideHint, setOutsideHint] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [drawMode, setDrawMode] = useState<DrawMode>('select')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [exported, setExported] = useState<ManifestSummary | null>(null)
  const [config, setConfig] = useState<ExtractionConfig | null>(null)
  const [model, setModel] = useState<string>(() => {
    try {
      return localStorage.getItem(MODEL_KEY) ?? ''
    } catch {
      return ''
    }
  })
  const [batchNote, setBatchNote] = useState<string | null>(null)
  const [blockedNote, setBlockedNote] = useState<string | null>(null)
  const [csvNote, setCsvNote] = useState<{ ok: boolean; text: string; problems: BillingCsvProblem[] } | null>(null)
  const [mode, setMode] = useState<ViewMode>(() => (remembered(VIEW_KEY) === 'layout' ? 'layout' : 'cards'))
  const [fitWidth, setFitWidth] = useState(() => remembered(FIT_KEY) !== 'false')
  const layoutRef = useRef<HTMLDivElement | null>(null)

  const chooseMode = (value: ViewMode) => {
    setMode(value)
    remember(VIEW_KEY, value)
  }

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
    try {
      localStorage.setItem(MODEL_KEY, value)
    } catch {
      // remembering the choice is only a convenience
    }
  }

  // While anything is extracting, keep the view current.
  const extracting = rows.some(
    (row) => row.extraction.state === 'running' || row.interpretation.state === 'running',
  )
  const wasExtracting = useRef(false)
  useEffect(() => {
    // When a batch finishes, drop its note: each row shows its own outcome.
    if (wasExtracting.current && !extracting) setBatchNote(null)
    wasExtracting.current = extracting
    if (!extracting) return
    const timer = window.setTimeout(() => {
      reload().catch(() => undefined)
    }, 1500)
    return () => window.clearTimeout(timer)
  }, [extracting, rows, reload])

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

  const selected = rows.find((row) => row.row_id === selectedId) ?? null
  const group = groups.find((g) => g.sourcePage === current) ?? groups[0]
  const zones = useMemo(() => group?.zones ?? [], [group])

  // Canvas items per section: each row's region -- the whole row is the evidence.
  const itemsByWorking = useMemo(() => {
    const out: Record<number, { rowId: string; box: Bounds }[]> = {}
    for (const zone of zones) out[zone.workingPage] = []
    for (const row of ordered) out[row.working_page]?.push({ rowId: row.row_id, box: row.bounds })
    return out
  }, [ordered, zones])

  const boxesByWorking = useMemo(() => {
    const out: Record<number, (Bounds & { confidence: number })[]> = {}
    for (const [page, items] of Object.entries(itemsByWorking)) {
      out[Number(page)] = items.map((item) => ({ ...item.box, confidence: 1 }))
    }
    return out
  }, [itemsByWorking])

  const itemAt = (placed: PlacedBox) => itemsByWorking[placed.workingPage]?.[placed.index]

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

  /** Only a hand-drawn row's region can be redrawn; any other drag is ignored. */
  const handleDraw = (_workingPage: number, rect: Bounds) => {
    if (drawMode !== 'bounds' || !selected) return
    void run(async () => {
      setOutsideHint(false)
      replace(await updateRow(jobId, selected.row_id, { bounds: rect }))
      setDrawMode('select')
    })
  }

  // Keep a row selected: the first one on the page being shown.
  useEffect(() => {
    if (selected && zones.some((zone) => zone.workingPage === selected.working_page)) return
    const first = ordered.find((row) => zones.some((zone) => zone.workingPage === row.working_page))
    setSelectedId(first?.row_id ?? null)
  }, [ordered, zones, selected])

  const removeRow = (row: Row) =>
    run(async () => {
      await deleteRow(jobId, row.row_id)
      setRows((previous) => previous.filter((existing) => existing.row_id !== row.row_id))
      setSelectedId(null)
    })

  const handleRemove = (workingPage: number, index: number) => {
    const item = itemsByWorking[workingPage]?.[index]
    const row = rows.find((r) => r.row_id === item?.rowId)
    if (!item || !row || row.origin === 'section') return
    if (window.confirm(`Delete row ${labels[row.row_id]}?`)) void removeRow(row)
  }

  const approve = (row: Row) =>
    run(async () => {
      if (reviewDirty) await onSaveReview()
      replace(await approveRow(jobId, row.row_id, row.revision))
    })

  // Extraction uses evidence rendered from the saved review: save first.
  const extractOne = (row: Row, stage: 'transcription' | 'interpretation' = 'transcription') =>
    run(async () => {
      if (reviewDirty) await onSaveReview()
      await extractRow(jobId, row.row_id, model, stage)
      await reload()
    })

  /** One click: prepare, transcribe where needed, then interpret. */
  const extractAll = () =>
    run(async () => {
      if (reviewDirty) await onSaveReview()
      const outcome = await extractRows(jobId, model)
      const plural = (n: number) => (n === 1 ? '' : 's')
      const skipped = Object.entries(outcome.skipped)
      const current = skipped.filter(([, reason]) => reason === 'already interpreted').length
      const needs = skipped
        .filter(([, reason]) => reason.startsWith('needs'))
        .map(([rowId, reason]) => `${labels[rowId] ?? rowId} ${reason}`)
      const parts = []
      if (outcome.transcribing.length) parts.push(`Transcribing ${outcome.transcribing.length} row${plural(outcome.transcribing.length)}, then interpreting`)
      if (outcome.interpreting.length) parts.push(`interpreting ${outcome.interpreting.length} row${plural(outcome.interpreting.length)}`)
      if (!parts.length) parts.push('Nothing to run')
      if (current) parts.push(`${current} already interpreted`)
      if (needs.length) parts.push(`not started — ${needs.join('; ')}`)
      const text = parts.join(' · ') + '.'
      setBatchNote(text.charAt(0).toUpperCase() + text.slice(1))
      setBlockedNote(outcome.interpretation_blocked?.message ?? null)
      await reload()
    })

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

  /** Select the row on its page (the page layout view). */
  const goTo = (row: Row) => {
    setCurrent(row.source_page)
    setSelectedId(row.row_id)
  }

  /** Show a row where the person is working: its card, or its page. */
  const reveal = (row: Row) => {
    if (mode === 'cards') {
      document.getElementById(`card-${row.row_id}`)?.scrollIntoView({ block: 'start' })
    } else {
      goTo(row)
    }
  }

  /** Patient, date and time are fixed on the page layout. */
  const editDetails = (row: Row) => {
    goTo(row)
    chooseMode('layout')
    window.requestAnimationFrame(() => layoutRef.current?.scrollIntoView({ block: 'start' }))
  }

  const fetchLatest = async (rowId: string) => {
    const body = await listRows(jobId)
    setRows(body.rows)
    return body.rows.find((row) => row.row_id === rowId)
  }

  const confirmedCount = rows.filter((row) => row.billing.state === 'confirmed').length
  const attentionCount = rows.filter(
    (row) => row.billing.state === 'outdated' || (row.billing.source === 'manual' && row.billing.edits_current === false),
  ).length

  const approvedCount = rows.filter((row) => row.approval_current).length
  const proposedCount = rows.filter((row) => row.interpretation.state === 'proposed').length
  const bust = selected ? `${selected.revision}-${reviewRevision}` : ''

  const fieldEditor = (row: Row, name: FieldName, input: React.ReactNode) => {
    const field = row.fields[name]
    // A blank time is expected: it will be read from the row evidence.
    const toExtract = name === 'time' && field.status === 'missing'
    const status = toExtract ? 'extract' : field.status
    return (
      <div className={`field field--${status}`}>
        <div className="field__head">
          <span className="field__label">{FIELD_LABELS[name]}</span>
          <span className={`chip chip--${status}`}>{toExtract ? 'To be extracted' : field.status}</span>
        </div>
        {input}
        <div className="field__foot">
          <span className="muted">
            {toExtract
              ? 'Leave blank to extract it from the row; type it only to override.'
              : field.source
                ? SOURCE_LABELS[field.source] ?? field.source
                : 'not set'}
          </span>
          {field.status === 'suggested' && (
            <button type="button" disabled={busy} onClick={() => void patch(row, { confirm: [name] })}>
              Confirm
            </button>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="rows">
      <div className="rows__bar">
        <p className="muted">
          {rows.length} row{rows.length === 1 ? '' : 's'} · {approvedCount} ready for extraction
          {reviewDirty && ' · unsaved redaction edits: save before marking rows ready'}
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
            onClick={() => void extractAll()}
            disabled={busy || !config?.configured || !rows.length}
            title="Transcribes rows that need it, then proposes billing with the assigned doctor profile; current results are reused"
          >
            Extract all rows
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
              (extracting ? ' · working…' : '')
            : 'Cloud extraction is not configured: add an OpenRouter API key below.'}
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
        <div className="segmented" role="tablist" aria-label="Rows view">
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'cards'}
            className={mode === 'cards' ? 'segmented__on' : undefined}
            onClick={() => chooseMode('cards')}
          >
            Appointments
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'layout'}
            className={mode === 'layout' ? 'segmented__on' : undefined}
            onClick={() => chooseMode('layout')}
          >
            Page layout
          </button>
        </div>
        <p className="rows__progress" aria-live="polite">
          <strong>
            {confirmedCount} of {rows.length} appointment{rows.length === 1 ? '' : 's'} confirmed
          </strong>
          {attentionCount > 0 && <span className="chip chip--warn">{attentionCount} need attention</span>}
        </p>
        {mode === 'cards' && (
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
        )}
      </div>

      {mode === 'cards' && (
        <div className="cards">
          {ordered.map((row) => (
            <AppointmentCard
              key={row.row_id}
              jobId={jobId}
              row={row}
              label={labels[row.row_id]}
              reviewRevision={reviewRevision}
              fitWidth={fitWidth}
              configured={Boolean(config?.configured)}
              doctor={doctor}
              busy={busy}
              onUpdated={replace}
              fetchLatest={() => fetchLatest(row.row_id)}
              onEditDetails={() => editDetails(row)}
              onInterpret={() => void extractOne(row, 'interpretation')}
              onExtract={() => void extractOne(row)}
              onManageDoctors={onManageDoctors}
            />
          ))}
          {!rows.length && <p className="muted">No appointments yet.</p>}
        </div>
      )}

      {mode === 'layout' && (
      <div className="rows__body" ref={layoutRef}>
        <nav className="page-nav" aria-label="Original pages">
          {groups.map(({ sourcePage, zones: pageZones }) => {
            const inPage = new Set(pageZones.map((zone) => zone.workingPage))
            const count = rows.filter((row) => inPage.has(row.working_page)).length
            return (
              <button
                key={sourcePage}
                type="button"
                className={sourcePage === current ? 'page-nav__item page-nav__item--on' : 'page-nav__item'}
                onClick={() => {
                  setCurrent(sourcePage)
                  setDrawMode('select')
                  setOutsideHint(false)
                }}
              >
                <span>
                  Page {sourcePage}
                  <span className="page-nav__date">
                    {pageZones.length} section{pageZones.length === 1 ? '' : 's'}
                  </span>
                </span>
                <span className={count ? 'found' : 'muted'}>{count}</span>
              </button>
            )
          })}
        </nav>

        <div className="stage">
          <p className="rows__mode">
            {drawMode === 'select' &&
              'Each section is one appointment, and the whole section is its evidence. Click a section to select it.'}
            {drawMode === 'bounds' && `Drag to redraw the region of ${labels[selectedId ?? '']}.`}
            {drawMode === 'bounds' && (
              <button type="button" className="link" onClick={() => setDrawMode('select')}>
                Cancel
              </button>
            )}
          </p>
          {outsideHint && (
            <p className="draw-hint">
              {`Draw inside the section that holds ${labels[selectedId ?? '']}.`}
            </p>
          )}
          {group && (
            <SourcePageCanvas
              imageUrl={sourceImageUrl(jobId, current, reviewRevision)}
              zones={zones}
              boxesByWorking={boxesByWorking}
              variant="row"
              tagFor={(placed) => {
                const item = itemAt(placed)
                if (!item) return undefined
                // A section's row already carries the section's own label.
                const row = rows.find((r) => r.row_id === item.rowId)
                return row?.origin === 'section' ? undefined : labels[item.rowId]
              }}
              classFor={(placed) => (itemAt(placed)?.rowId === selectedId ? 'box--active' : '')}
              onDraw={handleDraw}
              // A redrawn row stays in its row's section.
              drawWithin={drawMode === 'bounds' && selected ? selected.working_page : undefined}
              drawThrough
              onRejected={() => setOutsideHint(true)}
              onSectionClick={(workingPage) => {
                setOutsideHint(false)
                const row = rows.find((r) => r.origin === 'section' && r.working_page === workingPage)
                if (row) setSelectedId(row.row_id)
              }}
              onRemove={handleRemove}
              onSelect={(placed) => {
                const item = placed && itemAt(placed)
                if (item) setSelectedId(item.rowId)
              }}
            />
          )}
        </div>

        <aside className="row-panel">
          <ol className="row-list">
            {ordered.map((row) => {
              const status = approvalLabel(row)
              return (
                <li key={row.row_id}>
                  <button
                    type="button"
                    className={row.row_id === selectedId ? 'row-list__item row-list__item--on' : 'row-list__item'}
                    onClick={() => goTo(row)}
                  >
                    <span>
                      <strong>{labels[row.row_id]}</strong> · p{row.working_page}
                      <span className="muted">
                        {' '}
                        {row.fields.patient_id.value ?? '—'} · {row.fields.date.value ?? '—'}
                        {row.fields.time.value ? ` ${row.fields.time.value}` : ''}
                      </span>
                    </span>
                    <span className="row-list__chips">
                      <span className={status.className}>{status.text}</span>
                      {row.extraction.state !== 'not_ready' && (
                        <span className={extractionChip(row.extraction.state)}>
                          {EXTRACTION_LABELS[row.extraction.state]}
                        </span>
                      )}
                    </span>
                  </button>
                </li>
              )
            })}
            {!rows.length && <li className="muted">No rows yet.</li>}
          </ol>

          {selected && (
            <section className="row-editor" aria-label={`Row ${labels[selected.row_id]}`}>
              <header className="row-editor__head">
                <h2>
                  Row {labels[selected.row_id]}
                  <span className="muted">
                    {' '}
                    · page {selected.working_page} (source page {selected.source_page}) · rev {selected.revision}
                  </span>
                </h2>
                <span className={approvalLabel(selected).className}>{approvalLabel(selected).text}</span>
              </header>

              {selected.approval.status === 'invalidated' && !selected.approval_current && (
                <p className="notice notice--warn">
                  Readiness invalidated: {selected.approval.invalidated_reasons.join('; ') || 'changed'}.
                </p>
              )}

              <div className="evidence">
                <figure>
                  <img src={rowEvidenceUrl(jobId, selected.row_id, 'row', bust)} alt="Row evidence, redacted" />
                  <figcaption>The whole row, redacted: the evidence for extracting its time and notes</figcaption>
                </figure>
              </div>

              {fieldEditor(
                selected,
                'patient_id',
                <select
                  value={
                    selected.fields.patient_id.value ??
                    (selected.fields.patient_id.status === 'confirmed' ? NO_PATIENT : '')
                  }
                  disabled={busy}
                  onChange={(event) => {
                    const value = event.target.value
                    void patch(selected, value === NO_PATIENT ? { no_patient: true } : { patient_id: value || null })
                  }}
                >
                  <option value="">— choose a patient —</option>
                  <option value={NO_PATIENT}>No patient</option>
                  {patients.map((patient) => (
                    <option key={patient} value={patient}>
                      {patient}
                    </option>
                  ))}
                </select>,
              )}
              {fieldEditor(
                selected,
                'date',
                <input
                  type="date"
                  value={selected.fields.date.value ?? ''}
                  disabled={busy}
                  onChange={(event) => void patch(selected, { date: event.target.value || null })}
                />,
              )}
              {fieldEditor(
                selected,
                'time',
                <input
                  type="time"
                  value={selected.fields.time.value ?? ''}
                  disabled={busy}
                  onChange={(event) => void patch(selected, { time: event.target.value || null })}
                />,
              )}

              {selected.origin !== 'section' && (
                <div className="row-editor__tools">
                    <button type="button" onClick={() => setDrawMode('bounds')} disabled={busy}>
                      Redraw row region
                    </button>
                    <button
                      type="button"
                      className="danger"
                      disabled={busy}
                      onClick={() => {
                        if (window.confirm(`Delete row ${labels[selected.row_id]}?`)) void removeRow(selected)
                      }}
                    >
                      Delete row
                    </button>
                </div>
              )}

              {selected.unresolved.length > 0 ? (
                <p className="row-editor__todo">
                  To confirm: {selected.unresolved.map((name) => FIELD_LABELS[name] ?? name).join(', ')}
                </p>
              ) : (
                !selected.approval_current && <p className="muted">Patient and date confirmed.</p>
              )}
              <button
                type="button"
                className="primary"
                disabled={busy || selected.unresolved.length > 0 || selected.approval_current}
                onClick={() => void approve(selected)}
                title={reviewDirty ? 'Saves the redaction review first' : undefined}
              >
                {selected.approval_current
                  ? 'Ready for extraction'
                  : reviewDirty
                    ? 'Save review & mark ready for extraction'
                    : 'Mark ready for extraction'}
              </button>
              <p className="muted row-editor__note">
                Extracting marks the row ready automatically when its patient and date are known;
                use this button only to mark it by hand. Readiness is separate from reviewing the
                extraction below, and does not establish billing eligibility.
              </p>

              <InterpretationPanel
                row={selected}
                doctor={doctor}
                configured={Boolean(config?.configured)}
                busy={busy}
                onInterpret={() => void extractOne(selected, 'interpretation')}
                onManageDoctors={onManageDoctors}
              />

              <ExtractionPanel
                jobId={jobId}
                row={selected}
                configured={Boolean(config?.configured)}
                busy={busy}
                onExtract={() => void extractOne(selected)}
                onUpdated={replace}
              />
            </section>
          )}
        </aside>
      </div>
      )}
    </div>
  )
}
