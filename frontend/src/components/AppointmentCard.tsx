import { useEffect, useRef, useState } from 'react'
import { confirmBilling, reopenBilling, resetBilling, rowEvidenceUrl, saveBillingLines } from '../api'
import { type BillingColumn, type BillingLine, type DoctorAssignment, type Row } from '../types'
import ExtractionPanel from './ExtractionPanel'
import InterpretationPanel from './InterpretationPanel'

interface Props {
  jobId: string
  row: Row
  label: string
  reviewRevision: number
  /** Show crops scaled to the card's width instead of at full size. */
  fitWidth: boolean
  configured: boolean
  doctor: DoctorAssignment | null
  /** Something document-wide is running (saving the review, a batch). */
  busy: boolean
  onUpdated: (row: Row) => void
  /** Fetch the row as saved now (after a conflict). */
  fetchLatest: () => Promise<Row | undefined>
  onEditDetails: () => void
  onInterpret: () => void
  onExtract: () => void
  onManageDoctors: () => void
}

/** A line as typed. Units and contexts stay text until saved, so a
 *  half-typed value is never reformatted under the cursor. */
interface DraftLine {
  key: string
  code: string
  units: string
  contexts: string
  model_index: number | null
}

interface Draft {
  lines: DraftLine[]
  noBilling: boolean
  /** The proposal and billing version this draft was started from. */
  attemptId: string | null
  version: number
  /** Changed since the last successful save. */
  dirty: boolean
}

const SAVE_DELAY = 700
const EDITABLE: BillingColumn[] = ['ProcedureCode', 'Units', 'ContextualElements']
/** The columns shown on the card. The CSV still has all 17 (the others are
 *  always empty); these are the ones the person needs to see. */
const SHOWN: BillingColumn[] = [
  'doctorLicenceID',
  'doctorGroupNumber',
  'ClaimDate',
  'ProcedureTime',
  'HIN',
  'ProcedureCode',
  'Units',
  'ContextualElements',
]

const unitsText = (units: number | null) => (units === null ? '' : String(units))

const toDraftLines = (lines: BillingLine[]): DraftLine[] =>
  lines.map((line, index) => ({
    key: `l${index}`,
    code: line.code,
    units: unitsText(line.units),
    contexts: line.context_elements.join(','),
    model_index: line.model_index,
  }))

/** ``null`` for blank, a number, or ``undefined`` when it is not a number. */
function parseUnits(text: string): number | null | undefined {
  const value = text.trim().replace(',', '.')
  if (!value) return null
  if (!/^\d+(\.\d+)?$/.test(value)) return undefined
  return Number(value)
}

/** Comma-separated; spaces around each value go, empty values are dropped. */
const parseContexts = (text: string) =>
  text
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean)

const INTERPRETATION_NOTES: Partial<Record<Row['interpretation']['state'], string>> = {
  not_ready: 'Not transcribed yet.',
  pending: 'Billing not proposed yet.',
  running: 'Processing…',
  failed: 'The interpretation failed; you can retry it under Details, or enter the codes yourself.',
  stale: 'The proposal is outdated; Extract all rows will redo it.',
  blocked: 'Billing cannot be proposed for this document.',
}

/**
 * One appointment: its redacted crop, its billing lines in the practice's
 * CSV columns, and its billing confirmation. Edits are saved as the person
 * types, beside the model's proposal, and call no model.
 */
export default function AppointmentCard({
  jobId,
  row,
  label,
  reviewRevision,
  fitWidth,
  configured,
  doctor,
  busy,
  onUpdated,
  fetchLatest,
  onEditDetails,
  onInterpret,
  onExtract,
  onManageDoctors,
}: Props) {
  const view = row.billing
  const [draft, setDraft] = useState<Draft | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [conflict, setConflict] = useState(false)
  const [detailsOpen, setDetailsOpen] = useState(false)
  const draftRef = useRef<Draft | null>(null)
  draftRef.current = draft
  const timer = useRef<number | null>(null)
  const focusKey = useRef<string | null>(null)
  const nextKey = useRef(0)
  const inputs = useRef(new Map<string, HTMLInputElement>())

  // A clean draft follows the server when the billing changed elsewhere
  // (another tab, a reset). A dirty one is never replaced: polling cannot
  // overwrite what is being typed.
  useEffect(() => {
    if (draft && !draft.dirty && view.version > draft.version) setDraft(null)
  }, [draft, view.version])

  useEffect(() => () => {
    if (timer.current !== null) window.clearTimeout(timer.current)
  }, [])

  useEffect(() => {
    if (!focusKey.current) return
    inputs.current.get(focusKey.current)?.focus({ preventScroll: true })
    focusKey.current = null
  })

  const confirmed = view.state === 'confirmed'
  const running = view.interpretation_state === 'running' || row.extraction.state === 'running'
  const locked = confirmed || running || busy
  const shownLines: DraftLine[] = draft ? draft.lines : toDraftLines(view.lines)
  const noBilling = draft ? draft.noBilling : view.no_billing
  const invalidUnits = shownLines.some((line) => parseUnits(line.units) === undefined)

  const payload = (d: Draft) => ({
    acts: d.lines.map((line) => ({
      code: line.code.trim(),
      units: parseUnits(line.units) ?? null,
      context_elements: parseContexts(line.contexts),
      model_index: line.model_index,
    })),
    no_billing: d.noBilling,
    base_version: d.version,
    attempt_id: d.attemptId,
  })

  const failed = (cause: unknown) => {
    const message = cause instanceof Error ? cause.message : 'That did not work.'
    setError(message)
    setConflict(/changed since it was loaded|proposal changed/i.test(message))
  }

  const save = async () => {
    timer.current = null
    const current = draftRef.current
    if (!current || !current.dirty) return
    if (current.lines.some((line) => parseUnits(line.units) === undefined)) return
    const sent = current
    setSaving(true)
    try {
      const updated = await saveBillingLines(jobId, row.row_id, payload(sent))
      setError(null)
      setConflict(false)
      setDraft((d) =>
        d
          ? {
              ...d,
              version: updated.billing.version,
              attemptId: updated.billing.attempt_id,
              // Still dirty if the person kept typing while this was saving.
              dirty: d !== sent,
            }
          : d,
      )
      onUpdated(updated)
    } catch (cause) {
      failed(cause)
    } finally {
      setSaving(false)
    }
  }

  // Reschedule whenever a dirty draft changes.
  useEffect(() => {
    if (!draft?.dirty || saving || conflict) return
    if (timer.current !== null) window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => void save(), SAVE_DELAY)
    return () => {
      if (timer.current !== null) window.clearTimeout(timer.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft, saving, conflict])

  const change = (update: (d: Draft) => Draft) =>
    setDraft((d) => {
      const base: Draft = d ?? {
        lines: toDraftLines(view.lines),
        noBilling: view.no_billing,
        attemptId: view.attempt_id,
        version: view.version,
        dirty: false,
      }
      return { ...update(base), dirty: true }
    })

  const setLine = (key: string, field: 'code' | 'units' | 'contexts', value: string) =>
    change((d) => ({ ...d, lines: d.lines.map((line) => (line.key === key ? { ...line, [field]: value } : line)) }))

  const addLine = () => {
    const key = `n${nextKey.current++}`
    focusKey.current = `${key}:code`
    change((d) => ({
      ...d,
      noBilling: false,
      lines: [...d.lines, { key, code: '', units: '', contexts: '', model_index: null }],
    }))
  }

  const removeLine = (key: string) => change((d) => ({ ...d, lines: d.lines.filter((line) => line.key !== key) }))

  const act = async (action: () => Promise<Row>) => {
    if (timer.current !== null) window.clearTimeout(timer.current)
    setSaving(true)
    try {
      const updated = await action()
      setDraft(null)
      setError(null)
      setConflict(false)
      onUpdated(updated)
    } catch (cause) {
      failed(cause)
      // A refused confirmation still saved the lines: show them as saved.
      if (cause instanceof Error && cause.message.startsWith('Saved')) {
        setDraft(null)
        const latest = await fetchLatest().catch(() => undefined)
        if (latest) onUpdated(latest)
      }
    } finally {
      setSaving(false)
    }
  }

  const confirm = () => {
    if (invalidUnits) {
      setError('Units must be a number, or left empty.')
      return
    }
    const d: Draft = draft ?? {
      lines: toDraftLines(view.lines),
      noBilling: view.no_billing,
      attemptId: view.attempt_id,
      version: view.version,
      dirty: false,
    }
    void act(() => confirmBilling(jobId, row.row_id, payload(d)))
  }

  /** Save the lines shown against what the row is now based on. */
  const keepLines = () =>
    void act(async () => {
      const latest = (await fetchLatest()) ?? row
      const d: Draft = draft ?? {
        lines: toDraftLines(view.lines),
        noBilling: view.no_billing,
        attemptId: null,
        version: 0,
        dirty: true,
      }
      return saveBillingLines(jobId, row.row_id, {
        ...payload(d),
        base_version: latest.billing.version,
        attempt_id: latest.billing.attempt_id,
      })
    })

  const discard = () =>
    void act(async () => {
      const latest = (await fetchLatest()) ?? row
      if (latest.billing.source !== 'manual') return latest
      return resetBilling(jobId, row.row_id, latest.billing.version)
    })

  const reopen = () => void act(() => reopenBilling(jobId, row.row_id, view.version))

  const cell = (column: BillingColumn) => view.shared[column as keyof typeof view.shared] ?? ''

  const status =
    view.state === 'confirmed'
      ? { text: 'Confirmed', className: 'chip chip--ok' }
      : view.state === 'outdated' || (view.source === 'manual' && view.edits_current === false)
        ? { text: 'Needs attention', className: 'chip chip--warn' }
        : { text: 'Draft', className: 'chip' }
  const note = view.source === 'manual' ? null : INTERPRETATION_NOTES[view.interpretation_state]
  const outdated = view.source === 'manual' && view.edits_current === false
  const time = view.shared.ProcedureTime

  return (
    <article
      id={`card-${row.row_id}`}
      className={`card card--${view.state}${outdated ? ' card--attention' : ''}`}
      aria-labelledby={`card-${row.row_id}-title`}
    >
      <header className="card__head">
        <h3 id={`card-${row.row_id}-title`}>
          Row {label}
          <span className="muted card__meta">
            {' '}
            · {row.fields.patient_id.value ?? (row.fields.patient_id.status === 'confirmed' ? 'no patient' : 'patient not set')} · {row.fields.date.value ?? 'no date'}
            {time ? ` · ${time}` : ''}
          </span>
        </h3>
        <span className="card__chips">
          {view.warnings > 0 && (
            <span className="chip chip--issue" title="The model reported conflicts or unresolved notes: see Details">
              ⚠ {view.warnings} warning{view.warnings === 1 ? '' : 's'}
            </span>
          )}
          {view.source === 'manual' && !outdated && <span className="chip">Edited</span>}
          <span className={status.className}>{status.text}</span>
        </span>
      </header>

      <div className={fitWidth ? 'card__crop card__crop--fit' : 'card__crop'}>
        <img
          src={rowEvidenceUrl(jobId, row.row_id, 'row', `${row.revision}-${reviewRevision}`)}
          alt={`Row ${label}, redacted`}
          loading="lazy"
        />
      </div>

      {outdated && (
        <div className="notice notice--warn card__notice">
          <p>
            {view.state === 'outdated' ? 'This confirmation is outdated' : 'Your edits are outdated'}: the
            transcription, evidence, doctor profile or proposal changed since. Your lines are kept below; they are not
            exported until you keep them or switch to the new proposal.
          </p>
          <div className="card__actions">
            <button type="button" onClick={keepLines} disabled={saving || running}>
              Keep my lines
            </button>
            <button
              type="button"
              onClick={discard}
              disabled={saving || running || view.interpretation_state !== 'proposed'}
              title={view.interpretation_state !== 'proposed' ? 'There is no current proposal yet' : undefined}
            >
              Use the new proposal
            </button>
          </div>
        </div>
      )}

      {note && <p className="muted card__note">{note}</p>}

      <div className="billing-grid" role="region" aria-label={`Billing lines for row ${label}`} tabIndex={0}>
        <table>
          <thead>
            <tr>
              <th scope="col" className="billing-grid__n">
                #
              </th>
              {SHOWN.map((column) => (
                <th
                  key={column}
                  scope="col"
                  className={EDITABLE.includes(column) ? 'billing-grid__edit' : undefined}
                >
                  {column}
                </th>
              ))}
              <th scope="col" className="billing-grid__end">
                <span className="sr-only">Remove</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {shownLines.map((line, index) => {
              const n = index + 1
              const badUnits = parseUnits(line.units) === undefined
              return (
                <tr key={line.key}>
                  <td className="billing-grid__n">{n}</td>
                  {SHOWN.map((column) => {
                    if (column === 'ProcedureCode')
                      return (
                        <td key={column} className="billing-grid__edit">
                          <input
                            ref={(el) => {
                              if (el) inputs.current.set(`${line.key}:code`, el)
                              else inputs.current.delete(`${line.key}:code`)
                            }}
                            value={line.code}
                            onChange={(event) => setLine(line.key, 'code', event.target.value)}
                            aria-label={`ProcedureCode, line ${n}`}
                            aria-invalid={!line.code.trim() || undefined}
                            className="billing-grid__code"
                            spellCheck={false}
                            autoCapitalize="off"
                            autoComplete="off"
                            readOnly={locked}
                            maxLength={32}
                          />
                        </td>
                      )
                    if (column === 'Units')
                      return (
                        <td key={column} className="billing-grid__edit">
                          <input
                            value={line.units}
                            onChange={(event) => setLine(line.key, 'units', event.target.value)}
                            aria-label={`Units, line ${n}`}
                            aria-invalid={badUnits || undefined}
                            inputMode="decimal"
                            className="billing-grid__units"
                            autoComplete="off"
                            readOnly={locked}
                          />
                        </td>
                      )
                    if (column === 'ContextualElements')
                      return (
                        <td key={column} className="billing-grid__edit">
                          <input
                            value={line.contexts}
                            onChange={(event) => setLine(line.key, 'contexts', event.target.value)}
                            aria-label={`ContextualElements, line ${n} (comma-separated)`}
                            className="billing-grid__contexts"
                            spellCheck={false}
                            autoComplete="off"
                            readOnly={locked}
                          />
                        </td>
                      )
                    return (
                      <td key={column} className="billing-grid__shared">
                        {cell(column)}
                      </td>
                    )
                  })}
                  <td className="billing-grid__end">
                    {!locked && (
                      <button
                        type="button"
                        className="billing-grid__remove"
                        onClick={() => removeLine(line.key)}
                        aria-label={`Remove line ${n}`}
                        title="Remove this line"
                      >
                        ×
                      </button>
                    )}
                  </td>
                </tr>
              )
            })}
            {!shownLines.length && (
              <tr>
                <td colSpan={SHOWN.length + 2} className="muted billing-grid__empty">
                  {noBilling ? 'No billing acts for this appointment.' : 'No billing lines.'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {invalidUnits && <p className="error card__error">Units must be a number, or left empty.</p>}
      {error && (
        <div className="card__error">
          <p className="error">{error}</p>
          {conflict && (
            <div className="card__actions">
              <button type="button" onClick={keepLines} disabled={saving}>
                Keep my lines
              </button>
              <button
                type="button"
                onClick={() => {
                  setDraft(null)
                  setError(null)
                  setConflict(false)
                  void fetchLatest().then((latest) => latest && onUpdated(latest))
                }}
                disabled={saving}
              >
                Discard my changes
              </button>
            </div>
          )}
        </div>
      )}
      {view.problems.length > 0 && !outdated && (
        <p className="card__problems">
          Not exportable yet: {view.problems.join('; ')}.{' '}
          {view.problems.some((p) => /patient|health number|date/.test(p)) && (
            <button type="button" className="link" onClick={onEditDetails}>
              Fix patient or date
            </button>
          )}
        </p>
      )}

      <div className="card__actions card__foot">
        {confirmed ? (
          <>
            <span className="muted">
              Confirmed {view.confirmation ? new Date(view.confirmation.at).toLocaleString() : ''}
            </span>
            <button type="button" onClick={reopen} disabled={saving || busy}>
              Reopen for editing
            </button>
          </>
        ) : (
          <>
            <button type="button" onClick={addLine} disabled={locked || saving}>
              + Add code
            </button>
            <label className="card__nobilling">
              <input
                type="checkbox"
                checked={noBilling}
                disabled={locked || saving || shownLines.length > 0}
                onChange={(event) => change((d) => ({ ...d, noBilling: event.target.checked }))}
              />
              No billing acts
            </label>
            <span className="card__spacer" />
            <span className="muted card__saved" aria-live="polite">
              {saving ? 'Saving…' : draft?.dirty ? 'Unsaved changes' : view.source === 'manual' ? 'Saved' : ''}
            </span>
            {view.source === 'manual' && !outdated && (
              <button type="button" className="link" onClick={discard} disabled={saving || running}>
                Discard my edits
              </button>
            )}
            <button
              type="button"
              className="primary"
              onClick={confirm}
              disabled={locked || saving || outdated}
              title="Save these lines and mark the appointment done"
            >
              Confirm row
            </button>
          </>
        )}
      </div>

      <details className="card__details" onToggle={(event) => setDetailsOpen(event.currentTarget.open)}>
        <summary>Details: transcription, model reasoning, warnings, provenance</summary>
        {detailsOpen && (
          <div className="card__details-body">
            <InterpretationPanel
              row={row}
              doctor={doctor}
              configured={configured}
              busy={busy || saving}
              onInterpret={onInterpret}
              onManageDoctors={onManageDoctors}
            />
            <ExtractionPanel
              jobId={jobId}
              row={row}
              configured={configured}
              busy={busy || saving}
              onExtract={onExtract}
              onUpdated={onUpdated}
            />
            <p className="muted">
              Billing history: {view.edit_count} saved edit{view.edit_count === 1 ? '' : 's'},{' '}
              {view.confirmation_count} confirmation{view.confirmation_count === 1 ? '' : 's'}. The model&apos;s original
              proposal is kept unchanged above.
            </p>
          </div>
        )}
      </details>
    </article>
  )
}
