import { useEffect, useState } from 'react'
import { acceptExtraction } from '../api'
import type { ExtractionState, ExtractionView, HandwritingStatus, Row } from '../types'

interface Props {
  jobId: string
  row: Row
  configured: boolean
  busy: boolean
  onExtract: () => void
  onUpdated: (row: Row) => void
}

export const EXTRACTION_LABELS: Record<ExtractionState, string> = {
  not_ready: 'Not ready',
  pending: 'Not extracted',
  running: 'Extracting…',
  failed: 'Extraction failed',
  stale: 'Stale — re-extract',
  unreviewed: 'Transcribed',
  reviewed: 'Corrected',
}

export const extractionChip = (state: ExtractionState) =>
  ({
    reviewed: 'chip chip--ok',
    unreviewed: 'chip chip--suggested',
    failed: 'chip chip--warn',
    stale: 'chip chip--warn',
    running: 'chip chip--ready',
  })[state as string] ?? 'chip'

const HANDWRITING_LABELS: Record<HandwritingStatus, string> = {
  legible: 'Legible',
  partially_legible: 'Partly legible',
  unreadable: 'Present but unreadable',
  blank: 'No handwriting',
}

/**
 * What the review form starts from. A saved review is taken exactly as saved
 * -- a time deliberately left empty stays empty, blank handwriting stays
 * blank -- never topped up from the model's reading. Only without a review do
 * we fall back: a time confirmed during preparation, else the model's.
 */
function draftFrom(view: ExtractionView): { time: string; status: HandwritingStatus; text: string } {
  const review = view.review
  if (review) {
    return {
      time: review.time ?? '',
      status: review.handwriting_status,
      text: review.handwriting_text ?? '',
    }
  }
  const result = view.attempt?.result ?? null
  return {
    time: view.confirmed_time ?? result?.time ?? '',
    status: result?.handwriting_status ?? 'legible',
    text: result?.handwriting_text ?? '',
  }
}

const formatCost = (cost: number | null | undefined) =>
  cost === null || cost === undefined ? 'not reported' : `${cost.toFixed(4)} credits`

/**
 * A row's extraction: run it, review the model's reading beside the row
 * image, correct it, and accept it. The model's own result is never edited --
 * accepted values are stored beside it -- and stays viewable.
 */
export default function ExtractionPanel({
  jobId,
  row,
  configured,
  busy,
  onExtract,
  onUpdated,
}: Props) {
  const view = row.extraction
  const attempt = view.attempt
  const result = attempt?.result ?? null
  const review = view.review

  // Draft values, reset whenever a different result or review is shown.
  const initial = draftFrom(view)
  const [time, setTime] = useState(initial.time)
  const [status, setStatus] = useState<HandwritingStatus>(initial.status)
  const [text, setText] = useState(initial.text)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const draftKey = `${attempt?.attempt_id}-${review?.review_id}`
  useEffect(() => {
    const draft = draftFrom(view)
    setTime(draft.time)
    setStatus(draft.status)
    setText(draft.text)
    setError(null)
    // Only a different result or review resets what the reviewer typed.
  }, [draftKey])

  const noText = status === 'blank' || status === 'unreadable'
  // A correction is optional: there is something to save only when the
  // draft differs from what is shown (the saved correction, else the model's).
  const baseline = draftFrom(view)
  const changed =
    baseline.time !== time ||
    baseline.status !== status ||
    (baseline.status === 'blank' || baseline.status === 'unreadable' ? '' : baseline.text) !== (noText ? '' : text)

  const accept = async () => {
    if (!attempt) return
    setSaving(true)
    setError(null)
    try {
      onUpdated(
        await acceptExtraction(jobId, row.row_id, {
          attempt_id: attempt.attempt_id,
          time: time || null,
          handwriting_status: status,
          handwriting_text: noText ? null : text,
        }),
      )
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not accept the extraction.')
    } finally {
      setSaving(false)
    }
  }

  // A row whose patient and date are known is made ready on extraction; one
  // marked ready by hand (without a patient, say) is ready already.
  const ready = row.approval_current || row.unresolved.length === 0
  const canExtract = configured && ready && view.state !== 'running' && !busy
  const extractLabel =
    view.state === 'failed' ? 'Retry transcription' : view.attempt_count ? 'Transcribe again' : 'Transcribe row'

  return (
    <section className="extraction" aria-label="Extraction">
      <header className="extraction__head">
        <h3>Transcription</h3>
        <span className={extractionChip(view.state)}>
          {view.state === 'not_ready'
            ? ready ? 'Not extracted' : row.unresolved.includes('date') ? 'Needs date' : 'Needs patient'
            : EXTRACTION_LABELS[view.state]}
        </span>
      </header>

      {!ready && (
        <p className="muted">
          {row.unresolved.includes('date')
            ? 'Set the date before extracting this row.'
            : 'Choose the patient (or "No patient") before extracting this row.'}
        </p>
      )}
      {!configured && (
        <p className="muted">Cloud extraction is not configured on the server.</p>
      )}

      <div className="extraction__actions">
        <button type="button" onClick={onExtract} disabled={!canExtract}>
          {view.state === 'running' ? 'Extracting…' : extractLabel}
        </button>
        {view.attempt_count > 0 && (
          <span className="muted">
            {view.attempt_count} attempt{view.attempt_count === 1 ? '' : 's'}
            {view.review_count > 0 && ` · ${view.review_count} review${view.review_count === 1 ? '' : 's'}`}
          </span>
        )}
      </div>

      {view.last_error && (
        <p className="notice notice--warn">
          {view.last_error.message}
          {view.last_error.kind === 'route_unavailable' && ' Try another model.'}
        </p>
      )}
      {view.last_attempt_status === 'superseded' && (
        <p className="notice notice--warn">
          The row changed while the model was reading it; that result was set aside. Transcribe again.
        </p>
      )}
      {view.state === 'stale' && (
        <p className="notice notice--warn">
          The row's evidence changed since this result. It is shown for reference only; extract again to review.
        </p>
      )}

      {attempt && result && (
        <>
          {view.time_discrepancy && (
            <p className="notice notice--warn">
              The model read {view.time_discrepancy.proposed}, but {view.time_discrepancy.confirmed} was
              confirmed during preparation. The confirmed time is kept unless you change it here.
            </p>
          )}
          {view.possible_nam_in_text && (
            <p className="notice notice--warn">
              The handwriting contains something shaped like a health number. Remove it before accepting.
            </p>
          )}

          <div className="field">
            <div className="field__head">
              <span className="field__label">Time</span>
              <span className="muted">model: {result.time ?? 'unknown'}</span>
            </div>
            <input
              type="time"
              value={time}
              disabled={!view.attempt_is_current || saving}
              onChange={(event) => setTime(event.target.value)}
            />
            <div className="field__foot">
              <span className="muted">
                {result.time_evidence ? `read from “${result.time_evidence}”` : 'Leave empty if unknown.'}
              </span>
            </div>
          </div>

          <div className="field">
            <div className="field__head">
              <span className="field__label">Handwriting</span>
              <span className="muted">model: {HANDWRITING_LABELS[result.handwriting_status]}</span>
            </div>
            <select
              value={status}
              disabled={!view.attempt_is_current || saving}
              onChange={(event) => setStatus(event.target.value as HandwritingStatus)}
            >
              {(Object.keys(HANDWRITING_LABELS) as HandwritingStatus[]).map((value) => (
                <option key={value} value={value}>
                  {HANDWRITING_LABELS[value]}
                </option>
              ))}
            </select>
            {!noText && (
              <textarea
                className="extraction__text"
                value={text}
                rows={4}
                spellCheck={false}
                disabled={!view.attempt_is_current || saving}
                onChange={(event) => setText(event.target.value)}
              />
            )}
            {result.unreadable_or_ambiguous.length > 0 && (
              <ul className="extraction__doubts">
                {result.unreadable_or_ambiguous.map((doubt, index) => (
                  <li key={index}>{doubt}</li>
                ))}
              </ul>
            )}
            {result.remarks && <p className="muted">Model remark: {result.remarks}</p>}
          </div>

          {error && <p className="error">{error}</p>}
          {review && (
            <p className="muted">
              Correction saved {new Date(review.reviewed_at).toLocaleString()}
              {(review.corrected.time || review.corrected.handwriting) ? ' (differs from the model)' : ' (same as the model)'}.
            </p>
          )}
          <p className="muted">
            Optional: correct the reading above and save it. Billing is proposed from the model's
            reading without this step; a saved correction is used instead and the proposal updates.
          </p>
          <button
            type="button"
            disabled={!view.attempt_is_current || saving || busy || !changed}
            onClick={() => void accept()}
          >
            {saving ? 'Saving…' : 'Save correction'}
          </button>

          <details className="extraction__raw">
            <summary>Original model result</summary>
            <dl>
              <dt>Model</dt>
              <dd>
                {attempt.requested_model}
                {attempt.returned_model && attempt.returned_model !== attempt.requested_model
                  ? ` (answered by ${attempt.returned_model})`
                  : ''}
              </dd>
              <dt>Provider</dt>
              <dd>{attempt.provider ?? 'not reported'}</dd>
              <dt>Finished</dt>
              <dd>
                {attempt.finished_at ? new Date(attempt.finished_at).toLocaleString() : '—'} · {attempt.tries} tr
                {attempt.tries === 1 ? 'y' : 'ies'} · prompt {attempt.prompt_version}
              </dd>
              <dt>Usage</dt>
              <dd>
                {attempt.usage?.prompt_tokens ?? '?'} in / {attempt.usage?.completion_tokens ?? '?'} out tokens ·{' '}
                {formatCost(attempt.usage?.cost)}
              </dd>
            </dl>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </details>
        </>
      )}
    </section>
  )
}
