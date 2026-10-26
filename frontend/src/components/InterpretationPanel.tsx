import type { DoctorAssignment, Row } from '../types'
import BillingActs, { INTERPRETATION_LABELS } from './BillingActs'

interface Props {
  row: Row
  doctor: DoctorAssignment | null
  configured: boolean
  busy: boolean
  onInterpret: () => void
  onManageDoctors: () => void
}

const formatCost = (cost: number | null | undefined) =>
  cost === null || cost === undefined ? 'not reported' : `${cost.toFixed(4)} credits`

/**
 * A row's proposed billing, with what it was based on. Proposals are
 * unvalidated: nothing here approves one.
 */
export default function InterpretationPanel({ row, doctor, configured, busy, onInterpret, onManageDoctors }: Props) {
  const view = row.interpretation
  const attempt = view.attempt
  const result = attempt?.result ?? null
  const canRun =
    configured && !busy && view.state !== 'running' && view.state !== 'blocked' && view.state !== 'not_ready'
  const usedVersion = attempt?.inputs.doctor.version

  return (
    <section className="extraction" aria-label="Proposed billing">
      <header className="extraction__head">
        <h3>Proposed billing</h3>
        <span className={view.state === 'proposed' ? 'chip chip--unvalidated' : view.state === 'stale' || view.state === 'failed' || view.state === 'blocked' ? 'chip chip--warn' : 'chip'}>
          {INTERPRETATION_LABELS[view.state]}
        </span>
      </header>

      {doctor ? (
        <p className="muted">
          Interpreted with <strong>{doctor.snapshot.display_name}</strong>, profile v{doctor.version}
          {doctor.newer_version_available && doctor.current_version && ` (v${doctor.current_version} available — adopt it above to use it)`}
          .{' '}
          <button type="button" className="link" onClick={onManageDoctors}>
            Edit doctor profile
          </button>
        </p>
      ) : null}

      {view.blocked_message && (
        <div className="notice notice--warn">
          <p>{view.blocked_message}</p>
          {view.blocked_reason === 'empty_profile' && (
            <button type="button" onClick={onManageDoctors}>
              Edit doctor profile
            </button>
          )}
        </div>
      )}
      {view.state === 'not_ready' && !view.blocked_message && (
        <p className="muted">Interpretation runs automatically once the row is transcribed.</p>
      )}
      {view.last_error && <p className="notice notice--warn">{view.last_error.message}</p>}
      {view.last_attempt_status === 'superseded' && (
        <p className="notice notice--warn">An input changed while the model was working; that result was set aside.</p>
      )}
      {attempt && !view.is_current && (
        <p className="notice notice--warn">
          Outdated: the transcription, the evidence or the doctor changed since this proposal
          {usedVersion ? ` (made with profile v${usedVersion})` : ''}. Interpret again to update it.
        </p>
      )}

      <BillingActs view={view} showLabel={false} />

      {result && (
        <>
          {result.unresolved.length > 0 && (
            <div>
              <p className="field__label">Unresolved</p>
              <ul className="extraction__doubts">
                {result.unresolved.map((item, index) => (
                  <li key={index}>
                    “{item.text}” — {item.issue}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {result.conflicts.length > 0 && (
            <div>
              <p className="field__label">Conflicts</p>
              <ul className="extraction__doubts">
                {result.conflicts.map((conflict, index) => (
                  <li key={index}>{conflict}</li>
                ))}
              </ul>
            </div>
          )}
          {result.ignored_notes.length > 0 && (
            <div>
              <p className="field__label">Clinical notes (not billing)</p>
              <ul className="muted interp__ignored">
                {result.ignored_notes.map((note, index) => (
                  <li key={index}>
                    “{note.text}” — {note.reason}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {result.remarks && <p className="muted">Model remark: {result.remarks}</p>}
        </>
      )}

      <div className="extraction__actions">
        <button type="button" onClick={onInterpret} disabled={!canRun}>
          {view.state === 'running' ? 'Interpreting…' : view.attempt_count ? 'Interpret again' : 'Interpret'}
        </button>
        {view.attempt_count > 0 && (
          <span className="muted">
            {view.attempt_count} attempt{view.attempt_count === 1 ? '' : 's'}
          </span>
        )}
      </div>

      {attempt && result && (
        <details className="extraction__raw">
          <summary>How this proposal was made</summary>
          <dl>
            <dt>Doctor profile</dt>
            <dd>version {attempt.inputs.doctor.version}</dd>
            <dt>Transcription used</dt>
            <dd>
              {attempt.inputs.source === 'reviewer' ? 'your correction' : "the model's (not reviewed)"} · time{' '}
              {attempt.inputs.time ?? 'unknown'}
            </dd>
            <dt>Model</dt>
            <dd>
              {attempt.requested_model}
              {attempt.provider ? ` via ${attempt.provider}` : ''}
            </dd>
            <dt>Finished</dt>
            <dd>
              {attempt.finished_at ? new Date(attempt.finished_at).toLocaleString() : '—'} · prompt {attempt.prompt_version}
            </dd>
            <dt>Usage</dt>
            <dd>
              {attempt.usage?.prompt_tokens ?? '?'} in / {attempt.usage?.completion_tokens ?? '?'} out tokens ·{' '}
              {formatCost(attempt.usage?.cost)}
            </dd>
          </dl>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </details>
      )}
    </section>
  )
}
