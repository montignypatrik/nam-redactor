import type { InterpretationView } from '../types'

export const INTERPRETATION_LABELS: Record<InterpretationView['state'], string> = {
  not_ready: 'Waiting for transcription',
  blocked: 'Cannot interpret',
  pending: 'Not interpreted',
  running: 'Interpreting…',
  failed: 'Interpretation failed',
  stale: 'Outdated',
  proposed: 'Unvalidated',
}

const formatUnits = (units: number) => `${units} unit${units === 1 ? '' : 's'}`

/**
 * Proposed billing acts, one per line, each with its own context elements
 * and units. Always labelled: "Unvalidated" for a current proposal,
 * "Outdated" for one whose inputs have changed since.
 */
export default function BillingActs({
  view,
  compact = false,
  showLabel = true,
}: {
  view: InterpretationView
  compact?: boolean
  /** Off where the surrounding panel already shows the same label. */
  showLabel?: boolean
}) {
  const result = view.attempt?.result ?? null
  if (!result) return null
  const outdated = !view.is_current
  return (
    <div className={`acts${outdated ? ' acts--outdated' : ''}`}>
      {showLabel && (
        <span className={outdated ? 'chip chip--warn' : 'chip chip--unvalidated'}>
          {outdated ? 'Outdated' : 'Unvalidated'}
        </span>
      )}
      {result.no_billing_notation && <p className="muted acts__none">No billing notation found.</p>}
      {result.acts.length > 0 && (
        <ol className="acts__list">
          {result.acts.map((act, index) => (
            <li key={index} className="act">
              <span className={act.code ? 'act__code' : 'act__code act__code--none'}>{act.code ?? 'unresolved'}</span>
              {act.context_elements.map((context) => (
                <span key={context} className="act__context">
                  {context}
                </span>
              ))}
              {act.units !== null && <span className="act__units">{formatUnits(act.units)}</span>}
              {!compact && (
                <span className="act__source muted">
                  “{act.source_text}”{act.profile_reference ? ` · ${act.profile_reference}` : ''}
                </span>
              )}
            </li>
          ))}
        </ol>
      )}
      {(result.unresolved.length > 0 || result.conflicts.length > 0) && (
        <p className="acts__issues">
          {result.unresolved.length > 0 && `${result.unresolved.length} unresolved`}
          {result.unresolved.length > 0 && result.conflicts.length > 0 && ' · '}
          {result.conflicts.length > 0 && `${result.conflicts.length} conflict${result.conflicts.length === 1 ? '' : 's'}`}
        </p>
      )}
    </div>
  )
}
