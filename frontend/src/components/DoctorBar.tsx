import { useEffect, useState } from 'react'
import { assignDoctor, listDoctors } from '../api'
import { practiceSummary } from '../practice'
import type { DoctorAssignment, DoctorProfileSummary, Practice } from '../types'

interface Props {
  jobId: string
  doctor: DoctorAssignment | null
  /** The identifiers on the document -- for older documents, as typed at upload. */
  practice: Practice
  onChanged: (doctor: DoctorAssignment) => void
}

/**
 * The document's doctor. A profile edit never reaches a document on its own:
 * changing the doctor, or adopting a newer version of the same profile, is
 * an explicit action here. Redactions, rows and reviews are unaffected.
 */
export default function DoctorBar({ jobId, doctor, practice, onChanged }: Props) {
  const [choosing, setChoosing] = useState(false)
  const [profiles, setProfiles] = useState<DoctorProfileSummary[]>([])
  const [choice, setChoice] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!choosing) return
    listDoctors()
      .then((list) => {
        setProfiles(list)
        setChoice(list.find((p) => p.profile_id !== doctor?.profile_id)?.profile_id ?? list[0]?.profile_id ?? '')
      })
      .catch((cause) => setError(cause instanceof Error ? cause.message : 'Could not load profiles.'))
  }, [choosing, doctor?.profile_id])

  const assign = async (profileId: string, version?: number) => {
    setBusy(true)
    setError(null)
    try {
      const result = await assignDoctor(jobId, profileId, version)
      onChanged(result.doctor)
      setChoosing(false)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not change the doctor.')
    } finally {
      setBusy(false)
    }
  }

  const chosen = profiles.find((p) => p.profile_id === choice)

  return (
    <div className="doctor-bar">
      {doctor ? (
        <p>
          <span className="doctor-bar__label">Doctor</span> <strong>{doctor.snapshot.display_name}</strong>{' '}
          <span className="muted">
            · {practiceSummary(doctor.snapshot)} · profile v{doctor.version}
            {doctor.profile_status === 'archived' && ' (profile archived)'}
          </span>
        </p>
      ) : (
        <p>
          <span className="doctor-bar__label">Doctor</span>{' '}
          <strong className="doctor-bar__none">No doctor profile assigned</strong>{' '}
          <span className="muted">· {practiceSummary(practice)} (as entered at upload)</span>
        </p>
      )}

      <div className="doctor-bar__actions">
        {doctor?.newer_version_available && doctor.current_version && doctor.profile_status === 'active' && (
          <button
            type="button"
            disabled={busy}
            onClick={() => void assign(doctor.profile_id, doctor.current_version ?? undefined)}
            title="The profile was edited since this document was assigned. Adopt the new version's identifiers."
          >
            Adopt v{doctor.current_version}
          </button>
        )}
        {!choosing && (
          <button type="button" className="link" onClick={() => setChoosing(true)} disabled={busy}>
            {doctor ? 'Change doctor' : 'Assign a doctor'}
          </button>
        )}
      </div>

      {choosing && (
        <div className="doctor-bar__choose">
          <select value={choice} onChange={(event) => setChoice(event.target.value)} disabled={busy}>
            {profiles.map((p) => (
              <option key={p.profile_id} value={p.profile_id}>
                {p.display_name}
                {p.profile_id === doctor?.profile_id ? ' (current)' : ''}
              </option>
            ))}
          </select>
          {chosen && <span className="muted">{practiceSummary(chosen)}</span>}
          <button
            type="button"
            className="primary"
            disabled={busy || !chosen}
            onClick={() => chosen && void assign(chosen.profile_id, chosen.current_version)}
          >
            Assign
          </button>
          <button type="button" onClick={() => setChoosing(false)} disabled={busy}>
            Cancel
          </button>
          {!profiles.length && <span className="muted">No active doctor profiles.</span>}
        </div>
      )}
      {choosing && (
        <p className="muted doctor-bar__note">
          Applies to the whole document. Redactions, rows, extractions and reviews are kept.
        </p>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  )
}
