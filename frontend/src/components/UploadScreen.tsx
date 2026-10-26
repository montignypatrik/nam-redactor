import { useEffect, useRef, useState } from 'react'
import { listDoctors } from '../api'
import { practiceSummary } from '../practice'
import type { DoctorProfileSummary } from '../types'

interface Props {
  busy: boolean
  /** Kept by the app, so a trip to the doctor profiles screen keeps them. */
  file: File | null
  onFileChange: (file: File) => void
  dpi: number
  onDpiChange: (dpi: number) => void
  /** A profile to select when the list loads, e.g. one just created. */
  preferredDoctorId: string | null
  onUpload: (file: File, dpi: number, doctor: { profileId: string; version: number }) => void
  onManageDoctors: (createNew: boolean) => void
}

/** The last doctor chosen in this browser (an id only, no identifiers). */
const DOCTOR_KEY = 'nam-redactor.doctor'

const DPI_CHOICES = [
  { value: 150, label: '150 DPI', hint: 'Lower memory, fastest' },
  { value: 200, label: '200 DPI', hint: 'Default balance' },
  { value: 300, label: '300 DPI', hint: 'Sharpest, slowest' },
]

export default function UploadScreen({
  busy,
  file,
  onFileChange,
  dpi,
  onDpiChange,
  preferredDoctorId,
  onUpload,
  onManageDoctors,
}: Props) {
  const [dragging, setDragging] = useState(false)
  const [doctors, setDoctors] = useState<DoctorProfileSummary[] | null>(null)
  const [doctorId, setDoctorId] = useState('')
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    listDoctors()
      .then((profiles) => {
        setDoctors(profiles)
        let remembered = ''
        try {
          remembered = localStorage.getItem(DOCTOR_KEY) ?? ''
        } catch {
          // storage blocked: no preselection
        }
        const pick = [preferredDoctorId, remembered].find((id) => profiles.some((p) => p.profile_id === id))
        setDoctorId(pick ?? (profiles.length === 1 ? profiles[0].profile_id : ''))
      })
      .catch((cause) => setError(cause instanceof Error ? cause.message : 'Could not load doctor profiles.'))
  }, [preferredDoctorId])

  const doctor = doctors?.find((p) => p.profile_id === doctorId) ?? null

  const choose = (candidate: File | undefined) => {
    if (candidate) onFileChange(candidate)
  }

  const submit = () => {
    if (!file || !doctor) return
    try {
      localStorage.setItem(DOCTOR_KEY, doctor.profile_id)
    } catch {
      // remembering is a convenience
    }
    onUpload(file, dpi, { profileId: doctor.profile_id, version: doctor.current_version })
  }

  return (
    <section className="panel">
      <h1>Redact RAMQ numbers from a PDF</h1>
      <p className="lede">
        Detection runs entirely on this machine. The document is never uploaded
        anywhere and is deleted when you clear the job.
      </p>

      <div
        className={`dropzone${dragging ? ' dropzone--active' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault()
          setDragging(false)
          choose(event.dataTransfer.files?.[0])
        }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') inputRef.current?.click()
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          hidden
          onChange={(event) => choose(event.target.files?.[0])}
        />
        {file ? (
          <>
            <strong>{file.name}</strong>
            <span className="muted">{(file.size / 1024 / 1024).toFixed(1)} MB — click to choose another</span>
          </>
        ) : (
          <>
            <strong>Drop a PDF here</strong>
            <span className="muted">or click to browse</span>
          </>
        )}
      </div>

      <fieldset className="fieldset">
        <legend>Doctor</legend>
        {doctors === null && !error && <p className="muted">Loading doctor profiles…</p>}
        {error && <p className="error">{error}</p>}
        {doctors && doctors.length === 0 && (
          <div className="notice">
            <p>
              No doctor profiles yet. Every document belongs to one doctor, whose licence,
              group and sector numbers come from their profile.
            </p>
            <button type="button" className="primary" onClick={() => onManageDoctors(true)}>
              Create a doctor profile
            </button>
          </div>
        )}
        {doctors && doctors.length > 0 && (
          <div className="doctor-pick">
            <label className="doctor-pick__select">
              <span className="sr-only">Doctor profile</span>
              <select value={doctorId} onChange={(event) => setDoctorId(event.target.value)}>
                <option value="">— choose the doctor —</option>
                {doctors.map((p) => (
                  <option key={p.profile_id} value={p.profile_id}>
                    {p.display_name}
                  </option>
                ))}
              </select>
            </label>
            <p className="doctor-pick__summary muted">
              {doctor ? `${practiceSummary(doctor)} · profile v${doctor.current_version}` : 'The whole file belongs to one doctor.'}
            </p>
            <button type="button" className="link" onClick={() => onManageDoctors(false)}>
              Manage doctor profiles
            </button>
          </div>
        )}
      </fieldset>

      <fieldset className="fieldset">
        <legend>Rasterization DPI</legend>
        <div className="choices">
          {DPI_CHOICES.map((choice) => (
            <label key={choice.value} className={`choice${dpi === choice.value ? ' choice--on' : ''}`}>
              <input
                type="radio"
                name="dpi"
                value={choice.value}
                checked={dpi === choice.value}
                onChange={() => onDpiChange(choice.value)}
              />
              <span className="choice__label">{choice.label}</span>
              <span className="muted">{choice.hint}</span>
            </label>
          ))}
        </div>
      </fieldset>

      <p className="hint">
        Next you pick the zones to keep, then each one is rendered, run through OCR,
        then reviewed. The first run also downloads the OCR models (one time,
        about 20&nbsp;MB) — nothing from your document is sent anywhere.
      </p>

      <button
        type="button"
        className="primary"
        disabled={!file || !doctor || busy}
        title={!doctor ? 'Choose the doctor first' : undefined}
        onClick={submit}
      >
        {busy ? 'Uploading…' : 'Next: choose zones'}
      </button>
    </section>
  )
}
