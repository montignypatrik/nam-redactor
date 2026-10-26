import { useCallback, useEffect, useState } from 'react'
import {
  createDoctor,
  duplicateDoctor,
  getDoctor,
  getDoctorVersion,
  listDoctors,
  saveDoctor,
  setDoctorArchived,
  type DoctorContent,
} from '../api'
import { PRACTICE_FIELDS, invalidPractice, practiceSummary } from '../practice'
import type { DoctorExample, DoctorProfile, DoctorProfileSummary, DoctorVersion } from '../types'

interface Props {
  /** Open the editor on a new profile straight away. */
  startNew: boolean
  /** Back to where the user came from; with the profile to select, if any. */
  onDone: (selectProfileId: string | null) => void
}

const EMPTY: DoctorContent = {
  display_name: '',
  licence_number: '',
  group_number: '',
  sector_number: '',
  instructions: '',
  examples: [],
}

const INSTRUCTIONS_HINT = `Plain text, in your own words. For example:

Abbreviations:
- "sut" means suture; "pts" means points

Units and quantities:
- a number after "x" is a count

Context:
- a note in the margin applies to the whole row

Exclusions:
- never bill phone follow-ups

Exceptions:
- …`

const contentOf = (version: DoctorVersion): DoctorContent => ({
  display_name: version.display_name,
  licence_number: version.licence_number,
  group_number: version.group_number,
  sector_number: version.sector_number,
  instructions: version.instructions,
  examples: version.examples.map((example) => ({ ...example })),
})

const sameContent = (a: DoctorContent, b: DoctorContent) => JSON.stringify(a) === JSON.stringify(b)

/**
 * Doctor profiles: identifiers, and instructions and worked examples for
 * reading this doctor's annotations. Saving an edit adds a version; earlier
 * versions stay available to the documents that use them.
 */
export default function DoctorProfilesScreen({ startNew, onDone }: Props) {
  const [profiles, setProfiles] = useState<DoctorProfileSummary[]>([])
  const [showArchived, setShowArchived] = useState(false)
  const [selected, setSelected] = useState<DoctorProfile | null>(null)
  const [creating, setCreating] = useState(startNew)
  const [draft, setDraft] = useState<DoctorContent>(EMPTY)
  const [viewing, setViewing] = useState<DoctorVersion | null>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setProfiles(await listDoctors(true))
  }, [])

  useEffect(() => {
    refresh().catch((cause) => setError(cause instanceof Error ? cause.message : 'Could not load profiles.'))
  }, [refresh])

  const open = async (profileId: string) => {
    setError(null)
    setMessage(null)
    setViewing(null)
    const profile = await getDoctor(profileId)
    setSelected(profile)
    setCreating(false)
    setDraft(contentOf(profile.current))
  }

  const startCreating = () => {
    setSelected(null)
    setViewing(null)
    setCreating(true)
    setDraft(EMPTY)
    setMessage(null)
    setError(null)
  }

  const run = async (action: () => Promise<void>) => {
    setBusy(true)
    setError(null)
    setMessage(null)
    try {
      await action()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'That did not work.')
    } finally {
      setBusy(false)
    }
  }

  const save = () =>
    run(async () => {
      if (creating) {
        const profile = await createDoctor(draft)
        setSelected(profile)
        setCreating(false)
        setDraft(contentOf(profile.current))
        setMessage('Profile created.')
      } else if (selected) {
        const profile = await saveDoctor(selected.profile_id, draft, selected.current_version)
        setSelected(profile)
        setDraft(contentOf(profile.current))
        setMessage(
          profile.unchanged
            ? 'Nothing changed.'
            : `Saved as version ${profile.current_version}. Documents using an earlier version keep it until you adopt this one.`,
        )
      }
      await refresh()
    })

  const duplicate = () =>
    run(async () => {
      if (!selected) return
      const copy = await duplicateDoctor(selected.profile_id)
      await refresh()
      setSelected(copy)
      setDraft(contentOf(copy.current))
      setMessage('Duplicated. This copy is an independent profile — rename it and adjust as needed.')
    })

  const toggleArchived = () =>
    run(async () => {
      if (!selected) return
      const archiving = selected.status === 'active'
      if (archiving && !window.confirm(`Archive ${selected.display_name}? It can no longer be chosen for new documents; existing documents keep it.`)) return
      const profile = await setDoctorArchived(selected.profile_id, archiving)
      setSelected(profile)
      await refresh()
      setMessage(archiving ? 'Archived.' : 'Restored.')
    })

  const showVersion = (version: number) =>
    run(async () => {
      if (!selected) return
      setViewing(version === selected.current_version ? null : await getDoctorVersion(selected.profile_id, version))
    })

  const setExample = (index: number, change: Partial<DoctorExample>) =>
    setDraft((previous) => ({
      ...previous,
      examples: previous.examples.map((example, i) => (i === index ? { ...example, ...change } : example)),
    }))

  const invalid = invalidPractice(draft)
  const dirty = creating ? !sameContent(draft, EMPTY) : selected ? !sameContent(draft, contentOf(selected.current)) : false
  const archived = selected?.status === 'archived'
  const readOnly = Boolean(viewing) || archived
  const shown: DoctorContent = viewing ? contentOf(viewing) : draft
  const visible = profiles.filter((p) => showArchived || p.status === 'active')

  const leave = () => {
    if (dirty && !window.confirm('Leave without saving this profile?')) return
    onDone(selected && selected.status === 'active' ? selected.profile_id : null)
  }

  return (
    <section className="doctors">
      <header className="review__head">
        <div>
          <h1>Doctor profiles</h1>
          <p className="lede">
            One profile per doctor. Every document belongs to one doctor and keeps the exact profile
            version it was given.
          </p>
        </div>
        <div className="review__actions">
          <button type="button" onClick={leave}>
            Done
          </button>
        </div>
      </header>

      {error && <p className="error">{error}</p>}
      {message && <p className="notice">{message}</p>}

      <div className="doctors__body">
        <aside className="doctors__list">
          <button type="button" className="primary" onClick={startCreating} disabled={busy}>
            + New profile
          </button>
          <label className="doctors__toggle">
            <input type="checkbox" checked={showArchived} onChange={(e) => setShowArchived(e.target.checked)} />
            Show archived
          </label>
          <ul>
            {visible.map((profile) => (
              <li key={profile.profile_id}>
                <button
                  type="button"
                  className={profile.profile_id === selected?.profile_id ? 'row-list__item row-list__item--on' : 'row-list__item'}
                  onClick={() => void open(profile.profile_id)}
                >
                  <span>
                    <strong>{profile.display_name}</strong>
                    <span className="muted"> v{profile.current_version}</span>
                  </span>
                  {profile.status === 'archived' && <span className="chip">archived</span>}
                </button>
              </li>
            ))}
            {!visible.length && <li className="muted">No profiles yet.</li>}
          </ul>
        </aside>

        {(creating || selected) && (
          <div className="doctors__editor">
            <div className="doctors__editor-head">
              <h2>
                {creating ? 'New profile' : selected?.display_name}
                {selected && (
                  <span className="muted">
                    {' '}
                    · {viewing ? `viewing version ${viewing.version}` : `version ${selected.current_version}`}
                    {archived && ' · archived'}
                  </span>
                )}
              </h2>
              {selected && (
                <div className="row-editor__tools">
                  <button type="button" onClick={() => void duplicate()} disabled={busy}>
                    Duplicate
                  </button>
                  <button type="button" className={archived ? '' : 'danger'} onClick={() => void toggleArchived()} disabled={busy}>
                    {archived ? 'Restore' : 'Archive'}
                  </button>
                </div>
              )}
            </div>

            {viewing && (
              <p className="notice">
                Version {viewing.version} from {new Date(viewing.created_at).toLocaleString()}, read-only.{' '}
                <button type="button" className="link" onClick={() => setViewing(null)}>
                  Back to the current version
                </button>
              </p>
            )}
            {archived && !viewing && (
              <p className="notice notice--warn">Archived: restore it to edit or assign it.</p>
            )}

            <fieldset className="fieldset" disabled={readOnly || busy}>
              <legend>Doctor</legend>
              <label className="practice__field">
                <span>Name</span>
                <input
                  type="text"
                  value={shown.display_name}
                  maxLength={120}
                  placeholder="e.g. Dre Marie Tremblay"
                  onChange={(e) => setDraft((d) => ({ ...d, display_name: e.target.value }))}
                />
              </label>
              <div className="practice">
                {PRACTICE_FIELDS.map(({ key, label, placeholder }) => (
                  <label key={key} className="practice__field">
                    <span>{label}</span>
                    <input
                      type="text"
                      value={shown[key]}
                      placeholder={placeholder}
                      maxLength={30}
                      autoComplete="off"
                      spellCheck={false}
                      aria-invalid={invalid.includes(key)}
                      onChange={(e) => setDraft((d) => ({ ...d, [key]: e.target.value }))}
                    />
                  </label>
                ))}
              </div>
              {invalid.length > 0 && <p className="practice__error">Use only letters, digits, spaces and dashes.</p>}
            </fieldset>

            <fieldset className="fieldset" disabled={readOnly || busy}>
              <legend>Interpretation instructions</legend>
              <p className="muted doctors__help">
                How this doctor writes: abbreviations, units, what context matters, what to exclude,
                exceptions. Used later to interpret billing annotations; optional for now.
              </p>
              <textarea
                className="doctors__instructions"
                value={shown.instructions}
                rows={12}
                placeholder={INSTRUCTIONS_HINT}
                onChange={(e) => setDraft((d) => ({ ...d, instructions: e.target.value }))}
              />
            </fieldset>

            <fieldset className="fieldset" disabled={readOnly || busy}>
              <legend>Worked examples</legend>
              <p className="muted doctors__help">
                An annotation exactly as written, and how it should be read. Optional.
              </p>
              {shown.examples.map((example, index) => (
                <div key={index} className="doctors__example">
                  <label>
                    <span>Annotation as written</span>
                    <textarea
                      rows={2}
                      value={example.annotation}
                      spellCheck={false}
                      onChange={(e) => setExample(index, { annotation: e.target.value })}
                    />
                  </label>
                  <label>
                    <span>Expected interpretation</span>
                    <textarea
                      rows={2}
                      value={example.interpretation}
                      onChange={(e) => setExample(index, { interpretation: e.target.value })}
                    />
                  </label>
                  <label>
                    <span>Note (optional)</span>
                    <input
                      type="text"
                      value={example.note}
                      onChange={(e) => setExample(index, { note: e.target.value })}
                    />
                  </label>
                  {!readOnly && (
                    <button
                      type="button"
                      className="link"
                      onClick={() => setDraft((d) => ({ ...d, examples: d.examples.filter((_, i) => i !== index) }))}
                    >
                      Remove example
                    </button>
                  )}
                </div>
              ))}
              {!readOnly && (
                <button
                  type="button"
                  onClick={() =>
                    setDraft((d) => ({ ...d, examples: [...d.examples, { annotation: '', interpretation: '', note: '' }] }))
                  }
                >
                  + Add example
                </button>
              )}
            </fieldset>

            {!readOnly && (
              <div className="row">
                <button
                  type="button"
                  className="primary"
                  disabled={busy || !dirty || !draft.display_name.trim() || invalid.length > 0}
                  onClick={() => void save()}
                >
                  {creating ? 'Create profile' : 'Save as new version'}
                </button>
                {dirty && selected && (
                  <button type="button" onClick={() => setDraft(contentOf(selected.current))} disabled={busy}>
                    Discard changes
                  </button>
                )}
              </div>
            )}

            {selected && (
              <section className="doctors__history">
                <h3>Versions</h3>
                <ol>
                  {selected.versions.map((version) => (
                    <li key={version.version}>
                      <button
                        type="button"
                        className="link"
                        onClick={() => void showVersion(version.version)}
                      >
                        Version {version.version}
                        {version.version === selected.current_version && ' (current)'}
                      </button>{' '}
                      <span className="muted">
                        {new Date(version.created_at).toLocaleString()} · {version.display_name} ·{' '}
                        {practiceSummary(version)} · {version.example_count} example
                        {version.example_count === 1 ? '' : 's'}
                        {version.has_instructions ? ' · instructions' : ''}
                      </span>
                    </li>
                  ))}
                </ol>
              </section>
            )}
          </div>
        )}
        {!creating && !selected && (
          <p className="muted doctors__empty">Choose a profile to view or edit it, or create a new one.</p>
        )}
      </div>
    </section>
  )
}
