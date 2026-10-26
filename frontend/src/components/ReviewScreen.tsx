import { useEffect, useMemo, useState } from 'react'
import { resolveNam, sourceImageUrl } from '../api'
import type { Bounds, Box, DoctorAssignment, PageInfo, Practice } from '../types'
import SourcePageCanvas, { sourceGroups } from './SourcePageCanvas'
import DoctorBar from './DoctorBar'
import RowsWorkspace from './RowsWorkspace'

interface Props {
  jobId: string
  filename: string
  dpi: number
  practice: Practice
  doctor: DoctorAssignment | null
  onDoctorChanged: (doctor: DoctorAssignment) => void
  onManageDoctors: () => void
  /** Open on this tab (e.g. back from editing a doctor profile). */
  initialTab?: 'redactions' | 'rows'
  detected: PageInfo[]
  boxesByPage: Record<number, Box[]>
  onBoxesChange: (pageNum: number, boxes: Box[]) => void
  datesByPage: Record<number, string>
  onDateChange: (pageNum: number, date: string) => void
  onResetAll: () => void
  /** Revision of the saved review; 0 until the first save. */
  reviewRevision: number
  /** The on-screen redactions or dates differ from the saved review. */
  dirty: boolean
  saving: boolean
  onSave: () => Promise<void>
  /** A one-line outcome of the last save, e.g. approvals it invalidated. */
  notice: string | null
}

export default function ReviewScreen({
  jobId,
  filename,
  dpi,
  practice,
  doctor,
  onDoctorChanged,
  onManageDoctors,
  initialTab = 'redactions',
  detected,
  boxesByPage,
  onBoxesChange,
  datesByPage,
  onDateChange,
  onResetAll,
  reviewRevision,
  dirty,
  saving,
  onSave,
  notice,
}: Props) {
  const [tab, setTab] = useState<'redactions' | 'rows'>(initialTab)
  // Review works on original pages; each shows all the sections cut from it.
  const groups = useMemo(() => sourceGroups(detected), [detected])
  const [current, setCurrent] = useState(groups[0]?.sourcePage ?? 1)
  const [imageLoading, setImageLoading] = useState(true)
  /** The box just drawn, awaiting its NAM: which section, which index. */
  const [pending, setPending] = useState<{ workingPage: number; index: number } | null>(null)
  const [typedNam, setTypedNam] = useState('')
  const [namError, setNamError] = useState<string | null>(null)
  const [assigning, setAssigning] = useState(false)
  const [outsideHint, setOutsideHint] = useState(false)

  useEffect(() => {
    setImageLoading(true)
    setPending(null)
  }, [current])

  const group = groups.find((g) => g.sourcePage === current) ?? groups[0]
  const zones = group?.zones ?? []

  const handleDraw = (workingPage: number, rect: Bounds) => {
    // A newly drawn box arrives unlabelled; offer to type its NAM so the
    // record can share an ID with occurrences detection already found.
    const boxes = [...(boxesByPage[workingPage] ?? []), { ...rect, confidence: 1 }]
    onBoxesChange(workingPage, boxes)
    setPending({ workingPage, index: boxes.length - 1 })
    setOutsideHint(false)
    setTypedNam('')
    setNamError(null)
  }

  const handleRemove = (workingPage: number, index: number) => {
    onBoxesChange(workingPage, (boxesByPage[workingPage] ?? []).filter((_, i) => i !== index))
    setPending(null)
  }

  const updatePending = (change: (box: Box) => Box) => {
    if (!pending) return
    const boxes = [...(boxesByPage[pending.workingPage] ?? [])]
    if (pending.index >= boxes.length) return
    boxes[pending.index] = change(boxes[pending.index])
    onBoxesChange(pending.workingPage, boxes)
  }

  const assignTypedNam = async () => {
    if (!pending) return
    setAssigning(true)
    setNamError(null)
    try {
      const patientId = await resolveNam(jobId, typedNam)
      updatePending((box) => ({ ...box, patient_id: patientId }))
      setPending(null)
      setTypedNam('')
    } catch (cause) {
      setNamError(cause instanceof Error ? cause.message : 'Could not assign an ID.')
    } finally {
      setAssigning(false)
    }
  }

  const markAsName = () => {
    updatePending(({ patient_id: _dropped, ...box }) => ({ ...box, kind: 'name' }))
    dismissPrompt()
  }

  const dismissPrompt = () => {
    setPending(null)
    setTypedNam('')
    setNamError(null)
  }

  const detectedByPage = useMemo(() => {
    const map: Record<number, number> = {}
    for (const page of detected) map[page.page_num] = page.detected_boxes.length
    return map
  }, [detected])

  const allBoxes = Object.values(boxesByPage).flat()
  const totalBoxes = allBoxes.length
  const nameBoxes = allBoxes.filter((box) => box.kind === 'name').length
  const edited = detected.some(
    (page) => JSON.stringify(boxesByPage[page.page_num] ?? []) !== JSON.stringify(page.detected_boxes),
  )
  const undated = detected.filter((page) => !datesByPage[page.page_num]).map((page) => page.page_num)

  return (
    <section className="review">
      <header className="review__head">
        <div>
          <h1>{tab === 'redactions' ? 'Review the blackouts' : 'Appointment rows'}</h1>
          <p className="lede">
            <strong>{filename}</strong> · {detected.length} pages · {dpi} DPI
          </p>
          <DoctorBar jobId={jobId} doctor={doctor} practice={practice} onChanged={onDoctorChanged} />
          <ul className="legend" aria-label="Box types">
            <li>
              <span className="legend__swatch" aria-hidden="true" />
              {totalBoxes - nameBoxes} NAM{totalBoxes - nameBoxes === 1 ? '' : 's'}
            </li>
            <li>
              <span className="legend__swatch legend__swatch--name" aria-hidden="true" />
              {nameBoxes} name{nameBoxes === 1 ? '' : 's'}
            </li>
          </ul>
        </div>
        <div className="review__actions">
          <span className={dirty ? 'save-state save-state--dirty' : 'save-state'}>
            {dirty ? 'Unsaved changes' : reviewRevision ? `Review saved (rev ${reviewRevision})` : 'Not saved yet'}
          </span>
          <button
            type="button"
            onClick={() => void onSave()}
            disabled={saving || (!dirty && reviewRevision > 0) || undated.length > 0}
          >
            {saving ? 'Saving…' : 'Save review'}
          </button>
          {tab === 'redactions' && (
            <button type="button" onClick={onResetAll} disabled={!edited}>
              Reset to detected
            </button>
          )}
        </div>
      </header>

      <div className="tabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'redactions'}
          className={tab === 'redactions' ? 'tab tab--on' : 'tab'}
          onClick={() => setTab('redactions')}
        >
          Redactions
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'rows'}
          className={tab === 'rows' ? 'tab tab--on' : 'tab'}
          onClick={async () => {
            // Rows and their evidence are built on the saved review.
            if (dirty || !reviewRevision) await onSave()
            setTab('rows')
          }}
          disabled={saving || undated.length > 0}
          title={undated.length ? 'Every page needs a date first' : undefined}
        >
          Appointment rows
        </button>
      </div>

      {notice && <p className="notice">{notice}</p>}

      {tab === 'rows' && (
        <RowsWorkspace
          jobId={jobId}
          pages={detected.map((page) => ({ ...page, date: datesByPage[page.page_num] ?? page.date }))}
          reviewRevision={reviewRevision}
          reviewDirty={dirty}
          onSaveReview={onSave}
          doctor={doctor}
          onManageDoctors={onManageDoctors}
        />
      )}

      {tab === 'redactions' && (
      <>
      <p className="hint">
        Every black rectangle is burned into the appointment row images — nothing
        underneath survives. NAM boxes are stamped with the
        patient ID in white; patient names (amber outline) are blacked out with
        no label. <strong>Drag</strong> on the page to add a box;{' '}
        <strong>click</strong> a box and press <kbd>Delete</kbd> to remove it.
        For a number detection missed, draw a box and type the NAM so it shares
        an ID with that patient's other occurrences.
      </p>

      <div className="review__body">
        <nav className="page-nav" aria-label="Original pages">
          {groups.map(({ sourcePage, zones: pageZones }) => {
            const count = pageZones.reduce((sum, z) => sum + (boxesByPage[z.workingPage] ?? []).length, 0)
            const was = pageZones.reduce((sum, z) => sum + (detectedByPage[z.workingPage] ?? 0), 0)
            const undatedHere = pageZones.some((z) => !datesByPage[z.workingPage])
            return (
              <button
                key={sourcePage}
                type="button"
                className={sourcePage === current ? 'page-nav__item page-nav__item--on' : 'page-nav__item'}
                onClick={() => setCurrent(sourcePage)}
              >
                <span>
                  Page {sourcePage}
                  <span className={undatedHere ? 'page-nav__date page-nav__date--missing' : 'page-nav__date'}>
                    {pageZones.length} section{pageZones.length === 1 ? '' : 's'}
                    {undatedHere && ' · date missing'}
                  </span>
                </span>
                <span className={count ? 'found' : 'muted'}>
                  {count}
                  {count !== was ? ` (was ${was})` : ''}
                </span>
              </button>
            )
          })}
        </nav>

        <div className="stage">
          <div className="zone-dates">
            {zones.map((zone) => (
              <label
                key={zone.workingPage}
                className={`date-field${datesByPage[zone.workingPage] ? '' : ' date-field--missing'}`}
              >
                <span>{zones.length > 1 ? `Section ${zone.label} date` : 'Date'}</span>
                <input
                  type="date"
                  value={datesByPage[zone.workingPage] ?? ''}
                  required
                  aria-invalid={!datesByPage[zone.workingPage]}
                  onChange={(event) => onDateChange(zone.workingPage, event.target.value)}
                />
              </label>
            ))}
          </div>
          {outsideHint && <p className="draw-hint">Draw inside a section — the greyed-out area is not part of the document.</p>}
          {imageLoading && <p className="muted stage__loading">Rendering page…</p>}
          <SourcePageCanvas
            imageUrl={sourceImageUrl(jobId, current)}
            zones={zones}
            boxesByWorking={boxesByPage}
            onDraw={handleDraw}
            onRemove={handleRemove}
            onRejected={() => setOutsideHint(true)}
            onImageLoad={() => setImageLoading(false)}
          />

          {pending !== null && (
            <div className="nam-prompt">
              <p>
                New box in section #{pending.workingPage}. Type its NAM to give it the same ID
                as that patient's other records, mark it as a name, or skip to
                leave it blank.
              </p>
              <div className="row">
                <input
                  className="nam-prompt__input"
                  value={typedNam}
                  placeholder="abcd 1234 5678"
                  aria-label="NAM for the new box"
                  onChange={(event) => setTypedNam(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') void assignTypedNam()
                    if (event.key === 'Escape') dismissPrompt()
                  }}
                />
                <button
                  type="button"
                  className="primary"
                  disabled={assigning || !typedNam.trim()}
                  onClick={() => void assignTypedNam()}
                >
                  {assigning ? 'Checking…' : 'Assign ID'}
                </button>
                <button type="button" onClick={markAsName}>
                  It's a name
                </button>
                <button type="button" onClick={dismissPrompt}>
                  Skip
                </button>
              </div>
              {namError && <p className="error">{namError}</p>}
            </div>
          )}
        </div>
      </div>
      </>
      )}
    </section>
  )
}
