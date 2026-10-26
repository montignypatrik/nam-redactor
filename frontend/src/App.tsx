import { useCallback, useMemo, useRef, useState } from 'react'
import {
  fetchPages,
  getJob,
  listRows,
  saveReview,
  startDetection,
  streamEvents,
  uploadPdf,
} from './api'
import DocumentList from './components/DocumentList'
import ProgressScreen from './components/ProgressScreen'
import ReviewScreen from './components/ReviewScreen'
import UploadScreen from './components/UploadScreen'
import ZoneScreen from './components/ZoneScreen'
import DoctorProfilesScreen from './components/DoctorProfilesScreen'
import DoctorBar from './components/DoctorBar'
import RowsWorkspace from './components/RowsWorkspace'
import type { Box, DoctorAssignment, PageInfo, PagePayload, UploadResponse } from './types'

type Stage = 'upload' | 'doctors' | 'zones' | 'detecting' | 'review' | 'rows' | 'error'

interface PageResult {
  page_num: number
  nams_on_page: number
}

interface ZoneDraft {
  zones: Record<number, Box[]>
  dates: Record<number, string>
}

const datesOf = (pages: PageInfo[]): Record<number, string> => {
  const map: Record<number, string> = {}
  for (const page of pages) map[page.page_num] = page.date ?? ''
  return map
}

const cloneBoxes = (pages: PageInfo[], key: 'boxes' | 'detected_boxes'): Record<number, Box[]> => {
  const map: Record<number, Box[]> = {}
  for (const page of pages) map[page.page_num] = page[key].map((box) => ({ ...box }))
  return map
}

/** Comparable form of the review, to tell whether it differs from the saved one. */
const snapshot = (boxes: Record<number, Box[]>, dates: Record<number, string>) =>
  JSON.stringify({ boxes, dates })

export default function App() {
  const [stage, setStage] = useState<Stage>('upload')
  const [job, setJob] = useState<UploadResponse | null>(null)
  // The upload form's file and DPI live here, so a visit to the doctor
  // profiles screen (e.g. to create the first profile) keeps them.
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploadDpi, setUploadDpi] = useState(200)
  const [preferredDoctor, setPreferredDoctor] = useState<string | null>(null)
  const [doctorsStartNew, setDoctorsStartNew] = useState(false)
  /** Where "Done" on the profiles screen goes back to. */
  const [doctorsReturn, setDoctorsReturn] = useState<'upload' | 'rows'>('upload')
  /** The review was just confirmed: the appointments page processes every row. */
  const [autoProcess, setAutoProcess] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [totalPages, setTotalPages] = useState<number | null>(null)
  const [pageResults, setPageResults] = useState<PageResult[]>([])
  const [runningTotal, setRunningTotal] = useState(0)
  const [detected, setDetected] = useState<PageInfo[]>([])
  const [boxesByPage, setBoxesByPage] = useState<Record<number, Box[]>>({})
  const [datesByPage, setDatesByPage] = useState<Record<number, string>>({})
  const [reviewRevision, setReviewRevision] = useState(0)
  const [savedSnapshot, setSavedSnapshot] = useState('')
  const [saving, setSaving] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [zoneDraft, setZoneDraft] = useState<ZoneDraft | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const dirty = useMemo(
    () => stage === 'review' && snapshot(boxesByPage, datesByPage) !== savedSnapshot,
    [stage, boxesByPage, datesByPage, savedSnapshot],
  )

  const fail = (cause: unknown, fallback: string) => {
    setError(cause instanceof Error ? cause.message : fallback)
    setStage('error')
  }

  const handleUpload = async (file: File, dpi: number, doctor: { profileId: string; version: number }) => {
    setError(null)
    setUploading(true)
    setPageResults([])
    setRunningTotal(0)
    setTotalPages(null)

    let created: UploadResponse
    try {
      created = await uploadPdf(file, dpi, doctor)
    } catch (cause) {
      fail(cause, 'Upload failed.')
      setUploading(false)
      return
    }

    setUploading(false)
    setUploadFile(null)
    setJob(created)
    setZoneDraft(null)
    setTotalPages(created.total_pages)
    setStage('zones')
  }

  /** Open a saved document at whatever step it was left. */
  const openDocument = async (jobId: string) => {
    setError(null)
    setNotice(null)
    try {
      const status = await getJob(jobId)
      const opened: UploadResponse = {
        job_id: status.job_id,
        filename: status.filename,
        dpi: status.dpi,
        total_pages: status.source_page_count ?? status.total_pages ?? 0,
        licence_number: status.licence_number,
        group_number: status.group_number,
        sector_number: status.sector_number,
        doctor: status.doctor,
      }
      setJob(opened)
      if (status.status === 'awaiting_zones' || status.status === 'error') {
        const request = status.zone_request
        setZoneDraft(
          request
            ? {
                zones: Object.fromEntries(Object.entries(request.zones).map(([k, v]) => [Number(k), v])),
                dates: Object.fromEntries(Object.entries(request.dates).map(([k, v]) => [Number(k), v])),
              }
            : null,
        )
        if (status.status === 'error') {
          setNotice(`Detection did not finish last time${status.error ? ` (${status.error})` : ''}. Check the zones and continue.`)
        }
        setStage('zones')
      } else if (status.status === 'queued' || status.status === 'detecting') {
        void followDetection(opened.job_id)
      } else {
        // A document already processed reopens on its appointments.
        const { rows } = await listRows(opened.job_id)
        await showReview(opened.job_id, rows.some((row) => row.extraction.attempt_count > 0) ? 'rows' : 'review')
      }
    } catch (cause) {
      fail(cause, 'Could not open the document.')
    }
  }

  const followDetection = async (jobId: string) => {
    setTotalPages(null)
    setPageResults([])
    setRunningTotal(0)
    setStage('detecting')

    const controller = new AbortController()
    abortRef.current = controller

    try {
      await streamEvents(
        jobId,
        'detect',
        (event) => {
          if (event.type === 'start' && event.total_pages) setTotalPages(event.total_pages)
          if (event.type === 'page' && event.page_num) {
            setPageResults((previous) => [
              ...previous,
              { page_num: event.page_num!, nams_on_page: event.nams_on_page ?? 0 },
            ])
            setRunningTotal(event.total_nams ?? 0)
          }
          if (event.type === 'awaiting_review') void showReview(jobId)
          if (event.type === 'error') fail(new Error(event.message ?? 'Detection failed.'), '')
        },
        controller.signal,
      )
    } catch (cause) {
      if (!controller.signal.aborted) fail(cause, 'Lost connection to the server.')
    }
  }

  const handleStart = async (
    zonesByPage: Record<number, Box[]>,
    pageDates: Record<number, string>,
  ) => {
    if (!job) return
    const created = job
    setError(null)
    setNotice(null)
    setStarting(true)
    // Every page goes up with its date; the server keeps the zoned ones, or
    // all of them when there are no zones.
    const pages: PagePayload[] = Array.from({ length: created.total_pages }, (_, index) => {
      const pageNum = index + 1
      const date = pageDates[pageNum]
      return { page_num: pageNum, boxes: zonesByPage[pageNum] ?? [], ...(date ? { date } : {}) }
    })
    try {
      await startDetection(created.job_id, pages)
    } catch (cause) {
      fail(cause, 'Could not start detection.')
      return
    } finally {
      setStarting(false)
    }
    void followDetection(created.job_id)
  }

  const showReview = async (jobId: string, target: 'review' | 'rows' = 'review') => {
    try {
      const pages = await fetchPages(jobId)
      const boxes = cloneBoxes(pages.pages, 'boxes')
      const dates = datesOf(pages.pages)
      setDetected(pages.pages)
      setJob((current) =>
        current && {
          ...current,
          licence_number: pages.licence_number,
          group_number: pages.group_number,
          sector_number: pages.sector_number,
          doctor: pages.doctor,
        },
      )
      setBoxesByPage(boxes)
      setDatesByPage(dates)
      setReviewRevision(pages.review_revision)
      // Until a first save, the detections are unsaved work.
      setSavedSnapshot(pages.review_revision ? snapshot(boxes, dates) : '')
      setStage(target)
    } catch (cause) {
      fail(cause, 'Could not load the detected boxes.')
    }
  }

  const reviewPayload = (): PagePayload[] =>
    Object.entries(boxesByPage).map(([pageNum, boxes]) => ({
      page_num: Number(pageNum),
      boxes,
      date: datesByPage[Number(pageNum)],
    }))

  const handleSave = useCallback(async () => {
    if (!job) return
    setSaving(true)
    setNotice(null)
    try {
      const result = await saveReview(job.job_id, reviewPayload(), reviewRevision)
      setReviewRevision(result.review_revision)
      setSavedSnapshot(snapshot(boxesByPage, datesByPage))
      const count = result.invalidated_rows.length
      setNotice(
        count
          ? `Review saved. ${count} row${count === 1 ? '' : 's'} marked ready for extraction ${count === 1 ? 'shows' : 'show'} a changed redaction and must be marked ready again.`
          : 'Review saved.',
      )
    } catch (cause) {
      setNotice(cause instanceof Error ? cause.message : 'Could not save the review.')
      throw cause
    } finally {
      setSaving(false)
    }
    // reviewPayload reads boxesByPage and datesByPage, listed here.
  }, [job, boxesByPage, datesByPage, reviewRevision])

  /** Done reviewing: save if needed, then process every row on the appointments page. */
  const handleConfirmReview = async () => {
    if (dirty || !reviewRevision) {
      try {
        await handleSave()
      } catch {
        return // the notice says why
      }
    }
    setNotice(null)
    setAutoProcess(true)
    setStage('rows')
  }

  const handleBoxesChange = useCallback((pageNum: number, boxes: Box[]) => {
    setBoxesByPage((previous) => ({ ...previous, [pageNum]: boxes }))
  }, [])

  const handleDateChange = useCallback((pageNum: number, date: string) => {
    setDatesByPage((previous) => ({ ...previous, [pageNum]: date }))
  }, [])

  const handleResetAll = useCallback(() => {
    setBoxesByPage(cloneBoxes(detected, 'detected_boxes'))
  }, [detected])

  /** The document's doctor changed: identifiers follow the new snapshot. */
  const handleDoctorChanged = useCallback((doctor: DoctorAssignment) => {
    setJob((current) => current && { ...current, ...doctor.snapshot, doctor })
  }, [])

  const openDoctors = (createNew: boolean) => {
    setDoctorsStartNew(createNew)
    setDoctorsReturn('upload')
    setStage('doctors')
  }

  /** From the appointments: edit a profile, then come back to them. */
  const openDoctorsFromRows = () => {
    setDoctorsStartNew(false)
    setDoctorsReturn('rows')
    setStage('doctors')
  }

  /** Back to the document list. The document stays saved. */
  const handleDocuments = () => {
    if (dirty && !window.confirm('Leave without saving your redaction changes?')) return
    abortRef.current?.abort()
    abortRef.current = null
    setJob(null)
    setDetected([])
    setBoxesByPage({})
    setDatesByPage({})
    setSavedSnapshot('')
    setReviewRevision(0)
    setPageResults([])
    setRunningTotal(0)
    setTotalPages(null)
    setError(null)
    setNotice(null)
    setZoneDraft(null)
    setAutoProcess(false)
    setStage('upload')
  }

  return (
    <main className="app">
      <div className="app__top">
        <div className="app__badge">Local only — nothing leaves this machine</div>
        {stage === 'upload' && (
          <button type="button" className="link" onClick={() => openDoctors(false)}>
            Doctor profiles
          </button>
        )}
        {stage !== 'upload' && stage !== 'doctors' && (
          <button type="button" className="link" onClick={handleDocuments}>
            ← Documents
          </button>
        )}
      </div>

      {stage === 'upload' && (
        <>
          <UploadScreen
            busy={uploading}
            file={uploadFile}
            onFileChange={setUploadFile}
            dpi={uploadDpi}
            onDpiChange={setUploadDpi}
            preferredDoctorId={preferredDoctor}
            onUpload={handleUpload}
            onManageDoctors={openDoctors}
          />
          <DocumentList onOpen={(jobId) => void openDocument(jobId)} />
        </>
      )}

      {stage === 'doctors' && (
        <DoctorProfilesScreen
          startNew={doctorsStartNew}
          onDone={(profileId) => {
            if (doctorsReturn === 'rows' && job) {
              void showReview(job.job_id, 'rows')
              return
            }
            if (profileId) setPreferredDoctor(profileId)
            setStage('upload')
          }}
        />
      )}

      {stage === 'zones' && job && (
        <>
          {notice && <p className="notice notice--warn">{notice}</p>}
          <ZoneScreen
            key={job.job_id}
            jobId={job.job_id}
            filename={job.filename}
            totalPages={job.total_pages}
            onStart={handleStart}
            busy={starting}
            initialZones={zoneDraft?.zones}
            initialDates={zoneDraft?.dates}
          />
        </>
      )}

      {stage === 'detecting' && (
        <ProgressScreen
          filename={job?.filename ?? ''}
          totalPages={totalPages}
          pages={pageResults}
          runningTotal={runningTotal}
        />
      )}

      {stage === 'review' && job && (
        <ReviewScreen
          jobId={job.job_id}
          filename={job.filename}
          dpi={job.dpi}
          practice={job}
          doctor={job.doctor ?? null}
          onDoctorChanged={handleDoctorChanged}
          detected={detected}
          boxesByPage={boxesByPage}
          onBoxesChange={handleBoxesChange}
          datesByPage={datesByPage}
          onDateChange={handleDateChange}
          onResetAll={handleResetAll}
          reviewRevision={reviewRevision}
          dirty={dirty}
          saving={saving}
          onSave={handleSave}
          notice={notice}
          onConfirm={() => void handleConfirmReview()}
        />
      )}

      {stage === 'rows' && job && (
        <section className="review">
          <header className="review__head">
            <div>
              <h1>Appointments</h1>
              <p className="lede">
                <strong>{job.filename}</strong>
              </p>
              <DoctorBar jobId={job.job_id} doctor={job.doctor ?? null} practice={job} onChanged={handleDoctorChanged} />
            </div>
            <div className="review__actions">
              <button type="button" onClick={() => setStage('review')}>
                Back to review
              </button>
            </div>
          </header>
          <RowsWorkspace
            jobId={job.job_id}
            reviewRevision={reviewRevision}
            doctor={job.doctor ?? null}
            onManageDoctors={openDoctorsFromRows}
            autoProcess={autoProcess}
            onAutoProcessed={() => setAutoProcess(false)}
          />
        </section>
      )}

      {stage === 'error' && (
        <section className="panel">
          <h1>Something went wrong</h1>
          <p className="error">{error}</p>
          <p className="muted">Any saved work is kept; reopen the document from the list.</p>
          <button type="button" className="primary" onClick={handleDocuments}>
            Back to documents
          </button>
        </section>
      )}
    </main>
  )
}
