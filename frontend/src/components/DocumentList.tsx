import { useEffect, useState } from 'react'
import { deleteJob, listDocuments } from '../api'
import type { DocumentSummary } from '../types'

interface Props {
  onOpen: (jobId: string) => void
}

const STATUS_LABELS: Record<string, string> = {
  awaiting_zones: 'Choosing zones',
  queued: 'Detecting',
  detecting: 'Detecting',
  awaiting_review: 'In review',
  rendering: 'Rendering',
  done: 'Redacted PDF ready',
  error: 'Detection failed',
}

/** Saved documents: everything uploaded and not yet deleted, newest first. */
export default function DocumentList({ onOpen }: Props) {
  const [documents, setDocuments] = useState<DocumentSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = () =>
    listDocuments()
      .then(setDocuments)
      .catch((cause) => setError(cause instanceof Error ? cause.message : 'Could not list documents.'))

  useEffect(() => {
    void refresh()
  }, [])

  const remove = async (doc: DocumentSummary) => {
    const warning =
      `Delete "${doc.filename}"?\n\nThis permanently removes the original PDF, the review, ` +
      `${doc.rows_total} appointment row${doc.rows_total === 1 ? '' : 's'} and all evidence from this computer.`
    if (!window.confirm(warning)) return
    try {
      await deleteJob(doc.job_id)
      await refresh()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not delete the document.')
    }
  }

  if (documents === null && !error) return null

  return (
    <section className="panel documents">
      <h2>Saved documents</h2>
      <p className="muted">
        Kept on this computer until you delete them. Reopen one to continue where you left off.
      </p>
      {error && <p className="error">{error}</p>}
      {documents && documents.length === 0 && <p className="muted">None yet.</p>}
      {documents && documents.length > 0 && (
        <ul className="documents__list">
          {documents.map((doc) => (
            <li key={doc.job_id} className="documents__item">
              <div>
                <strong>{doc.filename}</strong>
                <span className="muted">
                  {doc.doctor ? doc.doctor.snapshot.display_name : 'No doctor profile assigned'}
                </span>
                <span className="muted">
                  {new Date(doc.created_at * 1000).toLocaleString()} ·{' '}
                  {STATUS_LABELS[doc.status] ?? doc.status}
                  {doc.total_pages ? ` · ${doc.total_pages} page${doc.total_pages === 1 ? '' : 's'}` : ''}
                  {doc.rows_total > 0 && ` · ${doc.rows_approved}/${doc.rows_total} rows ready for extraction`}
                </span>
              </div>
              <div className="documents__actions">
                <button type="button" className="primary" onClick={() => onOpen(doc.job_id)}>
                  Open
                </button>
                <button type="button" className="danger" onClick={() => void remove(doc)}>
                  Delete
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
