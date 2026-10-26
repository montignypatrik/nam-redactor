import { resultUrl } from '../api'

interface Props {
  jobId: string
  filename: string
  totalNams: number
  totalNames: number
  perPageCounts: number[]
  onBackToReview: () => void
  onDocuments: () => void
}

export default function DoneScreen({
  jobId,
  filename,
  totalNams,
  totalNames,
  perPageCounts,
  onBackToReview,
  onDocuments,
}: Props) {
  return (
    <section className="panel">
      <h1>Redaction complete</h1>
      <p className="lede">
        <strong>{filename}</strong> — {totalNams} NAM{totalNams === 1 ? '' : 's'} and{' '}
        {totalNames} name{totalNames === 1 ? '' : 's'} burned into{' '}
        {perPageCounts.length} page{perPageCounts.length === 1 ? '' : 's'}.
      </p>

      <p className="hint">
        The output is a fresh image-only PDF: the original text layer is gone,
        so a NAM or name cannot be recovered by selecting or copying text.
      </p>

      <ul className="page-list">
        {perPageCounts.map((count, index) => (
          <li key={index}>
            <span>Page {index + 1}</span>
            <span className={count ? 'found' : 'muted'}>
              {count === 0 ? 'clear' : `${count} redacted`}
            </span>
          </li>
        ))}
      </ul>

      <div className="row">
        <a className="primary" href={resultUrl(jobId)} download="redacted.pdf">
          Download redacted PDF
        </a>
        <button type="button" onClick={onBackToReview}>
          Back to review &amp; appointment rows
        </button>
        <button type="button" onClick={onDocuments}>
          Documents
        </button>
      </div>

      <p className="hint">
        The document, its review and its appointment rows are saved on this
        computer and reopen from the document list, even after the app restarts.
        They stay until you delete the document there.
      </p>
    </section>
  )
}
