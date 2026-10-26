interface PageResult {
  page_num: number
  nams_on_page: number
}

interface Props {
  filename: string
  totalPages: number | null
  pages: PageResult[]
  runningTotal: number
}

export default function ProgressScreen({ filename, totalPages, pages, runningTotal }: Props) {
  const done = pages.length
  const percent = totalPages ? Math.round((done / totalPages) * 100) : 0

  return (
    <section className="panel">
      <h1>Scanning for RAMQ numbers and names</h1>
      <p className="lede">
        <strong>{filename}</strong>
        {totalPages ? ` — ${done} of ${totalPages} pages` : ' — opening document…'}
      </p>

      <div className="bar" role="progressbar" aria-valuenow={percent} aria-valuemin={0} aria-valuemax={100}>
        <div className="bar__fill" style={{ width: `${percent}%` }} />
      </div>

      {done === 0 && (
        <p className="hint">
          The first page takes the longest. If this is the very first run, the
          OCR models are being downloaded once; later runs start immediately.
        </p>
      )}

      {totalPages !== null && done === totalPages && (
        <p className="hint">
          All pages read. Finding patient names across the document…
        </p>
      )}

      {done > 0 && (
        <>
          <p className="tally">
            {runningTotal === 0
              ? 'No numbers found yet'
              : `${runningTotal} number${runningTotal === 1 ? '' : 's'} found so far`}
          </p>
          <ul className="page-list">
            {pages.map((page) => (
              <li key={page.page_num}>
                <span>Page {page.page_num}</span>
                <span className={page.nams_on_page ? 'found' : 'muted'}>
                  {page.nams_on_page === 0
                    ? 'clear'
                    : `${page.nams_on_page} number${page.nams_on_page === 1 ? '' : 's'}`}
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  )
}
