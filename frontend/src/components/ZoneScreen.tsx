import { useMemo, useState } from 'react'
import { pageImageUrl } from '../api'
import type { Box } from '../types'
import BoxCanvas from './BoxCanvas'

interface Props {
  jobId: string
  filename: string
  totalPages: number
  /** Called with the zones and the date of every page. */
  onStart: (zonesByPage: Record<number, Box[]>, datesByPage: Record<number, string>) => void
  busy: boolean
  /** Zones and dates submitted before, when reopening a saved document. */
  initialZones?: Record<number, Box[]>
  initialDates?: Record<number, string>
}

/** Pages are shown many at once, so they load at a lighter resolution.
 *  Zones are normalized, so the resolution does not affect where they land. */
const PREVIEW_DPI = 110

const WHOLE_PAGE: Box = { x1: 0, y1: 0, x2: 1, y2: 1, confidence: 1 }

/**
 * Zone indices in the order the server will emit them as pages.
 *
 * Mirrors ``_reading_order`` in redactor.py: centre-y bucketed by half the
 * median zone height, then left to right. The numbers shown on screen are a
 * promise about the output, so they must follow the same rule.
 */
function readingOrder(boxes: Box[]): number[] {
  if (!boxes.length) return []
  const heights = boxes.map((box) => box.y2 - box.y1).sort((a, b) => a - b)
  const tolerance = heights[Math.floor(heights.length / 2)] / 2 || 1
  const key = (box: Box) => Math.round((box.y1 + box.y2) / 2 / tolerance)
  return boxes
    .map((_, index) => index)
    .sort((a, b) => key(boxes[a]) - key(boxes[b]) || boxes[a].x1 - boxes[b].x1)
}

export default function ZoneScreen({
  jobId,
  filename,
  totalPages,
  onStart,
  busy,
  initialZones,
  initialDates,
}: Props) {
  const [zonesByPage, setZonesByPage] = useState<Record<number, Box[]>>(initialZones ?? {})
  const [datesByPage, setDatesByPage] = useState<Record<number, string>>(initialDates ?? {})
  // Pages left out of the document entirely, whatever zones they hold.
  const [excluded, setExcluded] = useState<Set<number>>(() => new Set())

  const pageNums = useMemo(
    () => Array.from({ length: totalPages }, (_, index) => index + 1),
    [totalPages],
  )

  const keptPages = pageNums.filter((pageNum) => !excluded.has(pageNum))
  const zonesOf = (pageNum: number) => (excluded.has(pageNum) ? [] : zonesByPage[pageNum] ?? [])

  // The output page number each zone will become, per source page.
  const tagsByPage = useMemo(() => {
    const tags: Record<number, string[]> = {}
    let next = 1
    for (const pageNum of pageNums) {
      const zones = excluded.has(pageNum) ? [] : zonesByPage[pageNum] ?? []
      const pageTags: string[] = new Array(zones.length)
      for (const index of readingOrder(zones)) pageTags[index] = String(next++)
      tags[pageNum] = pageTags
    }
    return tags
  }, [pageNums, zonesByPage, excluded])

  const totalZones = keptPages.reduce((sum, pageNum) => sum + zonesOf(pageNum).length, 0)
  const pagesWithZones = keptPages.filter((pageNum) => zonesOf(pageNum).length).length

  const setZones = (pageNum: number, zones: Box[]) =>
    setZonesByPage((previous) => ({ ...previous, [pageNum]: zones }))

  const setDate = (pageNum: number, date: string) =>
    setDatesByPage((previous) => ({ ...previous, [pageNum]: date }))

  const toggleExcluded = (pageNum: number) =>
    setExcluded((previous) => {
      const next = new Set(previous)
      if (!next.delete(pageNum)) next.add(pageNum)
      return next
    })

  // Every page that will be kept needs a date: the zoned ones, or all of them
  // when the document is kept whole.
  const undatedZoned = keptPages.filter((pageNum) => zonesOf(pageNum).length && !datesByPage[pageNum])
  const undatedAll = keptPages.filter((pageNum) => !datesByPage[pageNum])

  const keptZones = () =>
    Object.fromEntries(keptPages.map((pageNum) => [pageNum, zonesOf(pageNum)]))

  // With no page excluded, no zones means the document as uploaded; with
  // some excluded, each kept page becomes one whole-page zone.
  const useWholePages = () =>
    onStart(
      excluded.size ? Object.fromEntries(keptPages.map((pageNum) => [pageNum, [WHOLE_PAGE]])) : {},
      datesByPage,
    )

  return (
    <section className="zones">
      <header className="review__head">
        <div>
          <h1>Choose the zones</h1>
          <p className="lede">
            <strong>{filename}</strong> · {totalPages} page{totalPages === 1 ? '' : 's'}
          </p>
        </div>
        <div className="review__actions">
          <button type="button" onClick={() => setZonesByPage({})} disabled={!totalZones || busy}>
            Clear all
          </button>
          <button
            type="button"
            onClick={useWholePages}
            disabled={busy || !keptPages.length || undatedAll.length > 0}
            title={undatedAll.length ? 'Every kept page needs a date first' : undefined}
          >
            {excluded.size ? `Use ${keptPages.length} whole page${keptPages.length === 1 ? '' : 's'}` : 'Use whole pages'}
          </button>
          <button
            type="button"
            className="primary"
            onClick={() => onStart(keptZones(), datesByPage)}
            disabled={!totalZones || busy || undatedZoned.length > 0}
            title={
              undatedZoned.length
                ? `Add a date to page${undatedZoned.length === 1 ? '' : 's'} ${undatedZoned.join(', ')}`
                : undefined
            }
          >
            {busy
              ? 'Starting…'
              : `Continue with ${totalZones} zone${totalZones === 1 ? '' : 's'}`}
          </button>
        </div>
      </header>

      <p className="hint">
        <strong>Drag</strong> on a page to draw a zone. Each zone becomes a page
        of its own, numbered in the order shown; <strong>click</strong> a zone and
        press <kbd>Delete</kbd> to remove it. Pages without a zone are left out
        {totalZones > 0 &&
          ` — ${totalPages - pagesWithZones} of ${totalPages} will be skipped`}
        ; <strong>Exclude page</strong> leaves a page out even when keeping whole pages. Every page you keep needs a <strong>date</strong>; each zone takes the
        date of its page, and you can still change it during review.
      </p>

      <ol className="zone-pages">
        {pageNums.map((pageNum) => {
          const isExcluded = excluded.has(pageNum)
          const zones = zonesByPage[pageNum] ?? []
          const needsDate = !isExcluded && zones.length > 0 && !datesByPage[pageNum]
          return (
            <li key={pageNum} className={isExcluded ? 'zone-page zone-page--excluded' : 'zone-page'}>
              <div className="zone-page__head">
                <span>
                  Page {pageNum}
                  <span className={zones.length && !isExcluded ? 'found' : 'muted'}>
                    {' · '}
                    {isExcluded
                      ? 'excluded'
                      : zones.length
                        ? `${zones.length} zone${zones.length === 1 ? '' : 's'}`
                        : 'skipped'}
                  </span>
                </span>
                <span className="zone-page__tools">
                  <button type="button" onClick={() => toggleExcluded(pageNum)} aria-pressed={isExcluded}>
                    {isExcluded ? 'Include page' : 'Exclude page'}
                  </button>
                  {!isExcluded && (
                    <>
                      <label className={`date-field${needsDate ? ' date-field--missing' : ''}`}>
                        {/* Said here rather than in a banner above the pages: a
                            banner would appear on the first zone drawn and shove
                            the page down mid-drag. */}
                        <span>{needsDate ? 'Date required' : 'Date'}</span>
                        <input
                          type="date"
                          value={datesByPage[pageNum] ?? ''}
                          required={zones.length > 0}
                          aria-invalid={needsDate}
                          onChange={(event) => setDate(pageNum, event.target.value)}
                        />
                      </label>
                      <button type="button" onClick={() => setZones(pageNum, [...zones, WHOLE_PAGE])}>
                        Whole page
                      </button>
                      <button
                        type="button"
                        onClick={() => setZones(pageNum, [])}
                        disabled={!zones.length}
                      >
                        Clear
                      </button>
                    </>
                  )}
                </span>
              </div>
              <div className="zone-page__canvas">
                <BoxCanvas
                  imageUrl={pageImageUrl(jobId, pageNum, PREVIEW_DPI)}
                  boxes={isExcluded ? [] : zones}
                  onChange={(next) => setZones(pageNum, next)}
                  variant="zone"
                  tags={tagsByPage[pageNum]}
                  lazy
                />
              </div>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
