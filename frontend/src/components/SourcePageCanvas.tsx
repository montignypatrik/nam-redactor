import { useMemo } from 'react'
import type { Bounds, Box } from '../types'
import { clipTo, pickZone, toSource, toWorking } from '../zoneMath'
import BoxCanvas from './BoxCanvas'

/** One section of the original page: its working page and where it sits. */
export interface SourceZone {
  workingPage: number
  crop: Bounds
  label: string
}

/** A box shown on the original page, with a way back to where it is stored. */
export interface PlacedBox {
  workingPage: number
  index: number
  box: Box
}

interface Props {
  imageUrl: string
  zones: SourceZone[]
  /** Boxes to show, per working page, in that page's own coordinates. */
  boxesByWorking: Record<number, Box[]>
  variant?: 'redact' | 'row'
  /** Tag shown on a box (outline variants). */
  tagFor?: (placed: PlacedBox) => string | undefined
  classFor?: (placed: PlacedBox) => string
  /** A drag finished: the rectangle in the working page's coordinates. */
  onDraw: (workingPage: number, rect: Bounds) => void
  onRemove: (workingPage: number, index: number) => void
  onSelect?: (placed: PlacedBox | null) => void
  /** A drag fell outside every section (or outside ``drawWithin``). */
  onRejected?: () => void
  /** Only accept drags inside this working page's section. */
  drawWithin?: number
  drawThrough?: boolean
  onImageLoad?: () => void
  /** A click inside a section (not a drag). */
  onSectionClick?: (workingPage: number) => void
}

/** Anything thinner than this, after clipping to a section, is a stray drag. */
const MIN_SIZE = 0.004

/**
 * The original page with every section drawn in place.
 *
 * Boxes are still stored per section (working page), so saving, output and
 * evidence are unchanged: this only maps them onto the original for display,
 * and maps each new drag back into the section it was drawn in. Everything
 * outside the sections is greyed out -- it is not part of the document.
 */
export default function SourcePageCanvas({
  imageUrl,
  zones,
  boxesByWorking,
  variant = 'redact',
  tagFor,
  classFor,
  onDraw,
  onRemove,
  onSelect,
  onRejected,
  drawWithin,
  drawThrough,
  onImageLoad,
  onSectionClick,
}: Props) {
  const placed = useMemo(() => {
    const out: PlacedBox[] = []
    for (const zone of zones) {
      ;(boxesByWorking[zone.workingPage] ?? []).forEach((box, index) => {
        out.push({ workingPage: zone.workingPage, index, box: toSource(box, zone.crop) })
      })
    }
    return out
  }, [zones, boxesByWorking])

  const handleDraw = (rect: Bounds) => {
    const candidates =
      drawWithin === undefined ? zones : zones.filter((zone) => zone.workingPage === drawWithin)
    const zone = pickZone(rect, candidates)
    const clipped = zone ? clipTo(rect, zone.crop) : null
    if (!zone || !clipped || clipped.x2 - clipped.x1 < MIN_SIZE || clipped.y2 - clipped.y1 < MIN_SIZE) {
      onRejected?.()
      return
    }
    onDraw(zone.workingPage, toWorking(clipped, zone.crop))
  }

  return (
    <BoxCanvas
      imageUrl={imageUrl}
      boxes={placed.map((item) => item.box)}
      onChange={() => undefined}
      variant={variant}
      regions={zones.map((zone) => ({ bounds: zone.crop, label: zone.label }))}
      tags={tagFor ? placed.map((item) => tagFor(item) ?? '') : undefined}
      boxClassName={classFor ? (index) => classFor(placed[index]) : undefined}
      onDraw={handleDraw}
      onRemove={(index) => onRemove(placed[index].workingPage, placed[index].index)}
      onSelect={(index) => onSelect?.(index === null ? null : placed[index])}
      drawThrough={drawThrough}
      onImageLoad={onImageLoad}
      onPoint={({ x, y }) => {
        const zone = zones.find(({ crop: c }) => x >= c.x1 && x <= c.x2 && y >= c.y1 && y <= c.y2)
        if (zone) onSectionClick?.(zone.workingPage)
      }}
    />
  )
}

/** An original page and the sections cut from it, in section order. */
export interface SourceGroup {
  sourcePage: number
  zones: SourceZone[]
}

/**
 * Group working pages by the original page they came from. A page with no
 * provenance (older data) stands alone, full-page.
 */
export function sourceGroups(
  pages: { page_num: number; source_page: number | null; source_crop: Bounds | null }[],
): SourceGroup[] {
  const groups = new Map<number, SourceZone[]>()
  for (const page of pages) {
    const key = page.source_page ?? page.page_num
    const zones = groups.get(key) ?? []
    zones.push({
      workingPage: page.page_num,
      crop: page.source_crop ?? { x1: 0, y1: 0, x2: 1, y2: 1 },
      label: `#${page.page_num}`,
    })
    groups.set(key, zones)
  }
  return [...groups.entries()]
    .sort(([a], [b]) => a - b)
    .map(([sourcePage, zones]) => ({ sourcePage, zones }))
}
