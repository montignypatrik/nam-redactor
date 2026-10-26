import type { Bounds } from './types'

/**
 * Mapping between a section (working page) and its original page.
 *
 * ``crop`` is the section's footprint on the original page, normalized. The
 * formulas match ``working_to_source`` in nam_redactor/provenance.py, so the
 * review screen and the exported manifest place things identically.
 */

export const FULL_PAGE: Bounds = { x1: 0, y1: 0, x2: 1, y2: 1 }

export function toSource<T extends Bounds>(box: T, crop: Bounds): T {
  const w = crop.x2 - crop.x1
  const h = crop.y2 - crop.y1
  return {
    ...box,
    x1: crop.x1 + box.x1 * w,
    y1: crop.y1 + box.y1 * h,
    x2: crop.x1 + box.x2 * w,
    y2: crop.y1 + box.y2 * h,
  }
}

export function toWorking<T extends Bounds>(box: T, crop: Bounds): T {
  const w = crop.x2 - crop.x1 || 1
  const h = crop.y2 - crop.y1 || 1
  return {
    ...box,
    x1: (box.x1 - crop.x1) / w,
    y1: (box.y1 - crop.y1) / h,
    x2: (box.x2 - crop.x1) / w,
    y2: (box.y2 - crop.y1) / h,
  }
}

export function clipTo(rect: Bounds, crop: Bounds): Bounds | null {
  const clipped = {
    x1: Math.max(rect.x1, crop.x1),
    y1: Math.max(rect.y1, crop.y1),
    x2: Math.min(rect.x2, crop.x2),
    y2: Math.min(rect.y2, crop.y2),
  }
  return clipped.x2 > clipped.x1 && clipped.y2 > clipped.y1 ? clipped : null
}

const area = (b: Bounds | null) => (b ? (b.x2 - b.x1) * (b.y2 - b.y1) : 0)

/** The zone a drawn rectangle belongs to: the one it overlaps most. */
export function pickZone<Z extends { crop: Bounds }>(rect: Bounds, zones: Z[]): Z | null {
  let best: Z | null = null
  let bestArea = 0
  for (const zone of zones) {
    const overlap = area(clipTo(rect, zone.crop))
    if (overlap > bestArea) {
      best = zone
      bestArea = overlap
    }
  }
  return best
}
