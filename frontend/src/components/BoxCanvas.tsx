import { useCallback, useEffect, useRef, useState } from 'react'
import type { Bounds, Box } from '../types'

interface Draft {
  x1: number
  y1: number
  x2: number
  y2: number
}

interface Props {
  imageUrl: string
  boxes: Box[]
  onChange: (boxes: Box[]) => void
  onImageLoad?: () => void
  /** ``redact`` draws solid blackouts; ``zone`` and ``row`` draw see-through
   *  outlines so the content being selected stays visible. */
  variant?: 'redact' | 'zone' | 'row'
  /** A tag shown in each box's corner, by index. Outline variants only. */
  tags?: string[]
  /** Defer loading the page image until it scrolls near the viewport. */
  lazy?: boolean
  /** Extra class names per box, by index. */
  boxClassName?: (index: number) => string
  /** When given, a finished drag is reported here instead of being appended
   *  to ``boxes`` through ``onChange``. */
  onDraw?: (rect: Draft) => void
  /** When given, removing a box is reported here instead of ``onChange``. */
  onRemove?: (index: number) => void
  /** Called when a box is selected (clicked) or the selection is cleared. */
  onSelect?: (index: number | null) => void
  /** Drags start anywhere, even on top of a box, instead of selecting it --
   *  for drawing a region inside another one. */
  drawThrough?: boolean
  /** Sections of the page to highlight: everything outside them is greyed
   *  out, and each is outlined with its label. Display only. */
  regions?: { bounds: Bounds; label: string }[]
  /** A click (a drag too small to be a box), in page-normalized coordinates. */
  onPoint?: (point: { x: number; y: number }) => void
}

/** Drags smaller than this fraction of a page dimension are treated as clicks. */
const MIN_SIZE = 0.004

const clamp01 = (value: number) => Math.min(1, Math.max(0, value))

/**
 * Page raster with editable blackout rectangles.
 *
 * Boxes are stored normalized (0..1) and rendered as percentage-positioned
 * overlays. Because the image and the overlays scale with the same element,
 * zoom, responsive resizing and device pixel ratio all cancel out -- there is
 * no scale factor to keep in sync. A drag is inverted the same way, against
 * the image's live bounding rect.
 */
export default function BoxCanvas({
  imageUrl,
  boxes,
  onChange,
  onImageLoad,
  variant = 'redact',
  tags,
  lazy = false,
  boxClassName,
  onDraw,
  onRemove,
  onSelect,
  drawThrough = false,
  regions,
  onPoint,
}: Props) {
  const rootRef = useRef<HTMLDivElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const dragStart = useRef<{ x: number; y: number } | null>(null)
  const [draft, setDraft] = useState<Draft | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [loaded, setLoaded] = useState(false)

  const toNormalized = useCallback((clientX: number, clientY: number) => {
    const image = imgRef.current
    if (!image) return null
    const rect = image.getBoundingClientRect()
    if (!rect.width || !rect.height) return null
    return {
      x: clamp01((clientX - rect.left) / rect.width),
      y: clamp01((clientY - rect.top) / rect.height),
    }
  }, [])

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return
    const point = toNormalized(event.clientX, event.clientY)
    if (!point) return
    event.currentTarget.setPointerCapture(event.pointerId)
    dragStart.current = point
    setSelected(null)
    onSelect?.(null)
    setDraft({ x1: point.x, y1: point.y, x2: point.x, y2: point.y })
  }

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = dragStart.current
    if (!start) return
    const point = toNormalized(event.clientX, event.clientY)
    if (!point) return
    setDraft({
      x1: Math.min(start.x, point.x),
      y1: Math.min(start.y, point.y),
      x2: Math.max(start.x, point.x),
      y2: Math.max(start.y, point.y),
    })
  }

  const handlePointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = dragStart.current
    dragStart.current = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    const pending = draft
    setDraft(null)
    if (!start || !pending) return
    // A tap or a negligible drag is a click, never a sliver of a box.
    if (pending.x2 - pending.x1 < MIN_SIZE || pending.y2 - pending.y1 < MIN_SIZE) {
      onPoint?.(start)
      return
    }
    if (onDraw) onDraw(pending)
    else onChange([...boxes, { ...pending, confidence: 1 }])
  }

  const removeBox = (index: number) => {
    if (onRemove) onRemove(index)
    else onChange(boxes.filter((_, i) => i !== index))
    setSelected(null)
  }

  // Delete/Backspace removes the selected box, but only when focus is not in a
  // control that needs the key itself.
  useEffect(() => {
    if (selected === null) return
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return
      if (event.key === 'Delete' || event.key === 'Backspace') {
        event.preventDefault()
        removeBox(selected)
      }
      if (event.key === 'Escape') setSelected(null)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  })

  useEffect(() => {
    setSelected(null)
    setLoaded(false)
  }, [imageUrl])

  // With several canvases on screen, a selection must not outlive a click
  // elsewhere -- otherwise Delete would remove a box from every canvas that
  // still holds one.
  useEffect(() => {
    if (selected === null) return
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setSelected(null)
    }
    window.addEventListener('pointerdown', onPointerDown)
    return () => window.removeEventListener('pointerdown', onPointerDown)
  }, [selected])

  const rendered = draft
    ? [...boxes, { ...draft, confidence: 1 }]
    : boxes

  return (
    <div
      ref={rootRef}
      className={`canvas${variant !== 'redact' ? ' canvas--zones' : ''}${loaded ? '' : ' canvas--loading'}`}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      <img
        ref={imgRef}
        src={imageUrl}
        alt="Page preview"
        draggable={false}
        loading={lazy ? 'lazy' : undefined}
        onLoad={() => {
          setLoaded(true)
          onImageLoad?.()
        }}
      />
      {regions && (
        <>
          <svg className="regions" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
            <path
              className="regions__outside"
              fillRule="evenodd"
              d={
                'M0 0H1V1H0Z' +
                regions
                  .map(({ bounds: b }) => `M${b.x1} ${b.y1}H${b.x2}V${b.y2}H${b.x1}Z`)
                  .join('')
              }
            />
            {regions.map(({ bounds: b }, index) => (
              <rect
                key={index}
                className="regions__edge"
                x={b.x1}
                y={b.y1}
                width={b.x2 - b.x1}
                height={b.y2 - b.y1}
                vectorEffect="non-scaling-stroke"
              />
            ))}
          </svg>
          {regions.map(({ bounds: b, label }, index) => (
            <span
              key={index}
              className="regions__label"
              style={{ right: `${(1 - b.x2) * 100}%`, top: `${b.y1 * 100}%` }}
            >
              {label}
            </span>
          ))}
        </>
      )}
      {rendered.map((box, index) => {
        const isDraft = Boolean(draft) && index === rendered.length - 1
        const isSelected = !isDraft && index === selected
        return (
          <div
            key={isDraft ? 'draft' : `box-${index}`}
            className={`box${variant !== 'redact' ? ' box--zone' : ''}${variant === 'row' ? ' box--row' : ''}${box.kind === 'name' ? ' box--name' : ''}${!isDraft && boxClassName ? ` ${boxClassName(index)}` : ''}${isSelected ? ' box--selected' : ''}${isDraft ? ' box--draft' : ''}`}
            style={{
              left: `${box.x1 * 100}%`,
              top: `${box.y1 * 100}%`,
              width: `${(box.x2 - box.x1) * 100}%`,
              height: `${(box.y2 - box.y1) * 100}%`,
            }}
            title={
              isDraft
                ? undefined
                : `${variant === 'zone' ? 'Zone' : variant === 'row' ? 'Region' : box.kind === 'name' ? 'Name' : 'NAM'} — click to select, then press Delete to remove`
            }
            onPointerDown={(event) => {
              if (drawThrough) return  // let the canvas start a drag here
              // Keep the canvas from starting a new draft underneath this box.
              event.stopPropagation()
              if (!isDraft) {
                setSelected(index)
                onSelect?.(index)
              }
            }}
          >
            {!isDraft && variant !== 'redact' && tags?.[index] && (
              <span className="box__tag">{tags[index]}</span>
            )}
            {!isDraft && variant === 'redact' && box.patient_id && (
              <span className="box__label">{box.patient_id}</span>
            )}
            {isSelected && (
              <button
                type="button"
                className="box__remove"
                aria-label="Remove this box"
                onPointerDown={(event) => event.stopPropagation()}
                onClick={(event) => {
                  event.stopPropagation()
                  removeBox(index)
                }}
              >
                ×
              </button>
            )}
          </div>
        )
      })}
    </div>
  )
}
