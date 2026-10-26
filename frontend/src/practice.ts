import type { Practice } from './types'

/** Licence, group and sector: optional, kept as text (leading zeros stay). */
export const PRACTICE_FIELDS: { key: keyof Practice; label: string; placeholder: string }[] = [
  { key: 'licence_number', label: 'Licence #', placeholder: 'e.g. 1-23456-7' },
  { key: 'group_number', label: 'Group #', placeholder: 'e.g. 01234' },
  { key: 'sector_number', label: 'Sector #', placeholder: 'e.g. 2' },
]

/** Letters, digits, spaces and dashes -- the same rule the server applies. */
export const PRACTICE_PATTERN = /^[A-Za-z0-9][A-Za-z0-9 -]{0,29}$/

export const invalidPractice = (practice: Practice): (keyof Practice)[] =>
  PRACTICE_FIELDS.filter(({ key }) => {
    const value = practice[key].trim()
    return value !== '' && !PRACTICE_PATTERN.test(value)
  }).map(({ key }) => key)

/** "Licence 1-23456-7 · Group 01234 · Sector 2", skipping blanks. */
export function practiceSummary(practice: Practice): string {
  const parts = [
    practice.licence_number && `Licence ${practice.licence_number}`,
    practice.group_number && `Group ${practice.group_number}`,
    practice.sector_number && `Sector ${practice.sector_number}`,
  ].filter(Boolean)
  return parts.length ? parts.join(' · ') : 'No licence, group or sector number'
}
