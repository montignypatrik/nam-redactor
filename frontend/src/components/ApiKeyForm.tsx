import { useState } from 'react'
import { removeApiKey, saveApiKey } from '../api'
import type { ExtractionConfig } from '../types'

interface Props {
  config: ExtractionConfig
  onChanged: (config: ExtractionConfig) => void
}

/** Add, replace or remove the OpenRouter key saved in the app. The key goes
 *  up once and is never sent back: the server only reports where it is from. */
export default function ApiKeyForm({ config, onChanged }: Props) {
  const [open, setOpen] = useState(!config.configured)
  const [key, setKey] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const act = async (work: () => Promise<ExtractionConfig>) => {
    setBusy(true)
    setError(null)
    try {
      const next = await work()
      setKey('')
      setOpen(!next.configured)
      onChanged(next)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not save the key.')
    } finally {
      setBusy(false)
    }
  }

  if (config.key_source === 'environment') return null

  if (!open) {
    return (
      <p className="muted api-key">
        OpenRouter API key saved on this computer.{' '}
        <button type="button" className="link" onClick={() => setOpen(true)}>
          Replace
        </button>{' '}
        <button
          type="button"
          className="link"
          disabled={busy}
          onClick={() => {
            if (window.confirm('Remove the saved OpenRouter API key?')) void act(removeApiKey)
          }}
        >
          Remove
        </button>
      </p>
    )
  }

  return (
    <form
      className="api-key"
      onSubmit={(event) => {
        event.preventDefault()
        void act(() => saveApiKey(key))
      }}
    >
      <label>
        <span>OpenRouter API key</span>
        <input
          type="password"
          autoComplete="off"
          spellCheck={false}
          placeholder="sk-or-v1-…"
          value={key}
          onChange={(event) => setKey(event.target.value)}
        />
      </label>
      <button type="submit" className="primary" disabled={busy || !key.trim()}>
        Save key
      </button>
      {config.configured && (
        <button type="button" onClick={() => setOpen(false)} disabled={busy}>
          Cancel
        </button>
      )}
      <span className="muted">Saved only on this computer; never shown again.</span>
      {error && <span className="error">{error}</span>}
    </form>
  )
}
