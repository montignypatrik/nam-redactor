import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  // Relative asset URLs so the built bundle works when served from the
  // FastAPI app at the site root.
  base: './',
  plugins: [react()],
  build: {
    // Committed to the repo so `pip install` yields a working app with no
    // node toolchain. A CI check rebuilds and diffs to catch staleness.
    outDir: '../nam_redactor/web/dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      // changeOrigin must stay false: the backend rejects cross-origin
      // mutations by comparing the Origin against the Host header, and
      // rewriting Host to :8000 while the browser sends Origin :5173 would
      // look like a cross-site request.
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: false },
    },
  },
})
