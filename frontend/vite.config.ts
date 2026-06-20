import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy /api and /data to the FastAPI backend so the browser sees one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8077',
      '/data': 'http://127.0.0.1:8077',
    },
  },
})
