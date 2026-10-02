import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('error', (err, _req, res) => {
            if (res && !res.headersSent) {
              res.writeHead(503, {
                'Content-Type': 'application/json'
              });
              res.end(JSON.stringify({
                status: 'error',
                message: 'FastAPI Backend is starting or offline at http://127.0.0.1:8001.',
                error: err.code
              }));
            }
          });
        }
      }
    }
  }
})
