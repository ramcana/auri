import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  root: 'renderer',
  plugins: [react()],
  base: './', // Use relative paths for assets
  build: {
    outDir: '../dist'
  },
  server: {
    port: 5173,
    host: true, // Listen on all addresses, including localhost and local network
    strictPort: true, // Fail if port is already in use
    hmr: {
      protocol: 'ws', // Use WebSocket for HMR
      host: 'localhost',
      port: 5173
    },
    proxy: {
      // Properly proxy WebSocket connections to our backend
      '/ws': {
        target: 'http://127.0.0.1:8080',
        ws: true,
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path
      },
      // Proxy API requests
      '/api': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path
      }
    }
  }
})
