import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/Techin513_Final/',
  build: {
    outDir: 'dist',
    rollupOptions: {
      output: {
        manualChunks: {
          plotly: ['plotly.js-dist-min'],
          vendor: ['react', 'react-dom', 'framer-motion'],
        },
      },
    },
  },
})
