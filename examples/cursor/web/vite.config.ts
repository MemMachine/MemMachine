import path from 'path'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Load env file based on `mode` in the current working directory.
  const env = loadEnv(mode, process.cwd(), '')
  
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: 8000,
      proxy: {
        '/api': {
          target: env.VITE_BACKEND_URL || 'http://localhost:8001',
          changeOrigin: true,
          secure: false,
          // rewrite: (path) => path.replace(/^\/uni-key/, '')
        },
      }
    }
  }
})
