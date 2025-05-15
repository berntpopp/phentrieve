import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { visualizer } from 'rollup-plugin-visualizer'
import viteCompression from 'vite-plugin-compression'
import viteImagemin from 'vite-plugin-imagemin'
import iconOptimizer from './vite-icon-optimizer'

export default defineConfig({
  plugins: [
    vue(),
    iconOptimizer(),
    viteCompression({
      algorithm: 'brotliCompress',
      ext: '.br',
      threshold: 1024, // Only compress files larger than 1KB
      compressionOptions: { level: 11 }, // Maximum compression level
      deleteOriginFile: false // Keep original files
    }),
    viteImagemin({
      gifsicle: { optimizationLevel: 7 },
      mozjpeg: { quality: 80 },
      pngquant: { quality: [0.7, 0.8] },
      svgo: true
    }),
    visualizer({
      filename: 'dist/stats.html',
      open: false,
      gzipSize: true,
      brotliSize: true,
    }),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  build: {
    target: 'es2015',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    },
    rollupOptions: {
      output: {
        manualChunks: {
          'vuetify': ['vuetify'],
          'vendor': ['vue', 'vue-router', 'pinia'],
          'components': [
            './src/components/DisclaimerDialog.vue',
            './src/components/LogViewer.vue',
            './src/components/ResultsDisplay.vue'
          ]
        }
      }
    },
    chunkSizeWarningLimit: 600
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true
      }
    },
    // Fix MIME type issues
    fs: {
      strict: false
    },
    // Properly configure headers for modules
    headers: {
      '*.js': {
        'Content-Type': 'application/javascript'
      },
      '*.mjs': {
        'Content-Type': 'application/javascript'
      },
      '*.vue': {
        'Content-Type': 'application/javascript'
      }
    }
  }
})
