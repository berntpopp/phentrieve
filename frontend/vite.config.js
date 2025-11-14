import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { configDefaults } from 'vitest/config'
import { visualizer } from 'rollup-plugin-visualizer'
import viteCompression from 'vite-plugin-compression'
import viteImagemin from 'vite-plugin-imagemin'
import iconOptimizer from './vite-icon-optimizer'
import commonjs from '@rollup/plugin-commonjs'

export default defineConfig({  
  optimizeDeps: {
    include: ['google-protobuf'],
    esbuildOptions: {
      // Node.js global to browser globalThis
      define: {
        global: 'globalThis'
      }
    }
  },
  plugins: [
    vue(),
    commonjs({
      transformMixedEsModules: true,
      include: [/node_modules/]
    }),
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
    commonjsOptions: {
      transformMixedEsModules: true,
      include: [/node_modules/],
      exclude: [/\.mjs$/]
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
    port: 5173,
    strictPort: true,
    // API proxy for local development
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        // Improve logging for debugging
        configure: (proxy) => {
          proxy.on('error', (err) => {
            console.log('[Proxy Error]', err)
          })
          proxy.on('proxyReq', (proxyReq, req) => {
            console.log('[Proxy Request]', req.method, req.url)
          })
        }
      }
    },
    // HMR configuration (optimized for fast refresh)
    hmr: {
      overlay: true, // Show errors in browser overlay
      timeout: 30000 // Increase timeout for slower connections
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
    },
    // Watch options for better HMR performance
    watch: {
      usePolling: false, // Use native file system events (faster)
      interval: 100
    }
  },
  test: {
    globals: true,
    environment: 'happy-dom',
    setupFiles: './src/test/setup.js',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        ...configDefaults.coverage.exclude,
        'src/test/**',
        '**/*.config.js',
        '**/dist/**'
      ]
    },
    exclude: [...configDefaults.exclude, 'e2e/*']
  }
})
