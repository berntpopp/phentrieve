import { fileURLToPath, URL } from 'node:url'
import { readFileSync } from 'fs'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { configDefaults } from 'vitest/config'
import { visualizer } from 'rollup-plugin-visualizer'
import viteCompression from 'vite-plugin-compression'
// vite-plugin-imagemin removed due to unmaintained status and 29+ security vulnerabilities
// Images are served as-is; consider vite-imagetools for future optimization needs
import iconOptimizer from './vite-icon-optimizer'

// Read version from package.json at build time
const packageJson = JSON.parse(
  readFileSync(new URL('./package.json', import.meta.url), 'utf-8')
)

export default defineConfig(({ mode }) => ({
  // Remove Vue devtools and debug code from production builds
  define: {
    __VUE_PROD_DEVTOOLS__: false,
    __VUE_PROD_HYDRATION_MISMATCH_DETAILS__: false,
    __APP_VERSION__: JSON.stringify(packageJson.version), // Inject version at build time
    global: 'globalThis',
  },
  optimizeDeps: {
    include: ['google-protobuf'],
  },
  plugins: [
    vue(),
    iconOptimizer(),
    // Brotli compression and bundle visualizer are deploy-time concerns.
    // Skip in CI to save 10-30s per build; keep for local builds so the
    // dev can inspect stats.html and see the actual deploy-ready payload.
    !process.env.CI &&
      viteCompression({
        algorithm: 'brotliCompress',
        ext: '.br',
        threshold: 1024, // Only compress files larger than 1KB
        compressionOptions: { level: 11 }, // Maximum compression level
        deleteOriginFile: false, // Keep original files
      }),
    // viteImagemin removed - see import comment above
    !process.env.CI &&
      visualizer({
        filename: 'dist/stats.html',
        open: false,
        gzipSize: true,
        brotliSize: true,
      }),
  ].filter(Boolean),
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  // Drop console/debugger in production builds only to preserve the dev
  // debugging experience.
  ...(mode === 'production' ? { esbuild: { drop: ['console', 'debugger'] } } : {}),
  build: {
    target: 'es2015',
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            {
              name: 'vuetify',
              test: /[\\/]node_modules[\\/]vuetify[\\/]/
            },
            {
              name: 'vendor',
              test: /[\\/]node_modules[\\/](vue|vue-router|pinia)[\\/]/
            },
            {
              name: 'components',
              test: /[\\/]src[\\/]components[\\/](DisclaimerDialog|LogViewer|ResultsDisplay)\.vue$/
            }
          ]
        }
      }
    },
    chunkSizeWarningLimit: 600,
    reportCompressedSize: false,  // skip gzip calc on every build (CI speedup)
  },
  server: {
    // Custom port 5734 (matches API 8734 pattern - HPOD project ports)
    // Avoids conflicts with other Vite projects (default 5173)
    port: 5734,
    strictPort: true, // Fail fast if port is in use
    // API proxy for local development
    proxy: {
      '/api': {
        target: 'http://localhost:8734', // Match custom API port
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
    // Enable polling for WSL2 with Windows mounted files (/mnt/c/)
    watch: {
      usePolling: true, // Required for WSL2 cross-filesystem watching
      interval: 100 // Poll every 100ms for changes
    }
  },
  test: {
    globals: true,
    environment: 'happy-dom',
    pool: 'threads',
    setupFiles: './src/test/setup.js',
    // Vitest deps inlining — Vuetify's CSS imports must be processed
    // by Vite, not left to Node's ESM loader (which rejects .css).
    server: {
      deps: {
        inline: ['vuetify'],
      },
    },
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
}))
