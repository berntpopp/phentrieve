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
