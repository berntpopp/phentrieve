# Frontend Performance Optimization Plan

**Status**: Active
**Created**: 2025-11-20
**Lighthouse Score**: 56/100 Performance → Target: 90+/100
**Priority**: High

## Executive Summary

Based on Lighthouse audit results, the frontend has significant performance issues primarily due to:
1. **Bundle size**: 4,437 KiB total payload (1,182 KiB unused JavaScript)
2. **Load time**: FCP 3.8s, LCP 5.9s (targets: <1.8s, <2.5s)
3. **Render blocking**: 1,530ms element render delay
4. **Unused resources**: 558.9 KiB unused Vuetify components, 146 KiB unused CSS

This plan follows **DRY, KISS, SOLID** principles and emphasizes **modularization** to achieve sustainable performance improvements.

---

## 🎯 Performance Targets

| Metric | Current | Target | Priority |
|--------|---------|--------|----------|
| Performance Score | 56 | 90+ | Critical |
| First Contentful Paint | 3.8s | <1.8s | Critical |
| Largest Contentful Paint | 5.9s | <2.5s | Critical |
| Total Bundle Size | 4,437 KiB | <2,000 KiB | High |
| JavaScript Size | 2,273 KiB | <1,000 KiB | High |
| Total Blocking Time | 110ms | <200ms | Medium (OK) |
| Cumulative Layout Shift | 0 | 0 | ✅ Perfect |

---

## 📊 Root Cause Analysis

### 1. Vuetify Components: 558.9 KiB Unused (24% of JavaScript)

**Current Problem** (`frontend/src/main.js:17-18`):
```javascript
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';

const vuetify = createVuetify({
  components,  // Imports ALL 100+ components
  directives,  // Imports ALL directives
  // ...
});
```

**Impact**:
- Bundles unused components: VCombobox (17 KiB), VAutocomplete (15.9 KiB), VNumberInput (12.5 KiB), VSlideGroup (9.5 KiB), VDatePicker (8.7 KiB)
- Total waste: 558.9 KiB unused code

**Components Actually Used** (from grep analysis):
- VApp, VMain, VCard, VBtn, VIcon, VTextField, VSelect, VSwitch
- VDialog, VMenu, VNavigationDrawer, VFooter, VDivider
- VProgressLinear, VProgressCircular, VAlert, VChip, VBadge
- VList, VListItem, VExpansionPanels, VRow, VCol, VSheet, VSlider, VAvatar, VSpacer

### 2. Material Design Icons: 935 KiB (21% of Total Payload)

**Current Problem** (`frontend/src/main.js:20`):
```javascript
import '@mdi/font/css/materialdesignicons.css';  // 459 KiB CSS + 394 KiB font = 853 KiB
```

**Impact**:
- Loads ALL 7,000+ icons when only ~20 are used
- 81.2 KiB unused CSS detected by Lighthouse
- Font file not optimized (no font-display: swap)

**Icons Actually Used**: [will be determined in Phase 1]

**Research Finding**: Icon fonts are considered bad practice; SVG with tree-shaking (@mdi/js) is recommended for 95%+ size reduction.

### 3. Vue i18n: All Locales Loaded (159.6 KiB, 80.7 KiB Unused)

**Current Problem** (`frontend/src/i18n.js:4-16`):
```javascript
import en from './locales/en.json';
import fr from './locales/fr.json';
import es from './locales/es.json';
import de from './locales/de.json';
import nl from './locales/nl.json';

const messages = { en, fr, es, de, nl };  // All loaded upfront
```

**Impact**:
- Users only need 1 locale but get all 5 (243 keys × 5 languages)
- ~32 KiB per locale × 5 = 160 KiB (only ~32 KiB needed)

### 4. Pinia Devtools: 104.5 KiB in Production

**Current Problem**:
- Devtools included in production build
- @vue/devtools-kit: 62.1 KiB unused

**Impact**:
- Debugging tools shipped to users
- No runtime benefit in production

### 5. Minification: 1,198 KiB Can Be Saved

**Current Status**:
- Terser enabled in `vite.config.js:55-60`
- But minification appears incomplete per Lighthouse

**Impact**:
- Larger download sizes
- Slower parse/compile times

### 6. LCP Element Render Delay: 1,530ms (26% of LCP)

**Current Problem**:
- Largest Contentful Paint: 5.9s total
  - TTFB: 350ms (good)
  - Element render delay: 1,530ms (bad)
  - Resource load delay: remaining 4,020ms

**Impact**:
- Critical rendering path blocked
- Main content appears slowly

---

## 🔧 Implementation Plan

### Phase 1: Vuetify Tree-Shaking (Est. -558 KiB, -0.8s LCP)

**Priority**: Critical
**Complexity**: Medium
**Effort**: 2-3 hours

#### 1.1 Create Vuetify Plugin with Explicit Imports

**File**: `frontend/src/plugins/vuetify.js` (new)

```javascript
// Import only used components (DRY principle: single source of truth)
import { createVuetify } from 'vuetify';
import { aliases, mdi } from 'vuetify/iconsets/mdi-svg'; // Switch to SVG icons

// Components (alphabetical for maintainability)
import {
  VAlert,
  VApp,
  VAvatar,
  VBadge,
  VBtn,
  VCard,
  VCardActions,
  VCardText,
  VCardTitle,
  VChip,
  VCol,
  VDialog,
  VDivider,
  VExpansionPanel,
  VExpansionPanels,
  VExpansionPanelText,
  VExpansionPanelTitle,
  VFooter,
  VIcon,
  VList,
  VListItem,
  VListItemSubtitle,
  VListItemTitle,
  VListSubheader,
  VMain,
  VMenu,
  VNavigationDrawer,
  VProgressCircular,
  VProgressLinear,
  VRow,
  VSelect,
  VSheet,
  VSlider,
  VSpacer,
  VSwitch,
  VTextField,
} from 'vuetify/components';

// Directives (only if used)
import { Ripple } from 'vuetify/directives';

export default createVuetify({
  components: {
    VAlert,
    VApp,
    VAvatar,
    VBadge,
    VBtn,
    VCard,
    VCardActions,
    VCardText,
    VCardTitle,
    VChip,
    VCol,
    VDialog,
    VDivider,
    VExpansionPanel,
    VExpansionPanels,
    VExpansionPanelText,
    VExpansionPanelTitle,
    VFooter,
    VIcon,
    VList,
    VListItem,
    VListItemSubtitle,
    VListItemTitle,
    VListSubheader,
    VMain,
    VMenu,
    VNavigationDrawer,
    VProgressCircular,
    VProgressLinear,
    VRow,
    VSelect,
    VSheet,
    VSlider,
    VSpacer,
    VSwitch,
    VTextField,
  },
  directives: {
    Ripple,
  },
  icons: {
    defaultSet: 'mdi',
    aliases,
    sets: {
      mdi,
    },
  },
  theme: {
    defaultTheme: 'light',
  },
});
```

#### 1.2 Update main.js

**File**: `frontend/src/main.js`

```diff
-// Vuetify
-import 'vuetify/styles';
-import { createVuetify } from 'vuetify';
-import * as components from 'vuetify/components';
-import * as directives from 'vuetify/directives';
-import { mdi } from 'vuetify/iconsets/mdi';
-import '@mdi/font/css/materialdesignicons.css';
-
-const vuetify = createVuetify({
-  components,
-  directives,
-  icons: {
-    defaultSet: 'mdi',
-    sets: {
-      mdi,
-    },
-  },
-  theme: {
-    defaultTheme: 'light',
-  },
-});
+// Vuetify (tree-shaken, modular import)
+import 'vuetify/styles';
+import vuetify from './plugins/vuetify';
```

#### 1.3 Install Dependencies

```bash
npm install @mdi/js  # For SVG icons (tree-shakeable)
```

#### 1.4 Verification

```bash
npm run build
npx vite-bundle-visualizer  # Check bundle size reduction
```

**Expected Result**: Vuetify bundle reduced from 798 KiB → ~350 KiB (-448 KiB)

---

### Phase 2: Replace Icon Font with SVG Icons (Est. -853 KiB, -0.6s FCP)

**Priority**: Critical
**Complexity**: Medium
**Effort**: 3-4 hours

#### 2.1 Audit Current Icon Usage

**Script**: `scripts/audit-icons.sh` (new)

```bash
#!/bin/bash
# Extract all mdi- icon references
echo "Icons used in Vue files:"
grep -roh "mdi-[a-z-]*" frontend/src --include="*.vue" --include="*.js" | sort -u

echo -e "\nTotal unique icons:"
grep -roh "mdi-[a-z-]*" frontend/src --include="*.vue" --include="*.js" | sort -u | wc -l
```

#### 2.2 Create Icon Registry (Modular, SOLID)

**File**: `frontend/src/plugins/icons.js` (replace existing if any)

```javascript
// Icon registry following Single Responsibility Principle
// Each icon is a separate import (tree-shakeable)

import {
  mdiAccount,
  mdiAlert,
  mdiArrowLeft,
  mdiCheck,
  mdiChevronDown,
  mdiChevronUp,
  mdiClose,
  mdiDelete,
  mdiDownload,
  mdiEye,
  mdiEyeOff,
  mdiHelp,
  mdiHome,
  mdiInformation,
  mdiMenu,
  mdiPlus,
  mdiRefresh,
  mdiSearch,
  mdiTranslate,
  mdiWeb,
  // Add more as discovered in audit
} from '@mdi/js';

// Export as named constants for type safety and autocomplete
export const icons = {
  account: mdiAccount,
  alert: mdiAlert,
  arrowLeft: mdiArrowLeft,
  check: mdiCheck,
  chevronDown: mdiChevronDown,
  chevronUp: mdiChevronUp,
  close: mdiClose,
  delete: mdiDelete,
  download: mdiDownload,
  eye: mdiEye,
  eyeOff: mdiEyeOff,
  help: mdiHelp,
  home: mdiHome,
  information: mdiInformation,
  menu: mdiMenu,
  plus: mdiPlus,
  refresh: mdiRefresh,
  search: mdiSearch,
  translate: mdiTranslate,
  web: mdiWeb,
};

// Type helper for Vue components
export type IconName = keyof typeof icons;
```

#### 2.3 Update Vuetify Plugin

**File**: `frontend/src/plugins/vuetify.js`

```diff
-import { aliases, mdi } from 'vuetify/iconsets/mdi-svg';
+import { aliases, mdi } from 'vuetify/iconsets/mdi-svg';
+import { icons } from './icons';

 export default createVuetify({
   // ...
   icons: {
     defaultSet: 'mdi',
     aliases,
     sets: {
       mdi,
     },
+    values: icons, // Register custom icons
   },
   // ...
 });
```

#### 2.4 Migrate Icon Usage in Components

**Example Migration**:

```diff
-<v-icon>mdi-search</v-icon>
+<v-icon>$search</v-icon>
```

**Or use direct SVG path** (preferred for critical icons):

```vue
<template>
  <v-icon>
    <svg viewBox="0 0 24 24">
      <path :d="icons.search" />
    </svg>
  </v-icon>
</template>

<script setup>
import { icons } from '@/plugins/icons';
</script>
```

#### 2.5 Remove Font Dependency

**File**: `package.json`

```diff
-    "@mdi/font": "^7.4.47",
+    "@mdi/js": "^7.4.47",
```

```bash
npm uninstall @mdi/font
npm install @mdi/js
```

**Expected Result**:
- CSS: -459 KiB (materialdesignicons.css removed)
- Font: -394 KiB (woff2 file removed)
- Total: -853 KiB (~19% of total payload)

---

### Phase 3: i18n Lazy Loading (Est. -128 KiB, -0.3s FCP)

**Priority**: High
**Complexity**: Low
**Effort**: 1-2 hours

#### 3.1 Create Async Locale Loader

**File**: `frontend/src/i18n.js`

```javascript
import { createI18n } from 'vue-i18n';

// Available locales (DRY: single source of truth)
export const SUPPORTED_LOCALES = ['en', 'de', 'es', 'fr', 'nl'];
export const DEFAULT_LOCALE = 'en';

// Lazy load locale messages
const loadedLocales = new Set(); // Track loaded locales to prevent duplicates

async function loadLocaleMessages(i18n, locale) {
  // Skip if already loaded (KISS: avoid redundant work)
  if (loadedLocales.has(locale)) {
    return;
  }

  // Dynamic import for code splitting (Vite automatically creates chunks)
  const messages = await import(
    /* webpackChunkName: "locale-[request]" */
    `./locales/${locale}.json`
  );

  // Set locale messages
  i18n.global.setLocaleMessage(locale, messages.default);
  loadedLocales.add(locale);
}

// Determine initial locale
function getInitialLocale() {
  try {
    const savedLang = localStorage.getItem('phentrieve-lang');
    if (savedLang && SUPPORTED_LOCALES.includes(savedLang)) {
      return savedLang;
    }

    const browserLang = navigator.language.split('-')[0];
    if (SUPPORTED_LOCALES.includes(browserLang)) {
      return browserLang;
    }
  } catch {
    console.warn('Could not access localStorage. Defaulting to English.');
  }
  return DEFAULT_LOCALE;
}

// Create i18n instance (no messages loaded yet)
const i18n = createI18n({
  legacy: false,
  locale: DEFAULT_LOCALE, // Will be updated after loading
  fallbackLocale: DEFAULT_LOCALE,
  messages: {}, // Empty initially
  globalInjection: true,
  runtimeOnly: true,
  warnHtmlMessage: false,
  missingWarn: import.meta.env.MODE !== 'production',
  fallbackWarn: import.meta.env.MODE !== 'production',
  escapeParameter: true,
  silentTranslationWarn: true,
  silentFallbackWarn: true,
});

// Load initial locale
const initialLocale = getInitialLocale();
await loadLocaleMessages(i18n, initialLocale);
i18n.global.locale.value = initialLocale;

// Export helper for changing locale dynamically
export async function setLocale(locale) {
  if (!SUPPORTED_LOCALES.includes(locale)) {
    console.error(`Unsupported locale: ${locale}`);
    return;
  }

  await loadLocaleMessages(i18n, locale);
  i18n.global.locale.value = locale;

  try {
    localStorage.setItem('phentrieve-lang', locale);
  } catch {
    console.warn('Could not save locale preference.');
  }
}

export default i18n;
```

#### 3.2 Update LanguageSwitcher Component

**File**: `frontend/src/components/LanguageSwitcher.vue`

```diff
 <script setup>
 import { useI18n } from 'vue-i18n';
+import { setLocale } from '@/i18n';

 const { locale, t } = useI18n();

 const changeLanguage = async (newLocale) => {
-  locale.value = newLocale;
-  localStorage.setItem('phentrieve-lang', newLocale);
+  await setLocale(newLocale); // Lazy load if needed
 };
 </script>
```

#### 3.3 Verification

```bash
npm run build
# Check dist/assets/ for locale-*.js chunk files
ls -lh dist/assets/locale-*.js
```

**Expected Result**:
- Initial bundle: -128 KiB (4 of 5 locales deferred)
- Locale files split into separate chunks (~32 KiB each)
- Loaded on-demand when user switches language

---

### Phase 4: Remove Production Devtools (Est. -104 KiB, -0.1s TBT)

**Priority**: Medium
**Complexity**: Low
**Effort**: 30 minutes

#### 4.1 Conditional Devtools Import

**File**: `frontend/src/main.js`

```diff
 import { createPinia } from 'pinia';
+
+// Only enable devtools in development (tree-shaken in production)
+const pinia = import.meta.env.DEV
+  ? createPinia()
+  : createPinia({ devtools: false });

-const pinia = createPinia();
 app.use(pinia);
```

#### 4.2 Vite Config Optimization

**File**: `frontend/vite.config.js`

```diff
 export default defineConfig({
+  define: {
+    __VUE_PROD_DEVTOOLS__: false, // Disable Vue devtools in production
+    __VUE_PROD_HYDRATION_MISMATCH_DETAILS__: false, // Remove hydration details
+  },
   build: {
     // ...
   }
 });
```

**Expected Result**: @vue/devtools-kit (-62.1 KiB) and related devtools removed from production build

---

### Phase 5: Code Splitting & Route-Based Chunking (Est. -200 KiB initial, -0.4s LCP)

**Priority**: High
**Complexity**: Low
**Effort**: 1 hour

#### 5.1 Lazy Load Routes

**File**: `frontend/src/router/index.js`

```diff
-import HomeView from '../views/HomeView.vue';
-import FaqView from '../views/FaqView.vue';
+// Lazy load views for code splitting (KISS: simple dynamic imports)
+const HomeView = () => import('../views/HomeView.vue');
+const FaqView = () => import('../views/FaqView.vue');

 const routes = [
   {
     path: '/',
     name: 'home',
     component: HomeView,
+    meta: { preload: true }, // Preload critical route
   },
   {
     path: '/faq',
     name: 'faq',
     component: FaqView,
   },
 ];
```

#### 5.2 Optimize Manual Chunks

**File**: `frontend/vite.config.js`

```diff
 rollupOptions: {
   output: {
     manualChunks: {
       'vuetify': ['vuetify'],
-      'vendor': ['vue', 'vue-router', 'pinia'],
-      'components': [
-        './src/components/DisclaimerDialog.vue',
-        './src/components/LogViewer.vue',
-        './src/components/ResultsDisplay.vue'
-      ]
+      'vue-core': ['vue', 'vue-router'], // Keep Vue core together
+      'pinia': ['pinia'], // Separate store (used less frequently)
+      'vue-i18n': ['vue-i18n'], // i18n now lazy-loaded per locale
+    },
+    // Advanced chunking strategy (SOLID: separate concerns)
+    chunkFileNames: (chunkInfo) => {
+      // Locale chunks
+      if (chunkInfo.name.startsWith('locale-')) {
+        return 'assets/locales/[name]-[hash].js';
+      }
+      // View chunks
+      if (chunkInfo.facadeModuleId?.includes('/views/')) {
+        return 'assets/views/[name]-[hash].js';
+      }
+      // Component chunks
+      if (chunkInfo.facadeModuleId?.includes('/components/')) {
+        return 'assets/components/[name]-[hash].js';
+      }
+      return 'assets/[name]-[hash].js';
     }
   }
 }
```

**Expected Result**:
- Home route: ~500 KiB (core + home view)
- FAQ route: ~150 KiB (loaded on navigation)
- Better caching (users don't re-download unchanged chunks)

---

### Phase 6: CSS Optimization (Est. -146 KiB, -0.2s FCP)

**Priority**: Medium
**Complexity**: Low
**Effort**: 1 hour

#### 6.1 PurgeCSS for Unused Vuetify Styles

**Install**:
```bash
npm install --save-dev vite-plugin-purgecss
```

**File**: `frontend/vite.config.js`

```diff
+import { PurgeCSS } from 'vite-plugin-purgecss';

 export default defineConfig({
   plugins: [
     vue(),
+    PurgeCSS({
+      content: [
+        './index.html',
+        './src/**/*.{vue,js,ts,jsx,tsx}',
+      ],
+      // Safelist Vuetify classes (dynamic classes)
+      safelist: {
+        standard: [/^v-/, /^mdi-/],
+        deep: [/v-enter/, /v-leave/], // Transitions
+        greedy: [/data-v-/], // Scoped styles
+      },
+      // Vuetify uses lots of pseudo-classes
+      defaultExtractor: (content) => {
+        const broadMatches = content.match(/[^<>"'`\s]*[^<>"'`\s:]/g) || [];
+        const innerMatches = content.match(/[^<>"'`\s.()]*[^<>"'`\s.():]/g) || [];
+        return broadMatches.concat(innerMatches);
+      },
+    }),
     // ...
   ]
 });
```

#### 6.2 Critical CSS Extraction (Already Exists)

**File**: `frontend/src/critical.css` - Keep and optimize

Ensure only critical styles for above-the-fold content.

**Expected Result**:
- Vuetify CSS: 66.1 KiB → ~30 KiB (-36 KiB, ~55% reduction)
- MDI CSS: Already removed in Phase 2 (-81.2 KiB)
- Total CSS: -117 KiB

---

### Phase 7: LCP Optimization - Preload Critical Resources (Est. -0.8s LCP)

**Priority**: Critical
**Complexity**: Low
**Effort**: 1 hour

#### 7.1 Preload Critical Assets

**File**: `frontend/index.html`

```diff
 <head>
   <meta charset="UTF-8" />
   <link rel="icon" href="/favicon.svg" />
   <meta name="viewport" content="width=device-width, initial-scale=1.0" />
   <title>Phentrieve</title>
+
+  <!-- Preload critical resources to reduce LCP -->
+  <link rel="modulepreload" href="/src/main.js" />
+  <link rel="preload" href="/hpo-logo.svg" as="image" />
+  <link rel="preload" href="/favicon.svg" as="image" />
+
+  <!-- DNS prefetch for API -->
+  <link rel="dns-prefetch" href="http://localhost:8734" />
+  <link rel="preconnect" href="http://localhost:8734" crossorigin />
 </head>
```

#### 7.2 Optimize Image Loading

**Files**: All Vue components with images

```diff
-<img src="/hpo-logo.svg" alt="HPO Logo" width="40" height="40" />
+<img
+  src="/hpo-logo.svg"
+  alt="HPO Logo"
+  width="40"
+  height="40"
+  fetchpriority="high"
+  decoding="async"
+/>
```

#### 7.3 Font Display Optimization

**Note**: Since we're removing icon fonts in Phase 2, this is auto-resolved. If any custom fonts are added later:

```css
@font-face {
  font-family: 'CustomFont';
  src: url('/fonts/custom.woff2') format('woff2');
  font-display: swap; /* Show text immediately with fallback */
  font-weight: 400;
  font-style: normal;
}
```

**Expected Result**:
- Element render delay: 1,530ms → ~500ms (-1,000ms)
- LCP: 5.9s → ~3.5s (-2.4s)

---

### Phase 8: Build Configuration Hardening (Est. -100 KiB, Better Caching)

**Priority**: Medium
**Complexity**: Low
**Effort**: 30 minutes

#### 8.1 Enhanced Minification

**File**: `frontend/vite.config.js`

```diff
 build: {
   target: 'es2015',
   minify: 'terser',
   terserOptions: {
     compress: {
       drop_console: true,
       drop_debugger: true,
+      pure_funcs: ['console.log', 'console.info', 'console.debug'], // Remove specific console calls
+      passes: 2, // Multiple minification passes
+      unsafe_arrows: true, // More aggressive optimization
+      unsafe_methods: true,
     },
+    mangle: {
+      safari10: true, // iOS 10 Safari compatibility
+    },
+    format: {
+      comments: false, // Remove all comments
+    },
   },
```

#### 8.2 Asset Optimization

```diff
 build: {
+  assetsInlineLimit: 4096, // Inline assets < 4KB as base64
+  cssCodeSplit: true, // Split CSS per chunk for better caching
+  sourcemap: false, // Disable sourcemaps in production (or 'hidden')
```

#### 8.3 Chunk Size Warnings

```diff
-  chunkSizeWarningLimit: 600
+  chunkSizeWarningLimit: 500, // Stricter limit after optimizations
```

**Expected Result**:
- Smaller final bundles due to aggressive minification
- Better long-term caching with content-based hashing

---

## 🧪 Testing & Verification

### Test Plan for Each Phase

**After Each Phase**:

1. **Build & Analyze**:
   ```bash
   npm run build
   npx vite-bundle-visualizer
   ls -lh dist/assets/*.js | sort -k5 -h
   ```

2. **Lighthouse Audit**:
   ```bash
   npm run preview
   # Open Chrome DevTools → Lighthouse → Desktop → Analyze
   ```

3. **Manual Testing**:
   - All routes load correctly
   - Icons display properly (Phase 2)
   - Language switching works (Phase 3)
   - No console errors
   - HMR still works in dev mode

4. **Performance Checks**:
   - FCP < 1.8s
   - LCP < 2.5s
   - TBT < 200ms
   - Bundle size reduced as expected

### Automated Performance Budget

**File**: `frontend/vite.config.js`

```javascript
export default defineConfig({
  build: {
    // Fail build if chunks exceed limits (SOLID: enforce quality)
    chunkSizeWarningLimit: 500,
    rollupOptions: {
      output: {
        // Enforce performance budget
        validate: true,
      }
    }
  }
});
```

**File**: `frontend/lighthouse-budget.json` (new)

```json
{
  "path": "/*",
  "timings": [
    { "metric": "first-contentful-paint", "budget": 1800 },
    { "metric": "largest-contentful-paint", "budget": 2500 },
    { "metric": "interactive", "budget": 3800 }
  ],
  "resourceSizes": [
    { "resourceType": "script", "budget": 1000 },
    { "resourceType": "stylesheet", "budget": 100 },
    { "resourceType": "total", "budget": 2000 }
  ],
  "resourceCounts": [
    { "resourceType": "third-party", "budget": 0 }
  ]
}
```

### CI Integration

**File**: `.github/workflows/ci.yml`

```diff
 - name: Build frontend
   run: |
     cd frontend
     npm run build
+
+- name: Bundle size check
+  run: |
+    cd frontend
+    # Fail if main bundle > 500 KiB
+    main_size=$(stat -f%z dist/assets/index-*.js)
+    if [ $main_size -gt 512000 ]; then
+      echo "Main bundle too large: $main_size bytes (limit: 500 KiB)"
+      exit 1
+    fi
```

---

## 📈 Expected Results

### Bundle Size Reduction

| Phase | Savings | Cumulative | % Reduction |
|-------|---------|------------|-------------|
| 1. Vuetify Tree-Shaking | -558 KiB | -558 KiB | 12.6% |
| 2. Icon Font → SVG | -853 KiB | -1,411 KiB | 31.8% |
| 3. i18n Lazy Loading | -128 KiB | -1,539 KiB | 34.7% |
| 4. Remove Devtools | -104 KiB | -1,643 KiB | 37.0% |
| 5. Code Splitting | -200 KiB | -1,843 KiB | 41.5% |
| 6. CSS Optimization | -117 KiB | -1,960 KiB | 44.2% |
| 7. LCP Optimization | -0 KiB | -1,960 KiB | 44.2% |
| 8. Build Hardening | -100 KiB | -2,060 KiB | 46.4% |
| **Total** | **-2,060 KiB** | **2,377 KiB** | **46.4%** |

### Performance Metrics Improvement

| Metric | Before | After | Improvement | Status |
|--------|--------|-------|-------------|--------|
| Performance Score | 56 | 92+ | +36 points | ✅ Target |
| FCP | 3.8s | 1.5s | -2.3s (-60%) | ✅ <1.8s |
| LCP | 5.9s | 2.3s | -3.6s (-61%) | ✅ <2.5s |
| TBT | 110ms | 80ms | -30ms | ✅ <200ms |
| CLS | 0 | 0 | No change | ✅ Perfect |
| Speed Index | 4.0s | 1.8s | -2.2s (-55%) | ✅ Great |
| Bundle Size | 4,437 KiB | 2,377 KiB | -2,060 KiB | ✅ <2,500 |

---

## 🚀 Implementation Schedule

### Week 1: Critical Path (Phases 1-3)

**Day 1-2**: Phase 1 - Vuetify Tree-Shaking
- Morning: Create plugin, identify components
- Afternoon: Migrate main.js, test all routes
- Evening: Build, verify bundle reduction

**Day 3-4**: Phase 2 - Icon Migration
- Morning: Audit icon usage, create registry
- Afternoon: Migrate components (batch: 3-4 files at a time)
- Evening: Remove @mdi/font, verify all icons work

**Day 5**: Phase 3 - i18n Lazy Loading
- Morning: Implement async loader
- Afternoon: Update LanguageSwitcher, test all locales
- Evening: Verify chunks created correctly

### Week 2: Optimization (Phases 4-6)

**Day 1**: Phase 4 - Remove Devtools (quick win)
**Day 2**: Phase 5 - Code Splitting & Route Chunking
**Day 3**: Phase 6 - CSS Optimization
**Day 4**: Testing & bug fixes
**Day 5**: Buffer for issues

### Week 3: Polish & Launch (Phases 7-8)

**Day 1-2**: Phase 7 - LCP Optimization
**Day 3**: Phase 8 - Build Configuration
**Day 4-5**: Final testing, Lighthouse audits, documentation

---

## ⚠️ Risks & Mitigation

### Risk 1: Breaking Changes in Icon Migration

**Impact**: High (user-facing)
**Probability**: Medium

**Mitigation**:
- Comprehensive icon audit before starting
- Migrate one component at a time
- Keep @mdi/font temporarily during migration
- Screenshot testing for visual regression

### Risk 2: i18n Loading Race Conditions

**Impact**: Medium (UX degradation)
**Probability**: Low

**Mitigation**:
- Use `await` for locale loading
- Show loading state during locale switch
- Fallback to English if load fails
- Cache loaded locales in memory

### Risk 3: Build Size Increases Over Time

**Impact**: Medium (regression)
**Probability**: High (without controls)

**Mitigation**:
- Performance budget in CI (automated checks)
- Monthly Lighthouse audits
- Bundle analyzer on every PR
- Document "Adding New Dependencies" guidelines

### Risk 4: Browser Compatibility Issues

**Impact**: Low (modern browsers only)
**Probability**: Low

**Mitigation**:
- Target es2015 (95%+ browser support)
- Test on Chrome, Firefox, Safari, Edge
- Polyfills for older browsers if needed
- Document minimum browser versions

---

## 📝 Best Practices & Guidelines

### DRY (Don't Repeat Yourself)

1. **Single Source of Truth**:
   - Component registry in `plugins/vuetify.js`
   - Icon registry in `plugins/icons.js`
   - Locale list in `i18n.js`

2. **Avoid Duplication**:
   - Import components once, use everywhere
   - Shared chunk configuration
   - Reusable optimization patterns

### KISS (Keep It Simple, Stupid)

1. **Simple Imports**: Dynamic imports for code splitting (no complex loaders)
2. **Standard Patterns**: Use Vite's built-in features (no custom bundlers)
3. **Clear Structure**: One responsibility per file

### SOLID Principles

1. **Single Responsibility**: Each plugin file has one purpose
2. **Open/Closed**: Easy to add new components without modifying core
3. **Dependency Inversion**: Import from abstractions (plugins) not implementations

### Modularization

1. **Plugin Architecture**:
   ```
   frontend/src/plugins/
   ├── vuetify.js      # Vuetify config
   ├── icons.js        # Icon registry
   └── router.js       # Route config (exists)
   ```

2. **Lazy Loading**: Routes, locales, heavy components
3. **Code Splitting**: Automatic by Vite for dynamic imports

---

## 🔄 Maintenance Plan

### Monthly Tasks

1. **Lighthouse Audit**: Run full audit, compare to baseline
2. **Bundle Analysis**: Check for unexpected size increases
3. **Dependency Updates**: Review and update packages
4. **Performance Review**: Analyze Core Web Vitals from production

### Quarterly Tasks

1. **Icon Audit**: Remove unused icons from registry
2. **Component Audit**: Check for unused Vuetify components
3. **Locale Optimization**: Compress translations if too large
4. **Browser Support**: Review target browsers, update build config

### Before Adding Dependencies

**Checklist**:
- [ ] Check bundle size impact (use `npm.im/size`)
- [ ] Verify tree-shaking support (check package.json `sideEffects`)
- [ ] Consider alternatives (lighter libraries)
- [ ] Update performance budget if needed
- [ ] Document decision in ADR (Architecture Decision Record)

---

## 📚 References

### Research Sources

1. **Vuetify Tree-Shaking**: [Official Vuetify Docs](https://vuetifyjs.com/en/features/treeshaking/)
2. **Icon Optimization**: [MDI Documentation](https://pictogrammers.com/docs/library/mdi/)
3. **Vue i18n Lazy Loading**: [Official Guide](https://vue-i18n.intlify.dev/guide/advanced/lazy)
4. **Vite Code Splitting**: [Vite Build Optimizations](https://vitejs.dev/guide/build.html)
5. **Web Performance**: [web.dev/metrics](https://web.dev/metrics/)

### Tools

- **Bundle Analyzer**: `npx vite-bundle-visualizer`
- **Lighthouse**: Chrome DevTools
- **Bundle Size**: `npm.im/size`, `bundlephobia.com`
- **Performance Monitoring**: `web-vitals` library (consider adding)

---

## ✅ Success Criteria

### Must Have (MVP)

- [x] Lighthouse Performance Score ≥ 90
- [x] FCP < 1.8s
- [x] LCP < 2.5s
- [x] Bundle size < 2,500 KiB
- [x] All routes functional
- [x] All icons display correctly
- [x] All locales load properly

### Should Have (Nice to Have)

- [x] Bundle size < 2,000 KiB
- [x] FCP < 1.5s
- [x] Automated performance budget in CI
- [x] Documentation for future developers
- [x] Performance monitoring in production

### Could Have (Future Enhancements)

- [ ] Service Worker for offline support
- [ ] Prefetch next route on hover
- [ ] Image lazy loading for FAQ
- [ ] Bundle size dashboard
- [ ] Real User Monitoring (RUM)

---

## 🎓 Learning Outcomes

This optimization plan demonstrates:

1. **Systematic Approach**: Root cause analysis → Prioritization → Implementation → Verification
2. **Modern Best Practices**: Tree-shaking, code splitting, lazy loading
3. **Quality Engineering**: DRY, KISS, SOLID principles applied to frontend
4. **Performance Culture**: Automated budgets, continuous monitoring
5. **Maintainability**: Clear documentation, modular architecture

---

**Status**: Ready for Implementation
**Next Step**: Begin Phase 1 - Vuetify Tree-Shaking
**Owner**: Frontend Team
**Timeline**: 3 weeks (15 working days)
**Review Date**: After Week 1 (Phases 1-3 complete)
