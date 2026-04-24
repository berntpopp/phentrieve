# Stream B: Frontend Refactoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose frontend mega-components into focused composables and sub-components, standardize Pinia stores, extract constants, and fix confirmed performance issues — all with zero behavioral regression.

**Architecture:** Characterization tests first (Tasks 1-2), then constants/store cleanup (Tasks 3-4), then composable extraction (Tasks 5-7), then component decomposition (Tasks 8-10), then performance fixes (Task 11). Each task produces an atomic, buildable commit.

**Tech Stack:** Vue 3 (Composition API), Vuetify 3, Pinia, Vitest, @vue/test-utils

**Branch:** `improve/frontend-refactor`

**Spec:** `../archived/2026-04-09-code-quality-improvements-design.md` (Stream B)

---

### Task 1: Characterization Tests for queryPreferences Store

**Files:**
- Create: `frontend/src/test/queryPreferences.test.js`
- Read: `frontend/src/stores/queryPreferences.js`

Test current options-API store behavior before migrating to setup store.

- [ ] **Step 1: Read the current store implementation**

Read `frontend/src/stores/queryPreferences.js` to understand state shape, actions, and persistence config.

- [ ] **Step 2: Write characterization tests**

Create `frontend/src/test/queryPreferences.test.js`:

```javascript
import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useQueryPreferencesStore } from '../stores/queryPreferences'

describe('queryPreferences store (characterization)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  describe('initial state', () => {
    it('includeDetails defaults to false', () => {
      const store = useQueryPreferencesStore()
      expect(store.includeDetails).toBe(false)
    })

    it('has no other state properties', () => {
      const store = useQueryPreferencesStore()
      // The store only manages includeDetails — no modelName, numResults, etc.
      // Those live in QueryInterface.vue component data, not in the store.
      expect(store.$state).toEqual({ includeDetails: false })
    })
  })

  describe('actions', () => {
    it('toggleIncludeDetails flips the value', () => {
      const store = useQueryPreferencesStore()
      expect(store.includeDetails).toBe(false)
      store.toggleIncludeDetails()
      expect(store.includeDetails).toBe(true)
      store.toggleIncludeDetails()
      expect(store.includeDetails).toBe(false)
    })

    it('setIncludeDetails sets to true', () => {
      const store = useQueryPreferencesStore()
      store.setIncludeDetails(true)
      expect(store.includeDetails).toBe(true)
    })

    it('setIncludeDetails sets to false', () => {
      const store = useQueryPreferencesStore()
      store.setIncludeDetails(true)
      store.setIncludeDetails(false)
      expect(store.includeDetails).toBe(false)
    })

    it('setIncludeDetails coerces truthy values to boolean', () => {
      const store = useQueryPreferencesStore()
      store.setIncludeDetails(1)
      expect(store.includeDetails).toBe(true)
      store.setIncludeDetails(0)
      expect(store.includeDetails).toBe(false)
    })
  })
})
```

- [ ] **Step 3: Run tests**

Run:
```bash
make frontend-test
```
Expected: All pass (existing + new).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/test/queryPreferences.test.js && git commit -m "test: add characterization tests for queryPreferences store

Lock current state shape and behavior before migrating from
options API to composition API (setup store)."
```

---

### Task 2: Characterization Tests for Key Component Behavior

**Files:**
- Create: `frontend/src/test/components/QueryInterface.test.js`
- Read: `frontend/src/components/QueryInterface.vue`

- [ ] **Step 1: Write mount helpers and basic tests**

Create `frontend/src/test/components/QueryInterface.test.js`:

```javascript
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import { createI18n } from 'vue-i18n'
import en from '../../locales/en.json'

// Mock the API service class — methods are queryHpo() and processText()
// (NOT query() — the real class in services/PhentrieveService.js uses queryHpo)
vi.mock('../../services/PhentrieveService', () => {
  const MockService = {
    queryHpo: vi.fn().mockResolvedValue({ results: [] }),
    processText: vi.fn().mockResolvedValue({ processed_chunks: [], aggregated_hpo_terms: [] }),
    getConfigInfo: vi.fn().mockResolvedValue({
      available_embedding_models: [{ name: 'test-model', loaded: true }],
      default_embedding_model: 'test-model',
    }),
  }
  return { default: MockService }
})

const vuetify = createVuetify({ components, directives })
const i18n = createI18n({ locale: 'en', messages: { en } })

function mountQueryInterface() {
  // QueryInterface uses Options API with setup() hook, has no props.
  // It renders a search container with query input.
  return mount(
    (await import('../../components/QueryInterface.vue')).default,
    {
      global: {
        plugins: [createPinia(), vuetify, i18n],
        stubs: {
          ResultsDisplay: true,
          ConversationSkeleton: true,
        },
      },
    }
  )
}

describe('QueryInterface (characterization)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('mounts without errors', async () => {
    const wrapper = await mountQueryInterface()
    expect(wrapper.exists()).toBe(true)
  })

  it('renders the search container', async () => {
    const wrapper = await mountQueryInterface()
    expect(wrapper.find('.search-container').exists()).toBe(true)
  })

  it('initializes with empty query text', async () => {
    const wrapper = await mountQueryInterface()
    expect(wrapper.vm.queryText).toBe('')
  })

  it('initializes with loading false', async () => {
    const wrapper = await mountQueryInterface()
    expect(wrapper.vm.isLoading).toBe(false)
  })

  it('has advanced options hidden by default', async () => {
    const wrapper = await mountQueryInterface()
    expect(wrapper.vm.showAdvancedOptions).toBe(false)
  })

  it('has default threshold values', async () => {
    const wrapper = await mountQueryInterface()
    expect(wrapper.vm.similarityThreshold).toBe(0.5)
    expect(wrapper.vm.numResults).toBe(10)
    expect(wrapper.vm.chunkRetrievalThreshold).toBe(0.7)
    expect(wrapper.vm.aggregatedTermConfidence).toBe(0.75)
  })
})
```

Note: `mountQueryInterface` is async because it uses dynamic import. QueryInterface.vue is an Options API component with a `setup()` hook that only initializes `conversationStore`. Test assertions use `wrapper.vm.*` to access Options API data properties.

- [ ] **Step 2: Run tests**

Run:
```bash
make frontend-test
```
Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/test/components/ && git commit -m "test: add characterization tests for QueryInterface component

Basic mount and render tests to catch regressions during
component decomposition."
```

---

### Task 3: Extract Constants

**Files:**
- Create: `frontend/src/constants/defaults.js`
- Create: `frontend/src/constants/urls.js`
- Modify: `frontend/src/components/QueryInterface.vue`
- Modify: `frontend/src/components/ResultsDisplay.vue`
- Modify: `frontend/src/components/SimilarityScore.vue`
- Modify: `frontend/src/App.vue`

- [ ] **Step 1: Create defaults.js**

Create `frontend/src/constants/defaults.js`:

```javascript
/**
 * Default values for query parameters and thresholds.
 * Single source of truth — used by QueryInterface and API calls.
 */

// Query defaults
export const DEFAULT_NUM_RESULTS = 10
export const DEFAULT_SIMILARITY_THRESHOLD = 0.5
export const DEFAULT_SPLIT_THRESHOLD = 0.5

// Chunking defaults
export const DEFAULT_CHUNK_RETRIEVAL_THRESHOLD = 0.7
export const DEFAULT_AGGREGATED_TERM_CONFIDENCE = 0.75
export const DEFAULT_INPUT_TEXT_LENGTH_THRESHOLD = 120
export const DEFAULT_WINDOW_SIZE = 3
export const DEFAULT_STEP_SIZE = 1
export const DEFAULT_MIN_SEGMENT_LENGTH = 2
export const DEFAULT_NUM_RESULTS_PER_CHUNK = 3

// Similarity score quality thresholds
export const SCORE_EXCELLENT = 0.9
export const SCORE_GOOD = 0.75
export const SCORE_MODERATE = 0.6
export const SCORE_LOW = 0.4
export const SCORE_ANIMATION_TRIGGER = 0.85
```

- [ ] **Step 2: Create urls.js**

Create `frontend/src/constants/urls.js`:

```javascript
/**
 * External URL templates used across the frontend.
 */

export const HPO_TERM_URL = (hpoId) => `https://hpo.jax.org/browse/term/${hpoId}`
export const GITHUB_REPO_URL = 'https://github.com/berntpopp/phentrieve'
export const PHENTRIEVE_PRODUCTION_URL = 'https://phentrieve.kidney-genetics.org/'
```

- [ ] **Step 3: Update components to import from constants**

In `frontend/src/components/QueryInterface.vue`, replace hardcoded values with imports:

```javascript
import {
  DEFAULT_NUM_RESULTS,
  DEFAULT_SIMILARITY_THRESHOLD,
  DEFAULT_SPLIT_THRESHOLD,
  DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
  DEFAULT_AGGREGATED_TERM_CONFIDENCE,
  DEFAULT_INPUT_TEXT_LENGTH_THRESHOLD,
  DEFAULT_WINDOW_SIZE,
  DEFAULT_STEP_SIZE,
  DEFAULT_MIN_SEGMENT_LENGTH,
  DEFAULT_NUM_RESULTS_PER_CHUNK,
} from '../constants/defaults'
```

Replace each magic number with the named constant. Similarly for `ResultsDisplay.vue` (HPO URLs), `SimilarityScore.vue` (score thresholds), and `App.vue` (GitHub URL).

- [ ] **Step 4: Verify build and tests pass**

Run:
```bash
make frontend-build && make frontend-test
```
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/constants/ frontend/src/components/ frontend/src/App.vue && git commit -m "refactor: extract hardcoded thresholds and URLs to constants modules

Move 12+ magic numbers from QueryInterface, ResultsDisplay,
SimilarityScore to constants/defaults.js. Move HPO and GitHub
URLs to constants/urls.js. Single source of truth."
```

---

### Task 4: Migrate queryPreferences Store to Setup Store

**Files:**
- Modify: `frontend/src/stores/queryPreferences.js`

- [ ] **Step 1: Read the current options-API store**

Read `frontend/src/stores/queryPreferences.js` to understand state, actions, getters, and persistence config.

- [ ] **Step 2: Rewrite as setup store**

Replace the entire file content with:

```javascript
/**
 * Pinia Store for Query Preferences (setup store / composition API)
 *
 * Migrated from options API to match the pattern used by all other stores
 * (conversation.js, disclaimer.js, log.js). State and behavior are identical.
 *
 * @module stores/queryPreferences
 */

import { ref } from 'vue'
import { defineStore } from 'pinia'

export const useQueryPreferencesStore = defineStore('queryPreferences', () => {
  /**
   * Whether to include HPO term details (definitions and synonyms) in results
   * @type {import('vue').Ref<boolean>}
   */
  const includeDetails = ref(false)

  /**
   * Toggle the include details preference
   */
  function toggleIncludeDetails() {
    includeDetails.value = !includeDetails.value
  }

  /**
   * Set the include details preference
   * @param {boolean} value - Whether to include details
   */
  function setIncludeDetails(value) {
    includeDetails.value = Boolean(value)
  }

  return { includeDetails, toggleIncludeDetails, setIncludeDetails }
}, {
  /**
   * Enable automatic persistence to localStorage
   * Plugin will automatically save/load state on changes
   */
  persist: true,
})
```

Key points: the store ID stays `'queryPreferences'` so existing localStorage data is preserved. The state shape `{ includeDetails }` is identical. Every ref and function is returned so Pinia tracks them.

- [ ] **Step 3: Run characterization tests**

Run:
```bash
make frontend-test
```
Expected: All characterization tests from Task 1 pass — same state shape, same behavior.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/stores/queryPreferences.js && git commit -m "refactor: migrate queryPreferences store from options API to setup store

Standardizes all Pinia stores on composition API (setup store pattern).
All 4 stores now use the same pattern. Characterization tests pass."
```

---

### Task 5: Extract useFileDownload Composable

**Files:**
- Create: `frontend/src/composables/useFileDownload.js`
- Modify: `frontend/src/components/QueryInterface.vue`
- Modify: `frontend/src/components/LogViewer.vue`

- [ ] **Step 1: Create the composable**

Create `frontend/src/composables/useFileDownload.js`:

```javascript
/**
 * Composable for file download functionality.
 * Replaces 3x duplicated document.createElement('a') pattern.
 */
export function useFileDownload() {
  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  function downloadText(content, filename, mimeType = 'text/plain') {
    const blob = new Blob([content], { type: mimeType })
    downloadBlob(blob, filename)
  }

  function downloadJson(data, filename) {
    const content = JSON.stringify(data, null, 2)
    downloadText(content, filename, 'application/json')
  }

  return { downloadBlob, downloadText, downloadJson }
}
```

- [ ] **Step 2: Replace download patterns in QueryInterface.vue**

Find the 2 locations in `QueryInterface.vue` where `document.createElement('a')` is used for downloads (around lines 1144 and 1210).

QueryInterface.vue uses Options API, so composables must be called inside `setup()` and their returns must be exposed to the template/methods. Update the existing `setup()` function:

```javascript
import { useFileDownload } from '../composables/useFileDownload'

export default {
  // ...
  setup() {
    const conversationStore = useConversationStore()
    const { downloadText, downloadJson } = useFileDownload()
    return { conversationStore, downloadText, downloadJson }
  },
  methods: {
    exportCollectionAsText() {
      // Before: document.createElement('a') ... document.body.appendChild ...
      // After:
      this.downloadText(textContent, 'phentrieve-export.txt')
    },
    exportPhenopacket() {
      // Before: document.createElement('a') ... document.body.appendChild ...
      // After:
      this.downloadJson(phenopacketData, 'phenopacket.json')
    },
  },
}
```

- [ ] **Step 3: Replace download pattern in LogViewer.vue**

LogViewer.vue also uses Options API. Read it first to confirm, then apply the same pattern — call `useFileDownload()` inside `setup()` and return the methods:

```javascript
import { useFileDownload } from '../composables/useFileDownload'

export default {
  setup() {
    const { downloadJson } = useFileDownload()
    return { downloadJson }
  },
  methods: {
    downloadLogs() {
      // Before: document.createElement('a') ... document.body.appendChild ...
      // After:
      this.downloadJson(logData, 'phentrieve-logs.json')
    },
  },
}
```

- [ ] **Step 4: Verify build and tests pass**

Run:
```bash
make frontend-build && make frontend-test
```
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/composables/useFileDownload.js frontend/src/components/QueryInterface.vue frontend/src/components/LogViewer.vue && git commit -m "refactor: extract useFileDownload composable, eliminate 3x duplication

The document.createElement('a') download pattern was duplicated
in QueryInterface.vue (2x) and LogViewer.vue (1x). Now shared
from a single composable with proper URL.revokeObjectURL cleanup."
```

---

### Task 6: Extract useVersionCheck Composable

**Files:**
- Create: `frontend/src/composables/useVersionCheck.js`
- Modify: `frontend/src/App.vue`

- [ ] **Step 1: Read App.vue version/health logic**

Read `frontend/src/App.vue` to identify all version-fetching and health-monitoring code.

- [ ] **Step 2: Create the composable**

Create `frontend/src/composables/useVersionCheck.js`:

```javascript
import { ref } from 'vue'
import { getAllVersions } from '../utils/version'

/**
 * Composable for version fetching.
 * Extracted from App.vue refreshVersions() method.
 *
 * Note: API health monitoring stays in App.vue via useApiHealth()
 * from services/api-health.js — that composable already exists and
 * is wired into App.vue's computed properties (apiConnected, responseTime).
 * We only extract the version-fetching concern here.
 */
export function useVersionCheck() {
  const frontendVersion = ref('Loading...')
  const apiVersion = ref('Loading...')
  const cliVersion = ref('Loading...')
  const environment = ref('unknown')
  const loadingVersions = ref(false)

  async function refreshVersions() {
    loadingVersions.value = true
    try {
      const versions = await getAllVersions()
      frontendVersion.value = versions.frontend?.version || 'unknown'
      apiVersion.value = versions.api?.version || 'unknown'
      cliVersion.value = versions.cli?.version || 'unknown'
      environment.value = versions.environment || 'unknown'
    } catch {
      // Versions stay at their current values on error
    } finally {
      loadingVersions.value = false
    }
  }

  return {
    frontendVersion, apiVersion, cliVersion, environment,
    loadingVersions, refreshVersions,
  }
}
```

- [ ] **Step 3: Update App.vue to use the composable**

App.vue uses Options API, so call the composable inside `setup()` and return its values. This replaces the `data()` properties (`frontendVersion`, `apiVersion`, `cliVersion`, `environment`, `loadingVersions`) and the `refreshVersions()` method:

```javascript
import { useVersionCheck } from './composables/useVersionCheck'

export default {
  setup() {
    const {
      frontendVersion, apiVersion, cliVersion, environment,
      loadingVersions, refreshVersions,
    } = useVersionCheck()

    return {
      frontendVersion, apiVersion, cliVersion, environment,
      loadingVersions, refreshVersions,
    }
  },
  // Remove from data(): frontendVersion, apiVersion, cliVersion, environment, loadingVersions
  // Remove from methods: refreshVersions()
  // Keep in mounted(): call this.refreshVersions() — it now comes from setup()
  // Keep: useApiHealth() usage in computed (apiConnected, responseTime) — unchanged
}
```

- [ ] **Step 4: Verify build passes**

Run:
```bash
make frontend-build && make frontend-test
```
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/composables/useVersionCheck.js frontend/src/App.vue && git commit -m "refactor: extract useVersionCheck composable from App.vue

Moves API version fetching and health monitoring into a reusable
composable. Reduces App.vue complexity."
```

---

### Task 7: Replace DOM Queries in App.vue with Template Refs

**Files:**
- Modify: `frontend/src/App.vue`

- [ ] **Step 1: Read App.vue DOM manipulation locations**

Read `frontend/src/App.vue` focusing on lines 445, 448, 461, 471, 475, 519, 546 — all `document.querySelector` calls used for tutorial panel management.

- [ ] **Step 2: Replace with template refs and event-driven patterns**

For each `document.querySelector` call:

1. Add a `ref` attribute to the target element in the template
2. Use `useTemplateRef()` or `ref()` in the script
3. Replace `querySelector` with the ref

App.vue uses **Options API** (not `<script setup>`), so template refs are accessed via `this.$refs`:

```html
<!-- Template: add ref attribute to target elements -->
<v-navigation-drawer ref="navDrawer" v-model="drawerOpen" ...>
```

```javascript
// Script (Options API methods):
export default {
  data() {
    return {
      drawerOpen: false, // Use reactive data instead of DOM manipulation
    }
  },
  methods: {
    closePanel() {
      // Before: document.querySelector('.v-navigation-drawer')?.click()
      // After: toggle the reactive v-model binding
      this.drawerOpen = false
    },
  },
}
```

**Strategy for each querySelector pattern:**

| Current DOM query | Replacement |
|---|---|
| `querySelector('.v-navigation-drawer')` | `v-model` reactive binding on drawer |
| `querySelector('.v-btn')` inside tutorial | `ref="tutorialTarget"` + `this.$refs.tutorialTarget` |
| `querySelectorAll(...)` for panel closing | Reactive boolean per panel (`logDrawerOpen`, `settingsDrawerOpen`) |
| `:contains(...)` selectors (invalid in DOM) | Data attributes: `data-tutorial-step="N"` + `querySelector('[data-tutorial-step="N"]')` |

For tutorial-related DOM queries that target Vuetify internal classes (`.v-btn`, `.v-icon`), replace with `data-tutorial-step` attributes on the target elements. The tutorial service can then use `querySelector('[data-tutorial-step="1"]')` — still a DOM query but on a stable, owned attribute rather than brittle Vuetify internals.

- [ ] **Step 3: Verify build and tests pass**

Run:
```bash
make frontend-build && make frontend-test
```
Expected: All pass.

- [ ] **Step 4: Manual smoke test**

Run `make dev-frontend` + `make dev-api`, click through the tutorial overlay, verify panels open and close correctly.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.vue && git commit -m "refactor: replace document.querySelector calls with template refs in App.vue

Eliminates 7+ fragile DOM queries that depended on Vuetify's internal
CSS class names. Uses Vue template refs and reactive state instead."
```

---

### Task 8: Decompose QueryInterface.vue — Extract Composables

**Files:**
- Create: `frontend/src/composables/useAdvancedOptions.js`
- Create: `frontend/src/composables/usePhenotypeCollection.js`
- Modify: `frontend/src/components/QueryInterface.vue`

This is the largest refactoring task. Extract composables one at a time, testing after each.

- [ ] **Step 1: Extract useAdvancedOptions**

Create `frontend/src/composables/useAdvancedOptions.js`:

```javascript
import { ref } from 'vue'
import {
  DEFAULT_NUM_RESULTS,
  DEFAULT_SIMILARITY_THRESHOLD,
  DEFAULT_SPLIT_THRESHOLD,
  DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
  DEFAULT_AGGREGATED_TERM_CONFIDENCE,
  DEFAULT_INPUT_TEXT_LENGTH_THRESHOLD,
  DEFAULT_WINDOW_SIZE,
  DEFAULT_STEP_SIZE,
  DEFAULT_MIN_SEGMENT_LENGTH,
  DEFAULT_NUM_RESULTS_PER_CHUNK,
} from '../constants/defaults'

/**
 * Composable for managing advanced query options state.
 * Extracted from QueryInterface.vue to reduce component size.
 */
export function useAdvancedOptions() {
  const showAdvancedOptions = ref(false)
  const numResults = ref(DEFAULT_NUM_RESULTS)
  const similarityThreshold = ref(DEFAULT_SIMILARITY_THRESHOLD)
  const splitThreshold = ref(DEFAULT_SPLIT_THRESHOLD)
  const chunkRetrievalThreshold = ref(DEFAULT_CHUNK_RETRIEVAL_THRESHOLD)
  const aggregatedTermConfidence = ref(DEFAULT_AGGREGATED_TERM_CONFIDENCE)
  const inputTextLengthThreshold = ref(DEFAULT_INPUT_TEXT_LENGTH_THRESHOLD)
  const windowSize = ref(DEFAULT_WINDOW_SIZE)
  const stepSize = ref(DEFAULT_STEP_SIZE)
  const minSegmentLength = ref(DEFAULT_MIN_SEGMENT_LENGTH)
  const numResultsPerChunk = ref(DEFAULT_NUM_RESULTS_PER_CHUNK)

  function toggleAdvancedOptions() {
    showAdvancedOptions.value = !showAdvancedOptions.value
  }

  return {
    showAdvancedOptions,
    numResults,
    similarityThreshold,
    splitThreshold,
    chunkRetrievalThreshold,
    aggregatedTermConfidence,
    inputTextLengthThreshold,
    windowSize,
    stepSize,
    minSegmentLength,
    numResultsPerChunk,
    toggleAdvancedOptions,
  }
}
```

- [ ] **Step 2: Wire composable into QueryInterface.vue**

Replace the corresponding data properties in QueryInterface.vue with:

```javascript
import { useAdvancedOptions } from '../composables/useAdvancedOptions'

const {
  showAdvancedOptions, numResults, similarityThreshold,
  splitThreshold, chunkRetrievalThreshold, aggregatedTermConfidence,
  inputTextLengthThreshold, windowSize, stepSize, minSegmentLength,
  numResultsPerChunk, toggleAdvancedOptions,
} = useAdvancedOptions()
```

Remove the corresponding data properties from the component.

- [ ] **Step 3: Extract usePhenotypeCollection**

Create `frontend/src/composables/usePhenotypeCollection.js` with the collection management logic (add/remove phenotype, toggle assertion, phenopacket export). Move the relevant methods and reactive state from QueryInterface.vue.

- [ ] **Step 4: Wire collection composable into QueryInterface.vue**

```javascript
import { usePhenotypeCollection } from '../composables/usePhenotypeCollection'

const {
  collectedPhenotypes, addPhenotype, removePhenotype,
  toggleAssertion, exportPhenopacket, exportCollectionText,
} = usePhenotypeCollection()
```

- [ ] **Step 5: Verify build and tests pass**

Run:
```bash
make frontend-build && make frontend-test
```
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/composables/useAdvancedOptions.js frontend/src/composables/usePhenotypeCollection.js frontend/src/components/QueryInterface.vue && git commit -m "refactor: extract useAdvancedOptions and usePhenotypeCollection from QueryInterface

First phase of QueryInterface decomposition. Moves advanced options
state (12 refs) and phenotype collection logic into composables.
QueryInterface.vue LOC reduced significantly."
```

---

### Task 9: Decompose QueryInterface.vue — Extract Sub-Components

**Files:**
- Create: `frontend/src/components/AdvancedOptionsPanel.vue`
- Create: `frontend/src/components/PhenotypeCollectionPanel.vue`
- Modify: `frontend/src/components/QueryInterface.vue`

- [ ] **Step 1: Extract AdvancedOptionsPanel.vue**

Create `frontend/src/components/AdvancedOptionsPanel.vue` containing the advanced options template section from QueryInterface.vue. The component receives all option values as `v-model` props.

```vue
<template>
  <v-expand-transition>
    <div v-show="modelValue" role="region" aria-label="Advanced options">
      <!-- Advanced options form fields moved from QueryInterface -->
    </div>
  </v-expand-transition>
</template>

<script setup>
defineProps({
  modelValue: Boolean,
  // All option refs passed as v-model props
})
defineEmits(['update:modelValue'])
</script>
```

- [ ] **Step 2: Extract PhenotypeCollectionPanel.vue**

Similarly extract the phenotype collection panel template and its local display logic.

- [ ] **Step 3: Update QueryInterface.vue to use sub-components**

```vue
<template>
  <!-- Replace inline advanced options with: -->
  <AdvancedOptionsPanel v-model="showAdvancedOptions" ... />

  <!-- Replace inline collection panel with: -->
  <PhenotypeCollectionPanel
    :phenotypes="collectedPhenotypes"
    @remove="removePhenotype"
    @toggle-assertion="toggleAssertion"
    @export-text="exportCollectionText"
    @export-json="exportPhenopacket"
  />
</template>
```

- [ ] **Step 4: Verify build and tests pass**

Run:
```bash
make frontend-build && make frontend-test && make frontend-i18n-check
```
Expected: All pass. i18n check is important since we're moving template strings between files.

- [ ] **Step 5: Check QueryInterface.vue LOC**

Run:
```bash
wc -l frontend/src/components/QueryInterface.vue
```
Expected: Significantly less than 1483 (target: <350).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/ && git commit -m "refactor: extract AdvancedOptionsPanel and PhenotypeCollectionPanel from QueryInterface

Second phase of QueryInterface decomposition. Template sections
moved to focused sub-components. QueryInterface is now a thin
coordinator."
```

---

### Task 10: Decompose ResultsDisplay.vue

**Files:**
- Create: `frontend/src/components/ResultItem.vue`
- Create: `frontend/src/components/ChunkResultsView.vue`
- Create: `frontend/src/components/AggregatedTermsView.vue`
- Modify: `frontend/src/components/ResultsDisplay.vue`

- [ ] **Step 1: Read ResultsDisplay.vue render paths**

Read `frontend/src/components/ResultsDisplay.vue` to identify the 3 render paths (simple query, textProcess, aggregated) and their template boundaries.

- [ ] **Step 2: Extract ResultItem.vue**

Extract the single-result rendering block (result card with HPO ID, score, details expansion, collection button) into `ResultItem.vue`:

```vue
<template>
  <v-card class="result-item">
    <!-- Single result rendering -->
  </v-card>
</template>

<script setup>
import { HPO_TERM_URL } from '../constants/urls'

defineProps({
  result: { type: Object, required: true },
  isCollected: { type: Boolean, default: false },
})
defineEmits(['add-to-collection', 'toggle-details'])
</script>
```

- [ ] **Step 3: Extract ChunkResultsView.vue**

Extract the chunk-based text processing display (chunk list with highlights, per-chunk results).

- [ ] **Step 4: Extract AggregatedTermsView.vue**

Extract the aggregated HPO terms display (ranked term list with evidence, confidence scores).

- [ ] **Step 5: Update ResultsDisplay.vue to compose sub-components**

```vue
<template>
  <!-- Simple query mode -->
  <template v-if="responseType === 'query'">
    <ResultItem
      v-for="(result, index) in responseData.results"
      :key="result.hpo_id"
      :result="result"
      :is-collected="isCollected(result.hpo_id)"
      @add-to-collection="$emit('add-to-collection', result)"
    />
  </template>

  <!-- Text process mode -->
  <template v-else-if="responseType === 'textProcess'">
    <ChunkResultsView :chunks="responseData.processed_chunks" ... />
  </template>

  <!-- Aggregated mode -->
  <template v-else-if="responseType === 'aggregated'">
    <AggregatedTermsView :terms="responseData.aggregated_hpo_terms" ... />
  </template>
</template>
```

- [ ] **Step 6: Verify build, tests, and i18n**

Run:
```bash
make frontend-build && make frontend-test && make frontend-i18n-check
```
Expected: All pass.

- [ ] **Step 7: Check ResultsDisplay.vue LOC**

Run:
```bash
wc -l frontend/src/components/ResultsDisplay.vue
```
Expected: Less than 1079 (target: <450).

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/ && git commit -m "refactor: decompose ResultsDisplay into ResultItem, ChunkResultsView, AggregatedTermsView

Each render path (query/textProcess/aggregated) now has its own
focused component. ResultsDisplay is a thin router between them."
```

---

### Task 11: Performance Fixes (Confirmed Issues)

**Files:**
- Modify: `frontend/src/components/QueryInterface.vue`
- Modify: `frontend/src/components/LogViewer.vue`
- Modify: `frontend/src/router/index.js`

- [ ] **Step 1: Fix deep watcher**

In `QueryInterface.vue` (or the relevant composable after decomposition), replace the deep watcher:

```javascript
// Before:
watch: {
  'conversationStore.queryHistory': {
    handler() { /* scroll logic */ },
    deep: true,
  }
}

// After:
import { watch, computed } from 'vue'

const historyLength = computed(() => conversationStore.queryHistory.length)
watch(historyLength, () => {
  // scroll logic — only fires when items are added/removed
})
```

- [ ] **Step 2: Fix JSON.stringify in LogViewer filteredLogs**

In `LogViewer.vue`, replace the expensive `JSON.stringify` search:

```javascript
// Before:
const filteredLogs = computed(() => {
  return logs.filter(log =>
    JSON.stringify(log.data).toLowerCase().includes(searchTerm.value)
  )
})

// After:
const filteredLogs = computed(() => {
  const term = searchTerm.value.toLowerCase()
  if (!term) return logs
  return logs.filter(log =>
    (log.message || '').toLowerCase().includes(term) ||
    (log.level || '').toLowerCase().includes(term) ||
    (log.source || '').toLowerCase().includes(term)
  )
})
```

- [ ] **Step 3: Add route-level code splitting**

In `frontend/src/router/index.js`, use lazy loading for views:

```javascript
// Before:
import HomeView from '../views/HomeView.vue'
import FAQView from '../views/FAQView.vue'

// After:
const HomeView = () => import('../views/HomeView.vue')
const FAQView = () => import('../views/FAQView.vue')
```

- [ ] **Step 4: Verify build and tests**

Run:
```bash
make frontend-build && make frontend-test
```
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/ && git commit -m "perf: fix deep watcher, JSON.stringify filter, add route code splitting

- Replace deep watcher on queryHistory with shallow length watch
- Replace JSON.stringify in filteredLogs with targeted field search
- Add lazy loading for route views (code splitting)"
```

---

### Task 12: Run Verification Gates

- [ ] **Step 1: Gate 1 — Lint and format**

Run:
```bash
make frontend-lint && make frontend-format
```
Expected: Zero errors.

- [ ] **Step 2: Gate 2 — Production build**

Run:
```bash
make frontend-build
```
Expected: Builds successfully.

- [ ] **Step 3: Gate 3 — All tests pass**

Run:
```bash
make frontend-test
```
Expected: All pass (existing + new characterization tests).

- [ ] **Step 4: Gate 4 — i18n check**

Run:
```bash
make frontend-i18n-check
```
Expected: All locale keys valid.

- [ ] **Step 5: Gate 5 — Component LOC check**

Run:
```bash
echo "QueryInterface.vue:" && wc -l frontend/src/components/QueryInterface.vue
echo "ResultsDisplay.vue:" && wc -l frontend/src/components/ResultsDisplay.vue
echo "App.vue:" && wc -l frontend/src/App.vue
```
Expected: QueryInterface <350, ResultsDisplay <450.

- [ ] **Step 6: Gate 6 — No document.querySelector in Vue components**

Run:
```bash
grep -r "document\.querySelector\|document\.getElementById\|document\.createElement" frontend/src/components/ frontend/src/App.vue
```
Expected: Only in `useFileDownload.js` composable (the centralized download helper). Zero in .vue files except if useFileDownload is used inline.

- [ ] **Step 7: Gate 7 — Manual smoke test**

Run `make dev-frontend` + `make dev-api`:
1. Submit a query — results render correctly
2. Switch to text processing mode — chunks display
3. Add phenotype to collection — collection panel works
4. Export phenopacket — download triggers
5. Open advanced options — all controls work
6. Click through tutorial — no errors
