# Frontend Conversation Persistence & Performance Optimization Plan

**Status**: Completed
**Priority**: High
**Complexity**: Medium
**Estimated Effort**: 2-3 days
**Target**: Alpha Release
**Completed**: 2025-11-24
**Last Reviewed**: 2025-01-24 (Senior Developer Review Applied)

## Problem Statement

The current frontend chat interface has several critical issues affecting performance and user experience:

### Current Issues

1. **DOM Performance Degradation**
   - `queryHistory` array grows indefinitely (line 910 in QueryInterface.vue)
   - Each query adds user bubble + results to DOM without cleanup
   - No virtual scrolling - all conversation items rendered simultaneously
   - Browser performance degrades with long conversations (100+ queries)

2. **Navigation State Loss**
   - All conversation history lost when navigating to FAQ or other pages
   - No persistence across page reloads
   - Poor user experience when accidentally navigating away

3. **No Reset Functionality**
   - Users cannot clear conversation and start fresh
   - No application state reset mechanism
   - Accumulated state can cause confusion

4. **No User Control**
   - Cannot configure conversation retention length
   - No settings for performance vs. history trade-off
   - One-size-fits-all approach doesn't suit all use cases

## Research Findings (2024/2025 Best Practices)

### Virtual Scrolling Performance

- **vue-virtual-scroller**: Reduces DOMContentLoaded from 22s to 563ms (39x improvement)
- **Memory savings**: 128MB to 79MB (38% reduction)
- **Recommendation**: Use `vue-virtual-scroller@next` for Vue 3 compatibility
- **Alternative**: `virtua` (3kB, zero-config, tree-shakeable)

### State Persistence

- **pinia-plugin-persistedstate**: Official recommendation for Pinia persistence
- **Features**: Configurable storage (localStorage/sessionStorage), SSR support, selective persistence
- **Size limits**: LocalStorage ~5MB (sufficient for conversation history)
- **Best practice**: Store only essential data, not computed/derived values

### Chat Interface Patterns

- **Windowing**: Render only visible items + buffer (e.g., 50 visible + 10 buffer)
- **Pagination**: Load older conversations on demand (lazy loading)
- **Automatic cleanup**: Remove DOM nodes outside viewport
- **Bi-directional scrolling**: New messages at bottom, load older ones upward

## Proposed Solution Architecture

### Phase 1: Pinia Conversation Store (DRY Principle)

**Objective**: Centralize conversation state management with persistence

> **Note**: `pinia-plugin-persistedstate@4.7.1` is already installed and configured in `main.js`.
> No additional installation or configuration is required.

#### 1.1 Create Conversation Store

**File**: `frontend/src/stores/conversation.js`

```javascript
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useConversationStore = defineStore('conversation', () => {
  // State
  const queryHistory = ref([])
  const collectedPhenotypes = ref([])
  const maxHistoryLength = ref(50) // User-configurable

  // Getters
  const conversationLength = computed(() => queryHistory.value.length)
  const hasConversation = computed(() => queryHistory.value.length > 0)

  // Actions
  function addQuery(queryItem) {
    // Add unique ID for virtual scroller key-field
    queryHistory.value.unshift({
      id: crypto.randomUUID(),
      timestamp: new Date().toISOString(),
      ...queryItem
    })

    // Trim history if exceeds max length
    if (queryHistory.value.length > maxHistoryLength.value) {
      queryHistory.value.length = maxHistoryLength.value
    }
  }

  function updateQueryResponse(id, responseData, error = null) {
    const item = queryHistory.value.find(q => q.id === id)
    if (item) {
      item.response = responseData
      item.error = error
      item.loading = false
    }
  }

  function clearConversation() {
    queryHistory.value = []
  }

  function addPhenotype(phenotype) {
    const isDuplicate = collectedPhenotypes.value.some(
      item => item.hpo_id === phenotype.hpo_id
    )
    if (!isDuplicate) {
      collectedPhenotypes.value.push({
        ...phenotype,
        added_at: new Date().toISOString()
      })
    }
  }

  function removePhenotype(index) {
    collectedPhenotypes.value.splice(index, 1)
  }

  function clearPhenotypes() {
    collectedPhenotypes.value = []
  }

  function resetAll() {
    // Complete application state reset
    clearConversation()
    clearPhenotypes()
  }

  return {
    // State
    queryHistory,
    collectedPhenotypes,
    maxHistoryLength,

    // Getters
    conversationLength,
    hasConversation,

    // Actions
    addQuery,
    updateQueryResponse,
    clearConversation,
    addPhenotype,
    removePhenotype,
    clearPhenotypes,
    resetAll
  }
}, {
  // pinia-plugin-persistedstate v4.x syntax
  persist: {
    key: 'phentrieve-conversation',
    storage: localStorage,
    pick: ['queryHistory', 'collectedPhenotypes', 'maxHistoryLength']
  }
})
```

### Phase 2: Virtual Scrolling Implementation (Performance Optimization)

**Objective**: Render only visible conversation items to prevent DOM bloat

#### 2.1 Install Virtual Scroller

```bash
npm install vue-virtual-scroller@next
# OR for smaller bundle
npm install virtua
```

#### 2.2 Update QueryInterface.vue

**Current**: Lines 556-599 (conversation-container with v-for)

**Replace with** (correct Vue SFC structure):

```vue
<script setup>
import { DynamicScroller, DynamicScrollerItem } from 'vue-virtual-scroller'
import 'vue-virtual-scroller/dist/vue-virtual-scroller.css'
import { useConversationStore } from '../stores/conversation'
import ResultsDisplay from './ResultsDisplay.vue'

const conversationStore = useConversationStore()
</script>

<template>
  <div ref="conversationContainer" class="conversation-container">
    <DynamicScroller
      :items="conversationStore.queryHistory"
      :min-item-size="100"
      class="scroller"
      key-field="id"
    >
      <template #default="{ item, index, active }">
        <DynamicScrollerItem
          :item="item"
          :active="active"
          :size-dependencies="[item.query, item.response]"
          :data-index="index"
        >
          <div class="mb-4">
            <!-- User query bubble -->
            <div class="user-query d-flex">
              <v-tooltip location="top" text="User Input">
                <template #activator="{ props }">
                  <v-avatar v-bind="props" color="primary" size="36" class="mt-1 mr-2">
                    <span class="white--text">U</span>
                  </v-avatar>
                </template>
              </v-tooltip>
              <div class="query-bubble">
                <p class="mb-0" style="white-space: pre-wrap">
                  {{ item.query }}
                </p>
              </div>
            </div>

            <!-- API response -->
            <div v-if="item.loading || item.response || item.error" class="bot-response d-flex mt-2">
              <v-tooltip location="top" text="Phentrieve Response">
                <template #activator="{ props }">
                  <v-avatar v-bind="props" color="info" size="36" class="mt-1 mr-2">
                    <v-icon color="white">mdi-robot-outline</v-icon>
                  </v-avatar>
                </template>
              </v-tooltip>
              <div class="response-bubble">
                <v-progress-circular v-if="item.loading" indeterminate color="primary" size="24" />

                <ResultsDisplay
                  v-else
                  :key="'results-' + item.id"
                  :response-data="item.response"
                  :result-type="item.type"
                  :error="item.error"
                  :collected-phenotypes="conversationStore.collectedPhenotypes"
                  @add-to-collection="conversationStore.addPhenotype"
                />
              </div>
            </div>
          </div>
        </DynamicScrollerItem>
      </template>
    </DynamicScroller>
  </div>
</template>
```

**Benefits**:
- Only renders ~10-20 visible items + buffer
- Automatically recycles DOM nodes
- Maintains scroll position
- Handles dynamic item heights
- Uses unique `id` as key (not index) for proper Vue reactivity

### Phase 3: User Settings Component (SOLID - Single Responsibility)

**Objective**: Give users control over conversation retention and performance

#### 3.1 Create Settings Dialog Component

**File**: `frontend/src/components/ConversationSettings.vue`

> **Note**: Component does NOT include positioning styles. Parent component handles layout.

```vue
<script setup>
import { ref, computed } from 'vue'
import { useConversationStore } from '../stores/conversation'

const conversationStore = useConversationStore()
const dialog = ref(false)
const confirmClearDialog = ref(false)

const estimatedStorageSize = computed(() => {
  const size = JSON.stringify({
    queryHistory: conversationStore.queryHistory,
    collectedPhenotypes: conversationStore.collectedPhenotypes
  }).length
  return size < 1024 ? `${size} bytes` : `${(size / 1024).toFixed(1)} KB`
})

function handleConfirmedClear() {
  conversationStore.clearConversation()
  confirmClearDialog.value = false
  dialog.value = false
}
</script>

<template>
  <!-- Main Settings Dialog -->
  <v-dialog v-model="dialog" max-width="500">
    <template #activator="{ props }">
      <v-btn
        v-bind="props"
        icon="mdi-cog"
        size="small"
        variant="text"
        aria-label="Conversation Settings"
      >
        <v-icon>mdi-cog</v-icon>
        <v-tooltip activator="parent" location="top">
          {{ $t('conversationSettings.tooltip', 'Conversation Settings') }}
        </v-tooltip>
      </v-btn>
    </template>

    <v-card>
      <v-card-title>
        <span class="text-h6">{{ $t('conversationSettings.title', 'Conversation Settings') }}</span>
      </v-card-title>

      <v-card-text>
        <!-- History Length Setting -->
        <div class="mb-4">
          <label class="text-subtitle-2 mb-2 d-block">
            {{ $t('conversationSettings.historyLength', 'Keep Last N Queries') }}
          </label>
          <v-slider
            v-model="conversationStore.maxHistoryLength"
            :min="10"
            :max="200"
            :step="10"
            thumb-label="always"
            color="primary"
          >
            <template #append>
              <v-text-field
                v-model.number="conversationStore.maxHistoryLength"
                type="number"
                style="width: 80px"
                density="compact"
                variant="outlined"
                hide-details
              />
            </template>
          </v-slider>
          <p class="text-caption mt-1">
            {{ $t('conversationSettings.historyHint', 'Lower values improve performance. Higher values keep more history.') }}
          </p>
        </div>

        <!-- Current Stats -->
        <v-divider class="my-4" />
        <div class="text-caption">
          <p><strong>{{ $t('conversationSettings.stats', 'Current Statistics') }}:</strong></p>
          <p>{{ $t('conversationSettings.totalQueries', 'Total Queries') }}: {{ conversationStore.conversationLength }}</p>
          <p>{{ $t('conversationSettings.totalPhenotypes', 'Collected Phenotypes') }}: {{ conversationStore.collectedPhenotypes.length }}</p>
          <p>{{ $t('conversationSettings.storageUsed', 'Estimated Storage') }}: {{ estimatedStorageSize }}</p>
        </div>
      </v-card-text>

      <v-card-actions>
        <v-spacer />
        <v-btn color="error" variant="text" @click="confirmClearDialog = true">
          {{ $t('conversationSettings.clearHistory', 'Clear History') }}
        </v-btn>
        <v-btn color="primary" variant="text" @click="dialog = false">
          {{ $t('conversationSettings.close', 'Close') }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>

  <!-- Confirmation Dialog (replaces native confirm()) -->
  <v-dialog v-model="confirmClearDialog" max-width="400">
    <v-card>
      <v-card-title class="text-h6">
        {{ $t('conversationSettings.confirmClearTitle', 'Clear Conversation?') }}
      </v-card-title>
      <v-card-text>
        {{ $t('conversationSettings.confirmClearMessage', 'This will permanently delete all conversation history. This action cannot be undone.') }}
      </v-card-text>
      <v-card-actions>
        <v-spacer />
        <v-btn variant="text" @click="confirmClearDialog = false">
          {{ $t('common.cancel', 'Cancel') }}
        </v-btn>
        <v-btn color="error" variant="flat" @click="handleConfirmedClear">
          {{ $t('common.delete', 'Delete') }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>
```

#### 3.2 Add Settings Button to Layout

**Location**: In `QueryInterface.vue` - parent controls positioning

```vue
<!-- In QueryInterface.vue template, bottom-right area -->
<div class="conversation-actions">
  <ConversationSettings />
  <!-- Other FABs like collection panel -->
</div>

<style scoped>
.conversation-actions {
  position: fixed;
  bottom: 90px;
  right: 16px;
  z-index: 1040;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
</style>
```

### Phase 4: Application Reset via Logo (UX Enhancement)

**Objective**: Provide intuitive way to reset entire application state

#### 4.1 Update App.vue or Main Header Component

**Add click handler with Vuetify confirmation dialog** (not native `confirm()`):

```vue
<script setup>
import { ref } from 'vue'
import { useConversationStore } from './stores/conversation'
import { useRouter } from 'vue-router'

const conversationStore = useConversationStore()
const router = useRouter()
const confirmResetDialog = ref(false)

function handleLogoClick() {
  if (conversationStore.hasConversation) {
    confirmResetDialog.value = true
  } else {
    router.push('/') // Just navigate if no conversation
  }
}

function handleConfirmedReset() {
  conversationStore.resetAll()
  confirmResetDialog.value = false
  router.push('/')
}
</script>

<template>
  <v-app-bar>
    <v-app-bar-title>
      <!-- Clickable logo with tooltip -->
      <v-tooltip location="bottom">
        <template #activator="{ props }">
          <div
            v-bind="props"
            class="logo-container"
            @click="handleLogoClick"
            role="button"
            tabindex="0"
            @keydown.enter="handleLogoClick"
            @keydown.space="handleLogoClick"
          >
            <img src="/logo.svg" alt="Phentrieve Logo" class="logo" />
            <span class="ml-2">Phentrieve</span>
          </div>
        </template>
        <span>{{ $t('app.logoTooltip', 'Click to reset application') }}</span>
      </v-tooltip>
    </v-app-bar-title>
  </v-app-bar>

  <!-- Reset Confirmation Dialog (replaces native confirm()) -->
  <v-dialog v-model="confirmResetDialog" max-width="400">
    <v-card>
      <v-card-title class="text-h6">
        {{ $t('app.confirmResetTitle', 'Reset Application?') }}
      </v-card-title>
      <v-card-text>
        {{ $t('app.confirmResetMessage', 'This will clear all conversation history and collected phenotypes. This action cannot be undone.') }}
      </v-card-text>
      <v-card-actions>
        <v-spacer />
        <v-btn variant="text" @click="confirmResetDialog = false">
          {{ $t('common.cancel', 'Cancel') }}
        </v-btn>
        <v-btn color="error" variant="flat" @click="handleConfirmedReset">
          {{ $t('common.reset', 'Reset') }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<style scoped>
.logo-container {
  cursor: pointer;
  display: flex;
  align-items: center;
  transition: opacity 0.2s;
}

.logo-container:hover {
  opacity: 0.8;
}

.logo-container:focus {
  outline: 2px solid var(--v-theme-primary);
  outline-offset: 2px;
  border-radius: 4px;
}

.logo {
  height: 40px;
  width: auto;
}
</style>
```

### Phase 5: Migration & Cleanup (KISS Principle)

**Objective**: Smoothly transition from component state to Pinia store and address tech debt

#### 5.1 Update QueryInterface.vue

**Remove from data()** (lines 872-941):
- `queryHistory`
- `collectedPhenotypes`
- All phenotype-related state

**Replace with**:

```javascript
import { useConversationStore } from '../stores/conversation'

export default {
  setup() {
    const conversationStore = useConversationStore()
    return { conversationStore }
  },

  // Remove data properties related to conversation
  // Update methods to use conversationStore instead
}
```

#### 5.2 Update Method Calls

**Old**:
```javascript
this.queryHistory.unshift(historyItem)
this.collectedPhenotypes.push(phenotype)
```

**New**:
```javascript
this.conversationStore.addQuery(historyItem)
this.conversationStore.addPhenotype(phenotype)
```

#### 5.3 Update Template References

**Old**:
```vue
v-for="(item, index) in queryHistory"
:collected-phenotypes="collectedPhenotypes"
```

**New**:
```vue
<!-- Handled by virtual scroller now -->
:collected-phenotypes="conversationStore.collectedPhenotypes"
```

#### 5.4 Tech Debt Cleanup: Refactor disclaimer.js

**Objective**: Standardize persistence patterns across all stores

The existing `disclaimer.js` store manually handles localStorage (lines 26-57). Refactor to use the persistence plugin for consistency:

**Current** (manual localStorage):
```javascript
function initialize() {
  const stored = localStorage.getItem(STORAGE_KEY)
  // ... manual parsing
}

function saveAcknowledgment() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
}
```

**Refactored** (using persist plugin):
```javascript
export const useDisclaimerStore = defineStore('disclaimer', () => {
  const isAcknowledged = ref(false)
  const acknowledgmentTimestamp = ref(null)

  // ... computed and actions (remove manual localStorage calls)

  function saveAcknowledgment() {
    isAcknowledged.value = true
    acknowledgmentTimestamp.value = new Date().toISOString()
    // No manual localStorage - plugin handles it
  }

  function reset() {
    isAcknowledged.value = false
    acknowledgmentTimestamp.value = null
    // No manual localStorage - plugin handles it
  }

  return { /* ... */ }
}, {
  persist: {
    key: 'phentrieve-disclaimer',
    storage: localStorage,
    pick: ['isAcknowledged', 'acknowledgmentTimestamp']
  }
})
```

## Performance Impact Analysis

### Before Optimization

| Metric | Value |
|--------|-------|
| DOM nodes (100 queries) | ~5,000-10,000 |
| Memory usage | ~150-200 MB |
| Initial render time | 2-5 seconds |
| Scroll lag | Noticeable stuttering |
| Navigation state loss | 100% data loss |

### After Optimization

| Metric | Value | Improvement |
|--------|-------|-------------|
| DOM nodes (100 queries) | ~200-400 (visible only) | 95% reduction |
| Memory usage | ~50-80 MB | 60-70% reduction |
| Initial render time | <500ms | 75-90% faster |
| Scroll lag | Smooth 60fps | Eliminated |
| Navigation state loss | 0% (persisted) | 100% improvement |

## Implementation Checklist

### Phase 1: Pinia Store Setup
- [ ] Create `stores/conversation.js` with persist v4 syntax
- [ ] Add unit tests for store actions
- [ ] Test persistence across page reloads

### Phase 2: Virtual Scrolling
- [ ] Install `vue-virtual-scroller@next` OR `virtua`
- [ ] Replace v-for with DynamicScroller
- [ ] Ensure items have unique `id` field
- [ ] Test scroll performance with 100+ items
- [ ] Verify dynamic height calculation
- [ ] Test on mobile devices

### Phase 3: User Settings
- [ ] Create ConversationSettings.vue component
- [ ] Use v-dialog for confirmations (not native confirm())
- [ ] Add settings button to QueryInterface layout
- [ ] Implement history length slider
- [ ] Add storage statistics display
- [ ] Add i18n translations

### Phase 4: Logo Reset
- [ ] Add click handler to logo
- [ ] Implement v-dialog confirmation (not native confirm())
- [ ] Add tooltip with hover hint
- [ ] Test keyboard navigation (a11y)
- [ ] Add i18n translations

### Phase 5: Migration & Cleanup
- [ ] Refactor QueryInterface.vue to use store
- [ ] Update ResultsDisplay.vue props
- [ ] Remove duplicate state management
- [ ] Update all method calls
- [ ] Refactor disclaimer.js to use persist plugin
- [ ] Run full E2E tests
- [ ] Performance testing before/after

## Testing Strategy

### Unit Tests

```javascript
// stores/conversation.spec.js
import { setActivePinia, createPinia } from 'pinia'
import { useConversationStore } from './conversation'

describe('Conversation Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('adds query to history with unique id', () => {
    const store = useConversationStore()
    store.addQuery({ query: 'test', loading: true })
    expect(store.queryHistory.length).toBe(1)
    expect(store.queryHistory[0].id).toBeDefined()
  })

  it('trims history when exceeds max length', () => {
    const store = useConversationStore()
    store.maxHistoryLength = 5

    for (let i = 0; i < 10; i++) {
      store.addQuery({ query: `test ${i}`, loading: false })
    }

    expect(store.queryHistory.length).toBe(5) // Trimmed to max
  })

  it('prevents duplicate phenotypes', () => {
    const store = useConversationStore()
    const phenotype = { hpo_id: 'HP:0000001', label: 'Test' }

    store.addPhenotype(phenotype)
    store.addPhenotype(phenotype)

    expect(store.collectedPhenotypes.length).toBe(1)
  })

  it('resets all state', () => {
    const store = useConversationStore()
    store.addQuery({ query: 'test' })
    store.addPhenotype({ hpo_id: 'HP:0000001' })

    store.resetAll()

    expect(store.queryHistory.length).toBe(0)
    expect(store.collectedPhenotypes.length).toBe(0)
  })

  it('updates query response by id', () => {
    const store = useConversationStore()
    store.addQuery({ query: 'test', loading: true })
    const queryId = store.queryHistory[0].id

    store.updateQueryResponse(queryId, { results: [] }, null)

    expect(store.queryHistory[0].loading).toBe(false)
    expect(store.queryHistory[0].response).toEqual({ results: [] })
  })
})
```

### E2E Tests

```javascript
// e2e/conversation-persistence.spec.js
describe('Conversation Persistence', () => {
  it('persists conversation across page reloads', () => {
    cy.visit('/')
    cy.get('[data-testid="query-input"]').type('test query')
    cy.get('[data-testid="search-button"]').click()
    cy.get('[data-testid="query-history"]').should('have.length', 1)

    // Reload page
    cy.reload()

    // Conversation should still be there
    cy.get('[data-testid="query-history"]').should('have.length', 1)
  })

  it('persists conversation across navigation', () => {
    cy.visit('/')
    cy.get('[data-testid="query-input"]').type('test query')
    cy.get('[data-testid="search-button"]').click()

    // Navigate to FAQ
    cy.get('[data-testid="nav-faq"]').click()
    cy.url().should('include', '/faq')

    // Navigate back
    cy.get('[data-testid="nav-home"]').click()

    // Conversation should still be there
    cy.get('[data-testid="query-history"]').should('have.length', 1)
  })

  it('resets app via logo click with confirmation dialog', () => {
    cy.visit('/')
    cy.get('[data-testid="query-input"]').type('test query')
    cy.get('[data-testid="search-button"]').click()

    // Click logo
    cy.get('[data-testid="app-logo"]').click()

    // Vuetify dialog should appear
    cy.get('.v-dialog').should('be.visible')
    cy.get('.v-dialog').contains('Reset').click()

    // Conversation should be cleared
    cy.get('[data-testid="query-history"]').should('have.length', 0)
  })
})
```

## Migration Path

### Option A: Big Bang Migration (Recommended)
- Implement all phases in single PR
- Less risk of state inconsistency
- Easier to test as complete feature
- **Estimated time**: 2-3 days

### Option B: Incremental Migration
1. Week 1: Pinia store setup + persistence
2. Week 2: Virtual scrolling implementation
3. Week 3: Settings UI + logo reset
- More gradual, lower risk
- Can deploy incrementally
- **Estimated time**: 2-3 weeks

**Recommendation**: Option A for alpha release timeline

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| LocalStorage limits (5MB) | High | Add storage monitoring, warn at 80% capacity |
| Virtual scroller bugs | Medium | Extensive testing, fallback to pagination |
| State migration issues | High | Add migration logic for existing users |
| Performance on slow devices | Medium | Make virtual scrolling optional via settings |
| Breaking existing functionality | High | Comprehensive E2E test coverage |

## Rollback Plan

If issues arise post-deployment:

1. **Feature flag**: Add `VITE_ENABLE_CONVERSATION_PERSISTENCE` env var
2. **Fallback**: Keep old component state logic as backup
3. **Data migration**: Export localStorage data before clearing
4. **Monitoring**: Track localStorage errors and virtual scroller performance

## Success Metrics

- [ ] DOM nodes reduced by >90% for 100+ query conversations
- [ ] Memory usage reduced by >60%
- [ ] Zero navigation state loss (100% persistence)
- [ ] Smooth scrolling (60fps) with 200+ queries
- [ ] User settings adoption >50% within first week
- [ ] Zero critical bugs in first week post-deployment

## Future Enhancements (Post-Alpha)

1. **Cloud Sync**: Sync conversations across devices (requires backend)
2. **Conversation Export**: Export as JSON/CSV for analysis
3. **Conversation Search**: Full-text search within history
4. **Conversation Bookmarking**: Mark important queries
5. **Conversation Sharing**: Generate shareable links
6. **Conversation Analytics**: Show usage patterns and stats

## References

- [Vue Virtual Scroller Documentation](https://github.com/Akryum/vue-virtual-scroller)
- [Pinia Plugin Persistedstate v4](https://prazdevs.github.io/pinia-plugin-persistedstate/)
- [Vue.js Performance Best Practices](https://vuejs.org/guide/best-practices/performance.html)
- [Pinia Core Concepts](https://pinia.vuejs.org/core-concepts/)
- [Modern Virtual Scrolling (2024)](https://blog.logrocket.com/create-performant-virtual-scrolling-list-vuejs/)

---

**Plan created**: 2025-11-18
**Last updated**: 2025-01-24
**Reviewed by**: Senior Developer
**Next review**: After Phase 1 implementation
