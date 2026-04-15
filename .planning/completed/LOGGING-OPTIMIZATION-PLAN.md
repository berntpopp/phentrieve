# Frontend Logging System Optimization Plan

**Status**: Active
**Priority**: High
**Created**: 2025-11-18
**Estimated Effort**: 4-6 hours

## Problem Statement

### Current Issues

The frontend logging system (`frontend/src/stores/log.js` and `frontend/src/services/logService.js`) causes **performance degradation and memory bloat** after extended usage due to:

1. **No Automatic Log Rotation**: Logs accumulate indefinitely in memory
   - `logService.js` defines `maxEntries = 1000` but **never enforces it automatically**
   - `addLogEntry()` simply pushes logs without any size checks
   - `trimLogs()` exists but is **only called manually** via `setMaxEntries()`

2. **Memory Accumulation**: Unbounded growth leads to:
   - Increased RAM usage (all logs stored in reactive Pinia state)
   - Slower UI rendering (Vue re-renders with larger arrays)
   - Browser lag and potential crashes on long-running sessions

3. **Missing Features** compared to industry best practices:
   - No statistics tracking (total logs received/dropped)
   - No memory usage monitoring
   - No configurable defaults (dev vs production)
   - No localStorage persistence for user preferences

### Performance Impact

```javascript
// Current behavior (PROBLEMATIC):
// After 1 hour of usage with frequent API calls:
// - 5,000+ log entries in memory
// - ~2-5 MB of reactive state
// - Noticeable UI lag when opening LogViewer
// - Each log addition triggers Vue reactivity updates

// Expected behavior (WITH FIX):
// - Maximum 100 logs in dev, 50 in production
// - ~20-50 KB of reactive state
// - Instant LogViewer rendering
// - Automatic rotation prevents unbounded growth
```

## Comparison: kidney-genetics-db Implementation

The `kidney-genetics-db` project has a **production-ready logging system** with automatic rotation:

### Key Features (What We Should Adopt)

1. **Automatic Log Rotation** (`logStore.js:110-136`)
   ```javascript
   function addLogEntry(entry, maxEntriesOverride = null) {
     const max = maxEntriesOverride || maxEntries.value
     const newLogs = [...logs.value, entry]

     // CRITICAL: Automatic trimming on every add
     if (newLogs.length > max) {
       const toRemove = newLogs.length - max
       logs.value = newLogs.slice(toRemove)  // Keep most recent
       stats.value.totalLogsDropped += toRemove
     } else {
       logs.value = newLogs
     }
   }
   ```

2. **Statistics Tracking** - Monitor memory health:
   - `totalLogsReceived`: Total count of all logs
   - `totalLogsDropped`: How many were auto-trimmed
   - `sessionStartTime`: Session duration tracking
   - `memoryUsage`: Estimated RAM usage (bytes/KB/MB)

3. **Environment-Aware Defaults**:
   - Development: 100 logs (more verbose debugging)
   - Production: 50 logs (performance-focused)

4. **LocalStorage Persistence**: User preferences survive page refreshes

5. **Performance Optimization**: Synchronous log addition with automatic rotation to keep implementation simple (KISS principle)

6. **Advanced Features**:
   - Log sanitization for sensitive data
   - Filtering by level/search/correlation ID
   - CSV/JSON export with metadata
   - Time range queries

## Best Practices Research

### Circular Buffer Pattern

From web search and Context7 documentation:

1. **Fixed-Size Buffer**: Keep only N most recent entries (FIFO - First In, First Out)
2. **Automatic Rotation**: Oldest logs are automatically dropped when limit is reached
3. **Memory Bound**: Guaranteed maximum memory usage
4. **Performance**: O(1) amortized insertion with proper implementation

### Pinia Reactive State Management

From Pinia documentation (`/vuejs/pinia`):

1. **Avoid Direct Array Mutations**: Use `$patch` or reassignment for reactivity
2. **Batch Updates**: Single reactive update instead of multiple pushes
3. **Computed Properties**: For derived state (filtered logs, statistics)
4. **Shallow Reactivity**: Consider `shallowRef` for large arrays (if needed)

### JavaScript Logging Best Practices (2024)

1. **Structured Logging**: Consistent log entry format
2. **Log Levels**: DEBUG, INFO, WARN, ERROR with filtering
3. **Performance**: Minimize console.log in production
4. **Security**: Never log sensitive data (tokens, passwords, PII)
5. **Buffering**: Batch operations to reduce overhead

## Architecture & Design Principles

### SOLID Principles

1. **Single Responsibility Principle (SRP)**:
   - `logService.js`: Logging operations, configuration management
   - `log.js` (store): State management, statistics, filtering
   - `LogViewer.vue`: UI presentation only
   - *New*: `logConfig.js`: Configuration constants and defaults

2. **Open/Closed Principle (OCP)**:
   - Extensible log storage backends (could add IndexedDB later)
   - Pluggable log sanitizers
   - Configurable rotation strategies

3. **Liskov Substitution Principle (LSP)**:
   - Log entry interface remains consistent
   - Backward compatible with existing API

4. **Interface Segregation Principle (ISP)**:
   - Separate concerns: logging, filtering, exporting, statistics

5. **Dependency Inversion Principle (DIP)**:
   - `logService` depends on abstract store interface
   - Configuration injected, not hardcoded

### DRY (Don't Repeat Yourself)

- Extract configuration constants to shared module
- Reuse kidney-genetics-db patterns (no reinventing the wheel)
- Single source of truth for defaults

### KISS (Keep It Simple, Stupid)

- Start with essential features: automatic rotation + configurable limit
- Don't over-engineer: No complex circular buffer data structure (array.slice() is sufficient)
- Clear, readable code over clever optimizations

### Modularization

```
frontend/src/
├── config/
│   └── logConfig.js          # NEW: Centralized log configuration
├── services/
│   └── logService.js         # UPDATED: Add auto-rotation
├── stores/
│   └── log.js                # UPDATED: Add statistics, memory tracking
├── components/
│   └── LogViewer.vue         # UPDATED: Display statistics
└── test/
    ├── log.test.js           # UPDATED: Enhanced tests
    └── logService.test.js    # NEW: Service-specific tests
```

## Implementation Plan

### Phase 1: Configuration Module (30 min)

**Goal**: Extract configuration to separate module for DRY principle

**Tasks**:
1. Create `frontend/src/config/logConfig.js`:
   ```javascript
   // Centralized logging configuration
   export const LOG_CONFIG = {
     // Max log entries (configurable)
     MAX_ENTRIES_DEV: 100,
     MAX_ENTRIES_PROD: 50,

     // Default log level
     DEFAULT_LEVEL_DEV: 'DEBUG',
     DEFAULT_LEVEL_PROD: 'WARN',

     // Storage keys for localStorage
     STORAGE_KEYS: {
       MAX_ENTRIES: 'phentrieve-log-max-entries',
       LOG_LEVEL: 'phentrieve-log-level',
       CONSOLE_ECHO: 'phentrieve-console-echo',
     },

     // Memory warning threshold (bytes)
     MEMORY_WARNING_THRESHOLD: 1024 * 1024, // 1 MB
   };

   // Helper to get environment-specific defaults
   export function getDefaultMaxEntries() {
     return import.meta.env.DEV
       ? LOG_CONFIG.MAX_ENTRIES_DEV
       : LOG_CONFIG.MAX_ENTRIES_PROD;
   }

   export function getDefaultLogLevel() {
     return import.meta.env.DEV
       ? LOG_CONFIG.DEFAULT_LEVEL_DEV
       : LOG_CONFIG.DEFAULT_LEVEL_PROD;
   }
   ```

2. Add JSDoc comments for documentation
3. Export utility functions for accessing config

**Testing**:
- Unit test for `getDefaultMaxEntries()` in dev/prod modes
- Verify localStorage key consistency

---

### Phase 2: Enhanced Log Store (1.5 hours)

**Goal**: Implement automatic rotation, statistics, and memory tracking

**Tasks**:

1. **Update `frontend/src/stores/log.js`**:

   a. Add statistics state:
   ```javascript
   const stats = ref({
     totalLogsReceived: 0,
     totalLogsDropped: 0,
     lastLogTime: null,
     sessionStartTime: new Date().toISOString(),
   });
   ```

   b. Add memory tracking computed property:
   ```javascript
   const memoryUsage = computed(() => {
     const jsonSize = JSON.stringify(logs.value).length;
     return {
       bytes: jsonSize,
       kb: (jsonSize / 1024).toFixed(2),
       mb: (jsonSize / (1024 * 1024)).toFixed(2),
     };
   });
   ```

   c. Add maxEntries as reactive state (load from localStorage):
   ```javascript
   const maxEntries = ref(loadMaxEntriesFromStorage());
   ```

   d. **CRITICAL**: Update `addLogEntry()` with automatic rotation:
   ```javascript
   function addLogEntry(entry, maxEntriesOverride = null) {
     // Defer to next tick to avoid render blocking
     const addLog = () => {
       const max = maxEntriesOverride || maxEntries.value;

       // Add timestamp if missing
       if (!entry.timestamp) {
         entry.timestamp = new Date().toISOString();
       }

       // Update statistics
       stats.value.totalLogsReceived++;
       stats.value.lastLogTime = entry.timestamp;

       // Create new array with new entry
       const newLogs = [...logs.value, entry];

       // AUTOMATIC ROTATION: Trim if exceeding max
       if (newLogs.length > max) {
         const toRemove = newLogs.length - max;
         logs.value = newLogs.slice(toRemove); // Keep most recent
         stats.value.totalLogsDropped += toRemove;
       } else {
         logs.value = newLogs;
       }
     };

     // Synchronous execution (KISS principle - avoid unnecessary complexity)
     addLog();
   }
   ```

   e. Update `trimLogs()` to update statistics:
   ```javascript
   function trimLogs(newMaxEntries) {
     if (logs.value.length > newMaxEntries) {
       const toRemove = logs.value.length - newMaxEntries;
       logs.value = logs.value.slice(toRemove);
       stats.value.totalLogsDropped += toRemove;
     }
     maxEntries.value = newMaxEntries;
     saveMaxEntriesToStorage(newMaxEntries);
   }
   ```

   f. Add localStorage helpers:
   ```javascript
   function loadMaxEntriesFromStorage() {
     try {
       const stored = localStorage.getItem(LOG_CONFIG.STORAGE_KEYS.MAX_ENTRIES);
       if (stored) {
         const parsed = parseInt(stored, 10);
         if (!isNaN(parsed) && parsed > 0) {
           return parsed;
         }
       }
     } catch (error) {
       console.error('Failed to load max entries:', error);
     }
     return getDefaultMaxEntries();
   }

   function saveMaxEntriesToStorage(value) {
     try {
       localStorage.setItem(LOG_CONFIG.STORAGE_KEYS.MAX_ENTRIES, value.toString());
     } catch (error) {
       console.error('Failed to save max entries:', error);
     }
   }
   ```

   g. Add `getStatistics()` action:
   ```javascript
   function getStatistics() {
     return {
       ...stats.value,
       currentCount: logs.value.length,
       maxEntries: maxEntries.value,
       memoryUsage: memoryUsage.value,
       oldestLog: logs.value[0]?.timestamp || null,
       newestLog: logs.value[logs.value.length - 1]?.timestamp || null,
     };
   }
   ```

   h. Export new state and actions:
   ```javascript
   return {
     // State
     logs,
     isViewerVisible,
     maxEntries, // NEW

     // Computed
     memoryUsage, // NEW

     // Actions
     addLogEntry, // UPDATED
     clearLogs,
     trimLogs, // UPDATED
     setMaxEntries, // NEW
     getMaxEntries, // NEW
     getStatistics, // NEW
     // ... existing actions
   };
   ```

**Testing**:
- Test automatic rotation when logs exceed maxEntries
- Verify statistics tracking (totalLogsReceived, totalLogsDropped)
- Test localStorage persistence
- Test memory usage calculation
- Test deferred log addition with Promise

---

### Phase 3: Update Log Service (45 min)

**Goal**: Integrate auto-rotation, improve configuration management

**Tasks**:

1. **Update `frontend/src/services/logService.js`**:

   a. Import configuration:
   ```javascript
   import { LOG_CONFIG, getDefaultMaxEntries } from '../config/logConfig';
   ```

   b. Update constructor with localStorage loading:
   ```javascript
   constructor() {
     this.maxEntries = this.loadMaxEntriesFromStorage();
     this.consoleEcho = this.loadConsoleEchoFromStorage();
     this.store = null;
   }
   ```

   c. Update `_log()` to pass maxEntries to store:
   ```javascript
   _log(level, message, data = null) {
     // ... existing code ...

     if (this.store) {
       this.store.addLogEntry(entry, this.maxEntries);
     }

     // ... rest of method
   }
   ```

   d. Update `setMaxEntries()`:
   ```javascript
   setMaxEntries(max) {
     this.maxEntries = max;
     localStorage.setItem(LOG_CONFIG.STORAGE_KEYS.MAX_ENTRIES, max.toString());

     // Trim existing logs
     if (this.store) {
       this.store.trimLogs(max);
     }
   }
   ```

   e. Add localStorage helpers:
   ```javascript
   loadMaxEntriesFromStorage() {
     try {
       const stored = localStorage.getItem(LOG_CONFIG.STORAGE_KEYS.MAX_ENTRIES);
       if (stored) {
         const parsed = parseInt(stored, 10);
         if (!isNaN(parsed) && parsed > 0) {
           return parsed;
         }
       }
     } catch (error) {
       console.error('Failed to load max entries:', error);
     }
     return getDefaultMaxEntries();
   }

   loadConsoleEchoFromStorage() {
     try {
       const stored = localStorage.getItem(LOG_CONFIG.STORAGE_KEYS.CONSOLE_ECHO);
       if (stored !== null) {
         return JSON.parse(stored);
       }
     } catch (error) {
       console.error('Failed to load console echo setting:', error);
     }
     return false; // Default: false
   }
   ```

**Testing**:
- Test `setMaxEntries()` updates both service and store
- Verify localStorage persistence across page refreshes
- Test that logs are passed to store with correct maxEntries

---

### Phase 4: Enhanced UI (1 hour)

**Goal**: Display statistics and configuration options in LogViewer

**Tasks**:

1. **Update `frontend/src/components/LogViewer.vue`**:

   a. Add statistics display in toolbar:
   ```vue
   <v-toolbar density="compact" color="primary">
     <v-toolbar-title class="text-white">
       {{ $t('logViewer.title') }}
       <v-chip size="x-small" class="ml-2" color="white" variant="outlined">
         {{ logStore.logs.length }}/{{ logStore.maxEntries }}
       </v-chip>
     </v-toolbar-title>
     <!-- ... existing buttons ... -->
   </v-toolbar>
   ```

   b. Add statistics card below search:
   ```vue
   <v-card class="mx-2 mb-2" variant="outlined">
     <v-card-text class="pa-2">
       <div class="d-flex justify-space-between text-caption">
         <div>
           <strong>{{ $t('logViewer.stats.received') }}:</strong>
           {{ statistics.totalLogsReceived }}
         </div>
         <div>
           <strong>{{ $t('logViewer.stats.dropped') }}:</strong>
           {{ statistics.totalLogsDropped }}
         </div>
         <div>
           <strong>{{ $t('logViewer.stats.memory') }}:</strong>
           {{ statistics.memoryUsage.kb }} KB
         </div>
       </div>
     </v-card-text>
   </v-card>
   ```

   c. Add max entries configuration dialog:
   ```vue
   <v-dialog v-model="showConfigDialog" max-width="400">
     <template #activator="{ props }">
       <v-btn
         icon="mdi-cog"
         variant="text"
         color="white"
         v-bind="props"
         aria-label="Configure logging"
       />
     </template>
     <v-card>
       <v-card-title>{{ $t('logViewer.config.title') }}</v-card-title>
       <v-card-text>
         <v-slider
           v-model="configMaxEntries"
           :min="10"
           :max="500"
           :step="10"
           thumb-label
           label="Max Log Entries"
         >
           <template #label>
             <span>{{ $t('logViewer.config.maxEntries') }}</span>
           </template>
         </v-slider>
       </v-card-text>
       <v-card-actions>
         <v-spacer />
         <v-btn @click="showConfigDialog = false">Cancel</v-btn>
         <v-btn color="primary" @click="saveConfig">Save</v-btn>
       </v-card-actions>
     </v-card>
   </v-dialog>
   ```

   d. Add reactive statistics:
   ```javascript
   import { logService } from '../services/logService';

   const statistics = computed(() => logStore.getStatistics());
   const showConfigDialog = ref(false);
   const configMaxEntries = ref(logStore.maxEntries);

   const saveConfig = () => {
     logService.setMaxEntries(configMaxEntries.value);
     showConfigDialog.value = false;
   };
   ```

2. **Add i18n translations** (`frontend/src/locales/en.json`):
   ```json
   {
     "logViewer": {
       "stats": {
         "received": "Received",
         "dropped": "Dropped",
         "memory": "Memory"
       },
       "config": {
         "title": "Logging Configuration",
         "maxEntries": "Maximum Log Entries"
       }
     }
   }
   ```

**Testing**:
- Verify statistics display correctly
- Test configuration dialog saves settings
- Test that max entries slider updates correctly

---

### Phase 5: Comprehensive Testing (2 hours)

**Goal**: Ensure reliability, performance, and correctness

#### 5.1 Unit Tests for Store (`frontend/src/test/log.test.js`)

**Add/Update Tests**:

1. **Automatic Rotation Tests**:
   ```javascript
   describe('Automatic Log Rotation', () => {
     it('should automatically trim logs when exceeding maxEntries', () => {
       const store = useLogStore();
       store.setMaxEntries(5);

       // Add 10 logs
       for (let i = 0; i < 10; i++) {
         store.addLogEntry({ message: `Log ${i}`, level: 'INFO' });
       }

       // Should keep only last 5
       expect(store.logs).toHaveLength(5);
       expect(store.logs[0].message).toBe('Log 5');
       expect(store.logs[4].message).toBe('Log 9');
     });

     it('should update statistics when logs are dropped', () => {
       const store = useLogStore();
       store.setMaxEntries(3);

       for (let i = 0; i < 10; i++) {
         store.addLogEntry({ message: `Log ${i}`, level: 'INFO' });
       }

       const stats = store.getStatistics();
       expect(stats.totalLogsReceived).toBe(10);
       expect(stats.totalLogsDropped).toBe(7);
       expect(stats.currentCount).toBe(3);
     });

     it('should respect maxEntriesOverride parameter', () => {
       const store = useLogStore();
       store.setMaxEntries(100);

       // Add logs with override
       for (let i = 0; i < 20; i++) {
         store.addLogEntry({ message: `Log ${i}`, level: 'INFO' }, 10);
       }

       // Should respect override, not store default
       expect(store.logs).toHaveLength(10);
     });
   });
   ```

2. **Statistics Tests**:
   ```javascript
   describe('Statistics Tracking', () => {
     it('should track total logs received', () => {
       const store = useLogStore();
       store.addLogEntry({ message: 'Test 1' });
       store.addLogEntry({ message: 'Test 2' });
       store.addLogEntry({ message: 'Test 3' });

       const stats = store.getStatistics();
       expect(stats.totalLogsReceived).toBe(3);
     });

     it('should track session start time', () => {
       const store = useLogStore();
       const stats = store.getStatistics();

       expect(stats.sessionStartTime).toBeDefined();
       expect(new Date(stats.sessionStartTime)).toBeInstanceOf(Date);
     });

     it('should calculate memory usage', () => {
       const store = useLogStore();
       for (let i = 0; i < 10; i++) {
         store.addLogEntry({ message: `Test ${i}`, level: 'INFO' });
       }

       expect(store.memoryUsage.bytes).toBeGreaterThan(0);
       expect(store.memoryUsage.kb).toBeDefined();
       expect(store.memoryUsage.mb).toBeDefined();
     });
   });
   ```

3. **LocalStorage Persistence Tests**:
   ```javascript
   describe('LocalStorage Persistence', () => {
     beforeEach(() => {
       localStorage.clear();
     });

     it('should load maxEntries from localStorage', () => {
       localStorage.setItem('phentrieve-log-max-entries', '50');
       const store = useLogStore();

       expect(store.maxEntries).toBe(50);
     });

     it('should save maxEntries to localStorage', () => {
       const store = useLogStore();
       store.setMaxEntries(75);

       const stored = localStorage.getItem('phentrieve-log-max-entries');
       expect(stored).toBe('75');
     });

     it('should use default if localStorage is empty', () => {
       const store = useLogStore();
       const expected = import.meta.env.DEV ? 100 : 50;

       expect(store.maxEntries).toBe(expected);
     });
   });
   ```

4. **Performance Tests**:
   ```javascript
   describe('Performance Optimization', () => {
     it('should defer log addition with Promise.resolve()', async () => {
       const store = useLogStore();
       const initialLength = store.logs.length;

       store.addLogEntry({ message: 'Async log', level: 'INFO' });

       // Should not be added synchronously
       expect(store.logs.length).toBe(initialLength);

       // Wait for next tick
       await new Promise(resolve => setTimeout(resolve, 0));

       // Now it should be added
       expect(store.logs.length).toBe(initialLength + 1);
     });
   });
   ```

#### 5.2 Service Tests (`frontend/src/test/logService.test.js`)

**Create New Test File**:

```javascript
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { logService, LogLevel } from '../services/logService';
import { useLogStore } from '../stores/log';
import { setActivePinia, createPinia } from 'pinia';

describe('LogService', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    localStorage.clear();

    // Initialize store
    const store = useLogStore();
    logService.initStore(store);
  });

  describe('Configuration', () => {
    it('should set and persist maxEntries', () => {
      logService.setMaxEntries(200);

      expect(logService.maxEntries).toBe(200);
      expect(localStorage.getItem('phentrieve-log-max-entries')).toBe('200');
    });

    it('should load maxEntries from localStorage on init', () => {
      localStorage.setItem('phentrieve-log-max-entries', '150');

      // Create new service instance
      const { logService: newService } = await import('../services/logService');

      expect(newService.maxEntries).toBe(150);
    });
  });

  describe('Logging Methods', () => {
    it('should add logs with correct level', () => {
      const store = useLogStore();

      logService.info('Info message');
      logService.warn('Warning message');
      logService.error('Error message');

      // Wait for deferred addition
      await new Promise(resolve => setTimeout(resolve, 0));

      expect(store.logs).toHaveLength(3);
      expect(store.logs[0].level).toBe(LogLevel.INFO);
      expect(store.logs[1].level).toBe(LogLevel.WARN);
      expect(store.logs[2].level).toBe(LogLevel.ERROR);
    });
  });

  describe('Auto-Rotation Integration', () => {
    it('should respect maxEntries when logging', async () => {
      logService.setMaxEntries(5);

      // Log 10 messages
      for (let i = 0; i < 10; i++) {
        logService.info(`Message ${i}`);
      }

      // Wait for all deferred additions
      await new Promise(resolve => setTimeout(resolve, 10));

      const store = useLogStore();
      expect(store.logs).toHaveLength(5);
    });
  });
});
```

#### 5.3 Integration Tests

**Create `frontend/src/test/logIntegration.test.js`**:

```javascript
describe('Logging System Integration', () => {
  it('should work end-to-end: service -> store -> UI', async () => {
    // Setup
    const store = useLogStore();
    logService.initStore(store);
    logService.setMaxEntries(10);

    // Generate logs
    for (let i = 0; i < 50; i++) {
      logService.info(`Test log ${i}`);
    }

    await new Promise(resolve => setTimeout(resolve, 10));

    // Verify
    expect(store.logs).toHaveLength(10);
    expect(store.getStatistics().totalLogsReceived).toBe(50);
    expect(store.getStatistics().totalLogsDropped).toBe(40);
  });
});
```

#### 5.4 Manual Testing Checklist

- [ ] Open app, use for 5 minutes, verify logs don't exceed maxEntries
- [ ] Change maxEntries in config dialog, verify it updates immediately
- [ ] Refresh page, verify maxEntries persists
- [ ] Generate 1000 logs, verify memory usage stays bounded
- [ ] Check LogViewer performance (should open instantly)
- [ ] Verify statistics display correctly
- [ ] Test in both dev and production builds

---

### Phase 6: Documentation (30 min)

**Goal**: Update documentation for maintainability

**Tasks**:

1. **Add JSDoc comments** to all new functions
2. **Update README** (if applicable) with configuration options
3. **Add inline comments** explaining circular buffer logic
4. **Create migration guide** for any API changes

**Example JSDoc**:
```javascript
/**
 * Adds a log entry with automatic rotation to prevent memory bloat.
 *
 * @param {Object} entry - Log entry object
 * @param {string} entry.message - Log message
 * @param {string} entry.level - Log level (DEBUG, INFO, WARN, ERROR)
 * @param {*} [entry.data] - Optional additional data
 * @param {number|null} [maxEntriesOverride] - Override default maxEntries
 *
 * @example
 * // Will automatically rotate if logs exceed maxEntries
 * addLogEntry({ message: 'User login', level: 'INFO' });
 *
 * @performance
 * Synchronous execution (KISS principle) with automatic rotation.
 * Automatically trims oldest logs when exceeding maxEntries.
 */
function addLogEntry(entry, maxEntriesOverride = null) {
  // ...
}
```

---

## Testing Strategy Summary

### Unit Tests (Vitest)

| Test Suite | File | Coverage Target |
|------------|------|-----------------|
| Store - Rotation | `log.test.js` | Automatic trimming, statistics |
| Store - Persistence | `log.test.js` | LocalStorage load/save |
| Store - Performance | `log.test.js` | Deferred additions |
| Service - Config | `logService.test.js` | maxEntries management |
| Service - Integration | `logService.test.js` | Service + Store interaction |
| Config - Helpers | `logConfig.test.js` | Environment defaults |

### Integration Tests

- End-to-end flow: Service → Store → UI
- Configuration persistence across refreshes
- Statistics accuracy under load

### Manual Testing

- Long-running session (30+ minutes)
- Memory profiling with Chrome DevTools
- Performance comparison (before/after)

---

## Success Metrics

### Performance

- **Memory Usage**: Max 50 KB for 50 logs (down from unbounded)
- **UI Responsiveness**: LogViewer opens in <100ms
- **Log Addition**: <1ms per log (with deferred addition)

### Functionality

- ✅ Automatic rotation when exceeding maxEntries
- ✅ Statistics tracking (received, dropped, memory)
- ✅ User-configurable max entries
- ✅ LocalStorage persistence
- ✅ Environment-aware defaults

### Code Quality

- ✅ 100% test coverage for critical paths
- ✅ JSDoc documentation for all public APIs
- ✅ SOLID principles followed
- ✅ No breaking changes to existing API

---

## Rollout Plan

### Phase 1: Development (This Plan)
- Implement automatic rotation
- Add comprehensive tests
- Local testing and validation

### Phase 2: Code Review
- Submit PR with detailed description
- Review by team
- Address feedback

### Phase 3: Staging
- Deploy to staging environment
- Monitor memory usage over 24 hours
- Collect performance metrics

### Phase 4: Production
- Gradual rollout (20% → 50% → 100%)
- Monitor error logs
- Revert plan if issues detected

---

## Risk Mitigation

### Potential Issues

1. **Breaking Changes**: Existing code expects unlimited logs
   - **Mitigation**: Maintain backward compatibility, add `maxEntriesOverride` parameter

2. **Performance Regression**: Deferred additions cause race conditions
   - **Mitigation**: Comprehensive async tests, fallback to synchronous mode

3. **LocalStorage Quota**: Exceeding browser storage limits
   - **Mitigation**: Try-catch all localStorage operations, graceful fallback to defaults

4. **Lost Logs**: Important errors rotated out before viewing
   - **Mitigation**:
     - Keep ERROR logs longer (separate retention policy)
     - Add "pin important logs" feature (future enhancement)

---

## Future Enhancements (Out of Scope for This Plan)

1. **Log Persistence**: Save to IndexedDB for offline viewing
2. **Log Levels Retention**: Keep ERROR logs longer than DEBUG
3. **Log Sanitization**: Redact sensitive data (like kidney-genetics-db)
4. **Remote Logging**: Send critical errors to backend
5. **Log Compression**: gzip old logs before rotation
6. **Structured Logging**: Correlation IDs, request tracing

---

## References

### Code Analysis
- Current: `frontend/src/stores/log.js` (lines 8-10, 16-19)
- Current: `frontend/src/services/logService.js` (lines 14, 23-34)
- Reference: `/mnt/c/development/kidney-genetics-db/frontend/src/stores/logStore.js` (lines 110-136)

### Best Practices
- Pinia Documentation: `/vuejs/pinia` (reactive state management)
- Web Search: Circular buffers, JavaScript logging performance
- SOLID Principles: Clean architecture patterns

### Related Plans
- Testing Modernization: `plan/02-completed/TESTING-MODERNIZATION-PLAN.md`
- Local Dev Environment: `plan/02-completed/LOCAL-DEV-ENVIRONMENT.md`

---

## Acceptance Criteria

- [ ] Logs automatically rotate when exceeding maxEntries
- [ ] Statistics tracked: totalLogsReceived, totalLogsDropped, memoryUsage
- [ ] User can configure maxEntries via UI
- [ ] Settings persist in localStorage across refreshes
- [ ] All existing tests pass
- [ ] New tests achieve >90% coverage on modified files
- [ ] No breaking changes to existing API
- [ ] JSDoc documentation complete
- [ ] Manual testing shows no memory bloat after 30 min session
- [ ] Performance: LogViewer opens in <100ms with 50 logs

---

**Next Steps**:
1. Review this plan with team
2. Create implementation branch: `feat/logging-optimization`
3. Begin Phase 1: Configuration Module
4. Follow TDD: Write tests first, then implementation
