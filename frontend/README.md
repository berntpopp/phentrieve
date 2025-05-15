# Frontend Documentation

## Client-Side Logging System

The frontend application includes a comprehensive client-side logging system that helps track application events, user interactions, and potential issues during development and debugging.

### Core Components

#### 1. Log Service (`src/services/logService.js`)

The LogService provides a centralized logging interface with the following features:

- Multiple log levels (DEBUG, INFO, WARN, ERROR)
- Optional console echoing
- Configurable maximum log entries
- JSON data support for structured logging

Usage:

```javascript
import { logService } from '../services/logService'

// Basic logging
logService.debug('Debug message')
logService.info('Info message')
logService.warn('Warning message')
logService.error('Error message')

// Logging with data
logService.info('User action', { action: 'click', target: 'submit-button' })
logService.error('API error', { status: 404, message: 'Not found' })

// Configure logging
logService.setConsoleEcho(false) // Disable console echoing
logService.setMaxEntries(500)    // Limit log history
```

#### 2. Log Store (`src/stores/log.js`)

The Pinia store that manages the log state:

- Stores log entries
- Manages LogViewer visibility
- Handles log trimming and clearing

#### 3. LogViewer Component (`src/components/LogViewer.vue`)

A Vuetify-based UI component that displays logs with the following features:

- Real-time log viewing
- Filtering by log level
- Text search functionality
- Log entry expansion for detailed data
- Download logs as JSON
- Clear log history

### Integration

The logging system is integrated throughout the application:

1. **API Interactions**: All API calls in PhentrieveService are logged
2. **User Actions**: Important user interactions in components
3. **Error Handling**: System-wide error capture and logging
4. **Application State**: Important state changes and initialization events

### Usage in Components

To use logging in a component:

```javascript
import { logService } from '../services/logService'

export default {
  methods: {
    async handleAction() {
      try {
        logService.info('Starting action', { params })
        // ... action code ...
        logService.info('Action completed', { result })
      } catch (error) {
        logService.error('Action failed', error)
      }
    }
  }
}
```

### Accessing Logs

1. Click the "Logs" button in the application footer to open the LogViewer
2. Use the search bar to filter logs by content
3. Use the level selector to filter by log level
4. Click on log entries to expand and view detailed data
5. Use the download button to export logs for further analysis

### Best Practices

1. Use appropriate log levels:
   - DEBUG: Detailed information for debugging
   - INFO: General operational information
   - WARN: Warning messages for potential issues
   - ERROR: Error conditions that need attention

2. Include relevant data:
   - Always include contextual data when available
   - Avoid logging sensitive information
   - Structure data objects for clarity

3. Performance considerations:
   - Use DEBUG level for verbose logging
   - Consider disabling console echo in production
   - Monitor log volume and adjust maxEntries if needed
