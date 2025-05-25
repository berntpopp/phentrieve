// The useLogStore import is used by components that initialize this service
// eslint-disable-next-line no-unused-vars
import { useLogStore } from '../stores/log'

export const LogLevel = {
  DEBUG: 'DEBUG',
  INFO: 'INFO',
  WARN: 'WARN',
  ERROR: 'ERROR'
}

class LogService {
  constructor() {
    this.maxEntries = 1000 // Configurable maximum number of log entries
    this.consoleEcho = true // Echo logs to console by default
    this.store = null
  }

  initStore(store) {
    this.store = store
  }

  _log(level, message, data = null) {
    const entry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      data: data ? JSON.parse(JSON.stringify(data)) : null
    }

    // Add to store if initialized
    if (this.store) {
      this.store.addLogEntry(entry)
    }

    // Echo to console if enabled
    if (this.consoleEcho) {
      const consoleMethod = {
        [LogLevel.DEBUG]: 'debug',
        [LogLevel.INFO]: 'info',
        [LogLevel.WARN]: 'warn',
        [LogLevel.ERROR]: 'error'
      }[level]
      console[consoleMethod](message, data || '')
    }
  }

  debug(message, data = null) {
    this._log(LogLevel.DEBUG, message, data)
  }

  info(message, data = null) {
    this._log(LogLevel.INFO, message, data)
  }

  warn(message, data = null) {
    this._log(LogLevel.WARN, message, data)
  }

  error(message, data = null) {
    this._log(LogLevel.ERROR, message, data)
  }

  setConsoleEcho(enabled) {
    this.consoleEcho = enabled
  }

  setMaxEntries(max) {
    this.maxEntries = max
    this.store.trimLogs(max)
  }

  clear() {
    this.store.clearLogs()
  }
}

export const logService = new LogService()
