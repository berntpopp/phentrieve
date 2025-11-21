/**
 * API health monitoring service for connection status tracking.
 *
 * Polls /api/v1/system/health endpoint every 30 seconds to verify API connectivity.
 * Provides reactive connection status for UI components.
 *
 * Design patterns:
 * - Singleton: Single health monitor across entire app
 * - Observer: Components subscribe to connection status changes
 * - Retry: Auto-retry on failures with configurable interval
 *
 * Following KISS principle: Simple polling, no WebSocket complexity for REST API.
 */

import { ref } from 'vue';
import axios from 'axios';

class ApiHealthService {
  constructor() {
    /**
     * Reactive connection status (true = connected, false = disconnected).
     * Bind directly to Vue components for automatic UI updates.
     * @type {import('vue').Ref<boolean>}
     */
    this.connected = ref(true);

    /**
     * Timestamp of last successful health check.
     * @type {import('vue').Ref<Date|null>}
     */
    this.lastCheck = ref(null);

    /**
     * Response time of last health check in milliseconds.
     * @type {import('vue').Ref<number|null>}
     */
    this.responseTime = ref(null);

    /**
     * Interval between health checks (milliseconds).
     * @type {number}
     */
    this.checkInterval = 30000; // 30 seconds

    /**
     * Timeout for health check requests (milliseconds).
     * @type {number}
     */
    this.requestTimeout = 5000; // 5 seconds

    /**
     * Interval timer ID for cleanup.
     * @type {number|null}
     * @private
     */
    this.intervalId = null;
  }

  /**
   * Perform a single health check.
   * Measures response time and updates connection status.
   *
   * @returns {Promise<boolean>} True if healthy, false otherwise
   */
  async checkHealth() {
    const startTime = performance.now();

    try {
      const response = await axios.get('/api/v1/system/health', {
        timeout: this.requestTimeout,
      });

      const endTime = performance.now();
      const responseTimeMs = Math.round(endTime - startTime);

      // Update reactive state
      this.connected.value = response.status === 200;
      this.lastCheck.value = new Date();
      this.responseTime.value = responseTimeMs;

      console.log(`[API Health] Connected (${responseTimeMs}ms)`);

      return true;
    } catch (error) {
      console.error('[API Health] Check failed:', error.message);

      // Update reactive state
      this.connected.value = false;
      this.lastCheck.value = new Date();
      this.responseTime.value = null;

      return false;
    }
  }

  /**
   * Start periodic health monitoring.
   * Performs initial check immediately, then repeats every checkInterval.
   */
  startMonitoring() {
    if (this.intervalId) {
      console.warn('[API Health] Monitoring already started');
      return;
    }

    console.log(`[API Health] Starting monitoring (every ${this.checkInterval / 1000}s)`);

    // Initial check
    this.checkHealth();

    // Periodic checks
    this.intervalId = setInterval(() => {
      this.checkHealth();
    }, this.checkInterval);
  }

  /**
   * Stop periodic health monitoring.
   * Call this in component's onUnmounted to prevent memory leaks.
   */
  stopMonitoring() {
    if (this.intervalId) {
      console.log('[API Health] Stopping monitoring');
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }
}

// Singleton instance - shared across entire app
export const apiHealthService = new ApiHealthService();

/**
 * Vue composable for using API health service in components.
 *
 * Provides reactive refs that automatically update UI on connection changes.
 *
 * @returns {Object} API health service interface
 *
 * @example
 * // In Vue component
 * import { onMounted, onUnmounted } from 'vue'
 * import { useApiHealth } from '@/services/api-health'
 *
 * const { connected, responseTime, startMonitoring, stopMonitoring } = useApiHealth()
 *
 * onMounted(() => startMonitoring())
 * onUnmounted(() => stopMonitoring())
 *
 * // In template
 * <div>API Status: {{ connected ? 'Online' : 'Offline' }}</div>
 * <div v-if="responseTime">Response Time: {{ responseTime }}ms</div>
 */
export function useApiHealth() {
  return {
    connected: apiHealthService.connected,
    lastCheck: apiHealthService.lastCheck,
    responseTime: apiHealthService.responseTime,
    checkHealth: () => apiHealthService.checkHealth(),
    startMonitoring: () => apiHealthService.startMonitoring(),
    stopMonitoring: () => apiHealthService.stopMonitoring(),
  };
}
