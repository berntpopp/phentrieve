/**
 * Version management utilities for Phentrieve frontend.
 *
 * Provides functions to:
 * - Get frontend version (build-time injected)
 * - Fetch API version from backend
 * - Aggregate all component versions
 *
 * Following DRY, KISS, SOLID principles:
 * - Single source of truth (package.json via Vite)
 * - Simple implementation (no complex logic)
 * - Graceful degradation (works even if API is down)
 */

import axios from 'axios';

/**
 * Get frontend version (build-time injection from package.json).
 *
 * Version is baked into bundle at build time - zero runtime cost.
 *
 * @returns {string} Frontend version (e.g., "0.1.0")
 *
 * @example
 * const version = getFrontendVersion()
 * console.log(version) // "0.1.0"
 */
export function getFrontendVersion() {
  return __APP_VERSION__; // Injected by Vite at build time
}

/**
 * Fetch all component versions from API and combine with frontend version.
 *
 * Makes HTTP GET request to /api/v1/system/version endpoint.
 * Gracefully handles API failures - returns frontend version even if API is down.
 *
 * @returns {Promise<Object>} All component versions
 *
 * @example
 * const versions = await getAllVersions()
 * console.log(versions.frontend.version) // "0.1.0"
 * console.log(versions.api.version)      // "0.2.0" or "unknown"
 * console.log(versions.cli.version)      // "0.2.0" or "unknown"
 */
export async function getAllVersions() {
  try {
    const response = await axios.get('/api/v1/system/version');

    // Combine API response with frontend version
    return {
      ...response.data, // CLI + API versions from backend
      frontend: {
        version: getFrontendVersion(),
        name: 'phentrieve-frontend',
        type: 'Vue.js',
      },
    };
  } catch (error) {
    console.error('Failed to fetch versions from API:', error);

    // Graceful degradation - return frontend version even if API fails
    return {
      frontend: {
        version: getFrontendVersion(),
        name: 'phentrieve-frontend',
        type: 'Vue.js',
      },
      api: { version: 'unknown', name: 'phentrieve-api', type: 'FastAPI' },
      cli: { version: 'unknown', name: 'phentrieve', type: 'Python CLI' },
      environment: 'unknown',
      timestamp: new Date().toISOString(),
    };
  }
}
