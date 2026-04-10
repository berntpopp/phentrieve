import { ref } from 'vue';
import { getAllVersions } from '../utils/version';
import { logService } from '../services/logService';

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
  const frontendVersion = ref('Loading...');
  const apiVersion = ref('Loading...');
  const cliVersion = ref('Loading...');
  const environment = ref('unknown');
  const loadingVersions = ref(false);
  const versionError = ref(null);

  async function refreshVersions() {
    loadingVersions.value = true;
    versionError.value = null;
    try {
      const versions = await getAllVersions();
      frontendVersion.value = versions.frontend?.version || 'unknown';
      apiVersion.value = versions.api?.version || 'unknown';
      cliVersion.value = versions.cli?.version || 'unknown';
      environment.value = versions.environment || 'unknown';
      logService.debug('Versions refreshed', versions);
    } catch (error) {
      versionError.value = error;
      logService.error('Failed to refresh versions', error);
    } finally {
      loadingVersions.value = false;
    }
  }

  return {
    frontendVersion,
    apiVersion,
    cliVersion,
    environment,
    loadingVersions,
    versionError,
    refreshVersions,
  };
}
