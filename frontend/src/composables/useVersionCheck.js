import { ref } from 'vue';
import { getAllVersions } from '../utils/version';

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

  async function refreshVersions() {
    loadingVersions.value = true;
    try {
      const versions = await getAllVersions();
      frontendVersion.value = versions.frontend?.version || 'unknown';
      apiVersion.value = versions.api?.version || 'unknown';
      cliVersion.value = versions.cli?.version || 'unknown';
      environment.value = versions.environment || 'unknown';
    } catch {
      // Versions stay at their current values on error
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
    refreshVersions,
  };
}
