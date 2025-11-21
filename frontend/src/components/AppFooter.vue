<template>
  <v-footer app class="bg-surface-light py-1" style="min-height: 48px">
    <v-container class="py-0">
      <v-row align="center" justify="space-between" no-gutters>
        <!-- Left: Version Information Button -->
        <v-col cols="auto" class="py-0">
          <v-menu location="top" :close-on-content-click="false">
            <template #activator="{ props: menuProps }">
              <v-btn variant="text" size="small" v-bind="menuProps">
                <v-icon icon="mdi-information-outline" size="small" start />
                v{{ frontendVersion }}
              </v-btn>
            </template>

            <!-- Version Details Popup -->
            <v-card min-width="340">
              <v-card-title class="d-flex align-center">
                <v-icon icon="mdi-package-variant" class="mr-2" />
                Version Information

                <v-spacer />

                <v-btn
                  icon="mdi-refresh"
                  variant="text"
                  size="small"
                  :loading="loading"
                  @click="refreshVersions"
                />
              </v-card-title>

              <v-divider />

              <v-card-text>
                <!-- Frontend Version -->
                <v-list-item class="px-0">
                  <template #prepend>
                    <v-icon icon="mdi-vuejs" color="success" />
                  </template>
                  <v-list-item-title>Frontend</v-list-item-title>
                  <v-list-item-subtitle> {{ frontendVersion }} (Vue.js) </v-list-item-subtitle>
                </v-list-item>

                <!-- API Version -->
                <v-list-item class="px-0">
                  <template #prepend>
                    <v-icon icon="mdi-api" color="primary" />
                  </template>
                  <v-list-item-title>API</v-list-item-title>
                  <v-list-item-subtitle> {{ apiVersion }} (FastAPI) </v-list-item-subtitle>
                </v-list-item>

                <!-- CLI Version -->
                <v-list-item class="px-0">
                  <template #prepend>
                    <v-icon icon="mdi-console" color="info" />
                  </template>
                  <v-list-item-title>CLI</v-list-item-title>
                  <v-list-item-subtitle> {{ cliVersion }} (Python) </v-list-item-subtitle>
                </v-list-item>

                <v-divider class="my-2" />

                <!-- Environment Badge -->
                <div class="d-flex align-center justify-space-between">
                  <span class="text-caption text-medium-emphasis">Environment:</span>
                  <v-chip :color="getEnvironmentColor(environment)" size="small" label>
                    {{ environment }}
                  </v-chip>
                </div>

                <!-- Last Updated -->
                <div v-if="timestamp" class="text-caption text-medium-emphasis mt-2">
                  Updated: {{ formatTimestamp(timestamp) }}
                </div>
              </v-card-text>
            </v-card>
          </v-menu>
        </v-col>

        <!-- Center: Connection Status Indicator -->
        <v-col cols="auto" class="py-0">
          <v-chip :color="apiConnected ? 'success' : 'error'" size="x-small" label class="mx-2">
            <v-icon size="x-small" start>
              {{ apiConnected ? 'mdi-lan-connect' : 'mdi-lan-disconnect' }}
            </v-icon>
            API {{ apiConnected ? 'Online' : 'Offline' }}
            <span v-if="apiConnected && responseTime" class="ml-1"> ({{ responseTime }}ms) </span>
          </v-chip>
        </v-col>

        <!-- Right: Project Links -->
        <v-col cols="auto" class="py-0">
          <v-btn
            href="https://github.com/berntpopp/phentrieve"
            target="_blank"
            variant="text"
            size="small"
          >
            <v-icon icon="mdi-github" size="small" start />
            GitHub
          </v-btn>
        </v-col>
      </v-row>
    </v-container>
  </v-footer>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import { getAllVersions } from '@/utils/version';
import { useApiHealth } from '@/services/api-health';

// Version state
const frontendVersion = ref(__APP_VERSION__); // Build-time injected
const apiVersion = ref('...');
const cliVersion = ref('...');
const environment = ref('...');
const timestamp = ref(null);
const loading = ref(false);

// API health monitoring
const { connected: apiConnected, responseTime, startMonitoring, stopMonitoring } = useApiHealth();

/**
 * Fetch versions from API and update state.
 */
async function refreshVersions() {
  loading.value = true;

  try {
    const versions = await getAllVersions();

    // Update state
    frontendVersion.value = versions.frontend.version;
    apiVersion.value = versions.api?.version || 'unknown';
    cliVersion.value = versions.cli?.version || 'unknown';
    environment.value = versions.environment || 'unknown';
    timestamp.value = versions.timestamp;

    console.log('[AppFooter] Versions refreshed:', versions);
  } catch (error) {
    console.error('[AppFooter] Failed to refresh versions:', error);
    apiVersion.value = 'error';
    cliVersion.value = 'error';
  } finally {
    loading.value = false;
  }
}

/**
 * Get color for environment badge.
 */
function getEnvironmentColor(env) {
  const colors = {
    production: 'success',
    staging: 'warning',
    development: 'info',
  };
  return colors[env] || 'default';
}

/**
 * Format timestamp for display.
 */
function formatTimestamp(ts) {
  if (!ts) return 'Unknown';

  try {
    const date = new Date(ts);
    return date.toLocaleString();
  } catch (error) {
    return 'Invalid';
  }
}

// Lifecycle hooks
onMounted(() => {
  refreshVersions(); // Fetch versions on mount
  startMonitoring(); // Start API health checks
});

onUnmounted(() => {
  stopMonitoring(); // Stop health checks on unmount (prevent memory leaks)
});
</script>

<style scoped>
/* Compact footer styling */
.v-footer {
  border-top: thin solid rgba(var(--v-border-color), var(--v-border-opacity));
}
</style>
