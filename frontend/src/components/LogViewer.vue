<template>
  <v-navigation-drawer
    v-model="logStore.isViewerVisible"
    location="right"
    temporary
    width="600"
    aria-label="Log viewer"
  >
    <v-toolbar density="compact" color="primary">
      <v-toolbar-title class="text-white d-flex align-center">
        {{ $t('logViewer.title') }}
        <v-chip size="x-small" class="ml-2" color="white" variant="outlined">
          {{ logStore.logCount }}/{{ logStore.maxEntries }}
        </v-chip>
      </v-toolbar-title>
      <v-spacer />
      <v-btn
        icon="mdi-cog"
        variant="text"
        color="white"
        aria-label="Configure logging"
        @click="showConfigDialog = true"
      />
      <v-btn
        icon="mdi-download"
        variant="text"
        color="white"
        :disabled="!logStore.logs.length"
        aria-label="Download logs"
        @click="downloadLogs"
      />
      <v-btn
        icon="mdi-delete"
        variant="text"
        color="white"
        :disabled="!logStore.logs.length"
        aria-label="Clear all logs"
        @click="clearLogs"
      />
      <v-btn
        icon="mdi-close"
        variant="text"
        color="white"
        aria-label="Close log viewer"
        @click="logStore.setViewerVisibility(false)"
      />
    </v-toolbar>

    <v-card class="mx-2 mt-2" variant="outlined">
      <v-card-text class="pa-2">
        <v-text-field
          v-model="search"
          density="compact"
          variant="outlined"
          label="Search logs"
          append-inner-icon="mdi-magnify"
          hide-details
          class="mb-2"
          bg-color="white"
          color="primary"
        >
          <template #label>
            <span class="text-high-emphasis">{{ $t('logViewer.searchPlaceholder') }}</span>
          </template>
        </v-text-field>

        <v-select
          v-model="selectedLevels"
          :items="logLevels"
          density="compact"
          variant="outlined"
          label="Log Levels"
          multiple
          chips
          hide-details
          class="mb-2"
          bg-color="white"
          color="primary"
        >
          <template #label>
            <span class="text-high-emphasis">{{ $t('logViewer.logLevels') }}</span>
          </template>
        </v-select>
      </v-card-text>
    </v-card>

    <!-- Statistics Card -->
    <v-card v-if="statistics" class="mx-2 mb-2" variant="outlined">
      <v-card-text class="pa-2">
        <div class="d-flex justify-space-between text-caption">
          <div>
            <strong>{{ $t('logViewer.stats.received') }}:</strong>
            {{ statistics.totalLogsReceived }}
          </div>
          <div>
            <strong>{{ $t('logViewer.stats.dropped') }}:</strong>
            <span :class="statistics.totalLogsDropped > 0 ? 'text-warning' : ''">
              {{ statistics.totalLogsDropped }}
            </span>
          </div>
          <div>
            <strong>{{ $t('logViewer.stats.memory') }}:</strong>
            {{ statistics.memoryUsage.kb }} KB
          </div>
        </div>
      </v-card-text>
    </v-card>

    <div class="log-container" role="log" aria-label="Application logs">
      <template v-if="filteredLogs.length">
        <v-card
          v-for="(log, index) in filteredLogs"
          :key="index"
          :color="getLogColor(log.level)"
          class="mb-2 mx-2"
          variant="outlined"
          role="article"
          :aria-label="`${log.level} log from ${formatTimestamp(log.timestamp)}`"
        >
          <v-card-text class="pa-2">
            <div class="d-flex align-center">
              <div class="text-caption text-high-emphasis">
                {{ formatTimestamp(log.timestamp) }}
              </div>
              <v-chip
                size="x-small"
                :color="getLogColor(log.level)"
                class="ml-2"
                label
                :aria-label="`Log level: ${log.level}`"
              >
                {{ log.level }}
              </v-chip>
            </div>
            <div class="text-body-2 mt-1 text-high-emphasis">
              {{ log.message }}
            </div>
            <v-expand-transition>
              <pre
                v-if="log.data"
                class="mt-2 text-caption text-high-emphasis"
                role="code"
                aria-label="Log details"
                >{{ JSON.stringify(log.data, null, 2) }}</pre
              >
            </v-expand-transition>
          </v-card-text>
        </v-card>
      </template>
      <v-card v-else class="ma-2" variant="outlined">
        <v-card-text class="text-center text-disabled">
          {{ $t('logViewer.emptyLogsMessage') }}
        </v-card-text>
      </v-card>
    </div>

    <!-- Configuration Dialog -->
    <v-dialog v-model="showConfigDialog" max-width="400">
      <v-card>
        <v-card-title>{{ $t('logViewer.config.title') }}</v-card-title>
        <v-card-text>
          <v-slider
            v-model="configMaxEntries"
            :min="configLimits.MIN_ENTRIES"
            :max="configLimits.MAX_ENTRIES"
            :step="configLimits.STEP"
            thumb-label="always"
            color="primary"
          >
            <template #prepend>
              <span class="text-caption">{{ configLimits.MIN_ENTRIES }}</span>
            </template>
            <template #append>
              <span class="text-caption">{{ configLimits.MAX_ENTRIES }}</span>
            </template>
          </v-slider>
          <div class="text-caption text-medium-emphasis mt-2">
            {{ $t('logViewer.config.description') }}
          </div>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn @click="showConfigDialog = false">{{ $t('logViewer.config.cancel') }}</v-btn>
          <v-btn color="primary" @click="saveConfig">{{ $t('logViewer.config.save') }}</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-navigation-drawer>
</template>

<script>
import { ref, computed, watch } from 'vue';
import { useLogStore } from '../stores/log';
import { LogLevel, logService } from '../services/logService';
import { LOG_CONFIG } from '../config/logConfig';

export default {
  name: 'LogViewer',
  setup() {
    const logStore = useLogStore();
    const search = ref('');
    const selectedLevels = ref(Object.values(LogLevel));
    const logLevels = Object.values(LogLevel);

    // Configuration dialog state
    const showConfigDialog = ref(false);
    const configMaxEntries = ref(logStore.maxEntries);
    const configLimits = LOG_CONFIG.UI_LIMITS;

    // Watch dialog open to sync configMaxEntries with current store value
    watch(showConfigDialog, (isOpen) => {
      if (isOpen) {
        configMaxEntries.value = logStore.maxEntries;
      }
    });

    // Statistics (computed to update reactively)
    // Pass true to include memory usage (expensive but needed for display)
    const statistics = computed(() => logStore.getStatistics(true));

    const filteredLogs = computed(() => {
      return logStore.logs.filter((log) => {
        const matchesSearch =
          !search.value ||
          log.message.toLowerCase().includes(search.value.toLowerCase()) ||
          (log.data && JSON.stringify(log.data).toLowerCase().includes(search.value.toLowerCase()));
        const matchesLevel = selectedLevels.value.includes(log.level);
        return matchesSearch && matchesLevel;
      });
    });

    const getLogColor = (level) => {
      const colors = {
        [LogLevel.DEBUG]: 'grey-lighten-3',
        [LogLevel.INFO]: 'blue-lighten-4',
        [LogLevel.WARN]: 'amber-lighten-4',
        [LogLevel.ERROR]: 'red-lighten-4',
      };
      return colors[level] || 'grey-lighten-3';
    };

    const formatTimestamp = (timestamp) => {
      return new Date(timestamp).toLocaleTimeString();
    };

    const downloadLogs = () => {
      const content = JSON.stringify(logStore.logs, null, 2);
      const blob = new Blob([content], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `logs-${new Date().toISOString()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    };

    const clearLogs = () => {
      logStore.clearLogs();
    };

    const saveConfig = () => {
      logService.setMaxEntries(configMaxEntries.value);
      showConfigDialog.value = false;
    };

    return {
      logStore,
      search,
      selectedLevels,
      logLevels,
      statistics,
      showConfigDialog,
      configMaxEntries,
      configLimits,
      filteredLogs,
      getLogColor,
      formatTimestamp,
      downloadLogs,
      clearLogs,
      saveConfig,
    };
  },
};
</script>

<style scoped>
.log-container {
  height: calc(100vh - 140px);
  overflow-y: auto;
}

pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  background: rgba(0, 0, 0, 0.03);
  padding: 8px;
  border-radius: 4px;
  margin: 0;
}
</style>
