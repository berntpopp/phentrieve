<template>
  <div class="search-controls-row d-flex justify-end mt-2">
    <div class="search-controls-group d-flex align-center flex-wrap justify-end ga-2">
      <div class="mode-switch" role="group" aria-label="Search mode switch">
        <v-btn
          data-testid="mode-pill-query"
          size="small"
          rounded="pill"
          :variant="!isTextProcessModeActive ? 'tonal' : 'text'"
          :color="!isTextProcessModeActive ? 'primary' : 'default'"
          class="mode-switch__pill"
          @click="$emit('set-mode', 'query')"
        >
          <v-icon start size="small">mdi-magnify</v-icon>
          Query
        </v-btn>
        <v-btn
          data-testid="mode-pill-text-process"
          size="small"
          rounded="pill"
          :variant="isTextProcessModeActive ? 'tonal' : 'text'"
          :color="isTextProcessModeActive ? 'primary' : 'default'"
          class="mode-switch__pill"
          @click="$emit('set-mode', 'textProcess')"
        >
          <v-icon start size="small">mdi-file-document-outline</v-icon>
          Full Text
        </v-btn>
      </div>

      <v-btn
        data-testid="search-settings-button"
        size="small"
        rounded="pill"
        variant="text"
        color="primary"
        class="search-controls-row__settings"
        :disabled="isLoading"
        :aria-label="advancedOptionsToggleLabel"
        :aria-expanded="showAdvancedOptions.toString()"
        aria-controls="advanced-options-panel"
        data-tutorial-step="advanced-options"
        @click="$emit('toggle-advanced')"
      >
        <v-icon start size="small">
          {{ showAdvancedOptions ? 'mdi-cog-outline' : 'mdi-tune-variant' }}
        </v-icon>
        Settings
      </v-btn>
    </div>
  </div>

  <div
    v-if="showAutoSwitchNotice"
    data-testid="mode-auto-switch-notice"
    class="mode-switch-notice px-3 pt-2"
  >
    {{ autoSwitchNoticeLabel }}
  </div>
</template>

<script>
export default {
  name: 'QueryModeControls',
  props: {
    isTextProcessModeActive: {
      type: Boolean,
      default: false,
    },
    isLoading: {
      type: Boolean,
      default: false,
    },
    showAdvancedOptions: {
      type: Boolean,
      default: false,
    },
    advancedOptionsToggleLabel: {
      type: String,
      required: true,
    },
    showAutoSwitchNotice: {
      type: Boolean,
      default: false,
    },
    autoSwitchNoticeLabel: {
      type: String,
      required: true,
    },
  },
  emits: ['set-mode', 'toggle-advanced'],
};
</script>

<style scoped>
.search-controls-row {
  padding-inline: 8px;
}

.search-controls-group {
  width: 100%;
}

.mode-switch {
  display: flex;
  align-items: center;
  gap: 2px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 999px;
  padding: 2px;
  background: rgba(245, 247, 252, 0.9);
}

.mode-switch__pill {
  text-transform: none;
  letter-spacing: 0;
  min-width: 0;
  font-size: 0.95rem;
}

.search-controls-row__settings {
  text-transform: none;
  letter-spacing: 0;
  min-height: 34px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  background: rgba(245, 247, 252, 0.9);
  font-size: 0.95rem;
}

.mode-switch-notice {
  font-size: 0.8rem;
  color: rgba(25, 82, 166, 0.92);
}
</style>
