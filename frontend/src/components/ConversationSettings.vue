<script setup>
/**
 * ConversationSettings Component
 *
 * Provides user control over conversation history settings including
 * history length limits and storage management. Uses Vuetify dialogs
 * for all confirmations (no native confirm()).
 *
 * @component
 */
import { ref, computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { useConversationStore } from '../stores/conversation';

const { t } = useI18n();
const conversationStore = useConversationStore();

// Dialog state
const settingsDialog = ref(false);
const confirmClearDialog = ref(false);

/**
 * Computed storage size in human-readable format
 */
const storageInfo = computed(() => {
  return conversationStore.getStorageSize();
});

/**
 * Handle confirmed reset all action
 */
function handleResetAll() {
  conversationStore.resetAll();
  confirmClearDialog.value = false;
  settingsDialog.value = false;
}
</script>

<template>
  <div>
    <!-- Settings Dialog Trigger Button -->
    <v-dialog v-model="settingsDialog" max-width="500">
      <template #activator="{ props }">
        <v-btn
          v-bind="props"
          icon
          size="small"
          variant="text"
          :aria-label="t('conversationSettings.tooltip', 'Conversation Settings')"
        >
          <v-icon>mdi-cog</v-icon>
          <v-tooltip activator="parent" location="top">
            {{ t('conversationSettings.tooltip', 'Conversation Settings') }}
          </v-tooltip>
        </v-btn>
      </template>

      <!-- Settings Dialog Content -->
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon class="mr-2">mdi-cog</v-icon>
          <span class="text-h6">{{
            t('conversationSettings.title', 'Conversation Settings')
          }}</span>
        </v-card-title>

        <v-card-text>
          <!-- History Length Setting -->
          <div class="mb-4">
            <label class="text-subtitle-2 mb-2 d-block">
              {{ t('conversationSettings.historyLength', 'Maximum History Length') }}
            </label>
            <v-slider
              v-model="conversationStore.maxHistoryLength"
              :min="10"
              :max="200"
              :step="10"
              thumb-label="always"
              color="primary"
              hide-details
            >
              <template #append>
                <v-text-field
                  v-model.number="conversationStore.maxHistoryLength"
                  type="number"
                  style="width: 80px"
                  density="compact"
                  variant="outlined"
                  hide-details
                  :min="10"
                  :max="200"
                />
              </template>
            </v-slider>
            <p class="text-caption text-medium-emphasis mt-1">
              {{
                t(
                  'conversationSettings.historyHint',
                  'Lower values improve performance. Higher values keep more history.'
                )
              }}
            </p>
          </div>

          <!-- Current Statistics -->
          <v-divider class="my-4" />

          <div class="text-body-2">
            <p class="font-weight-medium mb-2">
              {{ t('conversationSettings.stats', 'Current Statistics') }}
            </p>
            <v-list density="compact" class="bg-transparent pa-0">
              <v-list-item class="px-0">
                <template #prepend>
                  <v-icon size="small" class="mr-2">mdi-message-text</v-icon>
                </template>
                <v-list-item-title class="text-body-2">
                  {{ t('conversationSettings.totalQueries', 'Total Queries') }}:
                  {{ conversationStore.conversationLength }}
                </v-list-item-title>
              </v-list-item>

              <v-list-item class="px-0">
                <template #prepend>
                  <v-icon size="small" class="mr-2">mdi-format-list-checks</v-icon>
                </template>
                <v-list-item-title class="text-body-2">
                  {{ t('conversationSettings.totalPhenotypes', 'Collected Phenotypes') }}:
                  {{ conversationStore.phenotypeCount }}
                </v-list-item-title>
              </v-list-item>

              <v-list-item class="px-0">
                <template #prepend>
                  <v-icon size="small" class="mr-2">mdi-harddisk</v-icon>
                </template>
                <v-list-item-title class="text-body-2">
                  {{ t('conversationSettings.storageUsed', 'Storage Used') }}:
                  {{ storageInfo.formatted }}
                </v-list-item-title>
              </v-list-item>
            </v-list>
          </div>
        </v-card-text>

        <v-card-actions>
          <v-btn
            color="error"
            variant="text"
            :disabled="!conversationStore.hasConversation && !conversationStore.hasPhenotypes"
            @click="confirmClearDialog = true"
          >
            <v-icon start>mdi-delete-sweep</v-icon>
            {{ t('conversationSettings.clearAll', 'Clear All') }}
          </v-btn>
          <v-spacer />
          <v-btn color="primary" variant="text" @click="settingsDialog = false">
            {{ t('conversationSettings.close', 'Close') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Confirmation Dialog -->
    <v-dialog v-model="confirmClearDialog" max-width="400" persistent>
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon color="error" class="mr-2">mdi-alert-circle</v-icon>
          <span class="text-h6">{{
            t('conversationSettings.confirmClearTitle', 'Clear All Data?')
          }}</span>
        </v-card-title>

        <v-card-text>
          {{
            t(
              'conversationSettings.confirmClearMessage',
              'This will permanently delete all conversation history and collected phenotypes. This action cannot be undone.'
            )
          }}
        </v-card-text>

        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="confirmClearDialog = false">
            {{ t('common.cancel', 'Cancel') }}
          </v-btn>
          <v-btn color="error" variant="flat" @click="handleResetAll">
            {{ t('common.delete', 'Delete') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>
