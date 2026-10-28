<template>
  <div class="search-container mx-auto px-2">
    <!-- Clean Search Bar with Integrated Button -->
    <div class="search-bar-container pt-0 px-2 pb-2 pa-sm-3">
      <QueryForm
        v-model="queryText"
        :is-text-process-mode-active="isTextProcessModeActive"
        :is-loading="isLoading"
        :query-input-label="$t('queryInterface.accessibility.queryInputLabel')"
        :query-input-description="getQueryInputDescription()"
        :text-process-input-label="$t('queryInterface.accessibility.textProcessInputLabel')"
        :text-process-input-description="getTextProcessInputDescription()"
        :search-button-label="$t('queryInterface.accessibility.searchButton')"
        @submit="submitQuery"
        @cancel="queryInterfaceController.cancelQuery()"
      />

      <QueryModeControls
        :is-text-process-mode-active="isTextProcessModeActive"
        :is-loading="isLoading"
        :show-advanced-options="showAdvancedOptions"
        :show-auto-switch-notice="showAutoSwitchNotice"
        :auto-switch-notice-label="getAutoSwitchNoticeLabel()"
        @set-mode="setMode"
        @toggle-advanced="showAdvancedOptions = !showAdvancedOptions"
      />

      <!-- Advanced Options Panel -->
      <AdvancedOptionsPanel
        v-model:selected-model="selectedModel"
        v-model:selected-language="selectedLanguage"
        v-model:include-details="includeDetails"
        v-model:similarity-threshold="similarityThreshold"
        v-model:force-endpoint-mode="forceEndpointMode"
        v-model:text-process-options="textProcessOptions"
        v-model:chunking-strategy="chunkingStrategy"
        v-model:window-size="windowSize"
        v-model:step-size="stepSize"
        v-model:chunk-retrieval-threshold="chunkRetrievalThreshold"
        v-model:aggregated-term-confidence="aggregatedTermConfidence"
        v-model:no-assertion-detection-for-text-process="noAssertionDetectionForTextProcess"
        v-model:split-threshold="splitThreshold"
        v-model:min-segment-length="minSegmentLength"
        v-model:num-results-per-chunk="numResultsPerChunk"
        v-model:top-term-per-chunk-for-aggregation="topTermPerChunkForAggregation"
        :default-llm-model="defaultTextProcessLlmOptions.llmModel"
        :default-llm-mode="defaultTextProcessLlmOptions.llmMode"
        :visible="showAdvancedOptions"
        :disabled="isLoading"
        :available-models="availableModels"
        :available-languages="availableLanguages"
        :is-text-process-mode-active="isTextProcessModeActive"
      />
    </div>

    <!-- Chat-like conversation interface -->
    <div ref="conversationContainer" class="conversation-container">
      <ConversationSkeleton
        v-if="conversationStore.isHydrating"
        :count="2"
        class="hydration-skeleton"
      />

      <template v-else>
        <div v-for="item in conversationStore.queryHistory" :key="item.id" class="mb-4">
          <div class="user-query d-flex">
            <v-tooltip
              location="top"
              :text="$t('queryInterface.accessibility.userInput')"
              :aria-label="$t('queryInterface.accessibility.userInput')"
            >
              <template #activator="{ props }">
                <v-avatar v-bind="props" color="primary" size="36" class="mt-1 mr-2">
                  <span class="white--text">U</span>
                </v-avatar>
              </template>
            </v-tooltip>
            <div
              :class="
                item.type === 'textProcess' ? 'query-bubble query-bubble--document' : 'query-bubble'
              "
            >
              <template v-if="item.type === 'textProcess'">
                <FullTextNoteCurator
                  :item="item"
                  :note-text="getHistoryDisplayQuery(item)"
                  :expanded="isUserNoteExpanded(item.id)"
                  :active-phenotype-id="getHoveredNotePhenotype(item.id)"
                  :query-options="curationQueryOptions"
                  @toggle="toggleUserNote(item.id)"
                  @hover="handleAnnotatedTextHover(item.id, $event)"
                  @clear-hover="clearHoveredNotePhenotype(item.id)"
                  @add-to-collection="handleAddToCollection"
                />
              </template>
              <p v-else class="mb-0" style="white-space: pre-wrap">
                {{ getHistoryDisplayQuery(item) }}
              </p>
            </div>
          </div>

          <div v-if="item.loading || item.response || item.error" class="bot-response d-flex mt-2">
            <v-tooltip
              location="top"
              :text="$t('queryInterface.accessibility.response')"
              :aria-label="$t('queryInterface.accessibility.response')"
            >
              <template #activator="{ props }">
                <v-avatar v-bind="props" color="info" size="36" class="mt-1 mr-2">
                  <v-icon color="white"> mdi-robot-outline </v-icon>
                </v-avatar>
              </template>
            </v-tooltip>
            <div class="response-bubble">
              <v-progress-circular v-if="item.loading" indeterminate color="primary" size="24" />

              <ResultsDisplay
                v-else-if="item.error"
                :key="'results-' + item.id"
                :response-data="item.response"
                :result-type="item.type"
                :turn-id="item.id"
                :error="item.error"
                :collected-phenotypes="conversationStore.collectedPhenotypes"
                @add-to-collection="handleAddToCollection"
              />

              <FullTextResponseReceipt
                v-else-if="item.type === 'textProcess'"
                :item="item"
                :note-text="getHistoryDisplayQuery(item)"
                :collected-phenotypes="conversationStore.collectedPhenotypes"
                :hovered-phenotype-id="getHoveredNotePhenotype(item.id)"
                @add-to-collection="handleAddToCollection"
                @add-all-to-collection="handleAddAllToCollection"
                @hover-phenotype="setHoveredNotePhenotype(item.id, $event)"
                @clear-hover="clearHoveredNotePhenotype(item.id)"
              />

              <ResultsDisplay
                v-else
                :key="'results-' + item.id"
                :response-data="item.response"
                :result-type="item.type"
                :turn-id="item.id"
                :error="item.error"
                :collected-phenotypes="conversationStore.collectedPhenotypes"
                @add-to-collection="handleAddToCollection"
              />
            </div>
          </div>
        </div>
      </template>
    </div>

    <QueryResultActions
      v-model:subject-id="phenopacketSubjectId"
      v-model:sex="phenopacketSex"
      v-model:date-of-birth="phenopacketDateOfBirth"
      v-model:export-error-visible="exportErrorVisible"
      :phenotypes="conversationStore.collectedPhenotypes"
      :panel-open="conversationStore.showCollectionPanel"
      :sex-options="sexOptions"
      :export-error-message="exportErrorMessage"
      :dismiss-label="$t('common.dismiss')"
      @toggle-panel="toggleCollectionPanel"
      @update:panel-open="conversationStore.showCollectionPanel = $event"
      @remove="removePhenotype"
      @toggle-assertion="toggleAssertionStatus"
      @export-text="exportPhenotypes"
      @export-json="onExportPhenopacket"
      @clear="clearPhenotypeCollection"
    />

    <PiiReviewDialog
      v-model="piiReviewDialogVisible"
      :summary="pendingPiiSubmission?.scanResult?.summary || { high: {}, review: {} }"
      @cancel="cancelPiiReview"
      @redact="redactPiiInInput"
      @continue="continueWithPiiRedaction"
    />
  </div>
</template>

<script setup>
import { computed, nextTick, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { useRoute, useRouter } from 'vue-router';
import ResultsDisplay from './ResultsDisplay.vue';
import ConversationSkeleton from './ConversationSkeleton.vue';
import AdvancedOptionsPanel from './AdvancedOptionsPanel.vue';
import FullTextResponseReceipt from './FullTextResponseReceipt.vue';
import PiiReviewDialog from './PiiReviewDialog.vue';
import QueryForm from './query/QueryForm.vue';
import FullTextNoteCurator from './FullTextNoteCurator.vue';
import QueryModeControls from './query/QueryModeControls.vue';
import QueryResultActions from './query/QueryResultActions.vue';
import PhentrieveService from '../services/PhentrieveService';
import { logService } from '../services/logService';
import { useQueryPreferencesStore } from '../stores/queryPreferences';
import { useConversationStore } from '../stores/conversation';
import { useAdvancedOptions } from '../composables/useAdvancedOptions';
import { usePhenotypeCollection } from '../composables/usePhenotypeCollection';
import { useQueryInterfaceController } from '../composables/useQueryInterfaceController';
import { usePiiReviewFlow } from '../composables/usePiiReviewFlow';
import { useFullTextCurationStore } from '../stores/fullTextCuration';

defineOptions({
  name: 'QueryInterface',
});

const DEFAULT_TEXT_PROCESS_LLM_OPTIONS = Object.freeze({
  llmModel: 'gemini-3.1-flash-lite',
  llmMode: 'two_phase',
});
const REDACTED_QUERY_PLACEHOLDER = '[redacted]';

const { t, locale } = useI18n();
const route = useRoute();
const router = useRouter();
const conversationStore = useConversationStore();
const fullTextCurationStore = useFullTextCurationStore();
const queryPreferencesStore = useQueryPreferencesStore();
const piiReviewFlow = usePiiReviewFlow({ logService });
const { piiReviewDialogVisible, pendingPiiSubmission } = piiReviewFlow;

const {
  showAdvancedOptions,
  numResults,
  similarityThreshold,
  splitThreshold,
  chunkRetrievalThreshold,
  aggregatedTermConfidence,
  inputTextLengthThreshold,
  windowSize,
  stepSize,
  minSegmentLength,
  numResultsPerChunk,
  resetToDefaults,
} = useAdvancedOptions();

const {
  phenopacketSubjectId,
  phenopacketSex,
  phenopacketDateOfBirth,
  addPhenotype: addToPhenotypeCollection,
  removePhenotype,
  toggleAssertionStatus,
  clearCollection: clearPhenotypeCollection,
  toggleCollectionPanel,
  exportCollectionAsText: exportPhenotypes,
  exportAsPhenopacket: exportPhenotypesAsPhenopacket,
} = usePhenotypeCollection();

// Component state
const conversationContainer = ref(null);
const queryText = ref('');
const selectedModel = ref(null); // Will be set from API config
const availableModels = ref([]); // Populated from API /info endpoint
const modelsLoading = ref(true); // Track model loading state
const selectedLanguage = ref(null); // null = backend auto-detects the text language
const availableLanguages = ref([
  { text: t('common.auto'), value: null },
  { text: 'English', value: 'en' },
  { text: 'German (Deutsch)', value: 'de' },
  { text: 'French (Français)', value: 'fr' },
  { text: 'Spanish (Español)', value: 'es' },
  { text: 'Dutch (Nederlands)', value: 'nl' },
]);
const isLoading = ref(false);
const userHasScrolled = ref(false);
const shouldScrollToTop = ref(false);
const sexOptions = ref([
  { title: t('phenopacket.sexUnknown'), value: 0 },
  { title: t('phenopacket.sexFemale'), value: 1 },
  { title: t('phenopacket.sexMale'), value: 2 },
  { title: t('phenopacket.sexOther'), value: 3 },
]);
const forceEndpointMode = ref(null);
const modeSelectionSource = ref(null);
const autoSwitchSuppressed = ref(false);
const expandedUserNotes = ref({});
const defaultTextProcessLlmOptions = DEFAULT_TEXT_PROCESS_LLM_OPTIONS;
const textProcessOptions = ref({
  extractionBackend: 'llm',
  ...DEFAULT_TEXT_PROCESS_LLM_OPTIONS,
});
const chunkingStrategy = ref('sliding_window_punct_conj_cleaned');
const semanticModelForChunking = ref(null);
const retrievalModelForTextProcess = ref(null);
const noAssertionDetectionForTextProcess = ref(false);
const assertionPreferenceForTextProcess = ref('dependency');
const topTermPerChunkForAggregation = ref(false);
// Phenopacket export error surface (snackbar feedback)
const exportErrorVisible = ref(false);
const exportErrorMessage = ref('');
const hoveredTextProcessPhenotypes = ref({});

// Determine if text processing mode is active based on text length or user force setting
const isTextProcessModeActive = computed(() => {
  if (forceEndpointMode.value) {
    return forceEndpointMode.value === 'textProcess';
  }
  return queryText.value.length > inputTextLengthThreshold.value;
});
const showAutoSwitchNotice = computed(
  () => modeSelectionSource.value === 'auto' && forceEndpointMode.value === 'textProcess'
);
// Query parameters reused when a curator re-queries a single span / selection.
const curationQueryOptions = computed(() => ({
  model_name: selectedModel.value,
  language: selectedLanguage.value,
  num_results: 8,
  similarity_threshold: similarityThreshold.value,
}));
// Access includeDetails from Pinia store (persisted in localStorage)
const includeDetails = computed({
  get: () => queryPreferencesStore.includeDetails,
  set: (value) => queryPreferencesStore.setIncludeDetails(value),
});

const queryInterfaceController = useQueryInterfaceController({
  state: {
    queryText,
    uiLocale: locale,
    selectedModel,
    availableModels,
    modelsLoading,
    isLoading,
    shouldScrollToTop,
    userHasScrolled,
    showAdvancedOptions,
    forceEndpointMode,
    isTextProcessModeActive,
    selectedLanguage,
    similarityThreshold,
    numResults,
    includeDetails,
    expandedUserNotes,
    textProcessOptions,
    chunkingStrategy,
    windowSize,
    stepSize,
    splitThreshold,
    minSegmentLength,
    semanticModelForChunking,
    retrievalModelForTextProcess,
    chunkRetrievalThreshold,
    numResultsPerChunk,
    noAssertionDetectionForTextProcess,
    assertionPreferenceForTextProcess,
    aggregatedTermConfidence,
    topTermPerChunkForAggregation,
  },
  conversationStore,
  piiReviewFlow,
  route,
  router,
  nextTick,
  service: PhentrieveService,
  logService,
});

watch(
  () => conversationStore.queryHistory.length,
  (newLength) => {
    logService.debug('Query history updated', {
      newHistoryLength: newLength,
    });
    // Drop curation state for turns evicted from history (maxHistoryLength)
    // so the persisted curation store does not accumulate orphaned entries.
    const liveIds = new Set((conversationStore.queryHistory || []).map((entry) => entry.id));
    Object.keys(fullTextCurationStore.turns || {}).forEach((turnId) => {
      if (!liveIds.has(turnId)) {
        fullTextCurationStore.dropTurn(turnId);
      }
    });
    nextTick(() => {
      if (shouldScrollToTop.value && !userHasScrolled.value) {
        scrollToTop();
      }
    });
  }
);

watch(selectedModel, (newModel, oldModel) => {
  if (oldModel !== null) {
    // Avoid resetting on initial mount
    logService.info('Model changed', { newModel: newModel, oldModel: oldModel });
    // Reset some query-specific settings to defaults when model changes
    resetToDefaults();
    logService.info('Reset query-specific settings to defaults due to model change.');
  }
});

watch(queryText, (newValue, oldValue) => {
  if (typeof newValue !== 'string') {
    return;
  }

  if (newValue.length <= inputTextLengthThreshold.value) {
    autoSwitchSuppressed.value = false;
  }

  if (
    !autoSwitchSuppressed.value &&
    forceEndpointMode.value !== 'textProcess' &&
    newValue.length > inputTextLengthThreshold.value &&
    oldValue.length <= inputTextLengthThreshold.value
  ) {
    forceEndpointMode.value = 'textProcess';
    modeSelectionSource.value = 'auto';
  }
});

/**
 * Wrapper around usePhenotypeCollection.exportAsPhenopacket that surfaces
 * errors via a snackbar instead of letting them propagate unhandled.
 * The composable throws on failure (intentionally, so it stays testable
 * and accessibility-friendly); this function catches and renders.
 */
function onExportPhenopacket() {
  try {
    exportPhenotypesAsPhenopacket();
  } catch (error) {
    exportErrorMessage.value = t('queryInterface.phenotypeCollection.exportError');
    exportErrorVisible.value = true;
    // The composable already logs via logService.error; no need to re-log.
    void error;
  }
}

function handleAddToCollection(phenotype) {
  addToPhenotypeCollection(phenotype);
  // No auto-open: the drawer would cover the results; the row check + FAB badge confirm the add
}

function handleAddAllToCollection(phenotypes) {
  if (!Array.isArray(phenotypes)) {
    return;
  }

  phenotypes.forEach((phenotype) => {
    addToPhenotypeCollection(phenotype);
  });
}

/**
 * Fetch available embedding models from API /info endpoint
 * Falls back to BioLORD if API is unavailable
 */
function fetchAvailableModels() {
  return queryInterfaceController.fetchAvailableModels();
}

function scrollToTop() {
  nextTick(() => {
    const container = conversationContainer.value;
    if (container) {
      container.scrollTo({ top: 0, behavior: 'auto' }); // Use auto for instant jump
    }
  });
}

function getTextProcessInputDescription() {
  const base = t('queryInterface.accessibility.textProcessInputDescription');
  return isLoading.value
    ? `${base} ${t('queryInterface.accessibility.processingInProgress')}`
    : base;
}

function getQueryInputDescription() {
  const base = t('queryInterface.accessibility.queryInputDescription');
  return isLoading.value ? `${base} ${t('queryInterface.accessibility.searchInProgress')}` : base;
}


function getAutoSwitchNoticeLabel() {
  const translated = t('queryInterface.modeSwitch.autoFullTextNotice');
  return translated && translated !== 'queryInterface.modeSwitch.autoFullTextNotice'
    ? translated
    : 'Switched to Full Text for longer research text';
}

function getHistoryDisplayQuery(item) {
  const redactedQuery = typeof item?.redactedQuery === 'string' ? item.redactedQuery : '';
  if (redactedQuery.trim() && redactedQuery !== REDACTED_QUERY_PLACEHOLDER) {
    return redactedQuery;
  }

  const rawQuery = typeof item?.rawQuerySessionOnly === 'string' ? item.rawQuerySessionOnly : '';
  if (rawQuery.trim()) {
    return rawQuery;
  }

  return typeof item?.query === 'string' ? item.query : '';
}

function isUserNoteExpanded(turnId) {
  return Boolean(Reflect.get(expandedUserNotes.value, turnId));
}

function toggleUserNote(turnId) {
  const nextExpandedUserNotes = { ...expandedUserNotes.value };
  Reflect.set(nextExpandedUserNotes, turnId, !Reflect.get(expandedUserNotes.value, turnId));
  expandedUserNotes.value = nextExpandedUserNotes;
}

function getHoveredNotePhenotype(turnId) {
  return Reflect.get(hoveredTextProcessPhenotypes.value, turnId) || null;
}

function setHoveredNotePhenotype(turnId, phenotypeId) {
  const nextHoveredTextProcessPhenotypes = { ...hoveredTextProcessPhenotypes.value };
  Reflect.set(nextHoveredTextProcessPhenotypes, turnId, phenotypeId);
  hoveredTextProcessPhenotypes.value = nextHoveredTextProcessPhenotypes;
}

function clearHoveredNotePhenotype(turnId) {
  const nextHoveredTextProcessPhenotypes = { ...hoveredTextProcessPhenotypes.value };
  Reflect.set(nextHoveredTextProcessPhenotypes, turnId, null);
  hoveredTextProcessPhenotypes.value = nextHoveredTextProcessPhenotypes;
}

function handleAnnotatedTextHover(turnId, termIds) {
  if (!Array.isArray(termIds) || termIds.length === 0) {
    return;
  }

  setHoveredNotePhenotype(turnId, termIds[0]);
}

function setMode(mode) {
  if (mode === 'query' && showAutoSwitchNotice.value) {
    autoSwitchSuppressed.value = true;
  }
  forceEndpointMode.value = mode;
  modeSelectionSource.value = 'manual';
}

function submitQuery(isAutoSubmit = false) {
  return queryInterfaceController.submitQuery(isAutoSubmit);
}

function cancelPiiReview() {
  piiReviewFlow.cancelPiiReview();
}

function redactPiiInInput() {
  return queryInterfaceController.redactPiiInInput();
}

function continueWithPiiRedaction() {
  return queryInterfaceController.continueWithPiiRedaction();
}

// Initialization (formerly created())
fetchAvailableModels();

</script>

<style scoped>
.search-container {
  max-width: 900px; /* Slightly wider for comfort */
  width: 100%;
  margin-top: 0;
}

/* Advanced options panel specific styling */
#advanced-options-panel .text-subtitle-2 {
  font-size: 0.875rem !important;
  color: rgba(var(--v-theme-on-surface), 0.7);
}
#advanced-options-panel .text-caption {
  font-size: 0.75rem !important;
  color: rgba(var(--v-theme-on-surface), 0.7); /* Updated for WCAG AA contrast (7.0:1 ratio) */
}
#advanced-options-panel .v-select,
#advanced-options-panel .v-text-field,
#advanced-options-panel .v-switch {
  font-size: 0.8rem; /* Smaller font for inputs */
}
#advanced-options-panel .v-col {
  padding-top: 8px !important; /* Add top/bottom padding to columns */
  padding-bottom: 8px !important;
  padding-left: 6px !important;
  padding-right: 6px !important;
}
#advanced-options-panel .v-slider {
  margin-top: -4px; /* Adjust slider position slightly */
}

/* Conversation and Bubbles */
.conversation-container {
  max-height: calc(100vh - 280px); /* Adjust based on final header/footer height */
  min-height: 200px;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 16px 8px; /* Add some padding */
  scroll-behavior: smooth;
  margin-bottom: 24px;
  border-radius: 8px;
}

.query-bubble--document {
  padding: 12px 14px;
}

/* Webkit Scrollbar Styles */
.conversation-container::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
.conversation-container::-webkit-scrollbar-thumb {
  background: rgba(var(--v-theme-on-surface), 0.25);
  border-radius: 4px;
}
.conversation-container::-webkit-scrollbar-thumb:hover {
  background: rgba(var(--v-theme-on-surface), 0.35);
}
.conversation-container::-webkit-scrollbar-track {
  background: rgba(var(--v-theme-on-surface), 0.05);
  border-radius: 4px;
}

.query-bubble {
  background-color: rgba(var(--v-theme-primary), 0.12);
  color: rgb(var(--v-theme-on-surface));
  border-radius: 18px 18px 18px 4px; /* Chat bubble style */
  padding: 10px 15px;
  max-width: 80%;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.response-bubble {
  background-color: rgba(var(--v-theme-on-surface), 0.03);
  border: 1px solid rgba(var(--v-theme-on-surface), 0.08);
  border-radius: 18px 18px 4px 18px; /* Chat bubble style */
  padding: 10px 15px;
  width: 100%; /* Bot bubble can take more width */
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.user-query {
  justify-content: flex-end;
  margin-left: auto;
} /* Align user to right */
.user-query .v-avatar {
  order: 1;
  margin-left: 8px;
  margin-right: 0;
} /* Avatar on right */
.user-query .query-bubble {
  order: 0;
}

.bot-response {
  justify-content: flex-start;
}
.bot-response .v-avatar {
  margin-right: 8px;
}

/* Accessibility & General UI Polish */
:deep(.text-high-emphasis) {
  color: rgba(var(--v-theme-on-surface), 0.87) !important;
  font-weight: 500;
}
:deep(.v-field--variant-outlined) {
  background-color: rgb(var(--v-theme-surface)) !important;
  box-shadow: none;
}
:deep(.v-field__input) {
  color: rgba(var(--v-theme-on-surface), 0.87) !important;
}
:deep(.v-field__input::placeholder) {
  color: rgba(
    var(--v-theme-on-surface),
    0.7
  ) !important; /* Updated for WCAG AA contrast (7.0:1 ratio) */
}

.v-tooltip > .v-overlay__content {
  font-size: 0.75rem;
  padding: 4px 8px;
}

/* Collection Panel Enhancements */
.v-navigation-drawer .v-list-item-title {
  font-weight: 500;
}
.v-navigation-drawer .v-list-item-subtitle.wrap-text {
  white-space: normal;
  overflow: visible;
  text-overflow: clip;
  display: -webkit-box;
  -webkit-line-clamp: 3; /* Show up to 3 lines */
  -webkit-box-orient: vertical;
  line-height: 1.3em;
  max-height: 3.9em; /* 1.3em * 3 lines */
}
.v-navigation-drawer .v-btn {
  text-transform: none; /* Nicer button text */
}
</style>
