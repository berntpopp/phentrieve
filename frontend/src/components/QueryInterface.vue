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
      />

      <QueryModeControls
        :is-text-process-mode-active="isTextProcessModeActive"
        :is-loading="isLoading"
        :show-advanced-options="showAdvancedOptions"
        :advanced-options-toggle-label="getAdvancedOptionsToggleLabel()"
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
              :content-props="{ 'aria-label': $t('queryInterface.accessibility.userInput') }"
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
              :content-props="{ 'aria-label': $t('queryInterface.accessibility.response') }"
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

<script>
import { getCurrentInstance } from 'vue';
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

const DEFAULT_TEXT_PROCESS_LLM_OPTIONS = Object.freeze({
  llmModel: 'gemini-3.1-flash-lite',
  llmMode: 'two_phase',
});
const REDACTED_QUERY_PLACEHOLDER = '[redacted]';

export default {
  name: 'QueryInterface',
  components: {
    ResultsDisplay,
    ConversationSkeleton,
    AdvancedOptionsPanel,
    FullTextResponseReceipt,
    PiiReviewDialog,
    QueryForm,
    FullTextNoteCurator,
    QueryModeControls,
    QueryResultActions,
  },
  setup() {
    const instance = getCurrentInstance();
    const conversationStore = useConversationStore();
    const fullTextCurationStore = useFullTextCurationStore();
    const piiReviewFlow = usePiiReviewFlow({ logService });

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
    const queryInterfaceController = useQueryInterfaceController({
      getContext: () => {
        const vm = instance?.proxy;
        if (!vm) {
          return null;
        }

        return {
          routeQuery: vm.$route.query,
          nextTick: vm.$nextTick.bind(vm),
          replaceRouteQuery(query) {
            return vm.$router.replace({ query });
          },
          getState() {
            return {
              availableModels: vm.availableModels,
              selectedModel: vm.selectedModel,
              queryText: vm.queryText,
              similarityThreshold: vm.similarityThreshold,
              forceEndpointMode: vm.forceEndpointMode,
              chunkingStrategy: vm.chunkingStrategy,
              showAdvancedOptions: vm.showAdvancedOptions,
              modelsLoading: vm.modelsLoading,
              isLoading: vm.isLoading,
              shouldScrollToTop: vm.shouldScrollToTop,
              userHasScrolled: vm.userHasScrolled,
              isTextProcessModeActive: vm.isTextProcessModeActive,
              selectedLanguage: vm.selectedLanguage,
              textProcessOptions: vm.textProcessOptions,
              windowSize: vm.windowSize,
              stepSize: vm.stepSize,
              splitThreshold: vm.splitThreshold,
              minSegmentLength: vm.minSegmentLength,
              semanticModelForChunking: vm.semanticModelForChunking,
              retrievalModelForTextProcess: vm.retrievalModelForTextProcess,
              chunkRetrievalThreshold: vm.chunkRetrievalThreshold,
              numResultsPerChunk: vm.numResultsPerChunk,
              noAssertionDetectionForTextProcess: vm.noAssertionDetectionForTextProcess,
              assertionPreferenceForTextProcess: vm.assertionPreferenceForTextProcess,
              aggregatedTermConfidence: vm.aggregatedTermConfidence,
              topTermPerChunkForAggregation: vm.topTermPerChunkForAggregation,
              includeDetails: vm.includeDetails,
              numResults: vm.numResults,
              pendingPiiSubmission: vm.pendingPiiSubmission,
            };
          },
          setState(patch) {
            Object.assign(vm, patch);
          },
          setExpandedUserNote(turnId, expanded) {
            vm.expandedUserNotes = {
              ...vm.expandedUserNotes,
              [turnId]: expanded,
            };
          },
          piiReviewFlow,
          conversationStore: vm.conversationStore,
        };
      },
      service: PhentrieveService,
      logService,
    });
    return {
      conversationStore,
      fullTextCurationStore,
      piiReviewDialogVisible: piiReviewFlow.piiReviewDialogVisible,
      pendingPiiSubmission: piiReviewFlow.pendingPiiSubmission,
      piiReviewFlow,
      queryInterfaceController,
      // Advanced options
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
      // Phenotype collection
      phenopacketSubjectId,
      phenopacketSex,
      phenopacketDateOfBirth,
      addToPhenotypeCollection,
      removePhenotype,
      toggleAssertionStatus,
      clearPhenotypeCollection,
      toggleCollectionPanel,
      exportPhenotypes,
      exportPhenotypesAsPhenopacket,
    };
  },
  data() {
    return {
      queryText: '',
      selectedModel: null, // Will be set from API config
      availableModels: [], // Populated from API /info endpoint
      modelsLoading: true, // Track model loading state
      selectedLanguage: null, // Will be set to current locale in created()
      availableLanguages: [
        { text: this.$t('common.auto'), value: null },
        { text: 'English', value: 'en' },
        { text: 'German (Deutsch)', value: 'de' },
        { text: 'French (Français)', value: 'fr' },
        { text: 'Spanish (Español)', value: 'es' },
        { text: 'Dutch (Nederlands)', value: 'nl' },
      ],
      isLoading: false,
      lastUserScrollPosition: 0,
      userHasScrolled: false,
      shouldScrollToTop: false,
      sexOptions: [
        { title: this.$t('phenopacket.sexUnknown'), value: 0 },
        { title: this.$t('phenopacket.sexFemale'), value: 1 },
        { title: this.$t('phenopacket.sexMale'), value: 2 },
        { title: this.$t('phenopacket.sexOther'), value: 3 },
      ],
      forceEndpointMode: null,
      modeSelectionSource: null,
      autoSwitchSuppressed: false,
      expandedUserNotes: {},
      defaultTextProcessLlmOptions: DEFAULT_TEXT_PROCESS_LLM_OPTIONS,
      textProcessOptions: {
        extractionBackend: 'llm',
        ...DEFAULT_TEXT_PROCESS_LLM_OPTIONS,
      },
      chunkingStrategy: 'sliding_window_punct_conj_cleaned',
      semanticModelForChunking: null,
      retrievalModelForTextProcess: null,
      noAssertionDetectionForTextProcess: false,
      assertionPreferenceForTextProcess: 'dependency',
      topTermPerChunkForAggregation: false,
      // Phenopacket export error surface (snackbar feedback)
      exportErrorVisible: false,
      exportErrorMessage: '',
      hoveredTextProcessPhenotypes: {},
    };
  },
  computed: {
    // Determine if text processing mode is active based on text length or user force setting
    isTextProcessModeActive() {
      if (this.forceEndpointMode) {
        return this.forceEndpointMode === 'textProcess';
      }
      return this.queryText.length > this.inputTextLengthThreshold;
    },
    showAutoSwitchNotice() {
      return this.modeSelectionSource === 'auto' && this.forceEndpointMode === 'textProcess';
    },
    // Query parameters reused when a curator re-queries a single span / selection.
    curationQueryOptions() {
      return {
        model_name: this.selectedModel,
        language: this.selectedLanguage,
        num_results: 8,
        similarity_threshold: this.similarityThreshold,
      };
    },
    // Access includeDetails from Pinia store (persisted in localStorage)
    includeDetails: {
      get() {
        const store = useQueryPreferencesStore();
        return store.includeDetails;
      },
      set(value) {
        const store = useQueryPreferencesStore();
        store.setIncludeDetails(value);
      },
    },
  },
  watch: {
    'conversationStore.queryHistory.length': {
      handler(newLength) {
        logService.debug('Query history updated', {
          newHistoryLength: newLength,
        });
        // Drop curation state for turns evicted from history (maxHistoryLength)
        // so the persisted curation store does not accumulate orphaned entries.
        const liveIds = new Set(
          (this.conversationStore.queryHistory || []).map((entry) => entry.id)
        );
        Object.keys(this.fullTextCurationStore.turns || {}).forEach((turnId) => {
          if (!liveIds.has(turnId)) {
            this.fullTextCurationStore.dropTurn(turnId);
          }
        });
        this.$nextTick(() => {
          if (this.shouldScrollToTop && !this.userHasScrolled) {
            this.scrollToTop();
          }
        });
      },
    },
    selectedModel(newModel, oldModel) {
      if (oldModel !== null) {
        // Avoid resetting on initial mount
        logService.info('Model changed', { newModel: newModel, oldModel: oldModel });
        // Reset some query-specific settings to defaults when model changes
        this.resetToDefaults();
        logService.info('Reset query-specific settings to defaults due to model change.');
      }
    },
    queryText(newValue, oldValue) {
      if (typeof newValue !== 'string') {
        return;
      }

      if (newValue.length <= this.inputTextLengthThreshold) {
        this.autoSwitchSuppressed = false;
      }

      if (
        !this.autoSwitchSuppressed &&
        this.forceEndpointMode !== 'textProcess' &&
        newValue.length > this.inputTextLengthThreshold &&
        oldValue.length <= this.inputTextLengthThreshold
      ) {
        this.forceEndpointMode = 'textProcess';
        this.modeSelectionSource = 'auto';
      }
    },
  },
  created() {
    // Fetch available models from API
    this.fetchAvailableModels();

    // Set selectedLanguage based on current locale
    const currentLocale = this.$i18n.locale;
    // Only set if it's one of our supported languages
    const supportedLanguages = this.availableLanguages
      .filter((lang) => lang.value !== null)
      .map((lang) => lang.value);

    if (currentLocale && supportedLanguages.includes(currentLocale)) {
      this.selectedLanguage = currentLocale;
      logService.info(
        `Using UI locale '${currentLocale}' as default language for queries and text processing`
      );
    } else {
      // Keep the auto-detect (null) as fallback if locale is not supported
      logService.info(
        `Current locale '${currentLocale}' not in supported languages, using auto-detect`
      );
    }
  },
  methods: {
    /**
     * Wrapper around usePhenotypeCollection.exportAsPhenopacket that surfaces
     * errors via a snackbar instead of letting them propagate unhandled.
     * The composable throws on failure (intentionally, so it stays testable
     * and accessibility-friendly); this method catches and renders.
     */
    onExportPhenopacket() {
      try {
        this.exportPhenotypesAsPhenopacket();
      } catch (error) {
        this.exportErrorMessage = this.$t('queryInterface.phenotypeCollection.exportError');
        this.exportErrorVisible = true;
        // The composable already logs via logService.error; no need to re-log.
        void error;
      }
    },
    handleAddToCollection(phenotype) {
      this.addToPhenotypeCollection(phenotype);
      this.conversationStore.showCollectionPanel = true;
    },
    handleAddAllToCollection(phenotypes) {
      if (!Array.isArray(phenotypes)) {
        return;
      }

      phenotypes.forEach((phenotype) => {
        this.addToPhenotypeCollection(phenotype);
      });
      this.conversationStore.showCollectionPanel = true;
    },
    /**
     * Fetch available embedding models from API /info endpoint
     * Falls back to BioLORD if API is unavailable
     */
    async fetchAvailableModels() {
      return this.queryInterfaceController.fetchAvailableModels();
    },

    /**
     * Set fallback model when API is unavailable
     */
    setFallbackModel() {
      return this.queryInterfaceController.setFallbackModel();
    },

    applyUrlParametersAndAutoSubmit() {
      return this.queryInterfaceController.applyUrlParametersAndAutoSubmit();
    },
    handleUserScroll() {
      const container = this.$refs.conversationContainer;
      if (container) {
        if (Math.abs(container.scrollTop - this.lastUserScrollPosition) > 5) {
          // Threshold to detect actual scroll
          this.userHasScrolled = true;
        }
        this.lastUserScrollPosition = container.scrollTop;
      }
    },
    scrollToTop() {
      this.$nextTick(() => {
        const container = this.$refs.conversationContainer;
        if (container) {
          container.scrollTo({ top: 0, behavior: 'auto' }); // Use auto for instant jump
        }
      });
    },
    getTextProcessInputDescription() {
      const base = this.$t('queryInterface.accessibility.textProcessInputDescription');
      return this.isLoading
        ? `${base} ${this.$t('queryInterface.accessibility.processingInProgress')}`
        : base;
    },
    getQueryInputDescription() {
      const base = this.$t('queryInterface.accessibility.queryInputDescription');
      return this.isLoading
        ? `${base} ${this.$t('queryInterface.accessibility.searchInProgress')}`
        : base;
    },
    getAdvancedOptionsToggleLabel() {
      return this.showAdvancedOptions
        ? this.$t('queryInterface.accessibility.closeAdvancedOptions')
        : this.$t('queryInterface.accessibility.openAdvancedOptions');
    },
    getAutoSwitchNoticeLabel() {
      const translated = this.$t('queryInterface.modeSwitch.autoFullTextNotice');
      return translated && translated !== 'queryInterface.modeSwitch.autoFullTextNotice'
        ? translated
        : 'Switched to Full Text for longer clinical text';
    },
    getHistoryDisplayQuery(item) {
      const redactedQuery = typeof item?.redactedQuery === 'string' ? item.redactedQuery : '';
      if (redactedQuery.trim() && redactedQuery !== REDACTED_QUERY_PLACEHOLDER) {
        return redactedQuery;
      }

      const rawQuery =
        typeof item?.rawQuerySessionOnly === 'string' ? item.rawQuerySessionOnly : '';
      if (rawQuery.trim()) {
        return rawQuery;
      }

      return typeof item?.query === 'string' ? item.query : '';
    },
    isUserNoteExpanded(turnId) {
      return Boolean(Reflect.get(this.expandedUserNotes, turnId));
    },
    toggleUserNote(turnId) {
      const nextExpandedUserNotes = { ...this.expandedUserNotes };
      Reflect.set(nextExpandedUserNotes, turnId, !Reflect.get(this.expandedUserNotes, turnId));
      this.expandedUserNotes = nextExpandedUserNotes;
    },
    getHoveredNotePhenotype(turnId) {
      return Reflect.get(this.hoveredTextProcessPhenotypes, turnId) || null;
    },
    setHoveredNotePhenotype(turnId, phenotypeId) {
      const nextHoveredTextProcessPhenotypes = { ...this.hoveredTextProcessPhenotypes };
      Reflect.set(nextHoveredTextProcessPhenotypes, turnId, phenotypeId);
      this.hoveredTextProcessPhenotypes = nextHoveredTextProcessPhenotypes;
    },
    clearHoveredNotePhenotype(turnId) {
      const nextHoveredTextProcessPhenotypes = { ...this.hoveredTextProcessPhenotypes };
      Reflect.set(nextHoveredTextProcessPhenotypes, turnId, null);
      this.hoveredTextProcessPhenotypes = nextHoveredTextProcessPhenotypes;
    },
    handleAnnotatedTextHover(turnId, termIds) {
      if (!Array.isArray(termIds) || termIds.length === 0) {
        return;
      }

      this.setHoveredNotePhenotype(turnId, termIds[0]);
    },
    setMode(mode) {
      if (mode === 'query' && this.showAutoSwitchNotice) {
        this.autoSwitchSuppressed = true;
      }
      this.forceEndpointMode = mode;
      this.modeSelectionSource = 'manual';
    },
    async submitQuery(isAutoSubmit = false) {
      return this.queryInterfaceController.submitQuery(isAutoSubmit);
    },
    cancelPiiReview() {
      this.piiReviewFlow.cancelPiiReview();
    },
    redactPiiInInput() {
      return this.queryInterfaceController.redactPiiInInput();
    },
    continueWithPiiRedaction() {
      return this.queryInterfaceController.continueWithPiiRedaction();
    },
  },
};
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
  color: rgba(0, 0, 0, 0.7);
}
#advanced-options-panel .text-caption {
  font-size: 0.75rem !important;
  color: rgba(0, 0, 0, 0.7); /* Updated for WCAG AA contrast (7.0:1 ratio) */
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
  background: #ccc;
  border-radius: 4px;
}
.conversation-container::-webkit-scrollbar-thumb:hover {
  background: #bbb;
}
.conversation-container::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 4px;
}

.query-bubble {
  background-color: #e3f2fd; /* Lighter blue for user */
  color: #1e3a8a; /* Darker blue text for contrast */
  border-radius: 18px 18px 18px 4px; /* Chat bubble style */
  padding: 10px 15px;
  max-width: 80%;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.response-bubble {
  background-color: #f8f9fa; /* Very light grey for bot */
  border: 1px solid #e9ecef;
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
  color: rgba(0, 0, 0, 0.87) !important;
  font-weight: 500;
}
:deep(.v-field--variant-outlined) {
  background-color: #ffffff !important;
  box-shadow: none;
}
:deep(.v-field__input) {
  color: rgba(0, 0, 0, 0.87) !important;
}
:deep(.v-field__input::placeholder) {
  color: rgba(0, 0, 0, 0.7) !important; /* Updated for WCAG AA contrast (7.0:1 ratio) */
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
