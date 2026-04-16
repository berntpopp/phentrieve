<template>
  <div class="search-container mx-auto px-2">
    <!-- Clean Search Bar with Integrated Button -->
    <div class="search-bar-container pt-0 px-2 pb-2 pa-sm-3">
      <v-sheet rounded="pill" elevation="0" class="pa-1 pa-sm-2 search-bar" color="white">
        <div class="d-flex align-center flex-wrap flex-sm-nowrap">
          <v-textarea
            v-if="isTextProcessModeActive"
            v-model="queryText"
            density="comfortable"
            variant="outlined"
            hide-details
            class="search-input ml-2 ml-sm-3 flex-grow-1"
            :disabled="isLoading"
            bg-color="white"
            color="primary"
            rows="3"
            auto-grow
            clearable
            :aria-label="$t('queryInterface.accessibility.textProcessInputLabel')"
            :aria-description="getTextProcessInputDescription()"
            @keydown.enter.prevent="!isLoading && queryText.trim() ? submitQuery() : null"
          >
            <template #label>
              <span class="text-high-emphasis"
                >{{ $t('queryInterface.inputLabel') }} ({{
                  $t('queryInterface.documentModeLabel')
                }})</span
              >
            </template>
          </v-textarea>
          <v-text-field
            v-else
            ref="queryInput"
            v-model="queryText"
            density="comfortable"
            variant="outlined"
            hide-details
            class="search-input ml-2 ml-sm-3 flex-grow-1"
            :disabled="isLoading"
            bg-color="white"
            color="primary"
            clearable
            :aria-label="$t('queryInterface.accessibility.queryInputLabel')"
            :aria-description="getQueryInputDescription()"
            @keydown.enter.prevent="!isLoading && queryText.trim() ? submitQuery() : null"
          >
            <template #label>
              <span class="text-high-emphasis"
                >{{ $t('queryInterface.inputLabel') }} ({{
                  $t('queryInterface.queryModeLabel')
                }})</span
              >
            </template>
          </v-text-field>

          <div class="d-flex align-center">
            <v-tooltip
              location="top"
              :text="$t('queryInterface.tooltips.advancedOptions')"
              :content-props="{ 'aria-label': $t('queryInterface.tooltips.advancedOptions') }"
            >
              <template #activator="{ props }">
                <v-btn
                  v-bind="props"
                  icon
                  variant="text"
                  color="primary"
                  class="mx-1 mx-sm-2"
                  :disabled="isLoading"
                  :aria-label="getAdvancedOptionsToggleLabel()"
                  :aria-expanded="showAdvancedOptions.toString()"
                  aria-controls="advanced-options-panel"
                  size="small"
                  data-tutorial-step="advanced-options"
                  @click="showAdvancedOptions = !showAdvancedOptions"
                >
                  <v-icon>
                    {{ showAdvancedOptions ? 'mdi-cog-outline' : 'mdi-tune-variant' }}
                  </v-icon>
                </v-btn>
              </template>
            </v-tooltip>

            <v-btn
              ref="searchButton"
              color="primary"
              variant="tonal"
              icon
              rounded="circle"
              :loading="isLoading"
              :disabled="!queryText.trim()"
              class="mr-1 mr-sm-2"
              :aria-label="$t('queryInterface.accessibility.searchButton')"
              size="small"
              data-tutorial-step="search-button"
              @click="submitQuery"
            >
              <v-icon>mdi-magnify</v-icon>
            </v-btn>
          </div>
        </div>
      </v-sheet>

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
      <!-- Skeleton loading during hydration -->
      <ConversationSkeleton
        v-if="conversationStore.isHydrating"
        :count="2"
        class="hydration-skeleton"
      />

      <!-- Actual conversation history (shown after hydration) -->
      <template v-else>
        <div v-for="item in conversationStore.queryHistory" :key="item.id" class="mb-4">
          <!-- User query -->
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
            <div class="query-bubble">
              <p class="mb-0" style="white-space: pre-wrap">
                {{ item.query }}
              </p>
            </div>
          </div>

          <!-- API response -->
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
                v-else
                :key="'results-' + item.id"
                :response-data="item.response"
                :result-type="item.type"
                :error="item.error"
                :collected-phenotypes="conversationStore.collectedPhenotypes"
                @add-to-collection="addToPhenotypeCollection"
              />
            </div>
          </div>
        </div>
      </template>
    </div>

    <!-- Phenotype Collection Panel -->
    <PhenotypeCollectionPanel
      v-model:subject-id="phenopacketSubjectId"
      v-model:sex="phenopacketSex"
      v-model:date-of-birth="phenopacketDateOfBirth"
      :phenotypes="conversationStore.collectedPhenotypes"
      :panel-open="conversationStore.showCollectionPanel"
      :sex-options="sexOptions"
      @toggle-panel="toggleCollectionPanel"
      @update:panel-open="conversationStore.showCollectionPanel = $event"
      @remove="removePhenotype"
      @toggle-assertion="toggleAssertionStatus"
      @export-text="exportPhenotypes"
      @export-json="onExportPhenopacket"
      @clear="clearPhenotypeCollection"
    />

    <v-snackbar
      v-model="exportErrorVisible"
      color="error"
      location="bottom"
      timeout="6000"
      role="alert"
    >
      {{ exportErrorMessage }}
      <template #actions>
        <v-btn variant="text" @click="exportErrorVisible = false">
          {{ $t('common.dismiss') }}
        </v-btn>
      </template>
    </v-snackbar>
  </div>
</template>

<script>
import ResultsDisplay from './ResultsDisplay.vue';
import ConversationSkeleton from './ConversationSkeleton.vue';
import AdvancedOptionsPanel from './AdvancedOptionsPanel.vue';
import PhenotypeCollectionPanel from './PhenotypeCollectionPanel.vue';
import PhentrieveService from '../services/PhentrieveService';
import { logService } from '../services/logService';
import { useQueryPreferencesStore } from '../stores/queryPreferences';
import { useConversationStore } from '../stores/conversation';
import { useAdvancedOptions } from '../composables/useAdvancedOptions';
import { usePhenotypeCollection } from '../composables/usePhenotypeCollection';

const DEFAULT_TEXT_PROCESS_LLM_OPTIONS = Object.freeze({
  llmModel: 'gpt-5.4-mini',
  llmMode: 'two_phase',
});

export default {
  name: 'QueryInterface',
  components: {
    ResultsDisplay,
    ConversationSkeleton,
    AdvancedOptionsPanel,
    PhenotypeCollectionPanel,
  },
  setup() {
    const conversationStore = useConversationStore();

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

    return {
      conversationStore,
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
      defaultTextProcessLlmOptions: DEFAULT_TEXT_PROCESS_LLM_OPTIONS,
      textProcessOptions: {
        extractionBackend: 'standard',
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
    /**
     * Fetch available embedding models from API /info endpoint
     * Falls back to BioLORD if API is unavailable
     */
    async fetchAvailableModels() {
      this.modelsLoading = true;
      try {
        const config = await PhentrieveService.getConfigInfo();
        if (config.available_embedding_models && config.available_embedding_models.length > 0) {
          this.availableModels = config.available_embedding_models.map((m) => ({
            // Use the model name (last part of ID) as display text, e.g., "BioLORD-2023-M"
            text: m.id.split('/').pop(),
            value: m.id,
          }));
          // Set default model from API config
          if (config.default_embedding_model) {
            this.selectedModel = config.default_embedding_model;
          } else {
            this.selectedModel = this.availableModels[0]?.value || null;
          }
          logService.info('Loaded embedding models from API', {
            count: this.availableModels.length,
            defaultModel: this.selectedModel,
          });
        } else {
          this.setFallbackModel();
        }
      } catch (error) {
        logService.warn('Failed to fetch models from API, using fallback', {
          error: error.message,
        });
        this.setFallbackModel();
      } finally {
        this.modelsLoading = false;
      }
    },

    /**
     * Set fallback model when API is unavailable
     */
    setFallbackModel() {
      this.availableModels = [{ text: 'BioLORD 2023-M', value: 'FremyCompany/BioLORD-2023-M' }];
      this.selectedModel = 'FremyCompany/BioLORD-2023-M';
      logService.info('Using fallback embedding model', { model: this.selectedModel });
    },

    applyUrlParametersAndAutoSubmit() {
      const queryParams = this.$route.query;
      logService.debug('Raw URL query parameters:', { ...queryParams });
      let advancedOptionsWereSet = false;
      const parseBooleanParam = (val) =>
        typeof val === 'string' && (val.toLowerCase() === 'true' || val === '1');

      if (queryParams.text !== undefined) {
        this.queryText = queryParams.text;
      }
      if (queryParams.model !== undefined) {
        const validModels = this.availableModels.map((m) => m.value);
        if (validModels.includes(queryParams.model)) {
          this.selectedModel = queryParams.model;
          advancedOptionsWereSet = true;
        }
      }
      if (queryParams.threshold !== undefined) {
        const val = parseFloat(queryParams.threshold);
        if (!isNaN(val) && val >= 0 && val <= 1) {
          this.similarityThreshold = val;
          advancedOptionsWereSet = true;
        }
      }
      // Add processing for text process specific URL params
      if (queryParams.forceEndpointMode) {
        this.forceEndpointMode = queryParams.forceEndpointMode;
        advancedOptionsWereSet = true;
      }
      if (queryParams.chunkingStrategy) {
        this.chunkingStrategy = queryParams.chunkingStrategy;
        advancedOptionsWereSet = true;
      }

      if (advancedOptionsWereSet) this.showAdvancedOptions = true;

      const performAutoSubmit =
        queryParams.autoSubmit !== undefined
          ? parseBooleanParam(queryParams.autoSubmit)
          : queryParams.text !== undefined;

      if (performAutoSubmit && this.queryText && this.queryText.trim()) {
        logService.info('Auto-submitting query based on URL parameters.');
        this.$nextTick(() => setTimeout(() => this.submitQuery(true), 300)); // autoSubmit = true
      }
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
    async submitQuery(isAutoSubmit = false) {
      const queryTextTrimmed = this.queryText.trim();
      if (!queryTextTrimmed) {
        logService.warn('Empty query submission prevented');
        return;
      }

      const useTextProcessMode = this.isTextProcessModeActive;
      this.isLoading = true;
      const currentQuery = queryTextTrimmed;

      // Add query to store and get the generated ID
      const queryId = this.conversationStore.addQuery({
        query: currentQuery,
        loading: true,
        type: useTextProcessMode ? 'textProcess' : 'query',
      });

      if (!isAutoSubmit) this.queryText = ''; // Clear input only if not auto-submitting (URL params might be active)

      this.shouldScrollToTop = true;
      this.userHasScrolled = false;

      try {
        let response;
        if (useTextProcessMode) {
          const textProcessData = {
            text: currentQuery,
            extractionBackend: this.textProcessOptions.extractionBackend,
            llmModel: this.textProcessOptions.llmModel,
            llmMode: this.textProcessOptions.llmMode,
            language: this.selectedLanguage,
            chunkingStrategy: this.chunkingStrategy,
            windowSize: this.windowSize,
            stepSize: this.stepSize,
            splitThreshold: this.splitThreshold,
            minSegmentLength: this.minSegmentLength,
            semanticModelForChunking: this.semanticModelForChunking || this.selectedModel,
            retrievalModelForTextProcess: this.retrievalModelForTextProcess || this.selectedModel,
            trustRemoteCode: true,
            chunkRetrievalThreshold: this.chunkRetrievalThreshold,
            numResultsPerChunk: this.numResultsPerChunk,
            noAssertionDetectionForTextProcess: this.noAssertionDetectionForTextProcess,
            assertionPreferenceForTextProcess: this.assertionPreferenceForTextProcess,
            aggregatedTermConfidence: this.aggregatedTermConfidence,
            topTermPerChunkForAggregation: this.topTermPerChunkForAggregation,
            includeDetails: this.includeDetails,
          };
          logService.info('Sending to /text/process API', textProcessData);
          response = await PhentrieveService.processText(textProcessData);
        } else {
          const queryData = {
            text: currentQuery,
            model_name: this.selectedModel,
            language: this.selectedLanguage,
            num_results: this.numResults,
            similarity_threshold: this.similarityThreshold,
            query_assertion_language: this.selectedLanguage, // Pass selected language for query assertion
            detect_query_assertion: true, // Default to true for query mode now
            include_details: this.includeDetails,
          };
          logService.info('Sending to /query API', queryData);
          response = await PhentrieveService.queryHpo(queryData);
        }
        // Update the query response in the store
        this.conversationStore.updateQueryResponse(queryId, response);
      } catch (error) {
        // Update with error in the store
        this.conversationStore.updateQueryResponse(queryId, null, error);
        logService.error('Error submitting query/processing text', error);
      } finally {
        this.isLoading = false;
        if (!isAutoSubmit) {
          // Only clear URL params if it wasn't an auto-submit
          const newRouteQuery = { ...this.$route.query };
          delete newRouteQuery.autoSubmit; // Remove autoSubmit flag
          if (Object.keys(newRouteQuery).length !== Object.keys(this.$route.query).length - 1) {
            // Check if other params were there
            this.$router.replace({ query: newRouteQuery }).catch((err) => {
              if (err.name !== 'NavigationDuplicated' && err.name !== 'NavigationCancelled') {
                logService.warn('Error clearing autoSubmit from URL:', err);
              }
            });
          }
        }
      }
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

.collection-fab-position {
  margin: 16px;
  bottom: 72px !important; /* Increased to clear bottom menu bar */
  right: 16px !important;
  z-index: 1050; /* Ensure it's above navigation drawer backdrop */
}

.search-input {
  font-size: 1rem;
  line-height: 1.5;
}

.search-input :deep(.v-field) {
  border-radius: 28px; /* More rounded */
  min-height: 48px; /* Slightly taller */
  /* Removed box-shadow to rely on the search-bar border instead */
}
.search-input.v-textarea :deep(.v-field) {
  border-radius: 18px; /* Consistent rounding for textarea */
  padding-top: 8px;
  padding-bottom: 8px;
}

.search-input.v-textarea :deep(.v-field__input) {
  min-height: 80px; /* Keep textarea height if needed */
}

.search-bar {
  border: 1px solid rgba(0, 0, 0, 0.15); /* Added border back to search-bar */
  border-radius: 30px; /* Match outer radius if needed */
}

.search-input :deep(.v-field__outline) {
  --v-field-border-width: 0px; /* No internal outline needed */
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
