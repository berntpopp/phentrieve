<template>
  <div class="search-container mx-auto px-2">
    <!-- Clean Search Bar with Integrated Button -->
    <div class="search-bar-container pt-0 px-2 pb-2 pa-sm-3">
      <v-sheet
        rounded="xl"
        elevation="0"
        :class="[
          'search-bar',
          isTextProcessModeActive ? 'search-bar--text-process' : 'search-bar--query',
        ]"
        color="white"
      >
        <div
          :class="[
            'search-shell',
            'd-flex',
            'flex-wrap',
            'flex-sm-nowrap',
            isTextProcessModeActive ? 'search-shell--text-process' : 'align-center',
          ]"
        >
          <v-textarea
            v-if="isTextProcessModeActive"
            v-model="queryText"
            density="comfortable"
            variant="outlined"
            hide-details
            class="search-input search-input--text-process ml-2 ml-sm-3 flex-grow-1"
            :disabled="isLoading"
            bg-color="white"
            color="primary"
            rows="3"
            auto-grow
            clearable
            placeholder="Paste or type a clinical note"
            :aria-label="$t('queryInterface.accessibility.textProcessInputLabel')"
            :aria-description="getTextProcessInputDescription()"
            @keydown.enter.prevent="!isLoading && queryText.trim() ? submitQuery() : null"
          >
            <template #label>
              <span class="text-high-emphasis">Clinical note</span>
            </template>
          </v-textarea>
          <v-text-field
            v-else
            ref="queryInput"
            v-model="queryText"
            density="comfortable"
            variant="outlined"
            hide-details
            class="search-input search-input--query ml-2 ml-sm-3 flex-grow-1"
            :disabled="isLoading"
            bg-color="white"
            color="primary"
            clearable
            :aria-label="$t('queryInterface.accessibility.queryInputLabel')"
            :aria-description="getQueryInputDescription()"
            @keydown.enter.prevent="!isLoading && queryText.trim() ? submitQuery() : null"
          >
            <template #label>
              <span class="text-high-emphasis">Phenotype query</span>
            </template>
          </v-text-field>

          <div
            :class="[
              'search-action',
              'd-flex',
              'align-center',
              isTextProcessModeActive ? 'search-action--overlay' : null,
            ]"
          >
            <v-btn
              ref="searchButton"
              color="primary"
              variant="text"
              icon
              rounded="circle"
              :loading="isLoading"
              :disabled="!queryText.trim()"
              class="search-submit-button"
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
              @click="setMode('query')"
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
              @click="setMode('textProcess')"
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
            :aria-label="getAdvancedOptionsToggleLabel()"
            :aria-expanded="showAdvancedOptions.toString()"
            aria-controls="advanced-options-panel"
            data-tutorial-step="advanced-options"
            @click="showAdvancedOptions = !showAdvancedOptions"
          >
            <v-icon start size="small">
              {{ showAdvancedOptions ? 'mdi-cog-outline' : 'mdi-tune-variant' }}
            </v-icon>
            Settings
          </v-btn>
        </div>
      </div>

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

      <div
        v-if="showAutoSwitchNotice"
        data-testid="mode-auto-switch-notice"
        class="mode-switch-notice px-3 pt-2"
      >
        {{ getAutoSwitchNoticeLabel() }}
      </div>
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
                <div data-testid="user-note-summary" class="user-note-summary">
                  <div class="user-note-summary__header">
                    <v-icon size="small" color="primary">mdi-file-document-outline</v-icon>
                    <span class="text-body-2 font-weight-medium">Clinical note</span>
                    <span class="text-caption text-medium-emphasis">
                      {{ formatDocumentSummaryMeta(item.query) }}
                    </span>
                    <v-btn
                      data-testid="user-note-summary-toggle"
                      size="x-small"
                      variant="text"
                      icon
                      @click="toggleUserNote(item.id)"
                    >
                      <v-icon size="small">
                        {{ isUserNoteExpanded(item.id) ? 'mdi-chevron-up' : 'mdi-chevron-down' }}
                      </v-icon>
                    </v-btn>
                  </div>
                  <p
                    v-if="!isUserNoteExpanded(item.id)"
                    class="mb-0 text-body-2 text-medium-emphasis user-note-summary__preview"
                  >
                    {{ summarizeDocumentQuery(item.query) }}
                  </p>
                  <div
                    v-if="isUserNoteExpanded(item.id)"
                    data-testid="user-note-expanded"
                    class="user-note-summary__expanded"
                  >
                    <span v-for="segment in buildUserNoteSegments(item)" :key="segment.key">
                      <mark
                        v-if="segment.highlighted"
                        data-testid="annotated-note-span"
                        :title="segment.tooltip"
                        :class="{
                          'annotated-note-span--active':
                            getHoveredNotePhenotype(item.id) &&
                            segment.termIds.includes(getHoveredNotePhenotype(item.id)),
                        }"
                        @mouseenter="handleAnnotatedTextHover(item.id, segment.termIds)"
                        @mouseleave="clearHoveredNotePhenotype(item.id)"
                      >
                        {{ segment.text }}
                      </mark>
                      <span v-else>{{ segment.text }}</span>
                    </span>
                  </div>
                </div>
              </template>
              <p v-else class="mb-0" style="white-space: pre-wrap">
                {{ item.query }}
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

              <div v-else-if="item.type === 'textProcess'" class="full-text-receipt">
                <div class="full-text-receipt__title">Full-text analysis ready</div>
                <div class="text-caption text-medium-emphasis">
                  {{ formatFullTextReceipt(item) }}
                </div>
                <div
                  v-if="getTextProcessPhenotypes(item).length > 0"
                  class="d-flex justify-end mt-3"
                >
                  <v-btn
                    data-testid="full-text-response-add-all"
                    size="small"
                    rounded="pill"
                    color="primary"
                    variant="tonal"
                    @click="
                      handleAddAllToCollection(
                        getTextProcessPhenotypes(item).map((phenotype) =>
                          normalizeTextProcessPhenotype(phenotype)
                        )
                      )
                    "
                  >
                    <v-icon start size="small">mdi-plus-circle</v-icon>
                    Add all
                  </v-btn>
                </div>
                <v-list
                  v-if="getTextProcessPhenotypes(item).length > 0"
                  lines="two"
                  class="rounded-lg mt-3 full-text-response-list"
                >
                  <div
                    v-for="(phenotype, index) in getTextProcessPhenotypes(item)"
                    :key="`${item.id}-${phenotype.hpo_id}`"
                    data-testid="full-text-response-phenotype"
                    class="full-text-response-phenotype"
                    :class="{
                      'full-text-response-phenotype--active':
                        getHoveredNotePhenotype(item.id) === phenotype.hpo_id,
                    }"
                    @mouseenter="setHoveredNotePhenotype(item.id, phenotype.hpo_id)"
                    @mouseleave="clearHoveredNotePhenotype(item.id)"
                  >
                    <ResultItem
                      :result="mapTextProcessPhenotypeToResult(phenotype)"
                      :rank="index + 1"
                      :is-collected="
                        conversationStore.collectedPhenotypes.some(
                          (entry) => entry.hpo_id === phenotype.hpo_id
                        )
                      "
                      @add-to-collection="
                        handleAddToCollection(normalizeTextProcessPhenotype(phenotype))
                      "
                    />
                  </div>
                </v-list>
              </div>

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
import ResultItem from './ResultItem.vue';
import ConversationSkeleton from './ConversationSkeleton.vue';
import AdvancedOptionsPanel from './AdvancedOptionsPanel.vue';
import PhenotypeCollectionPanel from './PhenotypeCollectionPanel.vue';
import PhentrieveService from '../services/PhentrieveService';
import { logService } from '../services/logService';
import { useQueryPreferencesStore } from '../stores/queryPreferences';
import { useConversationStore } from '../stores/conversation';
import { useFullTextWorkspaceStore } from '../stores/fullTextWorkspace';
import { useAdvancedOptions } from '../composables/useAdvancedOptions';
import { usePhenotypeCollection } from '../composables/usePhenotypeCollection';

const DEFAULT_TEXT_PROCESS_LLM_OPTIONS = Object.freeze({
  llmModel: 'gemini-3.1-flash-lite-preview',
  llmMode: 'two_phase',
});

export default {
  name: 'QueryInterface',
  components: {
    ResultsDisplay,
    ResultItem,
    ConversationSkeleton,
    AdvancedOptionsPanel,
    PhenotypeCollectionPanel,
  },
  setup() {
    const conversationStore = useConversationStore();
    const fullTextWorkspaceStore = useFullTextWorkspaceStore();

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
      fullTextWorkspaceStore,
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
    getAutoSwitchNoticeLabel() {
      const translated = this.$t('queryInterface.modeSwitch.autoFullTextNotice');
      return translated && translated !== 'queryInterface.modeSwitch.autoFullTextNotice'
        ? translated
        : 'Switched to Full Text for longer clinical text';
    },
    summarizeDocumentQuery(query) {
      if (typeof query !== 'string') {
        return '';
      }

      const compact = query.replace(/\s+/g, ' ').trim();
      if (compact.length <= 120) {
        return compact;
      }

      return `${compact.slice(0, 117)}...`;
    },
    formatDocumentSummaryMeta(query) {
      if (typeof query !== 'string') {
        return '';
      }

      const charCount = query.trim().length;
      if (charCount === 0) {
        return '';
      }

      const wordCount = query.trim().split(/\s+/).filter(Boolean).length;

      return `${wordCount} words`;
    },
    isUserNoteExpanded(turnId) {
      return Boolean(this.expandedUserNotes[turnId]);
    },
    toggleUserNote(turnId) {
      this.expandedUserNotes = {
        ...this.expandedUserNotes,
        [turnId]: !this.expandedUserNotes[turnId],
      };
    },
    getTextProcessPhenotypes(item) {
      return Array.isArray(item?.response?.aggregated_hpo_terms)
        ? item.response.aggregated_hpo_terms.filter(
            (term) => term && typeof term.hpo_id === 'string' && typeof term.name === 'string'
          )
        : [];
    },
    parseScoreValue(value) {
      if (typeof value === 'number' && Number.isFinite(value)) {
        return value;
      }

      if (typeof value === 'string' && value.trim() !== '') {
        const parsed = Number.parseFloat(value);
        if (Number.isFinite(parsed)) {
          return parsed;
        }
      }

      return null;
    },
    mapTextProcessPhenotypeToResult(term) {
      const confidence = this.parseScoreValue(term.confidence);
      const similarity = this.parseScoreValue(term.similarity);

      return {
        hpo_id: term.hpo_id,
        label: term.name,
        confidence,
        score: confidence,
        scoreType: 'confidence',
        similarity: confidence ?? similarity ?? 0,
        definition: term.definition || '',
        synonyms: Array.isArray(term.synonyms) ? term.synonyms : [],
      };
    },
    getHoveredNotePhenotype(turnId) {
      return this.hoveredTextProcessPhenotypes[turnId] || null;
    },
    setHoveredNotePhenotype(turnId, phenotypeId) {
      this.hoveredTextProcessPhenotypes = {
        ...this.hoveredTextProcessPhenotypes,
        [turnId]: phenotypeId,
      };
    },
    clearHoveredNotePhenotype(turnId) {
      this.hoveredTextProcessPhenotypes = {
        ...this.hoveredTextProcessPhenotypes,
        [turnId]: null,
      };
    },
    normalizeAssertionStatus(status) {
      if (status === 'present') {
        return 'affirmed';
      }

      if (status === 'absent') {
        return 'negated';
      }

      if (status === 'affirmed' || status === 'negated' || status === 'unknown') {
        return status;
      }

      return 'unknown';
    },
    normalizeTextProcessPhenotype(term) {
      return {
        hpo_id: term.hpo_id,
        label: term.name,
        assertion_status: this.normalizeAssertionStatus(term.status),
      };
    },
    handleAnnotatedTextHover(turnId, termIds) {
      if (!Array.isArray(termIds) || termIds.length === 0) {
        return;
      }

      this.setHoveredNotePhenotype(turnId, termIds[0]);
    },
    buildUserNoteSegments(item) {
      const fallbackText = typeof item?.query === 'string' ? item.query : '';
      const chunks = Array.isArray(item?.response?.processed_chunks)
        ? item.response.processed_chunks
        : [];
      const terms = Array.isArray(item?.response?.aggregated_hpo_terms)
        ? item.response.aggregated_hpo_terms
        : [];
      const termLabels = new Map(
        terms
          .filter(
            (term) => term && typeof term.hpo_id === 'string' && typeof term.name === 'string'
          )
          .map((term) => [term.hpo_id, term.name])
      );
      const activePhenotypeId = this.getHoveredNotePhenotype(item?.id);

      if (!fallbackText || chunks.length === 0) {
        return [{ key: 'fallback-note', text: fallbackText, highlighted: false }];
      }

      const chunkOffsets = this.resolveChunkOffsetsInNote(fallbackText, chunks);
      const highlights = terms
        .filter((term) => !activePhenotypeId || term?.hpo_id === activePhenotypeId)
        .flatMap((term) =>
          (Array.isArray(term.text_attributions) ? term.text_attributions : []).map((attr) => ({
            ...attr,
            termId: term.hpo_id,
          }))
        )
        .map((attr) => {
          const offset =
            chunkOffsets.get(attr?.chunk_id) ??
            chunkOffsets.get(String(attr?.chunk_id)) ??
            chunkOffsets.get(Number(attr?.chunk_id));

          if (offset != null) {
            const start = offset + Math.max(0, attr.start_char ?? 0);
            const end = offset + Math.max(0, attr.end_char ?? 0);
            return {
              start: Math.max(0, Math.min(start, fallbackText.length)),
              end: Math.max(0, Math.min(end, fallbackText.length)),
              termIds: [attr.termId],
            };
          }

          const resolved = this.resolveMatchedTextRange(fallbackText, attr?.matched_text_in_chunk);
          return resolved
            ? {
                ...resolved,
                termIds: [attr.termId],
              }
            : null;
        })
        .filter(Boolean)
        .map((attr) => ({
          start: attr.start,
          end: attr.end,
          termIds: attr.termIds,
        }))
        .filter((attr) => attr.end > attr.start)
        .sort((left, right) => left.start - right.start);

      if (highlights.length === 0) {
        return [{ key: 'plain-note', text: fallbackText, highlighted: false }];
      }

      const merged = [];
      for (const range of highlights) {
        const previous = merged[merged.length - 1];
        if (previous && range.start <= previous.end) {
          previous.end = Math.max(previous.end, range.end);
          previous.termIds = [...new Set([...previous.termIds, ...range.termIds])];
        } else {
          merged.push({ ...range });
        }
      }

      const segments = [];
      let cursor = 0;
      merged.forEach((range, index) => {
        if (range.start > cursor) {
          segments.push({
            key: `plain-${index}-${cursor}`,
            text: fallbackText.slice(cursor, range.start),
            highlighted: false,
          });
        }
        segments.push({
          key: `mark-${index}-${range.start}`,
          text: fallbackText.slice(range.start, range.end),
          highlighted: true,
          termIds: range.termIds,
          tooltip: range.termIds
            .map((termId) => `${termLabels.get(termId) || termId} (${termId})`)
            .join(', '),
        });
        cursor = range.end;
      });

      if (cursor < fallbackText.length) {
        segments.push({
          key: `plain-tail-${cursor}`,
          text: fallbackText.slice(cursor),
          highlighted: false,
        });
      }

      return segments;
    },
    resolveChunkOffsetsInNote(noteText, chunks) {
      const offsets = new Map();
      let searchFrom = 0;

      chunks.forEach((chunk) => {
        const chunkText =
          typeof chunk?.text === 'string'
            ? chunk.text
            : typeof chunk?.chunk_text === 'string'
              ? chunk.chunk_text
              : '';

        if (!chunkText || typeof chunk?.chunk_id !== 'number') {
          return;
        }

        let offset = noteText.indexOf(chunkText, searchFrom);
        if (offset === -1) {
          offset = noteText.indexOf(chunkText);
        }
        if (offset === -1) {
          return;
        }

        offsets.set(chunk.chunk_id, offset);
        offsets.set(String(chunk.chunk_id), offset);
        searchFrom = Math.max(offset + 1, searchFrom);
      });

      return offsets;
    },
    resolveMatchedTextRange(noteText, matchedText) {
      if (typeof matchedText !== 'string' || matchedText.trim() === '') {
        return null;
      }

      const normalizedNote = noteText.toLowerCase();
      const normalizedMatch = matchedText.toLowerCase();
      const start = normalizedNote.indexOf(normalizedMatch);
      if (start === -1) {
        return null;
      }

      return {
        start,
        end: start + matchedText.length,
      };
    },
    formatFullTextReceipt(item) {
      const terms = item?.response?.aggregated_hpo_terms?.length ?? 0;
      const chunks = item?.response?.processed_chunks?.length ?? 0;
      if (chunks === 0 && typeof item?.query === 'string' && item.query.trim().length > 0) {
        return `${terms} findings in submitted note review`;
      }
      return `${terms} findings across ${chunks} document sections`;
    },
    setMode(mode) {
      if (mode === 'query' && this.showAutoSwitchNotice) {
        this.autoSwitchSuppressed = true;
      }
      this.forceEndpointMode = mode;
      this.modeSelectionSource = 'manual';
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

      if (useTextProcessMode) {
        this.expandedUserNotes = {
          ...this.expandedUserNotes,
          [queryId]: true,
        };
      }

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
        if (useTextProcessMode) {
          this.fullTextWorkspaceStore.initializeTurn(queryId);
          this.fullTextWorkspaceStore.setExpanded(queryId, true);
        }
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

.search-shell {
  gap: 8px;
}

.search-shell--text-process {
  align-items: flex-start;
}

.search-action {
  flex-shrink: 0;
}

.search-action--overlay {
  position: absolute;
  top: 12px;
  right: 14px;
  z-index: 1;
}

.search-submit-button {
  width: 34px !important;
  height: 34px !important;
  min-width: 34px !important;
  color: rgba(25, 82, 166, 0.7) !important;
}

.search-submit-button:hover {
  background: rgba(25, 82, 166, 0.08);
}

.search-input--text-process {
  margin-right: 64px !important;
}

.search-input--text-process :deep(.v-label) {
  font-size: 0.92rem;
  color: rgba(60, 72, 88, 0.62);
}

.search-input :deep(.v-field) {
  /* Removed box-shadow to rely on the search-bar border instead */
}

.search-input--query :deep(.v-field) {
  border-radius: 28px;
  min-height: 48px;
}

.search-input--text-process :deep(.v-field) {
  border-radius: 22px;
  min-height: 104px;
  padding-top: 8px;
  padding-bottom: 8px;
}

.search-input--text-process :deep(.v-field__input) {
  min-height: 88px;
}

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

.search-bar {
  border: 1px solid rgba(0, 0, 0, 0.15);
  border-radius: 24px;
  transition:
    padding 0.18s ease,
    border-color 0.18s ease;
}

.search-bar--query {
  padding: 6px 8px;
}

.search-bar--text-process {
  position: relative;
  padding: 10px 12px;
}

.search-input :deep(.v-field__outline) {
  --v-field-border-width: 0px; /* No internal outline needed */
}

@media (max-width: 600px) {
  .search-bar--query {
    padding: 4px 6px;
  }

  .search-bar--text-process {
    padding: 8px 10px;
    border-radius: 20px;
  }

  .search-shell--text-process {
    gap: 6px;
  }

  .search-action {
    align-self: flex-start;
    padding-top: 2px;
  }

  .search-action--overlay {
    top: 10px;
    right: 10px;
    padding-top: 0;
  }

  .search-input--text-process {
    margin-right: 56px !important;
  }

  .search-input--text-process :deep(.v-field) {
    border-radius: 18px;
    min-height: 96px;
  }

  .search-input--text-process :deep(.v-field__input) {
    min-height: 72px;
  }

  .search-input--text-process :deep(.v-label) {
    font-size: 0.88rem;
  }
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

.user-note-summary {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.user-note-summary__header {
  display: flex;
  align-items: center;
  gap: 8px;
}

.user-note-summary__header .text-caption {
  margin-left: auto;
}

.user-note-summary__preview {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.user-note-summary__expanded {
  margin-top: 8px;
  font-size: 0.95rem;
  line-height: 1.6;
  color: rgba(15, 23, 42, 0.9);
}

.user-note-summary__expanded mark {
  background: rgba(37, 99, 235, 0.16);
  color: inherit;
  border-radius: 4px;
  padding: 0 1px;
}

.annotated-note-span--active {
  background: rgba(37, 99, 235, 0.32) !important;
}

.full-text-receipt {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.full-text-receipt__title {
  font-weight: 600;
}

.full-text-response-list {
  background: transparent;
}

.full-text-response-phenotype {
  border-radius: 12px;
  transition:
    transform 0.16s ease,
    box-shadow 0.16s ease;
}

.full-text-response-phenotype--active {
  transform: translateY(-1px);
  box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
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

@media (max-width: 1100px) {
  .full-text-shell {
    grid-template-columns: 1fr;
  }

  .full-text-case-rail {
    position: static;
  }
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
