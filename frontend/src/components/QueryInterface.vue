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
            @keydown.enter.prevent="!isLoading && queryText.trim() ? submitQuery() : null"
            bg-color="white"
            color="primary"
            rows="3"
            auto-grow
            clearable
            aria-label="Clinical document input for text processing"
            :aria-description="'Enter longer clinical text for document processing' + (isLoading ? '. Processing in progress' : '')"
          >
            <template v-slot:label>
              <span class="text-high-emphasis">{{ $t('queryInterface.inputLabel') }} ({{ $t('queryInterface.documentModeLabel', 'Document Mode') }})</span>
            </template>
          </v-textarea>
          <v-text-field
            v-else
            v-model="queryText"
            density="comfortable"
            variant="outlined"
            hide-details
            class="search-input ml-2 ml-sm-3 flex-grow-1"
            :disabled="isLoading"
            @keydown.enter.prevent="!isLoading && queryText.trim() ? submitQuery() : null"
            bg-color="white"
            color="primary"
            clearable
            aria-label="Clinical text input field"
            :aria-description="'Enter clinical text to search for HPO terms' + (isLoading ? '. Search in progress' : '')"
            ref="queryInput"
          >
            <template v-slot:label>
              <span class="text-high-emphasis">{{ $t('queryInterface.inputLabel') }} ({{ $t('queryInterface.queryModeLabel', 'Query Mode') }})</span>
            </template>
          </v-text-field>

          <div class="d-flex align-center">
            <v-tooltip location="top" :text="$t('queryInterface.tooltips.advancedOptions')" role="tooltip">
              <template v-slot:activator="{ props }">
                <v-btn
                  v-bind="props"
                  icon
                  variant="text"
                  color="primary"
                  class="mx-1 mx-sm-2"
                  @click="showAdvancedOptions = !showAdvancedOptions"
                  :disabled="isLoading"
                  :aria-label="showAdvancedOptions ? 'Close Advanced Options' : 'Open Advanced Options'"
                  :aria-expanded="showAdvancedOptions.toString()"
                  aria-controls="advanced-options-panel"
                  size="small"
                >
                  <v-icon>{{ showAdvancedOptions ? 'mdi-cog-outline' : 'mdi-tune-variant' }}</v-icon>
                </v-btn>
              </template>
            </v-tooltip>

            <v-btn
              color="primary"
              variant="tonal"
              icon
              rounded="circle"
              @click="submitQuery"
              :loading="isLoading"
              :disabled="!queryText.trim()"
              class="mr-1 mr-sm-2"
              aria-label="Search HPO Terms"
              size="small"
              ref="searchButton"
            >
              <v-icon>mdi-magnify</v-icon>
            </v-btn>
          </div>
        </div>
      </v-sheet>

      <!-- Advanced Options Panel (Hidden by Default) -->
      <v-expand-transition>
        <v-sheet
          v-if="showAdvancedOptions && !isLoading"
          rounded="lg"
          elevation="1"
          class="mt-2 pa-3"
          id="advanced-options-panel"
          role="region"
          aria-label="Advanced search options"
          color="white"
          style="font-size: 0.8rem;"
        >
          <div class="text-subtitle-2 mb-2 px-1 font-weight-medium">{{ $t('queryInterface.advancedOptions.title') }}</div>

          <v-row dense>
            <v-col cols="12" md="6" class="pa-1">
              <v-tooltip location="bottom" :text="$t('queryInterface.tooltips.embeddingModel')" role="tooltip">
                <template v-slot:activator="{ props }">
                  <v-select
                    v-bind="props"
                    v-model="selectedModel"
                    :items="availableModels"
                    item-title="text"
                    item-value="value"
                    :disabled="isLoading"
                    variant="outlined"
                    density="compact"
                    aria-label="Select embedding model"
                    :aria-description="'Choose the model to use for text embedding. Currently selected: ' + selectedModel"
                    bg-color="white"
                    color="primary"
                    hide-details
                  >
                    <template v-slot:label>
                      <span class="text-caption">{{ $t('queryInterface.advancedOptions.embeddingModel') }}</span>
                    </template>
                  </v-select>
                </template>
              </v-tooltip>
            </v-col>

            <v-col cols="12" md="6" class="pa-1">
              <v-tooltip location="bottom" :text="$t('queryInterface.tooltips.similarityThreshold')" role="tooltip">
                <template v-slot:activator="{ props }">
                  <div>
                    <label :for="'similarity-slider'" class="text-caption mb-0 d-block" style="font-size: 0.7rem; padding-left: 4px;">{{ $t('queryInterface.advancedOptions.similarityThreshold') }}: {{ similarityThreshold.toFixed(2) }}</label>
                    <v-slider
                      v-bind="props"
                      v-model="similarityThreshold"
                      :disabled="isLoading"
                      id="similarity-slider"
                      class="mt-0 mb-1"
                      density="compact"
                      min="0"
                      max="1"
                      step="0.01"
                      color="primary"
                      track-color="grey-lighten-2"
                      thumb-label="hover"
                      hide-details
                      aria-label="Similarity threshold slider"
                      :aria-description="'Adjust minimum similarity threshold. Current value: ' + similarityThreshold.toFixed(2)"
                    ></v-slider>
                  </div>
                </template>
              </v-tooltip>
            </v-col>
          </v-row>

          <v-row dense>
            <v-col cols="12" md="6" class="pa-1">
              <v-tooltip location="bottom" :text="$t('queryInterface.tooltips.language')" role="tooltip">
                <template v-slot:activator="{ props }">
                  <v-select
                    v-bind="props"
                    v-model="selectedLanguage"
                    :items="availableLanguages"
                    item-title="text"
                    item-value="value"
                    :disabled="isLoading"
                    variant="outlined"
                    density="compact"
                    aria-label="Select query language"
                    :aria-description="'Choose the language for query processing. Currently selected: ' + selectedLanguage"
                    bg-color="white"
                    color="primary"
                    hide-details
                  >
                    <template v-slot:label>
                      <span class="text-caption">{{ $t('queryInterface.advancedOptions.language') }}</span>
                    </template>
                  </v-select>
                </template>
              </v-tooltip>
            </v-col>

            <v-col cols="12" md="6" class="pa-1 d-flex align-center">
              <v-tooltip location="bottom" :text="$t('queryInterface.tooltips.enableReranking')" role="tooltip">
                <template v-slot:activator="{ props }">
                  <v-switch
                    v-bind="props"
                    v-model="enableReranker"
                    :disabled="isLoading"
                    :label="$t('queryInterface.advancedOptions.enableReranking')"
                    color="primary"
                    inset
                    density="compact"
                    hide-details
                    class="mt-0 pt-0"
                    aria-label="Enable result re-ranking"
                  ></v-switch>
                </template>
              </v-tooltip>
            </v-col>
          </v-row>

          <v-row dense v-if="enableReranker">
            <v-col cols="12" md="6" class="pa-1">
              <v-tooltip location="bottom" :text="$t('queryInterface.tooltips.rerankerMode')" role="tooltip">
                <template v-slot:activator="{ props }">
                  <v-select
                    v-bind="props"
                    v-model="rerankerMode"
                    :items="rerankerModes"
                    item-title="text"
                    item-value="value"
                    :disabled="isLoading"
                    variant="outlined"
                    density="compact"
                    aria-label="Select reranker mode"
                    :aria-description="'Choose the reranker mode. Currently selected: ' + rerankerMode"
                    bg-color="white"
                    color="primary"
                    hide-details
                  >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.rerankerMode') }}</span>
                  </template>
                  </v-select>
                </template>
              </v-tooltip>
            </v-col>
          </v-row>

          <v-divider class="my-2"></v-divider>
          <div class="text-subtitle-2 mb-1 px-1 font-weight-medium">{{ $t('queryInterface.advancedOptions.processingModeTitle') }}</div>

          <v-row dense>
            <v-col cols="12" class="pa-1">
              <v-select
                v-model="forceEndpointMode"
                :items="[
                  { title: $t('queryInterface.advancedOptions.modeAutomatic'), value: null },
                  { title: $t('queryInterface.advancedOptions.modeQuery'), value: 'query' },
                  { title: $t('queryInterface.advancedOptions.modeTextProcess'), value: 'textProcess' }
                ]"
                item-title="title"
                item-value="value"
                variant="outlined"
                density="compact"
                :disabled="isLoading"
                bg-color="white"
                color="primary"
                hide-details
              >
                <template v-slot:label>
                  <span class="text-caption">{{ $t('queryInterface.advancedOptions.processingModeLabel') }}</span>
                </template>
              </v-select>
            </v-col>
          </v-row>

          <div v-if="isTextProcessModeActive">
            <v-divider class="my-2"></v-divider>
            <div class="text-subtitle-2 mb-1 px-1 font-weight-medium">{{ $t('queryInterface.advancedOptions.textProcessingTitle') }}</div>

            <v-row dense>
              <v-col cols="12" md="6" class="pa-1">
                <v-select
                  v-model="chunkingStrategy"
                  :items="['simple', 'semantic', 'detailed', 'sliding_window', 'sliding_window_cleaned']"
                  variant="outlined"
                  density="compact"
                  bg-color="white"
                  color="primary"
                  hide-details
                >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.chunkingStrategy') }}</span>
                  </template>
                </v-select>
              </v-col>
              <v-col cols="12" md="6" class="pa-1">
                <v-text-field
                  v-model.number="windowSize"
                  type="number"
                  min="1"
                  variant="outlined"
                  density="compact"
                  bg-color="white"
                  color="primary"
                  hide-details
                >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.windowSize') }}</span>
                  </template>
                </v-text-field>
              </v-col>
            </v-row>

            <v-row dense>
              <v-col cols="12" md="6" class="pa-1">
                <v-text-field
                  v-model.number="stepSize"
                  type="number"
                  min="1"
                  variant="outlined"
                  density="compact"
                  bg-color="white"
                  color="primary"
                  hide-details
                >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.stepSize') }}</span>
                  </template>
                </v-text-field>
              </v-col>
              <v-col cols="12" md="6" class="pa-1">
                <v-text-field
                  v-model.number="chunkRetrievalThreshold"
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  variant="outlined"
                  density="compact"
                  bg-color="white"
                  color="primary"
                  hide-details
                >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.chunkThreshold') }}</span>
                  </template>
                </v-text-field>
              </v-col>
            </v-row>

            <v-row dense>
              <v-col cols="12" md="6" class="pa-1">
                <v-text-field
                  v-model.number="aggregatedTermConfidence"
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  variant="outlined"
                  density="compact"
                  bg-color="white"
                  color="primary"
                  hide-details
                >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.aggConfidence') }}</span>
                  </template>
                </v-text-field>
              </v-col>
              <v-col cols="12" md="6" class="pa-1 d-flex align-center">
                <v-switch
                  v-model="noAssertionDetectionForTextProcess"
                  color="primary"
                  hide-details
                  density="compact"
                  inset
                  :true-value="false"
                  :false-value="true"
                >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.detectAssertions') }}</span>
                  </template>
                </v-switch>
              </v-col>
            </v-row>

            <v-row dense>
              <v-col cols="12" md="6" class="pa-1">
                <v-text-field
                  v-model.number="splitThreshold"
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  variant="outlined"
                  density="compact"
                  bg-color="white"
                  color="primary"
                  hide-details
                >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.splitThreshold') }}</span>
                  </template>
                </v-text-field>
              </v-col>
              <v-col cols="12" md="6" class="pa-1">
                <v-text-field
                  v-model.number="minSegmentLength"
                  type="number"
                  min="1"
                  variant="outlined"
                  density="compact"
                  bg-color="white"
                  color="primary"
                  hide-details
                >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.minSegmentLength') }}</span>
                  </template>
                </v-text-field>
              </v-col>
            </v-row>

            <v-row dense>
              <v-col cols="12" md="6" class="pa-1">
                <v-text-field
                  v-model.number="numResultsPerChunk"
                  type="number"
                  min="1"
                  variant="outlined"
                  density="compact"
                  bg-color="white"
                  color="primary"
                  hide-details
                >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.numResultsPerChunk') }}</span>
                  </template>
                </v-text-field>
              </v-col>
              <!-- Empty col for alignment if needed -->
              <v-col cols="12" md="6" class="pa-1"></v-col>
            </v-row>
          </div>
        </v-sheet>
      </v-expand-transition>
    </div>

    <!-- Chat-like conversation interface -->
    <div class="conversation-container" ref="conversationContainer">
      <div v-for="(item, index) in queryHistory" :key="index" class="mb-4">
        <!-- User query -->
        <div class="user-query d-flex">
          <v-tooltip location="top" text="User Input">
            <template v-slot:activator="{ props }">
              <v-avatar v-bind="props" color="primary" size="36" class="mt-1 mr-2">
                <span class="white--text">U</span>
              </v-avatar>
            </template>
          </v-tooltip>
          <div class="query-bubble">
            <p class="mb-0" style="white-space: pre-wrap;">{{ item.query }}</p>
          </div>
        </div>

        <!-- API response -->
        <div class="bot-response d-flex mt-2" v-if="item.loading || item.response || item.error">
          <v-tooltip location="top" text="Phentrieve Response">
            <template v-slot:activator="{ props }">
              <v-avatar v-bind="props" color="info" size="36" class="mt-1 mr-2">
                <v-icon color="white">mdi-robot-outline</v-icon>
              </v-avatar>
            </template>
          </v-tooltip>
          <div class="response-bubble">
            <v-progress-circular
              v-if="item.loading"
              indeterminate
              color="primary"
              size="24"
            ></v-progress-circular>

            <ResultsDisplay
              v-else
              :key="'results-' + index"
              :responseData="item.response"
              :resultType="item.type"
              :error="item.error"
              :collected-phenotypes="collectedPhenotypes"
              @add-to-collection="addToPhenotypeCollection"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- Floating action button for collection panel -->
    <v-tooltip location="left" :text="$t('queryInterface.tooltips.phenotypeCollection')" role="tooltip">
      <template v-slot:activator="{ props }">
        <v-btn
          v-bind="props"
          class="collection-fab collection-fab-position"
          color="secondary"
          icon
          position="fixed"
          location="bottom right"
          size="large"
          elevation="3"
          @click="toggleCollectionPanel"
          aria-label="Open HPO Collection Panel"
        >
          <v-badge
            :content="collectedPhenotypes.length"
            :model-value="collectedPhenotypes.length > 0"
            color="error" 
          >
            <v-icon>mdi-format-list-checks</v-icon>
          </v-badge>
        </v-btn>
      </template>
    </v-tooltip>

    <!-- Collection Panel -->
    <v-navigation-drawer
      v-model="showCollectionPanel"
      location="right"
      width="400"
      temporary
      style="z-index: 1500;" 
      aria-label="Phenotype collection"
    >
      <v-list-item class="pl-2 pr-1">
        <v-list-item-title class="text-h6">{{ $t('queryInterface.phenotypeCollection.title') }}</v-list-item-title>
        <template v-slot:append>
          <v-btn icon @click="toggleCollectionPanel" :aria-label="$t('queryInterface.phenotypeCollection.close')" variant="text" density="compact">
            <v-icon>mdi-close</v-icon>
          </v-btn>
        </template>
      </v-list-item>

      <v-divider></v-divider>

      <v-list v-if="collectedPhenotypes.length > 0" class="pt-0">
        <v-list-subheader>{{ $t('queryInterface.phenotypeCollection.count', { count: collectedPhenotypes.length }) }}</v-list-subheader>

        <v-list-item
          v-for="(phenotype, index) in collectedPhenotypes"
          :key="phenotype.hpo_id + '-' + index" 
          density="compact"
          class="py-1"
        >
          <v-list-item-title>
            <strong>{{ phenotype.hpo_id }}</strong>
            <v-chip
              size="x-small"
              class="ml-2"
              :color="phenotype.assertion_status === 'negated' ? 'pink-lighten-1' : 'green-lighten-1'"
              label
              variant="flat"
            >
              {{ $t(`queryInterface.phenotypeCollection.assertionStatus.${phenotype.assertion_status || 'affirmed'}`) }}
            </v-chip>
          </v-list-item-title>
          <v-list-item-subtitle class="wrap-text">
            {{ phenotype.label }}
          </v-list-item-subtitle>

          <template v-slot:append>
            <v-tooltip :text="$t('queryInterface.phenotypeCollection.assertionToggle')" location="start">
              <template v-slot:activator="{ props }">
                <v-btn
                  v-bind="props"
                  :icon="phenotype.assertion_status === 'negated' ? 'mdi-check-circle-outline' : 'mdi-close-circle-outline'"
                  variant="text"
                  density="compact"
                  :color="phenotype.assertion_status === 'negated' ? 'success' : 'error'"
                  class="mr-0"
                  @click="toggleAssertionStatus(index)"
                  :aria-label="`Toggle assertion status for ${phenotype.label} (${phenotype.hpo_id})`"
                ></v-btn>
              </template>
            </v-tooltip>
            <v-btn
              icon="mdi-delete-outline"
              variant="text"
              density="compact"
              color="grey-darken-1"
              @click="removePhenotype(index)"
              :aria-label="`Remove ${phenotype.label} (${phenotype.hpo_id}) from collection`"
            ></v-btn>
          </template>
        </v-list-item>
      </v-list>

      <v-sheet v-else class="pa-4 text-center">
        <v-icon size="x-large" color="grey-darken-1" class="mb-2">mdi-tray-plus</v-icon>
        <div class="text-body-1 text-grey-darken-2">{{ $t('queryInterface.phenotypeCollection.empty') }}</div>
        <div class="text-caption text-grey-darken-3 mt-2">
          {{ $t('queryInterface.phenotypeCollection.instructions') }} <v-icon size="small">mdi-plus-circle-outline</v-icon>
        </div>
      </v-sheet>

      <v-divider class="mt-4"></v-divider>
      <v-list-subheader>{{ $t('queryInterface.phenotypeCollection.subjectInfoHeader') }}</v-list-subheader>
      <div class="pa-3">
        <v-tooltip location="bottom" :text="$t('queryInterface.tooltips.subjectId')" role="tooltip">
          <template v-slot:activator="{ props }">
            <v-text-field
              v-bind="props"
              v-model="phenopacketSubjectId"
              density="compact"
              variant="outlined"
              hide-details="auto"
              class="mb-3"
              aria-label="Enter subject identifier for Phenopacket"
              bg-color="white"
              color="primary"
            >
              <template v-slot:label><span class="text-caption">{{ $t('queryInterface.phenotypeCollection.subjectId') }}</span></template>
            </v-text-field>
          </template>
        </v-tooltip>

        <v-tooltip location="bottom" :text="$t('queryInterface.tooltips.sex')" role="tooltip">
          <template v-slot:activator="{ props }">
            <v-select
              v-bind="props"
              v-model="phenopacketSex"
              :items="sexOptions"
              item-title="title"
              item-value="value"
              density="compact"
              variant="outlined"
              hide-details="auto"
              class="mb-3"
              clearable
              aria-label="Select subject sex for Phenopacket"
              bg-color="white"
              color="primary"
            >
              <template v-slot:label><span class="text-caption">{{ $t('queryInterface.phenotypeCollection.sex') }}</span></template>
            </v-select>
          </template>
        </v-tooltip>

        <v-tooltip location="bottom" :text="$t('queryInterface.tooltips.dateOfBirth')" role="tooltip">
          <template v-slot:activator="{ props }">
            <v-text-field
              v-bind="props"
              v-model="phenopacketDateOfBirth"
              placeholder="YYYY-MM-DD"
              density="compact"
              variant="outlined"
              hide-details="auto"
              class="mb-3"
              clearable
              type="date"
              aria-label="Enter subject date of birth for Phenopacket"
              bg-color="white"
              color="primary"
            >
              <template v-slot:label><span class="text-caption">{{ $t('queryInterface.phenotypeCollection.dateOfBirth') }}</span></template>
            </v-text-field>
          </template>
        </v-tooltip>
      </div>

      <template v-slot:append>
        <v-divider></v-divider>
        <div class="pa-3">
          <v-btn
            block
            color="primary"
            class="mb-2"
            prepend-icon="mdi-download-box-outline"
            @click="exportPhenotypesAsPhenopacket"
            :disabled="collectedPhenotypes.length === 0"
            aria-label="Export collected phenotypes as Phenopacket JSON"
          >
            {{ $t('queryInterface.phenotypeCollection.exportPhenopacket') }}
          </v-btn>
          <v-btn
            block
            variant="outlined"
            color="primary"
            class="mb-2"
            prepend-icon="mdi-text-box-outline"
            @click="exportPhenotypes"
            :disabled="collectedPhenotypes.length === 0"
          >
            {{ $t('queryInterface.phenotypeCollection.exportText') }}
          </v-btn>
          <v-btn
            block
            variant="tonal"
            color="error"
            prepend-icon="mdi-delete-sweep-outline"
            @click="clearPhenotypeCollection"
            :disabled="collectedPhenotypes.length === 0"
          >
            {{ $t('queryInterface.phenotypeCollection.clear') }}
          </v-btn>
        </div>
      </template>
    </v-navigation-drawer>
  </div>
</template>

<script>
import ResultsDisplay from './ResultsDisplay.vue';
import PhentrieveService from '../services/PhentrieveService';
import { logService } from '../services/logService';
// Direct JSON-based implementation instead of using @berntpopp/phenopackets-js

export default {
  name: 'QueryInterface',
  components: {
    ResultsDisplay
  },
  computed: {
    // Determine if text processing mode is active based on text length or user force setting
    isTextProcessModeActive() {
      if (this.forceEndpointMode) {
        return this.forceEndpointMode === 'textProcess';
      }
      return this.queryText.length > this.inputTextLengthThreshold;
    }
  },
  data() {
    return {
      queryText: '',
      selectedModel: null, // Will be set in mounted
      availableModels: [ // Example, should be fetched from API ideally
        { text: this.$t('models.biolord'), value: 'FremyCompany/BioLORD-2023-M' },
        { text: this.$t('models.jinaGerman'), value: 'jinaai/jina-embeddings-v2-base-de' },
        { text: this.$t('models.distiluse'), value: 'sentence-transformers/distiluse-base-multilingual-cased-v2' } // Corrected model name
      ],
      selectedLanguage: null,
      availableLanguages: [
        { text: this.$t('common.auto'), value: null },
        { text: 'English', value: 'en' },
        { text: 'German (Deutsch)', value: 'de' },
        { text: 'French (Français)', value: 'fr' },
        { text: 'Spanish (Español)', value: 'es' },
        { text: 'Dutch (Nederlands)', value: 'nl' }
      ],
      similarityThreshold: 0.3,
      numResults: 10,
      enableReranker: false,
      rerankerMode: 'cross-lingual', // Default for query mode
      rerankerModes: [ // These should match API schema literals
        {text: this.$t('queryInterface.advancedOptions.rerankerModeCrossLingual'), value: 'cross-lingual'},
        {text: this.$t('queryInterface.advancedOptions.rerankerModeMonolingual'), value: 'monolingual'}
      ],
      isLoading: false,
      queryHistory: [],
      showAdvancedOptions: false,
      collectedPhenotypes: [],
      showCollectionPanel: false,
      lastUserScrollPosition: 0,
      userHasScrolled: false,
      shouldScrollToTop: false,
      phenopacketSubjectId: '',
      phenopacketSex: null,
      phenopacketDateOfBirth: null,
      sexOptions: [
        { title: this.$t('phenopacket.sexUnknown'), value: 0 },
        { title: this.$t('phenopacket.sexFemale'), value: 1 },
        { title: this.$t('phenopacket.sexMale'), value: 2 },
        { title: this.$t('phenopacket.sexOther'), value: 3 }
      ],
      inputTextLengthThreshold: 120, // Increased threshold
      forceEndpointMode: null,
      chunkingStrategy: 'sliding_window_cleaned',
      windowSize: 7, // Default from config
      stepSize: 1, // Default from config
      splitThreshold: 0.5, // Default from config
      minSegmentLength: 3, // Default from config
      semanticModelForChunking: null,
      retrievalModelForTextProcess: null,
      chunkRetrievalThreshold: 0.3,
      numResultsPerChunk: 10,
      noAssertionDetectionForTextProcess: false,
      assertionPreferenceForTextProcess: 'dependency',
      aggregatedTermConfidence: 0.35,
      topTermPerChunkForAggregation: false
    };
  },
  watch: {
    queryHistory: {
      handler() {
        logService.info('Query history updated', { newHistoryLength: this.queryHistory.length });
        this.$nextTick(() => {
          if (this.shouldScrollToTop && !this.userHasScrolled) {
             this.scrollToTop();
          }
        });
      },
      deep: true
    },
    selectedModel(newModel, oldModel) {
      if (oldModel !== null) { // Avoid resetting on initial mount
        logService.info('Model changed', { newModel: newModel, oldModel: oldModel });
        // Reset some query-specific settings to defaults when model changes
        this.similarityThreshold = 0.3;
        this.enableReranker = false;
        this.rerankerMode = 'cross-lingual';
        logService.info('Reset query-specific settings to defaults due to model change.');
      }
    }
  },
  methods: {
    // ... (applyUrlParametersAndAutoSubmit, handleUserScroll, addToPhenotypeCollection, etc. remain largely the same) ...
    // Ensure i18n is used for labels in methods where static text was present
    // Add padding and scrollbar logic here if applicable within this component's scope.
    // The methods for data handling (submitQuery, export, etc.) are complex and largely functional,
    // so changes there will be minimal, focusing on passing the correct state.
    applyUrlParametersAndAutoSubmit() {
      const queryParams = this.$route.query;
      logService.debug('Raw URL query parameters:', { ...queryParams });
      let advancedOptionsWereSet = false;
      let performAutoSubmit = false;
      const parseBooleanParam = (val) => typeof val === 'string' && (val.toLowerCase() === 'true' || val === '1');

      if (queryParams.text !== undefined) { this.queryText = queryParams.text; }
      if (queryParams.model !== undefined) {
        const validModels = this.availableModels.map(m => m.value);
        if (validModels.includes(queryParams.model)) { this.selectedModel = queryParams.model; advancedOptionsWereSet = true; }
      }
      if (queryParams.threshold !== undefined) {
        const val = parseFloat(queryParams.threshold);
        if (!isNaN(val) && val >= 0 && val <= 1) { this.similarityThreshold = val; advancedOptionsWereSet = true; }
      }
      if (queryParams.reranker !== undefined) { this.enableReranker = parseBooleanParam(queryParams.reranker); advancedOptionsWereSet = true; }
      if (queryParams.rerankerMode !== undefined && this.enableReranker) {
        const validModes = this.rerankerModes.map(m => m.value);
        if (validModes.includes(queryParams.rerankerMode)) { this.rerankerMode = queryParams.rerankerMode; advancedOptionsWereSet = true;}
      }
      // Add processing for text process specific URL params
      if (queryParams.forceEndpointMode) { this.forceEndpointMode = queryParams.forceEndpointMode; advancedOptionsWereSet = true;}
      if (queryParams.chunkingStrategy) { this.chunkingStrategy = queryParams.chunkingStrategy; advancedOptionsWereSet = true;}


      if (advancedOptionsWereSet) this.showAdvancedOptions = true;

      performAutoSubmit = queryParams.autoSubmit !== undefined ? parseBooleanParam(queryParams.autoSubmit) : (queryParams.text !== undefined);

      if (performAutoSubmit && this.queryText && this.queryText.trim()) {
        logService.info('Auto-submitting query based on URL parameters.');
        this.$nextTick(() => setTimeout(() => this.submitQuery(true), 300)); // autoSubmit = true
      }
    },
    handleUserScroll() {
      const container = this.$refs.conversationContainer;
      if (container) {
        if (Math.abs(container.scrollTop - this.lastUserScrollPosition) > 5) { // Threshold to detect actual scroll
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
    addToPhenotypeCollection(phenotype, queryAssertionStatus = null) {
      logService.info('Adding phenotype to collection', { phenotype, queryAssertionStatus });
      const isDuplicate = this.collectedPhenotypes.some(item => item.hpo_id === phenotype.hpo_id);
      if (!isDuplicate) {
        const currentResponse = this.queryHistory.length > 0 ? this.queryHistory[0].response : null;
        const responseAssertionStatus = currentResponse?.query_assertion_status || null;
        const assertionStatus = queryAssertionStatus || responseAssertionStatus || 'affirmed';
        this.collectedPhenotypes.push({ ...phenotype, added_at: new Date(), assertion_status: assertionStatus });
        if (this.collectedPhenotypes.length === 1) this.showCollectionPanel = true;
      }
    },
    removePhenotype(index) {
      logService.info('Removing phenotype from collection', { index, phenotype: this.collectedPhenotypes[index] });
      this.collectedPhenotypes.splice(index, 1);
    },
    toggleAssertionStatus(index) {
      const phenotype = this.collectedPhenotypes[index];
      if (phenotype) {
        const newStatus = phenotype.assertion_status === 'negated' ? 'affirmed' : 'negated';
        phenotype.assertion_status = newStatus; // Directly mutate for reactivity
        logService.info('Toggled phenotype assertion status', { hpo_id: phenotype.hpo_id, newStatus });
      }
    },
    clearPhenotypeCollection() {
      logService.info('Clearing phenotype collection and subject information');
      this.collectedPhenotypes = [];
      this.phenopacketSubjectId = '';
      this.phenopacketSex = null;
      this.phenopacketDateOfBirth = null;
    },
    toggleCollectionPanel() {
      this.showCollectionPanel = !this.showCollectionPanel;
    },
    exportPhenotypes() {
      logService.info('Exporting phenotypes as text', { count: this.collectedPhenotypes.length });
      let exportText = "HPO Phenotypes Collection\n";
      exportText += "Exported on: " + new Date().toLocaleString() + "\n\n";
      exportText += "ID\tLabel\tAssertion Status\n";
      this.collectedPhenotypes.forEach(p => {
        exportText += `${p.hpo_id}\t${p.label}\t${p.assertion_status || 'affirmed'}\n`;
      });
      const blob = new Blob([exportText], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'hpo_phenotypes.txt';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    },
    exportPhenotypesAsPhenopacket() {
      if (this.collectedPhenotypes.length === 0) {
        logService.warn('Attempted to export empty phenopacket collection');
        return;
      }
      logService.info('Starting Phenopacket export process', { count: this.collectedPhenotypes.length });
      try {
        const timestamp = new Date().toISOString();
        const phenopacketId = `phentrieve-export-${Date.now()}`;
        const phenopacket = {
          id: phenopacketId,
          metaData: {
            created: timestamp, createdBy: "Phentrieve Frontend", phenopacketSchemaVersion: "2.0.0",
            resources: [{ id: "phentrieve", name: "Phentrieve", namespacePrefix: "Phentrieve", url: "https://phentrieve.kidney-genetics.org/", version: import.meta.env.VITE_APP_VERSION || "1.0.0", iriPrefix: "phentrieve" }]
          },
          phenotypicFeatures: []
        };
        if (this.phenopacketSubjectId || this.phenopacketSex !== null || this.phenopacketDateOfBirth) {
          phenopacket.subject = {};
          if (this.phenopacketSubjectId) phenopacket.subject.id = this.phenopacketSubjectId.trim();
          if (this.phenopacketSex !== null) {
            const sexMap = { 0: "UNKNOWN_SEX", 1: "FEMALE", 2: "MALE", 3: "OTHER_SEX" };
            phenopacket.subject.sex = sexMap[this.phenopacketSex];
          }
          if (this.phenopacketDateOfBirth) {
            const dob = new Date(this.phenopacketDateOfBirth + "T00:00:00Z");
            if (!isNaN(dob.getTime())) phenopacket.subject.timeAtLastEncounter = { timestamp: dob.toISOString() };
          }
          if (Object.keys(phenopacket.subject).length === 0) delete phenopacket.subject;
        }
        this.collectedPhenotypes.forEach(cp => {
          phenopacket.phenotypicFeatures.push({ type: { id: cp.hpo_id, label: cp.label }, excluded: cp.assertion_status === 'negated' });
        });
        const phenopacketJsonString = JSON.stringify(phenopacket, null, 2);
        const blob = new Blob([phenopacketJsonString], { type: 'application/json;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.download = `phentrieve_phenopacket_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
        a.href = url;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        logService.info('Phenopacket successfully exported', { filename: a.download });
      } catch (error) {
        logService.error('Error during Phenopacket export', { error });
        alert('Error exporting Phenopacket. Check console.');
      }
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
      const historyItem = { query: currentQuery, loading: true, response: null, error: null, type: useTextProcessMode ? 'textProcess' : 'query' };
      this.queryHistory.unshift(historyItem);
      if (!isAutoSubmit) this.queryText = ''; // Clear input only if not auto-submitting (URL params might be active)
      
      this.shouldScrollToTop = true;
      this.userHasScrolled = false; 

      try {
        let response;
        if (useTextProcessMode) {
          const textProcessData = {
            text_content: currentQuery, language: this.selectedLanguage,
            chunking_strategy: this.chunkingStrategy, window_size: this.windowSize,
            step_size: this.stepSize, split_threshold: this.splitThreshold,
            min_segment_length: this.minSegmentLength,
            semantic_model_name: this.semanticModelForChunking || this.selectedModel,
            retrieval_model_name: this.retrievalModelForTextProcess || this.selectedModel,
            trust_remote_code: true, 
            chunk_retrieval_threshold: this.chunkRetrievalThreshold,
            num_results_per_chunk: this.numResultsPerChunk,
            enable_reranker: this.enableReranker, reranker_mode: this.rerankerMode,
            no_assertion_detection: this.noAssertionDetectionForTextProcess,
            assertion_preference: this.assertionPreferenceForTextProcess,
            aggregated_term_confidence: this.aggregatedTermConfidence,
            top_term_per_chunk_for_aggregation: this.topTermPerChunkForAggregation,
          };
          logService.info('Sending to /text/process API', textProcessData);
          response = await PhentrieveService.processText(textProcessData);
        } else {
          const queryData = {
            text: currentQuery, model_name: this.selectedModel, language: this.selectedLanguage,
            num_results: this.numResults, similarity_threshold: this.similarityThreshold,
            enable_reranker: this.enableReranker, reranker_mode: this.rerankerMode,
            query_assertion_language: this.selectedLanguage, // Pass selected language for query assertion
            detect_query_assertion: true // Default to true for query mode now
          };
          logService.info('Sending to /query API', queryData);
          response = await PhentrieveService.queryHpo(queryData);
        }
        historyItem.response = response;
      } catch (error) {
        historyItem.error = error;
        logService.error('Error submitting query/processing text', error);
      } finally {
        historyItem.loading = false;
        this.isLoading = false;
        this.queryHistory = [...this.queryHistory]; // Trigger reactivity
         if (!isAutoSubmit) { // Only clear URL params if it wasn't an auto-submit
            const newRouteQuery = { ...this.$route.query };
            delete newRouteQuery.autoSubmit; // Remove autoSubmit flag
            if (Object.keys(newRouteQuery).length !== Object.keys(this.$route.query).length -1) { // Check if other params were there
              this.$router.replace({ query: newRouteQuery }).catch(err => {
                if (err.name !== 'NavigationDuplicated' && err.name !== 'NavigationCancelled') {
                  logService.warn('Error clearing autoSubmit from URL:', err);
                }
              });
            }
        }
      }
    }
  },
  created() {
    // Example: Populate availableModels from an API endpoint if needed
    // PhentrieveService.getConfigInfo().then(config => {
    //   this.availableModels = config.available_embedding_models.map(m => ({ text: `${m.description} (${m.id.split('/').pop()})`, value: m.id }));
    //   if (!this.selectedModel && config.default_embedding_model) {
    //     this.selectedModel = config.default_embedding_model;
    //   }
    // }).catch(error => {
    //   logService.error('Failed to fetch config info for models:', error);
    // });
    this.selectedModel = this.availableModels[0].value; // Set a default if not fetched
  }
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
  bottom: 16px !important; /* Adjusted for potential scrollbar */
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
  box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.16); /* Softer shadow */
}
.search-input.v-textarea :deep(.v-field) {
  border-radius: 18px; /* Consistent rounding for textarea */
  padding-top: 8px;
  padding-bottom: 8px;
}

.search-input :deep(.v-field__input) {
  min-height: 48px;
  padding-top: 0;
  padding-bottom: 0;
  align-items: center; /* Vertically align text */
}
.search-input.v-textarea :deep(.v-field__input) {
  min-height: 80px; /* Keep textarea height if needed */
}

.search-bar {
  /* Removed border from search-bar, it's on v-field now */
  border-radius: 30px; /* Match outer radius if needed */
}

.search-input :deep(.v-field__outline) {
  --v-field-border-width: 0px; /* No internal outline needed */
}


/* Advanced options panel specific styling */
#advanced-options-panel .text-subtitle-2 {
  font-size: 0.875rem !important;
  color: rgba(0,0,0,0.7);
}
#advanced-options-panel .text-caption {
  font-size: 0.75rem !important;
  color: rgba(0,0,0,0.6);
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
.conversation-container::-webkit-scrollbar { width: 8px; height: 8px; }
.conversation-container::-webkit-scrollbar-thumb { background: #ccc; border-radius: 4px; }
.conversation-container::-webkit-scrollbar-thumb:hover { background: #bbb; }
.conversation-container::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 4px; }

.query-bubble {
  background-color: #e3f2fd; /* Lighter blue for user */
  color: #1e3a8a; /* Darker blue text for contrast */
  border-radius: 18px 18px 18px 4px; /* Chat bubble style */
  padding: 10px 15px;
  max-width: 80%;
  box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.response-bubble {
  background-color: #f8f9fa; /* Very light grey for bot */
  border: 1px solid #e9ecef;
  border-radius: 18px 18px 4px 18px; /* Chat bubble style */
  padding: 10px 15px;
  width: 100%; /* Bot bubble can take more width */
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.user-query { justify-content: flex-end; margin-left: auto; } /* Align user to right */
.user-query .v-avatar { order: 1; margin-left: 8px; margin-right: 0; } /* Avatar on right */
.user-query .query-bubble { order: 0; }


.bot-response { justify-content: flex-start; }
.bot-response .v-avatar { margin-right: 8px; }

/* Accessibility & General UI Polish */
:deep(.text-high-emphasis) { color: rgba(0,0,0,0.87) !important; font-weight: 500; }
:deep(.v-field--variant-outlined) { background-color: #FFFFFF !important; box-shadow: none; }
:deep(.v-field__input) { color: rgba(0,0,0,0.87) !important; }
:deep(.v-field__input::placeholder) { color: rgba(0,0,0,0.6) !important; }

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