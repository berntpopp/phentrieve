<template>
  <div class="search-container mx-auto px-2">
    <!-- Clean Search Bar with Integrated Button -->
    <div class="search-bar-container pt-0 px-2 pb-2 pa-sm-3">
      <v-sheet rounded="pill" elevation="0" class="pa-1 pa-sm-2 search-bar" color="white">
        <div class="d-flex align-center flex-wrap flex-sm-nowrap">
          <v-textarea
            v-if="forceEndpointMode === 'textProcess' || queryText.length > inputTextLengthThreshold"
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
                  :aria-controls="'advanced-options-panel'"
                  size="small"
                >
                  <v-icon>{{ showAdvancedOptions ? 'mdi-cog' : 'mdi-tune' }}</v-icon>
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
          class="mt-1 py-1 px-2"
          id="advanced-options-panel"
          role="region"
          aria-label="Advanced search options"
          color="white"
          style="font-size: 0.8rem;"
        >
          <div class="text-caption mb-0 px-1 font-weight-medium">{{ $t('queryInterface.advancedOptions.title') }}</div>
          
          <v-row dense>
            <v-col cols="12" md="6" class="py-0">
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
                    size="x-small"
                    aria-label="Select embedding model"
                    :aria-description="'Choose the model to use for text embedding. Currently selected: ' + selectedModel"
                    bg-color="white"
                    color="primary"
                    class="my-0"
                    hide-details
                  >

                <template v-slot:label>
                  <span class="text-caption">{{ $t('queryInterface.advancedOptions.embeddingModel') }}</span>
                </template>
                  </v-select>
                </template>
              </v-tooltip>
            </v-col>
            
            <v-col cols="12" md="6" class="py-0">
              <v-tooltip location="bottom" :text="$t('queryInterface.tooltips.similarityThreshold')" role="tooltip">
                <template v-slot:activator="{ props }">
                  <div>
                    <label :for="'similarity-slider'" class="text-caption mb-0 d-block" style="font-size: 0.7rem;">{{ $t('queryInterface.advancedOptions.similarityThreshold') }}: {{ similarityThreshold.toFixed(2) }}</label>
                    <v-slider
                      v-bind="props"
                      v-model="similarityThreshold"
                      :disabled="isLoading"
                      id="similarity-slider"
                      class="py-0 mt-0 mb-1"
                      density="compact"
                      min="0"
                      max="1"
                      step="0.01"
                      color="primary"
                      track-color="grey-lighten-2"
                      thumb-label="hover"
                      :tick-size="0"
                      hide-details
                      aria-label="Similarity threshold slider"
                      :aria-description="'Adjust minimum similarity threshold. Current value: ' + similarityThreshold.toFixed(2)"
                    ></v-slider>
                  </div>
                </template>
              </v-tooltip>
            </v-col>
          </v-row>
          
          <v-row>
            <v-col cols="12" md="6">
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
                  >
                    <template v-slot:label>
                      <span class="text-high-emphasis">{{ $t('queryInterface.advancedOptions.language') }}</span>
                    </template>
                  </v-select>
                </template>
              </v-tooltip>
            </v-col>
            
            <v-col cols="12" md="6">
              <v-tooltip location="bottom" :text="$t('queryInterface.tooltips.enableReranking')" role="tooltip">
                <template v-slot:activator="{ props }">
                  <v-switch
                    v-bind="props"
                    v-model="enableReranker"
                    :disabled="isLoading"
                    :label="$t('queryInterface.advancedOptions.enableReranking')"
                    color="primary"
                    inset
                    aria-label="Enable result re-ranking"
                  ></v-switch>
                </template>
              </v-tooltip>
            </v-col>
          </v-row>

          <v-row>
            <v-col cols="12" md="6" v-if="enableReranker">
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
                  >
                  <template v-slot:label>
                  <span class="text-high-emphasis">{{ $t('queryInterface.advancedOptions.rerankerMode') }}</span>
                </template>
                  </v-select>
                </template>
              </v-tooltip>
            </v-col>
            
            <v-col cols="12" md="6" v-if="enableReranker">
              <!-- Placeholder for balance -->
            </v-col>
          </v-row>
          
          <!-- Processing Mode Selector -->
          <v-divider class="my-0"></v-divider>
          <div class="text-caption mb-0 px-1 font-weight-medium">{{ $t('queryInterface.advancedOptions.processingModeTitle') }}</div>
          
          <v-row dense>
            <v-col cols="12" class="py-0">
              <v-select
                v-model="forceEndpointMode"
                :items="[
                  { title: $t('queryInterface.advancedOptions.modeAutomatic'), value: null },
                  { title: $t('queryInterface.advancedOptions.modeQuery'), value: 'query' },
                  { title: $t('queryInterface.advancedOptions.modeTextProcess'), value: 'textProcess' }
                ]"
                item-title="title"
                item-value="value"
                :label="$t('queryInterface.advancedOptions.processingModeLabel')"
                variant="outlined"
                density="compact"
                size="x-small"
                :disabled="isLoading"
                bg-color="white"
                color="primary"
                class="my-0"
                hide-details
              >
                <template v-slot:label>
                  <span class="text-caption">{{ $t('queryInterface.advancedOptions.processingModeLabel') }}</span>
                </template>
              </v-select>
            </v-col>
          </v-row>
          
          <!-- Text Processing Specific Options (Only show when in text processing mode) -->
          <div v-if="forceEndpointMode === 'textProcess' || queryText.length > inputTextLengthThreshold">
            <v-divider class="my-0"></v-divider>
            <div class="text-caption mb-0 px-1 font-weight-medium">{{ $t('queryInterface.advancedOptions.textProcessingTitle') }}</div>
            
            <v-row dense>
              <v-col cols="12" md="6" class="py-0">
                <v-select 
                  v-model="chunkingStrategy" 
                  :items="[
                    'simple',
                    'semantic',
                    'detailed',
                    'sliding_window',
                    'sliding_window_cleaned'
                  ]" 
                  :label="$t('queryInterface.advancedOptions.chunkingStrategy')" 
                  variant="outlined" 
                  density="compact"
                  size="x-small"
                  bg-color="white" 
                  color="primary"
                  class="my-0"
                  hide-details
                >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.chunkingStrategy') }}</span>
                  </template>
                </v-select>
              </v-col>
              <v-col cols="12" md="6" class="py-0">
                <v-text-field 
                  v-model.number="windowSize" 
                  :label="$t('queryInterface.advancedOptions.windowSize')" 
                  type="number" 
                  variant="outlined" 
                  density="compact"
                  size="x-small"
                  bg-color="white" 
                  color="primary"
                  class="my-0"
                  hide-details
                >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.windowSize') }}</span>
                  </template>
                </v-text-field>
              </v-col>
            </v-row>
            
            <v-row dense>
              <v-col cols="12" md="6" class="py-0">
                <v-text-field 
                  v-model.number="stepSize" 
                  :label="$t('queryInterface.advancedOptions.stepSize')" 
                  type="number" 
                  variant="outlined" 
                  density="compact"
                  size="x-small"
                  bg-color="white" 
                  color="primary"
                  class="my-0"
                  hide-details
                >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.stepSize') }}</span>
                  </template>
                </v-text-field>
              </v-col>
              <v-col cols="12" md="6" class="py-0">
                <v-text-field 
                  v-model.number="chunkRetrievalThreshold" 
                  :label="$t('queryInterface.advancedOptions.chunkThreshold')" 
                  type="number" 
                  step="0.01" 
                  min="0" 
                  max="1" 
                  variant="outlined" 
                  density="compact"
                  size="x-small"
                  bg-color="white" 
                  color="primary"
                  class="my-0"
                  hide-details
                >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.chunkThreshold') }}</span>
                  </template>
                </v-text-field>
              </v-col>
            </v-row>
            
            <v-row dense>
              <v-col cols="12" md="6" class="py-0">
                <v-text-field 
                  v-model.number="aggregatedTermConfidence" 
                  :label="$t('queryInterface.advancedOptions.aggConfidence')" 
                  type="number" 
                  step="0.01" 
                  min="0" 
                  max="1" 
                  variant="outlined" 
                  density="compact"
                  size="x-small"
                  bg-color="white" 
                  color="primary"
                  class="my-0"
                  hide-details
                >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.aggConfidence') }}</span>
                  </template>
                </v-text-field>
              </v-col>
              <v-col cols="12" md="6" class="d-flex align-center py-0">
                <v-switch
                  v-model="noAssertionDetectionForTextProcess" 
                  :label="$t('queryInterface.advancedOptions.detectAssertions')" 
                  color="primary" 
                  hide-details
                  density="compact"
                  class="my-0"
                  :true-value="false"
                  :false-value="true"
                  size="x-small"
                >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.detectAssertions') }}</span>
                  </template>
                </v-switch>
              </v-col>
            </v-row>
            
            <!-- Additional text processing settings -->
            <v-row dense>
              <v-col cols="12" md="6" class="py-0">
                <v-text-field 
                  v-model.number="splitThreshold" 
                  type="number" 
                  step="0.01" 
                  min="0" 
                  max="1" 
                  variant="outlined" 
                  density="compact"
                  size="x-small"
                  bg-color="white" 
                  color="primary"
                  class="my-0"
                  hide-details
                >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.splitThreshold') }}</span>
                  </template>
                </v-text-field>
              </v-col>
              <v-col cols="12" md="6" class="py-0">
                <v-text-field 
                  v-model.number="minSegmentLength" 
                  type="number" 
                  min="1"
                  variant="outlined" 
                  density="compact"
                  size="x-small"
                  bg-color="white" 
                  color="primary"
                  class="my-0"
                  hide-details
                >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.minSegmentLength') }}</span>
                  </template>
                </v-text-field>
              </v-col>
            </v-row>
            
            <v-row dense>
              <v-col cols="12" md="6" class="py-0">
                <v-text-field 
                  v-model.number="numResultsPerChunk" 
                  type="number" 
                  min="1"
                  variant="outlined" 
                  density="compact"
                  size="x-small"
                  bg-color="white" 
                  color="primary"
                  class="my-0"
                  hide-details
                >
                  <template v-slot:label>
                    <span class="text-caption">{{ $t('queryInterface.advancedOptions.numResultsPerChunk') }}</span>
                  </template>
                </v-text-field>
              </v-col>
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
            <p class="mb-0">{{ item.query }}</p>
          </div>
        </div>
        
        <!-- API response -->
        <div class="bot-response d-flex mt-2" v-if="item.loading || item.response || item.error">
          <v-tooltip location="top" text="Phentrieve Response">
            <template v-slot:activator="{ props }">
              <v-avatar v-bind="props" color="info" size="36" class="mt-1 mr-2">
                <span class="white--text">P</span> <!-- P for Phentrieve -->
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
            color="primary"
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
      aria-label="Phenotype collection"
    >
      <v-list-item>
        <v-list-item-title class="text-h6">{{ $t('queryInterface.phenotypeCollection.title') }}</v-list-item-title>
        <template v-slot:append>
          <v-btn icon @click="toggleCollectionPanel" :aria-label="$t('queryInterface.phenotypeCollection.close')">
            <v-icon>mdi-close</v-icon>
          </v-btn>
        </template>
      </v-list-item>
      
      <v-divider></v-divider>
      
      <v-list v-if="collectedPhenotypes.length > 0">
        <v-list-subheader>{{ $t('queryInterface.phenotypeCollection.count', { count: collectedPhenotypes.length }) }}</v-list-subheader>
        
        <v-list-item
          v-for="(phenotype, index) in collectedPhenotypes"
          :key="phenotype.hpo_id"
          density="compact"
        >
          <v-list-item-title>
            <strong>{{ phenotype.hpo_id }}</strong>
            <v-chip
              size="x-small"
              class="ml-2"
              :color="phenotype.assertion_status === 'negated' ? 'error' : 'success'"
              text-color="white"
              label
            >
              {{ $t(`queryInterface.phenotypeCollection.${phenotype.assertion_status}`) }}
            </v-chip>
          </v-list-item-title>
          <v-list-item-subtitle>
            {{ phenotype.label }}
          </v-list-item-subtitle>
          
          <template v-slot:append>
            <v-tooltip :text="$t('queryInterface.phenotypeCollection.assertionToggle')" location="start">
              <template v-slot:activator="{ props }">
                <v-btn
                  v-bind="props"
                  :icon="phenotype.assertion_status === 'negated' ? 'mdi-check' : 'mdi-close-circle-outline'"
                  variant="text"
                  density="compact"
                  :color="phenotype.assertion_status === 'negated' ? 'success' : 'error'"
                  class="mr-1"
                  @click="toggleAssertionStatus(index)"
                  :aria-label="`Toggle assertion status for ${phenotype.label} (${phenotype.hpo_id})`"
                ></v-btn>
              </template>
            </v-tooltip>
            <v-btn
              icon="mdi-delete"
              variant="text"
              density="compact"
              color="error"
              @click="removePhenotype(index)"
              :aria-label="`Remove ${phenotype.label} (${phenotype.hpo_id}) from collection`"
            ></v-btn>
          </template>
        </v-list-item>
      </v-list>
      
      <v-sheet v-else class="pa-4 text-center">
        <v-icon size="large" color="grey-darken-2" class="mb-2">mdi-tray-plus</v-icon>
        <div class="text-body-1 text-grey-darken-3">{{ $t('queryInterface.phenotypeCollection.empty') }}</div>
        <div class="text-body-2 text-grey-darken-3 mt-2">
          {{ $t('queryInterface.phenotypeCollection.instructions') }} <v-icon size="small">mdi-plus-circle</v-icon>
        </div>
      </v-sheet>
      
      <v-divider class="mt-4"></v-divider>
      <v-list-subheader>{{ $t('queryInterface.phenotypeCollection.subjectInfoHeader') }}</v-list-subheader>
      <div class="pa-4">
        <v-tooltip location="bottom" :text="$t('queryInterface.tooltips.subjectId')" role="tooltip">
          <template v-slot:activator="{ props }">
            <v-text-field
              v-bind="props"
              v-model="phenopacketSubjectId"
              label="Subject ID"
              density="compact"
              variant="outlined"
              hide-details="auto"
              class="mb-3"
              aria-label="Enter subject identifier for Phenopacket"
              bg-color="surface"
              color="primary"
            >
              <template v-slot:label><span class="text-high-emphasis">{{ $t('queryInterface.phenotypeCollection.subjectId') }}</span></template>
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
              label="Sex"
              density="compact"
              variant="outlined"
              hide-details="auto"
              class="mb-3"
              clearable
              aria-label="Select subject sex for Phenopacket"
              bg-color="surface"
              color="primary"
            >
              <template v-slot:label><span class="text-high-emphasis">{{ $t('queryInterface.phenotypeCollection.sex') }}</span></template>
            </v-select>
          </template>
        </v-tooltip>

        <v-tooltip location="bottom" :text="$t('queryInterface.tooltips.dateOfBirth')" role="tooltip">
          <template v-slot:activator="{ props }">
            <v-text-field
              v-bind="props"
              v-model="phenopacketDateOfBirth"
              label="Date of Birth (YYYY-MM-DD)"
              placeholder="YYYY-MM-DD"
              density="compact"
              variant="outlined"
              hide-details="auto"
              class="mb-3"
              clearable
              type="date" 
              aria-label="Enter subject date of birth for Phenopacket"
              bg-color="surface"
              color="primary"
            >
              <template v-slot:label><span class="text-high-emphasis">{{ $t('queryInterface.phenotypeCollection.dateOfBirth') }}</span></template>
            </v-text-field>
          </template>
        </v-tooltip>
      </div>
      
      <template v-slot:append>
        <v-divider></v-divider>
        <div class="pa-2">
          <v-btn
            block
            color="primary"
            class="mb-2"
            prepend-icon="mdi-download"
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
            prepend-icon="mdi-download"
            @click="exportPhenotypes"
            :disabled="collectedPhenotypes.length === 0"
          >
            {{ $t('queryInterface.phenotypeCollection.exportText') }}
          </v-btn>
          <v-btn
            block
            variant="tonal"
            color="error"
            prepend-icon="mdi-delete"
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
      selectedModel: null,
      availableModels: [
        { text: 'BioLORD-2023-M (Medical)', value: 'FremyCompany/BioLORD-2023-M' },
        { text: 'Jina Embed v2 (German)', value: 'jinaai/jina-embeddings-v2-base-de' },
        { text: 'DistilUSE (Multilingual)', value: 'distiluse-base-multilingual-cased-v1' }
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
      rerankerMode: 'cross-lingual',
      rerankerModes: ['cross-lingual', 'monolingual'],
      isLoading: false,
      queryHistory: [],
      showAdvancedOptions: false,
      collectedPhenotypes: [],
      showCollectionPanel: false,
      lastUserScrollPosition: 0,
      userHasScrolled: false,
      shouldScrollToTop: false,
      // Subject information for Phenopacket export
      phenopacketSubjectId: '',
      phenopacketSex: null,
      phenopacketDateOfBirth: null,
      // Sex options based on phenopackets-js enum values
      sexOptions: [
        { title: 'Unknown', value: 0 }, // UNKNOWN_SEX
        { title: 'Female', value: 1 },  // FEMALE
        { title: 'Male', value: 2 },    // MALE
        { title: 'Other', value: 3 }    // OTHER_SEX
      ],
      inputTextLengthThreshold: 60, // Character limit to auto-switch modes
      forceEndpointMode: null, // 'query', 'textProcess', or null (for automatic)
      
      // Parameters for /text/process endpoint
      chunkingStrategy: 'sliding_window_cleaned', // API default
      windowSize: 2, // Default for sliding window
      stepSize: 1, // Default step size
      splitThreshold: 0.5, // Default semantic split threshold
      minSegmentLength: 3, // Default minimum segment length (words)
      semanticModelForChunking: null, // Default to same as retrieval model if not set
      retrievalModelForTextProcess: null, // Default to same as query model if not set
      
      chunkRetrievalThreshold: 0.3, // Default retrieval threshold for chunks
      numResultsPerChunk: 10, // Default number of HPO terms per chunk
      
      // Assertion detection options
      noAssertionDetectionForTextProcess: false, // Default: enabled
      assertionPreferenceForTextProcess: 'dependency', // Default preference
      
      // Aggregation options
      aggregatedTermConfidence: 0.35, // Default confidence threshold for aggregation
      topTermPerChunkForAggregation: false // Default: use all terms for aggregation
    };
  },
  watch: {
    queryHistory: {
      // Watch for any changes to the queryHistory array
      handler() {
        logService.info('Query history updated', { newHistory: this.queryHistory })
        // Force scroll to top whenever history changes
        this.scrollToTop();
      },
      deep: true
    },
    selectedModel: {
      handler() {
        logService.info('Model changed', { newModel: this.selectedModel })
        // Reset settings to defaults when model changes
        this.similarityThreshold = 0.3;
        this.enableReranker = false;
        this.rerankerMode = 'cross-lingual';
        logService.info('Reset settings to defaults');
      }
    }
  },
  mounted() {
    logService.debug('QueryInterface mounted')
    // Set default model
    this.selectedModel = this.availableModels[0].value;
    
    // Initialize language from current UI language if supported
    const currentUiLang = this.$i18n.locale;
    logService.debug('Current UI language:', { language: currentUiLang });
    
    // Only set language if UI language is in our supported languages
    const supportedLanguages = this.availableLanguages.filter(lang => lang.value !== null).map(lang => lang.value);
    if (supportedLanguages.includes(currentUiLang)) {
      this.selectedLanguage = currentUiLang;
      logService.info('Setting query language from UI language:', { language: currentUiLang });
    } else {
      // Keep auto-detect (null) as default
      logService.info('UI language not in supported languages, using auto-detect');
    }
    
    // Apply URL parameters and handle auto-submit if needed
    this.applyUrlParametersAndAutoSubmit();
    
    // Add a scroll event listener to handle user scrolling
    const container = this.$refs.conversationContainer;
    if (container) {
      this.lastUserScrollPosition = 0;
      container.addEventListener('scroll', this.handleUserScroll);
    }
    
    // Focus the input field when component mounts
    if (this.$refs.queryInput) {
      this.$refs.queryInput.focus()
    }
  },
  
  unmounted() {
    // Clean up event listener
    const container = this.$refs.conversationContainer;
    if (container) {
      container.removeEventListener('scroll', this.handleUserScroll);
    }
  },
  
  updated() {
    // After component updates, force scroll to top if this was triggered by a new query
    // Use a flag to track if the update was triggered by a new query
    if (this.shouldScrollToTop) {
      logService.debug('Scrolling to top after update')
      this.scrollToTop();
      this.shouldScrollToTop = false;
    }
  },
  methods: {
    applyUrlParametersAndAutoSubmit() {
      // Get URL query parameters
      const queryParams = this.$route.query;
      logService.debug('Raw URL query parameters:', { ...queryParams });
      
      // Track if any advanced options were set from URL parameters
      let advancedOptionsWereSet = false;
      let performAutoSubmit = false;
      
      // Helper function for parsing boolean parameters
      const parseBooleanParam = (val) => typeof val === 'string' && (val.toLowerCase() === 'true' || val === '1');
      
      // Process 'text' parameter
      if (queryParams.text !== undefined) {
        const newTextValue = queryParams.text;
        this.queryText = newTextValue;
        logService.info('Applying URL parameter: text', { value: newTextValue });
      }
      
      // Process 'model' parameter
      if (queryParams.model !== undefined) {
        const paramValue = queryParams.model;
        const validModels = this.availableModels.map(m => m.value);
        
        if (validModels.includes(paramValue)) {
          this.selectedModel = paramValue;
          advancedOptionsWereSet = true;
          logService.info('Applying URL parameter: model', { value: paramValue });
        } else {
          logService.warn(`URL 'model' value '${paramValue}' is invalid.`);
        }
      }
      
      // Process 'threshold' parameter
      if (queryParams.threshold !== undefined) {
        const thresholdValue = parseFloat(queryParams.threshold);
        
        if (!isNaN(thresholdValue) && thresholdValue >= 0 && thresholdValue <= 1) {
          this.similarityThreshold = thresholdValue;
          advancedOptionsWereSet = true;
          logService.info('Applying URL parameter: threshold', { value: thresholdValue });
        } else {
          logService.warn(`URL 'threshold' value '${queryParams.threshold}' is invalid. Must be a number between 0 and 1.`);
        }
      }
      
      // Process 'reranker' parameter
      if (queryParams.reranker !== undefined) {
        const rerankerValue = parseBooleanParam(queryParams.reranker);
        this.enableReranker = rerankerValue;
        advancedOptionsWereSet = true;
        logService.info('Applying URL parameter: reranker', { value: rerankerValue });
      }
      
      // Process 'rerankerMode' parameter (only if reranker is enabled)
      if (queryParams.rerankerMode !== undefined && this.enableReranker) {
        const modeValue = queryParams.rerankerMode;
        
        if (this.rerankerModes.includes(modeValue)) {
          this.rerankerMode = modeValue;
          advancedOptionsWereSet = true;
          logService.info('Applying URL parameter: rerankerMode', { value: modeValue });
        } else {
          logService.warn(`URL 'rerankerMode' value '${modeValue}' is invalid.`);
        }
      }
      
      // Show advanced options panel if any advanced parameters were set
      if (advancedOptionsWereSet) {
        this.showAdvancedOptions = true;
        logService.info('Opened advanced options panel due to URL parameters.');
      }
      
      // Process 'autoSubmit' parameter
      if (queryParams.autoSubmit !== undefined) {
        performAutoSubmit = parseBooleanParam(queryParams.autoSubmit);
        logService.info('Found URL parameter: autoSubmit', { value: performAutoSubmit });
      } else if (queryParams.text !== undefined) {
        // If text is provided in URL but no autoSubmit parameter, default to auto-submit
        performAutoSubmit = true;
        logService.info('Text found in URL, defaulting to auto-submit');
      }
      
      // Auto-submit the query if requested and query text is not empty
      if (performAutoSubmit && this.queryText && this.queryText.trim()) {
        logService.info('Auto-submitting query based on URL parameters.');
        
        // Keep a reference to the query text since it will be cleared in submitQuery
        const currentQuery = this.queryText.trim();
        
        // Use nextTick to ensure the component is fully updated before submitting
        this.$nextTick(() => {
          // Add a delay to ensure the component is fully mounted and stable
          setTimeout(() => {
            logService.info('Executing query submission for:', { text: currentQuery });
            
            try {
              // Manually simulate what happens in submitQuery to ensure it works
              this.isLoading = true;
              
              // Add the query to the history
              this.queryHistory.unshift({
                query: currentQuery,
                loading: true,
                response: null,
                error: null
              });
              
              // Clear input field
              this.queryText = '';
              
              // Set scroll flags
              this.shouldScrollToTop = true;
              this.userHasScrolled = false;
              this.scrollToTop();
              
              // Execute the actual API call
              logService.info('Preparing API request data');
              const queryData = {
                text: currentQuery,
                model_name: this.selectedModel || 'FremyCompany/BioLORD-2023-M',
                language: this.selectedLanguage,
                num_results: this.numResults,
                similarity_threshold: this.similarityThreshold,
                enable_reranker: this.enableReranker,
                reranker_mode: this.rerankerMode,
                query_assertion_language: this.selectedLanguage
              };
              
              // Make API call
              logService.info('Sending auto-submitted query to API', queryData);
              PhentrieveService.queryHpo(queryData).then(response => {
                logService.info('Received API response for auto-submitted query');
                // Update history item
                this.queryHistory[0].loading = false;
                this.queryHistory[0].response = response;
                // Update reactivity
                this.queryHistory = [...this.queryHistory];
              }).catch(error => {
                logService.error('Error submitting auto query', error);
                this.queryHistory[0].loading = false;
                this.queryHistory[0].error = error;
                // Update reactivity
                this.queryHistory = [...this.queryHistory];
              }).finally(() => {
                this.isLoading = false;
                
                // Remove the autoSubmit parameter from URL to prevent re-submission on refresh
                setTimeout(() => {
                  const newQuery = { ...this.$route.query };
                  delete newQuery.autoSubmit;
                  this.$router.replace({ query: newQuery }).catch(err => {
                    if (err.name !== 'NavigationDuplicated' && err.name !== 'NavigationCancelled') {
                      logService.warn('Error updating URL after auto-submit:', err);
                    }
                  });
                }, 500);
              });
            } catch (e) {
              logService.error('Exception during auto-submit process', e);
              this.isLoading = false;
            }
          }, 800); // Increased delay to ensure everything is ready
        });
      }
    },
    
    handleUserScroll() {
      // Track when the user manually scrolls
      const container = this.$refs.conversationContainer;
      if (container) {
        this.lastUserScrollPosition = container.scrollTop;
        this.userHasScrolled = true;
      }
      
      if (!this.userHasScrolled && this.shouldScrollToTop) {
        logService.debug('User initiated scroll in conversation')
        this.userHasScrolled = true;
      }
    },
    
    addToPhenotypeCollection(phenotype, queryAssertionStatus = null) {
      logService.info('Adding phenotype to collection', { 
        phenotype: phenotype,
        assertionStatus: queryAssertionStatus
      })
      // Check if this phenotype is already in the collection
      const isDuplicate = this.collectedPhenotypes.some(item => item.hpo_id === phenotype.hpo_id);
      
      if (!isDuplicate) {
        // Get the current query assertion status from the response if available
        const currentResponse = this.queryHistory.length > 0 ? this.queryHistory[0].response : null;
        const responseAssertionStatus = currentResponse?.query_assertion_status || null;
        
        // Use provided status, response status, or default to 'affirmed'
        const assertionStatus = queryAssertionStatus || responseAssertionStatus || 'affirmed';
        
        // Add phenotype to collection with assertion status
        this.collectedPhenotypes.push({
          ...phenotype,
          added_at: new Date(),
          assertion_status: assertionStatus
        });
        
        // Auto-show the collection panel if this is the first item
        if (this.collectedPhenotypes.length === 1) {
          this.showCollectionPanel = true;
        }
      }
    },
    
    removePhenotype(index) {
      logService.info('Removing phenotype from collection', { index: index, phenotype: this.collectedPhenotypes[index] })
      this.collectedPhenotypes.splice(index, 1);
    },
    
    toggleAssertionStatus(index) {
      if (index >= 0 && index < this.collectedPhenotypes.length) {
        const phenotype = this.collectedPhenotypes[index];
        
        // Toggle between 'affirmed' and 'negated'
        const newStatus = phenotype.assertion_status === 'negated' ? 'affirmed' : 'negated';
        
        logService.info('Toggling phenotype assertion status', { 
          phenotype: phenotype.hpo_id,
          oldStatus: phenotype.assertion_status,
          newStatus: newStatus
        });
        
        // Update the phenotype with the new assertion status
        this.collectedPhenotypes.splice(index, 1, {
          ...phenotype,
          assertion_status: newStatus
        });
      }
    },
    
    clearPhenotypeCollection() {
      logService.info('Clearing phenotype collection and subject information')
      this.collectedPhenotypes = [];
      
      // Also reset the subject information fields
      this.phenopacketSubjectId = '';
      this.phenopacketSex = null;
      this.phenopacketDateOfBirth = null;
      
      logService.debug('Subject information inputs have been reset');
    },
    
    toggleCollectionPanel() {
      logService.debug('Toggling collection panel')
      this.showCollectionPanel = !this.showCollectionPanel;
    },
    
    scrollToTop() {
      logService.debug('Attempting to scroll conversation to top')
      // Schedule multiple scroll attempts to ensure it works in all situations
      const doScroll = () => {
        const container = this.$refs.conversationContainer;
        if (container) {
          // Temporarily disable smooth scrolling for immediate jump
          container.style.scrollBehavior = 'auto';
          // Force scroll to top
          container.scrollTop = 0;
          // Restore smooth scrolling
          setTimeout(() => {
            container.style.scrollBehavior = 'smooth';
          }, 50);
        }
      };
      
      // Try multiple times with increasing delays
      doScroll();
      this.$nextTick(doScroll);
      setTimeout(doScroll, 100);
      setTimeout(doScroll, 300);
      setTimeout(doScroll, 500);
    },
    
    exportPhenotypes() {
      logService.info('Exporting phenotypes as text', { count: this.collectedPhenotypes.length })
      // Create a formatted text with the phenotypes
      let exportText = "HPO Phenotypes Collection\n";
      exportText += "Exported on: " + new Date().toLocaleString() + "\n\n";
      exportText += "ID\tLabel\tAssertion Status\n";
      
      this.collectedPhenotypes.forEach(phenotype => {
        // Include assertion status in the export
        const assertionStatus = phenotype.assertion_status || 'affirmed';
        exportText += `${phenotype.hpo_id}\t${phenotype.label}\t${assertionStatus}\n`;
      });
      
      // Create a blob and download it
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
        // Create the Phenopacket directly as a JavaScript object
        // following the GA4GH Phenopacket v2 schema
        const timestamp = new Date().toISOString();
        const phenopacketId = `phentrieve-export-${Date.now()}`;
        
        // Basic structure of a Phenopacket
        const phenopacket = {
          id: phenopacketId,
          metaData: {
            created: timestamp,
            createdBy: "Phentrieve Frontend Application",
            phenopacketSchemaVersion: "2.0.0",
            resources: [
              {
                id: "phentrieve",
                name: "Phentrieve: AI-Powered Clinical Text to HPO Term Mapping",
                namespacePrefix: "Phentrieve",
                url: "https://phentrieve.kidney-genetics.org/",
                version: import.meta.env.VITE_APP_VERSION || "1.0.0",
                iriPrefix: "phentrieve"
              }
            ]
          },
          phenotypicFeatures: []
        };
        
        // Add subject information if provided
        if (this.phenopacketSubjectId || this.phenopacketSex !== null || this.phenopacketDateOfBirth) {
          phenopacket.subject = {};
          let subjectInfoAdded = false;
          
          if (this.phenopacketSubjectId && this.phenopacketSubjectId.trim()) {
            phenopacket.subject.id = this.phenopacketSubjectId.trim();
            subjectInfoAdded = true;
            logService.debug('Adding subject ID to Phenopacket', { id: this.phenopacketSubjectId.trim() });
          }
          
          if (this.phenopacketSex !== null && this.phenopacketSex !== undefined) {
            // Map numeric sex values to string values expected in the schema
            const sexMap = {
              0: "UNKNOWN_SEX",
              1: "FEMALE",
              2: "MALE",
              3: "OTHER_SEX"
            };
            phenopacket.subject.sex = sexMap[this.phenopacketSex];
            subjectInfoAdded = true;
            logService.debug('Adding subject sex to Phenopacket', { sex: this.phenopacketSex });
          }
          
          if (this.phenopacketDateOfBirth) {
            try {
              // Parse date from YYYY-MM-DD format
              const dob = new Date(this.phenopacketDateOfBirth + "T00:00:00Z");
              if (!isNaN(dob.getTime())) {
                phenopacket.subject.timeAtLastEncounter = {
                  timestamp: dob.toISOString()
                };
                subjectInfoAdded = true;
                logService.debug('Adding subject date of birth to Phenopacket', { dob: this.phenopacketDateOfBirth });
              } else {
                logService.warn('Invalid date format for Date of Birth', { input: this.phenopacketDateOfBirth });
              }
            } catch (dateError) {
              logService.error('Error processing Date of Birth for Phenopacket', { 
                input: this.phenopacketDateOfBirth, 
                error: dateError 
              });
            }
          }
          
          // If no subject info was actually added, remove the empty subject object
          if (!subjectInfoAdded) {
            delete phenopacket.subject;
          } else {
            logService.info('Subject information added to Phenopacket');
          }
        }
        
        // Add phenotypic features
        this.collectedPhenotypes.forEach(collectedPheno => {
          // Create the phenotypic feature object
          const phenotypicFeature = {
            type: {
              id: collectedPheno.hpo_id,
              label: collectedPheno.label
            }
          };
          
          // Set the excluded property based on the assertion status
          // In GA4GH Phenopacket v2, 'excluded: true' means the phenotype is negated/absent
          if (collectedPheno.assertion_status === 'negated') {
            phenotypicFeature.excluded = true;
            logService.debug('Adding negated phenotype to Phenopacket', { 
              id: collectedPheno.hpo_id, 
              status: 'negated' 
            });
          } else {
            // For affirmed phenotypes, we can either set excluded:false or omit it
            // We'll set it explicitly for clarity
            phenotypicFeature.excluded = false;
            logService.debug('Adding affirmed phenotype to Phenopacket', { 
              id: collectedPheno.hpo_id, 
              status: 'affirmed' 
            });
          }
          
          phenopacket.phenotypicFeatures.push(phenotypicFeature);
        });
        
        // Convert to JSON string (pretty-printed)
        const phenopacketJsonString = JSON.stringify(phenopacket, null, 2);
        
        // Trigger download
        const blob = new Blob([phenopacketJsonString], { type: 'application/json;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        const isoTimestamp = new Date().toISOString().replace(/[:.]/g, '-');
        a.download = `phentrieve_phenopacket_${isoTimestamp}.json`;
        a.href = url;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        logService.info('Phenopacket successfully exported as JSON', { 
          phenopacketId: phenopacket.id, 
          filename: a.download 
        });
        
      } catch (error) {
        logService.error('Error during Phenopacket creation or export', { 
          errorMessage: error.message, 
          stack: error.stack,
          errorObject: error 
        });
        // Inform the user about the error
        alert('An error occurred while exporting the Phenopacket. Please check the console for details.');
      }
    },
  
    async submitQuery() {
      logService.debug('Submitting query')
      // Validate input - prevent empty queries
      const queryTextTrimmed = this.queryText.trim();
      if (!queryTextTrimmed) {
        logService.warn('Empty query submission prevented');
        return;
      }
      
      // Determine the mode BEFORE clearing the input field
      // This fixes the issue where the mode is determined incorrectly after clearing the input
      const useTextProcessMode = this.forceEndpointMode === 'textProcess' || 
                                queryTextTrimmed.length > this.inputTextLengthThreshold;
      
      logService.info('Mode determination', { 
        length: queryTextTrimmed.length, 
        threshold: this.inputTextLengthThreshold,
        forcedMode: this.forceEndpointMode,
        useTextProcessMode: useTextProcessMode
      });
      
      this.isLoading = true;
      
      // Save the query text before clearing input field
      const currentQuery = queryTextTrimmed;
      
      // Add the query result to the conversation at the beginning (newest first)
      this.queryHistory.unshift({
        query: currentQuery,
        loading: true,
        response: null,
        error: null,
        type: useTextProcessMode ? 'textProcess' : 'query' // Add type to distinguish between query and text process results
      });

      // Get reference to the latest history item (now at index 0)
      const historyIndex = 0;
      
      // Reset input
      this.queryText = '';
      
      // Set the flag to indicate a scroll to top should happen
      this.shouldScrollToTop = true;
      this.userHasScrolled = false;
      
      // Also use direct scrollToTop method for redundancy
      this.scrollToTop();

      try {
        let response;
        
        if (useTextProcessMode) {
          // Text Processing Mode - use /text/process endpoint
          logService.info('Using text processing endpoint for longer text');
          
          // Prepare text processing request data
          const textProcessData = {
            text_content: currentQuery,
            language: this.selectedLanguage, // Pass explicitly selected language or null for auto-detection
            chunking_strategy: this.chunkingStrategy,
            window_size: this.windowSize,
            step_size: this.stepSize,
            split_threshold: this.splitThreshold,
            min_segment_length: this.minSegmentLength,
            semantic_model_name: this.semanticModelForChunking || this.selectedModel, // Use specific or fallback
            retrieval_model_name: this.retrievalModelForTextProcess || this.selectedModel, // Use specific or fallback
            trust_remote_code: true,
            chunk_retrieval_threshold: this.chunkRetrievalThreshold,
            num_results_per_chunk: this.numResultsPerChunk,
            enable_reranker: this.enableReranker,
            reranker_mode: this.rerankerMode,
            no_assertion_detection: this.noAssertionDetectionForTextProcess,
            assertion_preference: this.assertionPreferenceForTextProcess,
            aggregated_term_confidence: this.aggregatedTermConfidence,
            top_term_per_chunk_for_aggregation: this.topTermPerChunkForAggregation,
          };
          
          // Make Text Processing API call
          logService.info('Sending request to text processing API', {
            strategy: textProcessData.chunking_strategy,
            textLength: textProcessData.text_content.length,
            model: textProcessData.retrieval_model_name
          });
          
          response = await PhentrieveService.processText(textProcessData);
          
          logService.info('Received text processing API response', {
            numChunks: response.processed_chunks?.length || 0,
            numAggregatedTerms: response.aggregated_hpo_terms?.length || 0
          });
        } else {
          // Query Mode - use /query/ endpoint (original behavior)
          // Prepare query request data
          const queryData = {
            text: currentQuery,
            model_name: this.selectedModel || 'FremyCompany/BioLORD-2023-M',
            language: this.selectedLanguage, // Pass explicitly selected language or null for auto-detection
            num_results: this.numResults,
            similarity_threshold: this.similarityThreshold,
            enable_reranker: this.enableReranker,
            reranker_mode: this.rerankerMode,
            query_assertion_language: this.selectedLanguage // Use same language for assertion detection
          };
          
          // Make Query API call
          logService.info('Sending request to query API', queryData);
          response = await PhentrieveService.queryHpo(queryData);
          logService.info('Received query API response', response);
        }
        
        // Update history item using index reference
        this.queryHistory[historyIndex].loading = false;
        this.queryHistory[historyIndex].response = response;
        
        // Make a shallow copy of the array to trigger reactivity
        this.queryHistory = [...this.queryHistory];
      } catch (error) {
        // Handle error
        this.queryHistory[historyIndex].loading = false;
        this.queryHistory[historyIndex].error = error;
        logService.error('Error submitting query/processing text', error);
        
        // Make a shallow copy of the array to trigger reactivity
        this.queryHistory = [...this.queryHistory];
      } finally {
        this.isLoading = false;
      }
    }
  }
};
</script>

<style scoped>
.search-container {
  max-width: 800px;
  width: 100%;
  margin-top: 0; /* Remove the top margin */
}

.collection-fab-position {
  margin: 16px;
  bottom: 60px !important;
  right: 16px !important;
  z-index: 1000;
}

.search-input {
  font-size: 1rem;
  line-height: 1.5;
}

.search-input :deep(.v-field) {
  border-radius: 24px;
  min-height: 44px;
  box-shadow: none;
}

.search-input :deep(.v-field__input) {
  min-height: 44px;
  padding-top: 0;
  padding-bottom: 0;
}

/* Specific styles for textarea in document mode */
.search-input.v-textarea :deep(.v-field) {
  border-radius: 16px;
  padding-top: 8px;
  padding-bottom: 8px;
}

.search-input.v-textarea :deep(.v-field__input) {
  min-height: 80px;
  padding-top: 4px;
}

/* Ensure search bar container has enough height for the textarea */
.search-bar {
  min-height: 60px;
}

.search-input :deep(.v-field__outline) {
  --v-field-border-width: 0px;
}

/* Remove the border on the container and handle it in the input field */
.search-bar {
  border: 1px solid rgba(0, 0, 0, 0.15);
  border-radius: 28px;
}

/* Accessibility improvements for form fields */
:deep(.text-high-emphasis) {
  color: rgba(0, 0, 0, 0.87) !important; /* Darker text for better contrast */
  font-weight: 500;
}

/* Light theme styles for form fields */
:deep(.v-field--variant-outlined) {
  background-color: #FFFFFF !important;
  box-shadow: none;
}

:deep(.v-field__input) {
  color: rgba(0, 0, 0, 0.87) !important; /* Darker text for better contrast */
}

/* Make sure placeholder text is also accessible */
:deep(.v-field__input::placeholder) {
  color: rgba(0, 0, 0, 0.6) !important;
}

:deep(.text-high-emphasis) {
  color: rgba(0, 0, 0, 0.87) !important;
  font-weight: 500;
}

:deep(.v-field__input) {
  color: rgba(0, 0, 0, 0.87) !important;
}

:deep(.v-field__input::placeholder) {
  color: rgba(0, 0, 0, 0.6) !important;
}

.conversation-container {
  max-height: calc(70vh - 100px); /* Further reduced height to add more space at bottom */
  min-height: 200px;
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 8px;
  scroll-behavior: smooth;
  margin-bottom: 24px; /* Increased bottom margin */
  border-radius: 8px;
}

/* Make scrollbar visible */
.conversation-container::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

.conversation-container::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.2);
  border-radius: 4px;
}

.conversation-container::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.05);
  border-radius: 4px;
}

.query-bubble {
  background-color: rgba(var(--v-theme-primary), 0.08);
  border-radius: 12px;
  padding: 12px 16px;
  max-width: 80%;
}

.response-bubble {
  background-color: rgba(var(--v-theme-secondary), 0.05);
  border-radius: 12px;
  padding: 12px 16px;
  width: 100%;
}

.user-query {
  justify-content: flex-start;
}

.bot-response {
  justify-content: flex-start;
}
</style>
