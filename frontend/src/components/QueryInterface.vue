<template>
  <div class="search-container mx-auto px-2">
    <!-- Clean Search Bar with Integrated Button -->
    <div class="search-bar-container pa-2 pa-sm-4">
      <v-sheet rounded="pill" elevation="2" class="pa-1 pa-sm-2 search-bar" color="surface">
        <div class="d-flex align-center flex-wrap flex-sm-nowrap">
          <v-text-field
            v-model="queryText"
            density="comfortable"
            variant="outlined"
            hide-details
            class="search-input ml-2 ml-sm-3 flex-grow-1"
            :disabled="isLoading"
            @keydown.enter.prevent="!isLoading && queryText.trim() ? submitQuery() : null"
            bg-color="surface"
            color="primary"
            aria-label="Clinical text input field"
            :aria-description="'Enter clinical text to search for HPO terms' + (isLoading ? '. Search in progress' : '')"
          >
            <template v-slot:label>
              <span class="text-high-emphasis">Enter clinical text...</span>
            </template>
          </v-text-field>
          
          <div class="d-flex align-center">
            <v-btn 
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
            >
              <v-icon>mdi-magnify</v-icon>
            </v-btn>
          </div>
        </div>
      </v-sheet>
      
      <!-- Advanced Options Panel (Hidden by Default) -->
      <v-expand-transition>
        <v-sheet 
          v-if="showAdvancedOptions" 
          rounded="lg" 
          elevation="1" 
          class="mt-3 pa-4"
          id="advanced-options-panel"
          role="region"
          aria-label="Advanced search options"
        >
          <div class="text-subtitle-2 mb-3">Advanced Options</div>
          
          <v-row>
            <v-col cols="12" md="6">
              <v-select
                v-model="selectedModel"
                :items="availableModels"
                item-title="text"
                item-value="value"
                :disabled="isLoading"
                variant="outlined"
                density="compact"
                aria-label="Select embedding model"
                :aria-description="'Choose the model to use for text embedding. Currently selected: ' + selectedModel"
                bg-color="surface"
                color="primary"
              >
                <template v-slot:label>
                  <span class="text-high-emphasis">Embedding Model</span>
                </template>
              </v-select>
            </v-col>
            
            <v-col cols="12" md="6">
              <v-slider
                v-model="similarityThreshold"
                min="0"
                max="1"
                step="0.05"
                thumb-label
                :disabled="isLoading"
                aria-label="Adjust similarity threshold"
                :aria-valuetext="`Similarity threshold set to ${(similarityThreshold * 100).toFixed(0)}%`"
                :aria-description="'Set the minimum similarity score required for matches. Higher values mean more precise but fewer results.'"
                color="primary"
              >
                <template v-slot:label>
                  <span class="text-high-emphasis">Similarity Threshold</span>
                </template>
              </v-slider>
            </v-col>
          </v-row>
          
          <v-row>
            <v-col cols="12" md="6">
              <v-switch
                v-model="enableReranker"
                :disabled="isLoading"
                color="primary"
                hide-details
                class="mb-2"
                aria-label="Toggle re-ranking feature"
                :aria-description="'Enable or disable re-ranking of search results'"
              >
                <template v-slot:label>
                  <span class="text-high-emphasis">Enable Re-ranking</span>
                </template>
              </v-switch>
            </v-col>
            
            <v-col cols="12" md="6" v-if="enableReranker">
              <v-select
                v-model="rerankerMode"
                :items="rerankerModes"
                :disabled="!enableReranker || isLoading"
                variant="outlined"
                density="compact"
                aria-label="Select reranker mode"
                :aria-description="'Choose between cross-lingual or monolingual re-ranking mode'"
                bg-color="surface"
                color="primary"
              >
                <template v-slot:label>
                  <span class="text-high-emphasis">Re-ranker Mode</span>
                </template>
              </v-select>
            </v-col>
          </v-row>
        </v-sheet>
      </v-expand-transition>
    </div>
    
    <!-- Chat-like conversation interface -->
    <div class="conversation-container" ref="conversationContainer">
      <div v-for="(item, index) in queryHistory" :key="index" class="mb-4">
        <!-- User query -->
        <div class="user-query d-flex">
          <v-avatar color="primary" size="36" class="mt-1 mr-2">
            <span class="white--text">U</span>
          </v-avatar>
          <div class="query-bubble">
            <p class="mb-0">{{ item.query }}</p>
          </div>
        </div>
        
        <!-- API response -->
        <div class="bot-response d-flex mt-2" v-if="item.loading || item.response || item.error">
          <v-avatar color="info" size="36" class="mt-1 mr-2">
            <span class="white--text">P</span> <!-- P for Phentrieve -->
          </v-avatar>
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
              :error="item.error"
              :collected-phenotypes="collectedPhenotypes"
              @add-to-collection="addToPhenotypeCollection"
            />
          </div>
        </div>
      </div>
    </div>
    
    <!-- Floating action button for collection panel -->
    <v-btn
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
    
    <!-- Collection Panel -->
    <v-navigation-drawer
      v-model="showCollectionPanel"
      location="right"
      width="400"
      temporary
      aria-label="Phenotype collection"
    >
      <v-list-item>
        <v-list-item-title class="text-h6">HPO Collection</v-list-item-title>
        <template v-slot:append>
          <v-btn icon @click="toggleCollectionPanel" aria-label="Close HPO Collection Panel">
            <v-icon>mdi-close</v-icon>
          </v-btn>
        </template>
      </v-list-item>
      
      <v-divider></v-divider>
      
      <v-list v-if="collectedPhenotypes.length > 0">
        <v-list-subheader>{{ collectedPhenotypes.length }} phenotype(s) collected</v-list-subheader>
        
        <v-list-item
          v-for="(phenotype, index) in collectedPhenotypes"
          :key="phenotype.hpo_id"
          density="compact"
        >
          <v-list-item-title>
            <strong>{{ phenotype.hpo_id }}</strong>
          </v-list-item-title>
          <v-list-item-subtitle>
            {{ phenotype.label }}
          </v-list-item-subtitle>
          
          <template v-slot:append>
            <v-btn
              icon="mdi-close"
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
        <div class="text-body-1 text-grey-darken-3">No phenotypes collected yet</div>
        <div class="text-body-2 text-grey-darken-3 mt-2">
          Click the <v-icon size="small">mdi-plus-circle</v-icon> button next to any HPO term to add it to your collection
        </div>
      </v-sheet>
      
      <v-divider class="mt-4"></v-divider>
      <v-list-subheader>Subject Information (Optional)</v-list-subheader>
      <div class="pa-4">
        <v-text-field
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
          <template v-slot:label><span class="text-high-emphasis">Subject ID</span></template>
        </v-text-field>

        <v-select
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
          <template v-slot:label><span class="text-high-emphasis">Sex</span></template>
        </v-select>

        <v-text-field
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
          <template v-slot:label><span class="text-high-emphasis">Date of Birth</span></template>
        </v-text-field>
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
            Export as Phenopacket (JSON)
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
            Export as Text
          </v-btn>
          <v-btn
            block
            variant="tonal"
            color="error"
            prepend-icon="mdi-delete"
            @click="clearPhenotypeCollection"
            :disabled="collectedPhenotypes.length === 0"
          >
            Clear Collection
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
  data() {
    return {
      queryText: '',
      selectedModel: null,
      availableModels: [
        { text: 'BioLORD-2023-M (Medical)', value: 'FremyCompany/BioLORD-2023-M' },
        { text: 'Jina Embed v2 (German)', value: 'jinaai/jina-embeddings-v2-base-de' },
        { text: 'DistilUSE (Multilingual)', value: 'distiluse-base-multilingual-cased-v1' }
      ],
      similarityThreshold: 0.3,
      numResults: 10,
      enableReranker: false,
      rerankerMode: 'cross-lingual',
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
      ]
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
    handleUserScroll(event) {
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
    
    addToPhenotypeCollection(phenotype) {
      logService.info('Adding phenotype to collection', { phenotype: phenotype })
      // Check if this phenotype is already in the collection
      const isDuplicate = this.collectedPhenotypes.some(item => item.hpo_id === phenotype.hpo_id);
      
      if (!isDuplicate) {
        // Add phenotype to collection
        this.collectedPhenotypes.push({
          ...phenotype,
          added_at: new Date()
        });
        
        // Auto-show the collection panel if this is the first item
        if (this.collectedPhenotypes.length === 1) {
          this.showCollectionPanel = true;
        }
      }
    },
    
    removePhenotype(index) {
      logService.info('Removing phenotype from collection', { index: index })
      this.collectedPhenotypes.splice(index, 1);
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
      exportText += "ID\tLabel\n";
      
      this.collectedPhenotypes.forEach(phenotype => {
        exportText += `${phenotype.hpo_id}\t${phenotype.label}\n`;
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
          phenopacket.phenotypicFeatures.push({
            type: {
              id: collectedPheno.hpo_id,
              label: collectedPheno.label
            }
          });
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
      
      this.isLoading = true;
      
      // Save the query text before clearing input field
      const currentQuery = queryTextTrimmed;
      
      // Add the query result to the conversation at the beginning (newest first)
      this.queryHistory.unshift({
        query: currentQuery,
        loading: true,
        response: null,
        error: null
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
        // Prepare request data matching the QueryRequest schema
        const queryData = {
          text: currentQuery,
          model_name: this.selectedModel || 'FremyCompany/BioLORD-2023-M',
          num_results: this.numResults,
          similarity_threshold: this.similarityThreshold,
          enable_reranker: this.enableReranker,
          reranker_mode: this.rerankerMode
        };
        
        // Make API call
        logService.info('Sending query to API', queryData);
        const response = await PhentrieveService.queryHpo(queryData);
        logService.info('Received API response', response);
        
        // Update history item using index reference
        this.queryHistory[historyIndex].loading = false;
        this.queryHistory[historyIndex].response = response;
        
        // Make a shallow copy of the array to trigger reactivity
        this.queryHistory = [...this.queryHistory];
      } catch (error) {
        // Handle error
        this.queryHistory[historyIndex].loading = false;
        this.queryHistory[historyIndex].error = error;
        logService.error('Error submitting query', error);
        
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
}

.search-input :deep(.v-field__input) {
  min-height: 44px;
  padding-top: 0;
  padding-bottom: 0;
}

.search-input :deep(.v-field__outline) {
  --v-field-border-width: 1px;
}

/* Accessibility improvements for form fields */
:deep(.text-high-emphasis) {
  color: rgba(0, 0, 0, 0.87) !important; /* Darker text for better contrast */
  font-weight: 500;
}

/* Improve contrast for form fields */
:deep(.v-field--variant-outlined) {
  background-color: #f8f9fa !important; /* Lighter background */
  border: 1px solid rgba(0, 0, 0, 0.23);
}

:deep(.v-field__input) {
  color: rgba(0, 0, 0, 0.87) !important; /* Darker text for better contrast */
}

/* Make sure placeholder text is also accessible */
:deep(.v-field__input::placeholder) {
  color: rgba(0, 0, 0, 0.6) !important;
}

/* Dark mode support - use media query for dark theme if needed */
@media (prefers-color-scheme: dark) {
  :deep(.text-high-emphasis) {
    color: rgba(255, 255, 255, 0.87) !important;
  }
  
  :deep(.v-field--variant-outlined) {
    background-color: rgba(30, 30, 30, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.23);
  }
  
  :deep(.v-field__input) {
    color: rgba(255, 255, 255, 0.87) !important;
  }
  
  :deep(.v-field__input::placeholder) {
    color: rgba(255, 255, 255, 0.6) !important;
  }
}

.conversation-container {
  max-height: 600px;
  overflow-y: auto;
  padding-right: 8px;
  scroll-behavior: smooth;
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
