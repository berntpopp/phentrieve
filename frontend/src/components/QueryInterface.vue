<template>
  <div class="search-container mx-auto" style="max-width: 800px">
    <!-- Clean Search Bar with Integrated Button -->
    <div class="search-bar-container pa-4">
      <v-sheet rounded="pill" elevation="2" class="pa-2 search-bar">
        <div class="d-flex align-center">
          <v-textarea
            v-model="queryText"
            density="compact"
            variant="plain"
            placeholder="Enter clinical text to search for HPO terms..."
            rows="1"
            auto-grow
            hide-details
            class="search-input ml-3"
            :disabled="isLoading"
            @keydown.enter.prevent="!isLoading && queryText.trim() ? submitQuery() : null"
          ></v-textarea>
          
          <v-btn 
            icon 
            variant="text" 
            color="primary" 
            class="mx-2"
            @click="showAdvancedOptions = !showAdvancedOptions"
            :disabled="isLoading"
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
            class="mr-2"
          >
            <v-icon>mdi-magnify</v-icon>
          </v-btn>
        </div>
      </v-sheet>
      
      <!-- Advanced Options Panel (Hidden by Default) -->
      <v-expand-transition>
        <v-sheet v-if="showAdvancedOptions" rounded="lg" elevation="1" class="mt-3 pa-4">
          <div class="text-subtitle-2 mb-3">Advanced Options</div>
          
          <v-row>
            <v-col cols="12" md="6">
              <v-select
                v-model="selectedModel"
                :items="availableModels"
                item-title="text"
                item-value="value"
                label="Embedding Model"
                :disabled="isLoading"
                variant="outlined"
                density="compact"
              ></v-select>
            </v-col>
            
            <v-col cols="12" md="6">
              <v-slider
                v-model="similarityThreshold"
                label="Similarity Threshold"
                min="0"
                max="1"
                step="0.05"
                thumb-label
                :disabled="isLoading"
              ></v-slider>
            </v-col>
          </v-row>
          
          <v-row>
            <v-col cols="12" md="6">
              <v-switch
                v-model="enableReranker"
                label="Enable Re-ranking"
                :disabled="isLoading"
                color="primary"
                hide-details
              ></v-switch>
            </v-col>
            
            <v-col cols="12" md="6" v-if="enableReranker">
              <v-select
                v-model="rerankerMode"
                :items="['cross-lingual', 'monolingual']"
                label="Reranker Mode"
                :disabled="isLoading"
                variant="outlined"
                density="compact"
              ></v-select>
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
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import ResultsDisplay from './ResultsDisplay.vue';
import PhentrieveService from '../services/PhentrieveService';

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
      showAdvancedOptions: false
    };
  },
  watch: {
    queryHistory: {
      handler() {
        this.$nextTick(() => {
          if (this.$refs.conversationContainer) {
            this.$refs.conversationContainer.scrollTop = this.$refs.conversationContainer.scrollHeight;
          }
        });
      },
      deep: true
    }
  },
  mounted() {
    // Set default model
    this.selectedModel = this.availableModels[0].value;
  },
  methods: {
    async submitQuery() {
      if (!this.queryText.trim()) return;
      
      this.isLoading = true;
      
      // Add to history
      const historyItem = {
        query: this.queryText,
        loading: true,
        response: null,
        error: null
      };
      
      this.queryHistory.push(historyItem);
      
      try {
        // Prepare request data matching the QueryRequest schema
        const queryData = {
          text: this.queryText,
          model_name: this.selectedModel,
          num_results: this.numResults,
          similarity_threshold: this.similarityThreshold,
          enable_reranker: this.enableReranker,
          reranker_mode: this.rerankerMode,
        };
        
        // Make API call
        console.log('Sending query to API:', queryData);
        const response = await PhentrieveService.queryHpo(queryData);
        console.log('Received API response:', response);
        
        // Update history item with direct property assignment
        historyItem.loading = false;
        historyItem.response = response;
        
        // Make a shallow copy of the array to trigger reactivity
        this.queryHistory = [...this.queryHistory];
      } catch (error) {
        // Handle error
        historyItem.loading = false;
        historyItem.error = error;
        console.error('Error submitting query:', error);
        
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
.conversation-container {
  max-height: 600px;
  overflow-y: auto;
  padding-right: 8px;
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
