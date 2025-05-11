<template>
  <div>
    <v-card class="mb-4">
      <v-card-title class="text-h6">Phentrieve HPO Query</v-card-title>
      <v-card-text>
        <v-textarea
          v-model="queryText"
          label="Enter clinical text to query for HPO terms"
          rows="4"
          counter
          hide-details="auto"
          :disabled="isLoading"
          outlined
          auto-grow
        ></v-textarea>
        
        <v-row class="mt-2">
          <v-col cols="12" md="6">
            <v-select
              v-model="selectedModel"
              :items="availableModels"
              item-title="text"
              item-value="value"
              label="Embedding Model"
              :disabled="isLoading"
              outlined
              dense
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
              outlined
              dense
            ></v-select>
          </v-col>
        </v-row>
      </v-card-text>
      
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn
          color="primary"
          @click="submitQuery"
          :loading="isLoading"
          :disabled="!queryText.trim()"
        >
          <v-icon left>mdi-magnify</v-icon>
          Query HPO Terms
        </v-btn>
      </v-card-actions>
    </v-card>
    
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
      queryHistory: []
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
