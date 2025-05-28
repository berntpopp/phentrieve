<template>
  <div class="results-container">
    <!-- Regular Query Results Display -->
    <div v-if="resultType === 'query' && responseData && responseData.results && responseData.results.length > 0">
      <v-card class="mb-4 info-card">
        <v-card-title class="text-subtitle-1 pa-2 pa-sm-4">
          <div class="d-flex flex-wrap align-center">
            <!-- Model info - always present -->
            <div class="info-item mr-4 mb-1">
              <v-icon color="info" class="mr-1" size="small">mdi-information</v-icon>
              <span class="model-name">
                <small class="text-caption text-medium-emphasis">{{ $t('resultsDisplay.modelLabel') }}:</small>
                {{ displayModelName(responseData.model_used_for_retrieval) }}
              </span>
            </div>
            
            <!-- Language info -->
            <div v-if="responseData.language_detected" class="info-item mr-4 mb-1">
              <v-icon color="info" class="mr-1" size="small">mdi-translate</v-icon>
              <span>
                <small class="text-caption text-medium-emphasis">{{ $t('resultsDisplay.languageLabel') }}:</small>
                {{ responseData.language_detected }}
              </span>
            </div>
            
            <!-- Assertion status -->
            <div v-if="responseData.query_assertion_status" class="info-item mb-1">
              <v-icon 
                :color="responseData.query_assertion_status === 'negated' ? 'error' : 'success'" 
                class="mr-1" 
                size="small"
              >
                {{ responseData.query_assertion_status === 'negated' ? 'mdi-block-helper' : 'mdi-check-circle' }}
              </v-icon>
              <span>
                <small class="text-caption text-medium-emphasis">{{ $t('resultsDisplay.assertionLabel', 'Assertion') }}:</small>
                <v-chip
                  size="x-small"
                  :color="responseData.query_assertion_status === 'negated' ? 'error' : 'success'"
                  class="ml-1"
                  density="comfortable"
                >
                  {{ responseData.query_assertion_status === 'negated' ? $t('resultsDisplay.negated', 'Negated') : $t('resultsDisplay.affirmed', 'Affirmed') }}
                </v-chip>
              </span>
            </div>
          </div>
          
          <!-- Reranker info on separate line -->
          <div v-if="responseData.reranker_used" class="info-item mt-2">
            <v-icon color="info" class="mr-1" size="small">mdi-filter</v-icon>
            <span class="model-name">
              <small class="text-caption text-medium-emphasis">{{ $t('resultsDisplay.rerankerLabel') }}:</small>
              {{ displayModelName(responseData.reranker_used) }}
            </span>
          </div>
        </v-card-title>
      </v-card>

      <v-list lines="two" class="rounded-lg mt-2">
        <v-list-item
          v-for="(result, index) in responseData.results"
          :key="index"
          class="mb-1 rounded-lg"
          :color="isAlreadyCollected(result.hpo_id) ? 'primary-lighten-5' : 'grey-lighten-5'"
          border
          density="compact"
        >
          <template v-slot:prepend>
            <v-badge
              :content="index + 1"
              color="primary"
              inline
              class="mr-2"
            ></v-badge>
          </template>
          
          <v-list-item-title class="pb-2">
            <div class="d-flex flex-wrap align-center">
              <!-- HPO ID and Label on the left -->
              <div class="flex-grow-1">
                <div class="d-flex align-center mb-1">
                  <a 
                    :href="`https://hpo.jax.org/browse/term/${result.hpo_id}`" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    class="hpo-link"
                    :title="`View ${result.hpo_id} in HPO Browser`"
                  >
                    <span class="hpo-id font-weight-bold">{{ result.hpo_id }}</span>
                    <v-icon size="x-small" class="ml-1">mdi-open-in-new</v-icon>
                  </a>
                </div>
                <div class="d-flex align-center">
                  <span class="text-body-2 text-high-emphasis hpo-label">{{ result.label }}</span>
                  <v-chip v-if="result.original_rank !== undefined"
                    class="ml-2"
                    size="x-small"
                    color="grey-lighten-1"
                    label
                  >
                    {{ $t('resultsDisplay.originalRank') }}: #{{ result.original_rank }}
                  </v-chip>
                </div>
              </div>
              
              <!-- Score chips and Add button on the right -->
              <div class="d-flex align-center ml-auto">
                <div class="d-flex flex-row align-center gap-1 mr-2 score-chips-container">
                  <v-chip
                    class="score-chip"
                    color="primary"
                    size="x-small"
                    label
                    variant="outlined"
                  >
                    <v-icon size="x-small" start>mdi-percent</v-icon>
                    {{ (result.similarity * 100).toFixed(1) }}%
                  </v-chip>
                  
                  <v-chip
                    v-if="result.cross_encoder_score !== undefined && result.cross_encoder_score !== null"
                    class="score-chip"
                    color="secondary"
                    size="x-small"
                    label
                    variant="outlined"
                  >
                    <v-icon size="x-small" start>mdi-filter</v-icon>
                    {{ formatRerankerScore(result.cross_encoder_score) }}
                  </v-chip>
                </div>
                
                <v-btn
                  :icon="isAlreadyCollected(result.hpo_id) ? 'mdi-check-circle' : 'mdi-plus-circle'"
                  size="small"
                  :color="isAlreadyCollected(result.hpo_id) ? 'success' : 'primary'"
                  variant="text"
                  class="flex-shrink-0 add-btn"
                  @click.stop="addToCollection(result)"
                  :disabled="isAlreadyCollected(result.hpo_id)"
                  :title="isAlreadyCollected(result.hpo_id) ? $t('resultsDisplay.alreadyInCollectionTooltip', { id: result.hpo_id }) : $t('resultsDisplay.addToCollectionTooltip', { id: result.hpo_id })"
                  :aria-label="isAlreadyCollected(result.hpo_id) ? $t('resultsDisplay.alreadyInCollectionAriaLabel', { id: result.hpo_id, label: result.label }) : $t('resultsDisplay.addToCollectionAriaLabel', { id: result.hpo_id, label: result.label })"
                ></v-btn>
              </div>
            </div>
          </v-list-item-title>
          
          <template v-slot:append>
            <!-- Append content moved to subtitle for better space usage -->
          </template>
        </v-list-item>
      </v-list>
    </div>

    <!-- Text Processing Results Display -->
    <div v-else-if="resultType === 'textProcess' && responseData && responseData.processed_chunks && responseData.aggregated_hpo_terms">
      <!-- Meta Information Card -->
      <v-card class="mb-4 info-card">
        <v-card-title class="text-subtitle-1 pa-2 pa-sm-4">
          <div class="d-flex flex-wrap align-center">
            <!-- Processing Strategy -->
            <div class="info-item mr-4 mb-1">
              <v-icon color="info" class="mr-1" size="small">mdi-information</v-icon>
              <span class="model-name">
                <small class="text-caption text-medium-emphasis">{{ $t('resultsDisplay.textProcess.strategyLabel', 'Strategy') }}:</small>
                {{ responseData.meta?.request_parameters?.chunking_strategy || 'sliding_window_cleaned' }}
              </span>
            </div>
            
            <!-- Number of Chunks -->
            <div class="info-item mr-4 mb-1">
              <v-icon color="info" class="mr-1" size="small">mdi-file-document-multiple</v-icon>
              <span>
                <small class="text-caption text-medium-emphasis">{{ $t('resultsDisplay.textProcess.chunksLabel', 'Chunks') }}:</small>
                {{ responseData.processed_chunks.length }}
              </span>
            </div>
            
            <!-- Language info -->
            <div v-if="responseData.meta?.effective_language || responseData.meta?.request_parameters?.language" class="info-item mr-4 mb-1">
              <v-icon color="info" class="mr-1" size="small">mdi-translate</v-icon>
              <span>
                <small class="text-caption text-medium-emphasis">{{ $t('resultsDisplay.languageLabel') }}:</small>
                {{ responseData.meta.effective_language || responseData.meta.request_parameters?.language }}
              </span>
            </div>
          </div>
          
          <!-- Model info on separate line -->
          <div v-if="responseData.meta?.effective_retrieval_model || responseData.meta?.request_parameters?.retrieval_model_name" class="info-item mt-2">
            <v-icon color="info" class="mr-1" size="small">mdi-brain</v-icon>
            <span class="model-name">
              <small class="text-caption text-medium-emphasis">{{ $t('resultsDisplay.modelLabel') }}:</small>
              {{ displayModelName(responseData.meta.effective_retrieval_model || responseData.meta.request_parameters?.retrieval_model_name) }}
            </span>
          </div>
        </v-card-title>
      </v-card>

      <!-- Processed Chunks Section -->
      <h3 class="text-h6 my-4">{{ $t('resultsDisplay.textProcess.chunksTitle', 'Processed Chunks & Per-Chunk HPO Terms') }}</h3>
      <v-expansion-panels v-if="responseData.processed_chunks && responseData.processed_chunks.length > 0">
        <v-expansion-panel 
          v-for="chunk in responseData.processed_chunks" 
          :key="chunk.chunk_id"
          :ref="el => { if (el) chunkPanelsRef[chunk.chunk_id] = el }"
        >
          <v-expansion-panel-title>
            <div class="d-flex align-center">
              <span class="text-truncate">{{ $t('resultsDisplay.textProcess.chunkLabel', 'Chunk') }} {{ chunk.chunk_id }}: {{ chunk.text.substring(0, 50) }}...</span>
              <v-chip 
                size="small" 
                :color="chunk.status === 'negated' ? 'error' : (chunk.status === 'affirmed' ? 'success' : 'grey')" 
                class="ml-2">
                {{ chunk.status || 'unknown' }}
              </v-chip>
            </div>
          </v-expansion-panel-title>
          <v-expansion-panel-text>
            <p class="font-italic mb-2" :id="`chunk-text-${chunk.chunk_id}`">
              <template v-if="highlightedAttributions.length > 0">
                <span 
                  v-for="(segment, segmentIndex) in getHighlightedChunkSegments(chunk)" 
                  :key="segmentIndex" 
                  :class="{ 'highlighted-text-span': segment.isHighlighted }"
                >{{ segment.text }}</span>
              </template>
              <template v-else>
                "{{ chunk.text }}"
              </template>
            </p>
            <div v-if="chunk.assertion_details && chunk.assertion_details.final_status">
              <small>({{ $t('resultsDisplay.textProcess.assertionDetail', 'Assertion Method:') }} {{ chunk.assertion_details.combination_strategy }}, {{ $t('resultsDisplay.textProcess.finalStatus', 'Final Status:') }} {{ chunk.assertion_details.final_status }})</small>
            </div>
            
            <!-- Per-chunk HPO terms display -->
            <div v-if="chunk.hpo_matches && chunk.hpo_matches.length > 0" class="mt-2">
              <v-divider class="my-2"></v-divider>
              <div class="text-caption font-weight-medium mb-1">{{ $t('resultsDisplay.textProcess.chunkHPOTerms', 'HPO Terms Identified in this Chunk:') }}</div>
              <v-chip-group>
                <v-chip
                  v-for="match in chunk.hpo_matches"
                  :key="match.hpo_id"
                  size="small"
                  color="primary"
                  variant="outlined"
                  class="mr-1 mb-1"
                  :prepend-icon="'mdi-tag'"
                  :title="`${match.name} (Score: ${match.score.toFixed(2)})`"
                >
                  <a 
                    :href="`https://hpo.jax.org/browse/term/${match.hpo_id}`" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    class="hpo-link"
                    style="color: inherit;"
                  >
                    {{ match.hpo_id }}
                  </a>
                  <span class="ms-1">({{ match.score.toFixed(2) }})</span>
                </v-chip>
              </v-chip-group>
            </div>
            <div v-else class="text-caption text-medium-emphasis mt-2">
              {{ $t('resultsDisplay.textProcess.noChunkHPOTerms', 'No HPO terms were identified for this chunk.') }}
            </div>
          </v-expansion-panel-text>
        </v-expansion-panel>
      </v-expansion-panels>
      <v-alert v-else type="info">
        {{ $t('resultsDisplay.textProcess.noChunksProcessed', 'No text chunks were processed.') }}
      </v-alert>

      <!-- Aggregated HPO Terms Section -->
      <h3 class="text-h6 my-4">{{ $t('resultsDisplay.textProcess.aggregatedTitle', 'Aggregated Document Phenotypes') }}</h3>
      <v-list v-if="responseData.aggregated_hpo_terms && responseData.aggregated_hpo_terms.length > 0" 
              lines="two" class="rounded-lg mt-2">
        <v-list-item
          v-for="(term, index) in responseData.aggregated_hpo_terms"
          :key="'agg-' + index"
          class="mb-1 rounded-lg"
          :color="isAlreadyCollected(term.hpo_id) ? 'primary-lighten-5' : 'grey-lighten-5'"
          border
          density="compact"
        >
          <template v-slot:prepend>
            <v-badge
              :content="index + 1"
              color="primary"
              inline
              class="mr-2"
            ></v-badge>
          </template>
          
          <v-list-item-title class="pb-2">
            <div class="d-flex flex-wrap align-center">
              <!-- HPO ID and Label on the left -->
              <div class="flex-grow-1">
                <div class="d-flex align-center mb-1">
                  <a 
                    :href="`https://hpo.jax.org/browse/term/${term.hpo_id}`" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    class="hpo-link"
                    :title="`View ${term.hpo_id} in HPO Browser`"
                  >
                    <span class="hpo-id font-weight-bold">{{ term.hpo_id }}</span>
                    <v-icon size="x-small" class="ml-1">mdi-open-in-new</v-icon>
                  </a>
                </div>
                <div class="hpo-label mt-1">{{ term.name || term.label }}</div>
                
                <!-- Source chunks -->                
                <div v-if="term.source_chunk_ids && term.source_chunk_ids.length > 0" class="text-caption mt-1">
                  <span class="text-secondary">{{ $t('resultsDisplay.textProcess.sourceChunks', 'Source chunks') }}:</span>
                  <v-chip-group class="d-inline-flex">
                    <v-chip
                      v-for="chunkId in term.source_chunk_ids"
                      :key="chunkId"
                      size="x-small"
                      color="secondary"
                      variant="outlined"
                      class="mr-1"
                      @click="scrollToChunk(chunkId)"
                      style="cursor: pointer;"
                    >
                      #{{ chunkId }}
                    </v-chip>
                  </v-chip-group>
                  <span v-if="term.top_evidence_chunk_id" class="ml-2">
                    <span class="text-secondary">{{ $t('resultsDisplay.textProcess.topEvidence', 'Top evidence') }}:</span>
                    <v-chip
                      size="x-small"
                      color="primary"
                      variant="outlined"
                      class="ml-1"
                      @click="scrollToChunk(term.top_evidence_chunk_id)"
                      style="cursor: pointer;"
                    >
                      #{{ term.top_evidence_chunk_id }}
                    </v-chip>
                  </span>
                </div>
              </div>
              
              <!-- Score chips and Add button on the right -->
              <div class="d-flex align-center ml-auto">
                <div class="d-flex flex-row align-center gap-1 mr-2 score-chips-container">
                  <!-- Confidence score (average) -->
                  <v-chip
                    class="score-chip"
                    color="primary"
                    size="x-small"
                    variant="flat"
                    :title="$t('resultsDisplay.textProcess.confidenceTooltip', 'Average confidence score from all evidence chunks')"
                  >
                    <v-icon size="x-small" start>mdi-numeric</v-icon>
                    {{ (term.confidence * 100).toFixed(0) }}%
                  </v-chip>
                  
                  <!-- Assertion status -->
                  <v-chip
                    class="score-chip"
                    :color="getAssertionColor(term.status)"
                    size="x-small"
                    variant="flat"
                  >
                    <v-icon size="x-small" start>
                      {{ term.status === 'negated' ? 'mdi-block-helper' : 'mdi-check-circle' }}
                    </v-icon>
                    {{ term.status || 'unknown' }}
                  </v-chip>
                  
                  <!-- Evidence count -->
                  <v-chip
                    v-if="term.evidence_count > 1"
                    class="score-chip"
                    color="secondary"
                    size="x-small"
                    variant="flat"
                    :title="$t('resultsDisplay.textProcess.evidenceCountTooltip', 'Number of chunks providing evidence for this term')"
                  >
                    <v-icon size="x-small" start>mdi-file-multiple</v-icon>
                    {{ term.evidence_count }}
                  </v-chip>
                </div>
                
                <v-btn
                  :icon="isAlreadyCollected(term.hpo_id) ? 'mdi-check-circle' : 'mdi-plus-circle'"
                  size="small"
                  :color="isAlreadyCollected(term.hpo_id) ? 'success' : 'primary'"
                  variant="text"
                  class="add-btn"
                  density="comfortable"
                  @click="addToCollection({hpo_id: term.hpo_id, label: term.name || term.label}, term.status || 'affirmed')"
                  :disabled="isAlreadyCollected(term.hpo_id)"
                  :aria-label="isAlreadyCollected(term.hpo_id) ? 'Already added to collection' : 'Add to collection'"
                ></v-btn>
              </div>
            </div>
          </v-list-item-title>
        </v-list-item>
      </v-list>
      <v-alert v-else type="info" class="mb-4">
        {{ $t('resultsDisplay.textProcess.noAggregatedTerms', 'No aggregated HPO terms found.') }}
      </v-alert>
    </div>

    <!-- No Results / Empty States -->
    <div v-else-if="resultType === 'query' && responseData && responseData.results && responseData.results.length === 0">
      <v-alert color="warning" icon="mdi-alert-circle">
        {{ $t('resultsDisplay.noTermsFound') }}
      </v-alert>
    </div>
    <div v-else-if="resultType === 'textProcess' && responseData && (!responseData.processed_chunks || responseData.processed_chunks.length === 0)">
      <v-alert color="warning" icon="mdi-alert-circle">
        {{ $t('resultsDisplay.textProcess.noChunksProcessed', 'No text chunks were processed.') }}
      </v-alert>
    </div>
    <div v-else-if="error">
      <v-alert type="error" icon="mdi-alert-circle">
        <template v-if="error.userMessageKey">
          {{ $t(error.userMessageKey, error.userMessageParams) }}
        </template>
        <template v-else>
          {{ error.detail || $t('resultsDisplay.defaultError') }}
        </template>
      </v-alert>
    </div>
  </div>
</template>

<script>
import { logService } from '../services/logService'

export default {
  name: 'ResultsDisplay',
  data() {
    return {
      highlightedAttributions: [],
      chunkPanelsRef: {}
    }
  },
  props: {
    responseData: {
      type: Object,
      default: null,
      validator(value) {
        if (value) {
          logService.debug('Results data received', {
            modelUsed: value.model_used_for_retrieval,
            rerankerUsed: value.reranker_used,
            resultsCount: value.results?.length,
            language: value.language_detected,
            queryAssertionStatus: value.query_assertion_status,
            processedChunks: value.processed_chunks?.length,
            aggregatedTerms: value.aggregated_hpo_terms?.length
          });
        }
        return true;
      }
    },
    resultType: {
      type: String,
      default: 'query',
      validator: value => ['query', 'textProcess'].includes(value)
    },
    error: {
      type: Object,
      default: null
    },
    collectedPhenotypes: {
      type: Array,
      default: () => []
    }
  },
  emits: ['add-to-collection'],
  methods: {
    highlightAttributions(term) {
      if (term.text_attributions && term.text_attributions.length > 0) {
        // Format attributions for easier use in the component
        this.highlightedAttributions = term.text_attributions.map(attr => ({
          chunkId: attr.chunk_id,
          start: attr.start_char,
          end: attr.end_char,
          text: attr.matched_text_in_chunk
        }));
      }
    },
    
    clearHighlights() {
      this.highlightedAttributions = [];
    },
    
    getHighlightedChunkSegments(chunk) {
      // Find attributions for this chunk
      const chunkAttributions = this.highlightedAttributions.filter(
        attr => attr.chunkId === chunk.chunk_id
      );
      
      if (chunkAttributions.length === 0) {
        return [{ text: `"${chunk.text}"`, isHighlighted: false }];
      }
      
      // Sort attributions by start position
      chunkAttributions.sort((a, b) => a.start - b.start);
      
      const segments = [];
      let lastEnd = 0;
      
      for (const attr of chunkAttributions) {
        // Add non-highlighted segment before this attribution if needed
        if (attr.start > lastEnd) {
          segments.push({
            text: chunk.text.substring(lastEnd, attr.start),
            isHighlighted: false
          });
        }
        
        // Add highlighted segment
        segments.push({
          text: chunk.text.substring(attr.start, attr.end),
          isHighlighted: true
        });
        
        lastEnd = attr.end;
      }
      
      // Add remaining text after last attribution
      if (lastEnd < chunk.text.length) {
        segments.push({
          text: chunk.text.substring(lastEnd),
          isHighlighted: false
        });
      }
      
      return segments;
    },
    
    scrollToChunk(chunkId) {
      // Get the expansion panel element
      const panel = this.chunkPanelsRef[chunkId];
      if (panel) {
        // Open the panel if closed
        panel.expand();
        
        // Scroll to the panel
        setTimeout(() => {
          const element = document.getElementById(`chunk-text-${chunkId}`);
          if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            // Add a temporary highlight effect
            element.classList.add('flash-highlight');
            setTimeout(() => {
              element.classList.remove('flash-highlight');
            }, 1500);
          }
        }, 300); // Wait for expansion animation
      }
    },
    
    getAssertionColor(status) {
      if (!status) return 'grey';
      return status === 'negated' ? 'error' : 'success';
    },
    
    isAlreadyCollected(hpoId) {
      const isCollected = this.collectedPhenotypes.some(item => item.hpo_id === hpoId);
      logService.debug('Checking if phenotype is collected', { hpoId, isCollected });
      return isCollected;
    },
    addToCollection(phenotype, assertionStatus = 'affirmed') {
      // Convert text processing format to the expected format for collection
      const normalizedPhenotype = {
        hpo_id: phenotype.hpo_id,
        label: phenotype.name || phenotype.label, // API uses 'name' in text processing mode
        assertion_status: assertionStatus
      };
      
      logService.info('Adding phenotype to collection from results', { 
        originalPhenotype: phenotype,
        normalizedPhenotype: normalizedPhenotype
      });
      
      this.$emit('add-to-collection', normalizedPhenotype);
    },
    formatRerankerScore(score) {
      logService.debug('Formatting reranker score', { originalScore: score });
      // Different rerankers use different score ranges/meanings
      // Some return negative scores (higher/less negative = better)
      // Others return probabilities (0-1)
      let formattedScore;
      if (score < 0) {
        // For models like cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
        // that return negative scores, transform to a percentile-like display
        formattedScore = (5 + score).toFixed(1); // Transform range, e.g., -5 to 0 → 0 to 5
      } else if (score <= 1) {
        // For models returning probabilities (entailment scores)
        formattedScore = (score * 100).toFixed(1) + '%';
      } else {
        // For any other type of score
        formattedScore = score.toFixed(2);
      }
      logService.debug('Formatted reranker score', { originalScore: score, formattedScore });
      return formattedScore;
    },
    displayModelName(name) {
      logService.debug('Formatting model name for display', { originalName: name });
      // Format model names to be more display-friendly on mobile
      if (!name) {
        logService.warn('Empty model name received');
        return '';
      }
      
      // For typical model paths like org/model-name
      if (name.includes('/')) {
        const parts = name.split('/');
        return parts[parts.length - 1]; // Return just the model name without organization
      }
      
      // Shorten long model names for mobile display
      if (name.length > 25) {
        return name.substring(0, 22) + '...';
      }
      
      return name;
    }
  }
}
</script>

<style scoped>
.results-container {
  margin-bottom: 16px;
}

.hpo-id {
  font-weight: bold;
  white-space: nowrap;
}

.hpo-link {
  text-decoration: none;
  color: inherit;
  display: inline-flex;
  align-items: center;
  transition: color 0.2s;
}

.hpo-link:hover {
  color: var(--v-theme-primary);
}

.hpo-label {
  max-width: 100%;
  display: inline-block;
}

.info-card .model-name {
  word-break: break-word;
}

.info-item {
  display: flex;
  align-items: flex-start;
}

.score-chip {
  margin-right: 4px;
  height: 24px !important;
}

.add-btn {
  transform: scale(1.2);
  margin-left: 8px;
}

.score-chips-container {
  margin-top: 4px;
}

.highlighted-text-span {
  background-color: rgba(255, 235, 59, 0.5);
  border-radius: 2px;
  padding: 0 2px;
}

.flash-highlight {
  animation: flash-animation 1.5s;
}

@keyframes flash-animation {
  0%, 100% { background-color: transparent; }
  25% { background-color: rgba(255, 235, 59, 0.5); }
  75% { background-color: rgba(255, 235, 59, 0.5); }
}

/* On small screens */
@media (max-width: 600px) {
  .hpo-id {
    font-size: 1rem;
  }
  
  /* Make card titles wrap better on mobile */
  :deep(.v-card-title) {
    display: block;
    font-size: 0.875rem;
    line-height: 1.4;
    padding: 8px 12px;
  }

  .v-list-item {
    padding: 8px 12px !important;
  }

  /* Improve layout for model info on mobile */
  .info-item {
    margin-bottom: 8px;
  }
  
  .info-item:last-child {
    margin-bottom: 0;
  }
  
  .score-chips-container {
    flex-wrap: wrap;
    justify-content: flex-end;
  }
  
  .score-chip {
    font-size: 0.75rem;
    height: 22px !important;
  }
}
</style>
