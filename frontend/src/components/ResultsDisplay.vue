<template>
  <div class="results-container">
    <div v-if="responseData && responseData.results && responseData.results.length > 0">
      <v-card class="mb-4 info-card">
        <v-card-title class="text-subtitle-1 pa-2 pa-sm-4">
          <div class="info-item">
            <v-icon color="info" class="mr-2" size="small">mdi-information</v-icon>
            <span class="model-name">
              <small class="text-caption d-block d-sm-inline text-medium-emphasis">{{ $t('resultsDisplay.modelLabel') }}:</small>
              {{ displayModelName(responseData.model_used_for_retrieval) }}
            </span>
          </div>
          
          <div v-if="responseData.reranker_used" class="info-item mt-2">
            <v-icon color="info" class="mr-2" size="small">mdi-filter</v-icon>
            <span class="model-name">
              <small class="text-caption d-block d-sm-inline text-medium-emphasis">{{ $t('resultsDisplay.rerankerLabel') }}:</small>
              {{ displayModelName(responseData.reranker_used) }}
            </span>
          </div>
          
          <div v-if="responseData.language_detected" class="info-item mt-2">
            <v-icon color="info" class="mr-2" size="small">mdi-translate</v-icon>
            <span>
              <small class="text-caption d-block d-sm-inline text-medium-emphasis">{{ $t('resultsDisplay.languageLabel') }}:</small>
              {{ responseData.language_detected }}
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
          
          <v-list-item-title class="font-weight-bold pb-1">
            <div class="d-flex align-center justify-space-between">
              <span class="hpo-id">{{ result.hpo_id }}</span>
              <v-btn
                :icon="isAlreadyCollected(result.hpo_id) ? 'mdi-check-circle' : 'mdi-plus-circle'"
                size="small"
                :color="isAlreadyCollected(result.hpo_id) ? 'success' : 'primary'"
                variant="text"
                class="ml-2 flex-shrink-0 add-btn"
                @click.stop="addToCollection(result)"
                :disabled="isAlreadyCollected(result.hpo_id)"
                :title="isAlreadyCollected(result.hpo_id) ? $t('resultsDisplay.alreadyInCollectionTooltip', { id: result.hpo_id }) : $t('resultsDisplay.addToCollectionTooltip', { id: result.hpo_id })"
                :aria-label="isAlreadyCollected(result.hpo_id) ? $t('resultsDisplay.alreadyInCollectionAriaLabel', { id: result.hpo_id, label: result.label }) : $t('resultsDisplay.addToCollectionAriaLabel', { id: result.hpo_id, label: result.label })"
              ></v-btn>
            </div>
            <div class="d-block mt-1">
              <span class="text-body-2 text-high-emphasis hpo-label">{{ result.label }}</span>
            </div>
          </v-list-item-title>
          
          <v-list-item-subtitle class="d-flex flex-wrap justify-space-between align-center">
            <div class="d-flex align-center mt-1">
              <span v-if="result.original_rank !== undefined" class="text-caption text-high-emphasis mr-3">
                {{ $t('resultsDisplay.originalRank') }}: #{{ result.original_rank }}
              </span>
            </div>

            <div class="d-flex flex-row align-center gap-2 mt-2 score-chips-container">
              <v-chip
                class="score-chip"
                color="primary"
                size="small"
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
                size="small"
                label
                variant="outlined"
              >
                <v-icon size="x-small" start>mdi-filter</v-icon>
                {{ formatRerankerScore(result.cross_encoder_score) }}
              </v-chip>
            </div>
          </v-list-item-subtitle>
          
          <template v-slot:append>
            <!-- Append content moved to subtitle for better space usage -->
          </template>
        </v-list-item>
      </v-list>
    </div>
    <div v-else-if="responseData && responseData.results && responseData.results.length === 0">
      <v-alert color="warning" icon="mdi-alert-circle">
        {{ $t('resultsDisplay.noTermsFound') }}
      </v-alert>
    </div>
    <div v-else-if="error">
      <v-alert color="error" icon="mdi-alert">
        {{ error.detail || $t('resultsDisplay.defaultError') }}
      </v-alert>
    </div>
  </div>
</template>

<script>
import { logService } from '../services/logService'

export default {
  name: 'ResultsDisplay',
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
            language: value.language_detected
          });
        }
        return true;
      }
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
    isAlreadyCollected(hpoId) {
      const isCollected = this.collectedPhenotypes.some(item => item.hpo_id === hpoId);
      logService.debug('Checking if phenotype is collected', { hpoId, isCollected });
      return isCollected;
    },
    addToCollection(phenotype) {
      logService.info('Adding phenotype to collection from results', { phenotype });
      this.$emit('add-to-collection', phenotype);
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
