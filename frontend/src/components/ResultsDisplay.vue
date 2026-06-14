<template>
  <div class="results-container">
    <!-- Regular Query Results Display -->
    <div
      v-if="
        resultType === 'query' &&
        responseData &&
        responseData.results &&
        responseData.results.length > 0
      "
    >
      <v-card class="mb-4 info-card">
        <v-card-title class="text-subtitle-1 pa-2 pa-sm-4">
          <div class="d-flex flex-wrap align-center">
            <!-- Model info - always present -->
            <div class="info-item mr-4 mb-1">
              <v-icon color="info" class="mr-1" size="small"> mdi-information </v-icon>
              <span class="model-name">
                <small class="text-caption text-medium-emphasis"
                  >{{ $t('resultsDisplay.modelLabel') }}:</small
                >
                {{ displayModelName(responseData.model_used_for_retrieval) }}
              </span>
            </div>

            <!-- Language info -->
            <div v-if="responseData.language_detected" class="info-item mr-4 mb-1">
              <v-icon color="info" class="mr-1" size="small"> mdi-translate </v-icon>
              <span>
                <small class="text-caption text-medium-emphasis"
                  >{{ $t('resultsDisplay.languageLabel') }}:</small
                >
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
                {{
                  responseData.query_assertion_status === 'negated'
                    ? 'mdi-block-helper'
                    : 'mdi-check-circle'
                }}
              </v-icon>
              <span>
                <small class="text-caption text-medium-emphasis"
                  >{{ $t('resultsDisplay.assertionLabel') }}:</small
                >
                <v-chip
                  size="x-small"
                  :color="responseData.query_assertion_status === 'negated' ? 'error' : 'success'"
                  class="ml-1"
                  density="comfortable"
                >
                  {{
                    responseData.query_assertion_status === 'negated'
                      ? $t('resultsDisplay.negated')
                      : $t('resultsDisplay.affirmed')
                  }}
                </v-chip>
              </span>
            </div>
          </div>
        </v-card-title>
      </v-card>

      <v-list lines="two" class="rounded-lg mt-2">
        <ResultItem
          v-for="(result, index) in responseData.results"
          :key="`${result.hpo_id}-${index}`"
          :result="result"
          :rank="index + 1"
          :is-collected="collectedPhenotypeIds.has(result.hpo_id)"
          @add-to-collection="addToCollection($event)"
        />
      </v-list>
    </div>

    <div
      v-else-if="
        resultType === 'textProcess' &&
        responseData &&
        responseData.aggregated_hpo_terms &&
        !hasValidTurnId
      "
    >
      <v-alert type="error" icon="mdi-alert-circle">
        {{ $t('resultsDisplay.defaultError') }}
      </v-alert>
    </div>

    <!-- No Results / Empty States -->
    <div
      v-else-if="
        resultType === 'query' &&
        responseData &&
        responseData.results &&
        responseData.results.length === 0
      "
    >
      <v-alert color="warning" icon="mdi-alert-circle">
        {{ $t('resultsDisplay.noTermsFound') }}
      </v-alert>
    </div>
    <div
      v-else-if="
        resultType === 'textProcess' &&
        responseData &&
        processedChunks.length === 0 &&
        extractionBackend !== 'llm'
      "
    >
      <v-alert color="warning" icon="mdi-alert-circle">
        {{ $t('resultsDisplay.textProcess.noChunksProcessed') }}
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
        <div v-if="showLoginNudge" class="mt-2">
          <v-btn
            variant="text"
            size="small"
            color="primary"
            prepend-icon="mdi-login"
            @click="openAuthDialog"
          >
            {{ $t('auth.nudge.loginForMore') }}
          </v-btn>
        </div>
      </v-alert>
    </div>
  </div>
</template>

<script>
import { logService } from '../services/logService';
import ResultItem from './ResultItem.vue';
import { HPO_TERM_URL } from '../constants/urls';
import { useAuthStore } from '../stores/auth';

export default {
  name: 'ResultsDisplay',
  components: {
    ResultItem,
  },
  props: {
    responseData: {
      type: Object,
      default: null,
      validator: (value) => value === null || typeof value === 'object',
    },
    resultType: {
      type: String,
      default: 'query',
      validator: (value) => ['query', 'textProcess'].includes(value),
    },
    error: {
      type: Object,
      default: null,
    },
    collectedPhenotypes: {
      type: Array,
      default: () => [],
    },
    turnId: {
      type: String,
      default: '',
    },
  },
  emits: ['add-to-collection', 'add-all-to-collection'],
  data() {
    return {
      modelNameCache: new Map(), // Cache for formatted model names (performance optimization)
      expandedQueryResults: new Set(), // Track which query results have details expanded
      expandedAggregatedTerms: new Set(), // Track which aggregated terms have details expanded
    };
  },
  computed: {
    // Create a Set of collected phenotype IDs for O(1) lookup performance
    // This prevents reactive dependency issues when checking collection status in templates
    collectedPhenotypeIds() {
      return new Set(this.collectedPhenotypes.map((item) => item.hpo_id));
    },
    processedChunks() {
      return this.responseData?.processed_chunks ?? [];
    },
    extractionBackend() {
      return this.responseData?.meta?.extraction_backend ?? 'standard';
    },
    quotaRemaining() {
      return this.responseData?.meta?.quota_remaining;
    },
    quotaLimit() {
      return this.responseData?.meta?.quota_limit;
    },
    showQuotaNotice() {
      return (
        this.extractionBackend === 'llm' && this.quotaRemaining != null && this.quotaLimit != null
      );
    },
    hasValidTurnId() {
      return typeof this.turnId === 'string' && this.turnId.length > 0;
    },
    showLoginNudge() {
      // Encourage anonymous users to sign in when they hit the LLM quota.
      return (
        this.error?.userMessageKey === 'errors.api.llmQuotaExceeded' &&
        !useAuthStore().isAuthenticated
      );
    },
  },
  methods: {
    openAuthDialog() {
      window.dispatchEvent(new CustomEvent('phentrieve:open-auth', { detail: { mode: 'login' } }));
    },
    toggleQueryResultDetails(index) {
      if (this.expandedQueryResults.has(index)) {
        this.expandedQueryResults.delete(index);
      } else {
        this.expandedQueryResults.add(index);
      }
      // Force reactivity update
      this.expandedQueryResults = new Set(this.expandedQueryResults);
    },
    toggleAggregatedTermDetails(hpoId) {
      if (this.expandedAggregatedTerms.has(hpoId)) {
        this.expandedAggregatedTerms.delete(hpoId);
      } else {
        this.expandedAggregatedTerms.add(hpoId);
      }
      // Force reactivity update
      this.expandedAggregatedTerms = new Set(this.expandedAggregatedTerms);
    },
    hasDetails(result) {
      return (
        (result.definition && result.definition.trim()) ||
        (result.synonyms && result.synonyms.length > 0)
      );
    },
    getAssertionColor(status) {
      if (!status) return 'grey';
      return status === 'negated' ? 'error' : 'success';
    },

    isAlreadyCollected(hpoId) {
      // Use the computed Set for O(1) lookup performance
      return this.collectedPhenotypeIds.has(hpoId);
    },
    addToCollection(phenotype, assertionStatus = 'affirmed') {
      // Convert text processing format to the expected format for collection
      const normalizedPhenotype = {
        hpo_id: phenotype.hpo_id,
        label: phenotype.name || phenotype.label, // API uses 'name' in text processing mode
        assertion_status: assertionStatus,
      };

      // Log the assertion status being used
      logService.debug('Setting assertion status for collection item', {
        hpoId: phenotype.hpo_id,
        originalStatus: assertionStatus,
        finalStatus: normalizedPhenotype.assertion_status,
      });

      logService.info('Adding phenotype to collection from results', {
        originalPhenotype: phenotype,
        normalizedPhenotype: normalizedPhenotype,
      });

      this.$emit('add-to-collection', normalizedPhenotype);
    },
    hpoTermUrl(hpoId) {
      return HPO_TERM_URL(hpoId);
    },
    displayModelName(name) {
      // Check cache first (performance optimization - prevents excessive re-computation)
      if (this.modelNameCache.has(name)) {
        return this.modelNameCache.get(name);
      }

      logService.debug('Formatting model name for display', { originalName: name });
      // Format model names to be more display-friendly on mobile
      if (!name) {
        logService.warn('Empty model name received');
        return '';
      }

      let formatted;
      // For typical model paths like org/model-name
      if (name.includes('/')) {
        const parts = name.split('/');
        formatted = parts[parts.length - 1]; // Return just the model name without organization
      }
      // Shorten long model names for mobile display
      else if (name.length > 25) {
        formatted = name.substring(0, 22) + '...';
      } else {
        formatted = name;
      }

      // Cache the result
      this.modelNameCache.set(name, formatted);
      return formatted;
    },
  },
};
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

.highlighted-text-span {
  background-color: rgba(255, 236, 179, 0.8); /* Vuetify yellow lighten-3 with opacity */
  border-radius: 3px;
  padding: 0.5px 2px;
  box-shadow: 0 0 3px rgba(255, 210, 50, 0.5);
}

.details-section {
  border-top: 1px solid rgba(0, 0, 0, 0.12);
  background-color: rgba(0, 0, 0, 0.02);
  padding: 8px 12px;
  border-radius: 4px;
}

.definition-text {
  line-height: 1.6;
  color: rgba(0, 0, 0, 0.87);
}

.synonym-chip {
  font-size: 0.75rem;
}

.custom-hpo-card {
  transition:
    box-shadow 0.2s,
    transform 0.2s;
}

.custom-hpo-card:hover {
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
  transform: translateY(-1px);
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
