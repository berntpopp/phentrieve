<template>
  <div class="results-container">
    <div v-if="responseData && responseData.results && responseData.results.length > 0">
      <v-card class="mb-4">
        <v-card-title class="text-subtitle-1">
          <v-icon color="info" class="mr-2">mdi-information</v-icon>
          Query processed with model: {{ responseData.model_used_for_retrieval }}
          <span v-if="responseData.reranker_used">
            <br>
            <v-icon color="info" class="mr-2">mdi-filter</v-icon>
            Results reranked with: {{ responseData.reranker_used }}
          </span>
          <span v-if="responseData.language_detected">
            <br>
            <v-icon color="info" class="mr-2">mdi-translate</v-icon>
            Language detected: {{ responseData.language_detected }}
          </span>
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
          
          <v-list-item-title class="font-weight-bold d-flex align-center">
            <div class="d-flex align-center justify-space-between" style="width: 100%">
              <div>
                {{ result.hpo_id }}
                <span class="text-body-2 ml-2 text-grey-darken-3">{{ result.label }}</span>
              </div>
              <v-btn
                :icon="isAlreadyCollected(result.hpo_id) ? 'mdi-check-circle' : 'mdi-plus-circle'"
                size="small"
                :color="isAlreadyCollected(result.hpo_id) ? 'success' : 'primary'"
                variant="text"
                class="ml-2"
                @click.stop="addToCollection(result)"
                :disabled="isAlreadyCollected(result.hpo_id)"
                :title="isAlreadyCollected(result.hpo_id) ? result.hpo_id + ' already in collection' : 'Add ' + result.hpo_id + ' to collection'"
                :aria-label="isAlreadyCollected(result.hpo_id) ? `HPO term ${result.hpo_id} (${result.label}) is in collection` : `Add ${result.label} (${result.hpo_id}) to collection`"
              ></v-btn>
            </div>
          </v-list-item-title>
          
          <v-list-item-subtitle>
            <div class="d-flex align-center mt-1">
              <span v-if="result.original_rank !== undefined" class="text-caption text-grey-darken-3 mr-3">
                Original rank: #{{ result.original_rank }}
              </span>
            </div>
          </v-list-item-subtitle>
          
          <template v-slot:append>
            <div class="d-flex flex-column align-end">
              <v-chip
                class="mb-1"
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
                color="secondary"
                size="small"
                label
                variant="outlined"
              >
                <v-icon size="x-small" start>mdi-filter</v-icon>
                {{ formatRerankerScore(result.cross_encoder_score) }}
              </v-chip>
            </div>
          </template>
        </v-list-item>
      </v-list>
    </div>
    <div v-else-if="responseData && responseData.results && responseData.results.length === 0">
      <v-alert color="warning" icon="mdi-alert-circle">
        No HPO terms found matching your query with the current threshold.
        Try lowering the similarity threshold or rewording your query.
      </v-alert>
    </div>
    <div v-else-if="error">
      <v-alert color="error" icon="mdi-alert">
        {{ error.detail || "An error occurred while processing your query." }}
      </v-alert>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ResultsDisplay',
  props: {
    responseData: {
      type: Object,
      default: null
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
      return this.collectedPhenotypes.some(item => item.hpo_id === hpoId);
    },
    addToCollection(phenotype) {
      this.$emit('add-to-collection', phenotype);
    },
    formatRerankerScore(score) {
      // Different rerankers use different score ranges/meanings
      // Some return negative scores (higher/less negative = better)
      // Others return probabilities (0-1)
      if (score < 0) {
        // For models like cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
        // that return negative scores, transform to a percentile-like display
        return (5 + score).toFixed(1); // Transform range, e.g., -5 to 0 → 0 to 5
      } else if (score <= 1) {
        // For models returning probabilities (entailment scores)
        return (score * 100).toFixed(1) + '%';
      } else {
        // For any other type of score
        return score.toFixed(2);
      }
    }
  }
}
</script>

<style scoped>
.results-container {
  margin-bottom: 16px;
}
</style>
