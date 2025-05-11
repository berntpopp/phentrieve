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

      <v-expansion-panels>
        <v-expansion-panel v-for="(result, index) in responseData.results" :key="index">
          <v-expansion-panel-title>
            <div class="d-flex align-center">
              <span class="font-weight-bold">{{ result.hpo_id }}</span>
              <v-spacer></v-spacer>
              <v-chip
                class="mr-2"
                color="primary"
                size="small"
                label
              >
                {{ (result.similarity * 100).toFixed(1) }}%
              </v-chip>
              <v-chip
                v-if="result.cross_encoder_score !== undefined"
                class="mr-2"
                color="secondary"
                size="small"
                label
              >
                Reranked: {{ formatRerankerScore(result.cross_encoder_score) }}
              </v-chip>
              <v-chip
                v-if="result.original_rank !== undefined"
                class="mr-2"
                color="grey-lighten-1"
                size="small"
                label
              >
                Orig: #{{ result.original_rank }}
              </v-chip>
            </div>
          </v-expansion-panel-title>
          <v-expansion-panel-text>
            <div class="py-2">
              <p class="mb-2"><strong>HPO Term:</strong> {{ result.label }}</p>
              <p class="mb-0"><strong>Similarity Score:</strong> {{ result.similarity !== null && result.similarity !== undefined ? result.similarity.toFixed(4) : 'N/A' }}</p>
              <p v-if="result.cross_encoder_score !== undefined && result.cross_encoder_score !== null" class="mb-0">
                <strong>Cross-Encoder Score:</strong> {{ result.cross_encoder_score.toFixed(4) }}
              </p>
            </div>
          </v-expansion-panel-text>
        </v-expansion-panel>
      </v-expansion-panels>
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
    }
  },
  methods: {
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
