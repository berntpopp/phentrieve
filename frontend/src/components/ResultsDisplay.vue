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
                  >{{ $t('resultsDisplay.assertionLabel', 'Assertion') }}:</small
                >
                <v-chip
                  size="x-small"
                  :color="responseData.query_assertion_status === 'negated' ? 'error' : 'success'"
                  class="ml-1"
                  density="comfortable"
                >
                  {{
                    responseData.query_assertion_status === 'negated'
                      ? $t('resultsDisplay.negated', 'Negated')
                      : $t('resultsDisplay.affirmed', 'Affirmed')
                  }}
                </v-chip>
              </span>
            </div>
          </div>

          <!-- Reranker info on separate line -->
          <div v-if="responseData.reranker_used" class="info-item mt-2">
            <v-icon color="info" class="mr-1" size="small"> mdi-filter </v-icon>
            <span class="model-name">
              <small class="text-caption text-medium-emphasis"
                >{{ $t('resultsDisplay.rerankerLabel') }}:</small
              >
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
          :color="collectedPhenotypeIds.has(result.hpo_id) ? 'primary-lighten-5' : 'grey-lighten-5'"
          border
          density="compact"
        >
          <template #prepend>
            <v-badge :content="index + 1" color="primary" inline class="mr-2" />
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
                  <v-chip
                    v-if="result.original_rank !== undefined"
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
                  <SimilarityScore
                    :score="result.similarity"
                    type="similarity"
                    :decimals="2"
                    :show-animation="false"
                  />

                  <SimilarityScore
                    v-if="
                      result.cross_encoder_score !== undefined &&
                      result.cross_encoder_score !== null
                    "
                    :score="result.cross_encoder_score"
                    type="rerank"
                    :decimals="2"
                    :show-animation="false"
                  />
                </div>

                <v-btn
                  :icon="collectedPhenotypeIds.has(result.hpo_id) ? 'mdi-check-circle' : 'mdi-plus-circle'"
                  size="small"
                  :color="collectedPhenotypeIds.has(result.hpo_id) ? 'success' : 'primary'"
                  variant="text"
                  class="flex-shrink-0 add-btn"
                  :disabled="collectedPhenotypeIds.has(result.hpo_id)"
                  :title="
                    collectedPhenotypeIds.has(result.hpo_id)
                      ? $t('resultsDisplay.alreadyInCollectionTooltip', { id: result.hpo_id })
                      : $t('resultsDisplay.addToCollectionTooltip', { id: result.hpo_id })
                  "
                  :aria-label="
                    collectedPhenotypeIds.has(result.hpo_id)
                      ? $t('resultsDisplay.alreadyInCollectionAriaLabel', {
                          id: result.hpo_id,
                          label: result.label,
                        })
                      : $t('resultsDisplay.addToCollectionAriaLabel', {
                          id: result.hpo_id,
                          label: result.label,
                        })
                  "
                  @click.stop="addToCollection(result)"
                />
              </div>
            </div>
          </v-list-item-title>

          <template #append>
            <!-- Append content moved to subtitle for better space usage -->
          </template>
        </v-list-item>
      </v-list>
    </div>

    <!-- Text Processing Results Display -->
    <div
      v-else-if="
        resultType === 'textProcess' &&
        responseData &&
        responseData.processed_chunks &&
        responseData.aggregated_hpo_terms
      "
    >
      <!-- Meta Information Card -->
      <v-card class="mb-4 info-card">
        <v-card-title class="text-subtitle-1 pa-2 pa-sm-4">
          <div class="d-flex flex-wrap align-center">
            <!-- Processing Strategy -->
            <div class="info-item mr-4 mb-1">
              <v-icon color="info" class="mr-1" size="small"> mdi-information </v-icon>
              <span class="model-name">
                <small class="text-caption text-medium-emphasis"
                  >{{ $t('resultsDisplay.textProcess.strategyLabel', 'Strategy') }}:</small
                >
                {{
                  responseData.meta?.request_parameters?.chunking_strategy ||
                  'sliding_window_cleaned'
                }}
              </span>
            </div>

            <!-- Number of Chunks -->
            <div class="info-item mr-4 mb-1">
              <v-icon color="info" class="mr-1" size="small"> mdi-file-document-multiple </v-icon>
              <span>
                <small class="text-caption text-medium-emphasis"
                  >{{ $t('resultsDisplay.textProcess.chunksLabel', 'Chunks') }}:</small
                >
                {{ responseData.processed_chunks.length }}
              </span>
            </div>

            <!-- Language info -->
            <div
              v-if="
                responseData.meta?.effective_language ||
                responseData.meta?.request_parameters?.language
              "
              class="info-item mr-4 mb-1"
            >
              <v-icon color="info" class="mr-1" size="small"> mdi-translate </v-icon>
              <span>
                <small class="text-caption text-medium-emphasis"
                  >{{ $t('resultsDisplay.languageLabel') }}:</small
                >
                {{
                  responseData.meta.effective_language ||
                  responseData.meta.request_parameters?.language
                }}
              </span>
            </div>
          </div>

          <!-- Model info on separate line -->
          <div
            v-if="
              responseData.meta?.effective_retrieval_model ||
              responseData.meta?.request_parameters?.retrieval_model_name
            "
            class="info-item mt-2"
          >
            <v-icon color="info" class="mr-1" size="small"> mdi-brain </v-icon>
            <span class="model-name">
              <small class="text-caption text-medium-emphasis"
                >{{ $t('resultsDisplay.modelLabel') }}:</small
              >
              {{
                displayModelName(
                  responseData.meta.effective_retrieval_model ||
                    responseData.meta.request_parameters?.retrieval_model_name
                )
              }}
            </span>
          </div>
        </v-card-title>
      </v-card>

      <!-- Processed Chunks Section -->
      <h3 class="text-h6 my-4">
        {{ $t('resultsDisplay.textProcess.chunksTitle', 'Processed Chunks & Per-Chunk HPO Terms') }}
      </h3>
      <v-expansion-panels
        v-if="responseData.processed_chunks && responseData.processed_chunks.length > 0"
        v-model="openChunkPanels"
      >
        <v-expansion-panel
          v-for="chunk in responseData.processed_chunks"
          :key="chunk.chunk_id"
          :ref="
            (el) => {
              if (el) chunkPanelRefs[chunk.chunk_id] = el;
            }
          "
        >
          <v-expansion-panel-title>
            <div class="d-flex align-center">
              <span class="text-truncate"
                >{{ $t('resultsDisplay.textProcess.chunkLabel', 'Chunk') }} {{ chunk.chunk_id }}:
                {{ chunk.text.substring(0, 50) }}...</span
              >
              <v-chip
                size="small"
                :color="
                  chunk.status === 'negated'
                    ? 'error'
                    : chunk.status === 'affirmed'
                      ? 'success'
                      : 'grey'
                "
                class="ml-2"
              >
                {{ chunk.status || 'unknown' }}
              </v-chip>
            </div>
          </v-expansion-panel-title>
          <v-expansion-panel-text>
            <p
              :id="`chunk-text-${chunk.chunk_id}`"
              :ref="`chunk-text-${chunk.chunk_id}`"
              class="font-italic mb-2 chunk-text-displayable"
            >
              <span
                v-for="(segment, segIdx) in getHighlightedChunkSegments(chunk)"
                :key="segIdx"
                :class="{ 'highlighted-text-span': segment.isHighlighted }"
              >
                {{ segment.text }}
              </span>
            </p>
            <div v-if="chunk.assertion_details && chunk.assertion_details.final_status">
              <small
                >({{ $t('resultsDisplay.textProcess.assertionDetail', 'Assertion Method:') }}
                {{ chunk.assertion_details.combination_strategy }},
                {{ $t('resultsDisplay.textProcess.finalStatus', 'Final Status:') }}
                {{ chunk.assertion_details.final_status }})</small
              >
            </div>

            <!-- Per-chunk HPO terms display -->
            <div
              v-if="chunk.hpo_matches && chunk.hpo_matches.length > 0"
              class="mt-3 per-chunk-matches"
            >
              <h4 class="text-subtitle-2 mb-1">
                {{
                  $t('resultsDisplay.textProcess.hpoInChunkTitle', 'HPO Terms found in this Chunk:')
                }}
              </h4>
              <v-list density="compact" class="pa-0" style="background-color: transparent">
                <!-- Use transparent background for list -->
                <v-list-item
                  v-for="(match, matchIndex) in chunk.hpo_matches"
                  :key="`chunk-${chunk.chunk_id}-match-${matchIndex}`"
                  class="mb-1 pa-1"
                  variant="tonal"
                  density="compact"
                  rounded="sm"
                  color="blue-grey-lighten-5"
                >
                  <div class="d-flex justify-space-between align-center w-100">
                    <!-- Use w-100 for full width -->
                    <div class="text-caption">
                      <a
                        :href="`https://hpo.jax.org/browse/term/${match.hpo_id}`"
                        target="_blank"
                        rel="noopener noreferrer"
                        class="hpo-link"
                      >
                        <strong class="mr-1">{{ match.hpo_id }}</strong
                        >{{ match.name }}
                        <v-icon size="x-small" class="ml-1" color="primary">mdi-open-in-new</v-icon>
                      </a>
                    </div>
                    <SimilarityScore
                      :score="match.score"
                      type="similarity"
                      :decimals="2"
                      :show-animation="false"
                      class="ml-2"
                    />
                  </div>
                </v-list-item>
              </v-list>
            </div>
            <div v-else class="text-caption text-medium-emphasis mt-2">
              {{
                $t(
                  'resultsDisplay.textProcess.noChunkHPOTermsMatched',
                  'No HPO terms met the retrieval threshold for this specific chunk.'
                )
              }}
            </div>
          </v-expansion-panel-text>
        </v-expansion-panel>
      </v-expansion-panels>
      <v-alert v-else type="info">
        {{ $t('resultsDisplay.textProcess.noChunksProcessed', 'No text chunks were processed.') }}
      </v-alert>

      <!-- Aggregated HPO Terms Section -->
      <h3 class="text-h6 my-4">
        {{ $t('resultsDisplay.textProcess.aggregatedTitle', 'Aggregated Document Phenotypes') }}
      </h3>
      <v-list
        v-if="responseData.aggregated_hpo_terms && responseData.aggregated_hpo_terms.length > 0"
        lines="two"
        class="rounded-lg mt-2"
      >
        <v-list-item
          v-for="(term, index) in responseData.aggregated_hpo_terms"
          :key="'agg-' + term.hpo_id + '-' + index"
          class="mb-2 pa-3 rounded-lg custom-hpo-card"
          :color="collectedPhenotypeIds.has(term.hpo_id) ? 'blue-grey-lighten-5' : 'white'"
          elevation="1"
          border
          @mouseenter="updateHighlightedAttributions(term.text_attributions || [])"
          @mouseleave="clearHighlightedAttributions"
        >
          <div class="d-flex flex-column">
            <!-- Top Row: ID, Name, Assertion Status -->
            <div class="d-flex align-start mb-1">
              <div class="flex-grow-1">
                <a
                  :href="`https://hpo.jax.org/browse/term/${term.hpo_id}`"
                  target="_blank"
                  rel="noopener noreferrer"
                  class="hpo-link"
                >
                  <span class="hpo-id font-weight-bold text-primary text-body-1">{{
                    term.hpo_id
                  }}</span>
                  <v-icon size="x-small" class="ml-1" color="primary">mdi-open-in-new</v-icon>
                </a>
              </div>
              <v-chip
                v-if="term.status && term.status !== 'unknown'"
                size="small"
                :color="getAssertionColor(term.status)"
                class="text-uppercase ml-2"
                label
                variant="flat"
                density="comfortable"
              >
                {{
                  $t(
                    `queryInterface.phenotypeCollection.assertionStatus.${term.status}`,
                    term.status
                  )
                }}
              </v-chip>
            </div>

            <!-- Middle Row: HPO Label -->
            <div class="text-body-1 text-high-emphasis hpo-label mb-2">
              {{ term.name || term.label }}
            </div>

            <!-- Bottom Row: Scores, Evidence Details, Add Button -->
            <div class="d-flex align-center justify-space-between flex-wrap">
              <div class="d-flex align-center flex-wrap ga-1">
                <!-- ga-1 for gap -->
                <SimilarityScore
                  :score="term.confidence"
                  type="confidence"
                  :decimals="2"
                  :show-animation="false"
                />
                <div
                  v-if="
                    term.max_score_from_evidence &&
                    term.max_score_from_evidence.toFixed(2) !== term.confidence.toFixed(2)
                  "
                  class="d-flex align-center"
                >
                  <v-icon size="small" class="mr-1 text-medium-emphasis">
                    mdi-arrow-up-bold-hexagon-outline
                  </v-icon>
                  <span class="text-caption text-medium-emphasis mr-1">Max:</span>
                  <SimilarityScore
                    :score="term.max_score_from_evidence"
                    type="similarity"
                    :decimals="2"
                    :show-animation="false"
                  />
                </div>
                <v-chip
                  v-if="term.source_chunk_ids && term.source_chunk_ids.length"
                  size="small"
                  label
                  variant="tonal"
                  style="cursor: pointer"
                  @click="scrollToChunk(term.top_evidence_chunk_id || term.source_chunk_ids[0])"
                >
                  <v-icon start size="small"> mdi-text-box-search-outline </v-icon>
                  {{ $t('resultsDisplay.textProcess.evidenceFromChunksShort', 'Chunks:') }} #{{
                    term.source_chunk_ids.join(', #')
                  }}
                  <v-tooltip activator="parent" location="top">
                    {{
                      $t(
                        'resultsDisplay.textProcess.evidenceTooltip',
                        'Source chunks. Click to see top evidence chunk.'
                      )
                    }}
                  </v-tooltip>
                </v-chip>
                <v-chip
                  v-if="term.source_chunk_ids && term.source_chunk_ids.length > 0"
                  size="small"
                  label
                  variant="tonal"
                >
                  <v-icon start size="small"> mdi-pound </v-icon>
                  {{ term.source_chunk_ids.length }}
                  {{ $t('resultsDisplay.textProcess.hitsText', 'hits') }}
                  <v-tooltip activator="parent" location="top">
                    {{
                      $t(
                        'resultsDisplay.textProcess.evidenceCountTooltip',
                        'Number of chunks providing evidence'
                      )
                    }}
                  </v-tooltip>
                </v-chip>
              </div>

              <v-btn
                :icon="collectedPhenotypeIds.has(term.hpo_id) ? 'mdi-check-circle' : 'mdi-plus-circle'"
                size="small"
                :color="collectedPhenotypeIds.has(term.hpo_id) ? 'success' : 'primary'"
                variant="text"
                class="flex-shrink-0 add-btn"
                :disabled="collectedPhenotypeIds.has(term.hpo_id)"
                :title="
                  collectedPhenotypeIds.has(term.hpo_id)
                    ? $t('resultsDisplay.alreadyInCollectionTooltip', { id: term.hpo_id })
                    : $t('resultsDisplay.addToCollectionTooltip', { id: term.hpo_id })
                "
                @click.stop="
                  addToCollection(
                    { hpo_id: term.hpo_id, name: term.name, label: term.name || term.label },
                    term.status || 'affirmed'
                  )
                "
              />
            </div>
          </div>
        </v-list-item>
      </v-list>
      <v-alert v-else type="info" class="mb-4">
        {{ $t('resultsDisplay.textProcess.noAggregatedTerms', 'No aggregated HPO terms found.') }}
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
        (!responseData.processed_chunks || responseData.processed_chunks.length === 0)
      "
    >
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
import { logService } from '../services/logService';
import SimilarityScore from './SimilarityScore.vue';

export default {
  name: 'ResultsDisplay',
  components: {
    SimilarityScore,
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
            aggregatedTerms: value.aggregated_hpo_terms?.length,
          });
        }
        return true;
      },
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
  },
  emits: ['add-to-collection'],
  data() {
    return {
      highlightedAttributions: [],
      chunkPanelRefs: {},
      openChunkPanels: [],
      modelNameCache: new Map(), // Cache for formatted model names (performance optimization)
    };
  },
  computed: {
    // Create a Set of collected phenotype IDs for O(1) lookup performance
    // This prevents reactive dependency issues when checking collection status in templates
    collectedPhenotypeIds() {
      return new Set(this.collectedPhenotypes.map((item) => item.hpo_id));
    },
  },
  methods: {
    updateHighlightedAttributions(attributions) {
      this.highlightedAttributions = attributions.map((attr) => ({
        chunkId: attr.chunk_id, // Assumes API sends 1-based chunk_id
        start: attr.start_char,
        end: attr.end_char,
        text: attr.matched_text_in_chunk,
      }));
    },

    clearHighlightedAttributions() {
      this.highlightedAttributions = [];
    },

    highlightAttributions(term) {
      if (term.text_attributions && term.text_attributions.length > 0) {
        // Format attributions for easier use in the component
        this.highlightedAttributions = term.text_attributions.map((attr) => ({
          chunkId: attr.chunk_id,
          start: attr.start_char,
          end: attr.end_char,
          text: attr.matched_text_in_chunk,
        }));
      }
    },

    clearHighlights() {
      this.highlightedAttributions = [];
    },

    getHighlightedChunkSegments(chunk) {
      // Find attributions for this chunk
      const chunkAttributions = this.highlightedAttributions.filter(
        (attr) => attr.chunkId === chunk.chunk_id
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
            isHighlighted: false,
          });
        }

        // Add highlighted segment
        segments.push({
          text: chunk.text.substring(attr.start, attr.end),
          isHighlighted: true,
        });

        lastEnd = attr.end;
      }

      // Add remaining text after last attribution
      if (lastEnd < chunk.text.length) {
        segments.push({
          text: chunk.text.substring(lastEnd),
          isHighlighted: false,
        });
      }

      return segments;
    },

    scrollToChunk(chunkId) {
      if (!chunkId) return;

      // Handle array of chunk IDs - use the first one
      if (Array.isArray(chunkId)) {
        chunkId = chunkId[0];
      }

      logService.debug(`Attempting to scroll to and open chunk ID: ${chunkId}`);

      const panelComponent = this.chunkPanelRefs[chunkId];
      if (panelComponent && panelComponent.$el) {
        panelComponent.$el.scrollIntoView({ behavior: 'smooth', block: 'center' });

        // Open the panel if closed
        const panelValueToOpen = chunkId - 1; // Assuming panel value is its 0-based index
        if (this.openChunkPanels === undefined) this.openChunkPanels = []; // Initialize if not array
        if (!Array.isArray(this.openChunkPanels)) this.openChunkPanels = [this.openChunkPanels]; // Ensure array for multiple

        if (!this.openChunkPanels.includes(panelValueToOpen)) {
          // If multiple panels can be open:
          this.openChunkPanels.push(panelValueToOpen);
        }

        // Flash highlight effect
        setTimeout(() => {
          const textDisplayEl = this.$refs[`chunk-text-${chunkId}`];
          if (textDisplayEl && textDisplayEl[0]) {
            textDisplayEl[0].classList.add('flash-highlight');
            setTimeout(() => {
              if (textDisplayEl && textDisplayEl[0])
                textDisplayEl[0].classList.remove('flash-highlight');
            }, 1500);
          }
        }, 300); // Small delay to allow panel to open
      } else {
        logService.warn(`Panel ref for chunk ID ${chunkId} not found.`);
      }
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
    formatRerankerScore(score) {
      logService.debug('Formatting reranker score', { originalScore: score });
      // Different rerankers use different score ranges/meanings
      // Some return negative scores (higher/less negative = better)
      // Others return probabilities (0-1)
      // NOTE: This function is kept for backward compatibility but is largely replaced by SimilarityScore component
      let formattedScore;
      if (score < 0) {
        // For models like cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
        // that return negative scores, transform to a 0-5 scale
        formattedScore = (5 + score).toFixed(1); // Transform range, e.g., -5 to 0 → 0 to 5
      } else if (score <= 1) {
        // For models returning probabilities (entailment scores) - display as decimal
        formattedScore = score.toFixed(2);
      } else {
        // For any other type of score
        formattedScore = score.toFixed(2);
      }
      logService.debug('Formatted reranker score', { originalScore: score, formattedScore });
      return formattedScore;
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

.flash-highlight {
  animation: flashHighlightAnimation 0.75s 2 ease-in-out;
}

@keyframes flashHighlightAnimation {
  0%,
  100% {
    background-color: transparent;
  }
  50% {
    background-color: rgba(var(--v-theme-primary), 0.15);
  }
}

.highlighted-text-span {
  background-color: rgba(255, 236, 179, 0.8); /* Vuetify yellow lighten-3 with opacity */
  border-radius: 3px;
  padding: 0.5px 2px;
  box-shadow: 0 0 3px rgba(255, 210, 50, 0.5);
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
