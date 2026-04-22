<template>
  <div>
    <h3 class="text-h6 my-4">
      {{ $t('resultsDisplay.textProcess.aggregatedTitle', 'Aggregated Document Phenotypes') }}
    </h3>
    <v-list v-if="terms && terms.length > 0" lines="two" class="rounded-lg mt-2">
      <v-list-item
        v-for="(term, index) in terms"
        :key="'agg-' + term.hpo_id + '-' + index"
        class="mb-2 pa-3 rounded-lg custom-hpo-card"
        :color="isCollected(term.hpo_id) ? 'blue-grey-lighten-5' : 'white'"
        elevation="1"
        border
        @mouseenter="$emit('highlight-attributions', term.text_attributions || [])"
        @mouseleave="$emit('clear-attributions')"
      >
        <div class="d-flex flex-column">
          <!-- Top Row: ID, Name, Assertion Status -->
          <div class="d-flex align-start mb-1">
            <div class="flex-grow-1 d-flex align-center">
              <a
                :href="hpoTermUrl(term.hpo_id)"
                target="_blank"
                rel="noopener noreferrer"
                class="hpo-link"
              >
                <span class="hpo-id font-weight-bold text-primary text-body-1">{{
                  term.hpo_id
                }}</span>
                <v-icon size="x-small" class="ml-1" color="primary">mdi-open-in-new</v-icon>
              </a>
              <v-btn
                v-if="hasDetails(term)"
                variant="text"
                size="x-small"
                icon
                density="compact"
                class="ml-1"
                @click.stop="toggleDetails(term.hpo_id)"
              >
                <v-icon size="small">
                  {{ expandedTerms.has(term.hpo_id) ? 'mdi-chevron-up' : 'mdi-information' }}
                </v-icon>
                <v-tooltip
                  activator="parent"
                  location="top"
                  :content-props="{
                    'aria-label': expandedTerms.has(term.hpo_id)
                      ? $t('resultsDisplay.hideDetails', 'Hide details')
                      : $t('resultsDisplay.showDetails', 'Show details'),
                  }"
                >
                  {{
                    expandedTerms.has(term.hpo_id)
                      ? $t('resultsDisplay.hideDetails', 'Hide details')
                      : $t('resultsDisplay.showDetails', 'Show details')
                  }}
                </v-tooltip>
              </v-btn>
            </div>
            <v-chip
              v-if="term.status && term.status !== 'unknown'"
              size="small"
              :color="term.status === 'negated' ? 'error' : 'success'"
              class="text-uppercase ml-2"
              label
              variant="flat"
              density="comfortable"
            >
              {{ $t(assertionStatusLabel(term.status)) }}
            </v-chip>
          </div>

          <!-- Middle Row: HPO Label -->
          <div class="text-body-1 text-high-emphasis hpo-label mb-2">
            {{ term.name || term.label }}
          </div>

          <!-- Bottom Row: Scores, Evidence Details, Add Button -->
          <div class="d-flex align-center justify-space-between flex-wrap">
            <div class="d-flex align-center flex-wrap ga-1">
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
                @click="
                  $emit('scroll-to-chunk', term.top_evidence_chunk_id || term.source_chunk_ids[0])
                "
              >
                <v-icon start size="small"> mdi-text-box-search-outline </v-icon>
                {{ $t('resultsDisplay.textProcess.evidenceFromChunksShort', 'Chunks:') }} #{{
                  term.source_chunk_ids.join(', #')
                }}
                <v-tooltip
                  activator="parent"
                  location="top"
                  :content-props="{
                    'aria-label': $t(
                      'resultsDisplay.textProcess.evidenceTooltip',
                      'Source chunks. Click to see top evidence chunk.'
                    ),
                  }"
                >
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
                <v-tooltip
                  activator="parent"
                  location="top"
                  :content-props="{
                    'aria-label': $t(
                      'resultsDisplay.textProcess.evidenceCountTooltip',
                      'Number of chunks providing evidence'
                    ),
                  }"
                >
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
              :icon="isCollected(term.hpo_id) ? 'mdi-check-circle' : 'mdi-plus-circle'"
              size="small"
              :color="isCollected(term.hpo_id) ? 'success' : 'primary'"
              variant="text"
              class="flex-shrink-0 add-btn"
              :disabled="isCollected(term.hpo_id)"
              :title="
                isCollected(term.hpo_id)
                  ? $t('resultsDisplay.alreadyInCollectionTooltip', { id: term.hpo_id })
                  : $t('resultsDisplay.addToCollectionTooltip', { id: term.hpo_id })
              "
              @click.stop="
                $emit(
                  'add-to-collection',
                  { hpo_id: term.hpo_id, name: term.name, label: term.name || term.label },
                  term.status || 'affirmed'
                )
              "
            />
          </div>

          <!-- Expandable Details Section for Aggregated Terms -->
          <v-expand-transition>
            <div
              v-if="expandedTerms.has(term.hpo_id) && hasDetails(term)"
              class="mt-3 pt-3 details-section"
            >
              <div v-if="term.definition && term.definition.trim()" class="mb-3">
                <div class="text-caption text-medium-emphasis mb-1 font-weight-bold">
                  {{ $t('resultsDisplay.definitionLabel', 'Definition') }}:
                </div>
                <div class="text-body-2 definition-text">
                  {{ term.definition }}
                </div>
              </div>
              <div v-if="term.synonyms && term.synonyms.length > 0" class="mb-2">
                <div class="text-caption text-medium-emphasis mb-1 font-weight-bold">
                  {{ $t('resultsDisplay.synonymsLabel', 'Synonyms') }}:
                </div>
                <div class="d-flex flex-wrap ga-1">
                  <v-chip
                    v-for="(synonym, synIdx) in term.synonyms"
                    :key="synIdx"
                    size="small"
                    variant="tonal"
                    color="blue-grey"
                    class="synonym-chip"
                  >
                    {{ synonym }}
                  </v-chip>
                </div>
              </div>
            </div>
          </v-expand-transition>
        </div>
      </v-list-item>
    </v-list>
    <v-alert v-else type="info" class="mb-4">
      {{ $t('resultsDisplay.textProcess.noAggregatedTerms', 'No aggregated HPO terms found.') }}
    </v-alert>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import SimilarityScore from './SimilarityScore.vue';
import { HPO_TERM_URL } from '../constants/urls';

const props = defineProps({
  terms: { type: Array, default: () => [] },
  collectedPhenotypeIds: { type: Set, default: () => new Set() },
});

defineEmits([
  'add-to-collection',
  'highlight-attributions',
  'clear-attributions',
  'scroll-to-chunk',
]);

const expandedTerms = ref(new Set());

function hpoTermUrl(hpoId) {
  return HPO_TERM_URL(hpoId);
}

function hasDetails(term) {
  return (term.definition && term.definition.trim()) || (term.synonyms && term.synonyms.length > 0);
}

function isCollected(hpoId) {
  return props.collectedPhenotypeIds?.has(hpoId) || false;
}

function toggleDetails(hpoId) {
  if (expandedTerms.value.has(hpoId)) {
    expandedTerms.value.delete(hpoId);
  } else {
    expandedTerms.value.add(hpoId);
  }
  expandedTerms.value = new Set(expandedTerms.value);
}

function normalizeAssertionStatus(status) {
  if (status === 'present') {
    return 'affirmed';
  }

  if (status === 'absent') {
    return 'negated';
  }

  if (status === 'affirmed' || status === 'negated' || status === 'unknown') {
    return status;
  }

  return 'unknown';
}

function assertionStatusLabel(status) {
  const normalizedStatus = normalizeAssertionStatus(status);

  if (normalizedStatus === 'affirmed') {
    return 'queryInterface.phenotypeCollection.assertionStatus.affirmed';
  }

  if (normalizedStatus === 'negated') {
    return 'queryInterface.phenotypeCollection.assertionStatus.negated';
  }

  return 'queryInterface.phenotypeCollection.assertionStatus.unknown';
}
</script>

<style scoped>
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

.add-btn {
  transform: scale(1.2);
  margin-left: 8px;
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
</style>
