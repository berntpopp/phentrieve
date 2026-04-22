<template>
  <PhenotypeCardRow
    :hpo-id="result.hpo_id"
    :label="result.label"
    :color="isCollected ? 'primary-lighten-5' : 'grey-lighten-5'"
  >
    <template #prepend>
      <v-badge :content="rank" color="primary" inline class="mr-2" />
    </template>

    <template #inline-tools>
      <v-btn
        v-if="hasDetails"
        variant="text"
        size="x-small"
        icon
        density="compact"
        class="ml-1"
        @click.stop="detailsExpanded = !detailsExpanded"
      >
        <v-icon size="small">
          {{ detailsExpanded ? 'mdi-chevron-up' : 'mdi-information' }}
        </v-icon>
        <v-tooltip
          activator="parent"
          location="top"
          :content-props="{
            'aria-label': detailsExpanded
              ? $t('resultsDisplay.hideDetails', 'Hide details')
              : $t('resultsDisplay.showDetails', 'Show details'),
          }"
        >
          {{
            detailsExpanded
              ? $t('resultsDisplay.hideDetails', 'Hide details')
              : $t('resultsDisplay.showDetails', 'Show details')
          }}
        </v-tooltip>
      </v-btn>
    </template>

    <template #actions>
      <div class="d-flex flex-row align-center gap-1 mr-2 score-chips-container">
        <SimilarityScore
          :score="displayScore"
          :type="scoreType"
          :decimals="2"
          :show-animation="false"
        />
      </div>

      <v-btn
        :icon="isCollected ? 'mdi-check-circle' : 'mdi-plus-circle'"
        size="small"
        :color="isCollected ? 'success' : 'primary'"
        variant="text"
        class="flex-shrink-0 add-btn"
        :disabled="isCollected"
        :title="
          isCollected
            ? $t('resultsDisplay.alreadyInCollectionTooltip', { id: result.hpo_id })
            : $t('resultsDisplay.addToCollectionTooltip', { id: result.hpo_id })
        "
        :aria-label="
          isCollected
            ? $t('resultsDisplay.alreadyInCollectionAriaLabel', {
                id: result.hpo_id,
                label: result.label,
              })
            : $t('resultsDisplay.addToCollectionAriaLabel', {
                id: result.hpo_id,
                label: result.label,
              })
        "
        @click.stop="$emit('add-to-collection', result)"
      />
    </template>

    <template #details>
      <v-expand-transition>
        <div v-if="detailsExpanded && hasDetails" class="mt-3 pt-3 details-section">
          <div v-if="result.definition && result.definition.trim()" class="mb-3">
            <div class="text-caption text-medium-emphasis mb-1 font-weight-bold">
              {{ $t('resultsDisplay.definitionLabel', 'Definition') }}:
            </div>
            <div class="text-body-2 definition-text">
              {{ result.definition }}
            </div>
          </div>

          <div v-if="result.synonyms && result.synonyms.length > 0" class="mb-2">
            <div class="text-caption text-medium-emphasis mb-1 font-weight-bold">
              {{ $t('resultsDisplay.synonymsLabel', 'Synonyms') }}:
            </div>
            <div class="d-flex flex-wrap ga-1">
              <v-chip
                v-for="(synonym, synIdx) in result.synonyms"
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
    </template>
  </PhenotypeCardRow>
</template>

<script setup>
import { ref, computed } from 'vue';
import SimilarityScore from './SimilarityScore.vue';
import PhenotypeCardRow from './PhenotypeCardRow.vue';

const props = defineProps({
  result: { type: Object, required: true },
  rank: { type: Number, required: true },
  isCollected: { type: Boolean, default: false },
});

defineEmits(['add-to-collection']);

const detailsExpanded = ref(false);

const hasDetails = computed(() => {
  return (
    (props.result.definition && props.result.definition.trim()) ||
    (props.result.synonyms && props.result.synonyms.length > 0)
  );
});

function parseScoreValue(value) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return null;
}

const displayScore = computed(() => {
  const explicitScore = parseScoreValue(props.result.score);
  if (explicitScore != null) {
    return explicitScore;
  }

  const confidenceScore = parseScoreValue(props.result.confidence);
  if (confidenceScore != null) {
    return confidenceScore;
  }

  return parseScoreValue(props.result.similarity) ?? 0;
});

const scoreType = computed(() => {
  if (props.result.scoreType === 'confidence' || props.result.scoreType === 'similarity') {
    return props.result.scoreType;
  }

  if (
    parseScoreValue(props.result.confidence) != null &&
    parseScoreValue(props.result.similarity) == null
  ) {
    return 'confidence';
  }

  return 'similarity';
});
</script>

<style scoped>
.add-btn {
  transform: scale(1.2);
  margin-left: 8px;
}

.score-chips-container {
  margin-top: 4px;
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
</style>
