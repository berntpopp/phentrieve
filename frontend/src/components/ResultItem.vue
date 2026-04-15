<template>
  <v-list-item
    class="mb-1 rounded-lg"
    :color="isCollected ? 'primary-lighten-5' : 'grey-lighten-5'"
    border
    density="compact"
  >
    <template #prepend>
      <v-badge :content="rank" color="primary" inline class="mr-2" />
    </template>

    <v-list-item-title class="pb-2">
      <div class="d-flex flex-wrap align-center">
        <!-- HPO ID and Label on the left -->
        <div class="flex-grow-1">
          <div class="d-flex align-center mb-1">
            <a
              :href="hpoTermUrl(result.hpo_id)"
              target="_blank"
              rel="noopener noreferrer"
              class="hpo-link"
              :title="`View ${result.hpo_id} in HPO Browser`"
            >
              <span class="hpo-id font-weight-bold">{{ result.hpo_id }}</span>
              <v-icon size="x-small" class="ml-1">mdi-open-in-new</v-icon>
            </a>
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
          </div>
          <div class="d-flex align-center">
            <span class="text-body-2 text-high-emphasis hpo-label">{{ result.label }}</span>
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
        </div>
      </div>

      <!-- Expandable Details Section -->
      <v-expand-transition>
        <div v-if="detailsExpanded && hasDetails" class="mt-3 pt-3 details-section">
          <!-- Definition -->
          <div v-if="result.definition && result.definition.trim()" class="mb-3">
            <div class="text-caption text-medium-emphasis mb-1 font-weight-bold">
              {{ $t('resultsDisplay.definitionLabel', 'Definition') }}:
            </div>
            <div class="text-body-2 definition-text">
              {{ result.definition }}
            </div>
          </div>

          <!-- Synonyms -->
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
    </v-list-item-title>

    <template #append>
      <!-- Append content moved to subtitle for better space usage -->
    </template>
  </v-list-item>
</template>

<script setup>
import { ref, computed } from 'vue';
import SimilarityScore from './SimilarityScore.vue';
import { HPO_TERM_URL } from '../constants/urls';

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

function hpoTermUrl(hpoId) {
  return HPO_TERM_URL(hpoId);
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
