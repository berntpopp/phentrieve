<template>
  <div class="full-text-receipt">
    <div class="full-text-receipt__title">Full-text analysis ready</div>
    <div class="text-caption text-medium-emphasis">
      {{ receiptMeta }}
    </div>
    <div v-if="phenotypes.length > 0" class="d-flex justify-end mt-3">
      <v-btn
        data-testid="full-text-response-add-all"
        size="small"
        rounded="pill"
        color="primary"
        variant="tonal"
        @click="emitAddAll"
      >
        <v-icon start size="small">mdi-plus-circle</v-icon>
        Add all
      </v-btn>
    </div>
    <v-list
      v-if="phenotypes.length > 0"
      lines="two"
      class="rounded-lg mt-3 full-text-response-list"
    >
      <div
        v-for="(phenotype, index) in phenotypes"
        :key="`${item.id}-${phenotype.hpo_id}`"
        data-testid="full-text-response-phenotype"
        class="full-text-response-phenotype"
        :class="{
          'full-text-response-phenotype--active': hoveredPhenotypeId === phenotype.hpo_id,
        }"
        @mouseenter="emit('hover-phenotype', phenotype.hpo_id)"
        @mouseleave="emit('clear-hover')"
      >
        <ResultItem
          :result="mapTextProcessPhenotypeToResult(phenotype)"
          :rank="index + 1"
          :is-collected="isCollected(phenotype.hpo_id)"
          @add-to-collection="emit('add-to-collection', normalizeTextProcessPhenotype(phenotype))"
        />
      </div>
    </v-list>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import ResultItem from './ResultItem.vue';

defineOptions({
  name: 'FullTextResponseReceipt',
});

const props = defineProps({
  item: {
    type: Object,
    required: true,
  },
  collectedPhenotypes: {
    type: Array,
    default: () => [],
  },
  hoveredPhenotypeId: {
    type: String,
    default: null,
  },
});

const emit = defineEmits([
  'add-to-collection',
  'add-all-to-collection',
  'hover-phenotype',
  'clear-hover',
]);

const phenotypes = computed(() =>
  Array.isArray(props.item?.response?.aggregated_hpo_terms)
    ? props.item.response.aggregated_hpo_terms.filter(
        (term) => term && typeof term.hpo_id === 'string' && typeof term.name === 'string'
      )
    : []
);

const receiptMeta = computed(() => {
  const terms = props.item?.response?.aggregated_hpo_terms?.length ?? 0;
  const chunks = props.item?.response?.processed_chunks?.length ?? 0;

  if (chunks === 0 && typeof props.item?.query === 'string' && props.item.query.trim().length > 0) {
    return `${terms} findings in submitted note review`;
  }

  return `${terms} findings across ${chunks} document sections`;
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

function mapTextProcessPhenotypeToResult(term) {
  const confidence = parseScoreValue(term.confidence);
  const similarity = parseScoreValue(term.similarity);

  return {
    hpo_id: term.hpo_id,
    label: term.name,
    confidence,
    score: confidence,
    scoreType: 'confidence',
    similarity: confidence ?? similarity ?? 0,
    definition: term.definition || '',
    synonyms: Array.isArray(term.synonyms) ? term.synonyms : [],
    assertion_status: normalizeAssertionStatus(term.status),
  };
}

function normalizeAssertionStatus(status) {
  if (status === 'present') {
    return 'affirmed';
  }

  if (status === 'absent') {
    return 'negated';
  }

  if (
    status === 'affirmed' ||
    status === 'negated' ||
    status === 'uncertain' ||
    status === 'unknown'
  ) {
    return status;
  }

  return 'unknown';
}

function normalizeTextProcessPhenotype(term) {
  return {
    hpo_id: term.hpo_id,
    label: term.name,
    assertion_status: normalizeAssertionStatus(term.status),
  };
}

function isCollected(hpoId) {
  return props.collectedPhenotypes.some((entry) => entry.hpo_id === hpoId);
}

function emitAddAll() {
  emit(
    'add-all-to-collection',
    phenotypes.value.map((phenotype) => normalizeTextProcessPhenotype(phenotype))
  );
}
</script>

<style scoped>
.full-text-receipt {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.full-text-receipt__title {
  font-weight: 600;
}

.full-text-response-list {
  background: transparent;
}

.full-text-response-phenotype {
  border-radius: 12px;
  transition:
    transform 0.16s ease,
    box-shadow 0.16s ease;
}

.full-text-response-phenotype--active {
  transform: translateY(-1px);
  box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
}
</style>
