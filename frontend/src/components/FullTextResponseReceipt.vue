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
        <div class="full-text-response-phenotype__row">
          <ResultItem
            :result="mapTextProcessPhenotypeToResult(phenotype)"
            :rank="index + 1"
            :is-collected="isCollected(phenotype.hpo_id)"
            @add-to-collection="emit('add-to-collection', normalizeTextProcessPhenotype(phenotype))"
          />
          <v-chip
            v-if="phenotype.origin === 'manual'"
            data-testid="finding-origin-badge"
            size="x-small"
            color="primary"
            variant="tonal"
            label
            class="full-text-response-phenotype__badge"
          >
            {{ translate('annotatedDocumentPane.origin.manual', 'Manual') }}
          </v-chip>
        </div>
      </div>
    </v-list>
  </div>
</template>

<script setup>
import { computed, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import ResultItem from './ResultItem.vue';
import { useFullTextCuration } from '../composables/useFullTextCuration';

defineOptions({
  name: 'FullTextResponseReceipt',
});

const props = defineProps({
  item: {
    type: Object,
    required: true,
  },
  noteText: {
    type: String,
    default: '',
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

let i18n = null;
try {
  i18n = useI18n();
} catch {
  i18n = null;
}
function translate(key, fallback) {
  return i18n ? i18n.t(key) : fallback;
}

const curation = useFullTextCuration(props.item.id);

// Seed once the response is available (idempotent; shared with the note curator).
watch(
  () => props.item?.response,
  () => curation.ensureSeeded(props.item, props.noteText || props.item?.query || ''),
  { immediate: true }
);

// Curated, term-level findings (auto terms minus removals, plus manual additions,
// with replaced terms reflecting their new HPO id). Origin drives the badge.
const phenotypes = computed(() =>
  curation.findings.value.filter(
    (term) => term && typeof term.hpo_id === 'string' && typeof term.name === 'string'
  )
);

// HPO term details (definition/synonyms) are only on the original response, not
// in the lightweight annotation model — look them up by id for display.
const detailsByHpoId = computed(() => {
  const map = new Map();
  const terms = Array.isArray(props.item?.response?.aggregated_hpo_terms)
    ? props.item.response.aggregated_hpo_terms
    : [];
  terms.forEach((term) => {
    if (term && typeof term.hpo_id === 'string') map.set(term.hpo_id, term);
  });
  return map;
});

const receiptMeta = computed(() => {
  const terms = phenotypes.value.length;
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
  const details = detailsByHpoId.value.get(term.hpo_id) || {};

  return {
    hpo_id: term.hpo_id,
    label: term.name,
    confidence,
    score: confidence,
    scoreType: 'confidence',
    similarity: confidence ?? similarity ?? 0,
    definition: details.definition || '',
    synonyms: Array.isArray(details.synonyms) ? details.synonyms : [],
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

.full-text-response-phenotype__row {
  position: relative;
}

.full-text-response-phenotype__badge {
  position: absolute;
  top: 6px;
  right: 8px;
  pointer-events: none;
}
</style>
