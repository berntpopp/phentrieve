<template>
  <section class="findings-pane">
    <v-list lines="three" class="rounded-lg findings-pane__list">
      <v-list-item
        v-for="term in normalizedTerms"
        :key="term.hpoId"
        class="findings-pane__item"
        @mouseenter="$emit('hover-term', hoverPayload(term))"
        @mouseleave="$emit('clear-hover')"
        @click="$emit('inspect-term', inspectPayload(term))"
      >
        <template #title>
          <div class="d-flex align-center justify-space-between ga-3">
            <div class="d-flex flex-column">
              <span class="text-body-1 font-weight-medium">{{ term.name }}</span>
              <span class="text-caption text-medium-emphasis">{{ term.hpoId }}</span>
            </div>
            <v-chip size="small" variant="tonal" color="primary">
              {{ confidenceBand(term.confidence) }}
            </v-chip>
          </div>
        </template>
        <template #subtitle>
          <div class="d-flex flex-wrap align-center ga-2 mt-2">
            <v-chip
              size="x-small"
              label
              :color="term.status === 'negated' ? 'error' : 'success'"
              variant="flat"
            >
              {{ assertionStatusLabel(term.status) }}
            </v-chip>
            <span class="text-caption">
              {{ t('resultsDisplay.textProcess.sourceChunks') }}: {{ term.sourceChunkIds.length }}
            </span>
            <span class="text-caption">
              {{ t('resultsDisplay.textProcess.topEvidence') }}: {{ topEvidenceValue(term) }}
            </span>
          </div>
        </template>
      </v-list-item>
    </v-list>
  </section>
</template>

<script setup>
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';

const props = defineProps({
  terms: {
    type: Array,
    default: () => [],
  },
});

defineEmits(['hover-term', 'clear-hover', 'inspect-term']);

const { t } = useI18n();

const normalizedTerms = computed(() =>
  props.terms
    .filter((term) => term && typeof term.hpo_id === 'string' && typeof term.name === 'string')
    .map((term) => ({
      hpoId: term.hpo_id,
      name: term.name,
      confidence: typeof term.confidence === 'number' ? term.confidence : 0,
      status: typeof term.status === 'string' ? term.status : 'affirmed',
      sourceChunkIds: Array.isArray(term.source_chunk_ids) ? term.source_chunk_ids : [],
      topEvidenceChunkId:
        term.top_evidence_chunk_id ??
        (Array.isArray(term.source_chunk_ids) ? term.source_chunk_ids[0] : null),
    }))
);

function confidenceBand(value) {
  if (value >= 0.85) return 'High';
  if (value >= 0.6) return 'Medium';
  return 'Low';
}

function assertionStatusLabel(status) {
  return t(`queryInterface.phenotypeCollection.assertionStatus.${status}`, status);
}

function topEvidenceValue(term) {
  return term.topEvidenceChunkId == null ? '-' : `#${term.topEvidenceChunkId}`;
}

function hoverPayload(term) {
  return {
    hpoId: term.hpoId,
  };
}

function inspectPayload(term) {
  return {
    hpoId: term.hpoId,
    sourceChunkIds: [...term.sourceChunkIds],
    topEvidenceChunkId: term.topEvidenceChunkId,
  };
}
</script>
