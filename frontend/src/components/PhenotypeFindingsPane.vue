<template>
  <section class="findings-pane">
    <div class="findings-pane__header d-flex align-center justify-space-between ga-3 mb-3">
      <div>
        <div class="text-subtitle-2">Phenotypes</div>
        <div class="text-caption text-medium-emphasis">
          {{ findingsSummary }}
        </div>
      </div>

      <v-btn
        data-testid="findings-add-all"
        size="small"
        rounded="pill"
        color="primary"
        variant="tonal"
        :disabled="normalizedTerms.length === 0"
        @click.stop="$emit('add-all-to-collection', collectionPayloads)"
      >
        <v-icon start size="small">mdi-plus-circle</v-icon>
        Add all
      </v-btn>
    </div>

    <v-list lines="three" class="rounded-lg findings-pane__list">
      <PhenotypeCardRow
        v-for="term in normalizedTerms"
        :key="term.hpoId"
        :hpo-id="term.hpoId"
        :label="term.name"
        color="grey-lighten-5"
        clickable
        @mouseenter="$emit('hover-term', hoverPayload(term))"
        @mouseleave="$emit('clear-hover')"
        @click="$emit('inspect-term', inspectPayload(term))"
      >
        <template #inline-tools>
          <div v-if="hasMeaningfulConfidence(term.confidence)" class="ml-2 findings-pane__score">
            <SimilarityScore
              :score="term.confidence"
              type="confidence"
              :decimals="2"
              :show-animation="false"
            />
          </div>
        </template>

        <template #metadata>
          <div class="d-flex flex-wrap align-center ga-2">
            <v-chip
              size="x-small"
              label
              :color="term.status === 'negated' ? 'error' : 'success'"
              variant="flat"
            >
              {{ assertionStatusLabel(term.status) }}
            </v-chip>
            <span v-if="term.sourceChunkIds.length > 0" class="text-caption">
              {{ t('resultsDisplay.textProcess.sourceChunks') }}: {{ term.sourceChunkIds.length }}
            </span>
            <span v-if="term.topEvidenceChunkId != null" class="text-caption">
              {{ t('resultsDisplay.textProcess.topEvidence') }}: {{ topEvidenceValue(term) }}
            </span>
            <span v-if="term.sourceChunkIds.length === 0" class="text-caption text-medium-emphasis">
              LLM inferred
            </span>
          </div>
        </template>
      </PhenotypeCardRow>
    </v-list>
  </section>
</template>

<script setup>
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import PhenotypeCardRow from './PhenotypeCardRow.vue';
import SimilarityScore from './SimilarityScore.vue';

const props = defineProps({
  terms: {
    type: Array,
    default: () => [],
  },
});

defineEmits(['hover-term', 'clear-hover', 'inspect-term', 'add-all-to-collection']);

const { t } = useI18n();

const normalizedTerms = computed(() =>
  props.terms
    .filter((term) => term && typeof term.hpo_id === 'string' && typeof term.name === 'string')
    .map((term) => ({
      hpoId: term.hpo_id,
      name: term.name,
      confidence: typeof term.confidence === 'number' ? term.confidence : null,
      status: typeof term.status === 'string' ? term.status : 'affirmed',
      sourceChunkIds: Array.isArray(term.source_chunk_ids) ? term.source_chunk_ids : [],
      topEvidenceChunkId:
        term.top_evidence_chunk_id ??
        (Array.isArray(term.source_chunk_ids) ? term.source_chunk_ids[0] : null),
    }))
);

const collectionPayloads = computed(() =>
  normalizedTerms.value.map((term) => ({
    hpo_id: term.hpoId,
    label: term.name,
    assertion_status: normalizeAssertionStatus(term.status),
  }))
);

const findingsSummary = computed(() => {
  const count = normalizedTerms.value.length;
  return `${count} extracted phenotype${count === 1 ? '' : 's'}`;
});

function assertionStatusLabel(status) {
  const normalizedStatus = normalizeAssertionStatus(status);

  if (normalizedStatus === 'affirmed') {
    return t('queryInterface.phenotypeCollection.assertionStatus.affirmed');
  }

  if (normalizedStatus === 'negated') {
    return t('queryInterface.phenotypeCollection.assertionStatus.negated');
  }

  return t('queryInterface.phenotypeCollection.assertionStatus.unknown');
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

function hasMeaningfulConfidence(value) {
  return typeof value === 'number' && value > 0;
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
    hpo_id: term.hpoId,
    name: term.name,
    confidence: term.confidence,
    status: term.status,
    source_chunk_ids: [...term.sourceChunkIds],
    top_evidence_chunk_id: term.topEvidenceChunkId,
  };
}
</script>

<style scoped>
.findings-pane__header {
  padding-inline: 4px;
}

.findings-pane__list {
  background: transparent;
}

.findings-pane__score {
  display: inline-flex;
  align-items: center;
}
</style>
