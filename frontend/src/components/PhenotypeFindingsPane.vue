<template>
  <section class="findings-pane">
    <v-list lines="three" class="rounded-lg findings-pane__list">
      <v-list-item
        v-for="term in terms"
        :key="term.hpo_id"
        class="findings-pane__item"
        @mouseenter="$emit('hover-term', term.hpo_id)"
        @mouseleave="$emit('clear-hover')"
        @click="$emit('inspect-term', term.hpo_id)"
      >
        <template #title>
          <div class="d-flex align-center justify-space-between ga-3">
            <div class="d-flex flex-column">
              <span class="text-body-1 font-weight-medium">{{ term.name }}</span>
              <span class="text-caption text-medium-emphasis">{{ term.hpo_id }}</span>
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
              {{ formatStatus(term.status) }}
            </v-chip>
            <span class="text-caption">{{ evidenceCountLabel(term.source_chunk_ids) }}</span>
            <span class="text-caption">{{ topEvidenceLabel(term) }}</span>
          </div>
        </template>
      </v-list-item>
    </v-list>
  </section>
</template>

<script setup>
defineProps({
  terms: {
    type: Array,
    default: () => [],
  },
});

defineEmits(['hover-term', 'clear-hover', 'inspect-term']);

function confidenceBand(value) {
  if (value >= 0.85) return 'High';
  if (value >= 0.6) return 'Medium';
  return 'Low';
}

function formatStatus(status) {
  if (!status) return 'Affirmed';
  return status.charAt(0).toUpperCase() + status.slice(1);
}

function evidenceCountLabel(sourceChunkIds) {
  const count = sourceChunkIds?.length ?? 0;
  return `${count} evidence chunk${count === 1 ? '' : 's'}`;
}

function topEvidenceLabel(term) {
  const topChunkId = term.top_evidence_chunk_id ?? term.source_chunk_ids?.[0];

  if (topChunkId == null) {
    return 'Top evidence: unavailable';
  }

  return `Top evidence: #${topChunkId}`;
}
</script>
