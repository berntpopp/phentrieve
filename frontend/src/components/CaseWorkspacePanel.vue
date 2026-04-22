<template>
  <aside class="case-workspace">
    <div class="panel-header">
      <div class="text-subtitle-2">Case Workspace</div>
      <v-btn
        size="small"
        variant="text"
        data-testid="create-case-button"
        @click="$emit('create-case')"
      >
        New case
      </v-btn>
    </div>
    <v-list>
      <v-list-item
        v-for="item in cases"
        :key="item.id"
        :active="item.id === activeCaseId"
        :data-testid="`case-item-${item.id}`"
        @click="$emit('select-case', item.id)"
      >
        <template #title>{{ item.label }}</template>
        <template #subtitle>{{ item.phenotypes.length }} phenotypes</template>
      </v-list-item>
    </v-list>

    <div v-if="activeCase" class="case-summary">
      <div class="text-caption text-medium-emphasis">Active case</div>
      <div class="text-body-2 font-weight-medium">{{ activeCase.label }}</div>
      <div class="text-caption text-medium-emphasis">
        {{ phenotypeSummaryLabel(activeCase.phenotypes.length) }}
      </div>
    </div>

    <div class="case-actions">
      <v-btn block color="primary" data-testid="add-all-button" @click="$emit('add-all')">
        Add all findings
      </v-btn>
      <v-btn
        block
        variant="tonal"
        data-testid="export-case-button"
        :disabled="!activeCase || activeCase.phenotypes.length === 0"
        @click="$emit('export-case')"
      >
        Export Phenopacket
      </v-btn>
    </div>

    <div v-if="activeCase && activeCase.phenotypes.length > 0" class="case-preview">
      <div class="text-caption text-medium-emphasis mb-2">Included phenotypes</div>
      <div
        v-for="phenotype in activeCase.phenotypes"
        :key="`${activeCase.id}-${phenotype.hpo_id}`"
        class="case-preview__item"
      >
        <div class="text-body-2">{{ phenotype.label }}</div>
        <div class="text-caption text-medium-emphasis">{{ phenotype.hpo_id }}</div>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { computed } from 'vue';

const props = defineProps({
  cases: { type: Array, default: () => [] },
  activeCaseId: { type: String, default: '' },
});

defineEmits(['create-case', 'select-case', 'add-all', 'export-case']);

const activeCase = computed(
  () => props.cases.find((item) => item.id === props.activeCaseId) || props.cases[0] || null
);

function phenotypeSummaryLabel(count) {
  return count === 1 ? '1 phenotype selected' : `${count} phenotypes selected`;
}
</script>

<style scoped>
.case-workspace {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px;
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.panel-header :deep(.v-btn) {
  text-transform: none;
}

.case-summary {
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 12px;
  padding: 12px;
  background: rgba(248, 250, 252, 0.9);
}

.case-actions {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.case-preview {
  border-top: 1px solid rgba(0, 0, 0, 0.08);
  padding-top: 12px;
}

.case-preview__item {
  padding: 8px 0;
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
}

.case-preview__item:last-child {
  border-bottom: 0;
}
</style>
