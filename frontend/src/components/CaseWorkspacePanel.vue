<template>
  <aside class="case-workspace">
    <div class="panel-header">
      <div class="text-subtitle-2">Case Workspace</div>
      <!-- TODO(Stream G): Connect this action to fullTextWorkspace.js case creation and active-case selection from QueryInterface.vue. -->
      <v-btn size="small" variant="text" @click="$emit('create-case')">New case</v-btn>
    </div>
    <v-list>
      <v-list-item
        v-for="item in cases"
        :key="item.id"
        :active="item.id === activeCaseId"
        @click="$emit('select-case', item.id)"
      >
        <template #title>{{ item.label }}</template>
        <template #subtitle>{{ item.phenotypes.length }} phenotypes</template>
      </v-list-item>
    </v-list>
    <!-- TODO(Stream G): Route add-all from QueryInterface.vue full-text results into addPhenotypeToActiveCase(...) instead of the legacy global collection. -->
    <v-btn block color="primary" @click="$emit('add-all')">Add all extracted phenotypes</v-btn>
    <!-- TODO(Stream G): Route export-case through usePhenotypeCollection.js and PhentrieveService.js backend phenopacket export for the active workspace case. -->
    <v-btn block variant="tonal" @click="$emit('export-case')">Export Phenopacket</v-btn>
  </aside>
</template>

<script setup>
defineProps({
  cases: { type: Array, default: () => [] },
  activeCaseId: { type: String, default: '' },
});

defineEmits(['create-case', 'select-case', 'add-all', 'export-case']);
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
</style>
