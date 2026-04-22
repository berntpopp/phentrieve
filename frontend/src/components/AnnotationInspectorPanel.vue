<template>
  <aside class="annotation-inspector annotation-inspector--active">
    <div class="panel-header d-flex align-center justify-space-between">
      <div class="text-subtitle-2">Annotation Inspector</div>
      <v-btn variant="text" size="small" icon="mdi-arrow-left" @click="$emit('back')" />
    </div>

    <div v-if="selectedTerm" class="mt-4">
      <div :id="`annotation-detail-${selectedTerm.hpo_id}`">
        {{ selectedTerm.name }}
      </div>
      <div class="text-caption">Confidence: {{ formattedConfidence }}</div>
    </div>
  </aside>
</template>

<script setup>
import { computed } from 'vue';

const props = defineProps({
  selectedTerm: {
    type: Object,
    default: null,
  },
});

defineEmits(['back']);

const formattedConfidence = computed(() => {
  if (props.selectedTerm?.confidence == null) return '';
  return props.selectedTerm.confidence.toFixed(2);
});
</script>
