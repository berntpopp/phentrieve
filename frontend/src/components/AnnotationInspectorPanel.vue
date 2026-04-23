<template>
  <aside class="annotation-inspector annotation-inspector--active">
    <div class="panel-header d-flex align-center justify-space-between">
      <div class="text-subtitle-2">{{ t('resultsDisplay.showDetails') }}</div>
      <v-btn
        variant="text"
        size="small"
        icon="mdi-arrow-left"
        :aria-label="t('common.close')"
        @click="$emit('back')"
      />
    </div>

    <div v-if="normalizedTerm" class="mt-4">
      <div :id="`annotation-detail-${normalizedTerm.hpoId}`">
        {{ normalizedTerm.name }}
      </div>
      <div class="text-caption">
        {{ t('resultsDisplay.confidenceHeader') }}: {{ formattedConfidence }}
      </div>
    </div>
  </aside>
</template>

<script setup>
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { normalizeSelectedTerm } from '../utils/annotationInspector';

const props = defineProps({
  selectedTerm: {
    type: Object,
    default: null,
    validator: (term) =>
      term == null ||
      (typeof term.hpo_id === 'string' &&
        typeof term.name === 'string' &&
        (term.confidence == null || typeof term.confidence === 'number')),
  },
});

defineEmits(['back']);

const { t } = useI18n();

const normalizedTerm = computed(() => normalizeSelectedTerm(props.selectedTerm));

const formattedConfidence = computed(() => {
  if (normalizedTerm.value?.confidence == null) return '';
  return normalizedTerm.value.confidence.toFixed(2);
});
</script>
