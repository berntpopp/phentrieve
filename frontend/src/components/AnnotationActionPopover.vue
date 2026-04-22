<template>
  <v-menu
    :model-value="visible"
    :target="target"
    location="top"
    location-strategy="connected"
    :scrim="false"
  >
    <v-list density="compact">
      <v-list-item :title="t('annotatedDocumentPane.actions.inspect', 'Inspect')" @click="$emit('inspect')" />
      <v-list-item :title="t('annotatedDocumentPane.actions.addToCase', 'Add to case')" @click="$emit('add-to-case')" />
      <v-list-item :title="t('annotatedDocumentPane.actions.changeTerm', 'Change term')" @click="$emit('change-term')" />
      <v-list-item :title="t('annotatedDocumentPane.actions.removeAnnotation', 'Remove annotation')" @click="$emit('remove-annotation')" />
    </v-list>
  </v-menu>
</template>

<script setup>
import { useI18n } from 'vue-i18n';

defineProps({
  visible: {
    type: Boolean,
    default: false,
  },
  target: {
    type: Object,
    default: null,
  },
  annotationId: {
    type: String,
    default: null,
  },
  selectedText: {
    type: String,
    default: '',
  },
});

defineEmits(['inspect', 'add-to-case', 'change-term', 'remove-annotation']);

let t = (_key, fallback) => fallback;

try {
  ({ t } = useI18n());
} catch {
  // Tests may mount without the i18n plugin; keep literal fallbacks in that case.
}
</script>
