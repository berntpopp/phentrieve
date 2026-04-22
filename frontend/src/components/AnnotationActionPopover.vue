<template>
  <v-menu
    :model-value="visible"
    :target="target"
    location="top"
    location-strategy="connected"
    :scrim="false"
    @update:modelValue="handleMenuVisibilityUpdate"
  >
    <v-list density="compact">
      <v-list-item
        :title="t('annotatedDocumentPane.actions.inspect', 'Inspect')"
        @click="handleAction('inspect')"
      />
      <v-list-item
        :title="t('annotatedDocumentPane.actions.addToCase', 'Add to case')"
        @click="handleAction('add-to-case')"
      />
      <v-list-item
        :title="t('annotatedDocumentPane.actions.changeTerm', 'Change term')"
        @click="handleAction('change-term')"
      />
      <v-list-item
        :title="t('annotatedDocumentPane.actions.removeAnnotation', 'Remove annotation')"
        @click="handleAction('remove-annotation')"
      />
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

const emit = defineEmits([
  'inspect',
  'add-to-case',
  'change-term',
  'remove-annotation',
  'update:visible',
  'close',
]);

function handleMenuVisibilityUpdate(nextVisible) {
  emit('update:visible', nextVisible);

  if (!nextVisible) {
    emit('close');
  }
}

function handleAction(action) {
  emit(action);
  emit('update:visible', false);
  emit('close');
}

let t = (_key, fallback) => fallback;

try {
  ({ t } = useI18n());
} catch {
  // Tests may mount without the i18n plugin; keep literal fallbacks in that case.
}
</script>
