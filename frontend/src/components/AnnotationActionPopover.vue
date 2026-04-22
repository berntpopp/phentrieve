<template>
  <v-menu
    :model-value="visible"
    :target="target"
    location="top"
    location-strategy="connected"
    :scrim="false"
    @update:model-value="handleMenuVisibilityUpdate"
  >
    <v-list density="compact">
      <v-list-item
        :title="translate('annotatedDocumentPane.actions.inspect', 'Inspect')"
        @click="handleAction('inspect')"
      />
      <v-list-item
        :title="translate('annotatedDocumentPane.actions.addToCase', 'Add to case')"
        @click="handleAction('add-to-case')"
      />
      <v-list-item
        :title="translate('annotatedDocumentPane.actions.changeTerm', 'Change term')"
        @click="handleAction('change-term')"
      />
      <v-list-item
        :title="translate('annotatedDocumentPane.actions.removeAnnotation', 'Remove annotation')"
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

let i18n = null;

try {
  i18n = useI18n();
} catch {
  // Tests may mount without the i18n plugin; keep literal fallbacks in that case.
}

function translate(key, fallback) {
  return i18n ? i18n.t(key) : fallback;
}

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
</script>
