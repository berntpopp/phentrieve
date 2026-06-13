<template>
  <v-menu
    :model-value="visible"
    :target="target"
    location="top"
    location-strategy="connected"
    :scrim="false"
    content-class="annotation-action-popover__content"
    @update:model-value="handleMenuVisibilityUpdate"
  >
    <v-list class="annotation-action-popover__list" density="compact" elevation="8">
      <div class="annotation-action-popover__header">
        <div class="annotation-action-popover__eyebrow">
          <v-icon icon="mdi-text-box-search-outline" size="x-small" />
          <span>{{ translate('annotatedDocumentPane.actions.title', 'Annotation tools') }}</span>
        </div>
        <div v-if="selectedText" class="annotation-action-popover__selection">
          {{ selectedText }}
        </div>
      </div>

      <v-divider class="annotation-action-popover__divider" />
      <template v-if="mode === 'selection'">
        <v-list-item
          data-testid="action-annotate-selection"
          prepend-icon="mdi-plus-circle-outline"
          :title="
            translate('annotatedDocumentPane.actions.annotateSelection', 'Annotate selection')
          "
          @click="handleAction('annotate-selection')"
        />
      </template>
      <template v-else>
        <v-list-item
          data-testid="action-change-term"
          prepend-icon="mdi-swap-horizontal"
          :title="translate('annotatedDocumentPane.actions.changeTerm', 'Change term')"
          @click="handleAction('change-term')"
        />
        <v-list-item
          data-testid="action-add-to-collection"
          prepend-icon="mdi-plus-circle-outline"
          :title="translate('annotatedDocumentPane.actions.addToCollection', 'Add to collection')"
          @click="handleAction('add-to-collection')"
        />
        <v-list-item
          v-if="canRevert"
          data-testid="action-revert"
          prepend-icon="mdi-undo-variant"
          :title="translate('annotatedDocumentPane.actions.revert', 'Revert to original')"
          @click="handleAction('revert')"
        />
        <v-list-item
          data-testid="action-remove-annotation"
          prepend-icon="mdi-delete-outline"
          :title="translate('annotatedDocumentPane.actions.removeAnnotation', 'Remove annotation')"
          @click="handleAction('remove-annotation')"
        />
      </template>
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
    type: [Object, Array, String],
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
  mode: {
    type: String,
    default: 'annotation', // 'annotation' | 'selection'
  },
  canRevert: {
    type: Boolean,
    default: false,
  },
});

const emit = defineEmits([
  'change-term',
  'remove-annotation',
  'add-to-collection',
  'annotate-selection',
  'revert',
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

<style scoped>
.annotation-action-popover__list {
  min-width: 220px;
  padding: 0.35rem;
  border: 1px solid rgba(var(--v-theme-outline), 0.14);
  border-radius: 0.9rem;
  background: linear-gradient(
    180deg,
    rgba(var(--v-theme-surface-bright), 0.98),
    rgba(var(--v-theme-surface), 1)
  );
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.16);
}

.annotation-action-popover__header {
  display: grid;
  gap: 0.35rem;
  padding: 0.55rem 0.7rem 0.45rem;
}

.annotation-action-popover__eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: rgba(var(--v-theme-primary), 0.92);
}

.annotation-action-popover__selection {
  font-size: 0.84rem;
  line-height: 1.45;
  color: rgba(var(--v-theme-on-surface), 0.92);
  display: -webkit-box;
  overflow: hidden;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
}

.annotation-action-popover__divider {
  margin: 0 0.35rem 0.2rem;
}

:deep(.annotation-action-popover__list .v-list-item) {
  border-radius: 0.7rem;
  margin-bottom: 0.12rem;
}

:deep(.annotation-action-popover__list .v-list-item:hover) {
  background: rgba(var(--v-theme-primary), 0.08);
}

:deep(.annotation-action-popover__list .v-list-item-title) {
  font-size: 0.9rem;
  font-weight: 600;
}
</style>
