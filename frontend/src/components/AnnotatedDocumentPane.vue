<template>
  <section ref="rootElement" class="annotated-document-pane">
    <article
      v-for="chunk in chunks"
      :key="chunk.chunk_id"
      :data-chunk-id="chunk.chunk_id"
      :data-chunk-evidence-mode="chunk.evidence_mode || 'chunk'"
      class="annotated-chunk"
    >
      <div
        class="chunk-gutter"
        :class="`chunk-gutter--${chunk.evidence_mode || 'chunk'}`"
        aria-hidden="true"
      />
      <div class="chunk-content">
        <p
          :data-chunk-text-id="chunk.chunk_id"
          class="chunk-text chunk-text--selectable"
          :class="{
            'chunk-text--annotation-hover': hoveredAnnotationChunkId === chunk.chunk_id,
          }"
          @mouseup="handleTextSelection(chunk)"
          @click="handleChunkClick(chunk, $event)"
          @mousemove="handleChunkPointerMove(chunk, $event)"
          @mouseleave="clearChunkPointerState(chunk)"
        >
          <template v-if="!needsFallbackMarks(chunk, supportsCustomHighlight)">
            {{ chunk.text }}
          </template>
          <template v-else>
            <span
              v-for="segment in buildMarkedSegments(chunk, selectedAnnotationSet)"
              :key="segment.key"
            >
              <NestedAnnotationMarks
                v-if="segment.annotations.length > 0"
                :annotations="segment.annotations"
                :text="segment.text"
                @annotation-click="openAnnotationPopover($event.event, $event.segment)"
              />
              <span v-else>{{ segment.text }}</span>
            </span>
            <span
              v-for="annotation in getChunkAnnotationDetails(chunk)"
              :id="`annotation-detail-${annotation.id}`"
              :key="`detail-${chunk.chunk_id}-${annotation.id}`"
              class="sr-only"
            >
              {{ annotation.detailText }}
            </span>
          </template>
        </p>
      </div>
    </article>

    <AnnotationActionPopover
      :visible="popoverVisible"
      :target="popoverTarget"
      :annotation-id="activeAnnotationId"
      :selected-text="activeSelectedText"
      @update:visible="handlePopoverVisibilityUpdate"
      @close="clearPopover"
      @inspect="noop"
      @add-to-case="noop"
      @change-term="noop"
      @remove-annotation="noop"
    />
  </section>
</template>

<script setup>
import { computed, defineComponent, h, ref, toRef } from 'vue';
import AnnotationActionPopover from './AnnotationActionPopover.vue';
import {
  buildMarkedSegments,
  getChunkAnnotationDetails,
  needsFallbackMarks,
} from '../composables/useDocumentAnnotations';
import { useCustomHighlightOverlay } from '../composables/useCustomHighlightOverlay';

const NestedAnnotationMarks = defineComponent({
  name: 'NestedAnnotationMarks',
  props: {
    annotations: {
      type: Array,
      required: true,
    },
    text: {
      type: String,
      required: true,
    },
  },
  emits: ['annotation-click'],
  setup(props, { emit }) {
    function renderAt(index) {
      const annotation = props.annotations.at(index);

      if (!annotation) {
        return props.text;
      }

      const child = index === props.annotations.length - 1 ? props.text : renderAt(index + 1);

      return h(
        'mark',
        {
          'data-annotation-id': annotation.id,
          'aria-details': `annotation-detail-${annotation.id}`,
          class: ['annotated-mark', { 'annotated-mark--selected': annotation.selected }],
          onClick: (event) => {
            event.stopPropagation();
            emit('annotation-click', {
              event,
              segment: {
                annotationId: annotation.id,
                text: props.text,
              },
            });
          },
        },
        child
      );
    }

    return () => renderAt(0);
  },
});

const props = defineProps({
  chunks: {
    type: Array,
    default: () => [],
  },
  selectedAnnotationIds: {
    type: Array,
    default: () => [],
  },
});

const rootElement = ref(null);
const popoverVisible = ref(false);
const popoverTarget = ref(null);
const activeAnnotationId = ref(null);
const activeSelectedText = ref('');
const activePopoverAnchor = ref(null);
const hoveredAnnotationChunkId = ref(null);

const selectedAnnotationSet = computed(() => new Set(props.selectedAnnotationIds));
const overlay = useCustomHighlightOverlay({
  chunks: toRef(props, 'chunks'),
  selectedAnnotationIds: toRef(props, 'selectedAnnotationIds'),
  rootElement,
  onLayoutRefresh: refreshPopoverTarget,
});
const { supportsCustomHighlight } = overlay;

function noop() {}

function handlePopoverVisibilityUpdate(nextVisible) {
  popoverVisible.value = nextVisible;

  if (!nextVisible) {
    clearPopover();
  }
}

function refreshPopoverTarget() {
  if (!popoverVisible.value) {
    return;
  }

  const nextTarget = overlay.getAnchorTarget(activePopoverAnchor.value, popoverTarget.value);

  if (nextTarget) {
    popoverTarget.value = nextTarget;
  }
}

function rectToTarget(rect) {
  if (!rect) {
    return null;
  }

  return {
    x: rect.left + rect.width / 2,
    y: rect.top,
  };
}

function clearPopover() {
  popoverVisible.value = false;
  popoverTarget.value = null;
  activeAnnotationId.value = null;
  activeSelectedText.value = '';
  activePopoverAnchor.value = null;
}

function clearChunkPointerState(chunk) {
  if (hoveredAnnotationChunkId.value === chunk.chunk_id) {
    hoveredAnnotationChunkId.value = null;
  }
}

function handleChunkPointerMove(chunk, event) {
  if (!supportsCustomHighlight || needsFallbackMarks(chunk, supportsCustomHighlight)) {
    clearChunkPointerState(chunk);
    return;
  }

  hoveredAnnotationChunkId.value = overlay.findHitboxForEvent(chunk.chunk_id, event)
    ? chunk.chunk_id
    : null;
}

function openPopover(target, options = {}) {
  if (!target) {
    clearPopover();
    return;
  }

  popoverTarget.value = target;
  activeAnnotationId.value = options.annotationId || null;
  activeSelectedText.value = options.selectedText || '';
  activePopoverAnchor.value = options.anchor || null;
  popoverVisible.value = true;
}

function getCurrentSelection() {
  return window.getSelection?.() || null;
}

function hasActiveTextSelection(selection = getCurrentSelection()) {
  return Boolean(selection && !selection.isCollapsed && selection.toString().trim());
}

function openAnnotationPopover(event, segment) {
  const selection = getCurrentSelection();

  if (hasActiveTextSelection(selection)) {
    return;
  }

  const rect = event?.currentTarget?.getBoundingClientRect?.();
  selection?.removeAllRanges?.();

  openPopover(rectToTarget(rect), {
    annotationId: segment.annotationId,
    selectedText: segment.text,
    anchor: {
      type: 'mark-element',
      annotationId: segment.annotationId,
      element: event?.currentTarget || null,
    },
  });
}

function openCustomHighlightPopover(hitbox) {
  openPopover(hitbox.target, {
    annotationId: hitbox.annotationId,
    selectedText: hitbox.selectedText,
    anchor: {
      type: 'custom-hitbox',
      key: hitbox.key,
      annotationId: hitbox.annotationId,
      target: hitbox.target,
    },
  });
}

function handleTextSelection(chunk) {
  const selection = getCurrentSelection();

  if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
    return;
  }

  const chunkElement =
    rootElement.value?.querySelector(`[data-chunk-text-id="${chunk.chunk_id}"]`) || null;
  if (!chunkElement) {
    return;
  }

  if (!chunkElement.contains(selection.anchorNode) || !chunkElement.contains(selection.focusNode)) {
    return;
  }

  const selectedText = selection.toString().trim();
  if (!selectedText) {
    return;
  }

  const range = selection.getRangeAt(0);
  const rect = range.getBoundingClientRect();
  openPopover(rectToTarget(rect), {
    selectedText,
    anchor: {
      type: 'selection',
      range: range.cloneRange?.() || range,
    },
  });
}

function handleChunkClick(chunk, event) {
  if (!supportsCustomHighlight || needsFallbackMarks(chunk, supportsCustomHighlight)) {
    return;
  }

  if (hasActiveTextSelection(getCurrentSelection())) {
    return;
  }

  overlay.refreshCustomHighlightGeometry();

  const resolvedHitbox = overlay.findHitboxForEvent(chunk.chunk_id, event);

  if (!resolvedHitbox) {
    return;
  }

  openCustomHighlightPopover(resolvedHitbox);
}
</script>

<style scoped>
.annotated-document-pane {
  display: grid;
  gap: 0.75rem;
}

.annotated-chunk {
  display: grid;
  grid-template-columns: 0.5rem minmax(0, 1fr);
  gap: 0.75rem;
  align-items: stretch;
  border-radius: 0.75rem;
  background: rgba(var(--v-theme-surface), 1);
  border: 1px solid rgba(var(--v-theme-outline), 0.18);
}

.chunk-gutter {
  border-radius: 0.75rem 0 0 0.75rem;
}

.chunk-gutter--chunk {
  background: rgba(var(--v-theme-primary), 0.12);
}

.chunk-gutter--span {
  background: rgba(var(--v-theme-warning), 0.2);
}

.chunk-content {
  padding: 0.875rem 1rem 0.875rem 0;
}

.chunk-text {
  margin: 0;
  line-height: 1.6;
  color: rgba(var(--v-theme-on-surface), 0.92);
  white-space: pre-wrap;
  word-break: break-word;
  cursor: text;
}

.chunk-text--annotation-hover {
  cursor: pointer;
}

.chunk-text mark,
.annotated-mark {
  background-color: rgba(var(--v-theme-warning), 0.24);
  color: inherit;
  border-radius: 0.2rem;
  box-shadow: inset 0 -0.4em 0 rgba(var(--v-theme-warning), 0.16);
  cursor: pointer;
  transition:
    background-color 0.18s ease,
    box-shadow 0.18s ease;
}

.chunk-text mark:hover,
.annotated-mark:hover {
  background-color: rgba(var(--v-theme-warning), 0.3);
  box-shadow: inset 0 -0.48em 0 rgba(var(--v-theme-warning), 0.2);
}

.chunk-text mark.annotated-mark--selected,
.annotated-mark.annotated-mark--selected {
  background-color: rgba(var(--v-theme-error), 0.22);
  box-shadow: inset 0 -0.4em 0 rgba(var(--v-theme-error), 0.14);
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
  border: 0;
}
</style>
