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
          class="chunk-text"
          @mouseup="handleTextSelection(chunk)"
          @click="handleChunkClick(chunk, $event)"
        >
          <template v-if="!needsFallbackMarks(chunk)">
            {{ chunk.text }}
          </template>
          <template v-else>
            <span v-for="segment in buildMarkedSegments(chunk)" :key="segment.key">
              <mark
                v-if="segment.annotationId"
                :data-annotation-id="segment.annotationId"
                :aria-details="`annotation-detail-${segment.annotationId}`"
                :class="{ 'annotated-mark--selected': segment.selected }"
                @click.stop="openAnnotationPopover($event, segment)"
              >
                {{ segment.text }}
              </mark>
              <span
                v-if="segment.annotationId"
                :id="`annotation-detail-${segment.annotationId}`"
                class="sr-only"
              >
                {{ segment.detailText }}
              </span>
              <span v-else>{{ segment.text }}</span>
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
      @inspect="noop"
      @add-to-case="noop"
      @change-term="noop"
      @remove-annotation="noop"
    />
  </section>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import AnnotationActionPopover from './AnnotationActionPopover.vue';

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

const supportsCustomHighlight =
  typeof globalThis.CSS !== 'undefined' &&
  typeof globalThis.Highlight !== 'undefined' &&
  typeof globalThis.CSS.highlights !== 'undefined';

const customHighlightNames = new Set();
let customHighlightStyleElement = null;
const rootElement = ref(null);
const customHighlightHitboxes = ref([]);

const popoverVisible = ref(false);
const popoverTarget = ref(null);
const activeAnnotationId = ref(null);
const activeSelectedText = ref('');

const selectedAnnotationSet = computed(() => new Set(props.selectedAnnotationIds));

function noop() {}

function findChunkTextElement(chunkId) {
  return rootElement.value?.querySelector(`[data-chunk-text-id="${chunkId}"]`) || null;
}

function getAnnotations(chunk) {
  return Array.isArray(chunk.annotations) ? [...chunk.annotations] : [];
}

function getSpanAnnotations(chunk) {
  if ((chunk.evidence_mode || 'chunk') !== 'span') {
    return [];
  }

  const textLength = (chunk.text || '').length;

  return getAnnotations(chunk)
    .filter((item) => item.start_char != null && item.end_char != null)
    .map((item, index) => {
      const start = Math.max(0, Math.min(item.start_char, textLength));
      const end = Math.max(start, Math.min(item.end_char, textLength));

      if (end <= start) {
        return null;
      }

      return {
        ...item,
        id: item.id || `annotation-${chunk.chunk_id}-${index}`,
        start_char: start,
        end_char: end,
      };
    })
    .filter(Boolean);
}

function needsFallbackMarks(chunk) {
  return getSpanAnnotations(chunk).length > 0 && !supportsCustomHighlight;
}

function buildMarkedSegments(chunk) {
  const text = chunk.text || '';
  const annotations = getSpanAnnotations(chunk).sort((left, right) => left.start_char - right.start_char);
  const segments = [];
  let cursor = 0;

  annotations.forEach((annotation, index) => {
    const start = Math.max(cursor, annotation.start_char);
    const end = Math.max(start, annotation.end_char);

    if (start > cursor) {
      segments.push({
        key: `plain-${index}-${cursor}-${start}`,
        text: text.slice(cursor, start),
      });
    }

    if (end > start) {
      segments.push({
        key: `annotation-${annotation.id}-${start}-${end}`,
        text: text.slice(start, end),
        annotationId: annotation.id,
        selected: selectedAnnotationSet.value.has(annotation.id),
        detailText: annotation.matched_text_in_chunk || text.slice(start, end),
      });
      cursor = end;
    }
  });

  if (cursor < text.length) {
    segments.push({
      key: `tail-${cursor}-${text.length}`,
      text: text.slice(cursor),
    });
  }

  return segments;
}

function getHighlightName(annotationId, selected) {
  return selected ? `annotation-selected-${annotationId}` : `annotation-${annotationId}`;
}

function ensureCustomHighlightStyleElement() {
  if (!supportsCustomHighlight) {
    return null;
  }

  if (!customHighlightStyleElement) {
    customHighlightStyleElement = document.createElement('style');
    customHighlightStyleElement.setAttribute('data-annotated-document-highlight-style', 'true');
    document.head.appendChild(customHighlightStyleElement);
  }

  return customHighlightStyleElement;
}

function syncCustomHighlightStyles() {
  if (!supportsCustomHighlight) {
    return;
  }

  const styleElement = ensureCustomHighlightStyleElement();
  const rules = [];
  const seenAnnotationIds = new Set();

  props.chunks.forEach((chunk) => {
    getSpanAnnotations(chunk).forEach((annotation) => {
      if (seenAnnotationIds.has(annotation.id)) {
        return;
      }

      seenAnnotationIds.add(annotation.id);

      const baseName = getHighlightName(annotation.id, false);
      const selectedName = getHighlightName(annotation.id, true);

      rules.push(
        `::highlight(${baseName}) { background: rgba(var(--v-theme-warning), 0.24); border-bottom: 1px solid rgba(var(--v-theme-warning), 0.72); }`
      );
      rules.push(
        `::highlight(${selectedName}) { background: rgba(var(--v-theme-error), 0.22); border-bottom: 1px solid rgba(var(--v-theme-error), 0.8); }`
      );
    });
  });

  styleElement.textContent = rules.join('\n');
}

function clearCustomHighlights() {
  if (!supportsCustomHighlight) {
    customHighlightHitboxes.value = [];
    return;
  }

  customHighlightNames.forEach((name) => {
    globalThis.CSS.highlights.delete(name);
  });
  customHighlightNames.clear();
  customHighlightHitboxes.value = [];
}

function findFirstTextNode(element) {
  if (!element) {
    return null;
  }

  if (element.firstChild?.nodeType === 3) {
    return element.firstChild;
  }

  const nodes = [...element.childNodes];

  while (nodes.length > 0) {
    const node = nodes.shift();

    if (node?.nodeType === 3) {
      return node;
    }

    if (node?.childNodes?.length) {
      nodes.unshift(...node.childNodes);
    }
  }

  return null;
}

function buildCustomHighlightRange(element, annotation, chunkText) {
  if (!supportsCustomHighlight || !element) {
    return null;
  }

  if (!element.textContent && chunkText) {
    element.textContent = chunkText;
  }

  const textNode = findFirstTextNode(element);
  if (!textNode) {
    return null;
  }

  const nodeLength = textNode.textContent?.length || chunkText.length || 0;
  const start = Math.max(0, Math.min(annotation.start_char, nodeLength));
  const end = Math.max(start, Math.min(annotation.end_char, nodeLength));

  if (end <= start) {
    return null;
  }

  const range = new globalThis.Range();

  range.setStart(textNode, start);
  range.setEnd(textNode, end);

  return range;
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

function buildHitboxesForRange(range, annotation, chunkIndex, annotationIndex) {
  const rects = typeof range.getClientRects === 'function' ? [...range.getClientRects()] : [];
  const usableRects = rects.length > 0 ? rects : [range.getBoundingClientRect?.()].filter(Boolean);

  return usableRects.map((rect, rectIndex) => ({
    key: `custom-hitbox-${annotation.id}-${chunkIndex}-${annotationIndex}-${rectIndex}`,
    chunkId: props.chunks[chunkIndex]?.chunk_id,
    annotationId: annotation.id,
    detailText: annotation.matched_text_in_chunk || '',
    selectedText: annotation.matched_text_in_chunk || '',
    rect,
    target: rectToTarget(rect),
  }));
}

async function syncCustomHighlights() {
  if (!supportsCustomHighlight) {
    return;
  }

  clearCustomHighlights();
  syncCustomHighlightStyles();

  const groupedRanges = new Map();
  const hitboxes = [];

  props.chunks.forEach((chunk, chunkIndex) => {
    const element = findChunkTextElement(chunk.chunk_id);

    getSpanAnnotations(chunk).forEach((annotation, annotationIndex) => {
      const range = buildCustomHighlightRange(element, annotation, chunk.text || '');
      if (!range) {
        return;
      }

      const highlightName = getHighlightName(
        annotation.id,
        selectedAnnotationSet.value.has(annotation.id)
      );

      if (!groupedRanges.has(highlightName)) {
        groupedRanges.set(highlightName, []);
      }

      groupedRanges.get(highlightName).push(range);
      hitboxes.push(...buildHitboxesForRange(range, annotation, chunkIndex, annotationIndex));
    });
  });

  groupedRanges.forEach((ranges, highlightName) => {
    globalThis.CSS.highlights.set(highlightName, new globalThis.Highlight(...ranges));
    customHighlightNames.add(highlightName);
  });

  customHighlightHitboxes.value = hitboxes;
}

function clearPopover() {
  popoverVisible.value = false;
  popoverTarget.value = null;
  activeAnnotationId.value = null;
  activeSelectedText.value = '';
}

function openPopover(target, options = {}) {
  if (!target) {
    clearPopover();
    return;
  }

  popoverTarget.value = target;
  activeAnnotationId.value = options.annotationId || null;
  activeSelectedText.value = options.selectedText || '';
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
  });
}

function openCustomHighlightPopover(hitbox) {
  openPopover(hitbox.target, {
    annotationId: hitbox.annotationId,
    selectedText: hitbox.selectedText,
  });
}

function hitboxContainsPoint(hitbox, event) {
  if (!hitbox?.rect) {
    return false;
  }

  return (
    event.clientX >= hitbox.rect.left &&
    event.clientX <= hitbox.rect.right &&
    event.clientY >= hitbox.rect.top &&
    event.clientY <= hitbox.rect.bottom
  );
}

function handleTextSelection(chunk) {
  const selection = getCurrentSelection();

  if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
    return;
  }

  const chunkElement = findChunkTextElement(chunk.chunk_id);
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

  const rect = selection.getRangeAt(0).getBoundingClientRect();
  openPopover(rectToTarget(rect), { selectedText });
}

function handleChunkClick(chunk, event) {
  if (!supportsCustomHighlight || needsFallbackMarks(chunk)) {
    return;
  }

  if (hasActiveTextSelection(getCurrentSelection())) {
    return;
  }

  const matchedHitbox = customHighlightHitboxes.value.find(
    (hitbox) => hitbox.chunkId === chunk.chunk_id && hitboxContainsPoint(hitbox, event)
  );

  if (!matchedHitbox) {
    return;
  }

  openCustomHighlightPopover(matchedHitbox);
}

watch(
  () => [props.chunks, props.selectedAnnotationIds],
  async () => {
    await nextTick();
    await syncCustomHighlights();
  },
  { deep: true, immediate: true, flush: 'post' }
);

onMounted(() => {
  nextTick(() => {
    syncCustomHighlights();
  });
});

onBeforeUnmount(() => {
  clearCustomHighlights();
  customHighlightStyleElement?.remove();
  customHighlightStyleElement = null;
});
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
}

.chunk-text mark {
  background: rgba(var(--v-theme-warning), 0.24);
  color: inherit;
  border-bottom: 1px solid rgba(var(--v-theme-warning), 0.72);
  border-radius: 0.2rem;
  padding: 0 0.08rem;
}

.chunk-text mark.annotated-mark--selected {
  background: rgba(var(--v-theme-error), 0.22);
  border-bottom-color: rgba(var(--v-theme-error), 0.8);
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
