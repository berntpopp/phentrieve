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
      @inspect="noop"
      @add-to-case="noop"
      @change-term="noop"
      @remove-annotation="noop"
    />
  </section>
</template>

<script setup>
import {
  computed,
  defineComponent,
  getCurrentInstance,
  h,
  nextTick,
  onBeforeUnmount,
  ref,
  watch,
} from 'vue';
import AnnotationActionPopover from './AnnotationActionPopover.vue';

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
      const annotation = props.annotations[index];

      if (!annotation) {
        return props.text;
      }

      const child =
        index === props.annotations.length - 1
          ? props.text
          : renderAt(index + 1);

      return h(
        'mark',
        {
          'data-annotation-id': annotation.id,
          'aria-details': `annotation-detail-${annotation.id}`,
          class: { 'annotated-mark--selected': annotation.selected },
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

const supportsCustomHighlight =
  typeof globalThis.CSS !== 'undefined' &&
  typeof globalThis.Highlight !== 'undefined' &&
  typeof globalThis.CSS.highlights !== 'undefined';

const paneInstanceId = `pane-${getCurrentInstance()?.uid ?? 'unknown'}`;
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

function getChunkAnnotationDetails(chunk) {
  const detailsById = new Map();

  getSpanAnnotations(chunk).forEach((annotation) => {
    if (!detailsById.has(annotation.id)) {
      detailsById.set(annotation.id, {
        id: annotation.id,
        detailText: annotation.matched_text_in_chunk || '',
      });
    }
  });

  return [...detailsById.values()];
}

function needsFallbackMarks(chunk) {
  return getSpanAnnotations(chunk).length > 0 && !supportsCustomHighlight;
}

function buildMarkedSegments(chunk) {
  const text = chunk.text || '';
  const annotations = getSpanAnnotations(chunk).sort((left, right) => left.start_char - right.start_char);
  const boundaries = Array.from(
    new Set([0, text.length, ...annotations.flatMap((annotation) => [annotation.start_char, annotation.end_char])])
  ).sort((left, right) => left - right);
  const segments = [];

  for (let index = 0; index < boundaries.length - 1; index += 1) {
    const start = boundaries[index];
    const end = boundaries[index + 1];

    if (end <= start) {
      continue;
    }

    const activeAnnotations = annotations
      .filter((annotation) => annotation.start_char < end && annotation.end_char > start)
      .map((annotation) => ({
        id: annotation.id,
        selected: selectedAnnotationSet.value.has(annotation.id),
        detailText: annotation.matched_text_in_chunk || text.slice(start, end),
        start_char: annotation.start_char,
        end_char: annotation.end_char,
      }))
      .sort((left, right) => {
        if (left.start_char !== right.start_char) {
          return left.start_char - right.start_char;
        }

        return right.end_char - left.end_char;
      });

    const nextSegment = {
      key: `segment-${start}-${end}`,
      text: text.slice(start, end),
      annotations: activeAnnotations,
    };
    const previousSegment = segments[segments.length - 1];
    const previousSignature = previousSegment?.annotations?.map((annotation) => annotation.id).join('|') || '';
    const nextSignature = activeAnnotations.map((annotation) => annotation.id).join('|');

    if (previousSegment && previousSignature === nextSignature) {
      previousSegment.text += nextSegment.text;
      previousSegment.key = `${previousSegment.key}-${end}`;
      continue;
    }

    segments.push(nextSegment);
  }

  return segments;
}

function getHighlightName(annotationId, selected) {
  return selected
    ? `${paneInstanceId}-annotation-selected-${annotationId}`
    : `${paneInstanceId}-annotation-${annotationId}`;
}

function ensureCustomHighlightStyleElement() {
  if (!supportsCustomHighlight) {
    return null;
  }

  if (!customHighlightStyleElement) {
    customHighlightStyleElement = document.createElement('style');
    customHighlightStyleElement.setAttribute('data-annotated-document-highlight-style', 'true');
    customHighlightStyleElement.setAttribute('data-highlight-owner', paneInstanceId);
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
    startChar: annotation.start_char,
    endChar: annotation.end_char,
    annotationIndex,
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

function compareHitboxesBySpecificity(left, right) {
  const leftWidth = left.endChar - left.startChar;
  const rightWidth = right.endChar - right.startChar;

  if (leftWidth !== rightWidth) {
    return leftWidth - rightWidth;
  }

  if (left.startChar !== right.startChar) {
    return right.startChar - left.startChar;
  }

  if (left.endChar !== right.endChar) {
    return left.endChar - right.endChar;
  }

  return right.annotationIndex - left.annotationIndex;
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

  const matchingHitboxes = customHighlightHitboxes.value
    .filter((hitbox) => hitbox.chunkId === chunk.chunk_id && hitboxContainsPoint(hitbox, event))
    .sort(compareHitboxesBySpecificity);

  const resolvedHitbox = matchingHitboxes[0] || matchedHitbox;

  if (!resolvedHitbox) {
    return;
  }

  openCustomHighlightPopover(resolvedHitbox);
}

watch(
  () => [props.chunks, props.selectedAnnotationIds],
  async () => {
    await nextTick();
    await syncCustomHighlights();
  },
  { deep: true, immediate: true, flush: 'post' }
);

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
