<template>
  <section class="annotated-document-pane">
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
          :ref="(el) => setChunkTextRef(chunk.chunk_id, el)"
          class="chunk-text"
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
  </section>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, watch } from 'vue';

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
  typeof CSS !== 'undefined' &&
  typeof Highlight !== 'undefined' &&
  typeof CSS.highlights !== 'undefined';

const chunkTextRefs = new Map();
const customHighlightNames = new Set();

const selectedAnnotationSet = computed(() => new Set(props.selectedAnnotationIds));

function setChunkTextRef(chunkId, element) {
  if (element) {
    chunkTextRefs.set(chunkId, element);
    return;
  }

  chunkTextRefs.delete(chunkId);
}

function getAnnotations(chunk) {
  return Array.isArray(chunk.annotations) ? [...chunk.annotations] : [];
}

function needsFallbackMarks(chunk) {
  return chunk.evidence_mode === 'span' && getAnnotations(chunk).length > 0 && !supportsCustomHighlight;
}

function buildMarkedSegments(chunk) {
  const text = chunk.text || '';
  const annotations = getAnnotations(chunk)
    .filter((item) => item.start_char != null && item.end_char != null)
    .sort((left, right) => left.start_char - right.start_char);

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
      const annotationId = annotation.id || `annotation-${index}`;

      segments.push({
        key: `annotation-${annotationId}-${start}-${end}`,
        text: text.slice(start, end),
        annotationId,
        selected: selectedAnnotationSet.value.has(annotationId),
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

function clearCustomHighlights() {
  if (!supportsCustomHighlight) {
    return;
  }

  customHighlightNames.forEach((name) => {
    CSS.highlights.delete(name);
  });
  customHighlightNames.clear();
}

function applyCustomHighlights(chunkId, element, annotations) {
  if (!supportsCustomHighlight || !element) return;

  const highlightName = `chunk-${chunkId}`;
  const ranges = annotations
    .filter((item) => item.start_char != null && item.end_char != null)
    .map((item) => {
      const firstTextNode = element.firstChild;
      if (!firstTextNode) return null;

      const range = new Range();
      range.setStart(firstTextNode, item.start_char);
      range.setEnd(firstTextNode, item.end_char);
      return range;
    })
    .filter(Boolean);

  if (ranges.length === 0) {
    CSS.highlights.delete(highlightName);
    customHighlightNames.delete(highlightName);
    return;
  }

  CSS.highlights.set(highlightName, new Highlight(...ranges));
  customHighlightNames.add(highlightName);
}

async function syncCustomHighlights() {
  if (!supportsCustomHighlight) {
    return;
  }

  clearCustomHighlights();

  props.chunks.forEach((chunk) => {
    const element = chunkTextRefs.get(chunk.chunk_id);
    applyCustomHighlights(chunk.chunk_id, element, getAnnotations(chunk));
  });
}

watch(
  () => props.chunks,
  async () => {
    await nextTick();
    await syncCustomHighlights();
  },
  { deep: true, immediate: true }
);

onMounted(() => {
  syncCustomHighlights();
});

onBeforeUnmount(() => {
  clearCustomHighlights();
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
