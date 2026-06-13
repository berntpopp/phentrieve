<template>
  <div data-testid="user-note-summary" class="user-note-summary">
    <div class="user-note-summary__header">
      <v-icon size="small" color="primary">mdi-file-document-outline</v-icon>
      <span class="text-body-2 font-weight-medium">Clinical note</span>
      <span class="text-caption text-medium-emphasis">
        {{ meta }}
      </span>
      <v-btn
        data-testid="user-note-summary-toggle"
        size="x-small"
        variant="text"
        icon
        :aria-label="expanded ? 'Collapse clinical note' : 'Expand clinical note'"
        @click="$emit('toggle')"
      >
        <v-icon size="small">
          {{ expanded ? 'mdi-chevron-up' : 'mdi-chevron-down' }}
        </v-icon>
      </v-btn>
    </div>
    <p v-if="!expanded" class="mb-0 text-body-2 text-medium-emphasis user-note-summary__preview">
      {{ summary }}
    </p>
    <div
      v-if="expanded"
      ref="expandedContainer"
      data-testid="user-note-expanded"
      class="user-note-summary__expanded"
      @mouseup="onNoteMouseUp"
    >
      <span v-for="segment in segments" :key="segment.key">
        <v-tooltip
          v-if="segment.highlighted"
          location="top"
          :open-delay="180"
          open-on-focus
          max-width="280"
          content-class="annotated-note-tooltip"
          :content-props="{ 'aria-label': segment.tooltip }"
        >
          <template #activator="{ props }">
            <mark
              v-bind="props"
              data-testid="annotated-note-span"
              class="annotated-note-span"
              :class="{
                'annotated-note-span--active':
                  activePhenotypeId && segment.termIds.includes(activePhenotypeId),
              }"
              role="button"
              aria-haspopup="menu"
              tabindex="0"
              :aria-label="`Edit annotation: ${segment.tooltip}`"
              @mouseenter="$emit('hover', segment.termIds)"
              @mouseleave="$emit('clear-hover')"
              @focus="$emit('hover', segment.termIds)"
              @blur="$emit('clear-hover')"
              @click="activateSpan(segment, $event)"
              @contextmenu.prevent="activateSpan(segment, $event)"
              @keydown.enter.prevent="activateSpan(segment, $event)"
              @keydown.space.prevent="activateSpan(segment, $event)"
            >
              {{ segment.text }}
            </mark>
          </template>
          <div class="annotated-note-tooltip__content">
            <div class="annotated-note-tooltip__eyebrow">Linked phenotype</div>
            <div class="annotated-note-tooltip__label">{{ segment.tooltip }}</div>
          </div>
        </v-tooltip>
        <span v-else>{{ segment.text }}</span>
      </span>
    </div>
  </div>
</template>

<script>
import { computeSelectionOffsets } from '../../composables/useUserNoteAnnotations';

export default {
  name: 'FullTextWorkspace',
  props: {
    summary: {
      type: String,
      required: true,
    },
    meta: {
      type: String,
      required: true,
    },
    expanded: {
      type: Boolean,
      default: false,
    },
    segments: {
      type: Array,
      default: () => [],
    },
    activePhenotypeId: {
      type: String,
      default: null,
    },
  },
  emits: ['toggle', 'hover', 'clear-hover', 'span-activate', 'text-select'],
  methods: {
    activateSpan(segment, event) {
      const rect = event?.currentTarget?.getBoundingClientRect?.() || null;
      this.$emit('span-activate', {
        annotationIds: Array.isArray(segment.annotationIds) ? segment.annotationIds : [],
        termIds: Array.isArray(segment.termIds) ? segment.termIds : [],
        rect,
        text: segment.text,
      });
    },
    onNoteMouseUp() {
      const selection =
        typeof window !== 'undefined' && window.getSelection ? window.getSelection() : null;
      if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
        return;
      }
      const text = selection.toString().trim();
      if (!text) {
        return;
      }
      const container = this.$refs.expandedContainer;
      if (!container) {
        return;
      }
      const range = selection.getRangeAt(0);
      if (!container.contains(range.startContainer) || !container.contains(range.endContainer)) {
        return;
      }
      // Selections fully inside a single existing mark are edited via click, not
      // turned into a new manual annotation.
      const startEl =
        range.startContainer.nodeType === 1
          ? range.startContainer
          : range.startContainer.parentElement;
      const endEl =
        range.endContainer.nodeType === 1 ? range.endContainer : range.endContainer.parentElement;
      const startMark = startEl?.closest?.('.annotated-note-span') || null;
      const endMark = endEl?.closest?.('.annotated-note-span') || null;
      if (startMark && startMark === endMark) {
        return;
      }
      const offsets = computeSelectionOffsets(container, range);
      if (!offsets) {
        return;
      }
      const rect = range.getBoundingClientRect?.() || null;
      this.$emit('text-select', { text, start: offsets.start, end: offsets.end, rect });
    },
  },
};
</script>

<style scoped>
.user-note-summary {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.user-note-summary__header {
  display: flex;
  align-items: center;
  gap: 8px;
}

.user-note-summary__header .text-caption {
  margin-left: auto;
}

.user-note-summary__preview {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.user-note-summary__expanded {
  margin-top: 8px;
  font-size: 0.95rem;
  line-height: 1.6;
  color: rgba(15, 23, 42, 0.9);
}

.annotated-note-span,
.user-note-summary__expanded mark {
  background: rgba(37, 99, 235, 0.16);
  color: inherit;
  border-radius: 4px;
  box-shadow: inset 0 -0.42em 0 rgba(37, 99, 235, 0.14);
  cursor: pointer;
  transition:
    background-color 0.18s ease,
    box-shadow 0.18s ease;
}

.annotated-note-span:hover,
.user-note-summary__expanded mark:hover {
  background: rgba(37, 99, 235, 0.22);
  box-shadow: inset 0 -0.5em 0 rgba(37, 99, 235, 0.18);
}

.annotated-note-span--active {
  background: rgba(37, 99, 235, 0.32) !important;
  box-shadow: inset 0 -0.5em 0 rgba(37, 99, 235, 0.24);
}

/* Keyboard focus parity: a focused span gets the same emphasis as hover, plus a
   visible focus ring for non-pointer users (WCAG 2.4.7 / 1.4.13). */
.annotated-note-span:focus-visible {
  outline: 2px solid rgb(var(--v-theme-primary));
  outline-offset: 2px;
  background: rgba(37, 99, 235, 0.22);
}
</style>

<!--
  The v-tooltip content is teleported to <body>, outside this component's
  scoped styles, so a scoped :deep() rule can never reach it. Style it from a
  GLOBAL block instead. Vuetify's default tooltip surface is dark
  (surface-variant); we override it with the theme surface/on-surface tokens so
  the text stays readable (WCAG AA contrast) in BOTH light and dark themes.
  The selector includes `.v-tooltip >` to out-specify Vuetify's own
  `.v-tooltip > .v-overlay__content` rule deterministically.
-->
<style>
.v-tooltip > .v-overlay__content.annotated-note-tooltip {
  border-radius: 14px;
  border: 1px solid rgba(var(--v-theme-on-surface), 0.12);
  background: rgb(var(--v-theme-surface));
  color: rgb(var(--v-theme-on-surface));
  box-shadow: 0 16px 36px rgba(0, 0, 0, 0.18);
  padding: 10px 12px;
}

.annotated-note-tooltip .annotated-note-tooltip__content {
  display: grid;
  gap: 4px;
}

.annotated-note-tooltip .annotated-note-tooltip__eyebrow {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: rgb(var(--v-theme-primary));
}

.annotated-note-tooltip .annotated-note-tooltip__label {
  font-size: 0.82rem;
  line-height: 1.4;
  color: rgb(var(--v-theme-on-surface));
}
</style>
