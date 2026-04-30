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
    <div v-if="expanded" data-testid="user-note-expanded" class="user-note-summary__expanded">
      <span v-for="segment in segments" :key="segment.key">
        <v-tooltip
          v-if="segment.highlighted"
          location="top"
          :open-delay="180"
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
              @mouseenter="$emit('hover', segment.termIds)"
              @mouseleave="$emit('clear-hover')"
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
  emits: ['toggle', 'hover', 'clear-hover'],
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

:deep(.annotated-note-tooltip) {
  border-radius: 14px;
  border: 1px solid rgba(var(--v-theme-outline), 0.12);
  background: linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(255, 255, 255, 1));
  box-shadow: 0 16px 36px rgba(15, 23, 42, 0.14);
  padding: 10px 12px;
}

.annotated-note-tooltip__content {
  display: grid;
  gap: 4px;
}

.annotated-note-tooltip__eyebrow {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: rgba(37, 99, 235, 0.92);
}

.annotated-note-tooltip__label {
  font-size: 0.82rem;
  line-height: 1.4;
  color: rgba(15, 23, 42, 0.92);
}
</style>
