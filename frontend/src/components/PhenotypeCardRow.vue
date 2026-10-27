<template>
  <v-list-item
    class="phenotype-card-row mb-1 rounded-lg"
    :class="{ 'phenotype-card-row--clickable': clickable }"
    :color="color"
    border
    density="compact"
    @click="clickable ? $emit('click') : undefined"
  >
    <template v-if="$slots.prepend" #prepend>
      <slot name="prepend" />
    </template>

    <v-list-item-title class="pb-2">
      <div class="d-flex flex-wrap align-center">
        <div class="flex-grow-1">
          <div class="d-flex align-center flex-wrap ga-1 mb-1">
            <span
              class="text-body-1 font-weight-medium text-high-emphasis phenotype-card-row__label"
              >{{ label }}</span
            >
            <slot name="inline-tools" />
          </div>
          <a
            :href="hpoTermUrl(hpoId)"
            target="_blank"
            rel="noopener noreferrer"
            class="phenotype-card-row__hpo-link text-caption"
            :title="`View ${hpoId} in HPO Browser`"
            @click.stop
          >
            <span class="phenotype-card-row__hpo-id">{{ hpoId }}</span>
            <v-icon size="x-small" class="ml-1">mdi-open-in-new</v-icon>
          </a>
        </div>

        <div v-if="$slots.actions" class="d-flex align-center ml-auto phenotype-card-row__actions">
          <slot name="actions" />
        </div>
      </div>

      <div v-if="$slots.metadata" class="mt-2">
        <slot name="metadata" />
      </div>

      <slot name="details" />
    </v-list-item-title>
  </v-list-item>
</template>

<script setup>
import { HPO_TERM_URL } from '../constants/urls';

defineProps({
  hpoId: { type: String, required: true },
  label: { type: String, required: true },
  color: { type: String, default: 'grey-lighten-5' },
  clickable: { type: Boolean, default: false },
});

defineEmits(['click']);

function hpoTermUrl(hpoId) {
  return HPO_TERM_URL(hpoId);
}
</script>

<style scoped>
.phenotype-card-row__hpo-id {
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.phenotype-card-row__hpo-link {
  text-decoration: none;
  color: rgba(var(--v-theme-on-surface), var(--v-medium-emphasis-opacity));
  display: inline-flex;
  align-items: center;
  transition: color 0.2s;
}

.phenotype-card-row__hpo-link:hover {
  color: rgb(var(--v-theme-primary));
  text-decoration: underline;
  text-underline-offset: 2px;
}

.phenotype-card-row__label {
  max-width: 100%;
  display: inline-block;
}

.phenotype-card-row__actions {
  gap: 4px;
}

/* Vuetify's default prepend spacer leaves a wide gap after the rank badge */
.phenotype-card-row :deep(.v-list-item__prepend .v-list-item__spacer) {
  width: 8px;
}

.phenotype-card-row--clickable {
  cursor: pointer;
}
</style>
