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
          <div class="d-flex align-center mb-1">
            <a
              :href="hpoTermUrl(hpoId)"
              target="_blank"
              rel="noopener noreferrer"
              class="phenotype-card-row__hpo-link"
              :title="`View ${hpoId} in HPO Browser`"
              @click.stop
            >
              <span class="phenotype-card-row__hpo-id font-weight-bold">{{ hpoId }}</span>
              <v-icon size="x-small" class="ml-1">mdi-open-in-new</v-icon>
            </a>
            <slot name="inline-tools" />
          </div>
          <div class="d-flex align-center">
            <span class="text-body-2 text-high-emphasis phenotype-card-row__label">{{
              label
            }}</span>
          </div>
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
  font-weight: bold;
  white-space: nowrap;
}

.phenotype-card-row__hpo-link {
  text-decoration: none;
  color: inherit;
  display: inline-flex;
  align-items: center;
  transition: color 0.2s;
}

.phenotype-card-row__hpo-link:hover {
  color: var(--v-theme-primary);
}

.phenotype-card-row__label {
  max-width: 100%;
  display: inline-block;
}

.phenotype-card-row__actions {
  gap: 4px;
}

.phenotype-card-row--clickable {
  cursor: pointer;
}
</style>
