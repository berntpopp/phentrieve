<template>
  <v-dialog
    :model-value="modelValue"
    max-width="560"
    scrollable
    @update:model-value="$emit('update:modelValue', $event)"
  >
    <v-card class="hpo-term-picker" role="dialog" :aria-labelledby="titleId">
      <v-card-title :id="titleId" class="hpo-term-picker__title">
        <v-icon start size="small">
          {{ mode === 'add' ? 'mdi-plus-circle-outline' : 'mdi-swap-horizontal' }}
        </v-icon>
        {{ mode === 'add' ? t('hpoTermPicker.addTitle') : t('hpoTermPicker.changeTitle') }}
      </v-card-title>

      <div class="hpo-term-picker__span px-4">
        <span class="text-caption text-medium-emphasis">{{ t('hpoTermPicker.spanLabel') }}:</span>
        <span class="font-italic">“{{ spanText }}”</span>
      </div>

      <div class="px-4 pt-2">
        <v-text-field
          v-model="searchText"
          data-testid="hpo-picker-search"
          :label="t('hpoTermPicker.searchLabel')"
          density="comfortable"
          variant="outlined"
          hide-details
          clearable
          prepend-inner-icon="mdi-magnify"
          @update:model-value="onSearchInput"
          @keydown.enter.prevent="emitRequery"
        />
        <div
          class="hpo-term-picker__assertion mt-3"
          role="group"
          :aria-label="
            t('hpoTermPicker.assertionAffirmed') + ' / ' + t('hpoTermPicker.assertionNegated')
          "
        >
          <v-btn
            size="small"
            data-testid="hpo-picker-affirmed"
            :variant="assertionValue === 'affirmed' ? 'flat' : 'outlined'"
            :color="assertionValue === 'affirmed' ? 'success' : undefined"
            :aria-pressed="assertionValue === 'affirmed' ? 'true' : 'false'"
            @click="assertionValue = 'affirmed'"
          >
            <v-icon start size="small">mdi-check-circle-outline</v-icon>
            {{ t('hpoTermPicker.assertionAffirmed') }}
          </v-btn>
          <v-btn
            size="small"
            class="ml-2"
            data-testid="hpo-picker-negated"
            :variant="assertionValue === 'negated' ? 'flat' : 'outlined'"
            :color="assertionValue === 'negated' ? 'error' : undefined"
            :aria-pressed="assertionValue === 'negated' ? 'true' : 'false'"
            @click="assertionValue = 'negated'"
          >
            <v-icon start size="small">mdi-block-helper</v-icon>
            {{ t('hpoTermPicker.assertionNegated') }}
          </v-btn>
        </div>
      </div>

      <v-card-text class="hpo-term-picker__results">
        <div v-if="loading" class="d-flex align-center justify-center py-6 text-medium-emphasis">
          <v-progress-circular indeterminate size="22" class="mr-2" />
          {{ t('hpoTermPicker.loading') }}
        </div>
        <div
          v-else-if="candidates.length === 0"
          data-testid="hpo-picker-empty"
          class="text-medium-emphasis py-6 text-center"
        >
          {{ t('hpoTermPicker.empty') }}
        </div>
        <v-list v-else role="listbox" density="comfortable" class="hpo-term-picker__list">
          <v-list-item
            v-for="candidate in candidates"
            :key="candidate.hpo_id"
            data-testid="hpo-candidate"
            role="option"
            :aria-selected="candidate.hpo_id === selectedId"
            :active="candidate.hpo_id === selectedId"
            @click="selectedId = candidate.hpo_id"
          >
            <template #prepend>
              <v-icon size="small">
                {{ candidate.hpo_id === selectedId ? 'mdi-radiobox-marked' : 'mdi-radiobox-blank' }}
              </v-icon>
            </template>
            <v-list-item-title class="d-flex align-center justify-space-between">
              <span>{{ candidate.label }}</span>
              <v-chip size="x-small" label class="ml-2">
                {{ t('hpoTermPicker.scoreLabel') }} {{ formatScore(candidate.similarity) }}
              </v-chip>
            </v-list-item-title>
            <v-list-item-subtitle class="hpo-term-picker__meta">
              <span class="font-weight-medium">{{ candidate.hpo_id }}</span>
              <span v-if="candidate.definition"> — {{ truncate(candidate.definition) }}</span>
            </v-list-item-subtitle>
          </v-list-item>
        </v-list>
      </v-card-text>

      <v-card-actions class="justify-end">
        <v-btn data-testid="hpo-picker-cancel" variant="text" @click="$emit('cancel')">
          {{ t('hpoTermPicker.cancel') }}
        </v-btn>
        <v-btn
          data-testid="hpo-picker-submit"
          color="primary"
          variant="flat"
          :disabled="!selectedCandidate"
          @click="submit"
        >
          {{ mode === 'add' ? t('hpoTermPicker.add') : t('hpoTermPicker.replace') }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

const props = defineProps({
  modelValue: { type: Boolean, default: false },
  mode: { type: String, default: 'replace' }, // 'replace' | 'add'
  spanText: { type: String, default: '' },
  candidates: { type: Array, default: () => [] },
  loading: { type: Boolean, default: false },
  assertion: { type: String, default: 'affirmed' },
});

const emit = defineEmits(['update:modelValue', 'requery', 'submit', 'cancel']);

let i18n = null;
try {
  i18n = useI18n();
} catch {
  i18n = null;
}
function t(key) {
  return i18n ? i18n.t(key) : key;
}

const titleId = `hpo-term-picker-title-${Math.round(performance.now())}`;
const searchText = ref(props.spanText);
const selectedId = ref(null);
const assertionValue = ref(props.assertion || 'affirmed');

let debounceTimer = null;

watch(
  () => props.modelValue,
  (open) => {
    if (open) {
      searchText.value = props.spanText;
      selectedId.value = null;
      assertionValue.value = props.assertion || 'affirmed';
    }
  }
);

watch(
  () => props.assertion,
  (next) => {
    assertionValue.value = next || 'affirmed';
  }
);

const selectedCandidate = computed(
  () => props.candidates.find((c) => c.hpo_id === selectedId.value) || null
);

function emitRequery() {
  const value = (searchText.value || '').trim();
  if (value) emit('requery', value);
}

function onSearchInput() {
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(emitRequery, 220);
}

function submit() {
  if (!selectedCandidate.value) return;
  emit('submit', { term: selectedCandidate.value, assertion: assertionValue.value });
}

function formatScore(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(2) : '—';
}

function truncate(text, max = 120) {
  if (typeof text !== 'string') return '';
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}
</script>

<style scoped>
.hpo-term-picker__title {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-weight: 600;
}

.hpo-term-picker__span {
  display: flex;
  gap: 0.4rem;
  align-items: baseline;
  flex-wrap: wrap;
}

.hpo-term-picker__results {
  min-height: 160px;
}

.hpo-term-picker__meta {
  white-space: normal;
}
</style>
