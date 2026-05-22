<template>
  <v-sheet
    rounded="xl"
    elevation="0"
    :class="[
      'search-bar',
      isTextProcessModeActive ? 'search-bar--text-process' : 'search-bar--query',
    ]"
    color="white"
  >
    <div
      :class="[
        'search-shell',
        'd-flex',
        'flex-wrap',
        'flex-sm-nowrap',
        isTextProcessModeActive ? 'search-shell--text-process' : 'align-center',
      ]"
    >
      <v-textarea
        v-if="isTextProcessModeActive"
        v-model="queryText"
        density="comfortable"
        variant="outlined"
        hide-details
        class="search-input search-input--text-process ml-2 ml-sm-3 flex-grow-1"
        :disabled="isLoading"
        bg-color="white"
        color="primary"
        rows="3"
        auto-grow
        clearable
        placeholder="Paste or type a clinical note"
        :aria-label="textProcessInputLabel"
        :aria-description="textProcessInputDescription"
        @keydown.enter.prevent="canSubmit ? emitSubmit() : null"
      >
        <template #label>
          <span class="text-high-emphasis">Clinical note</span>
        </template>
      </v-textarea>
      <v-text-field
        v-else
        ref="queryInput"
        v-model="queryText"
        density="comfortable"
        variant="outlined"
        hide-details
        class="search-input search-input--query ml-2 ml-sm-3 flex-grow-1"
        :disabled="isLoading"
        bg-color="white"
        color="primary"
        clearable
        :aria-label="queryInputLabel"
        :aria-description="queryInputDescription"
        @keydown.enter.prevent="canSubmit ? emitSubmit() : null"
      >
        <template #label>
          <span class="text-high-emphasis">Phenotype query</span>
        </template>
      </v-text-field>

      <div
        :class="[
          'search-action',
          'd-flex',
          'align-center',
          isTextProcessModeActive ? 'search-action--overlay' : null,
        ]"
      >
        <v-btn
          ref="searchButton"
          color="primary"
          variant="text"
          icon
          rounded="circle"
          :loading="isLoading"
          :disabled="!queryText.trim()"
          class="search-submit-button"
          :aria-label="searchButtonLabel"
          size="small"
          data-tutorial-step="search-button"
          @click="emitSubmit"
        >
          <v-icon>mdi-magnify</v-icon>
        </v-btn>
      </div>
    </div>
  </v-sheet>
</template>

<script>
export default {
  name: 'QueryForm',
  props: {
    modelValue: {
      type: String,
      default: '',
    },
    isTextProcessModeActive: {
      type: Boolean,
      default: false,
    },
    isLoading: {
      type: Boolean,
      default: false,
    },
    queryInputLabel: {
      type: String,
      required: true,
    },
    queryInputDescription: {
      type: String,
      required: true,
    },
    textProcessInputLabel: {
      type: String,
      required: true,
    },
    textProcessInputDescription: {
      type: String,
      required: true,
    },
    searchButtonLabel: {
      type: String,
      required: true,
    },
  },
  emits: ['update:modelValue', 'submit'],
  computed: {
    queryText: {
      get() {
        return this.modelValue;
      },
      set(value) {
        this.$emit('update:modelValue', value ?? '');
      },
    },
    canSubmit() {
      return !this.isLoading && this.queryText.trim();
    },
  },
  methods: {
    emitSubmit() {
      this.$emit('submit');
    },
  },
};
</script>

<style scoped>
.search-input {
  font-size: 1rem;
  line-height: 1.5;
}

.search-shell {
  gap: 8px;
}

.search-shell--text-process {
  align-items: flex-start;
}

.search-action {
  flex-shrink: 0;
}

.search-action--overlay {
  position: absolute;
  top: 12px;
  right: 14px;
  z-index: 1;
}

.search-submit-button {
  width: 34px !important;
  height: 34px !important;
  min-width: 34px !important;
  color: rgba(25, 82, 166, 0.7) !important;
}

.search-submit-button:hover {
  background: rgba(25, 82, 166, 0.08);
}

.search-input--text-process {
  margin-right: 64px !important;
}

.search-input--text-process :deep(.v-label) {
  font-size: 0.92rem;
  color: rgba(60, 72, 88, 0.62);
}

.search-input :deep(.v-field) {
  /* Removed box-shadow to rely on the search-bar border instead */
}

.search-input--query :deep(.v-field) {
  border-radius: 28px;
  min-height: 48px;
}

.search-input--text-process :deep(.v-field) {
  border-radius: 22px;
  min-height: 104px;
  padding-top: 8px;
  padding-bottom: 8px;
}

.search-input--text-process :deep(.v-field__input) {
  min-height: 88px;
}

.search-bar {
  border: 1px solid rgba(0, 0, 0, 0.15);
  border-radius: 24px;
  transition:
    padding 0.18s ease,
    border-color 0.18s ease;
}

.search-bar--query {
  padding: 6px 8px;
}

.search-bar--text-process {
  position: relative;
  padding: 10px 12px;
}

.search-input :deep(.v-field__outline) {
  --v-field-border-width: 0px;
}

@media (max-width: 600px) {
  .search-bar--query {
    padding: 4px 6px;
  }

  .search-bar--text-process {
    padding: 8px 10px;
    border-radius: 20px;
  }

  .search-shell--text-process {
    gap: 6px;
  }

  .search-action {
    align-self: flex-start;
    padding-top: 2px;
  }

  .search-action--overlay {
    top: 10px;
    right: 10px;
    padding-top: 0;
  }

  .search-input--text-process {
    margin-right: 56px !important;
  }

  .search-input--text-process :deep(.v-field) {
    border-radius: 18px;
    min-height: 96px;
  }

  .search-input--text-process :deep(.v-field__input) {
    min-height: 72px;
  }

  .search-input--text-process :deep(.v-label) {
    font-size: 0.88rem;
  }
}
</style>
