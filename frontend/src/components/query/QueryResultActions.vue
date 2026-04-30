<template>
  <PhenotypeCollectionPanel
    v-model:subject-id="subjectIdProxy"
    v-model:sex="sexProxy"
    v-model:date-of-birth="dateOfBirthProxy"
    :phenotypes="phenotypes"
    :panel-open="panelOpen"
    :sex-options="sexOptions"
    @toggle-panel="$emit('toggle-panel')"
    @update:panel-open="$emit('update:panel-open', $event)"
    @remove="$emit('remove', $event)"
    @toggle-assertion="$emit('toggle-assertion', $event)"
    @export-text="$emit('export-text')"
    @export-json="$emit('export-json')"
    @clear="$emit('clear')"
  />

  <v-snackbar
    v-model="exportErrorVisibleProxy"
    color="error"
    location="bottom"
    timeout="6000"
    role="alert"
  >
    {{ exportErrorMessage }}
    <template #actions>
      <v-btn variant="text" @click="exportErrorVisibleProxy = false">
        {{ dismissLabel }}
      </v-btn>
    </template>
  </v-snackbar>
</template>

<script>
import PhenotypeCollectionPanel from '../PhenotypeCollectionPanel.vue';

export default {
  name: 'QueryResultActions',
  components: {
    PhenotypeCollectionPanel,
  },
  props: {
    subjectId: {
      type: String,
      default: '',
    },
    sex: {
      type: [Number, String],
      default: 0,
    },
    dateOfBirth: {
      type: String,
      default: '',
    },
    phenotypes: {
      type: Array,
      default: () => [],
    },
    panelOpen: {
      type: Boolean,
      default: false,
    },
    sexOptions: {
      type: Array,
      default: () => [],
    },
    exportErrorVisible: {
      type: Boolean,
      default: false,
    },
    exportErrorMessage: {
      type: String,
      default: '',
    },
    dismissLabel: {
      type: String,
      required: true,
    },
  },
  emits: [
    'update:subject-id',
    'update:sex',
    'update:date-of-birth',
    'update:panel-open',
    'update:export-error-visible',
    'toggle-panel',
    'remove',
    'toggle-assertion',
    'export-text',
    'export-json',
    'clear',
  ],
  computed: {
    subjectIdProxy: {
      get() {
        return this.subjectId;
      },
      set(value) {
        this.$emit('update:subject-id', value);
      },
    },
    sexProxy: {
      get() {
        return this.sex;
      },
      set(value) {
        this.$emit('update:sex', value);
      },
    },
    dateOfBirthProxy: {
      get() {
        return this.dateOfBirth;
      },
      set(value) {
        this.$emit('update:date-of-birth', value);
      },
    },
    exportErrorVisibleProxy: {
      get() {
        return this.exportErrorVisible;
      },
      set(value) {
        this.$emit('update:export-error-visible', value);
      },
    },
  },
};
</script>
