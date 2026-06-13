<template>
  <div class="full-text-note-curator">
    <FullTextWorkspace
      :summary="summary"
      :meta="meta"
      :expanded="expanded"
      :segments="curation.segments.value"
      :active-phenotype-id="activePhenotypeId"
      @toggle="$emit('toggle')"
      @hover="$emit('hover', $event)"
      @clear-hover="$emit('clear-hover')"
      @span-activate="onSpanActivate"
      @text-select="onTextSelect"
    />

    <AnnotationActionPopover
      :visible="popoverVisible"
      :target="popoverTarget"
      :mode="popoverMode"
      :can-revert="canRevert"
      :selected-text="activeSpanText"
      @update:visible="onPopoverVisibility"
      @close="closePopover"
      @change-term="onChangeTerm"
      @remove-annotation="onRemoveAnnotation"
      @add-to-collection="onAddToCollection"
      @annotate-selection="onAnnotateSelection"
      @revert="onRevert"
    />

    <HpoTermPickerDialog
      v-model="dialogOpen"
      :mode="dialogMode"
      :span-text="dialogSpanText"
      :candidates="dialogCandidates"
      :loading="dialogLoading"
      :assertion="dialogAssertion"
      @requery="runRequery"
      @submit="onDialogSubmit"
      @cancel="dialogOpen = false"
    />

    <v-snackbar
      v-model="snackbarOpen"
      data-testid="curation-undo-snackbar"
      timeout="6000"
      location="bottom"
    >
      {{ translate('annotatedDocumentPane.undo.removed', 'Annotation removed') }}
      <template #actions>
        <v-btn
          data-testid="curation-undo-action"
          variant="text"
          color="primary"
          @click="onUndo"
        >
          {{ translate('annotatedDocumentPane.undo.action', 'Undo') }}
        </v-btn>
      </template>
    </v-snackbar>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import FullTextWorkspace from './query/FullTextWorkspace.vue';
import AnnotationActionPopover from './AnnotationActionPopover.vue';
import HpoTermPickerDialog from './HpoTermPickerDialog.vue';
import { useFullTextCuration } from '../composables/useFullTextCuration';
import {
  summarizeDocumentQuery,
  formatDocumentSummaryMeta,
} from '../composables/useUserNoteAnnotations';

const props = defineProps({
  item: { type: Object, required: true },
  noteText: { type: String, default: '' },
  expanded: { type: Boolean, default: false },
  activePhenotypeId: { type: String, default: null },
  queryOptions: { type: Object, default: () => ({}) },
});

const emit = defineEmits(['toggle', 'hover', 'clear-hover', 'add-to-collection']);

let i18n = null;
try {
  i18n = useI18n();
} catch {
  i18n = null;
}
function translate(key, fallback) {
  return i18n ? i18n.t(key) : fallback;
}

const curation = useFullTextCuration(props.item.id);
curation.ensureSeeded(props.item, props.noteText);

watch(
  () => props.noteText,
  (next) => curation.setNoteText(next)
);

const summary = computed(() => summarizeDocumentQuery(props.noteText));
const meta = computed(() => formatDocumentSummaryMeta(props.noteText));

// Popover state
const popoverVisible = ref(false);
const popoverTarget = ref([0, 0]);
const popoverMode = ref('annotation'); // 'annotation' | 'selection'
const activeAnnotationId = ref(null);
const activeSpanText = ref('');
const selectionSpan = ref(null);

const canRevert = computed(() => {
  const ann = activeAnnotationId.value ? curation.annotationById(activeAnnotationId.value) : null;
  return Boolean(ann && (ann.origin === 'manual' || ann.replacedFrom));
});

function rectToTarget(rect) {
  if (!rect) return [0, 0];
  return [rect.left + rect.width / 2, rect.top];
}

function closePopover() {
  popoverVisible.value = false;
}

function onPopoverVisibility(next) {
  popoverVisible.value = next;
}

function onSpanActivate({ annotationIds, rect, text }) {
  const ids = Array.isArray(annotationIds) ? annotationIds : [];
  if (ids.length === 0) return;
  activeAnnotationId.value = ids[0];
  activeSpanText.value = text || '';
  selectionSpan.value = null;
  popoverMode.value = 'annotation';
  popoverTarget.value = rectToTarget(rect);
  popoverVisible.value = true;
}

function onTextSelect({ text, start, end, rect }) {
  selectionSpan.value = { start, end, text };
  activeAnnotationId.value = null;
  activeSpanText.value = text || '';
  popoverMode.value = 'selection';
  popoverTarget.value = rectToTarget(rect);
  popoverVisible.value = true;
}

// Dialog state
const dialogOpen = ref(false);
const dialogMode = ref('replace'); // 'replace' | 'add'
const dialogSpanText = ref('');
const dialogCandidates = ref([]);
const dialogLoading = ref(false);
const dialogAssertion = ref('affirmed');

async function runRequery(text) {
  dialogLoading.value = true;
  dialogCandidates.value = [];
  try {
    const { results, assertion } = await curation.requery(text, props.queryOptions);
    dialogCandidates.value = results;
    if (assertion === 'affirmed' || assertion === 'negated') {
      dialogAssertion.value = assertion;
    }
  } finally {
    dialogLoading.value = false;
  }
}

function openDialog(mode, spanText) {
  dialogMode.value = mode;
  dialogSpanText.value = spanText || '';
  dialogAssertion.value = 'affirmed';
  dialogCandidates.value = [];
  dialogOpen.value = true;
  if (dialogSpanText.value.trim()) runRequery(dialogSpanText.value.trim());
}

function onChangeTerm() {
  closePopover();
  openDialog('replace', activeSpanText.value);
}

function onAnnotateSelection() {
  closePopover();
  openDialog('add', activeSpanText.value);
}

function onRemoveAnnotation() {
  closePopover();
  if (!activeAnnotationId.value) return;
  curation.remove(activeAnnotationId.value);
  snackbarOpen.value = true;
}

function onRevert() {
  closePopover();
  if (activeAnnotationId.value) curation.revert(activeAnnotationId.value);
}

function onAddToCollection() {
  closePopover();
  const ann = activeAnnotationId.value ? curation.annotationById(activeAnnotationId.value) : null;
  if (!ann) return;
  emit('add-to-collection', {
    hpo_id: ann.hpoId,
    label: ann.label,
    assertion_status: ann.status,
  });
}

function onDialogSubmit({ term, assertion }) {
  if (dialogMode.value === 'add') {
    if (selectionSpan.value) curation.addManual(selectionSpan.value, term, assertion);
  } else if (activeAnnotationId.value) {
    curation.replace(activeAnnotationId.value, term, assertion);
  }
  dialogOpen.value = false;
}

// Undo snackbar
const snackbarOpen = ref(false);
function onUndo() {
  curation.undo();
  snackbarOpen.value = false;
}
</script>

<style scoped>
.full-text-note-curator {
  display: contents;
}
</style>
