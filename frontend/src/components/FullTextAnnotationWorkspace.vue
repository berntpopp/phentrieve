<template>
  <section
    class="full-text-workspace"
    :class="{ 'full-text-workspace--embedded': !showCaseWorkspace }"
  >
    <v-alert v-if="fallbackBanner" type="info" density="compact" variant="tonal" class="mb-3">
      {{ fallbackBanner }}
    </v-alert>

    <template v-if="!showCaseWorkspace">
      <PhenotypeFindingsPane
        :terms="workspaceTerms"
        @inspect-term="inspectTerm"
        @hover-term="highlightTerm"
        @clear-hover="clearHighlight"
        @add-all-to-collection="$emit('add-all-to-collection', $event)"
      />

      <section v-if="showDocumentPane" class="full-text-embedded__document">
        <div class="text-subtitle-2 mb-2">Evidence in note</div>
        <AnnotatedDocumentPane
          :chunks="workspaceChunks"
          :selected-annotation-ids="selectedAnnotationIds"
        />
      </section>
    </template>

    <template v-else>
      <div class="full-text-workspace__header d-flex justify-space-between align-center ga-3">
        <div class="full-text-workspace__title-block">
          <div class="text-subtitle-1">Full-text analysis</div>
          <div v-if="bannerText" class="text-caption text-medium-emphasis">
            {{ bannerText }}
          </div>
        </div>
        <v-btn
          variant="text"
          size="small"
          :icon="expanded ? 'mdi-chevron-up' : 'mdi-chevron-down'"
          @click="toggleExpanded"
        />
      </div>

      <v-expand-transition>
        <div v-show="expanded">
          <div class="workspace-mobile-nav">
            <v-btn
              v-if="showDocumentPane"
              size="small"
              rounded="pill"
              :variant="mobilePane === 'document' ? 'tonal' : 'text'"
              color="primary"
              @click="mobilePane = 'document'"
            >
              Document
            </v-btn>
            <v-btn
              size="small"
              rounded="pill"
              :variant="mobilePane === 'findings' ? 'tonal' : 'text'"
              color="primary"
              @click="mobilePane = 'findings'"
            >
              Findings
            </v-btn>
            <v-btn
              v-if="sidebarMode === SIDEBAR_MODE_INSPECTOR"
              size="small"
              rounded="pill"
              :variant="mobilePane === 'inspector' ? 'tonal' : 'text'"
              color="primary"
              @click="mobilePane = 'inspector'"
            >
              Inspector
            </v-btn>
            <v-btn
              v-if="showCaseWorkspace"
              size="small"
              rounded="pill"
              :variant="mobilePane === 'case' ? 'tonal' : 'text'"
              color="primary"
              @click="mobilePane = 'case'"
            >
              Case
            </v-btn>
          </div>

          <div
            class="workspace-layout workspace-layout--desktop"
            :class="{
              'workspace-layout--desktop-findings-only': !showDocumentPane,
              'workspace-layout--desktop-no-sidebar': !showSidebarPane,
            }"
          >
            <AnnotatedDocumentPane
              v-if="showDocumentPane"
              :chunks="workspaceChunks"
              :selected-annotation-ids="selectedAnnotationIds"
            />
            <PhenotypeFindingsPane
              :terms="workspaceTerms"
              @inspect-term="inspectTerm"
              @hover-term="highlightTerm"
              @clear-hover="clearHighlight"
              @add-all-to-collection="$emit('add-all-to-collection', $event)"
            />
            <AnnotationInspectorPanel
              v-if="sidebarMode === SIDEBAR_MODE_INSPECTOR"
              :selected-term="selectedTerm"
              @back="showCaseSidebar"
            />
            <CaseWorkspacePanel
              v-else-if="showCaseWorkspace"
              :cases="cases"
              :active-case-id="activeCaseId || ''"
              @create-case="createCase"
              @select-case="selectCase"
              @add-all="addAllPhenotypes"
              @export-case="exportActiveCase"
            />
          </div>

          <div class="workspace-layout workspace-layout--mobile">
            <AnnotatedDocumentPane
              v-if="showDocumentPane && mobilePane === 'document'"
              :chunks="workspaceChunks"
              :selected-annotation-ids="selectedAnnotationIds"
            />
            <AnnotationInspectorPanel
              v-else-if="sidebarMode === SIDEBAR_MODE_INSPECTOR && mobilePane === 'inspector'"
              :selected-term="selectedTerm"
              @back="showCaseSidebar"
            />
            <CaseWorkspacePanel
              v-else-if="showCaseWorkspace && mobilePane === 'case'"
              :cases="cases"
              :active-case-id="activeCaseId || ''"
              @create-case="createCase"
              @select-case="selectCase"
              @add-all="addAllPhenotypes"
              @export-case="exportActiveCase"
            />
            <PhenotypeFindingsPane
              v-else
              :terms="workspaceTerms"
              @inspect-term="inspectTerm"
              @hover-term="highlightTerm"
              @clear-hover="clearHighlight"
              @add-all-to-collection="$emit('add-all-to-collection', $event)"
            />
          </div>
        </div>
      </v-expand-transition>
    </template>
  </section>
</template>

<script setup>
import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import AnnotatedDocumentPane from './AnnotatedDocumentPane.vue';
import AnnotationInspectorPanel from './AnnotationInspectorPanel.vue';
import CaseWorkspacePanel from './CaseWorkspacePanel.vue';
import PhenotypeFindingsPane from './PhenotypeFindingsPane.vue';
import PhentrieveService from '../services/PhentrieveService';
import { SIDEBAR_MODE_CASE, SIDEBAR_MODE_INSPECTOR } from '../constants/fullTextWorkspace';
import { useFileDownload } from '../composables/useFileDownload';
import { useFullTextWorkspaceStore } from '../stores/fullTextWorkspace';

defineEmits(['add-all-to-collection']);

const props = defineProps({
  responseData: {
    type: Object,
    required: true,
  },
  turnId: {
    type: String,
    required: true,
  },
  submittedText: {
    type: String,
    default: '',
  },
  showCaseWorkspace: {
    type: Boolean,
    default: true,
  },
});

const { t } = useI18n();
const workspaceStore = useFullTextWorkspaceStore();
const { downloadJson, downloadText } = useFileDownload();
const mobilePane = ref('document');

if (!workspaceStore.hasTurn(props.turnId)) {
  workspaceStore.initializeTurn(props.turnId);
}

const turnState = computed(() => workspaceStore.getTurnState(props.turnId));
const expanded = computed(() => turnState.value?.expanded ?? false);
const sidebarMode = computed(() => turnState.value?.sidebarMode ?? SIDEBAR_MODE_CASE);
const selectedPhenotypeId = computed(() => turnState.value?.selectedPhenotypeId ?? null);
const cases = computed(() => turnState.value?.cases ?? []);
const activeCaseId = computed(() => turnState.value?.activeCaseId ?? null);
const workspaceTerms = computed(() => props.responseData?.aggregated_hpo_terms ?? []);
const usesSubmittedTextFallback = computed(
  () =>
    (props.responseData?.processed_chunks?.length ?? 0) === 0 &&
    typeof props.submittedText === 'string' &&
    props.submittedText.trim().length > 0
);
const fallbackAnnotations = computed(() =>
  usesSubmittedTextFallback.value
    ? buildSubmittedTextAnnotations(props.submittedText, workspaceTerms.value)
    : []
);

const workspaceChunks = computed(() => {
  if (usesSubmittedTextFallback.value) {
    return [
      {
        chunk_id: '__submitted_note__',
        text: props.submittedText,
        evidence_mode: fallbackAnnotations.value.length > 0 ? 'span' : 'chunk',
        annotations: fallbackAnnotations.value,
      },
    ];
  }

  const annotationsByChunk = new Map();

  workspaceTerms.value.forEach((term) => {
    const textAttributions = Array.isArray(term.text_attributions) ? term.text_attributions : [];
    textAttributions.forEach((attribution, index) => {
      if (typeof attribution?.chunk_id !== 'number') {
        return;
      }

      if (!annotationsByChunk.has(attribution.chunk_id)) {
        annotationsByChunk.set(attribution.chunk_id, []);
      }

      annotationsByChunk.get(attribution.chunk_id).push({
        id: `${term.hpo_id}-${attribution.chunk_id}-${index}`,
        start_char: attribution.start_char,
        end_char: attribution.end_char,
        matched_text_in_chunk: attribution.matched_text_in_chunk,
      });
    });
  });

  return (props.responseData?.processed_chunks ?? []).map((chunk) => ({
    ...chunk,
    evidence_mode: (annotationsByChunk.get(chunk.chunk_id) || []).length > 0 ? 'span' : 'chunk',
    annotations: annotationsByChunk.get(chunk.chunk_id) || [],
  }));
});

const selectedAnnotationIds = computed(() => {
  if (!selectedPhenotypeId.value) {
    return [];
  }

  if (usesSubmittedTextFallback.value) {
    return fallbackAnnotations.value
      .filter((annotation) => annotation.termId === selectedPhenotypeId.value)
      .map((annotation) => annotation.id);
  }

  const matchingTerm = workspaceTerms.value.find(
    (term) => term.hpo_id === selectedPhenotypeId.value
  );
  if (!matchingTerm || !Array.isArray(matchingTerm.text_attributions)) {
    return [];
  }

  return matchingTerm.text_attributions.map(
    (attribution, index) => `${matchingTerm.hpo_id}-${attribution.chunk_id}-${index}`
  );
});

const selectedTerm = computed(
  () => workspaceTerms.value.find((term) => term.hpo_id === selectedPhenotypeId.value) || null
);
const showSidebarPane = computed(
  () => sidebarMode.value === SIDEBAR_MODE_INSPECTOR || props.showCaseWorkspace
);
const showDocumentPane = computed(() => {
  if ((props.responseData?.processed_chunks?.length ?? 0) > 0) {
    return true;
  }
  return fallbackAnnotations.value.length > 0;
});

const bannerText = computed(() => {
  const meta = props.responseData?.meta || {};
  if (
    meta.extraction_backend === 'llm' &&
    meta.quota_remaining != null &&
    meta.quota_limit != null
  ) {
    return `${meta.quota_remaining} / ${meta.quota_limit} LLM analyses remaining today`;
  }

  if (meta.fallback_reason === 'llm_quota_exhausted') {
    return 'Standard analysis shown because richer LLM analysis is unavailable today';
  }

  return '';
});

const fallbackBanner = computed(() => {
  if (props.responseData?.meta?.fallback_reason !== 'llm_quota_exhausted') {
    return '';
  }

  const resetAt =
    props.responseData?.meta?.llm_quota_reset_at || props.responseData?.meta?.quota_reset_at;
  const resetText = resetAt ? new Date(resetAt).toLocaleString() : t('common.unknown');
  return `Richer LLM analysis is unavailable for today. Standard full-text analysis is shown instead. LLM access resets ${resetText}.`;
});

function toggleExpanded() {
  workspaceStore.setExpanded(props.turnId, !expanded.value);
}

function inspectTerm(term) {
  workspaceStore.setSelectedPhenotypeId(props.turnId, term?.hpo_id || null);
  workspaceStore.setSidebarMode(props.turnId, SIDEBAR_MODE_INSPECTOR);
  mobilePane.value = 'inspector';
}

function highlightTerm(term) {
  workspaceStore.setSelectedPhenotypeId(props.turnId, term?.hpoId || term?.hpo_id || null);
}

function clearHighlight() {
  if (sidebarMode.value !== SIDEBAR_MODE_INSPECTOR) {
    workspaceStore.setSelectedPhenotypeId(props.turnId, null);
  }
}

function showCaseSidebar() {
  workspaceStore.setSidebarMode(props.turnId, SIDEBAR_MODE_CASE);
  mobilePane.value = props.showCaseWorkspace ? 'case' : 'findings';
}

function createCase() {
  workspaceStore.createCase(props.turnId, {
    inputText: props.responseData?.meta?.input_text || '',
  });
  workspaceStore.setSidebarMode(props.turnId, SIDEBAR_MODE_CASE);
}

function ensureActiveCase() {
  if (!workspaceStore.getActiveCase(props.turnId)) {
    createCase();
  }

  return workspaceStore.getActiveCase(props.turnId);
}

function selectCase(caseId) {
  workspaceStore.setActiveCaseId(props.turnId, caseId);
}

function normalizeExportPhenotype(term) {
  return {
    hpo_id: term.hpo_id,
    label: term.name || term.label,
    assertion_status: normalizeExportAssertionStatus(term.status || term.assertion_status),
    confidence: typeof term.confidence === 'number' ? term.confidence : null,
    source_chunk_ids: Array.isArray(term.source_chunk_ids) ? [...term.source_chunk_ids] : [],
    text_attributions: Array.isArray(term.text_attributions) ? [...term.text_attributions] : [],
  };
}

function normalizeExportAssertionStatus(status) {
  const normalized = typeof status === 'string' ? status.trim().toLowerCase() : '';
  return normalized === 'negated' || normalized === 'absent' ? 'negated' : 'affirmed';
}

function addAllPhenotypes() {
  ensureActiveCase();
  workspaceStore.addPhenotypesToActiveCase(
    props.turnId,
    workspaceTerms.value.map((term) => normalizeExportPhenotype(term))
  );
}

async function exportActiveCase() {
  const activeCase = ensureActiveCase();
  if (!activeCase || activeCase.phenotypes.length === 0) {
    return;
  }

  const bundle = await PhentrieveService.exportPhenopacket({
    case_id: activeCase.id,
    case_label: activeCase.label,
    input_text: props.responseData?.meta?.input_text || '',
    include_annotation_sidecar: true,
    phenotypes: activeCase.phenotypes,
  });

  downloadText(bundle.phenopacket_json, `${activeCase.id}.phenopacket.json`, 'application/json');
  if (bundle.annotation_sidecar) {
    downloadJson(bundle.annotation_sidecar, `${activeCase.id}.annotations.json`);
  }
}

function buildSubmittedTextAnnotations(submittedText, terms) {
  const noteText = typeof submittedText === 'string' ? submittedText : '';
  const normalizedNote = noteText.toLowerCase();

  return terms.flatMap((term) => {
    const attributions = Array.isArray(term.text_attributions) ? term.text_attributions : [];

    return attributions
      .map((attribution, index) => {
        const range = resolveSubmittedTextRange(normalizedNote, attribution);
        if (!range) {
          return null;
        }

        return {
          id: `${term.hpo_id}-submitted-note-${index}`,
          termId: term.hpo_id,
          start_char: range.start,
          end_char: range.end,
          matched_text_in_chunk:
            attribution.matched_text_in_chunk || noteText.slice(range.start, range.end),
        };
      })
      .filter(Boolean);
  });
}

function resolveSubmittedTextRange(normalizedNote, attribution) {
  const matchedText =
    typeof attribution?.matched_text_in_chunk === 'string' ? attribution.matched_text_in_chunk : '';
  if (!matchedText) {
    return null;
  }

  const start = normalizedNote.indexOf(matchedText.toLowerCase());
  if (start === -1) {
    return null;
  }

  return {
    start,
    end: start + matchedText.length,
  };
}
</script>

<style scoped>
.full-text-workspace {
  overflow: hidden;
  display: grid;
  gap: 12px;
}

.full-text-workspace--embedded {
  gap: 16px;
}

.full-text-workspace__header {
  padding-inline: 4px;
}

.full-text-workspace__title-block {
  min-width: 0;
}

.full-text-embedded__document {
  display: grid;
  gap: 8px;
}

.workspace-layout {
  gap: 16px;
}

.workspace-layout--desktop {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.95fr) minmax(260px, 320px);
  align-items: start;
}

.workspace-layout--desktop-no-sidebar {
  grid-template-columns: minmax(0, 1.3fr) minmax(320px, 0.95fr);
}

.workspace-layout--desktop-findings-only {
  grid-template-columns: minmax(0, 1.4fr);
}

.workspace-layout--desktop-findings-only > :deep(.findings-pane) {
  max-width: 860px;
}

.workspace-layout--desktop > * {
  min-width: 0;
}

.workspace-layout--mobile {
  display: none;
}

.workspace-mobile-nav {
  display: none;
}

@media (max-width: 960px) {
  .workspace-mobile-nav {
    display: flex;
    gap: 8px;
    padding: 0 16px 8px;
    overflow-x: auto;
  }

  .workspace-layout--desktop {
    display: none;
  }

  .workspace-layout--mobile {
    display: block;
  }
}
</style>
