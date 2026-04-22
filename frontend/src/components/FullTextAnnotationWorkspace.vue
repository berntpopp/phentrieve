<template>
  <v-card class="full-text-workspace" rounded="lg" elevation="1">
    <v-card-title class="d-flex justify-space-between align-center ga-3">
      <div>
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
    </v-card-title>

    <v-alert
      v-if="fallbackBanner"
      type="info"
      density="compact"
      variant="tonal"
      class="mx-4 mb-2"
    >
      {{ fallbackBanner }}
    </v-alert>

    <v-expand-transition>
      <div v-show="expanded" class="workspace-layout">
        <div class="workspace-main">
          <AnnotatedDocumentPane
            :chunks="workspaceChunks"
            :selected-annotation-ids="selectedAnnotationIds"
          />
          <PhenotypeFindingsPane
            :terms="workspaceTerms"
            @inspect-term="inspectTerm"
            @hover-term="highlightTerm"
            @clear-hover="clearHighlight"
          />
        </div>

        <AnnotationInspectorPanel
          v-if="sidebarMode === SIDEBAR_MODE_INSPECTOR"
          :selected-term="selectedTerm"
          @back="showCaseSidebar"
        />
        <CaseWorkspacePanel
          v-else
          :cases="cases"
          :active-case-id="activeCaseId || ''"
          @create-case="createCase"
          @select-case="selectCase"
          @add-all="addAllPhenotypes"
          @export-case="exportActiveCase"
        />
      </div>
    </v-expand-transition>
  </v-card>
</template>

<script setup>
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import AnnotatedDocumentPane from './AnnotatedDocumentPane.vue';
import AnnotationInspectorPanel from './AnnotationInspectorPanel.vue';
import CaseWorkspacePanel from './CaseWorkspacePanel.vue';
import PhenotypeFindingsPane from './PhenotypeFindingsPane.vue';
import PhentrieveService from '../services/PhentrieveService';
import {
  SIDEBAR_MODE_CASE,
  SIDEBAR_MODE_INSPECTOR,
} from '../constants/fullTextWorkspace';
import { useFileDownload } from '../composables/useFileDownload';
import { useFullTextWorkspaceStore } from '../stores/fullTextWorkspace';

const props = defineProps({
  responseData: {
    type: Object,
    required: true,
  },
  turnId: {
    type: String,
    required: true,
  },
});

const { t } = useI18n();
const workspaceStore = useFullTextWorkspaceStore();
const { downloadJson, downloadText } = useFileDownload();

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

const workspaceChunks = computed(() => {
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
    evidence_mode:
      (annotationsByChunk.get(chunk.chunk_id) || []).length > 0 ? 'span' : 'chunk',
    annotations: annotationsByChunk.get(chunk.chunk_id) || [],
  }));
});

const selectedAnnotationIds = computed(() => {
  if (!selectedPhenotypeId.value) {
    return [];
  }

  const matchingTerm = workspaceTerms.value.find((term) => term.hpo_id === selectedPhenotypeId.value);
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

const bannerText = computed(() => {
  const meta = props.responseData?.meta || {};
  if (meta.extraction_backend === 'llm' && meta.quota_remaining != null && meta.quota_limit != null) {
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

  const resetAt = props.responseData?.meta?.llm_quota_reset_at || props.responseData?.meta?.quota_reset_at;
  const resetText = resetAt ? new Date(resetAt).toLocaleString() : t('common.unknown');
  return `Richer LLM analysis is unavailable for today. Standard full-text analysis is shown instead. LLM access resets ${resetText}.`;
});

function toggleExpanded() {
  workspaceStore.setExpanded(props.turnId, !expanded.value);
}

function inspectTerm(term) {
  workspaceStore.setSelectedPhenotypeId(props.turnId, term?.hpo_id || null);
  workspaceStore.setSidebarMode(props.turnId, SIDEBAR_MODE_INSPECTOR);
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
    assertion_status: term.status || term.assertion_status || 'affirmed',
    confidence: typeof term.confidence === 'number' ? term.confidence : null,
    source_chunk_ids: Array.isArray(term.source_chunk_ids) ? [...term.source_chunk_ids] : [],
    text_attributions: Array.isArray(term.text_attributions) ? [...term.text_attributions] : [],
  };
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
</script>

<style scoped>
.full-text-workspace {
  overflow: hidden;
}

.workspace-layout {
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(280px, 360px);
  gap: 16px;
  padding: 16px;
}

.workspace-main {
  display: grid;
  gap: 16px;
  min-width: 0;
}

@media (max-width: 960px) {
  .workspace-layout {
    grid-template-columns: 1fr;
  }
}
</style>
