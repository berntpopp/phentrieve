import { defineStore } from 'pinia';
import { ref } from 'vue';
import { SIDEBAR_MODE_CASE } from '../constants/fullTextWorkspace';

function createEmptyTurnState() {
  return {
    expanded: false,
    sidebarMode: SIDEBAR_MODE_CASE,
    selectedPhenotypeId: null,
    selectedSpanId: null,
    quotaBanner: null,
    activeCaseId: null,
    cases: [],
    undoStack: [],
    redoStack: [],
  };
}

export const useFullTextWorkspaceStore = defineStore('fullTextWorkspace', () => {
  const turns = ref({});

  function initializeTurn(turnId) {
    if (!turns.value[turnId]) {
      turns.value[turnId] = createEmptyTurnState();
    }
  }

  function setSidebarMode(turnId, mode) {
    initializeTurn(turnId);
    turns.value[turnId].sidebarMode = mode;
  }

  function setQuotaBanner(turnId, banner) {
    initializeTurn(turnId);
    turns.value[turnId].quotaBanner = banner;
  }

  return {
    turns,
    initializeTurn,
    setSidebarMode,
    setQuotaBanner,
  };
});
