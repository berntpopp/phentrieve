import { defineStore } from 'pinia';
import { ref } from 'vue';
import {
  SIDEBAR_MODE_CASE,
  SIDEBAR_MODE_INSPECTOR,
} from '../constants/fullTextWorkspace';

const VALID_SIDEBAR_MODES = new Set([SIDEBAR_MODE_CASE, SIDEBAR_MODE_INSPECTOR]);

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

function assertTurnId(turnId) {
  if (typeof turnId !== 'string' || turnId.length === 0) {
    throw new Error('Workspace turn id must be a non-empty string');
  }
}

export const useFullTextWorkspaceStore = defineStore('fullTextWorkspace', () => {
  const turns = ref({});

  function initializeTurn(turnId) {
    assertTurnId(turnId);
    if (!turns.value[turnId]) {
      turns.value[turnId] = createEmptyTurnState();
    }

    return turns.value[turnId];
  }

  function requireTurn(turnId) {
    assertTurnId(turnId);

    if (!turns.value[turnId]) {
      throw new Error(`Unknown workspace turn: ${turnId}`);
    }

    return turns.value[turnId];
  }

  function assertSidebarMode(mode) {
    if (!VALID_SIDEBAR_MODES.has(mode)) {
      throw new Error(`Unsupported sidebar mode: ${mode}`);
    }
  }

  function setSidebarMode(turnId, mode) {
    assertSidebarMode(mode);
    requireTurn(turnId).sidebarMode = mode;
  }

  function setQuotaBanner(turnId, banner) {
    requireTurn(turnId).quotaBanner = banner;
  }

  return {
    turns,
    initializeTurn,
    setSidebarMode,
    setQuotaBanner,
  };
});
