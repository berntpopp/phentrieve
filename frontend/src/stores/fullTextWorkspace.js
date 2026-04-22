import { defineStore } from 'pinia';
import { readonly, ref } from 'vue';
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

function cloneWorkspaceValue(value) {
  if (Array.isArray(value)) {
    return value.map((item) => cloneWorkspaceValue(item));
  }

  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, nestedValue]) => [key, cloneWorkspaceValue(nestedValue)])
    );
  }

  return value;
}

function cloneTurnState(turnState) {
  return {
    expanded: turnState.expanded,
    sidebarMode: turnState.sidebarMode,
    selectedPhenotypeId: turnState.selectedPhenotypeId,
    selectedSpanId: turnState.selectedSpanId,
    quotaBanner: cloneWorkspaceValue(turnState.quotaBanner),
    activeCaseId: turnState.activeCaseId,
    cases: cloneWorkspaceValue(turnState.cases),
    undoStack: cloneWorkspaceValue(turnState.undoStack),
    redoStack: cloneWorkspaceValue(turnState.redoStack),
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

    return cloneTurnState(turns.value[turnId]);
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

  function hasTurn(turnId) {
    assertTurnId(turnId);
    return Object.hasOwn(turns.value, turnId);
  }

  function getTurnState(turnId) {
    assertTurnId(turnId);

    return turns.value[turnId] ? cloneTurnState(turns.value[turnId]) : null;
  }

  function setSidebarMode(turnId, mode) {
    assertSidebarMode(mode);
    requireTurn(turnId).sidebarMode = mode;
  }

  function setQuotaBanner(turnId, banner) {
    requireTurn(turnId).quotaBanner = cloneWorkspaceValue(banner);
  }

  function setCases(turnId, cases) {
    requireTurn(turnId).cases = cloneWorkspaceValue(cases);
  }

  function setUndoStack(turnId, undoStack) {
    requireTurn(turnId).undoStack = cloneWorkspaceValue(undoStack);
  }

  function setRedoStack(turnId, redoStack) {
    requireTurn(turnId).redoStack = cloneWorkspaceValue(redoStack);
  }

  function resetTurn(turnId) {
    requireTurn(turnId);
    turns.value[turnId] = createEmptyTurnState();

    return cloneTurnState(turns.value[turnId]);
  }

  function removeTurn(turnId) {
    requireTurn(turnId);
    delete turns.value[turnId];
  }

  return {
    turns: readonly(turns),
    initializeTurn,
    hasTurn,
    getTurnState,
    setSidebarMode,
    setQuotaBanner,
    setCases,
    setUndoStack,
    setRedoStack,
    resetTurn,
    removeTurn,
  };
});
