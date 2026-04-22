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

function assertNullableNonEmptyString(value, message) {
  if (value !== null && (typeof value !== 'string' || value.length === 0)) {
    throw new Error(message);
  }
}

function assertBoolean(value, message) {
  if (typeof value !== 'boolean') {
    throw new Error(message);
  }
}

function assertObjectEntries(items, message) {
  items.forEach((item) => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) {
      throw new Error(message);
    }
  });
}

function assertCases(cases) {
  if (!Array.isArray(cases)) {
    throw new Error('Workspace cases must be an array');
  }

  const seenIds = new Set();
  cases.forEach((item, index) => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) {
      throw new Error(`Workspace cases[${index}] must be an object`);
    }

    if (typeof item.id !== 'string' || item.id.length === 0) {
      throw new Error(`Workspace cases[${index}].id must be a non-empty string`);
    }

    if (seenIds.has(item.id)) {
      throw new Error(`Workspace cases must not contain duplicate ids: ${item.id}`);
    }

    seenIds.add(item.id);
  });
}

function assertStackItems(items, label) {
  if (!Array.isArray(items)) {
    throw new Error(`Workspace ${label} must be an array`);
  }

  assertObjectEntries(items, `Workspace ${label} items must be objects`);
}

function assertJsonSerializableValue(value) {
  if (value === null) {
    return;
  }

  if (Array.isArray(value)) {
    value.forEach((item) => assertJsonSerializableValue(item));
    return;
  }

  const valueType = typeof value;
  if (valueType === 'string' || valueType === 'number' || valueType === 'boolean') {
    return;
  }

  if (valueType === 'object') {
    Object.values(value).forEach((item) => assertJsonSerializableValue(item));
    return;
  }

  throw new Error('Workspace quota banner must be JSON-serializable');
}

function assertQuotaBanner(banner) {
  if (banner === null) {
    return;
  }

  if (!banner || typeof banner !== 'object' || Array.isArray(banner)) {
    throw new Error('Workspace quota banner must be null or an object');
  }

  if (
    'fallbackReason' in banner &&
    (typeof banner.fallbackReason !== 'string' || banner.fallbackReason.length === 0)
  ) {
    throw new Error('Workspace quota banner fallbackReason must be a non-empty string');
  }

  if (
    'quotaResetAt' in banner &&
    (typeof banner.quotaResetAt !== 'string' || banner.quotaResetAt.length === 0)
  ) {
    throw new Error('Workspace quota banner quotaResetAt must be a non-empty string');
  }

  assertJsonSerializableValue(banner);
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

  function setExpanded(turnId, expanded) {
    assertBoolean(expanded, 'Workspace expanded state must be a boolean');
    requireTurn(turnId).expanded = expanded;
  }

  function setSelectedPhenotypeId(turnId, phenotypeId) {
    assertNullableNonEmptyString(
      phenotypeId,
      'Selected phenotype id must be null or a non-empty string'
    );
    requireTurn(turnId).selectedPhenotypeId = phenotypeId;
  }

  function setSelectedSpanId(turnId, spanId) {
    assertNullableNonEmptyString(spanId, 'Selected span id must be null or a non-empty string');
    requireTurn(turnId).selectedSpanId = spanId;
  }

  function setQuotaBanner(turnId, banner) {
    assertQuotaBanner(banner);
    requireTurn(turnId).quotaBanner = cloneWorkspaceValue(banner);
  }

  function setCases(turnId, cases) {
    assertCases(cases);
    const turn = requireTurn(turnId);
    turn.cases = cloneWorkspaceValue(cases);

    if (turn.activeCaseId !== null && !turn.cases.some((item) => item.id === turn.activeCaseId)) {
      turn.activeCaseId = null;
    }
  }

  function setActiveCaseId(turnId, caseId) {
    assertNullableNonEmptyString(caseId, 'Active case id must be null or a non-empty string');

    const turn = requireTurn(turnId);
    if (caseId !== null && !turn.cases.some((item) => item.id === caseId)) {
      throw new Error(`Active case id must reference an existing case: ${caseId}`);
    }

    turn.activeCaseId = caseId;
  }

  function setUndoStack(turnId, undoStack) {
    assertStackItems(undoStack, 'undo stack');
    requireTurn(turnId).undoStack = cloneWorkspaceValue(undoStack);
  }

  function setRedoStack(turnId, redoStack) {
    assertStackItems(redoStack, 'redo stack');
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
    setExpanded,
    setSelectedPhenotypeId,
    setSelectedSpanId,
    setQuotaBanner,
    setCases,
    setActiveCaseId,
    setUndoStack,
    setRedoStack,
    resetTurn,
    removeTurn,
  };
});
