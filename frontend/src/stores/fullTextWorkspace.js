import { defineStore } from 'pinia';
import { readonly, ref } from 'vue';
import { SIDEBAR_MODE_CASE, SIDEBAR_MODE_INSPECTOR } from '../constants/fullTextWorkspace';

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

function createWorkspaceId(prefix) {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return `${prefix}-${crypto.randomUUID()}`;
  }

  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function cloneWorkspaceValue(value) {
  if (Array.isArray(value)) {
    return value.map((item) => cloneWorkspaceValue(item));
  }

  if (value && typeof value === 'object') {
    const cloned = {};

    Object.keys(value).forEach((key) => {
      cloned[key] = cloneWorkspaceValue(value[key]);
    });

    return cloned;
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

function isPlainObject(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false;
  }

  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function assertJsonLikeValue(value, message) {
  if (value === null) {
    return;
  }

  if (Array.isArray(value)) {
    value.forEach((item) => assertJsonLikeValue(item, message));
    return;
  }

  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new Error(message);
    }
    return;
  }

  if (typeof value === 'string' || typeof value === 'boolean') {
    return;
  }

  if (isPlainObject(value)) {
    Object.values(value).forEach((item) => assertJsonLikeValue(item, message));
    return;
  }

  throw new Error(message);
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

    assertJsonLikeValue(item, `Workspace cases[${index}] must contain only JSON-like data`);

    if (seenIds.has(item.id)) {
      throw new Error(`Workspace cases must not contain duplicate ids: ${item.id}`);
    }

    seenIds.add(item.id);
  });
}

function normalizePhenotypeRecord(phenotype, errorPrefix = 'Workspace phenotype') {
  if (!phenotype || typeof phenotype !== 'object' || Array.isArray(phenotype)) {
    throw new Error(`${errorPrefix} must be an object`);
  }

  if (typeof phenotype.hpo_id !== 'string' || phenotype.hpo_id.length === 0) {
    throw new Error(`${errorPrefix}.hpo_id must be a non-empty string`);
  }

  if (typeof phenotype.label !== 'string' || phenotype.label.length === 0) {
    throw new Error(`${errorPrefix}.label must be a non-empty string`);
  }

  const normalized = cloneWorkspaceValue(phenotype);
  normalized.assertion_status =
    typeof normalized.assertion_status === 'string' && normalized.assertion_status.length > 0
      ? normalized.assertion_status
      : 'affirmed';

  assertJsonLikeValue(normalized, `${errorPrefix} must contain only JSON-like data`);
  return normalized;
}

function assertStackItems(items, label) {
  if (!Array.isArray(items)) {
    throw new Error(`Workspace ${label} must be an array`);
  }

  assertObjectEntries(items, `Workspace ${label} items must be objects`);

  items.forEach((item) =>
    assertJsonLikeValue(item, `Workspace ${label} items must contain only JSON-like data`)
  );
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

  assertJsonLikeValue(banner, 'Workspace quota banner must contain only JSON-like data');
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
    return Object.prototype.hasOwnProperty.call(turns.value, turnId);
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

  function createCase(turnId, casePayload = {}) {
    const turn = requireTurn(turnId);
    const nextCase = {
      id:
        typeof casePayload.id === 'string' && casePayload.id.length > 0
          ? casePayload.id
          : createWorkspaceId('case'),
      label:
        typeof casePayload.label === 'string' && casePayload.label.length > 0
          ? casePayload.label
          : `Case ${turn.cases.length + 1}`,
      phenotypes: Array.isArray(casePayload.phenotypes)
        ? casePayload.phenotypes.map((phenotype, index) =>
            normalizePhenotypeRecord(phenotype, `Workspace case phenotype ${index}`)
          )
        : [],
      inputText: typeof casePayload.inputText === 'string' ? casePayload.inputText : '',
    };

    turn.cases.push(nextCase);
    turn.activeCaseId = nextCase.id;
    return cloneWorkspaceValue(nextCase);
  }

  function getActiveCase(turnId) {
    const turn = requireTurn(turnId);
    if (!turn.activeCaseId) {
      return null;
    }

    return cloneWorkspaceValue(turn.cases.find((item) => item.id === turn.activeCaseId) || null);
  }

  function addPhenotypeToActiveCase(turnId, phenotype) {
    const turn = requireTurn(turnId);
    if (!turn.activeCaseId) {
      throw new Error(`Cannot add phenotype without an active case for turn ${turnId}`);
    }

    const activeCase = turn.cases.find((item) => item.id === turn.activeCaseId);
    if (!activeCase) {
      throw new Error(`Active case id must reference an existing case: ${turn.activeCaseId}`);
    }

    const normalizedPhenotype = normalizePhenotypeRecord(phenotype);
    const duplicate = activeCase.phenotypes.some(
      (item) =>
        item.hpo_id === normalizedPhenotype.hpo_id &&
        item.assertion_status === normalizedPhenotype.assertion_status
    );

    if (duplicate) {
      return false;
    }

    activeCase.phenotypes.push(normalizedPhenotype);
    return true;
  }

  function addPhenotypesToActiveCase(turnId, phenotypes) {
    if (!Array.isArray(phenotypes)) {
      throw new Error('Workspace phenotypes must be an array');
    }

    let addedCount = 0;
    phenotypes.forEach((phenotype) => {
      if (addPhenotypeToActiveCase(turnId, phenotype)) {
        addedCount += 1;
      }
    });

    return addedCount;
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
    createCase,
    getActiveCase,
    addPhenotypeToActiveCase,
    addPhenotypesToActiveCase,
    resetTurn,
    removeTurn,
  };
});
