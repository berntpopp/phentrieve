import { createPinia, setActivePinia } from 'pinia';
import { describe, it, expect, beforeEach } from 'vitest';
import { useFullTextWorkspaceStore } from '../../stores/fullTextWorkspace';
import { SIDEBAR_MODE_CASE, SIDEBAR_MODE_INSPECTOR } from '../../constants/fullTextWorkspace';

describe('fullTextWorkspace store', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
  });

  it('initializes turn state with the default workspace shape', () => {
    const store = useFullTextWorkspaceStore();

    store.initializeTurn('turn-a');

    expect(store.getTurnState('turn-a')).toEqual({
      expanded: false,
      sidebarMode: SIDEBAR_MODE_CASE,
      selectedPhenotypeId: null,
      selectedSpanId: null,
      quotaBanner: null,
      activeCaseId: null,
      cases: [],
      undoStack: [],
      redoStack: [],
    });
  });

  it('isolates workspace state by conversation turn id', () => {
    const store = useFullTextWorkspaceStore();

    store.initializeTurn('turn-a');
    store.initializeTurn('turn-b');
    store.setSidebarMode('turn-a', SIDEBAR_MODE_INSPECTOR);

    expect(store.getTurnState('turn-a').sidebarMode).toBe(SIDEBAR_MODE_INSPECTOR);
    expect(store.getTurnState('turn-b').sidebarMode).toBe(SIDEBAR_MODE_CASE);
  });

  it('stores quota banners independently from conversation history', () => {
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-a');
    store.setQuotaBanner('turn-a', {
      fallbackReason: 'llm_quota_exhausted',
      quotaResetAt: '2026-04-23T00:00:00+00:00',
    });

    expect(store.getTurnState('turn-a').quotaBanner.fallbackReason).toBe('llm_quota_exhausted');
  });

  it('updates owned scalar workspace fields through explicit actions', () => {
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-a');
    store.setCases('turn-a', [{ id: 'case-1', label: 'Case 1' }]);

    store.setExpanded('turn-a', true);
    store.setSelectedPhenotypeId('turn-a', 'HP:0001250');
    store.setSelectedSpanId('turn-a', 'span-1');
    store.setActiveCaseId('turn-a', 'case-1');

    expect(store.getTurnState('turn-a')).toMatchObject({
      expanded: true,
      selectedPhenotypeId: 'HP:0001250',
      selectedSpanId: 'span-1',
      activeCaseId: 'case-1',
    });
  });

  it('returns turn snapshots without exposing mutable internal state', () => {
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-a');

    const snapshot = store.getTurnState('turn-a');
    snapshot.sidebarMode = SIDEBAR_MODE_INSPECTOR;
    snapshot.cases.push({ id: 'case-1' });

    expect(store.getTurnState('turn-a').sidebarMode).toBe(SIDEBAR_MODE_CASE);
    expect(store.getTurnState('turn-a').cases).toEqual([]);
  });

  it('defensively copies nested workspace structures on write and read', () => {
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-a');

    const cases = [{ id: 'case-1', details: { rank: 1 } }];
    const undoStack = [{ type: 'assign-case', payload: { caseId: 'case-1' } }];
    const redoStack = [{ type: 'restore-case', payload: { caseId: 'case-2' } }];
    const quotaBanner = {
      fallbackReason: 'llm_quota_exhausted',
      quotaResetAt: '2026-04-23T00:00:00+00:00',
      details: { remaining: 0 },
    };

    store.setCases('turn-a', cases);
    store.setUndoStack('turn-a', undoStack);
    store.setRedoStack('turn-a', redoStack);
    store.setQuotaBanner('turn-a', quotaBanner);

    cases[0].details.rank = 99;
    undoStack[0].payload.caseId = 'mutated-undo';
    redoStack[0].payload.caseId = 'mutated-redo';
    quotaBanner.details.remaining = 99;

    const snapshot = store.getTurnState('turn-a');
    expect(snapshot.cases[0].details.rank).toBe(1);
    expect(snapshot.undoStack[0].payload.caseId).toBe('case-1');
    expect(snapshot.redoStack[0].payload.caseId).toBe('case-2');
    expect(snapshot.quotaBanner.details.remaining).toBe(0);

    snapshot.cases[0].details.rank = -1;
    snapshot.undoStack[0].payload.caseId = 'snapshot-undo';
    snapshot.redoStack[0].payload.caseId = 'snapshot-redo';
    snapshot.quotaBanner.details.remaining = -1;

    const freshSnapshot = store.getTurnState('turn-a');
    expect(freshSnapshot.cases[0].details.rank).toBe(1);
    expect(freshSnapshot.undoStack[0].payload.caseId).toBe('case-1');
    expect(freshSnapshot.redoStack[0].payload.caseId).toBe('case-2');
    expect(freshSnapshot.quotaBanner.details.remaining).toBe(0);
  });

  it('rejects unsupported sidebar modes', () => {
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-a');

    expect(() => store.setSidebarMode('turn-a', 'invalid-mode')).toThrow(
      'Unsupported sidebar mode: invalid-mode'
    );
  });

  it('rejects invalid scalar workspace updates', () => {
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-a');

    expect(() => store.setExpanded('turn-a', 'yes')).toThrow(
      'Workspace expanded state must be a boolean'
    );
    expect(() => store.setSelectedPhenotypeId('turn-a', '')).toThrow(
      'Selected phenotype id must be null or a non-empty string'
    );
    expect(() => store.setSelectedSpanId('turn-a', 42)).toThrow(
      'Selected span id must be null or a non-empty string'
    );
  });

  it('rejects invalid collection payloads and active case invariants', () => {
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-a');

    expect(() => store.setCases('turn-a', 'invalid')).toThrow('Workspace cases must be an array');
    expect(() => store.setCases('turn-a', [{}])).toThrow(
      'Workspace cases[0].id must be a non-empty string'
    );
    expect(() =>
      store.setCases('turn-a', [
        { id: 'case-1', label: 'Case 1' },
        { id: 'case-1', label: 'Case 1 duplicate' },
      ])
    ).toThrow('Workspace cases must not contain duplicate ids: case-1');
    expect(() => store.setUndoStack('turn-a', {})).toThrow('Workspace undo stack must be an array');
    expect(() => store.setUndoStack('turn-a', [null])).toThrow(
      'Workspace undo stack items must be objects'
    );
    expect(() => store.setRedoStack('turn-a', {})).toThrow('Workspace redo stack must be an array');
    expect(() => store.setRedoStack('turn-a', [null])).toThrow(
      'Workspace redo stack items must be objects'
    );
    expect(() => store.setActiveCaseId('turn-a', 'missing-case')).toThrow(
      'Active case id must reference an existing case: missing-case'
    );
  });

  it('rejects invalid quota banner payloads', () => {
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-a');

    expect(() => store.setQuotaBanner('turn-a', 'invalid')).toThrow(
      'Workspace quota banner must be null or an object'
    );
    expect(() => store.setQuotaBanner('turn-a', { fallbackReason: '' })).toThrow(
      'Workspace quota banner fallbackReason must be a non-empty string'
    );
    expect(() => store.setQuotaBanner('turn-a', { quotaResetAt: '' })).toThrow(
      'Workspace quota banner quotaResetAt must be a non-empty string'
    );
    expect(() => store.setQuotaBanner('turn-a', { fallbackReason: 'x', retry: () => {} })).toThrow(
      'Workspace quota banner must contain only JSON-like data'
    );
  });

  it('rejects non-json-like values that the clone path cannot round-trip', () => {
    class CustomPayload {
      constructor(value) {
        this.value = value;
      }
    }

    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-a');

    expect(() =>
      store.setCases('turn-a', [{ id: 'case-1', observedAt: new Date('2026-04-23T00:00:00Z') }])
    ).toThrow('Workspace cases[0] must contain only JSON-like data');
    expect(() => store.setUndoStack('turn-a', [{ payload: new Map([['a', 1]]) }])).toThrow(
      'Workspace undo stack items must contain only JSON-like data'
    );
    expect(() => store.setRedoStack('turn-a', [{ payload: new Set(['a']) }])).toThrow(
      'Workspace redo stack items must contain only JSON-like data'
    );
    expect(() =>
      store.setQuotaBanner('turn-a', { fallbackReason: 'x', details: new CustomPayload(1) })
    ).toThrow('Workspace quota banner must contain only JSON-like data');
    expect(() => store.setQuotaBanner('turn-a', { fallbackReason: 'x', score: NaN })).toThrow(
      'Workspace quota banner must contain only JSON-like data'
    );
    expect(() => store.setCases('turn-a', [{ id: 'case-1', score: Infinity }])).toThrow(
      'Workspace cases[0] must contain only JSON-like data'
    );
  });

  it('rejects circular structures in workspace payloads with a controlled error', () => {
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-a');

    const circularDetails = {};
    circularDetails.self = circularDetails;

    expect(() =>
      store.setQuotaBanner('turn-a', { fallbackReason: 'x', details: circularDetails })
    ).toThrow('Workspace values must not contain circular references');

    const circularPhenotype = {
      hpo_id: 'HP:0001250',
      label: 'Seizure',
    };
    circularPhenotype.details = circularPhenotype;

    expect(() => store.addPhenotypeToActiveCase('turn-a', circularPhenotype)).toThrow(
      'Cannot add phenotype without an active case for turn turn-a'
    );

    store.createCase('turn-a', { id: 'case-1', label: 'Case 1' });

    expect(() => store.addPhenotypeToActiveCase('turn-a', circularPhenotype)).toThrow(
      'Workspace values must not contain circular references'
    );
  });

  it('requires turn initialization before mutating workspace state', () => {
    const store = useFullTextWorkspaceStore();

    expect(() => store.setSidebarMode('missing-turn', SIDEBAR_MODE_CASE)).toThrow(
      'Unknown workspace turn: missing-turn'
    );
    expect(() =>
      store.setQuotaBanner('missing-turn', {
        fallbackReason: 'llm_quota_exhausted',
        quotaResetAt: '2026-04-23T00:00:00+00:00',
      })
    ).toThrow('Unknown workspace turn: missing-turn');
  });

  it('supports resetting and removing per-turn workspace state', () => {
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-a');
    store.setSidebarMode('turn-a', SIDEBAR_MODE_INSPECTOR);
    store.setQuotaBanner('turn-a', {
      fallbackReason: 'llm_quota_exhausted',
      quotaResetAt: '2026-04-23T00:00:00+00:00',
    });

    store.resetTurn('turn-a');

    expect(store.getTurnState('turn-a')).toEqual({
      expanded: false,
      sidebarMode: SIDEBAR_MODE_CASE,
      selectedPhenotypeId: null,
      selectedSpanId: null,
      quotaBanner: null,
      activeCaseId: null,
      cases: [],
      undoStack: [],
      redoStack: [],
    });

    store.removeTurn('turn-a');

    expect(store.hasTurn('turn-a')).toBe(false);
    expect(store.getTurnState('turn-a')).toBe(null);
  });
});
