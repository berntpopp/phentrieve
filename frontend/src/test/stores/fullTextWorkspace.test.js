import { createPinia, setActivePinia } from 'pinia';
import { describe, it, expect, beforeEach } from 'vitest';
import { useFullTextWorkspaceStore } from '../../stores/fullTextWorkspace';
import {
  CONFIDENCE_BANDS,
  SIDEBAR_MODE_CASE,
  SIDEBAR_MODE_INSPECTOR,
} from '../../constants/fullTextWorkspace';

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

  it('returns turn snapshots without exposing mutable internal state', () => {
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-a');

    const snapshot = store.getTurnState('turn-a');
    snapshot.sidebarMode = SIDEBAR_MODE_INSPECTOR;
    snapshot.cases.push({ id: 'case-1' });

    expect(store.getTurnState('turn-a').sidebarMode).toBe(SIDEBAR_MODE_CASE);
    expect(store.getTurnState('turn-a').cases).toEqual([]);
  });

  it('rejects unsupported sidebar modes', () => {
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-a');

    expect(() => store.setSidebarMode('turn-a', 'invalid-mode')).toThrow(
      'Unsupported sidebar mode: invalid-mode'
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

describe('fullTextWorkspace constants', () => {
  it('freezes confidence bands at the top level and nested band level', () => {
    expect(Object.isFrozen(CONFIDENCE_BANDS)).toBe(true);
    expect(Object.isFrozen(CONFIDENCE_BANDS.high)).toBe(true);
    expect(() => {
      CONFIDENCE_BANDS.high.min = 0.5;
    }).toThrow();
    expect(CONFIDENCE_BANDS.high.min).toBe(0.85);
  });
});
