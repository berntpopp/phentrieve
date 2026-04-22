import { createPinia, setActivePinia } from 'pinia';
import { describe, it, expect, beforeEach } from 'vitest';
import { useFullTextWorkspaceStore } from '../../stores/fullTextWorkspace';
import {
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

    expect(store.turns['turn-a']).toEqual({
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

    expect(store.turns['turn-a'].sidebarMode).toBe(SIDEBAR_MODE_INSPECTOR);
    expect(store.turns['turn-b'].sidebarMode).toBe(SIDEBAR_MODE_CASE);
  });

  it('stores quota banners independently from conversation history', () => {
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-a');
    store.setQuotaBanner('turn-a', {
      fallbackReason: 'llm_quota_exhausted',
      quotaResetAt: '2026-04-23T00:00:00+00:00',
    });

    expect(store.turns['turn-a'].quotaBanner.fallbackReason).toBe('llm_quota_exhausted');
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
});
