import { createPinia, setActivePinia } from 'pinia';
import { describe, it, expect, beforeEach } from 'vitest';
import { useFullTextWorkspaceStore } from '../../stores/fullTextWorkspace';

describe('fullTextWorkspace store', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
  });

  it('isolates workspace state by conversation turn id', () => {
    const store = useFullTextWorkspaceStore();

    store.initializeTurn('turn-a');
    store.initializeTurn('turn-b');
    store.setSidebarMode('turn-a', 'inspector');

    expect(store.turns['turn-a'].sidebarMode).toBe('inspector');
    expect(store.turns['turn-b'].sidebarMode).toBe('case');
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
});
