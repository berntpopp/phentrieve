import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';
import { createApp } from 'vue';
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate';
import { useConversationStore } from '../../stores/conversation';

const STORAGE_KEY = 'phentrieve-conversation';

function installPersistedPinia() {
  const pinia = createPinia();
  pinia.use(piniaPluginPersistedstate);
  createApp({}).use(pinia);
  setActivePinia(pinia);
}

describe('Conversation Store persistence privacy', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it('stores redacted durable query history when a redacted query is provided', () => {
    const store = useConversationStore();

    store.addQuery({
      query: 'Patient Jane Doe has seizures',
      redactedQuery: 'Patient [REDACTED_NAME] has seizures',
      terms: [],
    });

    expect(store.queryHistory[0]).toMatchObject({
      query: '[redacted]',
      redactedQuery: 'Patient [REDACTED_NAME] has seizures',
      rawQuerySessionOnly: 'Patient Jane Doe has seizures',
    });
  });

  it('uses a safe fallback when no redacted query is provided', () => {
    const store = useConversationStore();

    store.addQuery({
      query: 'Patient Jane Doe has seizures',
      terms: [],
    });

    expect(store.queryHistory[0]).toMatchObject({
      query: '[redacted]',
      redactedQuery: '[redacted]',
      rawQuerySessionOnly: 'Patient Jane Doe has seizures',
    });
  });

  it('does not persist session-only raw query text', () => {
    installPersistedPinia();
    const store = useConversationStore();

    store.addQuery({
      query: 'Patient Jane Doe has seizures',
      redactedQuery: 'Patient [REDACTED_NAME] has seizures',
      terms: [],
    });
    store.$persist();

    const persisted = localStorage.getItem(STORAGE_KEY);

    expect(persisted).toContain('Patient [REDACTED_NAME] has seizures');
    expect(persisted).not.toContain('Patient Jane Doe has seizures');
    expect(JSON.parse(persisted).queryHistory[0]).not.toHaveProperty('rawQuerySessionOnly');
  });
});
