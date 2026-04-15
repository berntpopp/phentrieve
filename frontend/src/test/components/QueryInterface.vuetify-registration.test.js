import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';
import vuetify from '../../plugins/vuetify';

vi.mock('../../services/PhentrieveService', () => {
  const MockService = {
    queryHpo: vi.fn().mockResolvedValue({ results: [] }),
    processText: vi.fn().mockResolvedValue({ processed_chunks: [], aggregated_hpo_terms: [] }),
    getConfigInfo: vi.fn().mockResolvedValue({
      available_embedding_models: [{ name: 'test-model', id: 'test-model', loaded: true }],
      default_embedding_model: 'test-model',
    }),
  };
  return { default: MockService };
});

vi.mock('../../services/logService', () => ({
  logService: {
    info: vi.fn(),
    debug: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: { en },
  silentTranslationWarn: true,
  silentFallbackWarn: true,
  missing: () => '',
});

async function mountQueryInterfaceWithAppVuetify() {
  const pinia = createPinia();
  setActivePinia(pinia);
  const component = (await import('../../components/QueryInterface.vue')).default;
  return mount(component, {
    global: {
      plugins: [pinia, vuetify, i18n],
      stubs: {
        ResultsDisplay: true,
        ConversationSkeleton: true,
        'v-navigation-drawer': true,
      },
      mocks: {
        $route: { query: {} },
        $router: { replace: vi.fn().mockResolvedValue(undefined) },
      },
    },
  });
}

describe('QueryInterface Vuetify registration', () => {
  let warnSpy;

  beforeEach(() => {
    setActivePinia(createPinia());
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  it('mounts without unresolved Vuetify component warnings', async () => {
    const wrapper = await mountQueryInterfaceWithAppVuetify();

    expect(wrapper.exists()).toBe(true);
    expect(
      warnSpy.mock.calls.some(([message]) =>
        String(message).includes('Failed to resolve component: v-snackbar')
      )
    ).toBe(false);
  });
});
