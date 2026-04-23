import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mount, flushPromises } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';
import PhentrieveService from '../../services/PhentrieveService';
import { useFullTextWorkspaceStore } from '../../stores/fullTextWorkspace';

// Mock the API service class — methods are queryHpo() and processText()
// (NOT query() — the real class in services/PhentrieveService.js uses queryHpo)
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

// Mock logService to avoid side effects
vi.mock('../../services/logService', () => ({
  logService: {
    info: vi.fn(),
    debug: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

const vuetify = createVuetify({ components, directives });
const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: { en },
  silentTranslationWarn: true,
  silentFallbackWarn: true,
  missing: () => '',
});

async function mountQueryInterface({
  routeQuery = {},
  routerReplace = vi.fn().mockResolvedValue(undefined),
} = {}) {
  // QueryInterface uses Options API with setup() hook, has no props.
  // It renders a search container with query input.
  const pinia = createPinia();
  setActivePinia(pinia);
  const component = (await import('../../components/QueryInterface.vue')).default;
  return mount(component, {
    global: {
      plugins: [pinia, vuetify, i18n],
      stubs: {
        ResultsDisplay: {
          props: ['error'],
          template: '<div class="results-display-stub">{{ error?.detail || "" }}</div>',
        },
        ConversationSkeleton: true,
        'v-navigation-drawer': true,
      },
      mocks: {
        $route: { query: routeQuery },
        $router: { replace: routerReplace },
      },
    },
  });
}

describe('QueryInterface (characterization)', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('mounts without errors', async () => {
    const wrapper = await mountQueryInterface();
    expect(wrapper.exists()).toBe(true);
  });

  it('renders the search container', async () => {
    const wrapper = await mountQueryInterface();
    expect(wrapper.find('.search-container').exists()).toBe(true);
  });

  it('loads available models through the controller-owned bootstrap flow', async () => {
    const wrapper = await mountQueryInterface();

    await flushPromises();

    expect(PhentrieveService.getConfigInfo).toHaveBeenCalledTimes(1);
    expect(wrapper.vm.availableModels).toEqual(
      expect.arrayContaining([expect.objectContaining({ value: 'test-model' })])
    );
    expect(wrapper.vm.selectedModel).toBe('test-model');
  });

  it('initializes with empty query text', async () => {
    const wrapper = await mountQueryInterface();
    expect(wrapper.vm.queryText).toBe('');
  });

  it('initializes with loading false', async () => {
    const wrapper = await mountQueryInterface();
    expect(wrapper.vm.isLoading).toBe(false);
  });

  it('has advanced options hidden by default', async () => {
    const wrapper = await mountQueryInterface();
    expect(wrapper.vm.showAdvancedOptions).toBe(false);
  });

  it('has default threshold values', async () => {
    const wrapper = await mountQueryInterface();
    expect(wrapper.vm.similarityThreshold).toBe(0.5);
    expect(wrapper.vm.numResults).toBe(10);
    expect(wrapper.vm.chunkRetrievalThreshold).toBe(0.7);
    expect(wrapper.vm.aggregatedTermConfidence).toBe(0.75);
  });

  it('renders the closed input for the forced processing mode', async () => {
    const wrapper = await mountQueryInterface();
    const queryModeLabel = 'Phenotype query';
    const documentModeLabel = 'Clinical note';

    await wrapper.setData({ forceEndpointMode: 'query' });

    expect(wrapper.findComponent({ name: 'VTextField' }).exists()).toBe(true);
    expect(wrapper.findComponent({ name: 'VTextarea' }).exists()).toBe(false);
    expect(wrapper.text()).toContain(queryModeLabel);
    expect(wrapper.text()).not.toContain(documentModeLabel);

    await wrapper.setData({ forceEndpointMode: 'textProcess' });

    expect(wrapper.findComponent({ name: 'VTextField' }).exists()).toBe(false);
    expect(wrapper.findComponent({ name: 'VTextarea' }).exists()).toBe(true);
    expect(wrapper.text()).toContain(documentModeLabel);
    expect(wrapper.text()).not.toContain(queryModeLabel);
  });

  it('renders visible query and full-text mode pills in the search shell', async () => {
    const wrapper = await mountQueryInterface();

    expect(wrapper.find('[data-testid="mode-pill-query"]').exists()).toBe(true);
    expect(wrapper.find('[data-testid="mode-pill-text-process"]').exists()).toBe(true);
  });

  it('switches to full-text mode from the visible search-shell pill', async () => {
    const wrapper = await mountQueryInterface();

    await wrapper.find('[data-testid="mode-pill-text-process"]').trigger('click');

    expect(wrapper.vm.forceEndpointMode).toBe('textProcess');
    expect(wrapper.findComponent({ name: 'VTextarea' }).exists()).toBe(true);
    expect(wrapper.findComponent({ name: 'VTextField' }).exists()).toBe(false);
  });

  it('uses a distinct shell class for full-text mode so textarea styling can diverge from query mode', async () => {
    const wrapper = await mountQueryInterface();

    expect(wrapper.find('.search-bar').classes()).toContain('search-bar--query');
    expect(wrapper.find('.search-bar').classes()).not.toContain('search-bar--text-process');

    await wrapper.find('[data-testid="mode-pill-text-process"]').trigger('click');

    expect(wrapper.find('.search-bar').classes()).toContain('search-bar--text-process');
    expect(wrapper.find('.search-bar').classes()).not.toContain('search-bar--query');
  });

  it('overlays the submit affordance in full-text mode so the textarea keeps the full row width', async () => {
    const wrapper = await mountQueryInterface();

    expect(wrapper.find('.search-action').classes()).not.toContain('search-action--overlay');

    await wrapper.find('[data-testid="mode-pill-text-process"]').trigger('click');

    expect(wrapper.find('.search-action').classes()).toContain('search-action--overlay');
  });

  it('keeps the query, full-text, and settings controls mounted below the field', async () => {
    const wrapper = await mountQueryInterface();

    expect(wrapper.find('[data-testid="mode-pill-query"]').exists()).toBe(true);
    expect(wrapper.find('[data-testid="mode-pill-text-process"]').exists()).toBe(true);
    expect(wrapper.find('[data-testid="search-settings-button"]').exists()).toBe(true);
  });

  it('switches back to query mode from the visible search-shell pill', async () => {
    const wrapper = await mountQueryInterface();

    await wrapper.setData({ forceEndpointMode: 'textProcess' });
    await wrapper.find('[data-testid="mode-pill-query"]').trigger('click');

    expect(wrapper.vm.forceEndpointMode).toBe('query');
    expect(wrapper.findComponent({ name: 'VTextField' }).exists()).toBe(true);
    expect(wrapper.findComponent({ name: 'VTextarea' }).exists()).toBe(false);
  });

  it('auto-switches to full-text mode for long text and shows a helper notice', async () => {
    const wrapper = await mountQueryInterface();

    expect(wrapper.vm.forceEndpointMode).toBe(null);

    await wrapper.setData({
      queryText:
        'This is a deliberately long clinical note text that should cross the query threshold and trigger an automatic switch into full-text mode for document review.',
    });

    expect(wrapper.vm.forceEndpointMode).toBe('textProcess');
    expect(wrapper.find('[data-testid="mode-auto-switch-notice"]').exists()).toBe(true);
  });

  it('does not auto-switch to full-text mode for short query text', async () => {
    const wrapper = await mountQueryInterface();

    await wrapper.setData({
      queryText: 'short syndrome query',
    });

    expect(wrapper.vm.forceEndpointMode).toBe(null);
    expect(wrapper.find('[data-testid="mode-auto-switch-notice"]').exists()).toBe(false);
  });

  it('hydrates URL parameters and auto-submits once models finish loading', async () => {
    vi.useFakeTimers();

    const wrapper = await mountQueryInterface({
      routeQuery: {
        text: 'Patient had recurrent seizures.',
        model: 'test-model',
        threshold: '0.8',
        forceEndpointMode: 'textProcess',
        chunkingStrategy: 'by_sentence',
        autoSubmit: 'true',
      },
    });

    await flushPromises();
    await wrapper.vm.$nextTick();
    await vi.runAllTimersAsync();
    await flushPromises();

    expect(wrapper.vm.queryText).toBe('Patient had recurrent seizures.');
    expect(wrapper.vm.selectedModel).toBe('test-model');
    expect(wrapper.vm.similarityThreshold).toBe(0.8);
    expect(wrapper.vm.forceEndpointMode).toBe('textProcess');
    expect(wrapper.vm.chunkingStrategy).toBe('by_sentence');
    expect(wrapper.vm.showAdvancedOptions).toBe(true);
    expect(PhentrieveService.processText).toHaveBeenCalledTimes(1);

    vi.useRealTimers();
  });

  it('hydrates array query params safely and auto-submits using the first value', async () => {
    vi.useFakeTimers();

    const wrapper = await mountQueryInterface({
      routeQuery: {
        text: ['Patient had recurrent seizures.', 'ignored'],
        model: ['test-model', 'ignored-model'],
        threshold: ['0.8', '0.4'],
        forceEndpointMode: ['textProcess', 'query'],
        chunkingStrategy: ['by_sentence', 'sliding_window'],
        autoSubmit: ['true', 'false'],
      },
    });

    await flushPromises();
    await wrapper.vm.$nextTick();
    await vi.runAllTimersAsync();
    await flushPromises();

    expect(wrapper.vm.queryText).toBe('Patient had recurrent seizures.');
    expect(wrapper.vm.selectedModel).toBe('test-model');
    expect(wrapper.vm.similarityThreshold).toBe(0.8);
    expect(wrapper.vm.forceEndpointMode).toBe('textProcess');
    expect(wrapper.vm.chunkingStrategy).toBe('by_sentence');
    expect(PhentrieveService.processText).toHaveBeenCalledTimes(1);
  });

  it('uses the fallback model when config loading fails', async () => {
    PhentrieveService.getConfigInfo.mockRejectedValueOnce(new Error('network down'));

    const wrapper = await mountQueryInterface();
    await flushPromises();

    expect(wrapper.vm.availableModels).toEqual([
      { text: 'BioLORD 2023-M', value: 'FremyCompany/BioLORD-2023-M' },
    ]);
    expect(wrapper.vm.selectedModel).toBe('FremyCompany/BioLORD-2023-M');
  });

  it('uses the fallback model when config returns no available models', async () => {
    PhentrieveService.getConfigInfo.mockResolvedValueOnce({
      available_embedding_models: [],
      default_embedding_model: null,
    });

    const wrapper = await mountQueryInterface();
    await flushPromises();

    expect(wrapper.vm.availableModels).toEqual([
      { text: 'BioLORD 2023-M', value: 'FremyCompany/BioLORD-2023-M' },
    ]);
    expect(wrapper.vm.selectedModel).toBe('FremyCompany/BioLORD-2023-M');
  });

  it('submits query-mode requests through queryHpo with the selected options', async () => {
    const wrapper = await mountQueryInterface();
    await flushPromises();

    await wrapper.setData({
      queryText: 'short syndrome query',
      forceEndpointMode: 'query',
      selectedLanguage: 'de',
      selectedModel: 'test-model',
    });
    wrapper.vm.similarityThreshold = 0.65;
    await wrapper.vm.$nextTick();

    await wrapper.vm.submitQuery();

    expect(PhentrieveService.queryHpo).toHaveBeenCalledWith(
      expect.objectContaining({
        text: 'short syndrome query',
        model_name: 'test-model',
        language: 'de',
        similarity_threshold: 0.65,
        include_details: wrapper.vm.includeDetails,
      })
    );
    expect(wrapper.vm.queryText).toBe('');
  });

  it('submits text-process requests through processText with controller-managed options', async () => {
    const wrapper = await mountQueryInterface();
    await flushPromises();

    await wrapper.setData({
      queryText: 'Patient had recurrent seizures.',
      forceEndpointMode: 'textProcess',
      selectedLanguage: 'en',
      selectedModel: 'test-model',
      textProcessOptions: {
        extractionBackend: 'llm',
        llmModel: 'gemini-test',
        llmMode: 'two_phase',
      },
      chunkingStrategy: 'sliding_window',
      windowSize: 3,
      stepSize: 1,
    });

    await wrapper.vm.submitQuery();

    expect(PhentrieveService.processText).toHaveBeenCalledWith(
      expect.objectContaining({
        text: 'Patient had recurrent seizures.',
        llmModel: 'gemini-test',
        llmMode: 'two_phase',
        language: 'en',
        chunkingStrategy: 'sliding_window',
        windowSize: 3,
        stepSize: 1,
        semanticModelForChunking: 'test-model',
        retrievalModelForTextProcess: 'test-model',
        includeDetails: wrapper.vm.includeDetails,
      })
    );
  });

  it('clears autoSubmit from the URL after a manual submit', async () => {
    const routerReplace = vi.fn().mockResolvedValue(undefined);
    const wrapper = await mountQueryInterface({
      routeQuery: {
        autoSubmit: 'true',
        model: 'test-model',
        threshold: '0.8',
      },
      routerReplace,
    });
    await flushPromises();

    await wrapper.setData({
      queryText: 'short syndrome query',
      forceEndpointMode: 'query',
      selectedModel: 'test-model',
    });

    await wrapper.vm.submitQuery();
    await flushPromises();

    expect(routerReplace).toHaveBeenCalledWith({
      query: {
        model: 'test-model',
        threshold: '0.8',
      },
    });
  });

  it('initializes a workspace turn after a text processing response arrives', async () => {
    const wrapper = await mountQueryInterface();
    const workspaceStore = useFullTextWorkspaceStore();

    await wrapper.setData({
      queryText: 'Patient had recurrent seizures.',
      forceEndpointMode: 'textProcess',
    });

    await wrapper.vm.submitQuery();

    const latestQuery = wrapper.vm.conversationStore.queryHistory[0];
    expect(latestQuery.type).toBe('textProcess');
    expect(workspaceStore.hasTurn(latestQuery.id)).toBe(true);
    expect(workspaceStore.getTurnState(latestQuery.id)?.expanded).toBe(true);
  });

  it('expands the submitted clinical note by default for new text-processing turns', async () => {
    const wrapper = await mountQueryInterface();

    await wrapper.setData({
      queryText: 'Patient had recurrent seizures.',
      forceEndpointMode: 'textProcess',
    });

    await wrapper.vm.submitQuery();

    const latestQuery = wrapper.vm.conversationStore.queryHistory[0];
    expect(latestQuery.type).toBe('textProcess');
    expect(wrapper.vm.isUserNoteExpanded(latestQuery.id)).toBe(true);
  });

  it('hides the truncated note preview while the clinical note is expanded', async () => {
    const wrapper = await mountQueryInterface();

    wrapper.vm.conversationStore.queryHistory.unshift({
      id: 'turn-expanded-note',
      query:
        'Patient had recurrent seizures and developmental delay documented across multiple visits.',
      type: 'textProcess',
      loading: false,
      error: null,
      response: {
        aggregated_hpo_terms: [],
        processed_chunks: [],
      },
    });
    wrapper.vm.expandedUserNotes['turn-expanded-note'] = true;
    await wrapper.vm.$nextTick();

    expect(wrapper.find('[data-testid="user-note-expanded"]').exists()).toBe(true);
    expect(wrapper.find('.user-note-summary__preview').exists()).toBe(false);
  });

  it('keeps text-processing turns inside the submitted-note bubble instead of rendering a separate workspace', async () => {
    const wrapper = await mountQueryInterface();

    wrapper.vm.conversationStore.queryHistory.unshift({
      id: 'turn-inline-note',
      query: 'clinical note',
      type: 'textProcess',
      loading: false,
      error: null,
      response: {
        aggregated_hpo_terms: [{ hpo_id: 'HP:0001250', name: 'Seizure' }],
        processed_chunks: [{ chunk_id: 0, chunk_text: 'note' }],
      },
    });
    await wrapper.vm.$nextTick();

    expect(wrapper.find('[data-testid="user-note-summary"]').exists()).toBe(true);
    expect(wrapper.text()).toContain('Full-text analysis ready');
    expect(wrapper.findComponent({ name: 'FullTextResponseReceipt' }).exists()).toBe(true);
  });

  it('shows a real error for failed full-text requests instead of a zero-findings receipt', async () => {
    const wrapper = await mountQueryInterface();

    wrapper.vm.conversationStore.queryHistory.unshift({
      id: 'turn-text-process-error',
      query: 'clinical note',
      type: 'textProcess',
      loading: false,
      response: null,
      error: {
        detail: 'Gemini API key not configured',
      },
    });
    await wrapper.vm.$nextTick();

    expect(wrapper.text()).toContain('Gemini API key not configured');
    expect(wrapper.text()).not.toContain('Full-text analysis ready');
  });

  it('keeps the full-text receipt before older query results without adding a separate workspace block', async () => {
    const wrapper = await mountQueryInterface();

    wrapper.vm.conversationStore.queryHistory.push({
      id: 'older-query',
      query: 'Kleinwuchs',
      type: 'query',
      loading: false,
      error: null,
      response: {
        results: [{ hpo_id: 'HP:0004322', label: 'Short stature', similarity: 0.97 }],
      },
    });

    wrapper.vm.conversationStore.queryHistory.unshift({
      id: 'latest-text-process',
      query: 'clinical note',
      type: 'textProcess',
      loading: false,
      error: null,
      response: {
        aggregated_hpo_terms: [{ hpo_id: 'HP:0001250', name: 'Seizure' }],
        processed_chunks: [],
      },
    });
    await wrapper.vm.$nextTick();

    const html = wrapper.html();
    const receiptIndex = html.indexOf('Full-text analysis ready');
    const olderQueryIndex = html.indexOf('Kleinwuchs');

    expect(receiptIndex).toBeGreaterThan(-1);
    expect(receiptIndex).toBeLessThan(olderQueryIndex);
  });

  it('does not attach a second full-text workspace surface to the response bubble', async () => {
    const wrapper = await mountQueryInterface();

    wrapper.vm.conversationStore.queryHistory.unshift({
      id: 'attached-text-process',
      query: 'clinical note',
      type: 'textProcess',
      loading: false,
      error: null,
      response: {
        aggregated_hpo_terms: [{ hpo_id: 'HP:0001250', name: 'Seizure' }],
        processed_chunks: [],
      },
    });
    await wrapper.vm.$nextTick();

    const responseBubble = wrapper.find('.response-bubble');
    expect(responseBubble.exists()).toBe(true);
    expect(responseBubble.text()).not.toContain('add all');
  });

  it('keeps the shared phenotype collection sidebar mounted in full-text mode', async () => {
    const wrapper = await mountQueryInterface();

    expect(wrapper.findComponent({ name: 'PhenotypeCollectionPanel' }).exists()).toBe(true);

    wrapper.vm.conversationStore.queryHistory.unshift({
      id: 'turn-text-process',
      query: 'clinical note',
      type: 'textProcess',
      loading: false,
      error: null,
      response: {
        aggregated_hpo_terms: [{ hpo_id: 'HP:0001250', name: 'Seizure' }],
        processed_chunks: [{ chunk_id: 0, chunk_text: 'note' }],
      },
    });
    await wrapper.vm.$nextTick();

    expect(wrapper.findComponent({ name: 'PhenotypeCollectionPanel' }).exists()).toBe(true);
  });

  it('shows extracted phenotypes in the bot response instead of the clinical-note bubble', async () => {
    const wrapper = await mountQueryInterface();

    wrapper.vm.conversationStore.queryHistory.unshift({
      id: 'turn-inline-phenotypes',
      query: 'Patient had recurrent seizures and developmental delay.',
      type: 'textProcess',
      loading: false,
      error: null,
      response: {
        aggregated_hpo_terms: [
          {
            hpo_id: 'HP:0001250',
            name: 'Seizure',
            status: 'present',
            text_attributions: [
              {
                chunk_id: 1,
                start_char: 22,
                end_char: 30,
                matched_text_in_chunk: 'seizures',
              },
            ],
          },
        ],
        processed_chunks: [
          { chunk_id: 1, text: 'Patient had recurrent seizures and developmental delay.' },
        ],
      },
    });
    await wrapper.vm.$nextTick();

    expect(wrapper.find('[data-testid="note-phenotype-chip"]').exists()).toBe(false);
    expect(wrapper.find('[data-testid="full-text-response-phenotype"]').exists()).toBe(true);
    expect(wrapper.find('[data-testid="full-text-response-phenotype"]').text()).toContain(
      'Seizure'
    );
  });

  it('renders the full-text confidence score in the response phenotype card', async () => {
    const wrapper = await mountQueryInterface();

    wrapper.vm.conversationStore.queryHistory.unshift({
      id: 'turn-response-confidence',
      query: 'Patient had recurrent seizures.',
      type: 'textProcess',
      loading: false,
      error: null,
      response: {
        aggregated_hpo_terms: [
          {
            hpo_id: 'HP:0001250',
            name: 'Seizure',
            confidence: 0.91,
            status: 'present',
          },
        ],
        processed_chunks: [{ chunk_id: 1, text: 'Patient had recurrent seizures.' }],
      },
    });
    await wrapper.vm.$nextTick();

    expect(wrapper.find('[data-testid="full-text-response-phenotype"]').text()).toContain('0.91');
  });

  it('renders full-text confidence when the API payload provides it as a numeric string', async () => {
    const wrapper = await mountQueryInterface();

    wrapper.vm.conversationStore.queryHistory.unshift({
      id: 'turn-response-confidence-string',
      query: 'Patient had recurrent seizures.',
      type: 'textProcess',
      loading: false,
      error: null,
      response: {
        aggregated_hpo_terms: [
          {
            hpo_id: 'HP:0001250',
            name: 'Seizure',
            confidence: '0.91',
            status: 'present',
          },
        ],
        processed_chunks: [{ chunk_id: 1, text: 'Patient had recurrent seizures.' }],
      },
    });
    await wrapper.vm.$nextTick();

    expect(wrapper.find('[data-testid="full-text-response-phenotype"]').text()).toContain('0.91');
  });

  it('adds all full-text phenotypes from the response bubble into the collection', async () => {
    const wrapper = await mountQueryInterface();

    wrapper.vm.conversationStore.queryHistory.unshift({
      id: 'turn-response-add-all',
      query: 'Patient had recurrent seizures and developmental delay.',
      type: 'textProcess',
      loading: false,
      error: null,
      response: {
        aggregated_hpo_terms: [
          {
            hpo_id: 'HP:0001250',
            name: 'Seizure',
            confidence: 0.91,
            status: 'present',
          },
          {
            hpo_id: 'HP:0001263',
            name: 'Developmental delay',
            confidence: 0.83,
            status: 'present',
          },
        ],
        processed_chunks: [
          { chunk_id: 1, text: 'Patient had recurrent seizures and developmental delay.' },
        ],
      },
    });
    await wrapper.vm.$nextTick();

    await wrapper.get('[data-testid="full-text-response-add-all"]').trigger('click');

    expect(wrapper.vm.conversationStore.showCollectionPanel).toBe(true);
    expect(wrapper.vm.conversationStore.collectedPhenotypes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ hpo_id: 'HP:0001250', label: 'Seizure' }),
        expect.objectContaining({ hpo_id: 'HP:0001263', label: 'Developmental delay' }),
      ])
    );
  });

  it('renders text-processing turns as a compact submitted-note summary instead of repeating the full note', async () => {
    const wrapper = await mountQueryInterface();
    const longNote =
      'NAA10-related syndrome is an X-linked condition with a broad spectrum of findings ranging from a severe phenotype in males with p.Ser37Pro in NAA10 to milder intellectual disability with different variants in males and females.';

    wrapper.vm.conversationStore.queryHistory.unshift({
      id: 'turn-compact-note',
      query: longNote,
      type: 'textProcess',
      loading: false,
      error: null,
      response: {
        aggregated_hpo_terms: [{ hpo_id: 'HP:0001250', name: 'Seizure' }],
        processed_chunks: [{ chunk_id: 0, chunk_text: 'note' }],
      },
    });
    await wrapper.vm.$nextTick();

    expect(wrapper.find('[data-testid="user-note-summary"]').exists()).toBe(true);
    expect(wrapper.find('[data-testid="user-note-summary"]').text()).toContain('Clinical note');
    expect(wrapper.find('[data-testid="user-note-summary"]').text()).not.toContain(longNote);
  });

  it('expands the submitted clinical note and highlights evidence spans from processed chunks', async () => {
    const wrapper = await mountQueryInterface();
    const note =
      'Patient had recurrent seizures and developmental delay documented in the clinical note.';

    wrapper.vm.conversationStore.queryHistory.unshift({
      id: 'turn-highlight-note',
      query: note,
      type: 'textProcess',
      loading: false,
      error: null,
      response: {
        processed_chunks: [
          {
            chunk_id: 1,
            text: note,
          },
        ],
        aggregated_hpo_terms: [
          {
            hpo_id: 'HP:0001250',
            name: 'Seizure',
            text_attributions: [
              {
                chunk_id: 1,
                start_char: 22,
                end_char: 30,
                matched_text_in_chunk: 'seizures',
              },
            ],
          },
        ],
      },
    });
    await wrapper.vm.$nextTick();

    await wrapper.get('[data-testid="user-note-summary-toggle"]').trigger('click');

    expect(wrapper.find('[data-testid="user-note-expanded"]').exists()).toBe(true);
    expect(wrapper.findAll('[data-testid="annotated-note-span"]').length).toBeGreaterThan(0);
    expect(wrapper.find('[data-testid="user-note-expanded"]').text()).toContain('seizures');
    expect(wrapper.find('[data-testid="user-note-expanded"] mark').exists()).toBe(true);
  });

  it('maps multi-chunk evidence spans back into the submitted note bubble', async () => {
    const wrapper = await mountQueryInterface();
    const note = 'Patient had recurrent seizures and developmental delay in clinic.';

    wrapper.vm.conversationStore.queryHistory.unshift({
      id: 'turn-highlight-multi-chunk',
      query: note,
      type: 'textProcess',
      loading: false,
      error: null,
      response: {
        processed_chunks: [
          {
            chunk_id: 1,
            text: 'Patient had recurrent seizures',
          },
          {
            chunk_id: 2,
            text: 'developmental delay in clinic.',
          },
        ],
        aggregated_hpo_terms: [
          {
            hpo_id: 'HP:0001263',
            name: 'Developmental delay',
            text_attributions: [
              {
                chunk_id: 2,
                start_char: 0,
                end_char: 19,
                matched_text_in_chunk: 'developmental delay',
              },
            ],
          },
        ],
      },
    });
    await wrapper.vm.$nextTick();

    await wrapper.get('[data-testid="user-note-summary-toggle"]').trigger('click');

    const marks = wrapper.findAll('[data-testid="user-note-expanded"] mark');
    expect(marks.length).toBeGreaterThan(0);
    expect(marks.some((mark) => mark.text().includes('developmental delay'))).toBe(true);
  });

  it('focuses note highlighting to the hovered phenotype in the bot response', async () => {
    const wrapper = await mountQueryInterface();
    const note = 'Patient had recurrent seizures and developmental delay in clinic.';

    wrapper.vm.conversationStore.queryHistory.unshift({
      id: 'turn-hover-chip',
      query: note,
      type: 'textProcess',
      loading: false,
      error: null,
      response: {
        processed_chunks: [
          {
            chunk_id: 1,
            text: note,
          },
        ],
        aggregated_hpo_terms: [
          {
            hpo_id: 'HP:0001250',
            name: 'Seizure',
            text_attributions: [
              {
                chunk_id: 1,
                start_char: 22,
                end_char: 30,
                matched_text_in_chunk: 'seizures',
              },
            ],
          },
          {
            hpo_id: 'HP:0001263',
            name: 'Developmental delay',
            text_attributions: [
              {
                chunk_id: 1,
                start_char: 35,
                end_char: 54,
                matched_text_in_chunk: 'developmental delay',
              },
            ],
          },
        ],
      },
    });
    await wrapper.vm.$nextTick();

    await wrapper.get('[data-testid="user-note-summary-toggle"]').trigger('click');

    const phenotypes = wrapper.findAll('[data-testid="full-text-response-phenotype"]');
    await phenotypes[0].trigger('mouseenter');

    const marks = wrapper.findAll('[data-testid="user-note-expanded"] mark');
    expect(marks).toHaveLength(1);
    expect(marks[0].text()).toContain('seizures');
  });

  it('shows the linked HPO term in the annotated note span tooltip', async () => {
    const wrapper = await mountQueryInterface();
    const note = 'Patient had recurrent seizures and developmental delay in clinic.';

    wrapper.vm.conversationStore.queryHistory.unshift({
      id: 'turn-note-tooltip',
      query: note,
      type: 'textProcess',
      loading: false,
      error: null,
      response: {
        processed_chunks: [
          {
            chunk_id: 1,
            text: note,
          },
        ],
        aggregated_hpo_terms: [
          {
            hpo_id: 'HP:0001250',
            name: 'Seizure',
            text_attributions: [
              {
                chunk_id: 1,
                start_char: 22,
                end_char: 30,
                matched_text_in_chunk: 'seizures',
              },
            ],
          },
        ],
      },
    });
    await wrapper.vm.$nextTick();

    await wrapper.get('[data-testid="user-note-summary-toggle"]').trigger('click');

    const mark = wrapper.get('[data-testid="annotated-note-span"]');
    expect(mark.attributes('title')).toBeUndefined();
    expect(mark.classes()).toContain('annotated-note-span');
  });

  it('uses interactive annotation styling in the expanded note without inline padding', async () => {
    const wrapper = await mountQueryInterface();
    const note = 'Patient had recurrent seizures and developmental delay in clinic.';

    wrapper.vm.conversationStore.queryHistory.unshift({
      id: 'turn-note-styling',
      query: note,
      type: 'textProcess',
      loading: false,
      error: null,
      response: {
        processed_chunks: [
          {
            chunk_id: 1,
            text: note,
          },
        ],
        aggregated_hpo_terms: [
          {
            hpo_id: 'HP:0001250',
            name: 'Seizure',
            text_attributions: [
              {
                chunk_id: 1,
                start_char: 22,
                end_char: 30,
                matched_text_in_chunk: 'seizures',
              },
            ],
          },
        ],
      },
    });
    await wrapper.vm.$nextTick();

    await wrapper.get('[data-testid="user-note-summary-toggle"]').trigger('click');

    const mark = wrapper.get('[data-testid="annotated-note-span"]');
    expect(mark.classes()).toContain('annotated-note-span');
    expect(mark.attributes('style') || '').not.toContain('padding');
  });
});
