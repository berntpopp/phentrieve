import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { computed, ref } from 'vue';
import { useQueryInterfaceController } from '../../composables/useQueryInterfaceController';

function createState(overrides = {}) {
  const state = {
    queryText: ref(''),
    selectedModel: ref(null),
    availableModels: ref([]),
    modelsLoading: ref(true),
    isLoading: ref(false),
    shouldScrollToTop: ref(false),
    userHasScrolled: ref(true),
    showAdvancedOptions: ref(false),
    forceEndpointMode: ref(null),
    selectedLanguage: ref('en'),
    similarityThreshold: ref(0.3),
    numResults: ref(10),
    includeDetails: ref(false),
    expandedUserNotes: ref({}),
    textProcessOptions: ref({
      extractionBackend: 'llm',
      llmModel: 'llm-a',
      llmMode: 'two_phase',
    }),
    chunkingStrategy: ref('sliding_window_punct_conj_cleaned'),
    windowSize: ref(3),
    stepSize: ref(1),
    splitThreshold: ref(0.5),
    minSegmentLength: ref(2),
    semanticModelForChunking: ref(null),
    retrievalModelForTextProcess: ref(null),
    chunkRetrievalThreshold: ref(0.7),
    numResultsPerChunk: ref(3),
    noAssertionDetectionForTextProcess: ref(false),
    assertionPreferenceForTextProcess: ref('dependency'),
    aggregatedTermConfidence: ref(0.75),
    topTermPerChunkForAggregation: ref(false),
    ...overrides,
  };
  state.isTextProcessModeActive = computed(() => state.forceEndpointMode.value === 'textProcess');
  return state;
}

function createController({ state = createState(), routeQuery = {}, piiFindings = false } = {}) {
  const conversationStore = {
    addQuery: vi.fn(() => 'turn-1'),
    updateQueryResponse: vi.fn(),
  };
  const piiReviewFlow = {
    scanSubmissionForPii: vi.fn(() => ({
      hasFindings: piiFindings,
      scanResult: { summary: { high: {}, review: {} } },
    })),
    continueWithPiiRedaction: vi.fn(),
    redactPiiInInput: vi.fn(({ setQueryText }) => setQueryText('[redacted] text')),
  };
  const router = { replace: vi.fn().mockResolvedValue(undefined) };
  const service = {
    getConfigInfo: vi.fn().mockResolvedValue({
      available_embedding_models: [{ id: 'org/model-a' }, { id: 'org/model-b' }],
      default_embedding_model: 'org/model-b',
    }),
    queryHpo: vi.fn().mockResolvedValue({ results: [] }),
    processText: vi.fn().mockResolvedValue({ processed_chunks: [] }),
  };
  const logService = { info: vi.fn(), debug: vi.fn(), warn: vi.fn(), error: vi.fn() };
  const controller = useQueryInterfaceController({
    state,
    conversationStore,
    piiReviewFlow,
    route: { query: routeQuery },
    router,
    nextTick: (fn) => fn(),
    service,
    logService,
  });
  return { controller, state, conversationStore, piiReviewFlow, router, service };
}

describe('useQueryInterfaceController', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('loads models from the API into the state refs', async () => {
    const { controller, state } = createController();

    await controller.fetchAvailableModels();

    expect(state.availableModels.value).toEqual([
      { text: 'model-a', value: 'org/model-a' },
      { text: 'model-b', value: 'org/model-b' },
    ]);
    expect(state.selectedModel.value).toBe('org/model-b');
    expect(state.modelsLoading.value).toBe(false);
  });

  it('falls back to BioLORD when the config request fails', async () => {
    const { controller, state, service } = createController();
    service.getConfigInfo.mockRejectedValueOnce(new Error('offline'));

    await controller.fetchAvailableModels();

    expect(state.selectedModel.value).toBe('FremyCompany/BioLORD-2023-M');
    expect(state.modelsLoading.value).toBe(false);
  });

  it('applies URL parameters and auto-submits once after models load', async () => {
    const { controller, state, service } = createController({
      routeQuery: { text: 'Seizures', model: 'org/model-a', threshold: '0.8' },
    });

    await controller.fetchAvailableModels();

    expect(state.queryText.value).toBe('Seizures');
    expect(state.selectedModel.value).toBe('org/model-a');
    expect(state.similarityThreshold.value).toBe(0.8);
    expect(state.showAdvancedOptions.value).toBe(true);

    await vi.advanceTimersByTimeAsync(300);
    expect(service.queryHpo).toHaveBeenCalledTimes(1);
    expect(service.queryHpo).toHaveBeenCalledWith(
      expect.objectContaining({ text: 'Seizures', similarity_threshold: 0.8 })
    );
    // Auto-submit keeps the query text in the input.
    expect(state.queryText.value).toBe('Seizures');
  });

  it('submits a query payload from the refs and clears the input', async () => {
    const state = createState({ queryText: ref('  Seizures  '), selectedModel: ref('m') });
    const { controller, service, conversationStore } = createController({ state });

    await controller.submitQuery();

    expect(service.queryHpo).toHaveBeenCalledWith({
      text: 'Seizures',
      model_name: 'm',
      language: 'en',
      num_results: 10,
      similarity_threshold: 0.3,
      query_assertion_language: 'en',
      detect_query_assertion: true,
      include_details: false,
    });
    expect(conversationStore.updateQueryResponse).toHaveBeenCalledWith('turn-1', {
      results: [],
    });
    expect(state.queryText.value).toBe('');
    expect(state.isLoading.value).toBe(false);
    expect(state.shouldScrollToTop.value).toBe(true);
    expect(state.userHasScrolled.value).toBe(false);
  });

  it('submits full text with chunking options and expands the user note', async () => {
    const state = createState({
      queryText: ref('Long clinical note'),
      selectedModel: ref('m'),
      forceEndpointMode: ref('textProcess'),
    });
    const { controller, service } = createController({ state });

    await controller.submitQuery();

    expect(service.processText).toHaveBeenCalledWith(
      expect.objectContaining({
        text: 'Long clinical note',
        extractionBackend: 'llm',
        llmModel: 'llm-a',
        semanticModelForChunking: 'm',
        retrievalModelForTextProcess: 'm',
        windowSize: 3,
      })
    );
    expect(state.expandedUserNotes.value).toEqual({ 'turn-1': true });
  });

  it('stops for PII review and clears autoSubmit from the URL', async () => {
    const state = createState({ queryText: ref('jane@example.org') });
    const { controller, service, router } = createController({
      state,
      piiFindings: true,
      routeQuery: { autoSubmit: 'true', model: 'm' },
    });

    await controller.submitQuery();

    expect(service.queryHpo).not.toHaveBeenCalled();
    expect(router.replace).toHaveBeenCalledWith({ query: { model: 'm' } });
  });

  it('writes the redacted text back to the queryText ref', () => {
    const { controller, state } = createController();

    controller.redactPiiInInput();

    expect(state.queryText.value).toBe('[redacted] text');
  });
});
