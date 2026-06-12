import { describe, it, expect, vi, afterEach } from 'vitest';
import { mount } from '@vue/test-utils';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';

const vuetify = createVuetify({ components, directives });
const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: { en },
  silentTranslationWarn: true,
  silentFallbackWarn: true,
  missing: () => '',
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe('ResultsDisplay', () => {
  it('does not log while validating responseData props', async () => {
    const debugSpy = vi.spyOn(console, 'debug').mockImplementation(() => {});
    const logModule = await import('../../services/logService');
    const serviceDebugSpy = vi.spyOn(logModule.logService, 'debug').mockImplementation(() => {});
    const component = (await import('../../components/ResultsDisplay.vue')).default;

    mount(component, {
      props: {
        resultType: 'query',
        responseData: {
          model_used_for_retrieval: 'test-model',
          language_detected: 'en',
          query_assertion_status: 'affirmed',
          results: [],
        },
      },
      global: {
        plugins: [vuetify, i18n],
        stubs: {
          FullTextAnnotationWorkspace: true,
          ResultItem: true,
        },
      },
    });

    expect(debugSpy).not.toHaveBeenCalled();
    expect(serviceDebugSpy).not.toHaveBeenCalled();
  });

  it('mounts the unified full-text workspace for text processing results', async () => {
    const component = (await import('../../components/ResultsDisplay.vue')).default;
    const wrapper = mount(component, {
      props: {
        resultType: 'textProcess',
        turnId: 'turn-1',
        responseData: {
          meta: { extraction_backend: 'llm', quota_remaining: 2, quota_limit: 7 },
          processed_chunks: [],
          aggregated_hpo_terms: [],
        },
      },
      global: {
        plugins: [vuetify, i18n],
        stubs: {
          FullTextAnnotationWorkspace: true,
          ResultItem: true,
        },
      },
    });

    const workspace = wrapper.findComponent({ name: 'FullTextAnnotationWorkspace' });
    expect(workspace.exists()).toBe(true);
    expect(workspace.props('turnId')).toBe('turn-1');
  });

  it('passes full text response metadata through to the workspace shell', async () => {
    const component = (await import('../../components/ResultsDisplay.vue')).default;
    const wrapper = mount(component, {
      props: {
        resultType: 'textProcess',
        turnId: 'turn-2',
        responseData: {
          meta: { extraction_backend: 'llm', quota_remaining: 2, quota_limit: 3 },
          processed_chunks: [],
          aggregated_hpo_terms: [],
        },
      },
      global: {
        plugins: [vuetify, i18n],
        stubs: {
          FullTextAnnotationWorkspace: true,
          ResultItem: true,
        },
      },
    });

    const workspace = wrapper.findComponent({ name: 'FullTextAnnotationWorkspace' });
    expect(workspace.props('responseData')).toEqual({
      meta: { extraction_backend: 'llm', quota_remaining: 2, quota_limit: 3 },
      processed_chunks: [],
      aggregated_hpo_terms: [],
    });
  });

  it('does not mount the full-text workspace when turnId is missing', async () => {
    const component = (await import('../../components/ResultsDisplay.vue')).default;
    const wrapper = mount(component, {
      props: {
        resultType: 'textProcess',
        responseData: {
          meta: { extraction_backend: 'llm' },
          processed_chunks: [],
          aggregated_hpo_terms: [],
        },
      },
      global: {
        plugins: [vuetify, i18n],
        stubs: {
          FullTextAnnotationWorkspace: true,
          ResultItem: true,
        },
      },
    });

    expect(wrapper.findComponent({ name: 'FullTextAnnotationWorkspace' }).exists()).toBe(false);
    expect(wrapper.text()).toContain(i18n.global.t('resultsDisplay.defaultError'));
  });

  it('re-emits bulk full-text collection actions from the workspace', async () => {
    const component = (await import('../../components/ResultsDisplay.vue')).default;
    const workspacePayload = [
      { hpo_id: 'HP:0001250', label: 'Seizure', assertion_status: 'affirmed' },
    ];
    const wrapper = mount(component, {
      props: {
        resultType: 'textProcess',
        turnId: 'turn-3',
        responseData: {
          meta: { extraction_backend: 'llm' },
          processed_chunks: [],
          aggregated_hpo_terms: [],
        },
      },
      global: {
        plugins: [vuetify, i18n],
        stubs: {
          FullTextAnnotationWorkspace: {
            name: 'FullTextAnnotationWorkspace',
            template: '<button @click="$emit(\'add-all-to-collection\', payload)">Emit</button>',
            data() {
              return { payload: workspacePayload };
            },
          },
          ResultItem: true,
        },
      },
    });

    await wrapper.get('button').trigger('click');

    expect(wrapper.emitted('add-all-to-collection')).toEqual([[workspacePayload]]);
  });

  it('uses the exposed ChunkResultsView state when scrolling to attributed evidence', async () => {
    vi.useFakeTimers();

    const component = (await import('../../components/ResultsDisplay.vue')).default;
    const firstScrollIntoView = vi.fn();
    const firstFlashChunkText = vi.fn();
    const secondFlashChunkText = vi.fn();
    const firstChunkResultsView = {
      chunkPanelRefs: {
        2: {
          $el: { scrollIntoView: firstScrollIntoView },
        },
      },
      openChunkPanels: [],
      flashChunkText: firstFlashChunkText,
    };
    const secondChunkResultsView = {
      chunkPanelRefs: {
        2: {
          $el: { scrollIntoView: vi.fn() },
        },
      },
      openChunkPanels: [],
      flashChunkText: secondFlashChunkText,
    };
    const getElementByIdSpy = vi.spyOn(document, 'getElementById');

    component.methods.scrollToChunk.call(
      {
        $refs: { chunkResultsView: firstChunkResultsView },
        findChunkPanelComponent: component.methods.findChunkPanelComponent,
        getOpenChunkPanels: component.methods.getOpenChunkPanels,
        flashChunkTextForChunk: component.methods.flashChunkTextForChunk,
      },
      [2]
    );
    vi.advanceTimersByTime(300);

    expect(firstScrollIntoView).toHaveBeenCalledWith({ behavior: 'smooth', block: 'center' });
    expect(firstChunkResultsView.openChunkPanels).toEqual([1]);
    expect(firstFlashChunkText).toHaveBeenCalledWith(2);
    expect(secondFlashChunkText).not.toHaveBeenCalled();
    expect(secondChunkResultsView.openChunkPanels).toEqual([]);
    expect(getElementByIdSpy).not.toHaveBeenCalled();
  });
});

describe('AdvancedOptionsPanel', () => {
  it('uses a localized human-readable label for the visible LLM mode option', async () => {
    const component = (await import('../../components/AdvancedOptionsPanel.vue')).default;
    const wrapper = mount(component, {
      props: {
        visible: true,
        isTextProcessModeActive: true,
        textProcessOptions: {
          extractionBackend: 'llm',
          llmModel: 'gemini-3.1-flash-lite',
          llmMode: 'two_phase',
        },
      },
      global: {
        plugins: [vuetify, i18n],
      },
    });

    const llmModeSelect = wrapper
      .findAllComponents({ name: 'VSelect' })
      .find((select) => select.props('modelValue') === 'two_phase');

    expect(llmModeSelect).toBeDefined();
    expect(llmModeSelect.props('items')).toContainEqual({
      title: i18n.global.t('queryInterface.advancedOptions.llmModes.twoPhase'),
      value: 'two_phase',
    });
  });

  it('uses the shared LLM defaults supplied by the parent when textProcessOptions omits them', async () => {
    const component = (await import('../../components/AdvancedOptionsPanel.vue')).default;
    const wrapper = mount(component, {
      props: {
        visible: true,
        isTextProcessModeActive: true,
        defaultLlmModel: 'gemini-3.1-flash-lite',
        defaultLlmMode: 'two_phase',
        textProcessOptions: {
          extractionBackend: 'llm',
        },
      },
      global: {
        plugins: [vuetify, i18n],
      },
    });

    const textFields = wrapper.findAllComponents({ name: 'VTextField' });
    const llmModeSelect = wrapper
      .findAllComponents({ name: 'VSelect' })
      .find((select) => select.props('modelValue') === 'two_phase');

    expect(textFields.some((field) => field.props('modelValue') === 'gemini-3.1-flash-lite')).toBe(
      true
    );
    expect(llmModeSelect).toBeDefined();
  });
});
