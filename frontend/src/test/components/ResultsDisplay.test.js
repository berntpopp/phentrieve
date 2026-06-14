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
          ResultItem: true,
        },
      },
    });

    expect(debugSpy).not.toHaveBeenCalled();
    expect(serviceDebugSpy).not.toHaveBeenCalled();
  });

  it('shows the default error for text processing results without a valid turn id', async () => {
    const component = (await import('../../components/ResultsDisplay.vue')).default;
    const wrapper = mount(component, {
      props: {
        resultType: 'textProcess',
        responseData: {
          meta: { extraction_backend: 'standard' },
          processed_chunks: [],
          aggregated_hpo_terms: [{ hpo_id: 'HP:0001250', name: 'Seizure' }],
        },
      },
      global: {
        plugins: [vuetify, i18n],
        stubs: {
          ResultItem: true,
        },
      },
    });

    expect(wrapper.text()).toContain(i18n.global.t('resultsDisplay.defaultError'));
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
