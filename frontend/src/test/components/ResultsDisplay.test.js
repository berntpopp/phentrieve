import { describe, it, expect } from 'vitest';
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

describe('ResultsDisplay', () => {
  it('renders quota metadata and tolerates empty processed_chunks', async () => {
    const component = (await import('../../components/ResultsDisplay.vue')).default;
    const wrapper = mount(component, {
      props: {
        resultType: 'textProcess',
        responseData: {
          meta: { extraction_backend: 'llm', quota_remaining: 2, quota_limit: 3 },
          processed_chunks: [],
          aggregated_hpo_terms: [],
        },
      },
      global: {
        plugins: [vuetify, i18n],
        stubs: {
          ChunkResultsView: true,
          AggregatedTermsView: true,
          ResultItem: true,
        },
      },
    });

    expect(wrapper.text()).toContain('2 / 3');
  });
});
