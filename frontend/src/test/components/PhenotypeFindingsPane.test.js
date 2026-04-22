import { describe, it, expect, afterEach, vi } from 'vitest';
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

describe('PhenotypeFindingsPane', () => {
  it('renders confidence bands instead of raw numeric scores in the primary list', async () => {
    const component = (await import('../../components/PhenotypeFindingsPane.vue')).default;
    const wrapper = mount(component, {
      props: {
        terms: [
          {
            hpo_id: 'HP:0001250',
            name: 'Seizure',
            confidence: 0.92,
            status: 'affirmed',
            source_chunk_ids: [1],
          },
        ],
      },
      global: {
        plugins: [vuetify, i18n],
      },
    });

    expect(wrapper.text()).toContain('High');
    expect(wrapper.text()).not.toContain('0.92');
    expect(wrapper.text()).toContain('Affirmed');
    expect(wrapper.text()).toContain('1 evidence chunk');
    expect(wrapper.text()).toContain('Top evidence: #1');
  });

  it('emits hover, clear, and inspect events from the findings list interactions', async () => {
    const component = (await import('../../components/PhenotypeFindingsPane.vue')).default;
    const wrapper = mount(component, {
      props: {
        terms: [
          {
            hpo_id: 'HP:0001250',
            name: 'Seizure',
            confidence: 0.92,
            status: 'affirmed',
            source_chunk_ids: [1, 3],
            top_evidence_chunk_id: 3,
          },
        ],
      },
      global: {
        plugins: [vuetify, i18n],
      },
    });

    const termItem = wrapper.get('.v-list-item');

    await termItem.trigger('mouseenter');
    await termItem.trigger('mouseleave');
    await termItem.trigger('click');

    expect(wrapper.emitted('hover-term')).toEqual([['HP:0001250']]);
    expect(wrapper.emitted('clear-hover')).toEqual([[]]);
    expect(wrapper.emitted('inspect-term')).toEqual([['HP:0001250']]);
  });

  it('renders inspector-only numeric details in the annotation panel', async () => {
    const component = (await import('../../components/AnnotationInspectorPanel.vue')).default;
    const wrapper = mount(component, {
      props: {
        selectedTerm: {
          hpo_id: 'HP:0001250',
          name: 'Seizure',
          confidence: 0.92,
        },
      },
      global: {
        plugins: [vuetify, i18n],
      },
    });

    expect(wrapper.text()).toContain('Annotation Inspector');
    expect(wrapper.text()).toContain('Confidence: 0.92');
    expect(wrapper.get('[id="annotation-detail-HP:0001250"]').text()).toContain('Seizure');
  });

  it('emits back from the inspector return action', async () => {
    const component = (await import('../../components/AnnotationInspectorPanel.vue')).default;
    const wrapper = mount(component, {
      props: {
        selectedTerm: {
          hpo_id: 'HP:0001250',
          name: 'Seizure',
          confidence: 0.92,
        },
      },
      global: {
        plugins: [vuetify, i18n],
      },
    });

    await wrapper.get('button').trigger('click');

    expect(wrapper.emitted('back')).toEqual([[]]);
  });
});
