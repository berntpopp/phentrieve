import { describe, it, expect } from 'vitest';
import { mount } from '@vue/test-utils';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';
import FullTextResponseReceipt from '../../components/FullTextResponseReceipt.vue';

const vuetify = createVuetify({ components, directives });
const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: { en },
  silentTranslationWarn: true,
  silentFallbackWarn: true,
  missing: () => '',
});

function mountReceipt(props = {}) {
  return mount(FullTextResponseReceipt, {
    props: {
      item: {
        id: 'turn-1',
        query: 'Patient had recurrent seizures.',
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
      },
      collectedPhenotypes: [],
      hoveredPhenotypeId: null,
      ...props,
    },
    global: {
      plugins: [vuetify, i18n],
      stubs: {
        ResultItem: {
          name: 'ResultItem',
          props: ['result', 'rank', 'isCollected'],
          emits: ['add-to-collection'],
          template: `
            <button
              class="result-item-stub"
              type="button"
              @click="$emit('add-to-collection', result)"
            >
              {{ rank }} {{ result.label }} {{ result.score }}
            </button>
          `,
        },
      },
    },
  });
}

describe('FullTextResponseReceipt', () => {
  it('emits add-all-to-collection for normalized full-text phenotypes', async () => {
    const wrapper = mountReceipt({
      item: {
        id: 'turn-1',
        response: {
          aggregated_hpo_terms: [{ hpo_id: 'HP:0001250', name: 'Seizure', status: 'present' }],
          processed_chunks: [{ chunk_id: 1, text: 'Patient had recurrent seizures.' }],
        },
      },
    });

    await wrapper.get('[data-testid="full-text-response-add-all"]').trigger('click');

    expect(wrapper.emitted('add-all-to-collection')).toHaveLength(1);
    expect(wrapper.emitted('add-all-to-collection')[0][0]).toEqual([
      {
        hpo_id: 'HP:0001250',
        label: 'Seizure',
        assertion_status: 'affirmed',
      },
    ]);
  });

  it('emits a normalized add-to-collection payload from a phenotype row', async () => {
    const wrapper = mountReceipt();

    await wrapper.get('.result-item-stub').trigger('click');

    expect(wrapper.emitted('add-to-collection')).toHaveLength(1);
    expect(wrapper.emitted('add-to-collection')[0][0]).toEqual({
      hpo_id: 'HP:0001250',
      label: 'Seizure',
      assertion_status: 'affirmed',
    });
  });

  it('emits hover and clear events for phenotype rows and reflects active state', async () => {
    const wrapper = mountReceipt({
      hoveredPhenotypeId: 'HP:0001250',
    });

    const phenotype = wrapper.get('[data-testid="full-text-response-phenotype"]');
    expect(phenotype.classes()).toContain('full-text-response-phenotype--active');

    await phenotype.trigger('mouseenter');
    await phenotype.trigger('mouseleave');

    expect(wrapper.emitted('hover-phenotype')).toHaveLength(1);
    expect(wrapper.emitted('hover-phenotype')[0]).toEqual(['HP:0001250']);
    expect(wrapper.emitted('clear-hover')).toHaveLength(1);
  });
});
