import { describe, expect, it, vi, beforeEach } from 'vitest';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';
import { useFullTextWorkspaceStore } from '../../stores/fullTextWorkspace';

vi.mock('../../services/PhentrieveService', () => ({
  default: {
    exportPhenopacket: vi.fn().mockResolvedValue({
      phenopacket_json: '{"id":"case-1"}',
      annotation_sidecar: { phenopacket_id: 'case-1', annotations: [] },
    }),
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

describe('FullTextAnnotationWorkspace', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
  });

  it('shows a one-line fallback banner while keeping the workspace layout intact', async () => {
    const component = (await import('../../components/FullTextAnnotationWorkspace.vue')).default;
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-1');
    store.setExpanded('turn-1', true);

    const wrapper = mount(component, {
      props: {
        turnId: 'turn-1',
        responseData: {
          meta: {
            extraction_backend: 'standard',
            fallback_reason: 'llm_quota_exhausted',
            llm_quota_reset_at: '2026-04-23T00:00:00+00:00',
          },
          processed_chunks: [
            { chunk_id: 1, text: 'Patient had recurrent seizures.', status: 'affirmed' },
          ],
          aggregated_hpo_terms: [],
        },
      },
      global: {
        plugins: [vuetify, i18n],
      },
    });

    expect(wrapper.text()).toContain('Richer LLM analysis is unavailable for today');
    expect(wrapper.text()).toContain('LLM access resets');
    expect(wrapper.find('.workspace-layout').exists()).toBe(true);
  });

  it('adds extracted phenotypes into the active case workspace', async () => {
    const component = (await import('../../components/FullTextAnnotationWorkspace.vue')).default;
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-2');
    store.setExpanded('turn-2', true);

    const wrapper = mount(component, {
      props: {
        turnId: 'turn-2',
        responseData: {
          meta: { extraction_backend: 'standard' },
          processed_chunks: [
            { chunk_id: 1, text: 'Patient had recurrent seizures.', status: 'affirmed' },
          ],
          aggregated_hpo_terms: [
            {
              hpo_id: 'HP:0001250',
              name: 'Seizure',
              status: 'affirmed',
              confidence: 0.9,
              source_chunk_ids: [1],
              text_attributions: [
                {
                  chunk_id: 1,
                  start_char: 13,
                  end_char: 22,
                  matched_text_in_chunk: 'seizures',
                },
              ],
            },
          ],
        },
      },
      global: {
        plugins: [vuetify, i18n],
      },
    });

    await wrapper.find('[data-testid="create-case-button"]').trigger('click');
    await wrapper.find('[data-testid="add-all-button"]').trigger('click');

    const activeCase = store.getActiveCase('turn-2');
    expect(activeCase).toBeTruthy();
    expect(activeCase.phenotypes).toHaveLength(1);
    expect(activeCase.phenotypes[0]).toMatchObject({
      hpo_id: 'HP:0001250',
      label: 'Seizure',
      assertion_status: 'affirmed',
    });
  });

  it('emits bulk collection payloads when the integrated case workspace is disabled', async () => {
    const component = (await import('../../components/FullTextAnnotationWorkspace.vue')).default;
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-3');
    store.setExpanded('turn-3', true);

    const wrapper = mount(component, {
      props: {
        turnId: 'turn-3',
        showCaseWorkspace: false,
        responseData: {
          meta: { extraction_backend: 'llm' },
          processed_chunks: [
            { chunk_id: 1, text: 'Patient had recurrent seizures.', status: 'affirmed' },
          ],
          aggregated_hpo_terms: [
            {
              hpo_id: 'HP:0001250',
              name: 'Seizure',
              status: 'affirmed',
              confidence: 0.9,
              source_chunk_ids: [1],
            },
          ],
        },
      },
      global: {
        plugins: [vuetify, i18n],
      },
    });

    await wrapper.get('[data-testid="findings-add-all"]').trigger('click');

    expect(wrapper.emitted('add-all-to-collection')).toEqual([
      [[{ hpo_id: 'HP:0001250', label: 'Seizure', assertion_status: 'affirmed' }]],
    ]);
  });

  it('uses the simplified embedded full-text layout when the case workspace is disabled', async () => {
    const component = (await import('../../components/FullTextAnnotationWorkspace.vue')).default;
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-embedded');
    store.setExpanded('turn-embedded', true);

    const wrapper = mount(component, {
      props: {
        turnId: 'turn-embedded',
        showCaseWorkspace: false,
        responseData: {
          meta: { extraction_backend: 'llm' },
          processed_chunks: [{ chunk_id: 1, text: 'Patient had recurrent seizures.' }],
          aggregated_hpo_terms: [
            {
              hpo_id: 'HP:0001250',
              name: 'Seizure',
              status: 'present',
              confidence: 0.9,
              source_chunk_ids: [1],
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
      },
      global: {
        plugins: [vuetify, i18n],
      },
    });

    expect(wrapper.classes()).toContain('full-text-workspace--embedded');
    expect(wrapper.find('.workspace-layout').exists()).toBe(false);
    expect(wrapper.text()).toContain('Evidence in note');
    expect(wrapper.text()).toContain(
      en.queryInterface.phenotypeCollection.assertionStatus.affirmed
    );
  });

  it('falls back to the submitted note when LLM returns findings without document sections', async () => {
    const component = (await import('../../components/FullTextAnnotationWorkspace.vue')).default;
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-4');
    store.setExpanded('turn-4', true);

    const wrapper = mount(component, {
      props: {
        turnId: 'turn-4',
        showCaseWorkspace: false,
        submittedText: 'Patient had recurrent seizures and developmental delay.',
        responseData: {
          meta: { extraction_backend: 'llm' },
          processed_chunks: [],
          aggregated_hpo_terms: [
            {
              hpo_id: 'HP:0001250',
              name: 'Seizure',
              status: 'affirmed',
              confidence: 0.9,
              source_chunk_ids: [1],
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
      },
      global: {
        plugins: [vuetify, i18n],
      },
    });

    expect(wrapper.text()).toContain('Patient had recurrent seizures and developmental delay.');
    expect(wrapper.find('[data-annotation-id]').exists()).toBe(true);
  });
});
