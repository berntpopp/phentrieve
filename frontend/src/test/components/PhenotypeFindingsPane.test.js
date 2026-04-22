import { describe, it, expect, afterEach, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';
import { SCORE_EXCELLENT, SCORE_GOOD, SCORE_MODERATE, SCORE_LOW } from '../../constants/defaults';

const vuetify = createVuetify({ components, directives });
const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: { en },
  silentTranslationWarn: true,
  silentFallbackWarn: true,
  missing: () => '',
});

function findingsTerm(overrides = {}) {
  return {
    hpo_id: 'HP:0001250',
    name: 'Seizure',
    confidence: 0.92,
    status: 'affirmed',
    source_chunk_ids: [1],
    top_evidence_chunk_id: 1,
    ...overrides,
  };
}

function mountFindingsPane(terms = [findingsTerm()]) {
  return import('../../components/PhenotypeFindingsPane.vue').then(({ default: component }) =>
    mount(component, {
      props: { terms },
      global: {
        plugins: [vuetify, i18n],
      },
    })
  );
}

function mountInspector(selectedTerm = findingsTerm()) {
  return import('../../components/AnnotationInspectorPanel.vue').then(({ default: component }) =>
    mount(component, {
      props: { selectedTerm },
      global: {
        plugins: [vuetify, i18n],
      },
    })
  );
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe('PhenotypeFindingsPane', () => {
  it('uses one bulk add action instead of per-row add and inspect buttons', async () => {
    const wrapper = await mountFindingsPane();

    expect(wrapper.find('[data-testid="findings-add-all"]').exists()).toBe(true);
    expect(wrapper.find('[aria-label="Add HP:0001250"]').exists()).toBe(false);
    expect(wrapper.find('[aria-label="Inspect HP:0001250"]').exists()).toBe(false);
  });

  it('emits all normalized findings for collection from the bulk add action', async () => {
    const wrapper = await mountFindingsPane([
      findingsTerm(),
      findingsTerm({
        hpo_id: 'HP:0001263',
        name: 'Global developmental delay',
        status: 'negated',
      }),
    ]);

    await wrapper.get('[data-testid="findings-add-all"]').trigger('click');

    expect(wrapper.emitted('add-all-to-collection')).toEqual([
      [
        [
          { hpo_id: 'HP:0001250', label: 'Seizure', assertion_status: 'affirmed' },
          {
            hpo_id: 'HP:0001263',
            label: 'Global developmental delay',
            assertion_status: 'negated',
          },
        ],
      ],
    ]);
  });

  it('renders query-style confidence chips together with the findings metadata', async () => {
    const wrapper = await mountFindingsPane();

    const link = wrapper.get('a');

    expect(link.text()).toContain('HP:0001250');
    expect(wrapper.text()).toContain('0.92');
    expect(wrapper.text()).toContain(
      en.queryInterface.phenotypeCollection.assertionStatus.affirmed
    );
    expect(wrapper.text()).toContain('Seizure');
    expect(wrapper.text()).toContain(`${en.resultsDisplay.textProcess.sourceChunks}: 1`);
    expect(wrapper.text()).toContain(`${en.resultsDisplay.textProcess.topEvidence}: #1`);
  });

  it('emits an inspect payload that is directly compatible with the inspector term shape', async () => {
    const term = findingsTerm({
      hpo_id: 'HP:0004322',
      name: 'Spasticity',
      confidence: 0.88,
      status: 'negated',
      source_chunk_ids: [2, 7],
      top_evidence_chunk_id: 7,
    });
    const wrapper = await mountFindingsPane([term]);

    const termItem = wrapper.get('.v-list-item');

    await termItem.trigger('mouseenter');
    await termItem.trigger('mouseleave');
    await termItem.trigger('click');

    expect(wrapper.emitted('hover-term')).toEqual([
      [
        {
          hpoId: 'HP:0004322',
        },
      ],
    ]);
    expect(wrapper.emitted('clear-hover')).toEqual([[]]);
    expect(wrapper.emitted('inspect-term')).toEqual([
      [
        {
          hpo_id: 'HP:0004322',
          name: 'Spasticity',
          confidence: 0.88,
          status: 'negated',
          source_chunk_ids: [2, 7],
          top_evidence_chunk_id: 7,
        },
      ],
    ]);
  });

  it('renders per-row summaries from the matching term data', async () => {
    const wrapper = await mountFindingsPane([
      findingsTerm(),
      findingsTerm({
        hpo_id: 'HP:0001627',
        name: 'Abnormal heart morphology',
        confidence: 0.62,
        status: 'negated',
        source_chunk_ids: [4, 5, 9],
        top_evidence_chunk_id: 9,
      }),
    ]);

    expect(wrapper.text()).toContain('0.62');
    expect(wrapper.text()).toContain(en.queryInterface.phenotypeCollection.assertionStatus.negated);
    expect(wrapper.text()).toContain(`${en.resultsDisplay.textProcess.sourceChunks}: 3`);
    expect(wrapper.text()).toContain(`${en.resultsDisplay.textProcess.topEvidence}: #9`);
  });

  it('maps present assertions onto the shared affirmed status label without emitting missing-key lookups', async () => {
    const wrapper = await mountFindingsPane([
      findingsTerm({
        hpo_id: 'HP:0001507',
        name: 'Growth abnormality',
        status: 'present',
      }),
    ]);

    expect(wrapper.text()).toContain(
      en.queryInterface.phenotypeCollection.assertionStatus.affirmed
    );
  });

  it('preserves null confidence in the inspect payload so the inspector does not render 0.00', async () => {
    const wrapper = await mountFindingsPane([
      findingsTerm({
        hpo_id: 'HP:0004322',
        name: 'Spasticity',
        confidence: null,
        source_chunk_ids: [4],
        top_evidence_chunk_id: 4,
      }),
    ]);

    await wrapper.get('.v-list-item').trigger('click');

    const inspectPayload = wrapper.emitted('inspect-term')?.[0]?.[0];
    const inspector = await mountInspector(inspectPayload);

    expect(inspectPayload).toMatchObject({
      hpo_id: 'HP:0004322',
      name: 'Spasticity',
      confidence: null,
      source_chunk_ids: [4],
      top_evidence_chunk_id: 4,
    });
    expect(inspector.text()).toContain('Spasticity');
    expect(inspector.text()).not.toContain(`${en.resultsDisplay.confidenceHeader}: 0.00`);
  });

  it('uses the shared frontend score component and suppresses empty confidence chips', async () => {
    const wrapper = await mountFindingsPane([
      findingsTerm({ hpo_id: 'HP:1', name: 'Excellent edge', confidence: SCORE_EXCELLENT }),
      findingsTerm({ hpo_id: 'HP:2', name: 'High edge', confidence: SCORE_GOOD }),
      findingsTerm({ hpo_id: 'HP:3', name: 'Moderate edge', confidence: SCORE_MODERATE }),
      findingsTerm({ hpo_id: 'HP:4', name: 'Low edge', confidence: SCORE_LOW }),
      findingsTerm({ hpo_id: 'HP:5', name: 'Poor fallback', confidence: SCORE_LOW - 0.01 }),
      findingsTerm({ hpo_id: 'HP:6', name: 'Unknown fallback', confidence: null }),
    ]);

    const itemTexts = wrapper.findAll('.v-list-item').map((item) => item.text());

    expect(itemTexts[0]).toContain(SCORE_EXCELLENT.toFixed(2));
    expect(itemTexts[1]).toContain(SCORE_GOOD.toFixed(2));
    expect(itemTexts[2]).toContain(SCORE_MODERATE.toFixed(2));
    expect(itemTexts[3]).toContain(SCORE_LOW.toFixed(2));
    expect(itemTexts[4]).toContain((SCORE_LOW - 0.01).toFixed(2));
    expect(itemTexts[5]).not.toContain('0.00');
  });
});

describe('AnnotationInspectorPanel', () => {
  describe('rendering', () => {
    it('renders localized inspector labels and numeric confidence details for the selected term', async () => {
      const wrapper = await mountInspector();

      expect(wrapper.text()).toContain(en.resultsDisplay.showDetails);
      expect(wrapper.text()).toContain(`${en.resultsDisplay.confidenceHeader}: 0.92`);
      expect(wrapper.get('[id="annotation-detail-HP:0001250"]').text()).toContain('Seizure');
    });
  });

  describe('navigation', () => {
    it('emits back from the inspector return action without depending on one specific term row', async () => {
      const wrapper = await mountInspector(
        findingsTerm({
          hpo_id: 'HP:0001627',
          name: 'Abnormal heart morphology',
          confidence: 0.61,
        })
      );

      expect(wrapper.get('button').attributes('aria-label')).toBe(en.common.close);
      await wrapper.get('button').trigger('click');

      expect(wrapper.emitted('back')).toEqual([[]]);
    });
  });

  describe('selectedTerm hardening', () => {
    it('suppresses details when selectedTerm does not match the expected contract', async () => {
      const wrapper = await mountInspector({
        hpoId: 'HP:0004322',
        name: 'Spasticity',
        confidence: 0.88,
      });

      expect(wrapper.text()).toContain(en.resultsDisplay.showDetails);
      expect(wrapper.text()).not.toContain('Spasticity');
      expect(wrapper.text()).not.toContain(`${en.resultsDisplay.confidenceHeader}: 0.88`);
    });

    it('renders the selected term label even when numeric confidence is unavailable', async () => {
      const wrapper = await mountInspector(
        findingsTerm({
          hpo_id: 'HP:0004322',
          name: 'Spasticity',
          confidence: null,
        })
      );

      expect(wrapper.text()).toContain('Spasticity');
      expect(wrapper.text()).toContain(`${en.resultsDisplay.confidenceHeader}:`);
    });
  });
});
