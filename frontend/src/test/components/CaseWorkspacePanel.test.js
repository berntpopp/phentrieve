import { describe, it, expect, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';

const vuetify = createVuetify({ components, directives });

const cases = [
  {
    id: 'case-1',
    label: 'Case 1',
    phenotypes: [{ hpo_id: 'HP:0001250', label: 'Seizure' }],
  },
  {
    id: 'case-2',
    label: 'Case 2',
    phenotypes: [],
  },
];

async function mountPanel(props = {}) {
  const component = (await import('../../components/CaseWorkspacePanel.vue')).default;
  return mount(component, {
    props: {
      cases,
      activeCaseId: 'case-1',
      ...props,
    },
    global: {
      plugins: [vuetify],
    },
  });
}

async function mountLegacyBridge({ locale = 'en', ...props } = {}) {
  const component = (await import('../../components/PhenotypeCollectionPanel.vue')).default;
  return mount(component, {
    props: {
      phenotypes: [],
      panelOpen: true,
      sexOptions: [],
      ...props,
    },
    global: {
      plugins: [vuetify],
      stubs: {
        'v-navigation-drawer': {
          template:
            '<div class="v-navigation-drawer-stub" :aria-label="$attrs[\'aria-label\']" :data-testid="$attrs[\'data-testid\']"><slot /><slot name="append" /></div>',
        },
      },
      mocks: {
        $i18n: { locale },
        $t: (key) => `legacy-copy:${locale}:${key}`,
      },
    },
  });
}

describe('CaseWorkspacePanel', () => {
  it('renders the case workspace title and case summaries', async () => {
    const wrapper = await mountPanel();

    expect(wrapper.text()).toContain('Case Workspace');
    expect(wrapper.text()).toContain('Case 1');
    expect(wrapper.text()).toContain('1 phenotypes');
    expect(wrapper.text()).toContain('Case 2');
    expect(wrapper.text()).toContain('0 phenotypes');
  });

  it('shows the active case summary and included phenotype preview', async () => {
    const wrapper = await mountPanel();

    expect(wrapper.text()).toContain('Active case');
    expect(wrapper.text()).toContain('1 phenotype selected');
    expect(wrapper.text()).toContain('Included phenotypes');
    expect(wrapper.text()).toContain('Seizure');
  });

  it('emits create-case, add-all, and export-case actions', async () => {
    const wrapper = await mountPanel();

    await wrapper.get('[data-testid="create-case-button"]').trigger('click');
    await wrapper.get('[data-testid="add-all-button"]').trigger('click');
    await wrapper.get('[data-testid="export-case-button"]').trigger('click');

    expect(wrapper.emitted('create-case')).toHaveLength(1);
    expect(wrapper.emitted('add-all')).toHaveLength(1);
    expect(wrapper.emitted('export-case')).toHaveLength(1);
  });

  it('emits select-case when a case is clicked', async () => {
    const wrapper = await mountPanel();

    await wrapper.get('[data-testid="case-item-case-2"]').trigger('click');

    expect(wrapper.emitted('select-case')).toEqual([['case-2']]);
  });

  it('only clears the collection on a second click', async () => {
    const wrapper = await mountLegacyBridge({
      phenotypes: [{ hpo_id: 'HP:0001250', label: 'Seizure', assertion_status: 'affirmed' }],
    });
    const clear = wrapper.get('[data-testid="collection-clear"]');

    await clear.trigger('click');
    expect(wrapper.emitted('clear')).toBeUndefined();
    expect(clear.text()).toContain('clearConfirm');

    await clear.trigger('click');
    expect(wrapper.emitted('clear')).toHaveLength(1);
  });

  it('copies only present HPO IDs', async () => {
    const writeText = vi.fn().mockResolvedValue();
    vi.stubGlobal('navigator', { clipboard: { writeText } });
    const wrapper = await mountLegacyBridge({
      phenotypes: [
        { hpo_id: 'HP:0001250', label: 'Seizure', assertion_status: 'affirmed' },
        { hpo_id: 'HP:0001263', label: 'Developmental delay', assertion_status: 'negated' },
        { hpo_id: 'HP:0004322', label: 'Short stature' },
      ],
    });

    await wrapper.get('[data-testid="collection-copy-ids"]').trigger('click');

    expect(writeText).toHaveBeenCalledWith('HP:0001250\nHP:0004322');
    vi.unstubAllGlobals();
  });

  it('takes the workspace labels from the locale files', async () => {
    const wrapper = await mountLegacyBridge({ locale: 'de' });
    const key = (k) => `legacy-copy:de:queryInterface.phenotypeCollection.${k}`;

    expect(wrapper.get('[data-testid="case-workspace-title"]').text()).toBe(key('title'));
    expect(wrapper.get('[data-testid="case-workspace-open-button"]').attributes('aria-label')).toBe(
      key('aria.openPanel')
    );
    expect(
      wrapper.get('[data-testid="case-workspace-close-button"]').attributes('aria-label')
    ).toBe(key('close'));
    expect(wrapper.get('[data-testid="case-workspace-drawer"]').attributes('aria-label')).toBe(
      key('aria.panel')
    );
  });
});
