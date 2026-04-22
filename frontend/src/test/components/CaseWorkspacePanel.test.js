import { describe, it, expect } from 'vitest';
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

const englishBridgeMessages = {
  'queryInterface.phenotypeCollection.title': 'HPO Collection',
  'queryInterface.phenotypeCollection.close': 'Close HPO Collection Panel',
  'queryInterface.phenotypeCollection.aria.openPanel': 'Open HPO Collection Panel',
  'queryInterface.phenotypeCollection.aria.panel': 'Phenotype collection',
};

const germanBridgeMessages = {
  'queryInterface.phenotypeCollection.title': 'HPO-Sammlung',
  'queryInterface.phenotypeCollection.close': 'HPO-Sammlungsbereich schließen',
  'queryInterface.phenotypeCollection.aria.openPanel': 'HPO-Sammlung öffnen',
  'queryInterface.phenotypeCollection.aria.panel': 'Phänotyp-Sammlung',
};

async function mountLegacyBridge({ messages = englishBridgeMessages, ...props } = {}) {
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
        $t: (key) => messages[key] ?? key,
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

  it('keeps the temporary rename bridge consistent in the legacy panel UI', async () => {
    const wrapper = await mountLegacyBridge();

    expect(wrapper.get('[data-testid="case-workspace-title"]').text()).toBe('Case Workspace');
    expect(wrapper.text()).not.toContain('HPO Collection');
    expect(wrapper.get('[data-testid="case-workspace-open-button"]').attributes('aria-label')).toBe(
      'Open Case Workspace Panel'
    );
    expect(wrapper.get('[data-testid="case-workspace-close-button"]').attributes('aria-label')).toBe(
      'Close Case Workspace Panel'
    );
    expect(wrapper.get('[data-testid="case-workspace-drawer"]').attributes('aria-label')).toBe(
      'Case Workspace Panel'
    );
  });

  it('bridges the shipped German locale strings to case workspace wording', async () => {
    const wrapper = await mountLegacyBridge({ messages: germanBridgeMessages });

    expect(wrapper.get('[data-testid="case-workspace-title"]').text()).toBe('Fallarbeitsbereich');
    expect(wrapper.text()).not.toContain('HPO-Sammlung');
    expect(wrapper.get('[data-testid="case-workspace-open-button"]').attributes('aria-label')).toBe(
      'Fallarbeitsbereich öffnen'
    );
    expect(wrapper.get('[data-testid="case-workspace-close-button"]').attributes('aria-label')).toBe(
      'Fallarbeitsbereich schließen'
    );
    expect(wrapper.get('[data-testid="case-workspace-drawer"]').attributes('aria-label')).toBe(
      'Fallarbeitsbereich'
    );
  });
});
