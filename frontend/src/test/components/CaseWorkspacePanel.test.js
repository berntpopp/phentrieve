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

    const buttons = wrapper.findAll('button');
    await buttons[0].trigger('click');
    await buttons[buttons.length - 2].trigger('click');
    await buttons[buttons.length - 1].trigger('click');

    expect(wrapper.emitted('create-case')).toHaveLength(1);
    expect(wrapper.emitted('add-all')).toHaveLength(1);
    expect(wrapper.emitted('export-case')).toHaveLength(1);
  });

  it('emits select-case when a case is clicked', async () => {
    const wrapper = await mountPanel();

    const listItems = wrapper.findAll('.v-list-item');
    await listItems[1].trigger('click');

    expect(wrapper.emitted('select-case')).toEqual([['case-2']]);
  });
});
