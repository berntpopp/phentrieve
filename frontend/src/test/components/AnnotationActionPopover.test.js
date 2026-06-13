import { describe, it, expect } from 'vitest';
import { mount } from '@vue/test-utils';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';
import AnnotationActionPopover from '../../components/AnnotationActionPopover.vue';

const vuetify = createVuetify({ components, directives });
const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: { en },
  missing: () => '',
});

// Render the menu content inline so we test the action list logic without
// Vuetify's overlay positioning machinery (visualViewport / scroll strategies).
const VMenuStub = { name: 'VMenu', template: '<div class="v-menu-stub"><slot /></div>' };

function mountPopover(props = {}) {
  return mount(AnnotationActionPopover, {
    props: { visible: true, target: [0, 0], ...props },
    global: { plugins: [vuetify, i18n], stubs: { VMenu: VMenuStub } },
  });
}

describe('AnnotationActionPopover', () => {
  it('annotation mode shows change/remove and emits them on click', async () => {
    const wrapper = mountPopover({ mode: 'annotation' });

    expect(wrapper.find('[data-testid="action-change-term"]').exists()).toBe(true);
    expect(wrapper.find('[data-testid="action-remove-annotation"]').exists()).toBe(true);
    expect(wrapper.find('[data-testid="action-annotate-selection"]').exists()).toBe(false);

    await wrapper.get('[data-testid="action-change-term"]').trigger('click');
    expect(wrapper.emitted('change-term')).toBeTruthy();

    await wrapper.get('[data-testid="action-remove-annotation"]').trigger('click');
    expect(wrapper.emitted('remove-annotation')).toBeTruthy();
  });

  it('selection mode shows a single annotate-selection action', async () => {
    const wrapper = mountPopover({ mode: 'selection' });

    expect(wrapper.find('[data-testid="action-annotate-selection"]').exists()).toBe(true);
    expect(wrapper.find('[data-testid="action-change-term"]').exists()).toBe(false);

    await wrapper.get('[data-testid="action-annotate-selection"]').trigger('click');
    expect(wrapper.emitted('annotate-selection')).toBeTruthy();
  });

  it('shows the revert action only when canRevert is true', async () => {
    const without = mountPopover({ mode: 'annotation', canRevert: false });
    expect(without.find('[data-testid="action-revert"]').exists()).toBe(false);

    const withRevert = mountPopover({ mode: 'annotation', canRevert: true });
    expect(withRevert.find('[data-testid="action-revert"]').exists()).toBe(true);
    await withRevert.get('[data-testid="action-revert"]').trigger('click');
    expect(withRevert.emitted('revert')).toBeTruthy();
  });
});
