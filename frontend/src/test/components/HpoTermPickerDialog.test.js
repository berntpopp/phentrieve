import { describe, it, expect, afterEach } from 'vitest';
import { mount } from '@vue/test-utils';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';
import HpoTermPickerDialog from '../../components/HpoTermPickerDialog.vue';

const vuetify = createVuetify({ components, directives });
const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: { en },
  missing: () => '',
});

function mountDialog(props = {}) {
  return mount(HpoTermPickerDialog, {
    attachTo: document.body,
    props: {
      modelValue: true,
      mode: 'replace',
      spanText: 'normal birth weight',
      candidates: [],
      loading: false,
      assertion: 'affirmed',
      ...props,
    },
    global: { plugins: [vuetify, i18n] },
  });
}

afterEach(() => {
  document.body.innerHTML = '';
});

describe('HpoTermPickerDialog', () => {
  it('shows an empty state when there are no candidates and not loading', () => {
    mountDialog({ candidates: [] });
    expect(document.body.textContent).toMatch(/no matching/i);
  });

  it('emits submit with the selected term and assertion', async () => {
    const wrapper = mountDialog({
      candidates: [{ hpo_id: 'HP:0001518', label: 'Small for gestational age', similarity: 0.74 }],
    });
    await wrapper.vm.$nextTick();

    document.querySelector('[data-testid="hpo-candidate"]').click();
    await wrapper.vm.$nextTick();

    document.querySelector('[data-testid="hpo-picker-submit"]').click();
    await wrapper.vm.$nextTick();

    expect(wrapper.emitted('submit')[0][0]).toMatchObject({
      term: { hpo_id: 'HP:0001518' },
      assertion: 'affirmed',
    });
  });

  it('emits requery when the search field changes (debounced)', async () => {
    const wrapper = mountDialog({ candidates: [] });
    await wrapper.vm.$nextTick();

    const input = document.querySelector('[data-testid="hpo-picker-search"] input');
    input.value = 'low birth weight';
    input.dispatchEvent(new Event('input', { bubbles: true }));

    await new Promise((resolve) => setTimeout(resolve, 260));
    expect(wrapper.emitted('requery')).toBeTruthy();
    expect(wrapper.emitted('requery').at(-1)[0]).toBe('low birth weight');
  });
});
