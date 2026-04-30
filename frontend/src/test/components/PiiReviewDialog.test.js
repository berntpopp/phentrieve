import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { nextTick } from 'vue';
import { createI18n } from 'vue-i18n';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import en from '../../locales/en.json';
import PiiReviewDialog from '../../components/PiiReviewDialog.vue';

const vuetify = createVuetify({ components, directives });
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } });

Object.defineProperty(window, 'visualViewport', {
  configurable: true,
  value: {
    width: 1024,
    height: 768,
    offsetLeft: 0,
    offsetTop: 0,
    pageLeft: 0,
    pageTop: 0,
    scale: 1,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  },
});

function mountDialog(props = {}) {
  return mount(PiiReviewDialog, {
    props: {
      modelValue: true,
      summary: { high: { email: 1, medical_record: 1 }, review: { date: 1 } },
      ...props,
    },
    global: { plugins: [vuetify, i18n] },
    attachTo: document.body,
  });
}

afterEach(() => {
  document.body.innerHTML = '';
});

describe('PiiReviewDialog', () => {
  it('shows category counts without raw snippets', async () => {
    const wrapper = mountDialog();
    await nextTick();

    expect(document.body.textContent).toContain('Email');
    expect(document.body.textContent).toContain('Medical record');
    expect(document.body.textContent).toContain('Date');
    expect(document.body.textContent).not.toContain('jane@example.org');

    wrapper.unmount();
  });

  it('emits cancel, redact, and continue actions', async () => {
    const wrapper = mountDialog();
    await nextTick();

    document.querySelector('[data-testid="pii-cancel"]').click();
    document.querySelector('[data-testid="pii-redact"]').click();
    document.querySelector('[data-testid="pii-continue"]').click();
    await nextTick();

    expect(wrapper.emitted('cancel')).toHaveLength(1);
    expect(wrapper.emitted('redact')).toHaveLength(1);
    expect(wrapper.emitted('continue')).toHaveLength(1);

    wrapper.unmount();
  });
});
