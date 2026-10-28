import { describe, it, expect, beforeEach } from 'vitest';
import { mount } from '@vue/test-utils';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';
import ThemeToggle from '../../components/ThemeToggle.vue';
import { THEME_STORAGE_KEY, readStoredTheme } from '../../plugins/vuetify';

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } });

describe('ThemeToggle', () => {
  beforeEach(() => localStorage.clear());

  it('defaults to system and ignores unknown stored values', () => {
    expect(readStoredTheme()).toBe('system');
    localStorage.setItem(THEME_STORAGE_KEY, 'neon');
    expect(readStoredTheme()).toBe('system');
  });

  it('cycles system -> light -> dark -> system and persists the choice', async () => {
    const vuetify = createVuetify({ components, directives, theme: { defaultTheme: 'system' } });
    const wrapper = mount(ThemeToggle, { global: { plugins: [vuetify, i18n] } });
    const btn = wrapper.get('button');
    expect(btn.attributes('aria-label')).toBe(en.app.footer.theme.system);

    await btn.trigger('click');
    expect(vuetify.theme.global.name.value).toBe('light');
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('light');

    await btn.trigger('click');
    expect(vuetify.theme.global.name.value).toBe('dark');
    expect(readStoredTheme()).toBe('dark');
    expect(btn.attributes('aria-label')).toBe(en.app.footer.theme.dark);

    await btn.trigger('click');
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('system');
  });
});
