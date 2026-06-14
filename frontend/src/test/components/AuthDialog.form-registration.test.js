/**
 * Regression test for the tree-shaken Vuetify registration in
 * `src/plugins/vuetify.js`.
 *
 * The app registers Vuetify components explicitly (for bundle size). When
 * `VForm` was omitted from that list, `<v-form>` rendered as an inert custom
 * element: its `@submit` never fired, so clicking "Create account" / "Sign in"
 * did nothing and registration was impossible from the UI.
 *
 * Unlike AuthDialog.test.js (which registers ALL components via
 * `import * as components`), this test mounts with the REAL app plugin so a
 * missing registration is actually caught.
 */
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { createI18n } from 'vue-i18n';
import { createPinia } from 'pinia';
import en from '../../locales/en.json';
import vuetify from '../../plugins/vuetify';
import AuthDialog from '../../components/auth/AuthDialog.vue';

vi.mock('../../services/AuthService', () => ({
  default: {
    login: vi.fn(),
    register: vi.fn(),
    requestPasswordReset: vi.fn(),
  },
}));
vi.mock('../../services/apiClient', () => ({ default: { get: vi.fn() }, readCookie: vi.fn() }));

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en }, missing: () => '' });

function mountDialog() {
  return mount(AuthDialog, {
    attachTo: document.body,
    props: { modelValue: true, initialMode: 'register' },
    global: { plugins: [vuetify, i18n, createPinia()] },
  });
}

describe('AuthDialog Vuetify registration (real app plugin)', () => {
  let warnSpy;

  beforeEach(() => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    warnSpy.mockRestore();
    document.body.innerHTML = '';
  });

  it('renders a native <form> element (VForm is registered)', () => {
    mountDialog();
    // If VForm is unregistered, <v-form> stays an inert custom element and no
    // native <form> exists, so the submit button never triggers submit().
    expect(document.querySelector('form')).not.toBeNull();
  });

  it('mounts without "Failed to resolve component" warnings', () => {
    mountDialog();
    const unresolved = warnSpy.mock.calls
      .map(([m]) => String(m))
      .filter((m) => m.includes('Failed to resolve component'));
    expect(unresolved).toEqual([]);
  });
});
