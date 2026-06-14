import { describe, it, expect, afterEach, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import { createPinia } from 'pinia';
import en from '../../locales/en.json';
import AuthDialog from '../../components/auth/AuthDialog.vue';

vi.mock('../../services/AuthService', () => ({
  default: {
    login: vi.fn(),
    register: vi.fn(),
    requestPasswordReset: vi.fn(),
  },
}));
vi.mock('../../services/apiClient', () => ({ default: { get: vi.fn() } }));

const vuetify = createVuetify({ components, directives });
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en }, missing: () => '' });

function mountDialog() {
  return mount(AuthDialog, {
    attachTo: document.body,
    props: { modelValue: true, initialMode: 'login' },
    global: { plugins: [vuetify, i18n, createPinia()] },
  });
}

afterEach(() => {
  document.body.innerHTML = '';
});

describe('AuthDialog', () => {
  it('renders email and password fields in login mode', () => {
    mountDialog();
    expect(document.body.textContent).toContain('Sign in');
    expect(document.body.textContent).toContain('Email');
    expect(document.body.textContent).toContain('Password');
  });

  it('shows the confirm-password field in register mode', async () => {
    const wrapper = mountDialog();
    wrapper.vm.setMode('register');
    await wrapper.vm.$nextTick();
    expect(document.body.textContent).toContain('Confirm password');
    expect(document.body.textContent).toContain('Create account');
  });

  it('blocks submit and does not call login when email is empty', async () => {
    const AuthService = (await import('../../services/AuthService')).default;
    const wrapper = mountDialog();
    await wrapper.vm.submit();
    await wrapper.vm.$nextTick();
    expect(AuthService.login).not.toHaveBeenCalled();
  });
});
