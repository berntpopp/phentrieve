import { describe, it, expect, vi, afterEach } from 'vitest';
import { mount } from '@vue/test-utils';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';
import TutorialOverlay from '../../components/TutorialOverlay.vue';
import { tutorialService } from '../../services/tutorialService';

const vuetify = createVuetify({ components, directives });
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } });

afterEach(() => vi.restoreAllMocks());

describe('TutorialOverlay', () => {
  it('skips the tutorial on Escape while visible (#466)', async () => {
    const skip = vi.spyOn(tutorialService, 'skip').mockImplementation(() => {});
    const wrapper = mount(TutorialOverlay, { global: { plugins: [vuetify, i18n] } });

    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(skip).not.toHaveBeenCalled();

    await wrapper.setProps({ visible: true });
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(skip).toHaveBeenCalledTimes(1);

    wrapper.unmount();
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(skip).toHaveBeenCalledTimes(1);
  });
});
