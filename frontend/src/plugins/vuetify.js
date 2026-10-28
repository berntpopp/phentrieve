/**
 * Vuetify Plugin Configuration
 *
 * Tree-shaking optimization: Import only used components instead of entire library
 * Following SOLID principles: Single Responsibility (one plugin, one purpose)
 * Following DRY: Single source of truth for Vuetify configuration
 *
 * Impact: Reduces bundle size by ~558 KiB (unused components eliminated)
 */

import { h } from 'vue';
import { createVuetify } from 'vuetify';
import { aliases, mdi as mdiSvg } from 'vuetify/iconsets/mdi-svg';
import { icons } from './icons';

// Import only components actually used in the application
// Alphabetically sorted for maintainability (KISS principle)
import {
  VAlert,
  VApp,
  VAvatar,
  VBadge,
  VBtn,
  VCard,
  VCardActions,
  VCardText,
  VCardTitle,
  VChip,
  VCol,
  VContainer,
  VDialog,
  VDivider,
  VExpandTransition,
  VExpansionPanel,
  VExpansionPanels,
  VExpansionPanelText,
  VExpansionPanelTitle,
  VFooter,
  VForm,
  VIcon,
  VList,
  VListItem,
  VListItemSubtitle,
  VListItemTitle,
  VListSubheader,
  VMain,
  VMenu,
  VNavigationDrawer,
  VProgressCircular,
  VProgressLinear,
  VRow,
  VSelect,
  VSheet,
  VSlider,
  VSnackbar,
  VSpacer,
  VSwitch,
  VTextField,
  VTextarea,
  VToolbar,
  VToolbarTitle,
  VTooltip,
} from 'vuetify/components';

// Import only used directives
import { Ripple, Tooltip } from 'vuetify/directives';

export const THEME_STORAGE_KEY = 'phentrieve-theme';

/** Saved theme choice ('light' | 'dark' | 'system'); 'system' follows prefers-color-scheme. */
export function readStoredTheme() {
  try {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    return ['light', 'dark', 'system'].includes(saved) ? saved : 'system';
  } catch {
    return 'system';
  }
}

/**
 * Create and export Vuetify instance
 * Modular approach: Easy to extend without modifying this file (Open/Closed principle)
 */
export default createVuetify({
  // Explicitly register only used components (tree-shaking friendly)
  components: {
    VAlert,
    VApp,
    VAvatar,
    VBadge,
    VBtn,
    VCard,
    VCardActions,
    VCardText,
    VCardTitle,
    VChip,
    VCol,
    VContainer,
    VDialog,
    VDivider,
    VExpandTransition,
    VExpansionPanel,
    VExpansionPanels,
    VExpansionPanelText,
    VExpansionPanelTitle,
    VFooter,
    VForm,
    VIcon,
    VList,
    VListItem,
    VListItemSubtitle,
    VListItemTitle,
    VListSubheader,
    VMain,
    VMenu,
    VNavigationDrawer,
    VProgressCircular,
    VProgressLinear,
    VRow,
    VSelect,
    VSheet,
    VSlider,
    VSnackbar,
    VSpacer,
    VSwitch,
    VTextField,
    VTextarea,
    VToolbar,
    VToolbarTitle,
    VTooltip,
  },

  directives: {
    Ripple,
    Tooltip,
  },

  // Inline SVG icons: templates keep using `mdi-*` names, mapped to @mdi/js paths
  icons: {
    defaultSet: 'mdi',
    aliases,
    sets: {
      mdi: {
        component: (props) =>
          h(mdiSvg.component, { ...props, icon: icons[props.icon] ?? props.icon }),
      },
    },
  },

  theme: {
    defaultTheme: readStoredTheme(),
    themes: {
      // Darker status colors so white chip text meets WCAG AA (>= 4.5:1)
      light: {
        colors: {
          background: '#F5F5F5',
          success: '#2E7D32',
          info: '#1565C0',
          warning: '#B45309',
        },
      },
      // Lighter status colors so they read as text on dark surfaces (AA); on-* resolves to dark
      dark: {
        colors: { success: '#81C784', info: '#64B5F6', warning: '#FFB74D', error: '#EF9A9A' },
      },
    },
  },
});
