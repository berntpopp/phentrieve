// postcss.config.js
module.exports = {
  plugins: {
    autoprefixer: {},
    // Conditionally apply PurgeCSS only for production builds
    ...(process.env.NODE_ENV === 'production' ? {
      '@fullhuman/postcss-purgecss': {
        content: [
          './index.html',
          './src/**/*.vue',
          './src/**/*.js',
          './src/**/*.ts',
        ],
        safelist: {
          standard: [
            /v-((?!application).)*$/, // General Vuetify components
            /v-application.*/, // Vuetify application classes
            /density-.*/, // Density classes
            /col-.*/, // Vuetify grid col classes
            /text-.*/, // Vuetify text alignment/color classes
            /bg-.*/, // Vuetify background color classes
            /d-.*/, // Vuetify display classes (d-flex, etc.)
            /justify-.*/, /align-.*/, // Flex alignment
            /ma-\d+/, /mt-\d+/, /mr-\d+/, /mb-\d+/, /ml-\d+/, /mx-\d+/, /my-\d+/, 
            /pa-\d+/, /pt-\d+/, /pr-\d+/, /pb-\d+/, /pl-\d+/, /px-\d+/, /py-\d+/, // Vuetify spacing
            /theme--.*/, // Vuetify theme classes
            /primary.*/, /secondary.*/, /accent.*/, // Theme colors
            /error.*/, /info.*/, /success.*/, /warning.*/,
            /rounded.*/, // Border radius utilities
            /elevation.*/, // Elevation utilities
            /text-subtitle.*/, // Typography classes
            /text-caption.*/, // Typography classes
            /text-body.*/, // Typography classes
            /text-grey.*/, // Color utilities
            /font-weight.*/, // Font weight utilities
            /-(leave|enter|appear)(|-(to|from|active))$/, // Vue transitions
            /^router-link(|-exact)-active$/,
          ],
          deep: [
            /v-input[^$]*/, // All v-input variations
            /v-field[^$]*/, // All v-field variations
            /v-label[^$]*/, // All v-label variations
            /v-btn[^$]*/, // All v-btn variations
            /v-messages[^$]*/, // All v-messages variations
            /v-application[^$]*/, // All v-application variations
            /v-card[^$]*/, /v-list[^$]*/, /v-container[^$]*/,
            /v-row[^$]*/, /v-col[^$]*/, /v-icon[^$]*/, /v-badge[^$]*/, /v-alert[^$]*/, /v-chip[^$]*/,
            /v-textarea[^$]*/, // All v-textarea variations
            /v-text-field[^$]*/, // All v-text-field variations
            /v-field-label[^$]*/, // All v-field-label variations
            /v-input-control[^$]*/, // All v-input-control variations
            /v-sheet[^$]*/, /v-overlay[^$]*/, /v-selection-control[^$]*/,
            /v-field--variant-.*/, // Field variants
            /v-input--density-.*/, // Input density
            /v-field--density-.*/, // Field density
            /v-input--variant-.*/, // Input variants
            /v-field--focused/, // Field states
            /v-input--focused/, // Input states
            /v-field--active/, // Field states
            /v-input--active/, // Input states
            /v-field--dirty/, // Field states
            /v-input--dirty/ // Input states
          ],
          greedy: [
            /v-ripple/, /transition-.*/, /elevation-.*/, 
            /scroll-.*/, /data-v-.*/, /v-theme--.*/, /v-layout.*/,
            /--v-theme-overlay-multiplier/, // Theme overlay
            /--v-border-color/, // Border colors
            /--v-border-opacity/, // Border opacity
            /--v-theme-on-surface/, // Theme colors
            /--v-theme-surface/, // Theme colors
            /--v-activated-opacity/, // Opacity variants
            /--v-disabled-opacity/, // Opacity variants
            /--v-field-border-width/, // Field properties
            /--v-field-border-opacity/, // Field properties
            /--v-field-surface-opacity/ // Field properties
          ],
        },
        defaultExtractor: content => {
          // Capture class names using a regex that handles various characters
          const broadMatches = content.match(/[\w-/:]+(?<!:)/g) || [];
          // Capture Vuetify specific classes that might be missed
          const vuetifySpecific = content.match(/v-[\w-]+--[\w-]+/g) || [];
          return broadMatches.concat(vuetifySpecific);
        }
      }
    } : {})
  }
}
