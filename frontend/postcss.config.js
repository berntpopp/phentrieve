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
            /v-((?!application).)*$/, // General Vuetify components, but not v-application
            /col-.*/, // Vuetify grid col classes
            /text-.*/, // Vuetify text alignment/color classes
            /bg-.*/, // Vuetify background color classes
            /d-.*/, // Vuetify display classes (d-flex, etc.)
            /justify-.*/, /align-.*/, // Flex alignment
            /ma-\d+/, /mt-\d+/, /mr-\d+/, /mb-\d+/, /ml-\d+/, /mx-\d+/, /my-\d+/, 
            /pa-\d+/, /pt-\d+/, /pr-\d+/, /pb-\d+/, /pl-\d+/, /px-\d+/, /py-\d+/, // Vuetify spacing
            /theme--.*/, // Vuetify theme classes
            /primary$/, /secondary$/, /accent$/, // Theme colors
            /error$/, /info$/, /success$/, /warning$/,
            /-(leave|enter|appear)(|-(to|from|active))$/, // Vue transitions
            /^router-link(|-exact)-active$/,
          ],
          deep: [
            /v-input__.*/, /v-field__.*/, /v-label/, /v-btn__.*/, 
            /v-messages/, /v-application/
          ],
          greedy: [
            /v-ripple/, /transition-.*/, /elevation-.*/, 
            /scroll-.*/, /data-v-.*/
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
