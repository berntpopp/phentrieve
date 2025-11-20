import js from '@eslint/js';
import pluginVue from 'eslint-plugin-vue';
import globals from 'globals';

export default [
  // Base recommended configs
  js.configs.recommended,
  ...pluginVue.configs['flat/recommended'],

  // Global ignores
  {
    ignores: ['dist/**', 'node_modules/**', 'build/**', '.vite/**'],
  },

  // Main configuration
  {
    files: ['**/*.{js,mjs,cjs,vue}'],
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: 'module',
      globals: {
        ...globals.browser,
        ...globals.node,
        ...globals.es2021,
      },
    },
    rules: {
      // Vue-specific rules
      'vue/require-default-prop': 'off',
      'vue/multi-word-component-names': 'off',
      'vue/valid-v-slot': 'off',
      'vue/no-v-html': 'off',

      // Disable Vue formatting rules - let Prettier handle formatting
      'vue/max-attributes-per-line': 'off',
      'vue/singleline-html-element-content-newline': 'off',
      'vue/html-indent': 'off',
      'vue/html-closing-bracket-newline': 'off',
      'vue/html-self-closing': 'off', // Let Prettier decide on self-closing tags

      // General rules
      'no-console': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
      'no-debugger': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
      'no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
    },
  },

  // Vue files specific configuration
  {
    files: ['**/*.vue'],
    languageOptions: {
      parserOptions: {
        ecmaVersion: 2020,
        sourceType: 'module',
      },
    },
  },
];
