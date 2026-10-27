<template>
  <v-tooltip location="bottom" :text="label" :aria-label="label">
    <template #activator="{ props }">
      <v-btn
        v-bind="props"
        icon
        variant="text"
        size="small"
        :aria-label="label"
        @click="cycleTheme"
      >
        <v-icon>{{ ICONS[choice] }}</v-icon>
      </v-btn>
    </template>
  </v-tooltip>
</template>

<script setup>
import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { useTheme } from 'vuetify';
import { THEME_STORAGE_KEY, readStoredTheme } from '@/plugins/vuetify';

const ORDER = ['system', 'light', 'dark'];
const ICONS = {
  system: 'mdi-theme-light-dark',
  light: 'mdi-weather-sunny',
  dark: 'mdi-weather-night',
};

const { t } = useI18n();
const theme = useTheme();
const choice = ref(readStoredTheme());
const label = computed(() => t(`app.footer.theme.${choice.value}`));

function cycleTheme() {
  choice.value = ORDER[(ORDER.indexOf(choice.value) + 1) % ORDER.length];
  theme.change(choice.value);
  try {
    localStorage.setItem(THEME_STORAGE_KEY, choice.value);
  } catch {
    // Storage blocked (private mode): the choice still applies for this session
  }
}
</script>
