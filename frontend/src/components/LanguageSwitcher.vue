<template>
  <div class="language-selector-container">
    <v-menu location="bottom">
      <template #activator="{ props }">
        <v-btn
          variant="tonal"
          density="comfortable"
          v-bind="props"
          aria-label="Select language"
          color="primary"
          class="language-btn"
        >
          <v-icon :icon="currentLanguageIcon" size="small" :color="currentLanguageColor" />
          <v-icon size="x-small" class="ml-1"> mdi-chevron-down </v-icon>
        </v-btn>
      </template>
      <v-list density="compact">
        <v-list-item
          v-for="localeOption in availableLocales"
          :key="localeOption.code"
          :value="localeOption.code"
          :active="currentLocale === localeOption.code"
          @click="currentLocale = localeOption.code"
        >
          <template #prepend>
            <v-icon
              :icon="localeOption.icon"
              size="small"
              class="mr-2"
              :color="localeOption.color"
            />
          </template>
          <v-list-item-title>{{ localeOption.name }}</v-list-item-title>
        </v-list-item>
      </v-list>
    </v-menu>
  </div>
</template>

<script setup>
import { ref, watch, computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { logService } from '@/services/logService'; // Assuming logService is available

const { locale } = useI18n();

const availableLocales = ref([
  { code: 'en', name: 'English', icon: 'mdi-alpha-e-box', color: 'blue' },
  { code: 'de', name: 'Deutsch', icon: 'mdi-alpha-d-box', color: 'red' },
  { code: 'fr', name: 'Français', icon: 'mdi-alpha-f-box', color: 'blue-darken-1' },
  { code: 'es', name: 'Español', icon: 'mdi-alpha-s-box', color: 'yellow-darken-3' },
  { code: 'nl', name: 'Nederlands', icon: 'mdi-alpha-n-box', color: 'orange' },
]);

const currentLocale = ref(locale.value); // Initialize with current i18n locale

watch(currentLocale, (newLocale) => {
  logService.info('Language changed by user', { newLocale });
  locale.value = newLocale;
  try {
    localStorage.setItem('phentrieve-lang', newLocale);
  } catch (e) {
    logService.warn('Could not save language preference to localStorage.', e);
  }
});

// Watch for external changes to i18n locale (e.g., on initial load)
watch(locale, (newGlobalLocale) => {
  if (currentLocale.value !== newGlobalLocale) {
    currentLocale.value = newGlobalLocale;
  }
});

const currentLanguageIcon = computed(() => {
  const found = availableLocales.value.find((l) => l.code === currentLocale.value);
  return found ? found.icon : 'mdi-translate';
});

const currentLanguageColor = computed(() => {
  const found = availableLocales.value.find((l) => l.code === currentLocale.value);
  return found ? found.color : 'primary';
});
</script>

<style scoped>
.language-selector-container {
  display: inline-flex;
  align-items: center;
  margin-right: 8px;
}

.language-btn {
  min-width: 40px;
  background-color: rgba(var(--v-theme-primary), 0.1);
  font-weight: 500;
}

.language-label {
  font-size: 14px;
  color: rgba(var(--v-theme-on-surface), 0.7);
}

.current-lang-code {
  font-size: 12px;
  font-weight: bold;
}
</style>
