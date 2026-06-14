<template>
  <v-container class="d-flex justify-center align-center" style="min-height: 70vh">
    <v-card max-width="440" width="100%">
      <v-card-title class="d-flex align-center">
        <v-icon class="mr-2">mdi-lock-reset</v-icon>
        {{ t('auth.resetView.title') }}
      </v-card-title>
      <v-divider />
      <v-card-text>
        <v-alert
          v-if="done"
          type="success"
          variant="tonal"
          class="mb-2"
          :text="t('auth.resetView.success')"
        />
        <v-alert
          v-else-if="!token"
          type="error"
          variant="tonal"
          :text="t('auth.resetView.missingToken')"
        />
        <v-alert
          v-if="errorMessage"
          type="error"
          variant="tonal"
          density="compact"
          class="mb-2"
          :text="errorMessage"
        />

        <v-form v-if="token && !done" ref="formRef" @submit.prevent="submit">
          <v-text-field
            v-model="password"
            :label="t('auth.fields.newPassword')"
            :type="show ? 'text' : 'password'"
            autocomplete="new-password"
            prepend-inner-icon="mdi-lock-outline"
            :append-inner-icon="show ? 'mdi-eye-off' : 'mdi-eye'"
            :rules="passwordRules"
            variant="outlined"
            density="comfortable"
            @click:append-inner="show = !show"
          />
          <v-text-field
            v-model="confirmPassword"
            :label="t('auth.fields.confirmPassword')"
            type="password"
            autocomplete="new-password"
            prepend-inner-icon="mdi-lock-check-outline"
            :rules="confirmRules"
            variant="outlined"
            density="comfortable"
          />
          <v-btn type="submit" color="primary" block :loading="loading">
            {{ t('auth.resetView.submit') }}
          </v-btn>
        </v-form>
      </v-card-text>
      <v-divider />
      <v-card-actions>
        <v-spacer />
        <v-btn color="primary" variant="text" to="/">{{ t('auth.verifyView.goHome') }}</v-btn>
      </v-card-actions>
    </v-card>
  </v-container>
</template>

<script setup>
import { ref, computed } from 'vue';
import { useRoute } from 'vue-router';
import { useI18n } from 'vue-i18n';
import { useAuthStore } from '../stores/auth';

const { t } = useI18n();
const route = useRoute();
const authStore = useAuthStore();

const token = route.query.token || '';
const password = ref('');
const confirmPassword = ref('');
const show = ref(false);
const loading = ref(false);
const done = ref(false);
const errorMessage = ref('');
const formRef = ref(null);

const passwordRules = [
  (v) => !!v || t('auth.validation.passwordRequired'),
  (v) => (v && v.length >= 10) || t('auth.validation.passwordMin'),
];
const confirmRules = computed(() => [
  (v) => v === password.value || t('auth.validation.passwordMatch'),
]);

async function submit() {
  errorMessage.value = '';
  const validation = await formRef.value?.validate();
  if (validation && validation.valid === false) return;
  loading.value = true;
  try {
    await authStore.confirmPasswordReset(token, password.value);
    done.value = true;
  } catch {
    errorMessage.value = t('auth.resetView.failed');
  } finally {
    loading.value = false;
  }
}
</script>
