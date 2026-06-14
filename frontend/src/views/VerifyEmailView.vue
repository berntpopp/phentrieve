<template>
  <v-container class="d-flex justify-center align-center" style="min-height: 70vh">
    <v-card max-width="440" width="100%">
      <v-card-title class="d-flex align-center">
        <v-icon class="mr-2">mdi-email-check-outline</v-icon>
        {{ t('auth.verifyView.title') }}
      </v-card-title>
      <v-divider />
      <v-card-text class="text-center py-6">
        <v-progress-circular v-if="status === 'pending'" indeterminate color="primary" />
        <v-alert
          v-else
          :type="status === 'success' ? 'success' : 'error'"
          variant="tonal"
          :text="message"
        />
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
import { ref, onMounted } from 'vue';
import { useRoute } from 'vue-router';
import { useI18n } from 'vue-i18n';
import { useAuthStore } from '../stores/auth';

const { t } = useI18n();
const route = useRoute();
const authStore = useAuthStore();

const status = ref('pending'); // pending | success | error
const message = ref('');

onMounted(async () => {
  const token = route.query.token;
  if (!token) {
    status.value = 'error';
    message.value = t('auth.verifyView.missingToken');
    return;
  }
  try {
    await authStore.verify(token);
    status.value = 'success';
    message.value = t('auth.verifyView.success');
    // Refresh user state if signed in so the verified badge/quota update.
    if (authStore.isAuthenticated) {
      await authStore.fetchMe();
      await authStore.fetchQuota();
    }
  } catch {
    status.value = 'error';
    message.value = t('auth.verifyView.failed');
  }
});
</script>
