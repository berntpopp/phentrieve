<template>
  <div class="account-button">
    <!-- Signed out: subtle icon button -->
    <v-tooltip
      v-if="!authStore.isAuthenticated"
      location="bottom"
      :text="t('auth.account.loginRegister')"
    >
      <template #activator="{ props: tip }">
        <v-btn
          v-bind="tip"
          icon="mdi-account-outline"
          variant="text"
          density="comfortable"
          size="small"
          :aria-label="t('auth.account.loginRegister')"
          @click="openDialog('login')"
        />
      </template>
    </v-tooltip>

    <!-- Signed in: avatar with menu -->
    <v-menu v-else location="bottom end" :close-on-content-click="false">
      <template #activator="{ props: menu }">
        <v-btn
          v-bind="menu"
          variant="text"
          density="comfortable"
          size="small"
          :aria-label="t('auth.account.account')"
        >
          <v-avatar size="28" :color="authStore.isVerified ? 'primary' : 'grey'">
            <span class="text-caption text-white">{{ initial }}</span>
          </v-avatar>
        </v-btn>
      </template>

      <v-card min-width="260">
        <v-card-text class="pb-1">
          <div class="text-caption text-medium-emphasis">{{ t('auth.account.signedInAs') }}</div>
          <div class="text-body-2 font-weight-medium text-truncate">{{ authStore.email }}</div>
          <v-chip
            :color="authStore.isVerified ? 'success' : 'warning'"
            size="x-small"
            label
            class="mt-2"
          >
            <v-icon start size="x-small">
              {{ authStore.isVerified ? 'mdi-check-decagram' : 'mdi-alert-circle-outline' }}
            </v-icon>
            {{ authStore.isVerified ? t('auth.account.verified') : t('auth.account.unverified') }}
          </v-chip>
        </v-card-text>

        <v-card-text v-if="quotaText" class="py-1 text-caption">
          <v-icon size="x-small" class="mr-1">mdi-lightning-bolt-outline</v-icon>
          {{ quotaText }}
        </v-card-text>

        <v-divider />
        <v-list density="compact" nav>
          <v-list-item
            v-if="!authStore.isVerified"
            prepend-icon="mdi-email-fast-outline"
            :title="t('auth.account.resendVerification')"
            @click="resendVerification"
          />
          <v-list-item
            prepend-icon="mdi-logout"
            :title="t('auth.account.logout')"
            @click="logout"
          />
        </v-list>
      </v-card>
    </v-menu>

    <AuthDialog v-model="dialogOpen" :initial-mode="dialogMode" @authenticated="onAuthenticated" />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue';
import { useI18n } from 'vue-i18n';
import { useAuthStore } from '../../stores/auth';
import AuthDialog from './AuthDialog.vue';

const { t } = useI18n();
const authStore = useAuthStore();

const dialogOpen = ref(false);
const dialogMode = ref('login');

const initial = computed(() => (authStore.email ? authStore.email[0].toUpperCase() : '?'));

const quotaText = computed(() => {
  const q = authStore.quota;
  if (!q || q.enforced === false) return '';
  return t('auth.account.quotaToday', {
    remaining: q.quota_remaining,
    limit: q.quota_limit,
  });
});

function openDialog(mode) {
  dialogMode.value = mode;
  dialogOpen.value = true;
}

// Allow other components to request the dialog via a window event.
function handleOpenEvent(event) {
  openDialog(event?.detail?.mode || 'login');
}

onMounted(() => {
  window.addEventListener('phentrieve:open-auth', handleOpenEvent);
  if (authStore.isAuthenticated) authStore.fetchQuota();
});

async function onAuthenticated() {
  await authStore.fetchQuota();
}

async function resendVerification() {
  if (authStore.email) {
    await authStore.resendVerification(authStore.email);
  }
}

async function logout() {
  await authStore.logout();
}

defineExpose({ openDialog });
</script>

<style scoped>
.account-button {
  position: fixed;
  top: 8px;
  right: 12px;
  z-index: 2000;
}
</style>
