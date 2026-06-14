<template>
  <v-dialog :model-value="modelValue" max-width="440" @update:model-value="close">
    <v-card>
      <v-card-title class="d-flex align-center">
        <v-icon class="mr-2">mdi-account-circle-outline</v-icon>
        {{ title }}
      </v-card-title>
      <v-divider />

      <v-card-text>
        <v-alert
          v-if="notice"
          type="success"
          variant="tonal"
          density="compact"
          class="mb-3"
          :text="notice"
        />
        <v-alert
          v-if="errorMessage"
          type="error"
          variant="tonal"
          density="compact"
          class="mb-3"
          :text="errorMessage"
        />

        <v-form ref="formRef" @submit.prevent="submit">
          <v-text-field
            v-model="email"
            :label="t('auth.fields.email')"
            type="email"
            autocomplete="email"
            prepend-inner-icon="mdi-email-outline"
            :rules="emailRules"
            density="comfortable"
            variant="outlined"
          />

          <v-text-field
            v-if="mode !== 'forgot'"
            v-model="password"
            :label="t('auth.fields.password')"
            :type="showPassword ? 'text' : 'password'"
            :autocomplete="mode === 'register' ? 'new-password' : 'current-password'"
            prepend-inner-icon="mdi-lock-outline"
            :append-inner-icon="showPassword ? 'mdi-eye-off' : 'mdi-eye'"
            :rules="passwordRules"
            density="comfortable"
            variant="outlined"
            @click:append-inner="showPassword = !showPassword"
          />

          <v-text-field
            v-if="mode === 'register'"
            v-model="confirmPassword"
            :label="t('auth.fields.confirmPassword')"
            type="password"
            autocomplete="new-password"
            prepend-inner-icon="mdi-lock-check-outline"
            :rules="confirmRules"
            density="comfortable"
            variant="outlined"
          />

          <v-btn type="submit" color="primary" block :loading="loading" class="mt-1">
            {{ submitLabel }}
          </v-btn>
        </v-form>
      </v-card-text>

      <v-divider />
      <v-card-actions class="flex-column align-stretch px-4 py-3">
        <v-btn v-if="mode === 'login'" variant="text" size="small" @click="setMode('forgot')">
          {{ t('auth.links.forgotPassword') }}
        </v-btn>
        <v-btn v-if="mode === 'login'" variant="text" size="small" @click="setMode('register')">
          {{ t('auth.links.needAccount') }}
        </v-btn>
        <v-btn v-if="mode !== 'login'" variant="text" size="small" @click="setMode('login')">
          {{ t('auth.links.backToLogin') }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup>
import { ref, computed, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { useAuthStore } from '../../stores/auth';

const props = defineProps({
  modelValue: { type: Boolean, default: false },
  initialMode: { type: String, default: 'login' },
});
const emit = defineEmits(['update:modelValue', 'authenticated']);

const { t } = useI18n();
const authStore = useAuthStore();

const mode = ref(props.initialMode);
const email = ref('');
const password = ref('');
const confirmPassword = ref('');
const showPassword = ref(false);
const loading = ref(false);
const notice = ref('');
const errorMessage = ref('');
const formRef = ref(null);

const title = computed(() => {
  if (mode.value === 'register') return t('auth.register.title');
  if (mode.value === 'forgot') return t('auth.forgot.title');
  return t('auth.login.title');
});

const submitLabel = computed(() => {
  if (mode.value === 'register') return t('auth.actions.register');
  if (mode.value === 'forgot') return t('auth.actions.sendReset');
  return t('auth.actions.login');
});

const emailRules = [
  (v) => !!v || t('auth.validation.emailRequired'),
  (v) => /.+@.+\..+/.test(v) || t('auth.validation.emailInvalid'),
];
const passwordRules = computed(() =>
  mode.value === 'forgot'
    ? []
    : [
        (v) => !!v || t('auth.validation.passwordRequired'),
        (v) =>
          mode.value !== 'register' || (v && v.length >= 10) || t('auth.validation.passwordMin'),
      ]
);
const confirmRules = computed(() => [
  (v) => v === password.value || t('auth.validation.passwordMatch'),
]);

function resetMessages() {
  notice.value = '';
  errorMessage.value = '';
}

function setMode(next) {
  mode.value = next;
  resetMessages();
}

function close() {
  emit('update:modelValue', false);
}

watch(
  () => props.modelValue,
  (open) => {
    if (open) {
      mode.value = props.initialMode;
      resetMessages();
    }
  }
);

function describeError(error) {
  const status = error?.response?.status;
  if (status === 401) return t('auth.errors.loginFailed');
  if (status === 429) return t('auth.errors.locked');
  return t('auth.errors.generic');
}

async function submit() {
  resetMessages();
  const validation = await formRef.value?.validate();
  if (validation && validation.valid === false) return;

  loading.value = true;
  try {
    if (mode.value === 'login') {
      await authStore.login(email.value, password.value);
      emit('authenticated');
      close();
    } else if (mode.value === 'register') {
      await authStore.register(email.value, password.value);
      notice.value = t('auth.messages.registerSuccess');
      mode.value = 'login';
      password.value = '';
      confirmPassword.value = '';
    } else {
      await authStore.requestPasswordReset(email.value);
      notice.value = t('auth.messages.resetSent');
    }
  } catch (error) {
    errorMessage.value = describeError(error);
  } finally {
    loading.value = false;
  }
}

defineExpose({ mode, setMode, submit });
</script>
