/**
 * Pinia store for authentication state.
 *
 * The access token lives in memory only (see services/authToken); only the
 * lightweight user object is persisted so the UI can render the signed-in
 * state immediately on reload, after which a silent refresh restores the
 * token. Mirrors the setup-store + persistedstate pattern used by the
 * disclaimer store.
 *
 * @module stores/auth
 */

import { defineStore } from 'pinia';
import { ref, computed } from 'vue';

import AuthService from '../services/AuthService';
import apiClient from '../services/apiClient';
import { setAccessToken, clearAccessToken } from '../services/authToken';
import { logService } from '../services/logService';

export const useAuthStore = defineStore(
  'auth',
  () => {
    // ===========================
    // State
    // ===========================
    const user = ref(null); // { email, is_verified } | null (persisted)
    const quota = ref(null); // { quota_limit, quota_remaining, ... } | null
    const initialized = ref(false);

    // ===========================
    // Getters
    // ===========================
    const isAuthenticated = computed(() => user.value !== null);
    const isVerified = computed(() => user.value?.is_verified === true);
    const email = computed(() => user.value?.email || null);

    // ===========================
    // Internal helpers
    // ===========================
    function setSession(accessToken, userPayload) {
      setAccessToken(accessToken);
      user.value = userPayload || null;
    }

    function clearSession() {
      clearAccessToken();
      user.value = null;
      quota.value = null;
    }

    // ===========================
    // Actions
    // ===========================
    async function register(emailValue, password) {
      return AuthService.register(emailValue, password);
    }

    async function verify(token) {
      return AuthService.verify(token);
    }

    async function resendVerification(emailValue) {
      return AuthService.resendVerification(emailValue);
    }

    async function login(emailValue, password) {
      const data = await AuthService.login(emailValue, password);
      setSession(data.access_token, data.user);
      return data.user;
    }

    async function logout() {
      try {
        await AuthService.logout();
      } catch (error) {
        logService.debug('Logout request failed (clearing locally anyway)', {
          message: error?.message,
        });
      }
      clearSession();
    }

    async function fetchMe() {
      const me = await AuthService.me();
      user.value = me;
      return me;
    }

    /**
     * Silent session restore on app start: try to refresh using the HttpOnly
     * cookie. On failure, clear any stale persisted user. Never throws.
     */
    async function initialize() {
      if (initialized.value) return;
      try {
        const data = await AuthService.refresh();
        setSession(data.access_token, data.user);
      } catch {
        clearSession();
      } finally {
        initialized.value = true;
      }
    }

    async function requestPasswordReset(emailValue) {
      return AuthService.requestPasswordReset(emailValue);
    }

    async function confirmPasswordReset(token, newPassword) {
      return AuthService.confirmPasswordReset(token, newPassword);
    }

    async function fetchQuota() {
      try {
        const { data } = await apiClient.get('/text/quota');
        quota.value = data;
        return data;
      } catch (error) {
        logService.debug('Quota fetch failed', { message: error?.message });
        return null;
      }
    }

    return {
      // state
      user,
      quota,
      initialized,
      // getters
      isAuthenticated,
      isVerified,
      email,
      // actions
      setSession,
      clearSession,
      register,
      verify,
      resendVerification,
      login,
      logout,
      fetchMe,
      initialize,
      requestPasswordReset,
      confirmPasswordReset,
      fetchQuota,
    };
  },
  {
    persist: {
      key: 'phentrieve-auth',
      storage: localStorage,
      pick: ['user'],
    },
  }
);
