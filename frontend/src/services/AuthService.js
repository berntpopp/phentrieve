/**
 * Auth API wrapper for the /api/v1/auth endpoints.
 *
 * Uses the shared {@link module:services/apiClient} (bearer + cookies +
 * refresh-retry). Cookie-bearing endpoints (refresh/logout) echo the CSRF
 * token via the double-submit pattern.
 *
 * @module services/AuthService
 */

import apiClient, { readCookie } from './apiClient';

function csrfHeaders() {
  const csrf = readCookie('csrf_token');
  return csrf ? { 'X-CSRF-Token': csrf } : {};
}

class AuthService {
  async register(email, password) {
    const { data } = await apiClient.post('/auth/register', { email, password });
    return data;
  }

  async verify(token) {
    const { data } = await apiClient.post('/auth/verify', { token });
    return data;
  }

  async resendVerification(email) {
    const { data } = await apiClient.post('/auth/resend-verification', { email });
    return data;
  }

  async login(email, password) {
    const { data } = await apiClient.post('/auth/login', { email, password });
    return data; // { access_token, token_type, user }
  }

  async refresh() {
    const { data } = await apiClient.post('/auth/refresh', null, {
      headers: csrfHeaders(),
    });
    return data;
  }

  async logout() {
    const { data } = await apiClient.post('/auth/logout', null, {
      headers: csrfHeaders(),
    });
    return data;
  }

  async me() {
    const { data } = await apiClient.get('/auth/me');
    return data; // { email, is_verified }
  }

  async requestPasswordReset(email) {
    const { data } = await apiClient.post('/auth/password-reset/request', { email });
    return data;
  }

  async confirmPasswordReset(token, newPassword) {
    const { data } = await apiClient.post('/auth/password-reset/confirm', {
      token,
      new_password: newPassword,
    });
    return data;
  }
}

export default new AuthService();
