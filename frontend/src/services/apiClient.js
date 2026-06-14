/**
 * Shared axios instance for authenticated API access.
 *
 * - Attaches the bearer access token to every request.
 * - Sends cookies (refresh + CSRF) via `withCredentials`.
 * - On a 401, performs a single-flight `/auth/refresh` and retries the original
 *   request once. Concurrent 401s share the same refresh promise.
 *
 * @module services/apiClient
 */

import axios from 'axios';
import { getAccessToken, setAccessToken, clearAccessToken } from './authToken';

export const API_URL = import.meta.env.VITE_API_URL || '/api/v1';

/** Read a non-HttpOnly cookie (used for the double-submit CSRF token). */
export function readCookie(name) {
  const target = `${name}=`;
  for (const part of document.cookie.split('; ')) {
    if (part.startsWith(target)) {
      return decodeURIComponent(part.slice(target.length));
    }
  }
  return null;
}

const apiClient = axios.create({ baseURL: API_URL, withCredentials: true });

apiClient.interceptors.request.use((config) => {
  const token = getAccessToken();
  if (token) {
    config.headers = config.headers || {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

let refreshPromise = null;

/** Perform a single refresh (bare axios so it bypasses this interceptor). */
async function performRefresh() {
  const csrf = readCookie('csrf_token');
  const response = await axios.post(`${API_URL}/auth/refresh`, null, {
    withCredentials: true,
    headers: csrf ? { 'X-CSRF-Token': csrf } : {},
  });
  setAccessToken(response.data?.access_token || null);
  return response.data;
}

apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const original = error.config;
    const status = error.response?.status;
    const isRefreshCall = original?.url?.includes('/auth/refresh');
    if (status === 401 && original && !original._retried && !isRefreshCall) {
      original._retried = true;
      try {
        if (!refreshPromise) {
          refreshPromise = performRefresh().finally(() => {
            refreshPromise = null;
          });
        }
        await refreshPromise;
        return apiClient(original);
      } catch {
        clearAccessToken();
      }
    }
    return Promise.reject(error);
  }
);

export default apiClient;
