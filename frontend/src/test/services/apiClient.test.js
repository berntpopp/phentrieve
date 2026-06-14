import { describe, it, expect, beforeEach } from 'vitest';

import apiClient, { readCookie } from '../../services/apiClient';
import { setAccessToken, clearAccessToken } from '../../services/authToken';

describe('apiClient', () => {
  beforeEach(() => {
    clearAccessToken();
  });

  it('readCookie extracts a named cookie value', () => {
    document.cookie = 'csrf_token=abc123';
    expect(readCookie('csrf_token')).toBe('abc123');
    expect(readCookie('nonexistent_cookie')).toBe(null);
  });

  it('request interceptor attaches the bearer token when set', () => {
    setAccessToken('tok-123');
    const handler = apiClient.interceptors.request.handlers[0].fulfilled;
    const config = handler({ headers: {} });
    expect(config.headers.Authorization).toBe('Bearer tok-123');
  });

  it('request interceptor omits the auth header when signed out', () => {
    clearAccessToken();
    const handler = apiClient.interceptors.request.handlers[0].fulfilled;
    const config = handler({ headers: {} });
    expect(config.headers.Authorization).toBeUndefined();
  });
});
