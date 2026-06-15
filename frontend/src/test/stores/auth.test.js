import { describe, it, expect, beforeEach, vi } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';

vi.mock('../../services/AuthService', () => ({
  default: {
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
    refresh: vi.fn(),
    me: vi.fn(),
    verify: vi.fn(),
    resendVerification: vi.fn(),
    requestPasswordReset: vi.fn(),
    confirmPasswordReset: vi.fn(),
  },
}));
vi.mock('../../services/apiClient', () => ({ default: { get: vi.fn() }, readCookie: vi.fn() }));

import AuthService from '../../services/AuthService';
import { readCookie } from '../../services/apiClient';
import { useAuthStore } from '../../stores/auth';
import { getAccessToken } from '../../services/authToken';

describe('auth store', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    vi.clearAllMocks();
  });

  it('starts unauthenticated', () => {
    const store = useAuthStore();
    expect(store.isAuthenticated).toBe(false);
    expect(store.isVerified).toBe(false);
    expect(store.email).toBe(null);
  });

  it('login stores the in-memory token and user', async () => {
    AuthService.login.mockResolvedValue({
      access_token: 'abc',
      user: { email: 'a@ex.com', is_verified: true },
    });
    const store = useAuthStore();
    await store.login('a@ex.com', 'pw');
    expect(store.isAuthenticated).toBe(true);
    expect(store.isVerified).toBe(true);
    expect(store.email).toBe('a@ex.com');
    expect(getAccessToken()).toBe('abc');
  });

  it('logout clears the session even when the request fails', async () => {
    AuthService.login.mockResolvedValue({
      access_token: 'abc',
      user: { email: 'a@ex.com', is_verified: false },
    });
    AuthService.logout.mockRejectedValue(new Error('network'));
    const store = useAuthStore();
    await store.login('a@ex.com', 'pw');
    await store.logout();
    expect(store.isAuthenticated).toBe(false);
    expect(getAccessToken()).toBe(null);
  });

  it('initialize restores a session via refresh when the csrf cookie is present', async () => {
    readCookie.mockReturnValue('csrf-1');
    AuthService.refresh.mockResolvedValue({
      access_token: 'r1',
      user: { email: 'b@ex.com', is_verified: true },
    });
    const store = useAuthStore();
    await store.initialize();
    expect(AuthService.refresh).toHaveBeenCalledTimes(1);
    expect(store.isAuthenticated).toBe(true);
    expect(getAccessToken()).toBe('r1');
  });

  it('initialize clears a stale persisted user when refresh fails', async () => {
    readCookie.mockReturnValue('csrf-1');
    AuthService.refresh.mockRejectedValue(new Error('401'));
    const store = useAuthStore();
    store.user = { email: 'stale@ex.com', is_verified: false };
    await store.initialize();
    expect(store.isAuthenticated).toBe(false);
    expect(getAccessToken()).toBe(null);
  });

  it('initialize skips refresh entirely when no session cookie exists', async () => {
    // Anonymous visit: no csrf_token cookie -> no session to restore, so the
    // refresh request (which would 403) must not be made at all.
    readCookie.mockReturnValue(null);
    const store = useAuthStore();
    await store.initialize();
    expect(AuthService.refresh).not.toHaveBeenCalled();
    expect(store.isAuthenticated).toBe(false);
    expect(getAccessToken()).toBe(null);
  });

  it('initialize clears a stale persisted user without a network call when the cookie is gone', async () => {
    readCookie.mockReturnValue(null);
    const store = useAuthStore();
    store.user = { email: 'stale@ex.com', is_verified: false };
    await store.initialize();
    expect(AuthService.refresh).not.toHaveBeenCalled();
    expect(store.isAuthenticated).toBe(false);
  });
});
