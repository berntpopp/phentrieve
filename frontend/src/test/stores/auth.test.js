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
vi.mock('../../services/apiClient', () => ({ default: { get: vi.fn() } }));

import AuthService from '../../services/AuthService';
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

  it('initialize restores a session via refresh', async () => {
    AuthService.refresh.mockResolvedValue({
      access_token: 'r1',
      user: { email: 'b@ex.com', is_verified: true },
    });
    const store = useAuthStore();
    await store.initialize();
    expect(store.isAuthenticated).toBe(true);
    expect(getAccessToken()).toBe('r1');
  });

  it('initialize clears a stale persisted user when refresh fails', async () => {
    AuthService.refresh.mockRejectedValue(new Error('401'));
    const store = useAuthStore();
    store.user = { email: 'stale@ex.com', is_verified: false };
    await store.initialize();
    expect(store.isAuthenticated).toBe(false);
    expect(getAccessToken()).toBe(null);
  });
});
