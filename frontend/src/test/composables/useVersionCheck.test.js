import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock the version utility before importing the composable
vi.mock('../../utils/version', () => ({
  getAllVersions: vi.fn(),
}));

import { useVersionCheck } from '../../composables/useVersionCheck';
import { getAllVersions } from '../../utils/version';

describe('useVersionCheck', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns all expected refs and the refreshVersions function', () => {
    const result = useVersionCheck();
    expect(result).toHaveProperty('frontendVersion');
    expect(result).toHaveProperty('apiVersion');
    expect(result).toHaveProperty('cliVersion');
    expect(result).toHaveProperty('environment');
    expect(result).toHaveProperty('loadingVersions');
    expect(result).toHaveProperty('refreshVersions');
    expect(typeof result.refreshVersions).toBe('function');
  });

  it('initializes with loading state values', () => {
    const { frontendVersion, apiVersion, cliVersion, environment, loadingVersions } =
      useVersionCheck();
    expect(frontendVersion.value).toBe('Loading...');
    expect(apiVersion.value).toBe('Loading...');
    expect(cliVersion.value).toBe('Loading...');
    expect(environment.value).toBe('unknown');
    expect(loadingVersions.value).toBe(false);
  });

  it('refreshVersions populates version refs on success', async () => {
    getAllVersions.mockResolvedValue({
      frontend: { version: '1.0.0' },
      api: { version: '2.0.0' },
      cli: { version: '3.0.0' },
      environment: 'development',
    });

    const {
      frontendVersion,
      apiVersion,
      cliVersion,
      environment,
      loadingVersions,
      refreshVersions,
    } = useVersionCheck();

    await refreshVersions();

    expect(frontendVersion.value).toBe('1.0.0');
    expect(apiVersion.value).toBe('2.0.0');
    expect(cliVersion.value).toBe('3.0.0');
    expect(environment.value).toBe('development');
    expect(loadingVersions.value).toBe(false);
  });

  it('refreshVersions handles missing version fields gracefully', async () => {
    getAllVersions.mockResolvedValue({});

    const { frontendVersion, apiVersion, cliVersion, environment, refreshVersions } =
      useVersionCheck();

    await refreshVersions();

    expect(frontendVersion.value).toBe('unknown');
    expect(apiVersion.value).toBe('unknown');
    expect(cliVersion.value).toBe('unknown');
    expect(environment.value).toBe('unknown');
  });

  it('refreshVersions keeps current values on error', async () => {
    getAllVersions.mockRejectedValue(new Error('Network error'));

    const { frontendVersion, loadingVersions, refreshVersions } = useVersionCheck();

    await refreshVersions();

    // Should keep initial 'Loading...' value on error
    expect(frontendVersion.value).toBe('Loading...');
    expect(loadingVersions.value).toBe(false);
  });

  it('sets loadingVersions to true during fetch', async () => {
    let resolvePromise;
    getAllVersions.mockReturnValue(
      new Promise((resolve) => {
        resolvePromise = resolve;
      })
    );

    const { loadingVersions, refreshVersions } = useVersionCheck();

    const promise = refreshVersions();
    expect(loadingVersions.value).toBe(true);

    resolvePromise({ frontend: { version: '1.0.0' } });
    await promise;

    expect(loadingVersions.value).toBe(false);
  });
});
