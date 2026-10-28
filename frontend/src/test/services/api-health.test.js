import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

vi.mock('axios', () => ({
  default: { get: vi.fn() },
}));

vi.mock('../../services/logService', () => ({
  logService: { info: vi.fn(), debug: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

import axios from 'axios';
import { apiHealthService } from '../../services/api-health';

function setHidden(hidden) {
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden });
}

describe('apiHealthService polling', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.clearAllMocks();
    axios.get.mockResolvedValue({ status: 200 });
  });

  afterEach(() => {
    apiHealthService.stopMonitoring();
    vi.useRealTimers();
    setHidden(false);
  });

  it('skips interval polls while the tab is hidden (#431)', async () => {
    setHidden(false);
    apiHealthService.startMonitoring();
    expect(axios.get).toHaveBeenCalledTimes(1); // initial check

    setHidden(true);
    await vi.advanceTimersByTimeAsync(apiHealthService.checkInterval * 3);
    expect(axios.get).toHaveBeenCalledTimes(1);

    setHidden(false);
    await vi.advanceTimersByTimeAsync(apiHealthService.checkInterval);
    expect(axios.get).toHaveBeenCalledTimes(2);
  });
});
