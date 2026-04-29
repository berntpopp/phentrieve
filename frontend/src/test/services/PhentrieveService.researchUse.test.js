import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('axios', () => ({
  default: {
    post: vi.fn(),
    get: vi.fn(),
  },
}));

import axios from 'axios';
import PhentrieveService from '../../services/PhentrieveService';

describe('PhentrieveService research-use guardrails', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('sends research-use acknowledgement on text-bearing API calls', async () => {
    axios.post.mockResolvedValue({ data: { ok: true } });

    await PhentrieveService.queryHpo({ text: 'Research note.', num_results: 5 });
    await PhentrieveService.processText({ text: 'Research note.' });
    await PhentrieveService.exportPhenopacket({
      input_text: 'Research note.',
      phenotypes: [],
    });

    expect(axios.post).toHaveBeenNthCalledWith(
      1,
      '/api/v1/query/',
      expect.objectContaining({ text: 'Research note.' }),
      expect.objectContaining({
        headers: { 'X-Phentrieve-Research-Use-Acknowledged': 'true' },
      })
    );
    expect(axios.post).toHaveBeenNthCalledWith(
      2,
      '/api/v1/text/process',
      expect.objectContaining({ text: 'Research note.' }),
      expect.objectContaining({
        headers: { 'X-Phentrieve-Research-Use-Acknowledged': 'true' },
      })
    );
    expect(axios.post).toHaveBeenNthCalledWith(
      3,
      '/api/v1/phenopackets/export',
      expect.objectContaining({ input_text: 'Research note.' }),
      expect.objectContaining({
        headers: { 'X-Phentrieve-Research-Use-Acknowledged': 'true' },
      })
    );
  });
});
