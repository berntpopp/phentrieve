import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('axios', () => ({
  default: {
    post: vi.fn(),
    get: vi.fn(),
  },
}));

import axios from 'axios';
import PhentrieveService from '../../services/PhentrieveService';

describe('PhentrieveService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('sends extraction_backend=llm in text process requests', async () => {
    axios.post.mockResolvedValue({ data: { meta: { extraction_backend: 'llm' } } });

    await PhentrieveService.processText({
      text: 'Patient had recurrent seizures.',
      extractionBackend: 'llm',
      llmModel: 'gpt-5.4-mini',
      llmMode: 'two_phase',
    });

    expect(axios.post).toHaveBeenCalledWith(
      '/api/v1/text/process',
      expect.objectContaining({ extraction_backend: 'llm' })
    );
  });

  it('preserves structured quota fields for 429 responses', async () => {
    axios.post.mockRejectedValue({
      isAxiosError: true,
      message: 'Request failed with status code 429',
      response: {
        status: 429,
        data: {
          detail: 'LLM full-text limit reached for today.',
          quota_remaining: 0,
          quota_limit: 3,
          extraction_backend: 'llm',
        },
      },
    });

    await expect(
      PhentrieveService.processText({
        text: 'Patient had recurrent seizures.',
        extractionBackend: 'llm',
        llmModel: 'gpt-5.4-mini',
        llmMode: 'two_phase',
      })
    ).rejects.toMatchObject({
      status: 429,
      userMessageKey: 'errors.api.llmQuotaExceeded',
      quotaRemaining: 0,
      quotaLimit: 3,
      extractionBackend: 'llm',
      originalErrorDetails: expect.objectContaining({
        apiResponseData: expect.objectContaining({
          quota_remaining: 0,
          quota_limit: 3,
          extraction_backend: 'llm',
        }),
      }),
    });
  });
});
