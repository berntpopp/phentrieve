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
          detail: {
            quota_remaining: 0,
            quota_limit: 3,
            usage_date_utc: '2026-04-16',
            error_message: 'LLM daily quota exhausted.',
          },
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
        apiResponseMessage: 'LLM daily quota exhausted.',
      }),
    });

    await expect(
      PhentrieveService.processText({
        text: 'Patient had recurrent seizures.',
        extractionBackend: 'llm',
        llmModel: 'gpt-5.4-mini',
        llmMode: 'two_phase',
      })
    ).rejects.not.toHaveProperty('usageDateUtc');

    await expect(
      PhentrieveService.processText({
        text: 'Patient had recurrent seizures.',
        extractionBackend: 'llm',
        llmModel: 'gpt-5.4-mini',
        llmMode: 'two_phase',
      })
    ).rejects.toMatchObject({
      originalErrorDetails: expect.objectContaining({
        apiResponseData: expect.objectContaining({
          detail: expect.objectContaining({
            quota_remaining: 0,
            quota_limit: 3,
            usage_date_utc: '2026-04-16',
            error_message: 'LLM daily quota exhausted.',
          }),
        }),
      }),
    });
  });

  it('keeps generic handling for llm 429 responses without quota fields', async () => {
    axios.post.mockRejectedValue({
      isAxiosError: true,
      message: 'Request failed with status code 429',
      response: {
        status: 429,
        data: {
          detail: {
            message: 'Too many requests for this LLM endpoint.',
          },
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
      userMessageKey: 'errors.api.unknown',
    });

    await expect(
      PhentrieveService.processText({
        text: 'Patient had recurrent seizures.',
        extractionBackend: 'llm',
        llmModel: 'gpt-5.4-mini',
        llmMode: 'two_phase',
      })
    ).rejects.not.toHaveProperty('quotaRemaining');
    await expect(
      PhentrieveService.processText({
        text: 'Patient had recurrent seizures.',
        extractionBackend: 'llm',
        llmModel: 'gpt-5.4-mini',
        llmMode: 'two_phase',
      })
    ).rejects.not.toHaveProperty('quotaLimit');
  });

  it('keeps generic handling for non-LLM 429 responses', async () => {
    axios.post.mockRejectedValue({
      isAxiosError: true,
      message: 'Request failed with status code 429',
      response: {
        status: 429,
        data: {
          detail: {
            message: 'Too many requests for this endpoint.',
          },
        },
      },
    });

    await expect(
      PhentrieveService.processText({
        text: 'Patient had recurrent seizures.',
        extractionBackend: 'standard',
        chunkingStrategy: 'simple',
      })
    ).rejects.toMatchObject({
      status: 429,
      userMessageKey: 'errors.api.unknown',
    });

    await expect(
      PhentrieveService.processText({
        text: 'Patient had recurrent seizures.',
        extractionBackend: 'standard',
        chunkingStrategy: 'simple',
      })
    ).rejects.not.toHaveProperty('quotaRemaining');
    await expect(
      PhentrieveService.processText({
        text: 'Patient had recurrent seizures.',
        extractionBackend: 'standard',
        chunkingStrategy: 'simple',
      })
    ).rejects.not.toHaveProperty('quotaLimit');
  });

  it('maps topTermPerChunkForAggregation to top_term_per_chunk_for_aggregation', async () => {
    axios.post.mockResolvedValue({ data: { meta: { extraction_backend: 'standard' } } });

    await PhentrieveService.processText({
      text: 'Patient had recurrent seizures.',
      extractionBackend: 'standard',
      topTermPerChunkForAggregation: true,
    });

    expect(axios.post).toHaveBeenCalledWith(
      '/api/v1/text/process',
      expect.objectContaining({
        top_term_per_chunk_for_aggregation: true,
      })
    );
    expect(axios.post.mock.calls[0][1]).not.toHaveProperty('top_term_per_chunk');
  });

  it('omits null and undefined text process fields from the API payload', async () => {
    axios.post.mockResolvedValue({ data: { meta: { extraction_backend: 'llm' } } });

    await PhentrieveService.processText({
      text: 'Patient had recurrent seizures.',
      extractionBackend: 'llm',
      llmModel: 'gpt-5.4-mini',
      trustRemoteCode: false,
      chunkingStrategy: null,
      windowSize: undefined,
      llmMode: null,
      retrievalModelName: null,
    });

    const payload = axios.post.mock.calls[0][1];
    expect(payload).toMatchObject({
      text: 'Patient had recurrent seizures.',
      extraction_backend: 'llm',
      llm_model: 'gpt-5.4-mini',
      trust_remote_code: false,
    });
    expect(payload).not.toHaveProperty('chunking_strategy');
    expect(payload).not.toHaveProperty('window_size');
    expect(payload).not.toHaveProperty('llm_mode');
    expect(payload).not.toHaveProperty('retrieval_model_name');
  });

  describe('adaptive_rechunking pass-through', () => {
    it('forwards adaptive_rechunking when supplied', async () => {
      axios.post.mockResolvedValue({
        data: {
          meta: { adaptive_rechunking: { enabled: true, trigger_count: 2 } },
          processed_chunks: [],
          aggregated_hpo_terms: [],
        },
      });

      await PhentrieveService.processText({
        text: 'Patient.',
        adaptive_rechunking: { enabled: true, quality_threshold: 0.5 },
      });

      const payload = axios.post.mock.calls[0][1];
      expect(payload.adaptive_rechunking).toEqual({
        enabled: true,
        quality_threshold: 0.5,
      });
    });

    it('parses meta.adaptive_rechunking response without error', async () => {
      axios.post.mockResolvedValue({
        data: {
          meta: { adaptive_rechunking: { enabled: true, trigger_count: 1 } },
          processed_chunks: [],
          aggregated_hpo_terms: [],
        },
      });

      const result = await PhentrieveService.processText({ text: 'Patient.' });
      expect(result.meta.adaptive_rechunking.enabled).toBe(true);
    });

    it('omits adaptive_rechunking from payload when not supplied', async () => {
      axios.post.mockResolvedValue({
        data: { meta: {}, processed_chunks: [], aggregated_hpo_terms: [] },
      });

      await PhentrieveService.processText({ text: 'Patient.' });

      const payload = axios.post.mock.calls[0][1];
      expect(payload).not.toHaveProperty('adaptive_rechunking');
    });
  });

  it('posts phenopacket export payloads to the backend endpoint', async () => {
    axios.post.mockResolvedValue({
      data: {
        phenopacket_json: '{"id":"case-1"}',
        annotation_sidecar: null,
      },
    });

    await PhentrieveService.exportPhenopacket({
      case_id: 'case-1',
      phenotypes: [],
    });

    expect(axios.post).toHaveBeenCalledWith('/api/v1/phenopackets/export', {
      case_id: 'case-1',
      phenotypes: [],
    });
  });
});
