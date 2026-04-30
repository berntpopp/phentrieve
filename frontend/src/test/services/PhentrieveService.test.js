import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('axios', () => ({
  default: {
    post: vi.fn(),
    get: vi.fn(),
  },
}));

vi.mock('../../services/logService', () => ({
  logService: {
    info: vi.fn(),
    debug: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

import axios from 'axios';
import PhentrieveService from '../../services/PhentrieveService';
import { logService } from '../../services/logService';

function serializedServiceLogs() {
  return JSON.stringify([
    ...logService.info.mock.calls,
    ...logService.debug.mock.calls,
    ...logService.warn.mock.calls,
    ...logService.error.mock.calls,
  ]);
}

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
      expect.objectContaining({ extraction_backend: 'llm' }),
      expect.any(Object)
    );
  });

  it('does not send client-selected LLM model fields in text process requests', async () => {
    axios.post.mockResolvedValue({ data: { meta: { extraction_backend: 'llm' } } });

    await PhentrieveService.processText({
      text: 'Patient had recurrent seizures.',
      extractionBackend: 'llm',
      llmModel: 'gpt-5.4-mini',
      llm_model: 'gemini-3.1-flash-lite-preview',
      model_name: 'legacy-model-name',
      llmMode: 'two_phase',
    });

    const payload = axios.post.mock.calls[0][1];
    expect(payload).toMatchObject({
      text: 'Patient had recurrent seizures.',
      extraction_backend: 'llm',
      llm_mode: 'two_phase',
    });
    expect(payload).not.toHaveProperty('llm_model');
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
        apiResponseStatus: 429,
        apiResponseDataSize: expect.any(Number),
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
    ).rejects.not.toHaveProperty('originalErrorDetails.apiResponseData');
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
      }),
      expect.any(Object)
    );
    expect(axios.post.mock.calls[0][1]).not.toHaveProperty('top_term_per_chunk');
  });

  it('omits null, undefined, and model-loader-only text process fields from the API payload', async () => {
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
    });
    expect(payload).not.toHaveProperty('trust_remote_code');
    expect(payload).not.toHaveProperty('llm_model');
    expect(payload).not.toHaveProperty('chunking_strategy');
    expect(payload).not.toHaveProperty('window_size');
    expect(payload).not.toHaveProperty('llm_mode');
    expect(payload).not.toHaveProperty('retrieval_model_name');
  });

  it('logs query request and response metadata without raw text payloads', async () => {
    axios.post.mockResolvedValue({
      status: 200,
      data: {
        query_text_received: 'Email jane@example.org with seizures',
        results: [{ id: 'HP:0001250' }],
      },
    });

    await PhentrieveService.queryHpo({
      text: 'Email jane@example.org with seizures',
      num_results: 5,
      model_name: 'test-model',
    });

    const logs = serializedServiceLogs();
    expect(logs).not.toContain('jane@example.org');
    expect(logs).not.toContain('Email jane');
    expect(logs).not.toContain('query_text_received');
    expect(logs).toContain('textLength');
    expect(logs).toContain('resultsCount');
  });

  it('logs API errors with response metadata without raw response or request bodies', async () => {
    axios.post.mockRejectedValueOnce({
      isAxiosError: true,
      message: 'Request failed with status code 400',
      name: 'AxiosError',
      code: 'ERR_BAD_REQUEST',
      config: {
        url: '/api/v1/query/',
        method: 'post',
        timeout: 0,
        data: 'Email jane@example.org with seizures',
      },
      request: {},
      response: {
        status: 400,
        statusText: 'Bad Request',
        data: {
          detail: 'Email jane@example.org with seizures is invalid',
        },
      },
    });

    await expect(
      PhentrieveService.queryHpo({
        text: 'Email jane@example.org with seizures',
        num_results: 5,
      })
    ).rejects.toMatchObject({ status: 400 });

    axios.post.mockRejectedValueOnce({
      isAxiosError: true,
      message: 'Request failed with status code 422',
      name: 'AxiosError',
      code: 'ERR_BAD_RESPONSE',
      config: {
        url: '/api/v1/text/process',
        method: 'post',
        timeout: 0,
        data: JSON.stringify({ text: 'Email jane@example.org with seizures' }),
      },
      request: {},
      response: {
        status: 422,
        statusText: 'Unprocessable Content',
        data: {
          submitted_text: 'Email jane@example.org with seizures',
          redacted_text: '[REDACTED_EMAIL] with seizures',
          snippets: ['Email jane@example.org'],
        },
      },
    });

    await expect(
      PhentrieveService.processText({
        text: 'Email jane@example.org with seizures',
        extractionBackend: 'standard',
      })
    ).rejects.toMatchObject({ status: 422 });

    const logs = serializedServiceLogs();
    expect(logs).not.toContain('jane@example.org');
    expect(logs).not.toContain('Email jane');
    expect(logs).not.toContain('submitted_text');
    expect(logs).not.toContain('redacted_text');
    expect(logs).not.toContain('snippets');
    expect(logs).not.toContain('Bad Request');
    expect(logs).not.toContain('Unprocessable Content');
    expect(logs).toContain('dataSize');
  });

  it('does not expose raw API error details on thrown standardized errors', async () => {
    axios.post.mockRejectedValueOnce({
      isAxiosError: true,
      message: 'Request failed with status code 400',
      name: 'AxiosError',
      code: 'ERR_BAD_REQUEST',
      config: {
        url: '/api/v1/query/',
        method: 'post',
        data: 'Email jane@example.org with seizures',
      },
      response: {
        status: 400,
        data: {
          detail: 'Email jane@example.org with seizures is invalid',
          submitted_text: 'Email jane@example.org with seizures',
          snippets: ['Email jane@example.org'],
        },
      },
    });

    let thrownError;
    try {
      await PhentrieveService.queryHpo({
        text: 'Email jane@example.org with seizures',
        num_results: 5,
      });
    } catch (error) {
      thrownError = error;
    }

    const serializedError = JSON.stringify(thrownError);
    expect(serializedError).not.toContain('jane@example.org');
    expect(serializedError).not.toContain('Email jane');
    expect(serializedError).not.toContain('submitted_text');
    expect(serializedError).not.toContain('snippets');
    expect(thrownError.originalErrorDetails).toEqual(
      expect.objectContaining({
        apiResponseStatus: 400,
        apiResponseDataSize: expect.any(Number),
      })
    );
    expect(thrownError.originalErrorDetails).not.toHaveProperty('apiResponseData');
  });

  it('does not expose object-shaped API error detail messages on thrown errors', async () => {
    axios.post.mockRejectedValueOnce({
      isAxiosError: true,
      message: 'Request failed with status code 422',
      name: 'AxiosError',
      code: 'ERR_BAD_REQUEST',
      config: {
        url: '/api/v1/query/',
        method: 'post',
      },
      response: {
        status: 422,
        data: {
          detail: {
            message: 'Email jane@example.org with seizures is invalid',
            error_message: 'Email jane@example.org was rejected',
          },
        },
      },
    });

    let thrownError;
    try {
      await PhentrieveService.queryHpo({
        text: 'Email jane@example.org with seizures',
        num_results: 5,
      });
    } catch (error) {
      thrownError = error;
    }

    const serializedError = JSON.stringify(thrownError);
    expect(serializedError).not.toContain('jane@example.org');
    expect(serializedError).not.toContain('Email jane');
    expect(thrownError.userMessageParams.detail).toBe('API returned an error. See status code for details.');
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

    expect(axios.post).toHaveBeenCalledWith(
      '/api/v1/phenopackets/export',
      {
        case_id: 'case-1',
        phenotypes: [],
      },
      expect.any(Object)
    );
  });
});
