import axios from 'axios';
import { logService } from './logService';

/**
 * Determine the API URL based on environment variables and deployment context
 *
 * Priority order:
 * 1. VITE_API_URL (from .env) - for external URLs like https://phentrieve-api.kidney-genetics.org/api/v1
 * 2. Relative path '/api/v1' - when frontend Nginx proxies to API (Docker/NPM setup)
 * 3. Development fallback - localhost with port for local development
 */
const API_URL = import.meta.env.VITE_API_URL || '/api/v1'; // Default to relative path for Nginx proxy

class PhentrieveService {
  async queryHpo(queryData) {
    // queryData should match QueryRequest schema from FastAPI
    // Example: { text: "...", model_name: "...", num_results: 10, ... }
    try {
      logService.info('Querying HPO API', { query: queryData });
      const response = await axios.post(`${API_URL}/query/`, queryData);
      logService.debug('HPO API response received', {
        status: response.status,
        data: response.data,
      });
      return response.data; // Expected to match QueryResponse schema
    } catch (error) {
      logService.error('Original API Error in PhentrieveService queryHpo:', {
        message: error.message,
        name: error.name,
        code: error.code,
        config: error.config
          ? {
              url: error.config.url,
              method: error.config.method,
              timeout: error.config.timeout,
            }
          : null,
        request: error.request ? 'Exists' : 'DoesNotExist', // Avoid logging large objects
        response: error.response
          ? {
              status: error.response.status,
              data: error.response.data,
            }
          : null,
      });
      throw this._createStandardizedError(error, 'querying HPO terms');
    }
  }

  /**
   * Processes longer text content for HPO term extraction with detailed chunk-level analysis
   * @param {Object} textProcessingData - Data matching TextProcessingRequest schema from FastAPI
   * @returns {Object} Data matching TextProcessingResponseAPI schema
   */
  async processText(textProcessingData) {
    let normalizedPayload;
    try {
      normalizedPayload = this._normalizeTextProcessPayload(textProcessingData);
      logService.info('Calling Text Processing API', {
        requestSize: JSON.stringify(normalizedPayload).length,
        textLength: normalizedPayload.text?.length || 0,
        backend: normalizedPayload.extraction_backend,
        model: normalizedPayload.llm_model || normalizedPayload.retrieval_model_name,
      });

      const response = await axios.post(`${API_URL}/text/process`, normalizedPayload);

      logService.debug('Text Processing API response received', {
        status: response.status,
        dataSize: JSON.stringify(response.data).length,
        numChunks: response.data.processed_chunks?.length || 0,
        numAggregatedTerms: response.data.aggregated_hpo_terms?.length || 0,
      });

      return response.data; // Expected to match TextProcessingResponseAPI schema
    } catch (error) {
      logService.error('Original API Error in PhentrieveService processText:', {
        message: error.message,
        name: error.name,
        code: error.code,
        config: error.config
          ? {
              url: error.config.url,
              method: error.config.method,
              timeout: error.config.timeout,
            }
          : null,
        request: error.request ? 'Exists' : 'DoesNotExist', // Avoid logging large objects
        response: error.response
          ? {
              status: error.response.status,
              data: error.response.data,
            }
          : null,
      });
      throw this._createStandardizedError(error, 'processing text for HPO extraction', {
        isTextProcessing: true,
        extractionBackend: normalizedPayload.extraction_backend,
      });
    }
  }

  _normalizeTextProcessPayload(textProcessingData) {
    const extractionBackend =
      textProcessingData.extraction_backend ?? textProcessingData.extractionBackend ?? 'standard';

    return {
      text: textProcessingData.text ?? textProcessingData.text_content ?? '',
      extraction_backend: extractionBackend,
      llm_model:
        textProcessingData.llm_model ??
        textProcessingData.llmModel ??
        textProcessingData.model_name ??
        null,
      llm_mode: textProcessingData.llm_mode ?? textProcessingData.llmMode ?? null,
      language: textProcessingData.language ?? null,
      chunking_strategy:
        textProcessingData.chunking_strategy ?? textProcessingData.chunkingStrategy ?? null,
      window_size: textProcessingData.window_size ?? textProcessingData.windowSize ?? null,
      step_size: textProcessingData.step_size ?? textProcessingData.stepSize ?? null,
      split_threshold: textProcessingData.split_threshold ?? textProcessingData.splitThreshold ?? null,
      min_segment_length:
        textProcessingData.min_segment_length ?? textProcessingData.minSegmentLength ?? null,
      semantic_model_name:
        textProcessingData.semantic_model_name ??
        textProcessingData.semanticModelForChunking ??
        textProcessingData.semanticModelName ??
        textProcessingData.selectedModel ??
        null,
      retrieval_model_name:
        textProcessingData.retrieval_model_name ??
        textProcessingData.retrievalModelForTextProcess ??
        textProcessingData.retrievalModelName ??
        textProcessingData.selectedModel ??
        null,
      trust_remote_code: textProcessingData.trust_remote_code ?? textProcessingData.trustRemoteCode,
      chunk_retrieval_threshold:
        textProcessingData.chunk_retrieval_threshold ??
        textProcessingData.chunkRetrievalThreshold ??
        null,
      num_results_per_chunk:
        textProcessingData.num_results_per_chunk ?? textProcessingData.numResultsPerChunk ?? null,
      no_assertion_detection:
        textProcessingData.no_assertion_detection ??
        textProcessingData.noAssertionDetectionForTextProcess ??
        null,
      assertion_preference:
        textProcessingData.assertion_preference ??
        textProcessingData.assertionPreferenceForTextProcess ??
        null,
      aggregated_term_confidence:
        textProcessingData.aggregated_term_confidence ??
        textProcessingData.aggregatedTermConfidence ??
        null,
      top_term_per_chunk_for_aggregation:
        textProcessingData.top_term_per_chunk_for_aggregation ??
        textProcessingData.topTermPerChunkForAggregation ??
        null,
      include_details: textProcessingData.include_details ?? textProcessingData.includeDetails,
    };
  }

  /**
   * Fetches configuration info from the API including available models
   * @returns {Object} Data matching PhentrieveConfigInfoResponseAPI schema
   */
  async getConfigInfo() {
    try {
      logService.debug('Fetching API configuration info');
      const response = await axios.get(`${API_URL}/info`);
      logService.debug('API config info received', {
        embeddingModelsCount: response.data.available_embedding_models?.length || 0,
        defaultModel: response.data.default_embedding_model,
      });
      return response.data;
    } catch (error) {
      logService.error('Error fetching API config info:', {
        message: error.message,
        status: error.response?.status,
      });
      throw this._createStandardizedError(error, 'fetching configuration info');
    }
  }

  /**
   * Creates a standardized error object from an Axios error
   * @param {Error} error - The original Axios error
   * @param {string} contextMessage - Context describing what operation was being performed
   * @returns {Object} Standardized error object
   * @private
   */
  _createStandardizedError(
    error,
    contextMessage = 'interacting with the API',
    { isTextProcessing = false, extractionBackend = null } = {}
  ) {
    const responseData = error.response?.data;
    const responseDetail = responseData?.detail;
    const quotaData =
      responseDetail && typeof responseDetail === 'object' ? responseDetail : responseData;
    const standardError = {
      status: error.response?.status || 0,
      type: 'UNKNOWN_ERROR',
      userMessageKey: 'errors.api.unknown', // Default i18n key
      userMessageParams: {},
      originalErrorDetails: {
        message: error.message,
        code: error.code,
        apiResponseMessage:
          typeof responseDetail === 'string'
            ? responseDetail
            : responseDetail?.error_message || responseDetail?.message,
        apiResponseData: responseData,
        configUrl: error.config?.url,
      },
    };

    if (error.isAxiosError && !error.response) {
      standardError.type = 'NETWORK_ERROR';
      standardError.userMessageKey = 'errors.api.network';
    } else if (error.response) {
      standardError.type = 'API_ERROR';
      const { key, params } = this._getErrorMessageKeyForStatus(
        error.response.status,
        typeof responseDetail === 'string'
          ? responseDetail
          : responseDetail?.error_message || responseDetail?.message || ''
      );
      standardError.userMessageKey = key;
      standardError.userMessageParams = params;
      if (
        error.response.status === 429 &&
        isTextProcessing &&
        extractionBackend === 'llm' &&
        this._hasQuotaDetails(quotaData)
      ) {
        standardError.userMessageKey = 'errors.api.llmQuotaExceeded';
        standardError.userMessageParams = {
          quotaRemaining: quotaData?.quota_remaining,
          quotaLimit: quotaData?.quota_limit,
          extractionBackend: quotaData?.extraction_backend ?? extractionBackend,
        };
        standardError.quotaRemaining = quotaData?.quota_remaining;
        standardError.quotaLimit = quotaData?.quota_limit;
        standardError.extractionBackend = quotaData?.extraction_backend ?? extractionBackend;
      }
    } else {
      // Generic client-side error or unexpected issue
      standardError.userMessageKey = 'errors.api.clientSide';
      standardError.userMessageParams = { context: contextMessage };
    }
    return standardError;
  }

  _hasQuotaDetails(quotaData) {
    if (!quotaData || typeof quotaData !== 'object') {
      return false;
    }

    return (
      quotaData.quota_remaining !== undefined ||
      quotaData.quota_limit !== undefined ||
      quotaData.quota_used !== undefined ||
      quotaData.usage_date_utc !== undefined
    );
  }

  /**
   * Maps HTTP status codes to user-friendly i18n message keys
   * @param {number} status - HTTP status code
   * @param {string} detail - Additional error details from API
   * @returns {Object} Object with key and params properties
   * @private
   */
  _getErrorMessageKeyForStatus(status, detail = '') {
    let key = 'errors.api.unknown';
    let params = { status, detail: detail || 'No additional details.' };

    switch (status) {
      case 400:
        key = 'errors.api.badRequest';
        // Optionally, try to parse detail for more specific messages
        if (detail && detail.toLowerCase().includes('model')) key = 'errors.api.badRequestModel';
        break;
      case 401:
        key = 'errors.api.unauthorized';
        break;
      case 403:
        key = 'errors.api.forbidden';
        break;
      case 404:
        key = 'errors.api.notFound';
        if (detail && detail.toLowerCase().includes('model')) key = 'errors.api.notFoundModel';
        break;
      case 500:
        key = 'errors.api.serverError';
        break;
      case 503:
        key = 'errors.api.serviceUnavailable';
        break;
      // Add more specific cases as needed
    }
    return { key, params };
  }
}

export default new PhentrieveService();
