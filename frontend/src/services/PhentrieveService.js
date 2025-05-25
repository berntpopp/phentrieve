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
const API_URL = import.meta.env.VITE_API_URL || '/api/v1';  // Default to relative path for Nginx proxy

class PhentrieveService {
    async queryHpo(queryData) {
        // queryData should match QueryRequest schema from FastAPI
        // Example: { text: "...", model_name: "...", num_results: 10, ... }
        try {
            logService.info('Querying HPO API', { query: queryData });
            const response = await axios.post(`${API_URL}/query/`, queryData);
            logService.debug('HPO API response received', { 
                status: response.status,
                data: response.data
            });
            return response.data; // Expected to match QueryResponse schema
        } catch (error) {
            logService.error('Original API Error in PhentrieveService queryHpo:', {
                message: error.message,
                name: error.name,
                code: error.code,
                config: error.config ? {
                    url: error.config.url,
                    method: error.config.method,
                    timeout: error.config.timeout
                } : null,
                request: error.request ? 'Exists' : 'DoesNotExist', // Avoid logging large objects
                response: error.response ? { 
                    status: error.response.status, 
                    data: error.response.data 
                } : null
            });
            throw this._createStandardizedError(error, 'querying HPO terms');
        }
    }

    /**
     * Creates a standardized error object from an Axios error
     * @param {Error} error - The original Axios error
     * @param {string} contextMessage - Context describing what operation was being performed
     * @returns {Object} Standardized error object
     * @private
     */
    _createStandardizedError(error, contextMessage = 'interacting with the API') {
        const standardError = {
            status: error.response?.status || 0,
            type: 'UNKNOWN_ERROR',
            userMessageKey: 'errors.api.unknown', // Default i18n key
            userMessageParams: {},
            originalErrorDetails: {
                message: error.message,
                code: error.code,
                apiResponseMessage: error.response?.data?.detail,
                configUrl: error.config?.url
            }
        };

        if (error.isAxiosError && !error.response) {
            standardError.type = 'NETWORK_ERROR';
            standardError.userMessageKey = 'errors.api.network';
        } else if (error.response) {
            standardError.type = 'API_ERROR';
            const { key, params } = this._getErrorMessageKeyForStatus(error.response.status, error.response.data?.detail);
            standardError.userMessageKey = key;
            standardError.userMessageParams = params;
        } else {
            // Generic client-side error or unexpected issue
            standardError.userMessageKey = 'errors.api.clientSide';
            standardError.userMessageParams = { context: contextMessage };
        }
        return standardError;
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
