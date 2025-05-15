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
            const errorDetail = error.response?.data?.detail || "An unknown error occurred while fetching results.";
            logService.error('Error querying HPO API', {
                status: error.response?.status,
                detail: errorDetail,
                error: error.message
            });
            // Rethrow a simplified error object for the component to handle
            throw { 
                status: error.response?.status,
                detail: errorDetail
            };
        }
    }
}

export default new PhentrieveService();
