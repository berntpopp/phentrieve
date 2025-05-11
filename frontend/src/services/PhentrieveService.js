import axios from 'axios';

// Use import.meta.env for Vite instead of process.env
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001/api/v1';

class PhentrieveService {
    async queryHpo(queryData) {
        // queryData should match QueryRequest schema from FastAPI
        // Example: { text: "...", model_name: "...", num_results: 10, ... }
        try {
            const response = await axios.post(`${API_URL}/query/`, queryData);
            return response.data; // Expected to match QueryResponse schema
        } catch (error) {
            console.error("Error querying HPO:", error.response?.data || error.message);
            // Rethrow a simplified error object for the component to handle
            throw { 
                status: error.response?.status,
                detail: error.response?.data?.detail || "An unknown error occurred while fetching results."
            };
        }
    }
}

export default new PhentrieveService();
