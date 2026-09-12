const CONFIG = {
    API_BASE_URL: '',

    ENDPOINTS: {
        ANALYZE: '/get_result'
    },

    // Allow cold Ollama model load + retrieval under keep_alive=0.
    TIMEOUT_MS: 240000
};

window.CONFIG = CONFIG;
