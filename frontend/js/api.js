class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl.replace(/\/+$/, '');
    }

    async _fetchWithTimeout(
        url,
        options = {},
        timeoutMs = CONFIG.TIMEOUT_MS
    ) {
        const controller = new AbortController();

        const timeout = setTimeout(() => {
            controller.abort();
        }, timeoutMs);

        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });

            const contentType =
                response.headers.get('content-type') || '';
            let payload = null;
            if (contentType.includes('application/json')) {
                try {
                    payload = await response.json();
                } catch {
                    payload = null;
                }
            }

            if (!response.ok) {
                let message =
                    `Request failed with status ${response.status}`;
                if (payload?.message) {
                    message = payload.message;
                } else if (payload?.error) {
                    message = payload.error;
                }
                const error = new Error(message);
                error.resultState = payload?.result_state || null;
                error.status = response.status;
                throw error;
            }

            if (payload !== null) {
                return payload;
            }

            return await response.text();

        } catch (error) {
            if (error.name === 'AbortError') {
                const timeoutError = new Error(
                    'Request timed out. Please try again.'
                );
                timeoutError.resultState = 'dependency_unavailable';
                throw timeoutError;
            }

            throw error;

        } finally {
            clearTimeout(timeout);
        }
    }

    async analyzeSpecification(userInput) {
        const url =
            `${this.baseUrl}${CONFIG.ENDPOINTS.ANALYZE}`;

        const payload = {
            user_input: userInput
        };

        return await this._fetchWithTimeout(url, {
            method: 'POST',

            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },

            body: JSON.stringify(payload)
        });
    }
}

window.apiClient =
    new ApiClient(CONFIG.API_BASE_URL);
