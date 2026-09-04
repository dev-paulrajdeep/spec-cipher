class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl.replace(/\/+$/, '');
    }

    async _fetchWithTimeout(url, options = {}, timeoutMs = CONFIG.TIMEOUT_MS) {
        const controller = new AbortController();

        const timeout = setTimeout(() => {
            controller.abort();
        }, timeoutMs);

        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });

            if (!response.ok) {
                let message = `Request failed with status ${response.status}`;

                try {
                    const errorData = await response.json();

                    if (errorData?.message) {
                        message = errorData.message;
                    } else if (errorData?.error) {
                        message = errorData.error;
                    }
                } catch {
                    // Response JSON chilo na
                }

                throw new Error(message);
            }

            const contentType =
                response.headers.get('content-type') || '';

            if (contentType.includes('application/json')) {
                return await response.json();
            }

            return await response.text();

        } catch (error) {
            if (error.name === 'AbortError') {
                throw new Error(
                    'Request timed out. Please try again.'
                );
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
                'Accept': 'application/json, text/plain'
            },

            body: JSON.stringify(payload)
        });
    }

    async searchStandards(queryPayload) {
        const url =
            `${this.baseUrl}${CONFIG.ENDPOINTS.SEARCH}`;

        return await this._fetchWithTimeout(url, {
            method: 'POST',

            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },

            body: JSON.stringify(queryPayload)
        });
    }

    async checkHealth() {
        const url =
            `${this.baseUrl}${CONFIG.ENDPOINTS.HEALTH}`;

        return await this._fetchWithTimeout(url, {
            method: 'GET',

            headers: {
                'Accept': 'application/json, text/plain'
            }
        });
    }
}

window.apiClient =
    new ApiClient(CONFIG.API_BASE_URL);
