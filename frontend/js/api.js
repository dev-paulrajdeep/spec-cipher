class ApiClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  // Ekta fetch request ke timeout er moddhe rakha
  async _fetchWithTimeout(
    url,
    options = {},
    timeoutMs = CONFIG.TIMEOUT_MS
  ) {
    const controller = new AbortController();

    const timeoutId = setTimeout(() => {
      controller.abort();
    }, timeoutMs);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      // Server theke error ele handle korchi
      if (!response.ok) {
        let errorData = null;

        try {
          errorData = await response.json();
        } catch {
          // JSON na hole kichu korar nei
        }

        const message =
          errorData?.error ||
          errorData?.message ||
          `Server returned HTTP ${response.status} (${response.statusText})`;

        const error = new Error(message);
        error.status = response.status;
        error.data = errorData;

        throw error;
      }

      return await response.json();

    } catch (err) {
      clearTimeout(timeoutId);

      // Request beshi time nile
      if (err.name === 'AbortError') {
        throw new Error(
          `Request timed out after ${timeoutMs / 1000}s. Please check if the Flask server is running at ${this.baseUrl}`
        );
      }

      throw err;
    }
  }

  // Flask server cholche kina check korar jonno
  async checkHealth() {
    try {
      const url =
        `${this.baseUrl}${CONFIG.ENDPOINTS.HEALTH}`;

      const res = await this._fetchWithTimeout(
        url,
        { method: 'GET' },
        4000
      );

      return {
        online: true,
        data: res
      };

    } catch (err) {
      return {
        online: false,
        error: err.message
      };
    }
  }

  // Procurement specification Flask e pathanor function
  async analyzeSpecification(payload) {
    const url =
      `${this.baseUrl}${CONFIG.ENDPOINTS.ANALYZE}`;

    const options = {
      method: 'POST'
    };

    // File thakle FormData use hobe
    if (payload instanceof FormData) {
      options.body = payload;

    } else {
      // Normal text input hole JSON pathabo
      options.headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      };

      options.body = JSON.stringify(payload);
    }

    return await this._fetchWithTimeout(
      url,
      options
    );
  }

  // Standards search korar function
  async searchStandards(queryPayload) {
    const url =
      `${this.baseUrl}${CONFIG.ENDPOINTS.SEARCH}`;

    return await this._fetchWithTimeout(
      url,
      {
        method: 'POST',

        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },

        body: JSON.stringify(queryPayload)
      }
    );
  }
}

window.apiClient =
  new ApiClient(CONFIG.API_BASE_URL);