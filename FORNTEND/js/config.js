const CONFIG = {
  API_BASE_URL: window.SPECCIPHER_API_URL || 'http://localhost:5000',

  ENDPOINTS: {
    ANALYZE: '/analyze',
    SEARCH: '/search',
    HEALTH: '/health'
  },

  TIMEOUT_MS: 45000,

  UPLOAD: {
    MAX_SIZE_BYTES: 15 * 1024 * 1024,

    ALLOWED_EXTENSIONS: [
      '.pdf',
      '.txt'
    ],

    ALLOWED_MIME_TYPES: [
      'application/pdf',
      'text/plain'
    ]
  }
};

window.CONFIG = CONFIG;