document.addEventListener('DOMContentLoaded', () => {
  console.log('SpecCipher Frontend initialized.');

  // Flask server check
  initApiStatusChecker();

  // URL hash thakle oi page open hobe
  const hash = window.location.hash.replace('#', '');

  if (hash && window.ui?.views.includes(hash)) {
    window.ui.switchView(hash);
  }
});

function initApiStatusChecker() {
  const statusPill =
    document.getElementById('api-status-pill');

  const dot =
    document.getElementById('status-indicator-dot');

  const text =
    document.getElementById('status-indicator-text');

  async function check() {
    if (!window.apiClient) return;

    const res = await window.apiClient.checkHealth();

    if (res.online) {
      if (dot) {
        dot.className = 'status-indicator-dot online';
      }

      if (text) {
        text.textContent = 'API Connected';
      }

      if (statusPill) {
        statusPill.title = 'Flask Backend Connected';
      }

    } else {
      if (dot) {
        dot.className = 'status-indicator-dot offline';
      }

      if (text) {
        text.textContent = 'Backend Standby';
      }

      if (statusPill) {
        statusPill.title = 'Flask Backend Unavailable';
      }
    }
  }

  // Page load hole ekbar check korb0
  check();

  // Tarpor kichu por por abar check korb0
  setInterval(check, 15000);
}