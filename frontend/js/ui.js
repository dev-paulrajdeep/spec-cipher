class UIController {
  constructor() {
    this.currentView = 'dashboard';
    this.views = [
      'dashboard',
      'new-analysis',
      'results',
      'search',
      'compare',
      'history',
      'about'
    ];
    this.toastsContainer = null;
    this.init();
  }

  init() {
    // Toast container ready korchi
    let tc = document.getElementById('toast-container');

    if (!tc) {
      tc = document.createElement('div');
      tc.id = 'toast-container';
      tc.className = 'toast-container';
      document.body.appendChild(tc);
    }

    this.toastsContainer = tc;

    // Navbar / button click handle korchi
    document.querySelectorAll('[data-view-target]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const target =
          e.currentTarget.getAttribute('data-view-target');

        if (target) {
          this.switchView(target);
        }
      });
    });

    // Drawer er baire click korle close hobe
    const overlay =
      document.getElementById('standard-details-drawer');

    if (overlay) {
      overlay.addEventListener('click', (e) => {
        if (e.target === overlay) {
          this.closeDrawer();
        }
      });
    }

    // Close button
    document.querySelectorAll('.drawer-close-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        this.closeDrawer();
      });
    });

    // ESC dile drawer close hobe
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        this.closeDrawer();
      }
    });
  }

  switchView(viewName) {
    if (!this.views.includes(viewName)) return;

    this.currentView = viewName;

    // Active navigation update korchi
    document.querySelectorAll('.nav-btn').forEach((btn) => {
      const target =
        btn.getAttribute('data-view-target');

      if (target === viewName) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    // Sob view hide kore active view dekhacchi
    document.querySelectorAll('.view-section').forEach((section) => {
      section.classList.remove('active');
    });

    const activeSection =
      document.getElementById(`view-${viewName}`);

    if (activeSection) {
      activeSection.classList.add('active');

      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    }

    // History page khulle history update hobe
    if (
      viewName === 'history' &&
      window.historyManager
    ) {
      window.historyManager.renderHistoryList();
    }

    // Compare page khulle selected standards dekhabe
    if (
      viewName === 'compare' &&
      window.compareManager
    ) {
      window.compareManager.renderCompareView();
    }
  }

  openDrawer() {
    const drawer =
      document.getElementById('standard-details-drawer');

    if (drawer) {
      drawer.classList.add('active');
      document.body.style.overflow = 'hidden';
    }
  }

  closeDrawer() {
    const drawer =
      document.getElementById('standard-details-drawer');

    if (drawer) {
      drawer.classList.remove('active');
      document.body.style.overflow = '';
    }
  }

  showToast(message, type = 'info', durationMs = 4000) {
    const toast =
      document.createElement('div');

    toast.className = `toast toast-${type}`;

    let iconSvg = '';

    // Type er upor icon change hobe
    if (type === 'error') {
      iconSvg = `
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="15" y1="9" x2="9" y2="15"></line>
          <line x1="9" y1="9" x2="15" y2="15"></line>
        </svg>
      `;
    } else if (type === 'success') {
      iconSvg = `
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2">
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
      `;
    } else if (type === 'warning') {
      iconSvg = `
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
          <line x1="12" y1="9" x2="12" y2="13"></line>
          <line x1="12" y1="17" x2="12.01" y2="17"></line>
        </svg>
      `;
    } else {
      iconSvg = `
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="16" x2="12" y2="12"></line>
          <line x1="12" y1="8" x2="12.01" y2="8"></line>
        </svg>
      `;
    }

    toast.innerHTML = `
      ${iconSvg}
      <span style="flex:1;">
        ${this.escapeHtml(message)}
      </span>
    `;

    this.toastsContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.2s ease';

      setTimeout(() => {
        toast.remove();
      }, 250);
    }, durationMs);
  }

  updateCompareCount(count) {
    const badge =
      document.getElementById('nav-compare-count');

    if (!badge) return;

    badge.textContent = count;
    badge.style.display =
      count > 0 ? 'inline-block' : 'none';
  }

  escapeHtml(str) {
    if (str === null || str === undefined) {
      return '';
    }

    const div = document.createElement('div');
    div.textContent = String(str);

    return div.innerHTML;
  }
}

window.ui = new UIController();