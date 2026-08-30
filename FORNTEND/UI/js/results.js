class DetailsManager {
  constructor() {
    this.currentStandard = null;
  }

  openStandardDetails(standard) {
    if (!standard) return;

    this.currentStandard = standard;

    // Standard er basic information
    const idEl = document.getElementById('drawer-standard-id');
    const titleEl = document.getElementById('drawer-standard-title');
    const statusEl = document.getElementById('drawer-status-badge');
    const versionEl = document.getElementById('drawer-version-tag');

    if (idEl) {
      idEl.textContent = standard.standard_id || 'Indian Standard';
    }

    if (titleEl) {
      titleEl.textContent = standard.title || 'Standard Specification';
    }

    if (statusEl) {
      const statusText = standard.status || 'Active';

      statusEl.textContent = statusText;
      statusEl.className =
        `status-badge ${statusText.toLowerCase()}`;
    }

    if (versionEl) {
      if (standard.version) {
        versionEl.textContent = `v${standard.version}`;
        versionEl.style.display = 'inline-flex';
      } else {
        versionEl.style.display = 'none';
      }
    }

    // Relevance, confidence aar evidence count
    const scoreValEl =
      document.getElementById('drawer-score-val');

    const confValEl =
      document.getElementById('drawer-conf-val');

    const clauseCountEl =
      document.getElementById('drawer-clauses-count');

    if (scoreValEl) {
      let scoreText = '—';

      if (typeof standard.relevance_score === 'number') {
        scoreText =
          `${Math.round(standard.relevance_score * 100)}%`;
      } else if (standard.relevance_score != null) {
        scoreText = String(standard.relevance_score);
      }

      scoreValEl.textContent = scoreText;
    }

    if (confValEl) {
      const confidence = standard.confidence || 'Not available';

      confValEl.textContent = confidence;

      confValEl.className =
        `confidence-indicator ${confidence.toLowerCase()}`;
    }

    if (clauseCountEl) {
      const evidenceCount = Array.isArray(standard.evidence)
        ? standard.evidence.length
        : 0;

      clauseCountEl.textContent =
        evidenceCount > 0
          ? `${evidenceCount} Cited`
          : 'None available';
    }

    // Standard er scope
    const scopeBox =
      document.getElementById('drawer-scope-content');

    if (scopeBox) {
      const scope =
        standard.scope || standard.description;

      if (scope) {
        scopeBox.textContent = scope;
        scopeBox.style.fontStyle = 'normal';
      } else {
        scopeBox.textContent =
          'Scope information is not available.';

        scopeBox.style.fontStyle = 'italic';
      }
    }

    // Evidence show korchi
    const evidenceList =
      document.getElementById('drawer-evidence-list');

    if (evidenceList) {
      evidenceList.innerHTML = '';

      if (
        Array.isArray(standard.evidence) &&
        standard.evidence.length > 0
      ) {
        standard.evidence.forEach((ev) => {
          const item =
            document.createElement('div');

          item.className = 'evidence-quote-item';

          const clauseRef =
            ev.clause ||
            ev.reference ||
            ev.section ||
            'General';

          const srcType =
            ev.source_type ||
            'Standard source';

          const quoteText =
            ev.text ||
            ev.content ||
            ev.excerpt ||
            (
              typeof ev === 'string'
                ? ev
                : ''
            );

          item.innerHTML = `
            <div class="evidence-meta-header">
              <span class="evidence-clause-ref">
                ${window.ui.escapeHtml(clauseRef)}
              </span>

              <span style="color:var(--text-tertiary);">
                ${window.ui.escapeHtml(srcType)}
              </span>
            </div>

            <div class="evidence-body-text">
              ${window.ui.escapeHtml(quoteText)}
            </div>
          `;

          evidenceList.appendChild(item);
        });
      } else {
        evidenceList.innerHTML = `
          <div class="empty-state-box"
               style="padding:1.5rem 1rem; border-style:dashed;">
            <div class="empty-state-desc">
              Supporting evidence is not available.
            </div>
          </div>
        `;
      }
    }

    // Kon requirement gulo match koreche
    const matchedList =
      document.getElementById('drawer-matched-list');

    if (matchedList) {
      matchedList.innerHTML = '';

      if (
        Array.isArray(standard.matched_requirements) &&
        standard.matched_requirements.length > 0
      ) {
        standard.matched_requirements.forEach((requirement) => {
          const tag =
            document.createElement('span');

          tag.className = 'matched-tag';

          const icon = `
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2.5">
              <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
          `;

          tag.innerHTML = `
            ${icon}
            ${window.ui.escapeHtml(requirement)}
          `;

          matchedList.appendChild(tag);
        });
      } else {
        matchedList.innerHTML = `
          <span style="color:var(--text-tertiary); font-size:0.84rem;">
            No specific matches were returned.
          </span>
        `;
      }
    }

    // Related standards graph
    if (window.graphVisualizer) {
      window.graphVisualizer.renderGraph(standard);
    }

    // Details panel open korchi
    window.ui.openDrawer();
  }
}

window.detailsManager = new DetailsManager();