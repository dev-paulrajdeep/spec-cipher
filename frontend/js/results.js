class ResultsManager {
    constructor() {
        this.currentResponse = null;
        this.currentRecommendations = [];

        this.init();
    }

    init() {
        /*
         * Refine button goes back to input screen.
         */
        const refineBtn =
            document.getElementById('btn-refine-spec');

        if (refineBtn) {
            refineBtn.addEventListener('click', () => {
                window.ui.switchView('new-analysis');
            });
        }
    }

    renderResults(response, meta = {}) {
        this.currentResponse = response;
        this.currentRecommendations =
            Array.isArray(response?.recommendations)
                ? response.recommendations
                : [];

        /*
         * Hide "No analysis yet"
         */
        const emptyState =
            document.getElementById(
                'results-empty-state'
            );

        if (emptyState) {
            emptyState.style.display = 'none';
        }

        /*
         * Show results container
         */
        const resultsArea =
            document.getElementById(
                'results-content-area'
            );

        if (resultsArea) {
            resultsArea.style.display = 'flex';
        }

        /*
         * Show the user's original sentence
         */
        this.renderQuerySummary(
            response,
            meta
        );

        /*
         * Show what was received
         */
        this.renderUnderstanding(
            response
        );

        /*
         * Show backend warnings
         */
        this.renderWarnings(
            [
                ...(Array.isArray(response?.warnings) ? response.warnings : []),
                ...this.currentRecommendations.flatMap((recommendation) =>
                    Array.isArray(recommendation?.warnings)
                        ? recommendation.warnings
                        : []
                )
            ]
        );

        /*
         * Render retrieved standards
         */
        this.renderRecommendations(
            this.currentRecommendations
        );
    }

    renderQuerySummary(response, meta) {
        const timestampEl =
            document.getElementById(
                'result-query-timestamp'
            );

        const previewEl =
            document.getElementById(
                'result-query-preview'
            );

        const query =
            meta.querySummary ||
            response?.user_input ||
            'Procurement requirement';

        if (timestampEl) {
            const date =
                meta.timestamp
                    ? new Date(meta.timestamp)
                    : new Date();

            timestampEl.textContent =
                `Analysis • ${date.toLocaleString()}`;
        }

        if (previewEl) {
            previewEl.textContent = query;
        }
    }

    renderUnderstanding(response) {
        const grid =
            document.getElementById(
                'understood-specs-grid'
            );

        if (!grid) {
            return;
        }

        const userInput =
            response?.user_input ||
            '';

        const recommendations =
            Array.isArray(response?.recommendations)
                ? response.recommendations
                : [];

        /*
         * There is no requirement extraction object
         * in the current backend response.
         *
         * So we only display information that
         * actually exists.
         */
        grid.innerHTML = `
            <div class="spec-parameter-cell">
                <div class="param-key">
                    User requirement
                </div>

                <div class="param-val">
                    ${window.ui.escapeHtml(userInput)}
                </div>
            </div>

            <div class="spec-parameter-cell">
                <div class="param-key">
                    Recommended standards
                </div>

                <div class="param-val">
                    ${recommendations.length}
                </div>
            </div>
        `;

        /*
         * Current backend does not return
         * "missing information", so hide
         * that warning.
         */
        const missingInfoBox =
            document.getElementById(
                'missing-info-box'
            );

        if (missingInfoBox) {
            missingInfoBox.style.display = 'none';
        }
    }

    renderWarnings(warnings) {
        const warningBox =
            document.getElementById(
                'results-global-warnings'
            );

        if (!warningBox) {
            return;
        }

        if (!Array.isArray(warnings) ||
            warnings.length === 0) {

            warningBox.style.display =
                'none';

            warningBox.textContent = '';

            return;
        }

        warningBox.innerHTML = warnings
            .map((warning) => {
                return `
                    <div>
                        ${window.ui.escapeHtml(warning)}
                    </div>
                `;
            })
            .join('');

        warningBox.style.display =
            'block';
    }

    renderRecommendations(recommendations) {
        const list =
            document.getElementById(
                'ranked-standards-list'
            );

        const countText =
            document.getElementById(
                'recommendations-count-text'
            );

        if (!list) {
            return;
        }

        list.innerHTML = '';

        const items =
            Array.isArray(recommendations)
                ? recommendations
                : [];

        if (countText) {
            countText.textContent =
                items.length === 0
                    ? 'No recommended standards found'
                    : `${items.length} recommended standard${items.length === 1 ? '' : 's'}`;
        }

        if (items.length === 0) {
            list.innerHTML = `
                <div class="empty-state-box"
                     style="padding:1.5rem;">
                    <div class="empty-state-title">
                        No recommended standards found
                    </div>

                    <div class="empty-state-desc">
                        We could not identify a sufficiently relevant
                        Indian Standard from the available evidence.
                    </div>
                </div>
            `;

            return;
        }

        items.forEach((standard, index) => {
            const card =
                this.createRecommendationCard(
                    standard,
                    index
                );

            list.appendChild(card);
        });
    }

    createRecommendationCard(standard, index) {
        const card = document.createElement('div');
        card.className = 'standard-record-card';

        const rank = standard?.rank ?? index + 1;
        const standardNumber = standard?.standard_number || 'Indian Standard not identified';
        const title = standard?.title || 'Standard title not available';
        const evidence = standard?.evidence?.[0] || {};
        const sectionNumber = evidence.section_number || '';
        const sectionTitle = evidence.section_title || '';
        const text = evidence.text || 'Source text is not available.';
        const score = this.formatScore(standard?.final_score ?? standard?.score);
        const sourceFile = evidence.source_file || 'Source unavailable';
        const pages = this.formatPages(evidence.page_start, evidence.page_end);
        const preview = this.createPreview(text);
        const explanation = standard?.explanation;
        const explanationSummary = explanation && typeof explanation === 'object'
            ? explanation.summary
            : typeof explanation === 'string' ? explanation : '';
        const relatedStandards = Array.isArray(standard?.related_standards)
            ? standard.related_standards
            : [];

        const explanationMarkup = explanationSummary
            ? `
                <div data-testid="recommendation-explanation">
                    <div class="reason-box-title">Explanation</div>
                    <div class="recommendation-copy">${window.ui.escapeHtml(explanationSummary)}</div>
                </div>
            `
            : '';

        const relatedMarkup = relatedStandards.length > 0
            ? `
                <div data-testid="related-information">
                    <div class="reason-box-title">Related information</div>
                    <ul class="reason-bullets">
                        ${relatedStandards.map((related) => `<li>${window.ui.escapeHtml(related.standard_number || 'Related standard')}</li>`).join('')}
                    </ul>
                </div>
            `
            : '';

        card.innerHTML = `
            <div class="record-main-row">
                <div class="rank-badge">#${window.ui.escapeHtml(rank)}</div>
                <div class="standard-info-col">
                    <div class="standard-code-row">
                        <span class="standard-id-tag">${window.ui.escapeHtml(standardNumber)}</span>
                        <span class="status-badge active">Recommended</span>
                    </div>
                    <div class="standard-title-text">${window.ui.escapeHtml(title)}</div>
                </div>
                <div class="record-metrics-col">
                    <div class="metric-item">
                        <span class="metric-label">Relevance</span>
                        <span class="metric-score-val">${window.ui.escapeHtml(score)}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Rank</span>
                        <span class="metric-score-val">#${window.ui.escapeHtml(rank)}</span>
                    </div>
                </div>
                <div class="record-actions-col">
                    <button class="btn btn-outline btn-sm" type="button" data-action="details">
                        View Source Text
                    </button>
                </div>
            </div>

            <div class="record-reasoning-panel">
                <div>
                    <div class="reason-box-title">Source evidence</div>
                    <div class="recommendation-copy">
                        ${window.ui.escapeHtml(sourceFile)}
                        ${pages !== '—' ? ` · Page ${window.ui.escapeHtml(pages)}` : ''}
                        ${sectionTitle ? `<div>${window.ui.escapeHtml(sectionTitle)}</div>` : ''}
                    </div>
                </div>
                <div>
                    <div class="reason-box-title">Extracted content</div>
                    <div class="recommendation-copy">${window.ui.escapeHtml(preview)}</div>
                </div>
                ${sectionTitle ? `
                    <div>
                        <div class="reason-box-title">Context</div>
                        <div class="recommendation-copy">${window.ui.escapeHtml(sectionNumber ? `${sectionNumber} — ${sectionTitle}` : sectionTitle)}</div>
                    </div>
                ` : ''}
                ${explanationMarkup}
                ${relatedMarkup}
            </div>
        `;

        const detailsButton =
            card.querySelector(
                '[data-action="details"]'
            );

        if (detailsButton) {
            detailsButton.addEventListener(
                'click',
                () => {
                    if (
                        window.detailsManager &&
                        typeof window.detailsManager
                            .openStandardDetails ===
                            'function'
                    ) {
                        window.detailsManager
                            .openStandardDetails(
                                standard
                            );
                    }
                }
            );
        }

        return card;
    }

    formatScore(score) {
        if (
            typeof score !== 'number' ||
            Number.isNaN(score)
        ) {
            return '—';
        }

        /*
         * Backend score is expected to be 0..1.
         */
        if (score >= 0 && score <= 1) {
            return `${Math.round(score * 100)}%`;
        }

        return String(score);
    }

    formatPages(start, end) {
        if (
            typeof start !== 'number' &&
            typeof end !== 'number'
        ) {
            return '—';
        }

        if (
            typeof start === 'number' &&
            typeof end === 'number'
        ) {
            if (start === end) {
                return String(start);
            }

            return `${start}–${end}`;
        }

        return String(
            typeof start === 'number'
                ? start
                : end
        );
    }

    createPreview(text) {
        const safeText =
            String(text || '');

        if (safeText.length <= 280) {
            return safeText;
        }

        return (
            safeText.substring(0, 280) +
            '...'
        );
    }
}

window.resultsManager =
    new ResultsManager();
