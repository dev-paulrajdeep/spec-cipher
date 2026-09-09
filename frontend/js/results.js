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
                    .filter((recommendation) => recommendation?.is_code || recommendation?.standard_number)
                    .slice(0, 5)
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
                    Retrieved matches
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
                `${items.length} recommendation${items.length === 1 ? '' : 's'}`;
        }

        if (items.length === 0) {
            list.innerHTML = `
                <div class="empty-state-box"
                     style="padding:1.5rem;">
                    <div class="empty-state-title">
                        No matches found
                    </div>

                    <div class="empty-state-desc">
                        The retrieval system did not return
                        any matching document chunks.
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
        const card =
            document.createElement('div');

        card.className =
            'standard-record-card';

        const rank =
            standard?.rank ??
            index + 1;

        const standardNumber =
            standard?.is_code ||
            standard?.standard_number ||
            'Indian Standard not identified';

        const title =
            standard?.title ||
            standard?.section_title ||
            'Standard title not available';

        const sectionNumber =
            standard?.section_number ||
            '—';

        const sectionTitle =
            standard?.section_title ||
            standard?.evidence?.[0]?.section_title ||
            'Section not available';

        const text =
            standard?.text ||
            standard?.evidence?.[0]?.text ||
            'No retrieved text available.';

        const score =
            this.formatScore(
                standard?.score ??
                standard?.final_score
            );

        const evidence =
            standard?.evidence?.[0] || {};

        const sourceFile =
            standard?.source?.source_file ||
            evidence.source_file ||
            'Source unavailable';

        const pageStart =
            standard?.source?.page_start ??
            evidence.page_start;

        const pageEnd =
            standard?.source?.page_end ??
            evidence.page_end;

        const pages =
            this.formatPages(
                pageStart,
                pageEnd
            );

        const preview =
            this.createPreview(text);

        const explanation =
            standard?.explanation;

        const relatedStandards =
            Array.isArray(standard?.related_standards)
                ? standard.related_standards
                : [];

        const explanationMarkup =
            explanation && typeof explanation === 'object'
                ? `
                    <div data-testid="recommendation-explanation">
                        <div class="reason-box-title">
                            Explanation
                        </div>

                        <div style="color:var(--text-secondary); line-height:1.55;">
                            ${window.ui.escapeHtml(explanation.summary || '')}
                        </div>

                        ${Array.isArray(explanation.key_points) && explanation.key_points.length > 0 ? `
                            <ul class="reason-bullets">
                                ${explanation.key_points.map((point) => `<li>${window.ui.escapeHtml(point)}</li>`).join('')}
                            </ul>
                        ` : ''}

                        ${Array.isArray(explanation.caveats) && explanation.caveats.length > 0 ? `
                            <div data-testid="recommendation-caveats">
                                <div class="reason-box-title">
                                    Caveats
                                </div>
                                <ul class="reason-bullets">
                                    ${explanation.caveats.map((caveat) => `<li>${window.ui.escapeHtml(caveat)}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                    </div>
                `
                : '';

        const relatedMarkup =
            relatedStandards.length > 0
                ? `
                    <div data-testid="related-standards">
                        <div class="reason-box-title">
                            Related standards
                        </div>
                        <ul class="reason-bullets">
                            ${relatedStandards.map((related) => `<li>${window.ui.escapeHtml(related.standard_number || 'Related standard')}</li>`).join('')}
                        </ul>
                    </div>
                `
                : '';

        card.innerHTML = `
            <div class="record-main-row">

                <div class="rank-badge">
                    #${window.ui.escapeHtml(rank)}
                </div>

                <div class="standard-info-col">

                    <div class="standard-code-row">

                        <span class="standard-id-tag">
                            ${window.ui.escapeHtml(standardNumber)}
                        </span>

                        <span class="status-badge active">
                            Retrieved
                        </span>

                    </div>

                    <div class="standard-title-text">
                        ${window.ui.escapeHtml(title)}
                    </div>

                    <div class="standard-meta-chips">

                        <span>
                            Section:
                            ${window.ui.escapeHtml(sectionNumber)}
                        </span>

                        <span>
                            ${window.ui.escapeHtml(sectionTitle)}
                        </span>

                    </div>

                </div>

                <div class="record-metrics-col">

                    <div class="metric-item">
                        <span class="metric-label">
                            Similarity
                        </span>

                        <span class="metric-score-val">
                            ${window.ui.escapeHtml(score)}
                        </span>
                    </div>

                    <div class="metric-item">
                        <span class="metric-label">
                            Rank
                        </span>

                        <span class="metric-score-val">
                            #${window.ui.escapeHtml(rank)}
                        </span>
                    </div>

                </div>

                <div class="record-actions-col">

                    <button
                        class="btn btn-outline btn-sm"
                        type="button"
                        data-action="details">
                        View Retrieved Text
                    </button>

                </div>

            </div>

            <div class="record-reasoning-panel">

                <div>

                    <div class="reason-box-title">
                        Retrieved content
                    </div>

                    <div style="
                        color:var(--text-secondary);
                        line-height:1.55;
                    ">
                        ${window.ui.escapeHtml(preview)}
                    </div>

                </div>

                <div>

                    <div class="reason-box-title">
                        Source
                    </div>

                    <div style="
                        color:var(--text-secondary);
                        line-height:1.55;
                    ">
                        <div>
                            ${window.ui.escapeHtml(sourceFile)}
                        </div>

                        <div>
                            Pages:
                            ${window.ui.escapeHtml(pages)}
                        </div>

                    </div>

                </div>

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
