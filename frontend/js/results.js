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
                response?.analysis_timestamp || meta.timestamp
                    ? new Date(response?.analysis_timestamp || meta.timestamp)
                    : new Date();

            timestampEl.textContent =
                    `Analysis • ${date.toLocaleDateString()} · ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
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
                    ${response?.retrieved_match_count ?? recommendations.length}
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
                `${items.length} recommended standard${items.length === 1 ? '' : 's'}`;
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

        const sourceEvidence = standard?.source_evidence || {};
        const evidence = standard?.evidence?.[0] || {};
        const sectionNumber = sourceEvidence.section_number || evidence.section_number;
        const sectionTitle = sourceEvidence.section_name || evidence.section_title;
        const text = standard?.extracted_content || evidence.text || '';

        const score =
            this.formatScore(
                standard?.score ??
                standard?.final_score
            );

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

        const explanation = standard?.explanation_text || standard?.explanation?.summary || standard?.reason;
        const relatedTopics =
            Array.isArray(standard?.related_topics)
                ? standard.related_topics
                : [];
        const category = standard?.category || {};
        const categoryText = category.code || category.name
            ? `${category.code || ''}${category.code && category.name ? ' · ' : ''}${category.name || ''}`
            : 'Not available in source metadata';
        const titleMarkup = this.highlightTerms(title);
        const relatedMarkup = relatedTopics.length
            ? relatedTopics.map((topic) => `<li>${window.ui.escapeHtml(topic)}</li>`).join('')
            : '<li>No related topics recorded</li>';

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
                            Recommended
                        </span>

                    </div>

                    <div class="standard-title-text">
                        ${titleMarkup}
                    </div>

                    <div class="standard-category-line">
                        Category: ${window.ui.escapeHtml(standard?.category_breadcrumb || categoryText)}
                    </div>

                </div>

                <div class="record-metrics-col">

                    <div class="metric-item">
                        <span class="metric-label">
                            Relevance
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
                        View Source Text
                    </button>

                </div>

            </div>

            <div class="record-reasoning-panel">
                <div class="record-evidence-column">
                    <div class="evidence-block">
                        <div class="reason-box-title">Source evidence</div>
                        <div class="evidence-lines">
                            <div>${window.ui.escapeHtml(sourceEvidence.document_title || title)}</div>
                            <div>${window.ui.escapeHtml(sourceEvidence.edition || 'Edition not available')}</div>
                            <div>${window.ui.escapeHtml(sourceEvidence.part_volume || 'Part/volume not available')}</div>
                            <div>Page ${window.ui.escapeHtml(pages)}</div>
                            <div>${window.ui.escapeHtml(sectionNumber || 'Section not available')}${sectionTitle ? ` · ${window.ui.escapeHtml(sectionTitle)}` : ''}</div>
                        </div>
                    </div>
                    <div class="evidence-block">
                        <div class="reason-box-title">Extracted content</div>
                        <div class="evidence-content">${window.ui.escapeHtml(text || 'Retrieved text is not available.')}</div>
                    </div>
                    <div class="evidence-block">
                        <div class="reason-box-title">Context</div>
                        <div class="evidence-content">${window.ui.escapeHtml(standard?.context || 'Context is not available in source metadata.')}</div>
                    </div>
                    <div class="evidence-block">
                        <div class="reason-box-title">Explanation</div>
                        <div class="evidence-content">${window.ui.escapeHtml(explanation || 'Explanation is not available.')}</div>
                    </div>
                </div>
                <aside class="record-source-column">
                    <div class="evidence-block">
                        <div class="reason-box-title">Source</div>
                        <div class="evidence-content">${window.ui.escapeHtml(sourceFile)}<br/>Page ${window.ui.escapeHtml(pages)}</div>
                    </div>
                    <div class="evidence-block">
                        <div class="reason-box-title">Category (from source)</div>
                        <div class="evidence-content">${window.ui.escapeHtml(categoryText)}</div>
                    </div>
                    <div class="evidence-block" data-testid="related-topics">
                        <div class="reason-box-title">Related topics</div>
                        <ul class="reason-bullets">${relatedMarkup}</ul>
                    </div>
                </aside>

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
            return score.toFixed(3);
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

    highlightTerms(title) {
        const escaped = window.ui.escapeHtml(title || 'Standard title not available');
        const terms = Array.isArray(this.currentResponse?.highlight_terms)
            ? this.currentResponse.highlight_terms.filter(Boolean)
            : [];
        if (!terms.length) {
            return escaped;
        }
        const pattern = terms.map((term) => String(term).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
        return escaped.replace(new RegExp(`(${pattern})`, 'gi'), '<mark>$1</mark>');
    }
}

window.resultsManager =
    new ResultsManager();
