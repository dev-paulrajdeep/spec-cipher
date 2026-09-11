class DetailsManager {
    constructor() {
        this.currentStandard = null;
    }

    openStandardDetails(standard) {
        if (!standard) {
            return;
        }

        this.currentStandard =
            standard;

        this.renderHeader(
            standard
        );

        this.renderMeta(
            standard
        );

        this.renderSection(
            standard
        );

        this.renderSource(
            standard
        );

        this.renderRetrievedText(
            standard
        );

        /*
         * Current backend does not return
         * relationship data.
         *
         * So we deliberately do not call
         * graph.js here.
         */

        window.ui.openDrawer();
    }

    renderHeader(standard) {
        const idEl =
            document.getElementById(
                'drawer-standard-id'
            );

        const titleEl =
            document.getElementById(
                'drawer-standard-title'
            );

        const statusEl =
            document.getElementById(
                'drawer-status-badge'
            );

        const versionEl =
            document.getElementById(
                'drawer-version-tag'
            );

        const standardNumber =
            standard.standard_number ||
            'Indian Standard';

        const title =
            standard.title ||
            standard.source_evidence?.document_title ||
            standard.section_title ||
            'Retrieved standard match';

        if (idEl) {
            idEl.textContent =
                standardNumber;
        }

        if (titleEl) {
            titleEl.textContent =
                title;
        }

        /*
         * "Retrieved" is a UI state, NOT
         * BIS applicability/status.
         */
        if (statusEl) {
            statusEl.textContent =
                'Retrieved';

            statusEl.className =
                'status-badge active';
        }

        /*
         * Backend does not send version.
         */
        if (versionEl) {
            versionEl.textContent = '';
            versionEl.style.display =
                'none';
        }
    }

    renderMeta(standard) {
        const scoreEl =
            document.getElementById(
                'drawer-score-val'
            );

        const rankEl =
            document.getElementById(
                'drawer-conf-val'
            );

        const pagesEl =
            document.getElementById(
                'drawer-clauses-count'
            );

        if (scoreEl) {
            scoreEl.textContent =
                this.formatScore(
                    standard.score
                );
        }

        if (rankEl) {
            rankEl.textContent =
                standard.rank != null
                    ? `#${standard.rank}`
                    : '—';

            rankEl.className =
                'doc-meta-val';
        }

        if (pagesEl) {
            const source =
                standard.source || {};

            pagesEl.textContent =
                this.formatPages(
                    source.page_start,
                    source.page_end
                );
        }
    }

    renderSection(standard) {
        const sectionBox =
            document.getElementById(
                'drawer-scope-content'
            );

        if (!sectionBox) {
            return;
        }

        const number =
            standard.section_number ||
            '';

        const title =
            standard.section_title ||
            '';

        if (!number && !title) {
            sectionBox.textContent =
                'Section information is not available.';

            sectionBox.style.fontStyle =
                'italic';

            return;
        }

        const sectionParts = [];

        if (number) {
            sectionParts.push(number);
        }

        if (title) {
            sectionParts.push(title);
        }

        sectionBox.textContent =
            sectionParts.join(' — ');

        sectionBox.style.fontStyle =
            'normal';
    }

    renderSource(standard) {
        const source =
            standard.source || {};
        const sourceEvidence =
            standard.source_evidence || {};

        const matchedList =
            document.getElementById(
                'drawer-matched-list'
            );

        if (!matchedList) {
            return;
        }

        matchedList.innerHTML = '';

        const items = [];

        if (source.source_file) {
            items.push(
                `Source file: ${source.source_file}`
            );
        }

        if (sourceEvidence.edition) {
            items.push(`Edition: ${sourceEvidence.edition}`);
        }

        if (sourceEvidence.section_name) {
            items.push(`Section: ${sourceEvidence.section_name}`);
        }

        if (source.chunk_id) {
            items.push(
                `Chunk ID: ${source.chunk_id}`
            );
        }

        if (
            Array.isArray(
                source.element_ids
            ) &&
            source.element_ids.length > 0
        ) {
            items.push(
                `Elements: ${source.element_ids.join(', ')}`
            );
        }

        if (items.length === 0) {
            matchedList.innerHTML = `
                <span style="
                    color:var(--text-tertiary);
                    font-size:0.84rem;
                ">
                    Source information is not available.
                </span>
            `;

            return;
        }

        items.forEach((value) => {
            const tag =
                document.createElement('span');

            tag.className =
                'matched-tag';

            tag.innerHTML = `
                <svg
                    width="12"
                    height="12"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2.5">
                    <polyline
                        points="20 6 9 17 4 12">
                    </polyline>
                </svg>

                ${window.ui.escapeHtml(value)}
            `;

            matchedList.appendChild(tag);
        });
    }

    renderRetrievedText(standard) {
        const evidenceList =
            document.getElementById(
                'drawer-evidence-list'
            );

        if (!evidenceList) {
            return;
        }

        evidenceList.innerHTML = '';

        const text =
            standard.extracted_content ||
            standard.text ||
            '';

        if (!text) {
            evidenceList.innerHTML = `
                <div class="empty-state-box"
                     style="
                        padding:1.5rem 1rem;
                        border-style:dashed;
                     ">

                    <div class="empty-state-desc">
                        Retrieved text is not available.
                    </div>

                </div>
            `;

            return;
        }

        const item =
            document.createElement('div');

        item.className =
            'evidence-quote-item';

        const sectionNumber =
            standard.section_number ||
            'Section';

        const sectionTitle =
            standard.section_title ||
            'Retrieved chunk';

        item.innerHTML = `
            <div class="evidence-meta-header">

                <span class="evidence-clause-ref">
                    ${window.ui.escapeHtml(sectionNumber)}
                </span>

                <span style="
                    color:var(--text-tertiary);
                ">
                    ${window.ui.escapeHtml(sectionTitle)}
                </span>

            </div>

            <div class="evidence-body-text">
                ${window.ui.escapeHtml(text)}
            </div>
        `;

        evidenceList.appendChild(item);
    }

    formatScore(score) {
        if (
            typeof score !== 'number' ||
            Number.isNaN(score)
        ) {
            return '—';
        }

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
}

window.detailsManager =
    new DetailsManager();
