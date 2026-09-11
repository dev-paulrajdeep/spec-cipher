class AnalysisController {
    constructor() {
        this.progressInterval = null;
        this.lastUserInput = '';
        this.init();
    }

    init() {
        this.form =
            document.getElementById('spec-analysis-form');

        this.rawTextInput =
            document.getElementById('spec-raw-text');

        const clearBtn =
            document.getElementById('btn-clear-spec');

        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clearForm();
            });
        }

        const retryBtn =
            document.getElementById('btn-retry-analysis');

        if (retryBtn) {
            retryBtn.addEventListener('click', () => {
                if (this.lastUserInput && this.rawTextInput) {
                    this.rawTextInput.value = this.lastUserInput;
                }
                this.handleSubmit();
            });
        }

        if (this.form) {
            this.form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleSubmit();
            });
        }
    }

    clearForm() {
        this.hideErrorPanel();
        if (this.rawTextInput) {
            this.rawTextInput.value = '';
            this.rawTextInput.focus();
        }
    }

    showErrorPanel(message, resultState) {
        const panel =
            document.getElementById('analysis-error-panel');
        const retryBtn =
            document.getElementById('btn-retry-analysis');
        if (!panel) {
            return;
        }
        const stateLabel = resultState
            ? ` (${window.ui.escapeHtml(resultState)})`
            : '';
        panel.innerHTML = `
            <div><strong>Analysis could not be completed${stateLabel}.</strong></div>
            <div>${window.ui.escapeHtml(message)}</div>
        `;
        panel.style.display = 'block';
        if (retryBtn) {
            retryBtn.style.display = this.lastUserInput ? 'inline' : 'none';
        }
    }

    hideErrorPanel() {
        const panel =
            document.getElementById('analysis-error-panel');
        const retryBtn =
            document.getElementById('btn-retry-analysis');
        if (panel) {
            panel.style.display = 'none';
            panel.textContent = '';
        }
        if (retryBtn) {
            retryBtn.style.display = 'none';
        }
    }

    async handleSubmit() {
        const userInput =
            this.rawTextInput?.value?.trim() || '';

        if (!userInput) {
            this.showErrorPanel(
                'Please enter a procurement requirement.',
                'invalid_request'
            );
            window.ui.showToast(
                'Please enter a procurement requirement.',
                'warning'
            );
            this.rawTextInput?.focus();
            return;
        }

        if (userInput.length > 4000) {
            this.showErrorPanel(
                'Requirement must be at most 4000 characters.',
                'invalid_request'
            );
            return;
        }

        this.lastUserInput = userInput;
        this.hideErrorPanel();

        const submitButton =
            document.getElementById('btn-submit-analysis');

        if (submitButton) {
            submitButton.disabled = true;
            submitButton.textContent = 'Analyzing...';
        }

        this.showProgressModal();

        try {
            const response =
                await window.apiClient
                    .analyzeSpecification(userInput);

            if (
                !response ||
                response.status !== 'success'
            ) {
                const error = new Error(
                    response?.message ||
                    'Invalid response received from backend.'
                );
                error.resultState = response?.result_state;
                throw error;
            }

            this.completeAllProgressSteps();

            setTimeout(() => {
                this.hideProgressModal();

                if (window.resultsManager) {
                    window.resultsManager.renderResults(
                        response,
                        {
                            querySummary: userInput,
                            rawText: userInput,
                            timestamp:
                                new Date().toISOString()
                        }
                    );
                } else {
                    console.error(
                        'resultsManager is not available.'
                    );
                }

                window.ui.switchView('results');

                const toastMessage = response.result_state === 'no_match'
                    ? 'No sufficiently supported standard found.'
                    : 'Analysis completed successfully.';
                window.ui.showToast(
                    toastMessage,
                    response.result_state === 'no_match' ? 'warning' : 'success'
                );

                if (submitButton) {
                    submitButton.disabled = false;
                    submitButton.textContent = 'Analyze';
                }
            }, 300);

        } catch (error) {
            this.hideProgressModal();

            if (submitButton) {
                submitButton.disabled = false;
                submitButton.textContent = 'Analyze';
            }

            const message = error.message || 'Analysis failed.';
            this.showErrorPanel(message, error.resultState || error.result_state);
            window.ui.showToast(
                `Analysis failed: ${message}`,
                'error',
                6000
            );

            console.error(
                'Analysis API Error:',
                error
            );
        }
    }

    showProgressModal() {
        const modal =
            document.getElementById(
                'analysis-progress-modal'
            );

        if (!modal) {
            return;
        }

        modal.classList.add('active');

        const stepItems =
            modal.querySelectorAll(
                '.progress-step-item'
            );

        stepItems.forEach((item) => {
            item.className =
                'progress-step-item pending';
        });

        clearInterval(this.progressInterval);

        let currentStep = 0;

        if (stepItems.length > 0) {
            stepItems[0].className =
                'progress-step-item active';
        }

        this.progressInterval =
            setInterval(() => {
                if (currentStep <
                    stepItems.length - 1) {

                    stepItems[currentStep].className =
                        'progress-step-item completed';

                    currentStep++;

                    stepItems[currentStep].className =
                        'progress-step-item active';
                }
            }, 1200);
    }

    completeAllProgressSteps() {
        clearInterval(this.progressInterval);

        const modal =
            document.getElementById(
                'analysis-progress-modal'
            );

        if (!modal) {
            return;
        }

        const stepItems =
            modal.querySelectorAll(
                '.progress-step-item'
            );

        stepItems.forEach((item) => {
            item.className =
                'progress-step-item completed';
        });
    }

    hideProgressModal() {
        clearInterval(this.progressInterval);

        const modal =
            document.getElementById(
                'analysis-progress-modal'
            );

        if (modal) {
            modal.classList.remove('active');
        }
    }
}

window.analysisController =
    new AnalysisController();
