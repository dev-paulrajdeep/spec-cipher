class AnalysisController {
    constructor() {
        this.progressInterval = null;
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

        /*
         * Analyze button
         *
         * Button type="submit" causes the form's submit event.
         * This listener catches that event and calls handleSubmit().
         */
        if (this.form) {
            this.form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleSubmit();
            });
        }
    }

    clearForm() {
        if (this.rawTextInput) {
            this.rawTextInput.value = '';
            this.rawTextInput.focus();
        }
    }

    async handleSubmit() {
        const userInput =
            this.rawTextInput?.value?.trim() || '';

        /*
         * Empty input check
         */
        if (!userInput) {
            window.ui.showToast(
                'Please enter a procurement requirement.',
                'warning'
            );

            this.rawTextInput?.focus();
            return;
        }

        const submitButton =
            document.getElementById('btn-submit-analysis');

        if (submitButton) {
            submitButton.disabled = true;
            submitButton.textContent = 'Analyzing...';
        }

        this.showProgressModal();

        try {
            /*
             * Send exact sentence to Flask
             */
            const response =
                await window.apiClient
                    .analyzeSpecification(userInput);

            /*
             * Safety check
             */
            if (
                !response ||
                response.status !== 'success'
            ) {
                throw new Error(
                    response?.message ||
                    'Invalid response received from backend.'
                );
            }

            /*
             * Mark progress complete
             */
            this.completeAllProgressSteps();

            /*
             * Small delay so the user can see the
             * completed progress state.
             */
            setTimeout(() => {
                this.hideProgressModal();

                /*
                 * Give backend response to results.js
                 */
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

                /*
                 * Show Results section
                 */
                window.ui.switchView('results');

                window.ui.showToast(
                    'Analysis completed successfully.',
                    'success'
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

            window.ui.showToast(
                `Analysis failed: ${error.message}`,
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

        /*
         * Move through the progress steps while
         * the backend is working.
         */
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
