class AnalysisController {
  constructor() {
    this.selectedFile = null;
    this.progressInterval = null;
    this.init();
  }

  init() {
    // Form er input gulo
    this.form = document.getElementById('spec-analysis-form');
    this.rawTextInput = document.getElementById('spec-raw-text');
    this.productInput = document.getElementById('spec-product');
    this.appInput = document.getElementById('spec-application');
    this.techSpecInput = document.getElementById('spec-tech-specs');
    this.qtyInput = document.getElementById('spec-quantity');

    // Optional structured field khulbe / bondho hobe
    this.structuredToggleBtn = document.getElementById('toggle-structured-fields');
    this.structuredFieldsContainer = document.getElementById('structured-fields-container');

    if (this.structuredToggleBtn && this.structuredFieldsContainer) {
      this.structuredToggleBtn.addEventListener('click', () => {
        const isHidden =
          this.structuredFieldsContainer.style.display === 'none';

        this.structuredFieldsContainer.style.display =
          isHidden ? 'grid' : 'none';

        this.structuredToggleBtn.innerHTML = isHidden
          ? `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="18 15 12 9 6 15"></polyline></svg> Hide Structured Fields`
          : `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg> Add Structured Parameters (Optional)`;
      });
    }

    // File upload er jinishgulo
    this.dropzone = document.getElementById('spec-dropzone');
    this.fileInput = document.getElementById('spec-file-input');
    this.filePreviewCard = document.getElementById('selected-file-preview');
    this.fileNameText = document.getElementById('file-preview-name');
    this.fileSizeText = document.getElementById('file-preview-size');
    this.removeFileBtn = document.getElementById('btn-remove-file');

    this.setupDropzone();

    // Clear button
    const clearBtn = document.getElementById('btn-clear-spec');

    if (clearBtn) {
      clearBtn.addEventListener('click', () => this.clearForm());
    }

    // Form submit hole ei function cholbe
    if (this.form) {
      this.form.addEventListener('submit', (e) => {
        e.preventDefault();
        this.handleSubmit();
      });
    }
  }

  setupDropzone() {
    if (!this.dropzone || !this.fileInput) return;

    this.dropzone.addEventListener('click', () => {
      this.fileInput.click();
    });

    this.fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        this.handleFileSelection(e.target.files[0]);
      }
    });

    // Drag kore file anar somoy
    ['dragenter', 'dragover'].forEach((eventName) => {
      this.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      this.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.dropzone.classList.remove('dragover');
      });
    });

    // File drop korle
    this.dropzone.addEventListener('drop', (e) => {
      if (
        e.dataTransfer &&
        e.dataTransfer.files &&
        e.dataTransfer.files.length > 0
      ) {
        this.handleFileSelection(e.dataTransfer.files[0]);
      }
    });

    // File remove korar button
    if (this.removeFileBtn) {
      this.removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.clearFile();
      });
    }
  }

  handleFileSelection(file) {
    // File size check
    if (file.size > CONFIG.UPLOAD.MAX_SIZE_BYTES) {
      window.ui.showToast(
        `File size exceeds limit (${(
          CONFIG.UPLOAD.MAX_SIZE_BYTES /
          (1024 * 1024)
        ).toFixed(0)}MB max)`,
        'error'
      );
      return;
    }

    this.selectedFile = file;

    if (this.fileNameText) {
      this.fileNameText.textContent = file.name;
    }

    if (this.fileSizeText) {
      this.fileSizeText.textContent = this.formatFileSize(file.size);
    }

    if (this.filePreviewCard) {
      this.filePreviewCard.classList.add('visible');
    }

    window.ui.showToast(`Attached document: ${file.name}`, 'info');
  }

  clearFile() {
    this.selectedFile = null;

    if (this.fileInput) {
      this.fileInput.value = '';
    }

    if (this.filePreviewCard) {
      this.filePreviewCard.classList.remove('visible');
    }
  }

  clearForm() {
    if (this.rawTextInput) this.rawTextInput.value = '';
    if (this.productInput) this.productInput.value = '';
    if (this.appInput) this.appInput.value = '';
    if (this.techSpecInput) this.techSpecInput.value = '';
    if (this.qtyInput) this.qtyInput.value = '';

    this.clearFile();
  }

  formatFileSize(bytes) {
    if (bytes === 0) return '0 B';

    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return (
      parseFloat((bytes / Math.pow(k, i)).toFixed(1)) +
      ' ' +
      sizes[i]
    );
  }

  async handleSubmit() {
    const rawText = this.rawTextInput?.value?.trim() || '';
    const product = this.productInput?.value?.trim() || '';
    const application = this.appInput?.value?.trim() || '';
    const techSpecs = this.techSpecInput?.value?.trim() || '';
    const quantity = this.qtyInput?.value
      ? parseInt(this.qtyInput.value, 10)
      : undefined;

    // Kichu input na thakle submit korte dibo na
    if (!rawText && !product && !this.selectedFile) {
      window.ui.showToast(
        'Please provide a procurement specification or upload a tender document.',
        'warning'
      );

      this.rawTextInput?.focus();
      return;
    }

    let payload;
    let querySummary = '';

    // File thakle FormData jabe
    if (this.selectedFile) {
      payload = new FormData();

      payload.append('file', this.selectedFile);

      if (rawText) {
        payload.append('raw_text', rawText);
      }

      if (product) {
        payload.append('product_description', product);
      }

      if (application) {
        payload.append('application', application);
      }

      if (techSpecs) {
        payload.append('technical_specification', techSpecs);
      }

      if (quantity) {
        payload.append('quantity', String(quantity));
      }

      querySummary =
        `File: ${this.selectedFile.name}` +
        (product ? ` (${product})` : '');
    } else {
      // Sudhu text hole JSON pathabo
      payload = {
        input_type: 'text',
        raw_text: rawText,
        product_description: product || undefined,
        application: application || undefined,
        technical_specification: techSpecs || undefined,
        quantity: quantity || undefined
      };

      querySummary =
        product ||
        (rawText.length > 80
          ? rawText.substring(0, 80) + '...'
          : rawText);
    }

    // Analysis cholche dekhabe
    this.showProgressModal();

    try {
      const response =
        await window.apiClient.analyzeSpecification(payload);

      this.completeAllProgressSteps();

      setTimeout(() => {
        this.hideProgressModal();

        // Result page e response pathacchi
        if (window.resultsManager) {
          window.resultsManager.renderResults(response, {
            querySummary,
            rawText,
            product,
            application,
            timestamp: new Date().toISOString()
          });
        }

        // History te save korchi
        if (window.historyManager) {
          window.historyManager.addRecord({
            summary: querySummary,
            timestamp: new Date().toISOString(),
            recommendationsCount:
              response?.recommendations?.length || 0,
            response: response
          });
        }

        // Result page khulbe
        window.ui.switchView('results');

        window.ui.showToast(
          'Analysis completed successfully.',
          'success'
        );
      }, 500);

    } catch (err) {
      this.hideProgressModal();

      window.ui.showToast(
        `Analysis failed: ${err.message}`,
        'error',
        6000
      );

      console.error('Analysis API Error:', err);
    }
  }

  showProgressModal() {
    const modal =
      document.getElementById('analysis-progress-modal');

    if (!modal) return;

    modal.classList.add('active');

    // Sob step abar pending kore dichhi
    const stepItems =
      modal.querySelectorAll('.progress-step-item');

    stepItems.forEach((item) => {
      item.className = 'progress-step-item pending';
    });

    let currentStep = 0;

    if (stepItems.length > 0) {
      stepItems[0].className =
        'progress-step-item active';
    }

    clearInterval(this.progressInterval);

    this.progressInterval = setInterval(() => {
      if (currentStep < stepItems.length - 1) {
        stepItems[currentStep].className =
          'progress-step-item completed';

        currentStep++;

        stepItems[currentStep].className =
          'progress-step-item active';
      }
    }, 1800);
  }

  completeAllProgressSteps() {
    clearInterval(this.progressInterval);

    const modal =
      document.getElementById('analysis-progress-modal');

    if (!modal) return;

    const stepItems =
      modal.querySelectorAll('.progress-step-item');

    stepItems.forEach((item) => {
      item.className = 'progress-step-item completed';
    });
  }

  hideProgressModal() {
    clearInterval(this.progressInterval);

    const modal =
      document.getElementById('analysis-progress-modal');

    if (modal) {
      modal.classList.remove('active');
    }
  }
}

window.analysisController = new AnalysisController();