class UIController {
    constructor() {
        this.currentView = 'new-analysis';

        this.views = [
            'new-analysis',
            'results',
            'about'
        ];

        this.toastsContainer = null;

        this.init();
    }

    init() {
        /*
         * Create toast container if it doesn't exist.
         */
        let tc =
            document.getElementById(
                'toast-container'
            );

        if (!tc) {
            tc =
                document.createElement('div');

            tc.id =
                'toast-container';

            tc.className =
                'toast-container';

            document.body.appendChild(tc);
        }

        this.toastsContainer =
            tc;

        /*
         * Buttons that contain
         * data-view-target
         */
        document
            .querySelectorAll(
                '[data-view-target]'
            )
            .forEach((btn) => {

                btn.addEventListener(
                    'click',
                    (e) => {

                        e.preventDefault();

                        const target =
                            e.currentTarget
                                .getAttribute(
                                    'data-view-target'
                                );

                        if (target) {
                            this.switchView(
                                target
                            );
                        }
                    }
                );
            });

        /*
         * Close drawer if user clicks
         * outside the panel.
         */
        const overlay =
            document.getElementById(
                'standard-details-drawer'
            );

        if (overlay) {
            overlay.addEventListener(
                'click',
                (e) => {

                    if (
                        e.target ===
                        overlay
                    ) {
                        this.closeDrawer();
                    }
                }
            );
        }

        /*
         * Drawer close buttons
         */
        document
            .querySelectorAll(
                '.drawer-close-btn'
            )
            .forEach((btn) => {

                btn.addEventListener(
                    'click',
                    () => {
                        this.closeDrawer();
                    }
                );
            });

        /*
         * ESC closes drawer.
         */
        document.addEventListener(
            'keydown',
            (e) => {

                if (e.key === 'Escape') {
                    this.closeDrawer();
                }
            }
        );
    }

    switchView(viewName) {
        if (
            !this.views.includes(
                viewName
            )
        ) {
            return;
        }

        this.currentView =
            viewName;

        /*
         * Update navbar active state.
         */
        document
            .querySelectorAll('.nav-btn')
            .forEach((btn) => {

                const target =
                    btn.getAttribute(
                        'data-view-target'
                    );

                if (
                    target ===
                    viewName
                ) {
                    btn.classList.add(
                        'active'
                    );
                } else {
                    btn.classList.remove(
                        'active'
                    );
                }
            });

        /*
         * Hide every view.
         */
        document
            .querySelectorAll(
                '.view-section'
            )
            .forEach((section) => {

                section.classList.remove(
                    'active'
                );
            });

        /*
         * Show requested view.
         */
        const activeSection =
            document.getElementById(
                `view-${viewName}`
            );

        if (activeSection) {
            activeSection.classList.add(
                'active'
            );

            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        }
    }

    openDrawer() {
        const drawer =
            document.getElementById(
                'standard-details-drawer'
            );

        if (drawer) {
            drawer.classList.add(
                'active'
            );

            document.body.style.overflow =
                'hidden';
        }
    }

    closeDrawer() {
        const drawer =
            document.getElementById(
                'standard-details-drawer'
            );

        if (drawer) {
            drawer.classList.remove(
                'active'
            );

            document.body.style.overflow =
                '';
        }
    }

    showToast(
        message,
        type = 'info',
        durationMs = 4000
    ) {
        const toast =
            document.createElement('div');

        toast.className =
            `toast toast-${type}`;

        let iconSvg = '';

        if (type === 'error') {
            iconSvg = `
                <svg
                    width="18"
                    height="18"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2">

                    <circle
                        cx="12"
                        cy="12"
                        r="10">
                    </circle>

                    <line
                        x1="15"
                        y1="9"
                        x2="9"
                        y2="15">
                    </line>

                    <line
                        x1="9"
                        y1="9"
                        x2="15"
                        y2="15">
                    </line>
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

                    <polyline
                        points="20 6 9 17 4 12">
                    </polyline>

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

                    <path d="
                        M10.29 3.86
                        L1.82 18
                        a2 2 0 0 0 1.71 3
                        h16.94
                        a2 2 0 0 0 1.71-3
                        L13.71 3.86
                        a2 2 0 0 0-3.42 0z
                    "></path>

                    <line
                        x1="12"
                        y1="9"
                        x2="12"
                        y2="13">
                    </line>

                    <line
                        x1="12"
                        y1="17"
                        x2="12.01"
                        y2="17">
                    </line>

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

                    <circle
                        cx="12"
                        cy="12"
                        r="10">
                    </circle>

                    <line
                        x1="12"
                        y1="16"
                        x2="12"
                        y2="12">
                    </line>

                    <line
                        x1="12"
                        y1="8"
                        x2="12.01"
                        y2="8">
                    </line>

                </svg>
            `;
        }

        toast.innerHTML = `
            ${iconSvg}

            <span style="flex:1;">
                ${this.escapeHtml(message)}
            </span>
        `;

        this.toastsContainer.appendChild(
            toast
        );

        setTimeout(() => {

            toast.style.opacity =
                '0';

            toast.style.transform =
                'translateY(10px)';

            toast.style.transition =
                'all 0.2s ease';

            setTimeout(() => {
                toast.remove();
            }, 250);

        }, durationMs);
    }

    escapeHtml(str) {
        if (
            str === null ||
            str === undefined
        ) {
            return '';
        }

        const div =
            document.createElement(
                'div'
            );

        div.textContent =
            String(str);

        return div.innerHTML;
    }
}

window.ui =
    new UIController();
