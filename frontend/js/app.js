document.addEventListener(
    'DOMContentLoaded',
    () => {

        console.log(
            'SpecCipher Frontend initialized.'
        );

        const hash =
            window.location.hash
                .replace('#', '');

        if (
            hash &&
            window.ui?.views.includes(
                hash
            )
        ) {
            window.ui.switchView(
                hash
            );
        }
    }
);
