"""CLI management entry point for SpecCipher."""

from __future__ import annotations

import sys


def main() -> None:
    """CLI entry point for operational tasks."""
    print("SpecCipher CLI: operational management tools.")
    if len(sys.argv) > 1 and sys.argv[1] == "--version":
        from speccipher import __version__

        print(f"SpecCipher version {__version__}")


if __name__ == "__main__":
    main()
