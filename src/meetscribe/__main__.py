"""Allow ``python -m meetscribe`` (used by the Nix wrapper in bin/meetscribe)."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
