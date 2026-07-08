"""Allow `python -m atoll` to run the Atoll CLI."""

from __future__ import annotations

import sys

from atoll.cli import main

if __name__ == "__main__":
    sys.exit(main())
