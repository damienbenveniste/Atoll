"""Public package interface for Atoll.

Atoll is primarily a CLI package, but this module exposes the package version so
tests, integrations, and release checks can verify the installed distribution
without importing command internals.
"""

from atoll.core import greet

__all__ = ["greet"]
