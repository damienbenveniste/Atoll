"""Repository-local benchmark corpus orchestration.

The package owns external-project metadata and evidence handling for Atoll's
benchmark workflows.  It deliberately lives outside :mod:`atoll` so pinned
repositories and benchmark-specific adapters cannot become compiler behavior.
"""

from scripts.benchmark_corpus.cli import main
from scripts.benchmark_corpus.manifest import load_manifest, manifest_matrix

__all__ = ["load_manifest", "main", "manifest_matrix"]
