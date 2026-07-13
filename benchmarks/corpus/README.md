# Multi-Repository Corpus

This directory contains immutable external-project metadata, reviewed dependency
constraints, deterministic workload adapters, and manually promoted evidence for
Atoll's real-repository corpus.

Validate the current manifest without cloning or executing external code:

```bash
uv run python -m scripts.benchmark_corpus validate
```

Repository cases are added only after their dependency lock and canonical oracle
are present. External code is executed by the isolated lifecycle runner, never by
manifest validation.
