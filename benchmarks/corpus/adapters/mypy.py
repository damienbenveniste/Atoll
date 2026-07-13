"""Executable adapter for the deterministic mypy self-check workload."""

from _performance import main

if __name__ == "__main__":
    raise SystemExit(main("mypy"))
