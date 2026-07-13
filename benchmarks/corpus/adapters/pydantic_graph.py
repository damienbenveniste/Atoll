"""Executable adapter for the deterministic Pydantic Graph fan-out workload."""

from _performance import main

if __name__ == "__main__":
    raise SystemExit(main("pydantic-graph"))
