"""Executable adapter for the deterministic AnyIO memory-stream workload."""

from _performance import main

if __name__ == "__main__":
    raise SystemExit(main("anyio"))
