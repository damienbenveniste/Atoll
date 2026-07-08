"""Small runnable example for atoll."""

from atoll import greet


def main() -> None:
    """Print a deterministic greeting."""
    print(greet("Agent"))


if __name__ == "__main__":
    main()
