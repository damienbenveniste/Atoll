"""Command implementations behind Atoll's CLI.

Each command module exposes typed option and result objects so the CLI can stay
thin. Command functions own project mutation, report writing, and filesystem
side effects for their specific workflow.
"""
