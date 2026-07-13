"""Deterministic Mako template-rendering workload."""

from __future__ import annotations

from types import ModuleType

import mako
import mako.template

_BATCH = 30_000


def run(*, repetitions: int, seed: int) -> tuple[dict[str, object], tuple[ModuleType, ...]]:
    """Render a compiled loop-and-conditional template with seeded records."""
    template_type = vars(mako.template)["Template"]
    template = template_type(
        "<ul>\n"
        "% for value in values:\n"
        '<li class=\'${"even" if value % 2 == 0 else "odd"}\'>${value * factor}</li>\n'
        "% endfor\n"
        "</ul>"
    )
    values = tuple(seed % 31 + index for index in range(16))
    renders = _BATCH * repetitions
    checksum = 0
    final_length = 0
    for index in range(renders):
        rendered = template.render(values=values, factor=index % 5 + 1)
        final_length = len(rendered)
        checksum += final_length + rendered.count("even")
    return {"checksum": checksum, "last_length": final_length, "renders": renders}, (mako,)
