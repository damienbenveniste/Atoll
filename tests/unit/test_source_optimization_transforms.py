"""Tests for LibCST source-optimization transformations."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pytest

from atoll.models import SymbolId
from atoll.source_optimization.transforms import (
    DeclarationKind,
    SourceTransformationRequest,
    build_source_transformation_patch,
    materialize_transformed_files,
)

_POSIX_ROOT = "/"


@dataclass(frozen=True, slots=True)
class _RequestSpec:
    """Compact request helper input for tests.

    Attributes:
        path: POSIX fixture path.
        source: Expected source content.
        qualname: Target symbol qualname.
        replacement_body: Replacement body statements.
        declaration_kind: Expected declaration kind for the target.
        helpers: Optional helper statement snippets.
    """

    path: PurePosixPath
    source: str
    qualname: str
    replacement_body: str
    declaration_kind: DeclarationKind = "function"
    helpers: tuple[str, ...] = ()


def _sha256(source: str) -> str:
    """Return the SHA-256 digest used by transformation requests.

    Args:
        source: Source text to hash as UTF-8.

    Returns:
        str: Hex SHA-256 digest.
    """
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _write(root: Path, relative: str, source: str) -> PurePosixPath:
    """Write a source fixture under a test project root.

    Args:
        root: Test project root.
        relative: POSIX relative path to write.
        source: Source text for the file.

    Returns:
        PurePosixPath: POSIX path matching the transformation API.
    """
    path = PurePosixPath(relative)
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return path


def _request(spec: _RequestSpec) -> SourceTransformationRequest:
    """Build a representative transformation request for tests.

    Args:
        spec: Compact request helper input.

    Returns:
        SourceTransformationRequest: Immutable request for the test fixture.
    """
    return SourceTransformationRequest(
        path=spec.path,
        expected_sha256=_sha256(spec.source),
        target=SymbolId(module="pkg.mod", qualname=spec.qualname),
        declaration_kind=spec.declaration_kind,
        replacement_body=spec.replacement_body,
        helper_statements=spec.helpers,
        summary=f"rewrite {spec.qualname}",
        transformation_id=f"step:{spec.qualname}",
    )


def test_replaces_function_body_preserving_comments_declaration_and_formatting(
    tmp_path: Path,
) -> None:
    """Only the selected function body changes; comments and neighbors survive."""
    source = (
        '"""module docs"""\n'
        "from __future__ import annotations\n"
        "\n"
        "# keep decorator comment\n"
        "@decorator(\n"
        "    mode='fast',\n"
        ")\n"
        "def run(value: int) -> int:\n"
        "    # old body comment\n"
        "    total = value + 1\n"
        "    return total\n"
        "\n"
        "# untouched neighbor\n"
        "def other() -> str:\n"
        "    return 'ok'\n"
    )
    path = _write(tmp_path, "pkg/mod.py", source)

    patch = build_source_transformation_patch(
        tmp_path,
        (
            _request(
                _RequestSpec(
                    path=path,
                    source=source,
                    qualname="run",
                    replacement_body="# generated body comment\nreturn value * 3\n",
                )
            ),
        ),
    )

    after = patch.files[0].after_source
    preserved_declaration = (
        "# keep decorator comment\n@decorator(\n    mode='fast',\n)\ndef run(value: int) -> int:"
    )
    assert preserved_declaration in after
    assert "    # generated body comment\n    return value * 3\n" in after
    assert "# untouched neighbor\ndef other() -> str:\n    return 'ok'\n" in after
    assert "a/pkg/mod.py" in patch.patch_text
    assert "b/pkg/mod.py" in patch.patch_text
    assert patch.source_edits[0].touched_symbols == (SymbolId("pkg.mod", "run"),)


def test_generated_patch_is_git_apply_compatible(tmp_path: Path) -> None:
    """Generated headers and hunks pass Git's patch parser."""
    source = "def run():\n    return 1\n"
    path = _write(tmp_path, "pkg/mod.py", source)
    patch = build_source_transformation_patch(
        tmp_path,
        (
            _request(
                _RequestSpec(
                    path=path,
                    source=source,
                    qualname="run",
                    replacement_body="return 3\n",
                )
            ),
        ),
    )
    git = shutil.which("git")
    assert git is not None

    result = subprocess.run(
        (git, "apply", "--check", "-"),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        input=patch.patch_text,
        text=True,
    )

    assert patch.patch_text.startswith("--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@")
    assert result.returncode == 0, result.stderr


def test_generated_patch_marks_missing_terminal_newlines(tmp_path: Path) -> None:
    """Git-compatible markers preserve files that omit a terminal newline."""
    source = "def run():\n    return 1"
    path = _write(tmp_path, "pkg/mod.py", source)
    patch = build_source_transformation_patch(
        tmp_path,
        (
            _request(
                _RequestSpec(
                    path=path,
                    source=source,
                    qualname="run",
                    replacement_body="return 3",
                )
            ),
        ),
    )

    assert "\\ No newline at end of file\n" in patch.patch_text


def test_resolves_sync_async_and_class_method_targets(tmp_path: Path) -> None:
    """Supported declarations are selected by exact qualname and expected kind."""
    sync_source = "def run():\n    return 'old'\n"
    async_source = "async def fetch():\n    return 'old'\n"
    method_source = "class Worker:\n    def run(self):\n        return 'old'\n"
    sync_path = _write(tmp_path, "pkg/sync.py", sync_source)
    async_path = _write(tmp_path, "pkg/async_mod.py", async_source)
    method_path = _write(tmp_path, "pkg/method.py", method_source)

    patch = build_source_transformation_patch(
        tmp_path,
        (
            _request(
                _RequestSpec(
                    path=method_path,
                    source=method_source,
                    qualname="Worker.run",
                    declaration_kind="method",
                    replacement_body="return 'method-new'\n",
                )
            ),
            _request(
                _RequestSpec(
                    path=async_path,
                    source=async_source,
                    qualname="fetch",
                    declaration_kind="async_function",
                    replacement_body="return 'async-new'\n",
                )
            ),
            _request(
                _RequestSpec(
                    path=sync_path,
                    source=sync_source,
                    qualname="run",
                    replacement_body="return 'sync-new'\n",
                )
            ),
        ),
    )

    assert [file.path.as_posix() for file in patch.files] == [
        "pkg/async_mod.py",
        "pkg/method.py",
        "pkg/sync.py",
    ]
    assert "return 'async-new'" in patch.files[0].after_source
    assert "return 'method-new'" in patch.files[1].after_source
    assert "return 'sync-new'" in patch.files[2].after_source


def test_helper_insertion_after_docstring_and_future_imports(tmp_path: Path) -> None:
    """Generated helpers are inserted after module docs and future imports."""
    source = (
        '"""docs"""\n'
        "from __future__ import annotations\n"
        "from __future__ import generator_stop\n"
        "\n"
        "import os\n"
        "\n"
        "def run():\n"
        "    return os.name\n"
    )
    path = _write(tmp_path, "pkg/mod.py", source)

    patch = build_source_transformation_patch(
        tmp_path,
        (
            _request(
                _RequestSpec(
                    path=path,
                    source=source,
                    qualname="run",
                    replacement_body="return _helper()\n",
                    helpers=("def _helper():\n    return 'generated'\n", "_HELPER_FLAG = True\n"),
                )
            ),
        ),
    )

    after = patch.files[0].after_source
    assert after.index("from __future__ import generator_stop") < after.index("def _helper")
    assert after.index("_HELPER_FLAG = True") < after.index("import os")


def test_patch_output_is_stable_regardless_of_request_order(tmp_path: Path) -> None:
    """Requests are sorted by path before transformation and diff generation."""
    first_source = "def first():\n    return 1\n"
    second_source = "def second():\n    return 2\n"
    first_path = _write(tmp_path, "pkg/a.py", first_source)
    second_path = _write(tmp_path, "pkg/b.py", second_source)
    first_request = _request(
        _RequestSpec(
            path=first_path,
            source=first_source,
            qualname="first",
            replacement_body="return 10\n",
        )
    )
    second_request = _request(
        _RequestSpec(
            path=second_path,
            source=second_source,
            qualname="second",
            replacement_body="return 20\n",
        )
    )

    forward = build_source_transformation_patch(tmp_path, (first_request, second_request))
    reversed_patch = build_source_transformation_patch(tmp_path, (second_request, first_request))

    assert forward == reversed_patch
    assert forward.patch_text.index("a/pkg/a.py") < forward.patch_text.index("a/pkg/b.py")


def test_rejects_stale_hash_without_mutating_source(tmp_path: Path) -> None:
    """A stale expected hash blocks transformation before any materialization."""
    source = "def run():\n    return 1\n"
    path = _write(tmp_path, "pkg/mod.py", source)
    request = SourceTransformationRequest(
        path=path,
        expected_sha256="0" * 64,
        target=SymbolId("pkg.mod", "run"),
        declaration_kind="function",
        replacement_body="return 2\n",
    )

    with pytest.raises(ValueError, match="stale source"):
        build_source_transformation_patch(tmp_path, (request,))

    assert (tmp_path / "pkg/mod.py").read_text(encoding="utf-8") == source


@pytest.mark.parametrize(
    ("unsafe_path", "message"),
    [
        (PurePosixPath("../pkg/mod.py"), "unsafe source path"),
        (PurePosixPath(_POSIX_ROOT, "pkg/mod.py"), "unsafe source path"),
    ],
)
def test_rejects_out_of_root_paths(
    tmp_path: Path,
    unsafe_path: PurePosixPath,
    message: str,
) -> None:
    """Absolute paths and parent traversal are rejected."""
    request = SourceTransformationRequest(
        path=unsafe_path,
        expected_sha256="0" * 64,
        target=SymbolId("pkg.mod", "run"),
        declaration_kind="function",
        replacement_body="return 2\n",
    )

    with pytest.raises(ValueError, match=message):
        build_source_transformation_patch(tmp_path, (request,))


def test_rejects_duplicate_request_paths(tmp_path: Path) -> None:
    """Only one transformation request may target a file in a patch."""
    source = "def run():\n    return 1\n"
    path = _write(tmp_path, "pkg/mod.py", source)
    request = _request(
        _RequestSpec(
            path=path,
            source=source,
            qualname="run",
            replacement_body="return 2\n",
        )
    )

    with pytest.raises(ValueError, match="duplicate transformation path"):
        build_source_transformation_patch(tmp_path, (request, request))


@pytest.mark.parametrize(
    ("source", "qualname", "kind", "message"),
    [
        ("def run():\n    return 1\n", "missing", "function", "missing target symbol"),
        (
            "def run():\n    return 1\n\ndef run():\n    return 2\n",
            "run",
            "function",
            "duplicate target symbol",
        ),
        ("async def run():\n    return 1\n", "run", "function", "declaration kind mismatch"),
    ],
)
def test_rejects_target_resolution_errors(
    tmp_path: Path,
    source: str,
    qualname: str,
    kind: DeclarationKind,
    message: str,
) -> None:
    """Missing, duplicate, and kind-mismatched targets fail clearly."""
    path = _write(tmp_path, "pkg/mod.py", source)

    with pytest.raises(ValueError, match=message):
        build_source_transformation_patch(
            tmp_path,
            (
                _request(
                    _RequestSpec(
                        path=path,
                        source=source,
                        qualname=qualname,
                        declaration_kind=kind,
                        replacement_body="return 3\n",
                    )
                ),
            ),
        )


@pytest.mark.parametrize(
    ("replacement_body", "helpers", "message"),
    [
        ("return (\n", (), "invalid replacement body"),
        ("return 1\n", ("def bad(:\n    pass\n",), "invalid helper statements"),
    ],
)
def test_rejects_invalid_replacement_or_helper_code(
    tmp_path: Path,
    replacement_body: str,
    helpers: tuple[str, ...],
    message: str,
) -> None:
    """Generated body and helper snippets must parse before patch output."""
    source = "def run():\n    return 1\n"
    path = _write(tmp_path, "pkg/mod.py", source)

    with pytest.raises(ValueError, match=message):
        build_source_transformation_patch(
            tmp_path,
            (
                _request(
                    _RequestSpec(
                        path=path,
                        source=source,
                        qualname="run",
                        replacement_body=replacement_body,
                        helpers=helpers,
                    )
                ),
            ),
        )


def test_materializes_into_caller_owned_copy_without_mutating_original_root(
    tmp_path: Path,
) -> None:
    """The materializer writes only into the caller-owned project copy."""
    root = tmp_path / "root"
    copy_root = tmp_path / "copy"
    source = "def run():\n    return 1\n"
    path = _write(root, "pkg/mod.py", source)
    (root / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
    patch = build_source_transformation_patch(
        root,
        (
            _request(
                _RequestSpec(
                    path=path,
                    source=source,
                    qualname="run",
                    replacement_body="return 42\n",
                )
            ),
        ),
    )

    shutil.copytree(root, copy_root)
    written = materialize_transformed_files(root, copy_root, patch)

    assert written == (copy_root / "pkg/mod.py",)
    assert (copy_root / "pkg/mod.py").read_text(encoding="utf-8") == patch.files[0].after_source
    assert (copy_root / "pyproject.toml").read_text(encoding="utf-8") == (
        "[project]\nname = 'fixture'\n"
    )
    assert (root / "pkg/mod.py").read_text(encoding="utf-8") == source


def test_materialize_transformed_files_rejects_original_root(tmp_path: Path) -> None:
    """Direct materialization refuses to write into the checkout/root input."""
    source = "def run():\n    return 1\n"
    path = _write(tmp_path, "pkg/mod.py", source)
    patch = build_source_transformation_patch(
        tmp_path,
        (
            _request(
                _RequestSpec(
                    path=path,
                    source=source,
                    qualname="run",
                    replacement_body="return 42\n",
                )
            ),
        ),
    )

    with pytest.raises(ValueError, match="temporary project copy must be distinct"):
        materialize_transformed_files(tmp_path, tmp_path, patch)


def test_materialize_transformed_files_writes_existing_temp_copy(tmp_path: Path) -> None:
    """The lightweight materializer writes transformed files under an existing copy."""
    root = tmp_path / "root"
    copy_root = tmp_path / "copy"
    source = "def run():\n    return 1\n"
    path = _write(root, "pkg/mod.py", source)
    shutil.copytree(root, copy_root)
    patch = build_source_transformation_patch(
        root,
        (
            _request(
                _RequestSpec(
                    path=path,
                    source=source,
                    qualname="run",
                    replacement_body="return 99\n",
                )
            ),
        ),
    )

    written = materialize_transformed_files(root, copy_root, patch)

    assert written == (copy_root / "pkg/mod.py",)
    assert (copy_root / "pkg/mod.py").read_text(encoding="utf-8") == "def run():\n    return 99\n"
    assert (root / "pkg/mod.py").read_text(encoding="utf-8") == source


def test_materialization_preflights_all_copy_sources_before_writing(tmp_path: Path) -> None:
    """A stale copied file prevents every write in the requested patch."""
    root = tmp_path / "root"
    copy_root = tmp_path / "copy"
    first_source = "def first():\n    return 1\n"
    second_source = "def second():\n    return 2\n"
    first_path = _write(root, "pkg/first.py", first_source)
    second_path = _write(root, "pkg/second.py", second_source)
    patch = build_source_transformation_patch(
        root,
        (
            _request(
                _RequestSpec(
                    path=first_path,
                    source=first_source,
                    qualname="first",
                    replacement_body="return 10\n",
                )
            ),
            _request(
                _RequestSpec(
                    path=second_path,
                    source=second_source,
                    qualname="second",
                    replacement_body="return 20\n",
                )
            ),
        ),
    )
    shutil.copytree(root, copy_root)
    (copy_root / "pkg/second.py").write_text("def second():\n    return 200\n", encoding="utf-8")

    with pytest.raises(ValueError, match="stale source in temporary project copy"):
        materialize_transformed_files(root, copy_root, patch)

    assert (copy_root / "pkg/first.py").read_text(encoding="utf-8") == first_source
