"""Tests for the generated source-optimization patch cache."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import cast

import pytest

from atoll.models import SymbolId
from atoll.source_optimization.cache import restore_or_build_source_patch
from atoll.source_optimization.transforms import (
    CallableBodyReplacement,
    SourceTransformationRequest,
)

type JsonObject = dict[str, object]


def _sha256(source: str) -> str:
    """Return the SHA-256 digest used by transformation requests.

    Args:
        source: Source text to hash as UTF-8.

    Returns:
        str: Hex SHA-256 digest.
    """
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _write(root: Path, relative: str, source: str) -> PurePosixPath:
    """Write a Python source fixture under a test project root.

    Args:
        root: Project root for the fixture.
        relative: POSIX relative file path.
        source: Source text to write.

    Returns:
        PurePosixPath: Relative POSIX path for cache and transformation APIs.
    """
    path = PurePosixPath(relative)
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return path


def _request(
    path: PurePosixPath,
    source: str,
    replacement: str = "return value * 3\n",
) -> SourceTransformationRequest:
    """Build a deterministic source transformation request for tests.

    Args:
        path: Relative source path to transform.
        source: Expected source content before transformation.
        replacement: Replacement function body.

    Returns:
        SourceTransformationRequest: Request targeting `run`.
    """
    return SourceTransformationRequest(
        path=path,
        expected_sha256=_sha256(source),
        target=SymbolId(module="pkg.mod", qualname="run"),
        declaration_kind="function",
        replacement_body=replacement,
        summary="rewrite run",
        transformation_id="step:run",
    )


def _complex_request(path: PurePosixPath, source: str) -> SourceTransformationRequest:
    """Build a request that serializes all optional request identity fields.

    Args:
        path: Relative source path to transform.
        source: Expected source content before transformation.

    Returns:
        SourceTransformationRequest: Request with helpers, trailers, and an
        additional callable replacement.
    """
    return SourceTransformationRequest(
        path=path,
        expected_sha256=_sha256(source),
        target=SymbolId(module="pkg.mod", qualname="run"),
        declaration_kind="function",
        replacement_body="return value * 3\n",
        helper_statements=("HELPER = 3",),
        trailing_statements=("TRAILER = run(1)",),
        additional_replacements=(
            CallableBodyReplacement(
                target=SymbolId(module="pkg.mod", qualname="helper"),
                declaration_kind="function",
                replacement_body="return value - 1\n",
            ),
        ),
        summary="rewrite run and helper",
        transformation_id=None,
    )


def _seed_cache_manifest(tmp_path: Path) -> tuple[Path, Path, SourceTransformationRequest]:
    """Create a valid manifest and return paths needed to corrupt it.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        tuple[Path, Path, SourceTransformationRequest]: Project root, manifest
        path, and request that generated the manifest.
    """
    project_root = tmp_path / "project"
    cache_root = tmp_path / "cache"
    source = (
        "def helper(value: int) -> int:\n"
        "    return value + 1\n"
        "\n"
        "def run(value: int) -> int:\n"
        "    return helper(value)\n"
    )
    path = _write(project_root, "pkg/mod.py", source)
    request = _complex_request(path, source)
    restore_or_build_source_patch(
        cache_root,
        "candidate-a",
        ("plan-a",),
        ("step:run",),
        project_root,
        (request,),
    )
    return project_root, next(cache_root.glob("*/manifest.json")), request


def _read_json_object(path: Path) -> JsonObject:
    """Read a JSON object fixture with a runtime assertion for tests.

    Args:
        path: JSON file to load.

    Returns:
        JsonObject: Parsed object.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(JsonObject, value)


def _write_json(path: Path, value: JsonObject) -> None:
    """Write a JSON object fixture using deterministic key order.

    Args:
        path: JSON file to write.
        value: Object payload to serialize.
    """
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _patch_payload(manifest: JsonObject) -> JsonObject:
    """Return the nested manifest patch payload.

    Args:
        manifest: Cache manifest object.

    Returns:
        JsonObject: Nested patch payload.
    """
    patch = manifest["patch"]
    assert isinstance(patch, dict)
    return cast(JsonObject, patch)


def _source_edit_payload(manifest: JsonObject) -> JsonObject:
    """Return the first serialized source edit payload.

    Args:
        manifest: Cache manifest object.

    Returns:
        JsonObject: First source edit object.
    """
    patch = _patch_payload(manifest)
    source_edits = patch["source_edits"]
    assert isinstance(source_edits, list)
    edit = cast(list[object], source_edits)[0]
    assert isinstance(edit, dict)
    return cast(JsonObject, edit)


def _transformed_file_payload(manifest: JsonObject) -> JsonObject:
    """Return the first serialized transformed-file payload.

    Args:
        manifest: Cache manifest object.

    Returns:
        JsonObject: First transformed-file object.
    """
    patch = _patch_payload(manifest)
    files = patch["files"]
    assert isinstance(files, list)
    file_payload = cast(list[object], files)[0]
    assert isinstance(file_payload, dict)
    return cast(JsonObject, file_payload)


def _restore_after_manifest_corruption(
    tmp_path: Path,
    mutate: Callable[[JsonObject], None],
) -> tuple[bool, tuple[str, ...]]:
    """Mutate a valid manifest and restore through the public cache API.

    Args:
        tmp_path: Pytest temporary directory.
        mutate: Callback that receives the manifest and corrupts it.

    Returns:
        tuple[bool, tuple[str, ...]]: Cache hit flag and diagnostics.
    """
    project_root, manifest_path, request = _seed_cache_manifest(tmp_path)
    manifest = _read_json_object(manifest_path)
    mutate(manifest)
    _write_json(manifest_path, manifest)

    restored = restore_or_build_source_patch(
        tmp_path / "cache",
        "candidate-a",
        ("plan-a",),
        ("step:run",),
        project_root,
        (request,),
    )

    return restored.hit, restored.diagnostics


def _set_manifest_field(field: str, value: object) -> Callable[[JsonObject], None]:
    """Build a typed mutator for a top-level manifest field.

    Args:
        field: Manifest field to replace.
        value: Replacement value.

    Returns:
        Callable[[JsonObject], None]: Mutator for parametrized invalid-manifest tests.
    """

    def mutate(manifest: JsonObject) -> None:
        manifest[field] = value

    return mutate


def _set_patch_field(field: str, value: object) -> Callable[[JsonObject], None]:
    """Build a typed mutator for a nested patch field.

    Args:
        field: Patch payload field to replace.
        value: Replacement value.

    Returns:
        Callable[[JsonObject], None]: Mutator for parametrized invalid-manifest tests.
    """

    def mutate(manifest: JsonObject) -> None:
        _patch_payload(manifest)[field] = value

    return mutate


def _set_source_edit_field(field: str, value: object) -> Callable[[JsonObject], None]:
    """Build a typed mutator for a serialized source-edit field.

    Args:
        field: Source-edit field to replace.
        value: Replacement value.

    Returns:
        Callable[[JsonObject], None]: Mutator for parametrized invalid-manifest tests.
    """

    def mutate(manifest: JsonObject) -> None:
        _source_edit_payload(manifest)[field] = value

    return mutate


def _delete_source_edit_field(field: str) -> Callable[[JsonObject], None]:
    """Build a typed mutator that removes a serialized source-edit field.

    Args:
        field: Source-edit field to delete.

    Returns:
        Callable[[JsonObject], None]: Mutator for parametrized invalid-manifest tests.
    """

    def mutate(manifest: JsonObject) -> None:
        del _source_edit_payload(manifest)[field]

    return mutate


def _set_transformed_file_field(field: str, value: object) -> Callable[[JsonObject], None]:
    """Build a typed mutator for a serialized transformed-file field.

    Args:
        field: Transformed-file field to replace.
        value: Replacement value.

    Returns:
        Callable[[JsonObject], None]: Mutator for parametrized invalid-manifest tests.
    """

    def mutate(manifest: JsonObject) -> None:
        _transformed_file_payload(manifest)[field] = value

    return mutate


def _delete_transformed_file_field(field: str) -> Callable[[JsonObject], None]:
    """Build a typed mutator that removes a transformed-file field.

    Args:
        field: Transformed-file field to delete.

    Returns:
        Callable[[JsonObject], None]: Mutator for parametrized invalid-manifest tests.
    """

    def mutate(manifest: JsonObject) -> None:
        del _transformed_file_payload(manifest)[field]

    return mutate


def test_source_patch_cache_misses_then_restores_exact_hit(tmp_path: Path) -> None:
    """A successful miss writes a manifest that restores the same patch payload."""
    project_root = tmp_path / "project"
    cache_root = tmp_path / "cache"
    source = "def run(value: int) -> int:\n    return value + 1\n"
    path = _write(project_root, "pkg/mod.py", source)
    request = _request(path, source)

    miss = restore_or_build_source_patch(
        cache_root,
        "candidate-a",
        ("plan-a",),
        ("step:run",),
        project_root,
        (request,),
    )
    hit = restore_or_build_source_patch(
        cache_root,
        "candidate-a",
        ("plan-a",),
        ("step:run",),
        project_root,
        (request,),
    )

    assert miss.hit is False
    assert miss.diagnostics == ("cache miss",)
    assert hit.hit is True
    assert hit.patch == miss.patch
    assert hit.diagnostics == ("cache hit",)


@pytest.mark.parametrize(
    ("candidate_id", "plan_ids", "transformation_ids", "replacement"),
    [
        ("candidate-b", ("plan-a",), ("step:run",), "return value * 3\n"),
        ("candidate-a", ("plan-b",), ("step:run",), "return value * 3\n"),
        ("candidate-a", ("plan-a",), ("step:other",), "return value * 3\n"),
        ("candidate-a", ("plan-a",), ("step:run",), "return value * 4\n"),
    ],
)
def test_source_patch_cache_invalidates_identity_and_request_changes(
    tmp_path: Path,
    candidate_id: str,
    plan_ids: tuple[str, ...],
    transformation_ids: tuple[str, ...],
    replacement: str,
) -> None:
    """Candidate, step, and request identity changes use distinct cache entries."""
    project_root = tmp_path / "project"
    cache_root = tmp_path / "cache"
    source = "def run(value: int) -> int:\n    return value + 1\n"
    path = _write(project_root, "pkg/mod.py", source)
    request = _request(path, source)
    restore_or_build_source_patch(
        cache_root,
        "candidate-a",
        ("plan-a",),
        ("step:run",),
        project_root,
        (request,),
    )

    changed = restore_or_build_source_patch(
        cache_root,
        candidate_id,
        plan_ids,
        transformation_ids,
        project_root,
        (replace(request, replacement_body=replacement),),
    )

    assert changed.hit is False


def test_source_patch_cache_invalidates_source_hash_changes(tmp_path: Path) -> None:
    """A request with a different expected source digest gets a different key."""
    project_root = tmp_path / "project"
    cache_root = tmp_path / "cache"
    source = "def run(value: int) -> int:\n    return value + 1\n"
    changed_source = "def run(value: int) -> int:\n    return value + 2\n"
    path = _write(project_root, "pkg/mod.py", source)
    request = _request(path, source)
    restore_or_build_source_patch(
        cache_root,
        "candidate-a",
        ("plan-a",),
        ("step:run",),
        project_root,
        (request,),
    )
    _write(project_root, "pkg/mod.py", changed_source)

    changed = restore_or_build_source_patch(
        cache_root,
        "candidate-a",
        ("plan-a",),
        ("step:run",),
        project_root,
        (_request(path, changed_source),),
    )

    assert changed.hit is False
    assert changed.patch.files[0].before_source == changed_source


def test_source_patch_cache_rebuilds_corrupt_manifest(tmp_path: Path) -> None:
    """Malformed manifests are ignored and replaced by a fresh generated patch."""
    project_root = tmp_path / "project"
    cache_root = tmp_path / "cache"
    source = "def run(value: int) -> int:\n    return value + 1\n"
    path = _write(project_root, "pkg/mod.py", source)
    request = _request(path, source)
    initial = restore_or_build_source_patch(
        cache_root,
        "candidate-a",
        ("plan-a",),
        ("step:run",),
        project_root,
        (request,),
    )
    manifest_path = next(cache_root.glob("*/manifest.json"))
    manifest = cast(dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8")))
    manifest["patch_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rebuilt = restore_or_build_source_patch(
        cache_root,
        "candidate-a",
        ("plan-a",),
        ("step:run",),
        project_root,
        (request,),
    )

    assert rebuilt.hit is False
    assert rebuilt.patch == initial.patch
    assert rebuilt.diagnostics[0].startswith("rebuilt invalid cache entry:")


def test_source_patch_cache_does_not_mutate_project_root(tmp_path: Path) -> None:
    """Both misses and hits leave project source bytes untouched."""
    project_root = tmp_path / "project"
    cache_root = tmp_path / "cache"
    source = "def run(value: int) -> int:\n    return value + 1\n"
    path = _write(project_root, "pkg/mod.py", source)
    request = _request(path, source)
    source_path = project_root / "pkg" / "mod.py"

    restore_or_build_source_patch(
        cache_root,
        "candidate-a",
        ("plan-a",),
        ("step:run",),
        project_root,
        (request,),
    )
    restore_or_build_source_patch(
        cache_root,
        "candidate-a",
        ("plan-a",),
        ("step:run",),
        project_root,
        (request,),
    )

    assert source_path.read_text(encoding="utf-8") == source


def test_source_patch_cache_does_not_cache_generation_exceptions(tmp_path: Path) -> None:
    """Failed generation leaves no manifest that could be mistaken for a hit."""
    project_root = tmp_path / "project"
    cache_root = tmp_path / "cache"
    source = "def run(value: int) -> int:\n    return value + 1\n"
    path = _write(project_root, "pkg/mod.py", source)
    request = replace(_request(path, source), expected_sha256="1" * 64)

    with pytest.raises(ValueError, match="stale source"):
        restore_or_build_source_patch(
            cache_root,
            "candidate-a",
            ("plan-a",),
            ("step:run",),
            project_root,
            (request,),
        )

    assert list(cache_root.glob("*/manifest.json")) == []


def test_source_patch_cache_requires_four_identity_arguments(tmp_path: Path) -> None:
    """The public restore wrapper rejects incomplete argument payloads."""
    with pytest.raises(TypeError, match="expects four identity arguments"):
        restore_or_build_source_patch(tmp_path / "cache", "candidate-a")


def test_source_patch_cache_rejects_non_path_project_root(tmp_path: Path) -> None:
    """The project root argument must be a pathlib Path."""
    request = _request(PurePosixPath("pkg/mod.py"), "def run() -> None:\n    pass\n")

    with pytest.raises(TypeError, match="project_root must be a Path"):
        restore_or_build_source_patch(
            tmp_path / "cache",
            "candidate-a",
            ("plan-a",),
            ("step:run",),
            cast(Path, "not-a-path"),
            (request,),
        )


def test_source_patch_cache_rejects_non_tuple_requests(tmp_path: Path) -> None:
    """The requests argument must be a tuple of transformation requests."""
    project_root = tmp_path / "project"
    request = _request(PurePosixPath("pkg/mod.py"), "def run() -> None:\n    pass\n")

    with pytest.raises(TypeError, match="requests must be SourceTransformationRequest tuple"):
        restore_or_build_source_patch(
            tmp_path / "cache",
            "candidate-a",
            ("plan-a",),
            ("step:run",),
            project_root,
            cast(tuple[SourceTransformationRequest, ...], [request]),
        )


def test_source_patch_cache_rejects_non_string_plan_tuple(tmp_path: Path) -> None:
    """Plan and transformation identity payloads must be string tuples."""
    project_root = tmp_path / "project"
    request = _request(PurePosixPath("pkg/mod.py"), "def run() -> None:\n    pass\n")

    with pytest.raises(TypeError, match="plan_ids must be a string tuple"):
        restore_or_build_source_patch(
            tmp_path / "cache",
            "candidate-a",
            cast(tuple[str, ...], ["plan-a"]),
            ("step:run",),
            project_root,
            (request,),
        )


@pytest.mark.parametrize(
    ("candidate_id", "plan_ids", "transformation_ids", "match"),
    [
        ("", ("plan-a",), ("step:run",), "candidate ID is empty"),
        ("candidate-a", (), ("step:run",), "plan IDs are empty"),
        ("candidate-a", ("plan-a",), (), "transformation IDs are empty"),
        ("../candidate", ("plan-a",), ("step:run",), "unsafe source patch cache candidate ID"),
        ("candidate-a", ("plan\\a",), ("step:run",), "unsafe source patch cache plan ID"),
        (
            "candidate-a",
            ("plan-a",),
            (f"{chr(47)}step",),
            "unsafe source patch cache transformation ID",
        ),
    ],
)
def test_source_patch_cache_rejects_unsafe_identity(
    tmp_path: Path,
    candidate_id: str,
    plan_ids: tuple[str, ...],
    transformation_ids: tuple[str, ...],
    match: str,
) -> None:
    """Unsafe candidate, plan, and transformation IDs never reach generation."""
    project_root = tmp_path / "project"
    source = "def run(value: int) -> int:\n    return value + 1\n"
    path = _write(project_root, "pkg/mod.py", source)
    request = _request(path, source)

    with pytest.raises(ValueError, match=match):
        restore_or_build_source_patch(
            tmp_path / "cache",
            candidate_id,
            plan_ids,
            transformation_ids,
            project_root,
            (request,),
        )


def test_source_patch_cache_rejects_empty_requests(tmp_path: Path) -> None:
    """At least one request is required to build a cache identity."""
    with pytest.raises(ValueError, match="requests are empty"):
        restore_or_build_source_patch(
            tmp_path / "cache",
            "candidate-a",
            ("plan-a",),
            ("step:run",),
            tmp_path / "project",
            (),
        )


def test_source_patch_cache_rejects_invalid_request_digest(tmp_path: Path) -> None:
    """Request fingerprints must use lowercase SHA-256 hex digests."""
    request = replace(
        _request(PurePosixPath("pkg/mod.py"), "def run() -> None:\n    pass\n"),
        expected_sha256="not-a-digest",
    )

    with pytest.raises(ValueError, match=r"source patch request digest is invalid: pkg/mod\.py"):
        restore_or_build_source_patch(
            tmp_path / "cache",
            "candidate-a",
            ("plan-a",),
            ("step:run",),
            tmp_path / "project",
            (request,),
        )


@pytest.mark.parametrize(
    "path",
    [
        PurePosixPath(chr(47), "pkg/mod.py"),
        PurePosixPath("pkg/../mod.py"),
        PurePosixPath("."),
    ],
)
def test_source_patch_cache_rejects_unsafe_request_path(
    tmp_path: Path,
    path: PurePosixPath,
) -> None:
    """Request paths must be clean POSIX paths below the project root."""
    request = replace(
        _request(PurePosixPath("pkg/mod.py"), "def run() -> None:\n    pass\n"),
        path=path,
    )

    with pytest.raises(ValueError, match="unsafe source patch cache path"):
        restore_or_build_source_patch(
            tmp_path / "cache",
            "candidate-a",
            ("plan-a",),
            ("step:run",),
            tmp_path / "project",
            (request,),
        )


def test_source_patch_cache_rebuilds_non_object_manifest(tmp_path: Path) -> None:
    """A manifest JSON value must be an object."""
    project_root, manifest_path, request = _seed_cache_manifest(tmp_path)
    manifest_path.write_text("[]", encoding="utf-8")

    restored = restore_or_build_source_patch(
        tmp_path / "cache",
        "candidate-a",
        ("plan-a",),
        ("step:run",),
        project_root,
        (request,),
    )

    assert restored.hit is False
    assert restored.diagnostics == (
        "rebuilt invalid cache entry: source patch cache manifest is not an object",
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            _set_manifest_field("candidate_id", "candidate-b"),
            "source patch cache manifest candidate_id mismatch",
        ),
        (
            _set_manifest_field("patch_sha256", "not-a-digest"),
            "source patch cache manifest patch_sha256 is invalid",
        ),
        (
            _set_manifest_field("patch", "not-a-payload"),
            "source patch cache manifest patch is invalid",
        ),
        (
            _set_patch_field("patch_text", 1),
            "source patch cache patch_text is invalid",
        ),
        (
            _set_patch_field("source_edits", "not-a-list"),
            "source patch cache source_edits is invalid",
        ),
        (
            _set_patch_field("files", "not-a-list"),
            "source patch cache files is invalid",
        ),
        (
            _set_patch_field("source_edits", ["bad"]),
            "source patch cache source edit is invalid",
        ),
        (
            _delete_source_edit_field("summary"),
            "source patch cache source edit fields are invalid",
        ),
        (
            _set_source_edit_field("path", 1),
            "source patch cache source edit path is invalid",
        ),
        (
            _set_source_edit_field("path", "../mod.py"),
            "unsafe source patch cache path: ../mod.py",
        ),
        (
            _set_source_edit_field("before_hash", "bad"),
            "source patch cache source edit before_hash is invalid",
        ),
        (
            _set_source_edit_field("after_hash", "bad"),
            "source patch cache source edit after_hash is invalid",
        ),
        (
            _set_source_edit_field("summary", 1),
            "source patch cache source edit summary is invalid",
        ),
        (
            _set_source_edit_field("touched_symbols", "not-symbols"),
            "source patch cache source edit symbols are invalid",
        ),
        (
            _set_source_edit_field("transformation_id", 1),
            "source patch cache source edit transformation_id is invalid",
        ),
        (
            _set_source_edit_field("start_line", "1"),
            "source patch cache source edit start_line is invalid",
        ),
        (
            _set_source_edit_field("end_line", "2"),
            "source patch cache source edit end_line is invalid",
        ),
        (
            _set_source_edit_field("touched_symbols", ["bad"]),
            "source patch cache symbol is invalid",
        ),
        (
            _set_source_edit_field(
                "touched_symbols",
                [{"module": "pkg.mod"}],
            ),
            "source patch cache symbol fields are invalid",
        ),
        (
            _set_source_edit_field(
                "touched_symbols",
                [{"module": 1, "qualname": "run"}],
            ),
            "source patch cache symbol is invalid",
        ),
        (
            _set_patch_field("files", ["bad"]),
            "source patch cache transformed file is invalid",
        ),
        (
            _delete_transformed_file_field("after_source"),
            "source patch cache transformed file fields are invalid",
        ),
        (
            _set_transformed_file_field("before_source", 1),
            "source patch cache transformed file is invalid",
        ),
    ],
)
def test_source_patch_cache_rebuilds_invalid_manifest_payloads(
    tmp_path: Path,
    mutate: Callable[[JsonObject], None],
    message: str,
) -> None:
    """Malformed manifest payloads are diagnosed and replaced by fresh cache data."""
    hit, diagnostics = _restore_after_manifest_corruption(tmp_path, mutate)

    assert hit is False
    assert diagnostics == (f"rebuilt invalid cache entry: {message}",)


def test_source_patch_cache_rebuilds_manifest_payload_digest_mismatch(
    tmp_path: Path,
) -> None:
    """A valid payload with the wrong digest is rebuilt instead of restored."""
    hit, diagnostics = _restore_after_manifest_corruption(
        tmp_path,
        _set_patch_field("patch_text", "changed\n"),
    )

    assert hit is False
    assert diagnostics == (
        "rebuilt invalid cache entry: source patch cache payload digest mismatch",
    )


def test_source_patch_cache_refuses_to_replace_symlink_manifest(tmp_path: Path) -> None:
    """Atomic manifest writes do not replace an existing symlink."""
    project_root, manifest_path, request = _seed_cache_manifest(tmp_path)
    invalid_target = tmp_path / "invalid-manifest.json"
    invalid_target.write_text("[]", encoding="utf-8")
    manifest_path.unlink()
    manifest_path.symlink_to(invalid_target)

    with pytest.raises(ValueError, match="refusing to replace symlink"):
        restore_or_build_source_patch(
            tmp_path / "cache",
            "candidate-a",
            ("plan-a",),
            ("step:run",),
            project_root,
            (request,),
        )


def test_source_patch_cache_removes_temp_file_when_atomic_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed atomic manifest write cleans up its temporary file."""
    project_root = tmp_path / "project"
    cache_root = tmp_path / "cache"
    source = "def run(value: int) -> int:\n    return value + 1\n"
    path = _write(project_root, "pkg/mod.py", source)
    request = _request(path, source)

    def fail_fsync(file_descriptor: int) -> None:
        """Raise during fsync after the temporary file has been created.

        Args:
            file_descriptor: File descriptor passed by the writer.

        Raises:
            OSError: Always raised to exercise cleanup.
        """
        del file_descriptor
        raise OSError("fsync failed")

    monkeypatch.setattr(os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="fsync failed"):
        restore_or_build_source_patch(
            cache_root,
            "candidate-a",
            ("plan-a",),
            ("step:run",),
            project_root,
            (request,),
        )

    assert list(cache_root.glob("*/manifest.json")) == []
    assert list(cache_root.glob("*/.manifest.json.*.tmp")) == []
