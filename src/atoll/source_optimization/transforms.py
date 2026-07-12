"""LibCST source transformations for source-optimization trials.

This module owns the narrow source-rewrite core used after source plans have
already been selected. It verifies file identity, rewrites only requested
declaration bodies with LibCST, inserts generated module helpers at the earliest
syntax-correct location, and returns reviewable patch text without mutating the
checkout that supplied the source.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast, override

import libcst as cst

from atoll.models import SymbolId
from atoll.source_optimization.models import SourceEdit

DeclarationKind = Literal["function", "async_function", "method", "async_method"]


@dataclass(frozen=True, slots=True)
class SourceTransformationRequest:
    """Deterministic request to rewrite one source file.

    The request identifies a single module-relative file, verifies its expected
    SHA-256 digest, replaces one declaration body selected by exact
    `SymbolId.qualname`, and may insert module-level helpers. Helper statements
    are inserted after the module docstring and contiguous `__future__` imports.

    Attributes:
        path: POSIX path to a Python source file under the caller-provided root.
        expected_sha256: Expected SHA-256 hex digest for the current source text.
        target: Symbol whose module-local `qualname` selects the declaration.
        declaration_kind: Expected declaration shape. A mismatch means the source
            no longer matches the planned transformation.
        replacement_body: Python statements that become the selected declaration body.
        helper_statements: Module-level Python statements inserted before ordinary
            imports, constants, classes, and functions.
        trailing_statements: Module-level helpers appended after existing declarations.
            Use these for identity captures that require source callables to exist first.
        summary: Stable human-readable summary recorded on the generated `SourceEdit`.
        transformation_id: Stable transformation step identifier recorded on the edit.
    """

    path: PurePosixPath
    expected_sha256: str
    target: SymbolId
    declaration_kind: DeclarationKind
    replacement_body: str
    helper_statements: tuple[str, ...] = ()
    trailing_statements: tuple[str, ...] = ()
    summary: str = "rewrite source-optimization declaration body"
    transformation_id: str | None = None


@dataclass(frozen=True, slots=True)
class TransformedSourceFile:
    """One transformed source file retained for later materialization.

    Attributes:
        path: POSIX path of the transformed file relative to the project root.
        before_source: Source text read from the immutable checkout/root input.
        after_source: Source text produced by LibCST after the requested rewrite.
    """

    path: PurePosixPath
    before_source: str
    after_source: str


@dataclass(frozen=True, slots=True)
class GeneratedSourcePatch:
    """Deterministic patch output for source-optimization source edits.

    Attributes:
        patch_text: Git-compatible unified diff with `a/` and `b/` paths and no timestamps.
        source_edits: Stable `SourceEdit` metadata for every transformed file.
        files: Complete transformed source payloads used by the materializer.
    """

    patch_text: str
    source_edits: tuple[SourceEdit, ...]
    files: tuple[TransformedSourceFile, ...]


@dataclass(frozen=True, slots=True)
class _Declaration:
    qualname: str
    kind: DeclarationKind


class _DeclarationCollector(cst.CSTVisitor):
    """Collect supported top-level functions and one-hop class methods."""

    def __init__(self) -> None:
        self.declarations: list[_Declaration] = []
        self._class_stack: list[str] = []
        self._function_depth = 0

    @override
    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        """Track classes so methods can be addressed as `Class.method`.

        Args:
            node: Class definition being entered.

        Returns:
            bool | None: `None`, allowing LibCST to continue visiting children.
        """
        self._class_stack.append(node.name.value)
        return None

    @override
    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        """Pop class context after visiting its body.

        Args:
            original_node: Class definition being exited.
        """
        del original_node
        self._class_stack.pop()

    @override
    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        """Record supported declarations before entering nested function bodies.

        Args:
            node: Function definition being entered.

        Returns:
            bool | None: `None`, allowing child nodes to be visited.
        """
        if self._function_depth == 0:
            declaration = _declaration_for(
                node.name.value,
                is_async=node.asynchronous is not None,
                class_stack=tuple(self._class_stack),
            )
            if declaration is not None:
                self.declarations.append(declaration)
        self._function_depth += 1
        return None

    @override
    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        """Leave the current function context.

        Args:
            original_node: Function definition being exited.
        """
        del original_node
        self._function_depth -= 1


class _BodyReplacementTransformer(cst.CSTTransformer):
    """Replace the body of one exact declaration qualname."""

    def __init__(self, target: str, body: cst.IndentedBlock) -> None:
        self._target = target
        self._body = body
        self._class_stack: list[str] = []
        self._function_depth = 0

    @override
    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        """Track class context for method qualnames.

        Args:
            node: Class definition being entered.

        Returns:
            bool | None: `None`, allowing child nodes to be visited.
        """
        self._class_stack.append(node.name.value)
        return None

    @override
    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        """Pop class context after transforming a class body.

        Args:
            original_node: Original class node.
            updated_node: Updated class node.

        Returns:
            cst.ClassDef: Updated class node.
        """
        del original_node
        self._class_stack.pop()
        return updated_node

    @override
    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        """Enter a function while preserving nested declarations.

        Args:
            node: Function definition being entered.

        Returns:
            bool | None: `None`, allowing child nodes to be visited.
        """
        del node
        self._function_depth += 1
        return None

    @override
    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        """Replace the selected declaration body and leave all other nodes intact.

        Args:
            original_node: Original function node.
            updated_node: Updated function node after child transformations.

        Returns:
            cst.FunctionDef: Function node with the requested body replacement
            when its exact qualname matches the request.
        """
        self._function_depth -= 1
        if self._function_depth == 0:
            declaration = _declaration_for(
                original_node.name.value,
                is_async=original_node.asynchronous is not None,
                class_stack=tuple(self._class_stack),
            )
            if declaration is not None and declaration.qualname == self._target:
                return updated_node.with_changes(body=self._body)
        return updated_node


def build_source_transformation_patch(
    project_root: Path,
    requests: tuple[SourceTransformationRequest, ...],
) -> GeneratedSourcePatch:
    """Build deterministic LibCST source edits without mutating the project root.

    Args:
        project_root: Root of the checkout or source tree to read from.
        requests: Per-file transformation requests. Requests are sorted by path
            before execution so patch output is stable across caller ordering.

    Returns:
        GeneratedSourcePatch: Unified diff, source-edit metadata, and transformed
        file payloads that can be materialized into a temporary project copy.

    Raises:
        ValueError: If paths are unsafe or duplicated, source hashes are stale,
            the target declaration is missing or duplicated, declaration kind
            does not match, generated code is invalid, or the transformed module
            is not valid Python source.
    """
    root = project_root.resolve()
    sorted_requests = tuple(sorted(requests, key=lambda request: request.path.as_posix()))
    _reject_duplicate_paths(sorted_requests)

    transformed_files: list[TransformedSourceFile] = []
    source_edits: list[SourceEdit] = []
    patch_parts: list[str] = []
    for request in sorted_requests:
        source_path = _safe_project_path(root, request.path)
        before_source = source_path.read_text(encoding="utf-8")
        before_hash = _sha256(before_source)
        if before_hash != request.expected_sha256:
            raise ValueError(
                f"stale source for {request.path.as_posix()}: expected "
                f"{request.expected_sha256}, found {before_hash}"
            )
        after_source = _transform_source(before_source, request)
        after_hash = _sha256(after_source)
        transformed = TransformedSourceFile(
            path=request.path,
            before_source=before_source,
            after_source=after_source,
        )
        transformed_files.append(transformed)
        start_line, end_line = _changed_line_range(before_source, after_source)
        source_edits.append(
            SourceEdit(
                path=request.path,
                before_hash=before_hash,
                after_hash=after_hash,
                summary=request.summary,
                touched_symbols=(request.target,),
                transformation_id=request.transformation_id,
                start_line=start_line,
                end_line=end_line,
            )
        )
        patch_parts.append(_unified_diff(transformed))

    return GeneratedSourcePatch(
        patch_text="".join(patch_parts),
        source_edits=tuple(source_edits),
        files=tuple(transformed_files),
    )


def materialize_transformed_files(
    project_root: Path,
    temporary_project_copy: Path,
    patch: GeneratedSourcePatch,
) -> tuple[Path, ...]:
    """Write transformed files only into a distinct temporary project copy.

    Args:
        project_root: Original checkout/root input. It is used only as a safety
            boundary and is never written by this function.
        temporary_project_copy: Disposable project copy that should receive the
            transformed file contents.
        patch: Generated patch result returned by `build_source_transformation_patch`.

    Returns:
        tuple[Path, ...]: Absolute paths written under `temporary_project_copy`,
        sorted in the same deterministic order as `patch.files`.

    Raises:
        ValueError: If the temporary copy resolves to the original root, if a
            stored file path is unsafe, or if a write target would escape the
            temporary copy.
    """
    root = project_root.resolve()
    copy_root = temporary_project_copy.resolve()
    if copy_root == root:
        raise ValueError("temporary project copy must be distinct from the source root")
    if not copy_root.is_dir():
        raise ValueError(f"temporary project copy does not exist: {copy_root}")
    destinations: list[tuple[TransformedSourceFile, Path]] = []
    for transformed in patch.files:
        destination = _safe_project_path(copy_root, transformed.path)
        if not _is_relative_to(destination, copy_root):
            raise ValueError(f"transformed path escapes temporary copy: {transformed.path}")
        if not destination.is_file():
            raise ValueError(
                f"temporary project copy is missing source file: {transformed.path.as_posix()}"
            )
        current_source = destination.read_text(encoding="utf-8")
        if _sha256(current_source) != _sha256(transformed.before_source):
            raise ValueError(
                f"stale source in temporary project copy: {transformed.path.as_posix()}"
            )
        destinations.append((transformed, destination))

    written: list[Path] = []
    for transformed, destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(transformed.after_source, encoding="utf-8")
        written.append(destination)
    return tuple(written)


def _transform_source(source: str, request: SourceTransformationRequest) -> str:
    module = _parse_module(source, request.path)
    declarations = _supported_declarations(module)
    matching = [
        declaration
        for declaration in declarations
        if declaration.qualname == request.target.qualname
    ]
    if not matching:
        raise ValueError(
            f"missing target symbol {request.target.stable_id} in {request.path.as_posix()}"
        )
    if len(matching) > 1:
        raise ValueError(
            f"duplicate target symbol {request.target.stable_id} in {request.path.as_posix()}"
        )
    declaration = matching[0]
    if declaration.kind != request.declaration_kind:
        raise ValueError(
            f"declaration kind mismatch for {request.target.stable_id}: "
            f"expected {request.declaration_kind}, found {declaration.kind}"
        )

    body = _parse_replacement_body(request.replacement_body, request)
    helper_statements = _parse_generated_statements(
        request.helper_statements,
        request=request,
        label="helper statements",
    )
    trailing_statements = _parse_generated_statements(
        request.trailing_statements,
        request=request,
        label="trailing statements",
    )
    replaced = module.visit(_BodyReplacementTransformer(request.target.qualname, body))
    if helper_statements:
        replaced = _insert_module_helpers(replaced, helper_statements)
    if trailing_statements:
        replaced = replaced.with_changes(body=(*replaced.body, *trailing_statements))
    after_source = replaced.code
    try:
        ast.parse(after_source, filename=request.path.as_posix())
    except SyntaxError as exc:
        raise ValueError(
            f"transformed source is invalid for {request.path.as_posix()}: {exc.msg}"
        ) from exc
    return after_source


def _parse_module(source: str, path: PurePosixPath) -> cst.Module:
    try:
        return cst.parse_module(source)
    except cst.ParserSyntaxError as exc:
        raise ValueError(f"invalid source for {path.as_posix()}: {exc}") from exc


def _supported_declarations(module: cst.Module) -> tuple[_Declaration, ...]:
    collector = _DeclarationCollector()
    module.visit(collector)
    return tuple(collector.declarations)


def _declaration_for(
    name: str,
    *,
    is_async: bool,
    class_stack: tuple[str, ...],
) -> _Declaration | None:
    if not class_stack:
        return _Declaration(
            qualname=name,
            kind="async_function" if is_async else "function",
        )
    if len(class_stack) == 1:
        return _Declaration(
            qualname=f"{class_stack[0]}.{name}",
            kind="async_method" if is_async else "method",
        )
    return None


def _parse_replacement_body(
    body_source: str,
    request: SourceTransformationRequest,
) -> cst.IndentedBlock:
    if not body_source.strip():
        raise ValueError(f"invalid replacement body for {request.target.stable_id}: empty body")
    normalized = body_source if body_source.endswith("\n") else f"{body_source}\n"
    indented = "".join(
        f"    {line}" if line.strip() else line for line in normalized.splitlines(True)
    )
    try:
        wrapper = cst.parse_module(f"def _atoll_replacement_body():\n{indented}")
    except cst.ParserSyntaxError as exc:
        raise ValueError(f"invalid replacement body for {request.target.stable_id}: {exc}") from exc
    statement = cast(cst.FunctionDef, wrapper.body[0])
    return cast(cst.IndentedBlock, statement.body)


def _parse_generated_statements(
    statements: tuple[str, ...],
    *,
    request: SourceTransformationRequest,
    label: str,
) -> tuple[cst.BaseStatement, ...]:
    if not statements:
        return ()
    helper_source = "\n".join(statement.rstrip("\n") for statement in statements)
    if helper_source:
        helper_source = f"{helper_source}\n"
    try:
        helper_module = cst.parse_module(helper_source)
    except cst.ParserSyntaxError as exc:
        raise ValueError(f"invalid {label} for {request.path.as_posix()}: {exc}") from exc
    if not helper_module.body:
        raise ValueError(f"invalid {label} for {request.path.as_posix()}: empty statements")
    return tuple(helper_module.body)


def _insert_module_helpers(
    module: cst.Module,
    helper_statements: tuple[cst.BaseStatement, ...],
) -> cst.Module:
    body = tuple(module.body)
    insertion_index = _helper_insertion_index(body)
    return module.with_changes(
        body=body[:insertion_index] + helper_statements + body[insertion_index:]
    )


def _helper_insertion_index(body: tuple[cst.BaseStatement, ...]) -> int:
    index = 0
    if body and _is_module_docstring(body[0]):
        index = 1
    while index < len(body) and _is_future_import(body[index]):
        index += 1
    return index


def _is_module_docstring(statement: cst.BaseStatement) -> bool:
    return (
        isinstance(statement, cst.SimpleStatementLine)
        and len(statement.body) == 1
        and isinstance(statement.body[0], cst.Expr)
        and isinstance(statement.body[0].value, cst.SimpleString)
    )


def _is_future_import(statement: cst.BaseStatement) -> bool:
    if not isinstance(statement, cst.SimpleStatementLine) or len(statement.body) != 1:
        return False
    small_statement = statement.body[0]
    return (
        isinstance(small_statement, cst.ImportFrom)
        and isinstance(small_statement.module, cst.Name)
        and small_statement.module.value == "__future__"
    )


def _reject_duplicate_paths(requests: tuple[SourceTransformationRequest, ...]) -> None:
    seen: set[PurePosixPath] = set()
    for request in requests:
        if request.path in seen:
            raise ValueError(f"duplicate transformation path: {request.path.as_posix()}")
        seen.add(request.path)


def _safe_project_path(root: Path, relative_path: PurePosixPath) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"unsafe source path: {relative_path.as_posix()}")
    candidate = (root / Path(relative_path.as_posix())).resolve()
    if not _is_relative_to(candidate, root):
        raise ValueError(f"source path escapes project root: {relative_path.as_posix()}")
    return candidate


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _unified_diff(transformed: TransformedSourceFile) -> str:
    path_text = transformed.path.as_posix()
    diff_lines = difflib.unified_diff(
        transformed.before_source.splitlines(keepends=True),
        transformed.after_source.splitlines(keepends=True),
        fromfile=f"a/{path_text}",
        tofile=f"b/{path_text}",
        lineterm="\n",
    )
    rendered: list[str] = []
    for line in diff_lines:
        if line.endswith("\n"):
            rendered.append(line)
        else:
            rendered.extend((f"{line}\n", "\\ No newline at end of file\n"))
    return "".join(rendered)


def _changed_line_range(before_source: str, after_source: str) -> tuple[int | None, int | None]:
    if before_source == after_source:
        return None, None
    matcher = difflib.SequenceMatcher(
        a=before_source.splitlines(),
        b=after_source.splitlines(),
        autojunk=False,
    )
    changed = [
        (before_start, before_end)
        for tag, before_start, before_end, _after_start, _after_end in matcher.get_opcodes()
        if tag != "equal"
    ]
    if not changed:
        return None, None
    start_line = min(start for start, _end in changed) + 1
    end_line = max(end for _start, end in changed)
    return start_line, max(start_line, end_line)
