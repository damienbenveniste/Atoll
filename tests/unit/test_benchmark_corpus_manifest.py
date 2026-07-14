"""Contract tests for the pinned benchmark-corpus manifest."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest
from scripts.benchmark_corpus import load_manifest, main, manifest_matrix

_SHA_A = "a" * 40
_SHA_B = "b" * 40
_INVALID_MANIFEST_EXIT_CODE = 2
_VALID_HASH = "a" * 64
_TREE_HASH = "b" * 64
_ARCHIVE_SIZE = 1234
_LARGE_COMPILE_TIMEOUT_SECONDS = 90 * 60
_EXPECTED_REPOSITORY_CASES = {
    "anyio",
    "attrs",
    "cattrs",
    "click",
    "dulwich",
    "html5lib",
    "httpx",
    "jsonschema",
    "mako",
    "markdown",
    "marshmallow",
    "mypy",
    "networkx",
    "pluggy",
    "pydantic",
    "pydantic-graph",
    "rich",
    "sortedcontainers",
    "sqlalchemy",
    "sqlglot",
    "sympy",
    "tomli",
    "tornado",
    "trio",
    "websockets",
}


@dataclass(frozen=True, slots=True)
class _CaseSpec:
    """Inputs varied by the manifest validation tests."""

    case_id: str = "alpha"
    revision: str = _SHA_A
    project_subroot: str = "packages/alpha"
    dependency_lock: str = "uv.lock"
    tiers: tuple[str, ...] = ("compatibility",)
    platforms: tuple[str, ...] = ("ubuntu-24.04",)


_DEFAULT_CASE = _CaseSpec()


def _case(spec: _CaseSpec = _DEFAULT_CASE, *, extra: tuple[str, ...] = ()) -> str:
    """Return one complete TOML case table for focused invalid inputs."""
    tier_values = ", ".join(f'"{tier}"' for tier in spec.tiers)
    platform_values = ", ".join(f'"{platform}"' for platform in spec.platforms)
    return "\n".join(
        (
            "[[case]]",
            f'id = "{spec.case_id}"',
            f'name = "{spec.case_id.title()}"',
            f'repository = "https://example.invalid/{spec.case_id}.git"',
            f'revision = "{spec.revision}"',
            f'project_subroot = "{spec.project_subroot}"',
            f'dependency_lock = "{spec.dependency_lock}"',
            'focused_test_command = ["python", "-m", "pytest", "tests/unit"]',
            f'oracle_adapter = "adapters.{spec.case_id}"',
            f"tiers = [{tier_values}]",
            f"platforms = [{platform_values}]",
            *extra,
        )
    )


def _write_manifest(
    path: Path,
    *,
    schema_version: str = "1",
    python_version: str = '"3.12"',
    backends: str = '["mypyc", "cython"]',
    cases: tuple[str, ...] | None = None,
) -> Path:
    """Write a complete minimal manifest, allowing one-field mutations."""
    case_tables = cases or (_case(),)
    path.write_text(
        "\n".join(
            (
                f"schema_version = {schema_version}",
                f"python_version = {python_version}",
                f"backends = {backends}",
                "",
                "\n\n".join(case_tables),
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("schema_version", ["0", "2", '"1"'])
def test_load_manifest_requires_integer_schema_version_one(
    tmp_path: Path,
    schema_version: str,
) -> None:
    """Only integer schema version 1 is accepted."""
    path = _write_manifest(tmp_path / "corpus.toml", schema_version=schema_version)

    with pytest.raises(ValueError, match="schema_version"):
        load_manifest(path)


def test_load_manifest_accepts_exact_schema_version_one(tmp_path: Path) -> None:
    """A complete version-1 document reaches the typed manifest boundary."""
    manifest = load_manifest(_write_manifest(tmp_path / "corpus.toml"))

    assert manifest.schema_version == 1
    assert manifest.python_version == "3.12"
    assert manifest.backends == ("mypyc", "cython")


def test_schema_one_accepts_optional_content_addressed_sdist(tmp_path: Path) -> None:
    """Schema one adds archive sources without changing existing Git cases."""
    sdist = (
        "[case.sdist]",
        'url = "https://files.example.invalid/source/alpha-1.0.tar.gz"',
        f'archive_sha256 = "{_VALID_HASH}"',
        f"archive_size = {_ARCHIVE_SIZE}",
        f'tree_sha256 = "{_TREE_HASH}"',
    )
    manifest = load_manifest(_write_manifest(tmp_path / "corpus.toml", cases=(_case(extra=sdist),)))

    assert manifest.schema_version == 1
    assert manifest.cases[0].sdist is not None
    assert manifest.cases[0].sdist.archive_size == _ARCHIVE_SIZE
    assert manifest.cases[0].revision == _SHA_A


@pytest.mark.parametrize(
    "sdist_field",
    [
        'url = "http://files.example.invalid/alpha.tar.gz"',
        'archive_sha256 = "short"',
        "archive_size = 0",
        'tree_sha256 = "short"',
    ],
)
def test_sdist_identity_rejects_mutable_or_incomplete_locks(
    tmp_path: Path,
    sdist_field: str,
) -> None:
    """Every archive representation has a canonical URL and complete lock."""
    fields = {
        "url": 'url = "https://files.example.invalid/alpha.tar.gz"',
        "archive_sha256": f'archive_sha256 = "{_VALID_HASH}"',
        "archive_size": f"archive_size = {_ARCHIVE_SIZE}",
        "tree_sha256": f'tree_sha256 = "{_TREE_HASH}"',
    }
    key = sdist_field.split(" =", maxsplit=1)[0]
    fields[key] = sdist_field
    case = _case(extra=("[case.sdist]", *fields.values()))

    with pytest.raises(ValueError, match=key):
        load_manifest(_write_manifest(tmp_path / "corpus.toml", cases=(case,)))


def test_load_manifest_rejects_missing_schema_version(tmp_path: Path) -> None:
    """Schema version is mandatory rather than inferred from case fields."""
    complete = _write_manifest(tmp_path / "complete.toml").read_text(encoding="utf-8")
    path = tmp_path / "corpus.toml"
    path.write_text(
        "\n".join(
            line for line in complete.splitlines() if not line.startswith("schema_version =")
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version"):
        load_manifest(path)


def test_load_manifest_rejects_malformed_toml(tmp_path: Path) -> None:
    """TOML syntax errors are surfaced as manifest validation failures."""
    path = tmp_path / "corpus.toml"
    path.write_text("schema_version = [1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"TOML|toml|manifest"):
        load_manifest(path)


@pytest.mark.parametrize(
    "revision",
    ["a" * 39, "a" * 41, "A" * 40, "g" * 40, "refs/tags/v1.0.0"],
)
def test_load_manifest_requires_full_lowercase_commit_sha(
    tmp_path: Path,
    revision: str,
) -> None:
    """Pins must be immutable 40-character lowercase hexadecimal SHAs."""
    path = _write_manifest(
        tmp_path / "corpus.toml",
        cases=(_case(_CaseSpec(revision=revision)),),
    )

    with pytest.raises(ValueError, match=r"revision|SHA"):
        load_manifest(path)


def test_load_manifest_rejects_missing_revision(tmp_path: Path) -> None:
    """Every corpus case must carry an immutable upstream revision."""
    case_without_revision = "\n".join(
        line for line in _case().splitlines() if not line.startswith("revision =")
    )
    path = _write_manifest(tmp_path / "corpus.toml", cases=(case_without_revision,))

    with pytest.raises(ValueError, match=r"revision|SHA"):
        load_manifest(path)


@pytest.mark.parametrize(
    ("field", "unsafe_path"),
    [
        ("project_subroot", "/root/project"),
        ("project_subroot", "../project"),
        ("project_subroot", "pkg/../project"),
        ("dependency_lock", "/root/uv.lock"),
        ("dependency_lock", "../uv.lock"),
    ],
)
def test_load_manifest_rejects_unsafe_project_paths(
    tmp_path: Path,
    field: str,
    unsafe_path: str,
) -> None:
    """Checkout-relative project paths cannot be absolute or traverse parents."""
    spec = (
        _CaseSpec(project_subroot=unsafe_path)
        if field == "project_subroot"
        else _CaseSpec(dependency_lock=unsafe_path)
    )
    path = _write_manifest(tmp_path / "corpus.toml", cases=(_case(spec),))

    with pytest.raises(ValueError, match=rf"{field}|relative|traversal"):
        load_manifest(path)


def test_load_manifest_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    """Case IDs uniquely identify matrix selections and artifact directories."""
    path = _write_manifest(
        tmp_path / "corpus.toml",
        cases=(_case(), _case(_CaseSpec(platforms=("macos-14",)))),
    )

    with pytest.raises(ValueError, match=r"duplicate|alpha"):
        load_manifest(path)


@pytest.mark.parametrize(
    ("cases", "message"),
    [
        ((_case(extra=('unexpected = "value"',)),), "unexpected|unknown"),
        ((_case(_CaseSpec(tiers=("nightly",))),), "tier"),
        ((_case(_CaseSpec(platforms=("solaris",))),), "platform"),
        ((_case(extra=("has_tool_atoll_compile = true",)),), "compile|unknown"),
    ],
)
def test_load_manifest_rejects_invalid_case_contracts(
    tmp_path: Path,
    cases: tuple[str, ...],
    message: str,
) -> None:
    """Case fields are closed and require a supported execution shape."""
    path = _write_manifest(tmp_path / "corpus.toml", cases=cases)

    with pytest.raises(ValueError, match=message):
        load_manifest(path)


@pytest.mark.parametrize("backends", ["[]", '["mypyc"]', '["cython"]'])
def test_load_manifest_requires_both_supported_backends(
    tmp_path: Path,
    backends: str,
) -> None:
    """The initial corpus always validates both compiler backends together."""
    path = _write_manifest(tmp_path / "corpus.toml", backends=backends)

    with pytest.raises(ValueError, match="backends"):
        load_manifest(path)


def test_manifest_matrix_is_stable_and_filters_without_reordering(tmp_path: Path) -> None:
    """Matrix expansion is sorted and remains sorted under every filter."""
    path = _write_manifest(
        tmp_path / "corpus.toml",
        cases=(
            _case(
                _CaseSpec(
                    case_id="beta",
                    revision=_SHA_B,
                    project_subroot=".",
                    tiers=("negative-control", "calibration"),
                    platforms=("macos-14", "ubuntu-24.04"),
                )
            ),
            _case(),
        ),
    )
    manifest = load_manifest(path)

    assert tuple((row.case_id, row.tier, row.platform) for row in manifest_matrix(manifest)) == (
        ("alpha", "compatibility", "ubuntu-24.04"),
        ("beta", "calibration", "macos-14"),
        ("beta", "calibration", "ubuntu-24.04"),
        ("beta", "negative-control", "macos-14"),
        ("beta", "negative-control", "ubuntu-24.04"),
    )
    assert tuple(
        (row.case_id, row.tier, row.platform)
        for row in manifest_matrix(manifest, tier="calibration")
    ) == (
        ("beta", "calibration", "macos-14"),
        ("beta", "calibration", "ubuntu-24.04"),
    )
    assert tuple(
        (row.case_id, row.tier, row.platform)
        for row in manifest_matrix(manifest, platform="macos-14")
    ) == (
        ("beta", "calibration", "macos-14"),
        ("beta", "negative-control", "macos-14"),
    )
    assert tuple(
        (row.case_id, row.tier, row.platform)
        for row in manifest_matrix(manifest, case_ids=("beta", "alpha"))
    ) == tuple((row.case_id, row.tier, row.platform) for row in manifest_matrix(manifest))
    assert manifest_matrix(manifest, tier="compatibility", platform="macos-14") == ()


def test_validate_cli_returns_success_for_a_valid_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The validate command reports success through its process return code."""
    path = _write_manifest(tmp_path / "corpus.toml")

    assert main(("--manifest", str(path), "validate")) == 0
    captured = capsys.readouterr()
    assert '"schema_version":1' in captured.out
    assert captured.err == ""


def test_validate_cli_returns_failure_for_an_invalid_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validation errors use exit code two and stderr, without a traceback."""
    path = _write_manifest(tmp_path / "corpus.toml", schema_version="2")

    assert main(("--manifest", str(path), "validate")) == _INVALID_MANIFEST_EXIT_CODE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "schema_version" in captured.err
    assert "Traceback" not in captured.err


def test_lock_cli_reports_reviewed_dependency_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Lock inspection hashes the exact repository-local constraints file."""
    lock = tmp_path / "uv.lock"
    lock.write_text(f"package==1.0 --hash=sha256:{_VALID_HASH}\n", encoding="utf-8")
    manifest = _write_manifest(tmp_path / "corpus.toml")

    assert (
        main(
            (
                "--manifest",
                str(manifest),
                "lock",
                "--atoll-root",
                str(tmp_path),
                "--case",
                "alpha",
            )
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert hashlib.sha256(lock.read_bytes()).hexdigest() in payload
    assert '"case":"alpha"' in payload


def test_lock_cli_rejects_dependency_lock_symlink(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A reviewed lock identity cannot be redirected through a symlink."""
    target = tmp_path / "constraints.txt"
    target.write_text("package==1.0\n", encoding="utf-8")
    (tmp_path / "uv.lock").symlink_to(target)
    manifest = _write_manifest(tmp_path / "corpus.toml")

    assert (
        main(
            (
                "--manifest",
                str(manifest),
                "lock",
                "--atoll-root",
                str(tmp_path),
            )
        )
        == _INVALID_MANIFEST_EXIT_CODE
    )
    assert "unsafe dependency lock" in capsys.readouterr().err


def test_repository_manifest_keeps_httpx_and_dulwich_tests_with_their_projects() -> None:
    """Focused test argv cannot be silently exchanged between corpus cases."""
    manifest = load_manifest(Path("benchmarks/corpus/manifest.toml"))
    commands = {case.id: case.focused_test_command for case in manifest.cases}

    assert commands["httpx"][-1] == (
        "tests/client/test_async_client.py::test_async_mock_transport[asyncio]"
    )
    assert commands["dulwich"][-1] == (
        "tests.test_objects.BlobReadTests.test_create_blob_from_string"
    )
    assert set(commands) == _EXPECTED_REPOSITORY_CASES
    cases = {case.id: case for case in manifest.cases}
    html5lib_source = cases["html5lib"].sdist
    assert html5lib_source is not None
    assert html5lib_source.archive_sha256 == (
        "b2e5b40261e20f354d198eae92afc10d750afb487ed5e50f9c4eaf07c184146f"
    )
    assert html5lib_source.tree_sha256 == (
        "ce3f57e64d28229d48b210b348d831126546abe0c8290250f74144e8bffef3f5"
    )
    assert all(case.sdist is None for case in manifest.cases if case.id != "html5lib")
    assert cases["dulwich"].compile_timeout_seconds == _LARGE_COMPILE_TIMEOUT_SECONDS
    assert cases["sqlglot"].compile_timeout_seconds == _LARGE_COMPILE_TIMEOUT_SECONDS


def test_lock_cli_write_uses_reviewed_input_and_reproducible_uv_policy(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock generation fixes Python, platform coverage, hashes, and time cutoff."""
    lock_input = tmp_path / "uv.in"
    lock_input.write_text("package>=1\n", encoding="utf-8")
    manifest = _write_manifest(tmp_path / "corpus.toml")
    observed: list[tuple[str, ...]] = []

    class _Completed:
        returncode = 0

    def fake_run(argv: tuple[str, ...], **_kwargs: object) -> _Completed:
        observed.append(argv)
        output = Path(argv[argv.index("--output-file") + 1])
        output.write_text(
            f"package==1.0 --hash=sha256:{_VALID_HASH}\n",
            encoding="utf-8",
        )
        return _Completed()

    def fake_which(_name: str) -> str:
        return "/uv"

    monkeypatch.setattr("scripts.benchmark_corpus.cli.shutil.which", fake_which)
    monkeypatch.setattr("scripts.benchmark_corpus.cli.subprocess.run", fake_run)

    assert (
        main(
            (
                "--manifest",
                str(manifest),
                "lock",
                "--atoll-root",
                str(tmp_path),
                "--write",
                "--case",
                "alpha",
            )
        )
        == 0
    )
    assert observed
    command = observed[0]
    assert command[:3] == ("/uv", "pip", "compile")
    assert "--universal" in command
    assert command[command.index("--python-version") + 1] == "3.12"
    assert "--generate-hashes" in command
    assert command[command.index("--exclude-newer") + 1] == "2026-07-13T23:59:59Z"
    assert '"case":"alpha"' in capsys.readouterr().out


@pytest.mark.parametrize(
    "requirement",
    [
        "package==1.0",
        f"package>=1.0 --hash=sha256:{_VALID_HASH}",
        f"package @ https://example.invalid/package.whl --hash=sha256:{_VALID_HASH}",
        "--index-url https://example.invalid/simple",
        "-e ../package",
    ],
)
def test_lock_cli_rejects_unhashed_unpinned_or_external_requirements(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    requirement: str,
) -> None:
    """Reviewed locks contain only exact index artifacts with SHA-256 hashes."""
    (tmp_path / "uv.lock").write_text(f"{requirement}\n", encoding="utf-8")
    manifest = _write_manifest(tmp_path / "corpus.toml")

    assert (
        main(
            (
                "--manifest",
                str(manifest),
                "lock",
                "--atoll-root",
                str(tmp_path),
            )
        )
        == _INVALID_MANIFEST_EXIT_CODE
    )
    assert "exact hashed requirements" in capsys.readouterr().err
