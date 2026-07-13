"""Contract tests for the pinned benchmark-corpus manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from scripts.benchmark_corpus import load_manifest, main, manifest_matrix

_SHA_A = "a" * 40
_SHA_B = "b" * 40
_INVALID_MANIFEST_EXIT_CODE = 2


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
