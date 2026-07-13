"""Tests for manual, compact, and deterministic corpus-history promotion."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from typing import cast

import pytest
from scripts.benchmark_corpus import aggregation, history
from scripts.benchmark_corpus.cli import main as corpus_main
from scripts.benchmark_corpus.history import (
    HISTORY_END,
    HISTORY_START,
    HistoryError,
    PromotionOptions,
    load_snapshot,
    promote_results,
)
from scripts.benchmark_corpus.identity import case_digest, comparison_key
from scripts.benchmark_corpus.models import (
    CaseResult,
    CaseStatus,
    CorpusCase,
    CorpusDefaults,
    CorpusManifest,
    EnvironmentEvidence,
    PolicyEvidence,
    RatioEvidence,
)

_ACCELERATED_SPEEDUP = 2.5
_RUN_ID = 1001


def test_promotion_retains_compact_reviewed_evidence_and_updates_only_marker(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, ("alpha", "beta"))
    results_root = tmp_path / "results"
    result_paths = tuple(
        _write_result(
            results_root / case_id / "case-result.json",
            _result(
                manifest,
                case_id,
                status="accelerated" if case_id == "alpha" else "supported-no-op",
                speedup=_ACCELERATED_SPEEDUP if case_id == "alpha" else None,
            ),
        )
        for case_id in ("alpha", "beta")
    )
    docs = _docs(tmp_path)
    _write_experiments(result_paths, "2026-07-13-initial")

    snapshot_path, docs_path = promote_results(
        manifest,
        tuple(reversed(result_paths)),
        PromotionOptions(
            tier="performance",
            platform="ubuntu-24.04",
            label="2026-07-13-initial",
            reviewed_by="Benchmark Maintainer",
            history_root=tmp_path / "history",
            docs_path=docs,
        ),
    )

    assert docs_path == docs
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["reviewed_by"] == "Benchmark Maintainer"
    assert payload["atoll_revision"] == "f" * 40
    assert payload["experiment"]["run_id"] == _RUN_ID
    assert [case["case_id"] for case in payload["cases"]] == ["alpha", "beta"]
    assert payload["cases"][0]["final_wheel_vs_original"] == _ACCELERATED_SPEEDUP
    assert "samples_seconds" not in snapshot_path.read_text(encoding="utf-8")
    rendered = docs.read_text(encoding="utf-8")
    assert rendered.startswith("# Benchmarks\n\nAuthored contract.\n")
    assert rendered.endswith("\nAuthored tail.\n")
    assert "`performance`" in rendered
    assert "2.500x" in rendered
    assert load_snapshot(snapshot_path).label == "2026-07-13-initial"

    repeated, _ = promote_results(
        manifest,
        result_paths,
        PromotionOptions(
            tier="performance",
            platform="ubuntu-24.04",
            label="2026-07-13-initial",
            reviewed_by="Benchmark Maintainer",
            history_root=tmp_path / "history",
            docs_path=docs,
        ),
    )
    assert repeated.read_bytes() == snapshot_path.read_bytes()


def test_promotion_refuses_label_reuse_for_different_reviewed_evidence(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, ("alpha",))
    result = _write_result(
        tmp_path / "results" / "case-result.json",
        _result(manifest, "alpha"),
    )
    docs = _docs(tmp_path)
    _write_experiments((result,), "reviewed")
    promote_results(
        manifest,
        (result,),
        PromotionOptions(
            tier="performance",
            platform="ubuntu-24.04",
            label="reviewed",
            reviewed_by="One",
            history_root=tmp_path / "history",
            docs_path=docs,
        ),
    )

    with pytest.raises(HistoryError, match="different evidence"):
        promote_results(
            manifest,
            (result,),
            PromotionOptions(
                tier="performance",
                platform="ubuntu-24.04",
                label="reviewed",
                reviewed_by="Two",
                history_root=tmp_path / "history",
                docs_path=docs,
            ),
        )


@pytest.mark.parametrize("label", ["../escape", "Uppercase", "space label", ""])
def test_promotion_rejects_unsafe_labels(tmp_path: Path, label: str) -> None:
    manifest = _manifest(tmp_path, ("alpha",))
    result = _write_result(
        tmp_path / "results" / "case-result.json",
        _result(manifest, "alpha"),
    )

    with pytest.raises(HistoryError, match="promotion label"):
        promote_results(
            manifest,
            (result,),
            PromotionOptions(
                tier="performance",
                platform="ubuntu-24.04",
                label=label,
                reviewed_by="Reviewer",
                history_root=tmp_path / "history",
                docs_path=_docs(tmp_path),
            ),
        )


def test_promotion_rejects_mixed_atoll_revisions(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ("alpha", "beta"))
    alpha = _write_result(
        tmp_path / "results" / "alpha.json",
        _result(manifest, "alpha"),
    )
    beta_result = _result(manifest, "beta")
    assert beta_result.environment is not None
    changed_environment = replace(beta_result.environment, atoll_revision="e" * 40)
    assert beta_result.policy is not None
    beta_result = replace(
        beta_result,
        environment=changed_environment,
        comparison_key=comparison_key(manifest.cases[1], changed_environment, beta_result.policy),
    )
    beta = _write_result(tmp_path / "results" / "beta.json", beta_result)
    _write_experiments((alpha, beta), "mixed")

    with pytest.raises(HistoryError, match="one Atoll revision"):
        promote_results(
            manifest,
            (alpha, beta),
            PromotionOptions(
                tier="performance",
                platform="ubuntu-24.04",
                label="mixed",
                reviewed_by="Reviewer",
                history_root=tmp_path / "history",
                docs_path=_docs(tmp_path),
            ),
        )


def test_promotion_requires_exact_documentation_markers(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ("alpha",))
    result = _write_result(
        tmp_path / "results" / "case-result.json",
        _result(manifest, "alpha"),
    )
    docs = tmp_path / "benchmarks.md"
    docs.write_text("# Missing markers\n", encoding="utf-8")
    _write_experiments((result,), "reviewed")

    with pytest.raises(HistoryError, match="history marker pair"):
        promote_results(
            manifest,
            (result,),
            PromotionOptions(
                tier="performance",
                platform="ubuntu-24.04",
                label="reviewed",
                reviewed_by="Reviewer",
                history_root=tmp_path / "history",
                docs_path=docs,
            ),
        )
    assert not (tmp_path / "history" / "reviewed--performance--ubuntu-24.04.json").exists()


def test_promote_cli_writes_snapshot_and_reports_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _manifest(tmp_path, ("alpha",))
    result = _write_result(
        tmp_path / "results" / "case-result.json",
        _result(manifest, "alpha"),
    )
    docs = _docs(tmp_path)
    _write_experiments((result,), "manual")

    def load_selected_manifest(_path: Path) -> CorpusManifest:
        return manifest

    monkeypatch.setattr("scripts.benchmark_corpus.cli.load_manifest", load_selected_manifest)
    exit_code = corpus_main(
        (
            "--manifest",
            str(manifest.path),
            "promote",
            "--tier",
            "performance",
            "--platform",
            "ubuntu-24.04",
            "--results-root",
            str(result.parent),
            "--label",
            "manual",
            "--reviewed-by",
            "Reviewer",
            "--history-root",
            str(tmp_path / "history"),
            "--docs-path",
            str(docs),
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert Path(payload["snapshot"]).is_file()
    assert payload["docs"] == str(docs)


def test_promotion_rejects_missing_or_mismatched_experiment_labels(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ("alpha",))
    result = _write_result(
        tmp_path / "results" / "alpha" / "case-result.json",
        _result(manifest, "alpha"),
    )
    options = PromotionOptions(
        tier="performance",
        platform="ubuntu-24.04",
        label="reviewed",
        reviewed_by="Reviewer",
        history_root=tmp_path / "history",
        docs_path=_docs(tmp_path),
    )

    with pytest.raises(HistoryError, match="missing experiment identity"):
        promote_results(manifest, (result,), options)

    _write_experiments((result,), "different")
    with pytest.raises(HistoryError, match="does not match promotion label"):
        promote_results(manifest, (result,), options)


def test_promotion_rejects_same_label_from_different_workflow_runs(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ("alpha", "beta"))
    alpha = _write_result(
        tmp_path / "results" / "alpha" / "case-result.json",
        _result(manifest, "alpha"),
    )
    beta = _write_result(
        tmp_path / "results" / "beta" / "case-result.json",
        _result(manifest, "beta"),
    )
    _write_experiments((alpha,), "reviewed", run_id=1001)
    _write_experiments((beta,), "reviewed", run_id=1002)

    with pytest.raises(HistoryError, match="mixed workflow run identities"):
        promote_results(
            manifest,
            (alpha, beta),
            PromotionOptions(
                tier="performance",
                platform="ubuntu-24.04",
                label="reviewed",
                reviewed_by="Reviewer",
                history_root=tmp_path / "history",
                docs_path=_docs(tmp_path),
            ),
        )


def test_invalid_atoll_revision_has_no_promotion_side_effects(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ("alpha",))
    malformed = _result(manifest, "alpha")
    assert malformed.environment is not None
    malformed = replace(
        malformed,
        environment=replace(malformed.environment, atoll_revision="short"),
    )
    result = _write_result(tmp_path / "results" / "case-result.json", malformed)
    _write_experiments((result,), "reviewed")
    docs = _docs(tmp_path)
    original_docs = docs.read_bytes()
    history = tmp_path / "history"

    with pytest.raises(HistoryError, match="full lowercase Atoll revision"):
        promote_results(
            manifest,
            (result,),
            PromotionOptions(
                tier="performance",
                platform="ubuntu-24.04",
                label="reviewed",
                reviewed_by="Reviewer",
                history_root=history,
                docs_path=docs,
            ),
        )

    assert not history.exists()
    assert docs.read_bytes() == original_docs


def test_promotion_uses_one_validated_read_of_each_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path, ("alpha",))
    result_path = _write_result(
        tmp_path / "results" / "case-result.json",
        _result(manifest, "alpha"),
    )
    _write_experiments((result_path,), "reviewed")
    original_loader = aggregation.load_case_result
    calls = 0

    def mutating_loader(path: Path) -> CaseResult:
        nonlocal calls
        calls += 1
        result = original_loader(path)
        _write_result(path, replace(result, status="unsupported"))
        return result

    monkeypatch.setattr(aggregation, "load_case_result", mutating_loader)
    snapshot, _docs_path = promote_results(
        manifest,
        (result_path,),
        PromotionOptions(
            tier="performance",
            platform="ubuntu-24.04",
            label="reviewed",
            reviewed_by="Reviewer",
            history_root=tmp_path / "history",
            docs_path=_docs(tmp_path),
        ),
    )

    assert calls == 1
    assert load_snapshot(snapshot).cases[0].status == "supported-no-op"


def test_immutable_snapshot_creation_never_overwrites_a_racing_writer(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    write_immutable = cast(
        Callable[[Path, str], None],
        vars(history)["_write_immutable"],
    )

    def write(content: str) -> str:
        try:
            write_immutable(path, content)
        except HistoryError:
            return "rejected"
        return "created"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(write, ("first\n", "second\n")))

    assert sorted(outcomes) == ["created", "rejected"]
    assert path.read_text(encoding="utf-8") in {"first\n", "second\n"}


def test_concurrent_promotions_keep_every_snapshot_in_generated_docs(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ("alpha",))
    first = _write_result(
        tmp_path / "first" / "case-result.json",
        _result(manifest, "alpha"),
    )
    second = _write_result(
        tmp_path / "second" / "case-result.json",
        _result(manifest, "alpha"),
    )
    _write_experiments((first,), "first", run_id=1001)
    _write_experiments((second,), "second", run_id=1002)
    docs = _docs(tmp_path)
    history_root = tmp_path / "history"

    def promote(item: tuple[Path, str]) -> Path:
        result, label = item
        snapshot, _docs_path = promote_results(
            manifest,
            (result,),
            PromotionOptions(
                tier="performance",
                platform="ubuntu-24.04",
                label=label,
                reviewed_by="Reviewer",
                history_root=history_root,
                docs_path=docs,
            ),
        )
        return snapshot

    with ThreadPoolExecutor(max_workers=2) as pool:
        snapshots = tuple(pool.map(promote, ((first, "first"), (second, "second"))))

    assert all(path.is_file() for path in snapshots)
    rendered = docs.read_text(encoding="utf-8")
    assert "`first`" in rendered
    assert "`second`" in rendered


def _docs(tmp_path: Path) -> Path:
    path = tmp_path / "benchmarks.md"
    path.write_text(
        "\n".join(
            (
                "# Benchmarks",
                "",
                "Authored contract.",
                HISTORY_START,
                "No snapshots.",
                HISTORY_END,
                "Authored tail.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def _write_experiments(
    paths: tuple[Path, ...],
    label: str,
    *,
    run_id: int = _RUN_ID,
) -> None:
    payload = {
        "head_sha": "f" * 40,
        "label": label,
        "repository": "damienbenveniste/Atoll",
        "run_attempt": 1,
        "run_id": run_id,
        "schema_version": 1,
        "workflow_ref": (
            "damienbenveniste/Atoll/.github/workflows/corpus-performance.yml@refs/heads/main"
        ),
    }
    for parent in {path.parent for path in paths}:
        (parent / "experiment.json").write_text(
            f"{json.dumps(payload, sort_keys=True)}\n",
            encoding="utf-8",
        )


def _manifest(tmp_path: Path, case_ids: tuple[str, ...]) -> CorpusManifest:
    path = tmp_path / "manifest.toml"
    path.write_text("schema_version = 1\n", encoding="utf-8")
    cases = tuple(
        CorpusCase(
            id=case_id,
            name=case_id,
            repository=f"https://example.invalid/{case_id}.git",
            revision="a" * 40,
            project_subroot=PurePosixPath("."),
            dependency_lock=PurePosixPath(f"locks/{case_id}.txt"),
            focused_test_command=("python", "-m", "pytest"),
            oracle_adapter="compatibility",
            oracle_arguments=(),
            tiers=("performance",),
            platforms=("ubuntu-24.04",),
        )
        for case_id in sorted(case_ids)
    )
    return CorpusManifest(
        path=path,
        schema_version=1,
        python_version="3.12",
        backends=("mypyc", "cython"),
        defaults=CorpusDefaults(
            test_timeout_seconds=1,
            compile_timeout_seconds=1,
            performance_timeout_seconds=1,
            max_log_bytes=1,
        ),
        cases=cases,
    )


def _result(
    manifest: CorpusManifest,
    case_id: str,
    *,
    status: CaseStatus = "supported-no-op",
    speedup: float | None = None,
) -> CaseResult:
    case = next(item for item in manifest.cases if item.id == case_id)
    environment = EnvironmentEvidence(
        python="cpython 3.12.11",
        atoll_revision="f" * 40,
        uv="uv 0.8.0",
        mypy="1.18.2",
        cython="3.1.4",
        compiler="clang 17",
        operating_system="Linux 6.8",
        architecture="x86_64",
        runner_image="ubuntu-24.04",
        hardware_class="test-runner",
        dependency_lock_digest="d" * 64,
    )
    policy = PolicyEvidence(
        digest="e" * 64,
        patch_path=PurePosixPath("compile-policy.patch"),
        source_path=PurePosixPath("pyproject.toml"),
    )
    return CaseResult(
        schema_version=1,
        case_id=case.id,
        tier="performance",
        platform="ubuntu-24.04",
        status=status,
        repository=case.repository,
        revision=case.revision,
        manifest_digest=hashlib.sha256(manifest.path.read_bytes()).hexdigest(),
        case_digest=case_digest(case),
        comparison_key=comparison_key(case, environment, policy),
        diagnostics=(),
        source_digest_before=None,
        source_digest_after=None,
        source_unchanged=True,
        policy=policy,
        environment=environment,
        phases=(),
        baseline_wheel_digest=None,
        compiled_wheel_digest=None,
        cold_report_path=None,
        warm_report_path=None,
        baseline_oracle_digest=None,
        compiled_oracle_digest=None,
        cold_compiler_invocations=1,
        warm_compiler_invocations=0,
        ratios=RatioEvidence(
            python_rewrite_vs_original=2.0 if speedup is not None else None,
            final_wheel_vs_original=speedup,
            native_vs_source_only=1.25 if speedup is not None else None,
            baseline_samples_seconds=(1.0, 1.1),
            source_only_samples_seconds=(0.5, 0.55) if speedup is not None else (),
            final_wheel_samples_seconds=(0.4, 0.44) if speedup is not None else (),
        ),
    )


def _write_result(path: Path, result: CaseResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw: object = json.loads(json.dumps(asdict(result), default=_json_default))
    payload = cast(dict[str, object], raw)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _json_default(value: object) -> str:
    if isinstance(value, PurePosixPath):
        return value.as_posix()
    raise TypeError(type(value).__name__)
