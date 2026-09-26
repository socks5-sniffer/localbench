from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

import pytest

from localbench.benchmark.schemas import BenchmarkResult
from localbench.community import (
    CommunityExportError,
    CommunityHardwareFacts,
    CommunityModelFacts,
    assess_benchmark_batch,
    bucket_memory_bytes,
    build_community_envelope,
    render_community_preview,
)


def _hardware() -> CommunityHardwareFacts:
    return CommunityHardwareFacts(
        source_batch_id="private-local-batch",
        os_family="Windows",
        architecture="AMD64",
        cpu_model="Intel Core i5-8265U",
        physical_cores=4,
        logical_cores=8,
        installed_memory_bucket_bytes=24 * 1024**3,
        gpu_model="Intel UHD 620",
        gpu_memory_bucket_bytes=1024**3,
        driver_version="26.20.100.7926",
        execution_backend="cpu",
        gpu_offload_layers=0,
        thread_count=8,
    )


def _model() -> CommunityModelFacts:
    return CommunityModelFacts(
        source_batch_id="private-local-batch",
        digest="sha256:abcdef",
        format="gguf",
        quantization="Q4_K_M",
        parameter_size="4B",
        weights_bytes=4 * 1024**3,
        public_name=None,
    )


def _batch(sample: BenchmarkResult, count: int = 3) -> tuple[BenchmarkResult, ...]:
    return tuple(
        replace(
            sample,
            run_id=f"private-run-{index}",
            batch_id="private-local-batch",
            batch_index=index,
            batch_size=count,
            warnings=(f"private warning C:/Users/person/{index}",),
        )
        for index in range(1, count + 1)
    )


def test_build_preview_is_allow_listed_and_omits_local_identifiers(
    sample_benchmark: BenchmarkResult,
) -> None:
    envelope = build_community_envelope(
        _batch(sample_benchmark),
        _hardware(),
        _model(),
        submission_id="00000000-0000-0000-0000-000000000123",
        created_month="2026-08",
    )

    preview = render_community_preview(envelope)
    decoded = json.loads(preview)

    assert decoded["envelope_version"] == 1
    assert decoded["created_month"] == "2026-08"
    assert decoded["records"][0]["completed_runs"] == 3
    assert [item["ordinal"] for item in decoded["records"][0]["observations"]] == [
        1,
        2,
        3,
    ]
    for forbidden in (
        "private-run",
        "private-local-batch",
        "private warning",
        "C:/Users/person",
        "started_at",
        "warnings",
        "source_batch_id",
    ):
        assert forbidden not in preview
    assert "public_name" not in decoded["records"][0]["model"]


def test_default_submission_ids_are_one_use(sample_benchmark: BenchmarkResult) -> None:
    first = build_community_envelope(_batch(sample_benchmark), _hardware(), _model())
    second = build_community_envelope(_batch(sample_benchmark), _hardware(), _model())

    assert first.submission_id != second.submission_id


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda batch: batch[:2], "incomplete_batch"),
        (lambda batch: (replace(batch[0], batch_size=1),), "singleton_batch"),
        (
            lambda batch: (batch[0], replace(batch[1], pre_run_unload_succeeded=False), batch[2]),
            "failed_cold_unload",
        ),
        (
            lambda batch: (batch[0], replace(batch[1], generation_tokens=3), batch[2]),
            "incomplete_generation",
        ),
        (
            lambda batch: (
                batch[0],
                replace(
                    batch[1],
                    measurement_signature=replace(
                        batch[1].measurement_signature, context_length=4096
                    ),
                ),
                batch[2],
            ),
            "mixed_measurement_signature",
        ),
    ],
)
def test_ineligible_batches_report_structured_reasons(
    sample_benchmark: BenchmarkResult,
    mutation: Callable[
        [tuple[BenchmarkResult, ...]], tuple[BenchmarkResult, ...]
    ],
    expected_code: str,
) -> None:
    mutated = mutation(_batch(sample_benchmark))

    eligibility = assess_benchmark_batch(mutated, _hardware(), _model())

    assert not eligibility.eligible
    assert expected_code in eligibility.exclusion_codes


def test_missing_digest_is_ineligible(sample_benchmark: BenchmarkResult) -> None:
    eligibility = assess_benchmark_batch(
        _batch(sample_benchmark), _hardware(), replace(_model(), digest="")
    )

    assert eligibility.exclusion_codes == ("missing_model_digest",)


def test_snapshot_provenance_must_match_batch_without_being_exported(
    sample_benchmark: BenchmarkResult,
) -> None:
    eligibility = assess_benchmark_batch(
        _batch(sample_benchmark),
        replace(_hardware(), source_batch_id="some-other-batch"),
        _model(),
    )

    assert eligibility.exclusion_codes == ("source_provenance_mismatch",)


def test_unknown_execution_placement_is_ineligible(
    sample_benchmark: BenchmarkResult,
) -> None:
    eligibility = assess_benchmark_batch(
        _batch(sample_benchmark), replace(_hardware(), thread_count=0), _model()
    )

    assert "unknown_execution_placement" in eligibility.exclusion_codes


def test_invalid_metrics_are_ineligible(sample_benchmark: BenchmarkResult) -> None:
    batch = _batch(sample_benchmark)
    broken = (batch[0], replace(batch[1], ttft_seconds=float("nan")), batch[2])

    eligibility = assess_benchmark_batch(broken, _hardware(), _model())

    assert eligibility.exclusion_codes == ("invalid_metrics",)


def test_inconsistent_resource_metrics_are_ineligible(
    sample_benchmark: BenchmarkResult,
) -> None:
    batch = _batch(sample_benchmark)
    broken = (
        batch[0],
        replace(batch[1], peak_system_ram_increase_bytes=1),
        batch[2],
    )

    eligibility = assess_benchmark_batch(broken, _hardware(), _model())

    assert eligibility.exclusion_codes == ("inconsistent_resource_metrics",)


def test_builder_refuses_ineligible_batch(sample_benchmark: BenchmarkResult) -> None:
    with pytest.raises(CommunityExportError) as caught:
        build_community_envelope(
            _batch(sample_benchmark), _hardware(), replace(_model(), digest="")
        )

    assert caught.value.code == "ineligible_export"
    assert caught.value.reasons == ("missing_model_digest",)


def test_memory_bucketing_is_coarse_and_never_zero() -> None:
    gib = 1024**3

    assert bucket_memory_bytes(23 * gib) == 24 * gib
    assert bucket_memory_bytes(gib) == 4 * gib


def test_fixed_ids_make_preview_deterministic(sample_benchmark: BenchmarkResult) -> None:
    arguments = (
        _batch(sample_benchmark),
        _hardware(),
        _model(),
    )
    options = {
        "submission_id": "00000000-0000-0000-0000-000000000123",
        "created_month": "2026-08",
    }

    first = render_community_preview(build_community_envelope(*arguments, **options))
    second = render_community_preview(build_community_envelope(*arguments, **options))

    assert first == second


@pytest.mark.parametrize(
    ("submission_id", "created_month", "expected_code"),
    [
        ("not-a-uuid", "2026-08", "invalid_submission_id"),
        ("00000000-0000-0000-0000-000000000123", "2026-13", "invalid_created_month"),
    ],
)
def test_builder_validates_envelope_metadata(
    sample_benchmark: BenchmarkResult,
    submission_id: str,
    created_month: str,
    expected_code: str,
) -> None:
    with pytest.raises(CommunityExportError) as caught:
        build_community_envelope(
            _batch(sample_benchmark),
            _hardware(),
            _model(),
            submission_id=submission_id,
            created_month=created_month,
        )

    assert caught.value.code == expected_code
