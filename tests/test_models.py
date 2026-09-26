from __future__ import annotations

from localbench.models import SystemProfile


def test_profile_dictionary_is_json_compatible(sample_profile: SystemProfile) -> None:
    result = sample_profile.to_dict()

    assert result["schema_version"] == "1"
    assert result["collected_at"] == "2026-08-30T12:00:00+00:00"
    assert result["gpus"][0]["shared_memory_bytes"] is None
    assert result["warnings"] == ("Example warning.",)

