"""Versioned deterministic benchmark workloads."""

from localbench.benchmark.schemas import MeasurementSignature

# v1 sent this same prompt through Ollama's chat template (no `raw` mode). Several
# instruction-tuned models treated "continue this list" as one chat turn to satisfy,
# emitted their end-of-turn token after a single item, and stopped well short of
# QUICK_GENERATION_TOKENS — producing a throughput figure measured over 3-15 tokens
# instead of 64. v2 keeps the same prompt text but is sent in Ollama's `raw` mode
# (bypasses chat templating), which is the correct delivery mode for a raw
# text-continuation prompt like this one and lets generation run to the full target.
# v1 remains documented here for anyone comparing against results stored under it;
# it is not reproduced by v2 and should not be treated as comparable.
QUICK_PROMPT_ID = "benchmark_generation_v2"
QUICK_PROMPT = (
    "Continue the following numbered sequence with concise factual sentences, one sentence "
    "per number, until generation stops. Do not add a title.\n"
    "1. Water freezes at zero degrees Celsius.\n"
    "2. Earth orbits the Sun.\n"
    "3."
)
QUICK_CONTEXT_LENGTH = 2048
QUICK_GENERATION_TOKENS = 64
QUICK_SEED = 42
QUICK_BENCHMARK_METHOD_ID = "ollama_generate_timing_v1"
QUICK_RESOURCE_SAMPLER_ID = "system_psutil_v1"
QUICK_RESOURCE_SAMPLE_INTERVAL_SECONDS = 0.1


def quick_measurement_signature(
    runtime_version: str | None,
    *,
    prompt_id: str = QUICK_PROMPT_ID,
) -> MeasurementSignature | None:
    """Build the canonical signature for known quick-profile workload versions.

    Old result JSON did not embed a signature, but it retained every source field needed
    to reconstruct the two known quick workloads. Unknown workloads remain readable but
    deliberately uncomparable. An unknown runtime version
    stays explicit in the signature and is rejected by the comparison eligibility check.
    """
    raw_by_prompt = {
        "benchmark_generation_v1": False,
        "benchmark_generation_v2": True,
    }
    raw = raw_by_prompt.get(prompt_id)
    if raw is None:
        return None
    return MeasurementSignature(
        contract_version="1",
        benchmark_method_id=QUICK_BENCHMARK_METHOD_ID,
        profile="quick",
        prompt_id=prompt_id,
        context_length=QUICK_CONTEXT_LENGTH,
        generation_token_target=QUICK_GENERATION_TOKENS,
        raw=raw,
        stream=True,
        think=False,
        temperature=0.0,
        seed=QUICK_SEED,
        keep_alive_seconds=0,
        cold_start_requested=True,
        runtime="ollama",
        runtime_version=runtime_version,
        resource_sampler_id=QUICK_RESOURCE_SAMPLER_ID,
        resource_sample_interval_seconds=QUICK_RESOURCE_SAMPLE_INTERVAL_SECONDS,
    )
