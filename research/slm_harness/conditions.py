"""Default registry of the six experimental conditions (2x3 factorial).

|     | generic harness | custom harness |
|-----|-----------------|----------------|
| large general    | C1 (baseline)   | C6 (upper bound) |
| small general    | C2 (naive cut)  | C3 (harness-only) |
| small fine-tuned | C4 (FT-only)    | C5 (proposed) |

Model names and prices below are PLACEHOLDERS for the smoke pipeline. Replace them
(or supply your own ExperimentConfig) before running real experiments.
"""

from __future__ import annotations

from research.slm_harness.schemas.config import (
    ConditionDef,
    ExperimentConfig,
    HarnessProfile,
    ModelProfile,
    PricingConfig,
    RunSettings,
)

# --- PLACEHOLDER pricing (USD per 1M tokens). Set real, dated prices before reporting. ---
_LARGE_PRICING = PricingConfig(usd_per_million_input=3.0, usd_per_million_output=15.0)
_SMALL_API_PRICING = PricingConfig(usd_per_million_input=0.10, usd_per_million_output=0.30)
_SLM_SELF_HOST_PRICING = PricingConfig(usd_per_million_input=0.05, usd_per_million_output=0.10)


def default_model_profiles(provider: str = "stub") -> dict[str, ModelProfile]:
    """Default model profiles. provider='stub' keeps everything offline."""
    return {
        "large_general": ModelProfile(
            profile_id="large_general",
            provider=provider,  # real runs: "anthropic"
            model_name="claude-sonnet-4-6",
            factor="large_general",
            pricing=_LARGE_PRICING,
            api_key_env="ANTHROPIC_API_KEY",
            stub_skill=0.95,
        ),
        "small_general": ModelProfile(
            profile_id="small_general",
            provider=provider,  # real runs: "openai_compat" against vLLM
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            factor="small_general",
            pricing=_SMALL_API_PRICING,
            base_url="http://localhost:8000/v1",
            api_key_env="VLLM_API_KEY",
            stub_skill=0.55,
        ),
        "small_finetuned": ModelProfile(
            profile_id="small_finetuned",
            provider=provider,
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            factor="small_finetuned",
            pricing=_SLM_SELF_HOST_PRICING,
            base_url="http://localhost:8000/v1",
            api_key_env="VLLM_API_KEY",
            adapter_path="research/slm_harness/training/checkpoints/qwen3-4b-navsearch-lora",
            is_finetuned=True,
            stub_skill=0.90,
        ),
    }


def default_harness_profiles() -> dict[str, HarnessProfile]:
    """Default generic + custom harness profiles."""
    return {
        "generic": HarnessProfile(profile_id="generic", kind="generic", max_steps=12),
        "custom": HarnessProfile(
            profile_id="custom", kind="custom", max_steps=10, max_retries_per_step=2
        ),
    }


CONDITION_TABLE: list[tuple[str, str, str, str]] = [
    # condition_id, model_profile, harness_profile, description
    ("C1", "large_general", "generic", "Standard baseline: large general model + generic harness"),
    ("C2", "small_general", "generic", "Naive cost-cut: small general model + generic harness"),
    ("C3", "small_general", "custom", "Harness-only effect: small general model + custom harness"),
    ("C4", "small_finetuned", "generic", "Fine-tuning-only effect: small FT model + generic harness"),
    ("C5", "small_finetuned", "custom", "Proposed system: small FT model + custom harness"),
    ("C6", "large_general", "custom", "Upper-bound control: large general model + custom harness"),
]


def default_conditions(models: dict[str, ModelProfile]) -> dict[str, ConditionDef]:
    """Build C1-C6 from a model profile mapping."""
    harness_kind = {"generic": "generic", "custom": "custom"}
    return {
        cid: ConditionDef(
            condition_id=cid,  # type: ignore[arg-type]
            model_profile_id=mp,
            harness_profile_id=hp,
            model_factor=models[mp].factor,
            harness_factor=harness_kind[hp],  # type: ignore[arg-type]
            description=desc,
        )
        for cid, mp, hp, desc in CONDITION_TABLE
    }


def default_experiment_config(
    tasks_path: str,
    *,
    experiment_id: str = "smoke",
    provider: str = "stub",
    seeds: list[int] | None = None,
    output_dir: str = "research/slm_harness/results/smoke",
) -> ExperimentConfig:
    """Assemble a complete default config (offline stub provider by default)."""
    models = default_model_profiles(provider=provider)
    return ExperimentConfig(
        experiment_id=experiment_id,
        models=models,
        harnesses=default_harness_profiles(),
        conditions=default_conditions(models),
        tasks_path=tasks_path,
        run=RunSettings(seeds=seeds or [0, 1, 2, 3, 4], output_dir=output_dir),
    )
