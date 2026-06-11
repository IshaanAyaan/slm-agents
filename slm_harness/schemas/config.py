"""Experiment configuration schemas (model profiles, harness profiles, conditions C1-C6)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

ProviderKind = Literal["anthropic", "openai_compat", "stub"]
ModelFactor = Literal["large_general", "small_general", "small_finetuned"]
HarnessFactor = Literal["generic", "custom"]


class PricingConfig(BaseModel):
    """Per-model token pricing used for cost-per-successful-task.

    Prices are USD per million tokens. Defaults are PLACEHOLDERS — set real,
    dated prices in the experiment config before reporting any cost numbers.
    """

    usd_per_million_input: float = Field(ge=0.0)
    usd_per_million_output: float = Field(ge=0.0)
    is_placeholder: bool = Field(
        default=True,
        description="True until replaced with verified, dated provider pricing.",
    )

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Compute USD cost for a token count pair."""
        return (
            input_tokens * self.usd_per_million_input
            + output_tokens * self.usd_per_million_output
        ) / 1_000_000.0


class ModelProfile(BaseModel):
    """One model participating in the experiment."""

    profile_id: str
    provider: ProviderKind
    model_name: str = Field(description="Provider model string, e.g. 'claude-sonnet-4-6'")
    factor: ModelFactor
    pricing: PricingConfig
    max_tokens: int = Field(default=1024, ge=1)
    base_url: str | None = Field(
        default=None, description="For openai_compat (e.g. local vLLM endpoint)."
    )
    api_key_env: str | None = Field(
        default=None,
        description="NAME of the environment variable holding the key. Never the key itself.",
    )
    adapter_path: str | None = Field(
        default=None, description="LoRA adapter path for fine-tuned SLM profiles."
    )
    is_finetuned: bool = False
    stub_skill: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Stub-client-only: probability of a competent step. Used exclusively for "
            "pipeline smoke tests; has no effect with real providers."
        ),
    )

    @model_validator(mode="after")
    def _check_finetuned(self) -> "ModelProfile":
        if self.factor == "small_finetuned" and not self.is_finetuned:
            object.__setattr__(self, "is_finetuned", True)
        return self


class HarnessProfile(BaseModel):
    """Harness-side configuration."""

    profile_id: str
    kind: HarnessFactor
    max_steps: int = Field(default=12, ge=1)
    max_retries_per_step: int = Field(default=2, ge=0)
    observation_char_budget: int = Field(
        default=2000, ge=200, description="Custom harness: max chars in the JSON observation."
    )
    tool_output_char_budget: int = Field(
        default=16_000,
        ge=256,
        description="Generic harness: inline tool-output budget (mirrors production default).",
    )
    include_distractor_tools: bool = Field(
        default=True,
        description=(
            "Generic harness: include extra unused tool schemas to reproduce the "
            "realistic prompt bloat of a general-purpose subagent."
        ),
    )


class ConditionDef(BaseModel):
    """One cell of the 2x3 factorial design."""

    condition_id: Literal["C1", "C2", "C3", "C4", "C5", "C6"]
    model_profile_id: str
    harness_profile_id: str
    model_factor: ModelFactor
    harness_factor: HarnessFactor
    description: str = ""


class RunSettings(BaseModel):
    """Settings for one experiment run."""

    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    output_dir: str = "slm_harness/results"
    runs_filename: str = "runs.jsonl"
    max_concurrency: int = Field(default=1, ge=1)
    fail_fast: bool = False


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    experiment_id: str
    models: dict[str, ModelProfile]
    harnesses: dict[str, HarnessProfile]
    conditions: dict[str, ConditionDef]
    tasks_path: str
    run: RunSettings = Field(default_factory=RunSettings)
    finetune_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="One-time fine-tuning cost, used by the break-even analysis.",
    )

    @model_validator(mode="after")
    def _check_refs(self) -> "ExperimentConfig":
        for cid, cond in self.conditions.items():
            if cond.condition_id != cid:
                raise ValueError(f"condition key {cid!r} != condition_id {cond.condition_id!r}")
            if cond.model_profile_id not in self.models:
                raise ValueError(f"{cid}: unknown model profile {cond.model_profile_id!r}")
            if cond.harness_profile_id not in self.harnesses:
                raise ValueError(f"{cid}: unknown harness profile {cond.harness_profile_id!r}")
            model = self.models[cond.model_profile_id]
            harness = self.harnesses[cond.harness_profile_id]
            if model.factor != cond.model_factor:
                raise ValueError(f"{cid}: model factor mismatch")
            if harness.kind != cond.harness_factor:
                raise ValueError(f"{cid}: harness factor mismatch")
        return self
