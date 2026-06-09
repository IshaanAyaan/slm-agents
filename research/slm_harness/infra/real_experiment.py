"""Assemble a real (non-stub) ExperimentConfig for the self-hosted run.

All three model factors are served locally via vLLM (OpenAI-compatible). Pricing for
each factor comes from measured GPU throughput (``cost_model``), so cost-per-successful
-task is hardware-grounded with zero paid API.

Endpoints (defaults match infra/serve_vllm.sh):
  - teacher  (large_general):     served name = TEACHER_MODEL  at TEACHER_URL
  - student base (small_general): served name = STUDENT_MODEL  at STUDENT_URL
  - student LoRA (small_finetuned): served name = ADAPTER_NAME at STUDENT_URL
"""

from __future__ import annotations

import json
from pathlib import Path

from research.slm_harness import _bootstrap  # noqa: F401

from research.slm_harness.conditions import default_conditions, default_harness_profiles
from research.slm_harness.schemas.config import (
    ExperimentConfig,
    ModelProfile,
    PricingConfig,
    RunSettings,
)


def _load_pricing(path: str | None, fallback: PricingConfig) -> PricingConfig:
    """Load a measured pricing fragment (output of cost_model.py) or use fallback."""
    if path and Path(path).is_file():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return PricingConfig.model_validate(data["pricing"])
    return fallback


def build_real_config(
    *,
    experiment_id: str,
    tasks_path: str,
    teacher_url: str,
    teacher_model: str,
    student_url: str,
    student_model: str,
    adapter_name: str,
    output_dir: str,
    seeds: list[int],
    teacher_pricing_json: str | None = None,
    student_base_pricing_json: str | None = None,
    student_ft_pricing_json: str | None = None,
    finetune_cost_usd: float = 0.0,
    api_key_env: str = "VLLM_API_KEY",
) -> ExperimentConfig:
    """Build the real 2x3 config wired to local vLLM endpoints."""
    # Conservative fallbacks (clearly placeholder) until cost_model measurements exist.
    teacher_pricing = _load_pricing(
        teacher_pricing_json,
        PricingConfig(usd_per_million_input=0.60, usd_per_million_output=0.60),
    )
    base_pricing = _load_pricing(
        student_base_pricing_json,
        PricingConfig(usd_per_million_input=0.04, usd_per_million_output=0.04),
    )
    ft_pricing = _load_pricing(
        student_ft_pricing_json,
        PricingConfig(usd_per_million_input=0.04, usd_per_million_output=0.04),
    )

    models = {
        "large_general": ModelProfile(
            profile_id="large_general",
            provider="openai_compat",
            model_name=teacher_model,
            factor="large_general",
            pricing=teacher_pricing,
            base_url=teacher_url,
            api_key_env=api_key_env,
            max_tokens=1024,
        ),
        "small_general": ModelProfile(
            profile_id="small_general",
            provider="openai_compat",
            model_name=student_model,
            factor="small_general",
            pricing=base_pricing,
            base_url=student_url,
            api_key_env=api_key_env,
            max_tokens=1024,
        ),
        "small_finetuned": ModelProfile(
            profile_id="small_finetuned",
            provider="openai_compat",
            model_name=adapter_name,  # vLLM serves the LoRA under this name
            factor="small_finetuned",
            pricing=ft_pricing,
            base_url=student_url,
            api_key_env=api_key_env,
            is_finetuned=True,
            max_tokens=1024,
        ),
    }
    return ExperimentConfig(
        experiment_id=experiment_id,
        models=models,
        harnesses=default_harness_profiles(),
        conditions=default_conditions(models),
        tasks_path=tasks_path,
        run=RunSettings(seeds=seeds, output_dir=output_dir, max_concurrency=8),
        finetune_cost_usd=finetune_cost_usd,
    )


def write_real_config(out_path: str, **kwargs) -> ExperimentConfig:
    """Build and persist a real config snapshot."""
    cfg = build_real_config(**kwargs)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    return cfg
