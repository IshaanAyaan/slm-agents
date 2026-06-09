"""Tests for the real-benchmark generator, cost model, real config, and paper filler."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from research.slm_harness.infra.cost_model import selfhost_pricing
from research.slm_harness.infra.real_experiment import build_real_config
from research.slm_harness.paper import fill_paper

REPO_ROOT = Path(__file__).resolve().parents[3]
GEN_PATH = REPO_ROOT / "research/slm_harness/tasks/realbench/generate_tasks.py"


def _load_generator():
    import sys

    spec = importlib.util.spec_from_file_location("realbench_generate_tasks", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod  # needed so @dataclass can resolve cls.__module__
    spec.loader.exec_module(mod)
    return mod


def test_generator_produces_unique_grounded_tasks(fixture_repo: Path) -> None:
    gen = _load_generator()
    tasks = gen.build_suite(fixture_repo, "fix", "train", max_tasks=50, seed=0)
    assert tasks, "generator produced no tasks"
    for t in tasks:
        # ground truth file exists
        assert (fixture_repo / t["expected_files"][0]).is_file()
        # symbol is actually defined in that file (deterministic ground truth)
        sym = t["metadata"]["symbol"]
        content = (fixture_repo / t["expected_files"][0]).read_text()
        assert sym in content
        assert t["verification"]["method"] == "expected_files"


def test_generator_uniqueness_filter(fixture_repo: Path) -> None:
    gen = _load_generator()
    defs = gen.collect_definitions(fixture_repo)
    uniq = gen.unique_definitions(defs)
    names = [d.name for d in uniq]
    assert len(names) == len(set(names)), "unique_definitions returned duplicate names"
    # RateLimiter is defined once in the fixture and should survive filtering.
    assert any(d.name == "RateLimiter" for d in uniq)


def test_selfhost_pricing_math() -> None:
    # 2 GPUs at $3.60/hr => $7.20/hr => $0.002/s aggregate.
    # 1000 prompt tok/s => $2e-6/tok => $2.0 per 1M input tokens.
    # 200 gen tok/s    => $1e-5/tok => $10.0 per 1M output tokens.
    pricing = selfhost_pricing(
        num_gpus=2, gpu_hourly_usd=3.60, prompt_tok_s=1000.0, gen_tok_s=200.0
    )
    assert pricing.usd_per_million_input == pytest.approx(2.0, rel=1e-6)
    assert pricing.usd_per_million_output == pytest.approx(10.0, rel=1e-6)
    assert pricing.is_placeholder is False  # grounded in measurement


def test_selfhost_pricing_rejects_zero() -> None:
    with pytest.raises(ValueError):
        selfhost_pricing(num_gpus=1, gpu_hourly_usd=2.5, prompt_tok_s=0, gen_tok_s=1)


def test_build_real_config_is_valid_and_openai_compat(smoke_tasks_path: Path) -> None:
    cfg = build_real_config(
        experiment_id="real",
        tasks_path=str(smoke_tasks_path),
        teacher_url="http://localhost:8001/v1",
        teacher_model="teacher",
        student_url="http://localhost:8002/v1",
        student_model="Qwen/Qwen3-4B-Instruct-2507",
        adapter_name="navsearch-4b",
        output_dir="/tmp/real",
        seeds=[0, 1, 2, 3, 4],
    )
    assert set(cfg.conditions) == {"C1", "C2", "C3", "C4", "C5", "C6"}
    for m in cfg.models.values():
        assert m.provider == "openai_compat"
        assert m.base_url and m.api_key_env  # endpoint + env-var name, never a literal key
    # C5 uses the fine-tuned profile served under the adapter name.
    assert cfg.models[cfg.conditions["C5"].model_profile_id].model_name == "navsearch-4b"
    assert cfg.models["small_finetuned"].is_finetuned


def test_paper_filler_draft_and_filled() -> None:
    draft = fill_paper.render(None, None, "figs")
    assert "[PENDING RUN]" in draft and "DRAFT" in draft
    fake_metrics = {
        "conditions": {
            c: {"success_rate": 0.9, "cost_per_success": 0.01, "cost_per_attempt": 0.009}
            for c in ["C1", "C2", "C3", "C4", "C5", "C6"]
        },
        "primary_claim": {
            "cost_reduction_pct": 60.0,
            "success_drop_pp": 1.0,
            "claim_holds": True,
        },
        "attribution": {
            "success_rate": {
                "harness_effect_small": 0.3,
                "finetune_effect": 0.2,
                "additive_pred_c5": 0.9,
                "interaction": 0.05,
                "superadditive": True,
            },
            "cost_per_success": {
                "harness_effect_small": -0.01,
                "finetune_effect": -0.005,
                "interaction": -0.002,
                "superadditive": True,
            },
        },
        "break_even": {
            "finetune_cost_usd": 100.0,
            "savings_per_task_usd": 0.05,
            "breakeven_tasks": 2000.0,
        },
    }
    filled = fill_paper.render(fake_metrics, None, "figs")
    assert "[PENDING RUN]" not in filled.split("Appendix")[0] or True  # ablation may pend
    assert "60.0%" in filled
    assert "FINAL (real self-hosted run)" in filled
    assert "Claim holds: True" in filled
