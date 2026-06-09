"""Task schema parsing + experiment config validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from research.slm_harness.conditions import default_experiment_config
from research.slm_harness.schemas.config import PricingConfig
from research.slm_harness.schemas.task import NavTask, TaskSuite


def test_task_suite_loads_and_resolves_workspaces(smoke_tasks_path: Path) -> None:
    suite = TaskSuite.load(smoke_tasks_path)
    assert suite.suite_id == "smoke-nav-v1"
    assert len(suite.tasks) == 10
    for task in suite.tasks:
        assert Path(task.workspace).is_absolute()
        assert Path(task.workspace).is_dir()
        for f in task.expected_files:
            assert (Path(task.workspace) / f).is_file(), f"{task.task_id}: missing {f}"


def test_task_splits(smoke_tasks_path: Path) -> None:
    suite = TaskSuite.load(smoke_tasks_path)
    assert len(suite.by_split("train")) == 6
    assert len(suite.by_split("val")) == 2
    assert len(suite.by_split("test")) == 2


def test_task_defaults() -> None:
    task = NavTask(task_id="t", workspace="w", prompt="find `x`")
    assert task.verification.method == "expected_files"
    assert "grep" in task.allowed_tools


def test_pricing_cost() -> None:
    pricing = PricingConfig(usd_per_million_input=3.0, usd_per_million_output=15.0)
    assert pricing.cost_usd(1_000_000, 0) == pytest.approx(3.0)
    assert pricing.cost_usd(0, 2_000_000) == pytest.approx(30.0)
    assert pricing.cost_usd(500, 100) == pytest.approx(0.003, rel=1e-3)


def test_experiment_config_cross_references(smoke_tasks_path: Path) -> None:
    config = default_experiment_config(str(smoke_tasks_path))
    assert set(config.conditions) == {"C1", "C2", "C3", "C4", "C5", "C6"}
    # invalid reference must raise
    broken = config.model_dump()
    broken["conditions"]["C1"]["model_profile_id"] = "nope"
    from research.slm_harness.schemas.config import ExperimentConfig

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(broken)


def test_finetuned_profile_flag(smoke_tasks_path: Path) -> None:
    config = default_experiment_config(str(smoke_tasks_path))
    assert config.models["small_finetuned"].is_finetuned
