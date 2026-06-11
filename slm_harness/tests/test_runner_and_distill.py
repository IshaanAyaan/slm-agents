"""End-to-end: runner across conditions + distillation builder, all offline."""

from __future__ import annotations

from pathlib import Path

from slm_harness.conditions import default_experiment_config
from slm_harness.distill import build_distillation_dataset
from slm_harness.metrics import compute_condition_metrics
from slm_harness.runner import run_experiment
from slm_harness.schemas.runlog import load_jsonl
from slm_harness.schemas.task import TaskSuite


async def test_run_experiment_all_conditions(
    smoke_tasks_path: Path, tmp_path: Path
) -> None:
    config = default_experiment_config(
        str(smoke_tasks_path),
        provider="stub",
        seeds=[0],
        output_dir=str(tmp_path / "out"),
    )
    records = await run_experiment(config, splits=["test"], quiet=True)
    # 6 conditions x 2 test tasks x 1 seed
    assert len(records) == 12
    assert {r.condition_id for r in records} == {"C1", "C2", "C3", "C4", "C5", "C6"}
    # JSONL round-trips.
    loaded = load_jsonl(tmp_path / "out" / "runs.jsonl")
    assert len(loaded) == 12
    assert (tmp_path / "out" / "config_snapshot.json").is_file()
    # Every record carries usage + cost.
    for r in records:
        assert r.input_tokens > 0 and r.cost_usd >= 0
        assert r.steps, "steps must be logged"
    # Skill-1.0-equivalents: large model (0.95 skill) should mostly succeed.
    metrics = compute_condition_metrics(records)
    assert metrics["C6"].success_rate >= 0.5


async def test_distillation_builder(smoke_tasks_path: Path, tmp_path: Path) -> None:
    config = default_experiment_config(
        str(smoke_tasks_path),
        provider="stub",
        seeds=[0, 1],
        output_dir=str(tmp_path / "out"),
    )
    # Make the teacher perfect so trajectories are clean.
    config.models["large_general"].stub_skill = 1.0
    records = await run_experiment(config, condition_ids=["C1", "C6"], quiet=True)
    suite = TaskSuite.load(smoke_tasks_path)
    counts = await build_distillation_dataset(
        records, suite, out_dir=tmp_path / "distill"
    )
    assert counts["train"] > 0
    assert sum(counts.values()) > 0

    # Validate output format with the trainer's own loader.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
    from train_lora import load_chat_jsonl

    rows = load_chat_jsonl(str(tmp_path / "distill" / "train.jsonl"))
    first = rows[0]["messages"]
    assert first[0]["role"] == "system"
    import json

    obs = json.loads(first[1]["content"])  # observation is valid JSON
    assert "goal" in obs and "valid_actions" in obs
    action = json.loads(first[2]["content"])  # action is valid JSON
    assert action["action"] in {"SEARCH", "READ", "ANSWER", "ESCALATE"}


async def test_figures_generation(smoke_tasks_path: Path, tmp_path: Path) -> None:
    from slm_harness.figures import generate_all

    config = default_experiment_config(
        str(smoke_tasks_path), provider="stub", seeds=[0], output_dir=str(tmp_path / "out")
    )
    records = await run_experiment(config, splits=["test"], quiet=True)
    written = generate_all(records, tmp_path / "figs", synthetic=True)
    names = {p.name for p in written}
    assert "fig1_success_by_condition.png" in names
    assert "fig2_cost_per_success.png" in names
    assert "fig3_attribution_success.png" in names
    assert "fig5_break_even.png" in names
    assert "tables.md" in names
    tables = (tmp_path / "figs" / "tables.md").read_text(encoding="utf-8")
    assert "Table 2 — Attribution decomposition" in tables
    assert "SYNTHETIC" in tables.upper() or "STUB" in tables.upper()


async def test_distillation_val_split_never_starved(
    smoke_tasks_path: Path, tmp_path: Path
) -> None:
    """All-train-declared source tasks must still yield a non-empty SFT val split.

    Regression for the 2026-06-10 real run: every distillation source task is
    declared split="train" (the benchmark split), which starved val.jsonl and
    aborted LoRA training.
    """
    config = default_experiment_config(
        str(smoke_tasks_path),
        provider="stub",
        seeds=[0, 1],
        output_dir=str(tmp_path / "out"),
    )
    config.models["large_general"].stub_skill = 1.0
    records = await run_experiment(config, condition_ids=["C6"], quiet=True)
    suite = TaskSuite.load(smoke_tasks_path)
    for task in suite.tasks:
        task.split = "train"  # mirror the real benchmark: all sources are train
    counts = await build_distillation_dataset(
        records, suite, out_dir=tmp_path / "distill"
    )
    assert counts["train"] > 0
    assert counts["val"] > 0, "val split must never be empty when train has examples"
