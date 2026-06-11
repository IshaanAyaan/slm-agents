"""Tests for the dedup + significance analysis (research.slm_harness.analysis)."""

from __future__ import annotations

import json
import math

from research.slm_harness.analysis import (
    bootstrap_cost_ratio,
    load_unique_attempts,
    mcnemar_exact,
    paired_success_test,
    significance_report,
    summarize_condition,
    wilson_interval,
)


def _record(cond: str, task: str, seed: int, success: bool, cost: float, output: str) -> dict:
    return {
        "condition_id": cond,
        "task_id": task,
        "seed": seed,
        "success": success,
        "cost_usd": cost,
        "steps": [{"model_output": output}],
    }


def _write_runs(tmp_path, records):
    path = tmp_path / "runs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_dedup_collapses_identical_replays_and_counts_divergent(tmp_path):
    records = [
        # task t0: identical replay -> one attempt, one identical replay
        _record("C1", "t0", 0, True, 0.01, "same"),
        _record("C1", "t0", 1, True, 0.01, "same"),
        # task t1: divergent rerun -> keeps seed 0, counts divergence
        _record("C1", "t1", 0, False, 0.02, "a"),
        _record("C1", "t1", 1, True, 0.02, "b"),
    ]
    attempts, dedup = load_unique_attempts([_write_runs(tmp_path, records)])
    assert len(attempts) == 2
    kept = {a.task_id: a for a in attempts}
    assert kept["t1"].seed == 0 and kept["t1"].success is False
    assert dedup["C1"].tasks == 2
    assert dedup["C1"].replays_identical == 1
    assert dedup["C1"].replays_divergent == 1


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(39, 40)
    assert lo < 39 / 40 < hi
    # Wilson for 39/40 at z=1.96: roughly [0.871, 0.996]
    assert math.isclose(lo, 0.8714, abs_tol=5e-3)
    assert math.isclose(hi, 0.9957, abs_tol=5e-3)
    # degenerate cases stay in [0, 1]
    assert wilson_interval(0, 40)[0] == 0.0
    assert wilson_interval(40, 40)[1] == 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_mcnemar_exact_matches_binomial_tail():
    # b=10, c=2: p = 2 * P(X >= 10 | n=12, 1/2) = 2*(66+12+1)/4096
    expected = 2 * (66 + 12 + 1) / 4096
    assert math.isclose(mcnemar_exact(10, 2), expected, rel_tol=1e-12)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == 1.0  # symmetric discordance is not evidence


def test_paired_test_and_summary(tmp_path):
    records = []
    # C5 succeeds on 9/10 tasks, C1 on 5/10; C1's successes are a subset of C5's.
    for i in range(10):
        records.append(_record("C5", f"t{i}", 0, i != 9, 0.001, f"o{i}"))
        records.append(_record("C1", f"t{i}", 0, i < 5, 0.004, f"o{i}"))
    attempts, dedup = load_unique_attempts([_write_runs(tmp_path, records)])
    by_cond = {}
    for a in attempts:
        by_cond.setdefault(a.condition_id, []).append(a)

    s5 = summarize_condition(by_cond["C5"], dedup.get("C5"))
    assert s5.n_tasks == 10 and s5.n_success == 9
    assert s5.cost_per_success is not None
    assert math.isclose(s5.cost_per_success, 0.01 / 9, rel_tol=1e-9)

    cmp = paired_success_test(by_cond["C5"], by_cond["C1"])
    assert cmp.a_only_success == 4 and cmp.b_only_success == 0
    assert cmp.mcnemar_p == 2 * (1 / 16)  # 2 * P(X>=4 | n=4, 1/2), capped later

    boot = bootstrap_cost_ratio(by_cond["C5"], by_cond["C1"], n_resamples=200, rng_seed=7)
    assert boot.point_ratio is not None and boot.point_ratio < 1.0
    assert boot.reduction_pct_point is not None and boot.reduction_pct_point > 0
    # deterministic under a fixed rng seed
    boot2 = bootstrap_cost_ratio(by_cond["C5"], by_cond["C1"], n_resamples=200, rng_seed=7)
    assert boot.ci95 == boot2.ci95


def test_significance_report_end_to_end(tmp_path):
    records = []
    for i in range(8):
        records.append(_record("C1", f"t{i}", 0, i < 6, 0.004, f"x{i}"))
        records.append(_record("C1", f"t{i}", 1, i < 6, 0.004, f"x{i}"))
        records.append(_record("C5", f"t{i}", 0, i < 7, 0.001, f"y{i}"))
    report = significance_report(
        [_write_runs(tmp_path, records)], n_resamples=50, rng_seed=1
    )
    assert report["conditions"]["C1"]["n_tasks"] == 8
    assert report["conditions"]["C1"]["replays_identical"] == 8
    assert report["conditions"]["C5"]["replays_identical"] == 0
    pairs = {(p["condition_a"], p["condition_b"]) for p in report["paired_success"]}
    assert ("C5", "C1") in pairs
    json.dumps(report)  # must be JSON-serializable
