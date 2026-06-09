"""Metric calculations + attribution decomposition + claim/break-even checks."""

from __future__ import annotations

import math

import pytest

from research.slm_harness.metrics import (
    attribution_decomposition,
    break_even,
    compute_condition_metrics,
    full_report,
    primary_claim_check,
)
from research.slm_harness.schemas.runlog import RunRecord


def _rec(cid: str, ok: bool, cost: float, **kw) -> RunRecord:
    return RunRecord(
        run_id="r",
        condition_id=cid,
        task_id=kw.pop("task_id", "t"),
        seed=kw.pop("seed", 0),
        model_profile="m",
        harness_profile="h",
        success=ok,
        cost_usd=cost,
        **kw,
    )


def _grid(success_rates: dict[str, float], costs: dict[str, float], n: int = 10) -> list[RunRecord]:
    records = []
    for cid in success_rates:
        n_ok = round(success_rates[cid] * n)
        for i in range(n):
            records.append(_rec(cid, i < n_ok, costs[cid], seed=i))
    return records


def test_condition_metrics_basics() -> None:
    records = [
        _rec("C1", True, 0.02, input_tokens=1000, output_tokens=100, retries=1),
        _rec("C1", False, 0.03, input_tokens=2000, output_tokens=200, escalated=True),
    ]
    m = compute_condition_metrics(records)["C1"]
    assert m.n_attempts == 2 and m.n_success == 1
    assert m.success_rate == pytest.approx(0.5)
    assert m.total_cost_usd == pytest.approx(0.05)
    assert m.cost_per_attempt == pytest.approx(0.025)
    assert m.cost_per_success == pytest.approx(0.05)  # all cost / successes
    assert m.retry_rate == pytest.approx(0.5)
    assert m.escalation_rate == pytest.approx(0.5)
    assert m.input_tokens_total == 3000


def test_cost_per_success_infinite_when_no_success() -> None:
    m = compute_condition_metrics([_rec("C2", False, 0.01)])["C2"]
    assert math.isinf(m.cost_per_success)


def test_attribution_decomposition_hand_computed() -> None:
    # success: C2=0.4, C3=0.7, C4=0.6, C5=0.95, C1=0.9, C6=0.92
    records = _grid(
        {"C1": 0.9, "C2": 0.4, "C3": 0.7, "C4": 0.6, "C5": 0.95, "C6": 0.9},
        {c: 0.01 for c in ["C1", "C2", "C3", "C4", "C5", "C6"]},
        n=20,
    )
    cm = compute_condition_metrics(records)
    attr = attribution_decomposition(cm, "success_rate", higher_is_better=True)
    assert attr.harness_effect_small == pytest.approx(0.30)
    assert attr.finetune_effect == pytest.approx(0.20)
    assert attr.additive_pred_c5 == pytest.approx(0.90)
    assert attr.interaction == pytest.approx(0.05)
    assert attr.superadditive  # 0.95 > 0.90
    assert attr.harness_effect_large == pytest.approx(0.0)


def test_attribution_cost_metric_lower_is_better() -> None:
    records = _grid(
        {c: 1.0 for c in ["C1", "C2", "C3", "C4", "C5", "C6"]},
        {"C1": 0.10, "C2": 0.06, "C3": 0.03, "C4": 0.04, "C5": 0.005, "C6": 0.05},
        n=10,
    )
    cm = compute_condition_metrics(records)
    attr = attribution_decomposition(cm, "cost_per_success", higher_is_better=False)
    # additive prediction: 0.03 + 0.04 - 0.06 = 0.01; observed 0.005 < pred -> super-additive
    assert attr.additive_pred_c5 == pytest.approx(0.01)
    assert attr.interaction == pytest.approx(-0.005)
    assert attr.superadditive


def test_attribution_handles_missing_conditions() -> None:
    cm = compute_condition_metrics([_rec("C1", True, 0.01)])
    attr = attribution_decomposition(cm)
    assert "C5" in attr.missing_conditions and attr.interaction == 0.0


def test_primary_claim_check() -> None:
    records = _grid(
        {"C1": 0.9, "C5": 0.89},
        {"C1": 0.10, "C5": 0.02},
        n=100,
    )
    cm = compute_condition_metrics(records)
    claim = primary_claim_check(cm)
    # cost/success: C1 = 0.10*100/90 = 0.1111; C5 = 0.02*100/89 = 0.02247
    assert claim.cost_reduction_pct == pytest.approx(79.78, abs=0.1)
    assert claim.success_drop_pp == pytest.approx(1.0, abs=1e-6)
    assert claim.claim_holds


def test_primary_claim_fails_on_success_drop() -> None:
    records = _grid({"C1": 0.9, "C5": 0.8}, {"C1": 0.10, "C5": 0.02}, n=10)
    claim = primary_claim_check(compute_condition_metrics(records))
    assert claim.meets_cost_target and not claim.meets_success_target
    assert not claim.claim_holds


def test_break_even() -> None:
    records = _grid({"C1": 1.0, "C5": 1.0}, {"C1": 0.10, "C5": 0.02}, n=10)
    cm = compute_condition_metrics(records)
    be = break_even(cm, finetune_cost_usd=400.0)
    assert be.savings_per_task_usd == pytest.approx(0.08)
    assert be.breakeven_tasks == pytest.approx(5000.0)


def test_full_report_is_json_safe() -> None:
    import json

    records = [_rec("C2", False, 0.01)]  # infinite cost/success
    report = full_report(records, finetune_cost_usd=100.0)
    json.dumps(report)  # must not raise
    assert report["conditions"]["C2"]["cost_per_success"] is None
