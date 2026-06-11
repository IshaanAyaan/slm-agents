"""Metrics, 2x3 factorial attribution decomposition, and break-even analysis.

The methodological core: separating the harness effect, the fine-tuning effect, and
their interaction across C1-C6, plus the super-additivity check
``C5 vs (C3 + C4 - C2)`` and the primary claim check
``C5 cuts cost-per-successful-task >= 50% vs C1 with <= 2pp success drop``.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from slm_harness.schemas.runlog import RunRecord


@dataclass
class ConditionMetrics:
    """Aggregates for one condition."""

    condition_id: str
    n_attempts: int = 0
    n_success: int = 0
    success_rate: float = 0.0
    total_cost_usd: float = 0.0
    cost_per_attempt: float = 0.0
    cost_per_success: float = math.inf
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    input_tokens_mean: float = 0.0
    output_tokens_mean: float = 0.0
    retry_rate: float = 0.0  # retries per attempt
    invalid_action_rate: float = 0.0  # invalid actions per attempt
    verifier_failure_rate: float = 0.0  # verifier failures per attempt
    escalation_rate: float = 0.0  # fraction of attempts escalated
    latency_mean_s: float = 0.0
    latency_p50_s: float = 0.0
    latency_p95_s: float = 0.0
    success_rate_stderr: float = 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def compute_condition_metrics(records: list[RunRecord]) -> dict[str, ConditionMetrics]:
    """Group records by condition and compute all aggregates."""
    by_cond: dict[str, list[RunRecord]] = {}
    for r in records:
        by_cond.setdefault(r.condition_id, []).append(r)

    out: dict[str, ConditionMetrics] = {}
    for cid, recs in sorted(by_cond.items()):
        n = len(recs)
        n_success = sum(1 for r in recs if r.success)
        total_cost = sum(r.cost_usd for r in recs)
        latencies = [r.latency_s for r in recs]
        p = n_success / n if n else 0.0
        m = ConditionMetrics(
            condition_id=cid,
            n_attempts=n,
            n_success=n_success,
            success_rate=p,
            total_cost_usd=total_cost,
            cost_per_attempt=total_cost / n if n else 0.0,
            cost_per_success=(total_cost / n_success) if n_success else math.inf,
            input_tokens_total=sum(r.input_tokens for r in recs),
            output_tokens_total=sum(r.output_tokens for r in recs),
            input_tokens_mean=statistics.mean([r.input_tokens for r in recs]) if recs else 0.0,
            output_tokens_mean=statistics.mean([r.output_tokens for r in recs]) if recs else 0.0,
            retry_rate=sum(r.retries for r in recs) / n if n else 0.0,
            invalid_action_rate=sum(r.invalid_actions for r in recs) / n if n else 0.0,
            verifier_failure_rate=sum(r.verifier_failures for r in recs) / n if n else 0.0,
            escalation_rate=sum(1 for r in recs if r.escalated) / n if n else 0.0,
            latency_mean_s=statistics.mean(latencies) if latencies else 0.0,
            latency_p50_s=_percentile(latencies, 0.5),
            latency_p95_s=_percentile(latencies, 0.95),
            success_rate_stderr=math.sqrt(p * (1 - p) / n) if n else 0.0,
        )
        out[cid] = m
    return out


@dataclass
class Attribution:
    """2x3 factorial attribution on a chosen metric (default: success_rate).

    Effects within the small-model 2x2 sub-design {C2, C3, C4, C5}:
      harness_effect    = C3 - C2   (custom harness, holding small general model fixed)
      finetune_effect   = C4 - C2   (fine-tuning, holding generic harness fixed)
      additive_pred_C5  = C3 + C4 - C2
      interaction       = C5 - additive_pred_C5  (super-additive iff > 0 for benefit
                          metrics; < 0 for cost metrics)
    Plus the large-model harness effect: C6 - C1.
    """

    metric: str
    values: dict[str, float] = field(default_factory=dict)
    harness_effect_small: float = 0.0
    finetune_effect: float = 0.0
    additive_pred_c5: float = 0.0
    interaction: float = 0.0
    superadditive: bool = False
    harness_effect_large: float = 0.0
    missing_conditions: list[str] = field(default_factory=list)


def attribution_decomposition(
    metrics: dict[str, ConditionMetrics],
    metric: str = "success_rate",
    higher_is_better: bool = True,
) -> Attribution:
    """Compute the factorial decomposition on one scalar metric."""
    attr = Attribution(metric=metric)
    needed = ["C1", "C2", "C3", "C4", "C5", "C6"]
    vals: dict[str, float] = {}
    for cid in needed:
        if cid in metrics:
            vals[cid] = float(getattr(metrics[cid], metric))
        else:
            attr.missing_conditions.append(cid)
    attr.values = vals
    if any(c in attr.missing_conditions for c in ("C2", "C3", "C4", "C5")):
        return attr
    attr.harness_effect_small = vals["C3"] - vals["C2"]
    attr.finetune_effect = vals["C4"] - vals["C2"]
    attr.additive_pred_c5 = vals["C3"] + vals["C4"] - vals["C2"]
    attr.interaction = vals["C5"] - attr.additive_pred_c5
    attr.superadditive = (
        attr.interaction > 0 if higher_is_better else attr.interaction < 0
    )
    if "C1" in vals and "C6" in vals:
        attr.harness_effect_large = vals["C6"] - vals["C1"]
    return attr


@dataclass
class PrimaryClaim:
    """C5 vs C1 headline check."""

    c1_cost_per_success: float = math.inf
    c5_cost_per_success: float = math.inf
    cost_reduction_pct: float = 0.0
    c1_success_rate: float = 0.0
    c5_success_rate: float = 0.0
    success_drop_pp: float = 0.0
    meets_cost_target: bool = False  # >= 50% reduction
    meets_success_target: bool = False  # <= 2pp drop
    claim_holds: bool = False


def primary_claim_check(
    metrics: dict[str, ConditionMetrics],
    *,
    cost_reduction_target_pct: float = 50.0,
    max_success_drop_pp: float = 2.0,
) -> PrimaryClaim:
    """Evaluate the headline claim from condition metrics."""
    claim = PrimaryClaim()
    if "C1" not in metrics or "C5" not in metrics:
        return claim
    c1, c5 = metrics["C1"], metrics["C5"]
    claim.c1_cost_per_success = c1.cost_per_success
    claim.c5_cost_per_success = c5.cost_per_success
    claim.c1_success_rate = c1.success_rate
    claim.c5_success_rate = c5.success_rate
    claim.success_drop_pp = (c1.success_rate - c5.success_rate) * 100.0
    if math.isfinite(c1.cost_per_success) and c1.cost_per_success > 0:
        claim.cost_reduction_pct = (
            (c1.cost_per_success - c5.cost_per_success) / c1.cost_per_success * 100.0
        )
    claim.meets_cost_target = claim.cost_reduction_pct >= cost_reduction_target_pct
    claim.meets_success_target = claim.success_drop_pp <= max_success_drop_pp
    claim.claim_holds = claim.meets_cost_target and claim.meets_success_target
    return claim


@dataclass
class BreakEven:
    """One-time fine-tuning cost amortization."""

    finetune_cost_usd: float = 0.0
    savings_per_task_usd: float = 0.0  # C1 cost/success - C5 cost/success
    breakeven_tasks: float = math.inf


def break_even(
    metrics: dict[str, ConditionMetrics], finetune_cost_usd: float
) -> BreakEven:
    """Tasks needed for fine-tuning cost to be recovered by per-task savings."""
    be = BreakEven(finetune_cost_usd=finetune_cost_usd)
    if "C1" not in metrics or "C5" not in metrics:
        return be
    c1, c5 = metrics["C1"], metrics["C5"]
    if not (math.isfinite(c1.cost_per_success) and math.isfinite(c5.cost_per_success)):
        return be
    be.savings_per_task_usd = c1.cost_per_success - c5.cost_per_success
    if be.savings_per_task_usd > 0:
        be.breakeven_tasks = finetune_cost_usd / be.savings_per_task_usd
    return be


def full_report(
    records: list[RunRecord], finetune_cost_usd: float = 0.0
) -> dict[str, Any]:
    """All metrics in one JSON-serializable dict."""
    cm = compute_condition_metrics(records)

    def _ser(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {
                k: _ser(getattr(obj, k)) for k in obj.__dataclass_fields__
            }
        if isinstance(obj, dict):
            return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, float) and math.isinf(obj):
            return None  # JSON-safe
        return obj

    return {
        "conditions": {cid: _ser(m) for cid, m in cm.items()},
        "attribution": {
            "success_rate": _ser(attribution_decomposition(cm, "success_rate", True)),
            "cost_per_success": _ser(
                attribution_decomposition(cm, "cost_per_success", False)
            ),
            "cost_per_attempt": _ser(
                attribution_decomposition(cm, "cost_per_attempt", False)
            ),
        },
        "primary_claim": _ser(primary_claim_check(cm)),
        "break_even": _ser(break_even(cm, finetune_cost_usd)),
    }
