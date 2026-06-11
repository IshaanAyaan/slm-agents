"""Statistical analysis over real run logs: dedup, exact CIs, paired tests.

The real experiment decodes greedily (temperature 0) and never threads the
nominal seed into the model client, so re-running a task under a second "seed"
mostly replays the identical trajectory (any divergence comes from vLLM batch
nondeterminism, not controlled sampling). Treating those replays as independent
attempts overstates precision by ~sqrt(2). This module therefore:

1. keeps exactly one attempt per (condition, task) — the lowest seed — and
   reports how many replays were byte-identical vs divergent;
2. computes Wilson 95% intervals for per-condition success on unique tasks;
3. runs exact McNemar tests on paired per-task success between conditions;
4. bootstraps (over tasks) the cost-per-success ratio between conditions.

Pure stdlib: safe to run anywhere the run logs exist, no GPU or extra deps.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

Z95 = 1.959963984540054


# ----------------------------------------------------------------------
# Loading and deduplication
# ----------------------------------------------------------------------


@dataclass
class Attempt:
    """One unique (condition, task) attempt kept for analysis."""

    condition_id: str
    task_id: str
    seed: int
    success: bool
    cost_usd: float


@dataclass
class DedupStats:
    """How the nominal seed replays related to the kept attempt."""

    tasks: int = 0
    replays_identical: int = 0
    replays_divergent: int = 0


def _trajectory_fingerprint(record: dict[str, Any]) -> tuple:
    """Outputs that must match for two runs to be considered the same trajectory."""
    return (
        record.get("success"),
        tuple(step.get("model_output", "") for step in record.get("steps", [])),
    )


def load_unique_attempts(
    paths: Iterable[str | Path],
) -> tuple[list[Attempt], dict[str, DedupStats]]:
    """Read runs JSONL files and keep one attempt per (condition, task).

    The lowest-seed record is kept. Higher-seed records are classified as
    byte-identical replays or divergent reruns (vLLM nondeterminism).
    """
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                by_key[(record["condition_id"], record["task_id"])].append(record)

    attempts: list[Attempt] = []
    dedup: dict[str, DedupStats] = defaultdict(DedupStats)
    for (condition_id, task_id), records in sorted(by_key.items()):
        records.sort(key=lambda r: r.get("seed", 0))
        kept = records[0]
        stats = dedup[condition_id]
        stats.tasks += 1
        kept_fp = _trajectory_fingerprint(kept)
        for extra in records[1:]:
            if _trajectory_fingerprint(extra) == kept_fp:
                stats.replays_identical += 1
            else:
                stats.replays_divergent += 1
        attempts.append(
            Attempt(
                condition_id=condition_id,
                task_id=task_id,
                seed=int(kept.get("seed", 0)),
                success=bool(kept["success"]),
                cost_usd=float(kept.get("cost_usd", 0.0)),
            )
        )
    return attempts, dict(dedup)


# ----------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant-pair counts.

    b = tasks where A succeeded and B failed; c = the reverse. Under H0 the
    discordant pairs are Binomial(b+c, 1/2).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


@dataclass
class ConditionSummary:
    """Per-condition statistics over unique attempts."""

    condition_id: str
    n_tasks: int
    n_success: int
    success_rate: float
    success_ci95: tuple[float, float]
    total_cost_usd: float
    cost_per_attempt: float
    cost_per_success: float | None
    replays_identical: int = 0
    replays_divergent: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["success_ci95"] = list(self.success_ci95)
        return d


def summarize_condition(
    attempts: list[Attempt], dedup: DedupStats | None = None
) -> ConditionSummary:
    """Success and cost statistics for one condition's unique attempts."""
    n = len(attempts)
    wins = sum(a.success for a in attempts)
    cost = sum(a.cost_usd for a in attempts)
    return ConditionSummary(
        condition_id=attempts[0].condition_id,
        n_tasks=n,
        n_success=wins,
        success_rate=wins / n if n else 0.0,
        success_ci95=wilson_interval(wins, n),
        total_cost_usd=cost,
        cost_per_attempt=cost / n if n else 0.0,
        cost_per_success=cost / wins if wins else None,
        replays_identical=dedup.replays_identical if dedup else 0,
        replays_divergent=dedup.replays_divergent if dedup else 0,
    )


@dataclass
class PairedComparison:
    """Paired per-task comparison between two conditions on shared tasks."""

    condition_a: str
    condition_b: str
    n_shared_tasks: int
    a_only_success: int  # discordant: A succeeded, B failed
    b_only_success: int  # discordant: B succeeded, A failed
    mcnemar_p: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def paired_success_test(
    attempts_a: list[Attempt], attempts_b: list[Attempt]
) -> PairedComparison:
    """Exact McNemar test of success between two conditions on shared tasks."""
    a_by_task = {a.task_id: a.success for a in attempts_a}
    b_by_task = {a.task_id: a.success for a in attempts_b}
    shared = sorted(set(a_by_task) & set(b_by_task))
    b_count = sum(1 for t in shared if a_by_task[t] and not b_by_task[t])
    c_count = sum(1 for t in shared if b_by_task[t] and not a_by_task[t])
    return PairedComparison(
        condition_a=attempts_a[0].condition_id,
        condition_b=attempts_b[0].condition_id,
        n_shared_tasks=len(shared),
        a_only_success=b_count,
        b_only_success=c_count,
        mcnemar_p=mcnemar_exact(b_count, c_count),
    )


@dataclass
class BootstrapRatio:
    """Bootstrap CI for the cost-per-success ratio A/B over shared tasks."""

    condition_a: str
    condition_b: str
    point_ratio: float | None
    ci95: tuple[float | None, float | None]
    reduction_pct_point: float | None
    reduction_pct_ci95: tuple[float | None, float | None]
    n_resamples: int
    n_degenerate: int  # resamples where either condition had zero successes
    rng_seed: int

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["ci95"] = list(self.ci95)
        d["reduction_pct_ci95"] = list(self.reduction_pct_ci95)
        return d


def bootstrap_cost_ratio(
    attempts_a: list[Attempt],
    attempts_b: list[Attempt],
    n_resamples: int = 10_000,
    rng_seed: int = 0,
) -> BootstrapRatio:
    """Percentile bootstrap (resampling tasks) of cost-per-success A/B.

    Tasks are resampled jointly so the per-task pairing (same benchmark item
    under both conditions) is preserved.
    """
    a_by_task = {a.task_id: a for a in attempts_a}
    b_by_task = {a.task_id: a for a in attempts_b}
    shared = sorted(set(a_by_task) & set(b_by_task))
    rng = random.Random(rng_seed)

    def cps(attempts: list[Attempt]) -> float | None:
        wins = sum(a.success for a in attempts)
        if wins == 0:
            return None
        return sum(a.cost_usd for a in attempts) / wins

    point_a = cps([a_by_task[t] for t in shared])
    point_b = cps([b_by_task[t] for t in shared])
    point = (point_a / point_b) if point_a is not None and point_b is not None else None

    ratios: list[float] = []
    degenerate = 0
    for _ in range(n_resamples):
        sample = [shared[rng.randrange(len(shared))] for _ in shared]
        ra = cps([a_by_task[t] for t in sample])
        rb = cps([b_by_task[t] for t in sample])
        if ra is None or rb is None:
            degenerate += 1
            continue
        ratios.append(ra / rb)
    ratios.sort()

    def pct(q: float) -> float | None:
        if not ratios:
            return None
        idx = min(len(ratios) - 1, max(0, int(round(q * (len(ratios) - 1)))))
        return ratios[idx]

    lo, hi = pct(0.025), pct(0.975)
    to_red = lambda r: None if r is None else (1.0 - r) * 100.0  # noqa: E731
    return BootstrapRatio(
        condition_a=attempts_a[0].condition_id,
        condition_b=attempts_b[0].condition_id,
        point_ratio=point,
        ci95=(lo, hi),
        reduction_pct_point=to_red(point),
        # ratio CI flips: low ratio = high reduction
        reduction_pct_ci95=(to_red(hi), to_red(lo)),
        n_resamples=n_resamples,
        n_degenerate=degenerate,
        rng_seed=rng_seed,
    )


# ----------------------------------------------------------------------
# Full report
# ----------------------------------------------------------------------


def significance_report(
    paths: Iterable[str | Path],
    comparisons: list[tuple[str, str]] | None = None,
    n_resamples: int = 10_000,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """End-to-end: load, dedup, summarize, test, bootstrap. JSON-serializable."""
    attempts, dedup = load_unique_attempts(paths)
    by_cond: dict[str, list[Attempt]] = defaultdict(list)
    for a in attempts:
        by_cond[a.condition_id].append(a)

    if comparisons is None:
        comparisons = [("C5", "C1"), ("C5", "C2"), ("C2", "C1"), ("C5", "C4"), ("C5", "C3")]
    comparisons = [(a, b) for a, b in comparisons if a in by_cond and b in by_cond]

    return {
        "method": {
            "dedup": "one attempt per (condition, task), lowest seed kept",
            "success_ci": "Wilson 95%",
            "paired_test": "exact two-sided McNemar on shared tasks",
            "cost_ci": f"paired percentile bootstrap over tasks, {n_resamples} resamples",
            "rng_seed": rng_seed,
        },
        "conditions": {
            cond: summarize_condition(atts, dedup.get(cond)).to_dict()
            for cond, atts in sorted(by_cond.items())
        },
        "paired_success": [
            paired_success_test(by_cond[a], by_cond[b]).to_dict() for a, b in comparisons
        ],
        "cost_ratio_bootstrap": [
            bootstrap_cost_ratio(by_cond[a], by_cond[b], n_resamples, rng_seed).to_dict()
            for a, b in comparisons
        ],
    }
