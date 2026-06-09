"""Figure and numbered-table generation for the paper.

All functions take the metrics report (``metrics.full_report`` output or RunRecords)
and write PNG figures + markdown/LaTeX tables. Figures generated from STUB runs are
watermarked as synthetic pipeline-validation output.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from research.slm_harness.metrics import (
    ConditionMetrics,
    attribution_decomposition,
    break_even,
    compute_condition_metrics,
    primary_claim_check,
)
from research.slm_harness.schemas.runlog import RunRecord

CONDITION_ORDER = ["C1", "C2", "C3", "C4", "C5", "C6"]
CONDITION_LABELS = {
    "C1": "C1\nlarge+generic",
    "C2": "C2\nsmall+generic",
    "C3": "C3\nsmall+custom",
    "C4": "C4\nFT+generic",
    "C5": "C5\nFT+custom",
    "C6": "C6\nlarge+custom",
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _watermark(ax: Any, synthetic: bool) -> None:
    if synthetic:
        ax.text(
            0.5,
            0.5,
            "SYNTHETIC (stub models)\npipeline validation only",
            transform=ax.transAxes,
            fontsize=18,
            color="red",
            alpha=0.18,
            ha="center",
            va="center",
            rotation=20,
        )


def _ordered(metrics: dict[str, ConditionMetrics]) -> list[ConditionMetrics]:
    return [metrics[c] for c in CONDITION_ORDER if c in metrics]


def condition_bar_chart(
    metrics: dict[str, ConditionMetrics],
    out_path: str | Path,
    *,
    synthetic: bool = True,
) -> Path:
    """Figure 1: success rate per condition with binomial stderr bars."""
    plt = _plt()
    ms = _ordered(metrics)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = range(len(ms))
    ax.bar(
        xs,
        [m.success_rate for m in ms],
        yerr=[m.success_rate_stderr for m in ms],
        capsize=4,
        color=["#4878a8" if m.condition_id != "C5" else "#d1495b" for m in ms],
    )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([CONDITION_LABELS.get(m.condition_id, m.condition_id) for m in ms])
    ax.set_ylabel("Task success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Success rate by condition")
    _watermark(ax, synthetic)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def cost_per_success_chart(
    metrics: dict[str, ConditionMetrics],
    out_path: str | Path,
    *,
    synthetic: bool = True,
) -> Path:
    """Figure 2: cost per successful task (the headline metric)."""
    plt = _plt()
    ms = _ordered(metrics)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = [m.cost_per_success if math.isfinite(m.cost_per_success) else 0.0 for m in ms]
    bars = ax.bar(
        range(len(ms)),
        vals,
        color=["#4878a8" if m.condition_id != "C5" else "#d1495b" for m in ms],
    )
    for bar, m in zip(bars, ms):
        label = "∞" if not math.isfinite(m.cost_per_success) else f"${m.cost_per_success:.4f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([CONDITION_LABELS.get(m.condition_id, m.condition_id) for m in ms])
    ax.set_ylabel("Cost per successful task (USD)")
    ax.set_title("Cost per successful task by condition")
    _watermark(ax, synthetic)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def attribution_figure(
    metrics: dict[str, ConditionMetrics],
    out_path: str | Path,
    *,
    metric: str = "success_rate",
    higher_is_better: bool = True,
    synthetic: bool = True,
) -> Path:
    """Figure 3: factorial decomposition (harness, fine-tune, interaction)."""
    plt = _plt()
    attr = attribution_decomposition(metrics, metric, higher_is_better)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    names = [
        "Harness effect\n(C3−C2)",
        "Fine-tune effect\n(C4−C2)",
        "Additive pred. C5\n(C3+C4−C2)",
        "Observed C5",
        "Interaction\n(C5−pred)",
    ]
    vals = [
        attr.harness_effect_small,
        attr.finetune_effect,
        attr.additive_pred_c5,
        attr.values.get("C5", float("nan")),
        attr.interaction,
    ]
    colors = ["#4878a8", "#4878a8", "#999999", "#d1495b", "#2a9d8f" if attr.superadditive else "#e9c46a"]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(
        f"Attribution decomposition on {metric} "
        f"({'super-additive' if attr.superadditive else 'not super-additive'})"
    )
    _watermark(ax, synthetic)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def data_efficiency_curve(
    points: list[tuple[int, float]] | None,
    out_path: str | Path,
    *,
    synthetic: bool = True,
) -> Path:
    """Figure 4: success vs number of fine-tuning examples.

    PLACEHOLDER until multiple fine-tuned checkpoints exist. Pass real
    (n_train_examples, success_rate) points once LoRA checkpoints trained on
    nested data subsets have been evaluated; with ``points=None`` an explicitly
    labeled empty placeholder axis is produced.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if points:
        xs, ys = zip(*sorted(points))
        ax.plot(xs, ys, marker="o")
    else:
        ax.text(
            0.5,
            0.5,
            "PLACEHOLDER\nrequires checkpoints trained on nested subsets\n"
            "(see training/README.md, step 4)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="gray",
        )
    ax.set_xlabel("Fine-tuning examples")
    ax.set_ylabel("C5 success rate (held-out)")
    ax.set_title("Data-efficiency curve")
    ax.set_ylim(0, 1.05)
    _watermark(ax, synthetic)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def break_even_curve(
    metrics: dict[str, ConditionMetrics],
    out_path: str | Path,
    *,
    finetune_costs_usd: tuple[float, ...] = (50.0, 100.0, 250.0, 500.0, 1000.0),
    synthetic: bool = True,
) -> Path:
    """Figure 5: tasks-to-break-even vs one-time fine-tuning cost."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs, ys = [], []
    for cost in finetune_costs_usd:
        be = break_even(metrics, cost)
        if math.isfinite(be.breakeven_tasks):
            xs.append(cost)
            ys.append(be.breakeven_tasks)
    if xs:
        ax.plot(xs, ys, marker="o")
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.text(
            0.5,
            0.5,
            "No positive per-task savings (C5 ≥ C1 cost/success)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
    ax.set_xlabel("One-time fine-tuning cost (USD)")
    ax.set_ylabel("Tasks to break even")
    ax.set_title("Break-even analysis")
    _watermark(ax, synthetic)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# ----------------------------- numbered tables -----------------------------


def _fmt_cost(v: float) -> str:
    return "—" if not math.isfinite(v) else f"{v:.5f}"


def results_tables_markdown(
    metrics: dict[str, ConditionMetrics], *, synthetic: bool = True
) -> str:
    """Tables 1-3 in markdown, ready for paper insertion."""
    ms = _ordered(metrics)
    note = (
        "\n> **NOTE: numbers below were produced with STUB model clients and are "
        "synthetic pipeline-validation output, not research results.**\n"
        if synthetic
        else ""
    )
    t1 = [
        "## Table 1 — Main results by condition",
        note,
        "| Condition | n | Success rate | Cost/attempt (USD) | Cost/success (USD) | Escalation rate |",
        "|---|---|---|---|---|---|",
    ]
    for m in ms:
        t1.append(
            f"| {m.condition_id} | {m.n_attempts} | {m.success_rate:.3f} ± {m.success_rate_stderr:.3f} "
            f"| {m.cost_per_attempt:.5f} | {_fmt_cost(m.cost_per_success)} | {m.escalation_rate:.3f} |"
        )

    attr_s = attribution_decomposition(metrics, "success_rate", True)
    attr_c = attribution_decomposition(metrics, "cost_per_success", False)
    t2 = [
        "",
        "## Table 2 — Attribution decomposition",
        "",
        "| Effect | Success rate | Cost/success (USD) |",
        "|---|---|---|",
        f"| Harness effect, small model (C3−C2) | {attr_s.harness_effect_small:+.3f} | {attr_c.harness_effect_small:+.5f} |",
        f"| Fine-tuning effect (C4−C2) | {attr_s.finetune_effect:+.3f} | {attr_c.finetune_effect:+.5f} |",
        f"| Additive prediction for C5 (C3+C4−C2) | {attr_s.additive_pred_c5:.3f} | {attr_c.additive_pred_c5:.5f} |",
        f"| Observed C5 | {attr_s.values.get('C5', float('nan')):.3f} | {attr_c.values.get('C5', float('nan')):.5f} |",
        f"| Interaction (C5 − additive pred.) | {attr_s.interaction:+.3f} | {attr_c.interaction:+.5f} |",
        f"| Super-additive? | {attr_s.superadditive} | {attr_c.superadditive} |",
        f"| Harness effect, large model (C6−C1) | {attr_s.harness_effect_large:+.3f} | {attr_c.harness_effect_large:+.5f} |",
    ]

    t3 = [
        "",
        "## Table 3 — Token and reliability breakdown",
        "",
        "| Condition | In-tok/attempt | Out-tok/attempt | Retry rate | Invalid-action rate | Verifier-failure rate | p50 latency (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in ms:
        t3.append(
            f"| {m.condition_id} | {m.input_tokens_mean:.0f} | {m.output_tokens_mean:.0f} "
            f"| {m.retry_rate:.2f} | {m.invalid_action_rate:.2f} | {m.verifier_failure_rate:.2f} "
            f"| {m.latency_p50_s:.2f} |"
        )

    claim = primary_claim_check(metrics)
    t4 = [
        "",
        "## Table 4 — Primary claim check (C5 vs C1)",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| C1 cost/success (USD) | {_fmt_cost(claim.c1_cost_per_success)} |",
        f"| C5 cost/success (USD) | {_fmt_cost(claim.c5_cost_per_success)} |",
        f"| Cost reduction | {claim.cost_reduction_pct:.1f}% (target ≥ 50%) |",
        f"| Success drop | {claim.success_drop_pp:.1f} pp (target ≤ 2 pp) |",
        f"| **Claim holds** | **{claim.claim_holds}** |",
    ]
    return "\n".join(t1 + t2 + t3 + t4) + "\n"


def generate_all(
    records: list[RunRecord],
    out_dir: str | Path,
    *,
    synthetic: bool = True,
    finetune_cost_usd: float = 250.0,
) -> list[Path]:
    """Generate every figure + the tables file. Returns written paths."""
    metrics = compute_condition_metrics(records)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = [
        condition_bar_chart(metrics, out / "fig1_success_by_condition.png", synthetic=synthetic),
        cost_per_success_chart(metrics, out / "fig2_cost_per_success.png", synthetic=synthetic),
        attribution_figure(
            metrics, out / "fig3_attribution_success.png", metric="success_rate", synthetic=synthetic
        ),
        attribution_figure(
            metrics,
            out / "fig3b_attribution_cost.png",
            metric="cost_per_success",
            higher_is_better=False,
            synthetic=synthetic,
        ),
        data_efficiency_curve(None, out / "fig4_data_efficiency_PLACEHOLDER.png", synthetic=synthetic),
        break_even_curve(metrics, out / "fig5_break_even.png", synthetic=synthetic),
    ]
    tables = out / "tables.md"
    tables.write_text(results_tables_markdown(metrics, synthetic=synthetic), encoding="utf-8")
    written.append(tables)
    return written
