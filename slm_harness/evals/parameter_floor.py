"""Aggregate eval cells into the parameter-floor map, cost model, and figures.

The floor for a subroutine is the smallest model whose fine-tuned success (with the
single schema-retry the harness allows) clears the reliability bar AND meaningfully
beats the rules-only baseline. Subroutines where rules alone clear the bar are
flagged "rules suffice" — an honest "no model needed" verdict.

Usage:
    python -m slm_harness.evals.parameter_floor \
        --eval .../subroutines/eval --data .../subroutines/data \
        --out .../subroutines [--gpu-usd-hr 3.19] [--figures]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from slm_harness.subroutines.registry import MODEL_GRID, REGISTRY

RELIABILITY_BAR = 0.90   # success_retry needed to call a size "working"
RULES_MARGIN = 0.02      # model must beat rules by at least this to matter
SIZE_ORDER = ["smollm2-135m", "smollm2-360m", "qwen2.5-0.5b", "qwen2.5-1.5b"]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant-pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * 0.5 ** n
    return min(1.0, 2 * tail)


def rules_per_item(data_dir: Path, name: str) -> dict[str, bool]:
    """Deterministic per-example rules outcomes on the test split (recomputable)."""
    sub = REGISTRY[name]
    outcomes: dict[str, bool] = {}
    test_path = data_dir / name / "test.jsonl"
    for line in test_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        ex = json.loads(line)
        out = sub.rules_baseline(ex)
        outcomes[ex["id"]] = bool(out is not None and sub.verify(ex, out))
    return outcomes


def paired_vs_rules(cell: dict, rules_items: dict[str, bool]) -> dict:
    """Exact McNemar of the fine-tuned cell against rules on identical items."""
    b = c = both = neither = 0
    for rec in cell.get("records", []):
        ft_ok = rec["success"] or rec.get("retry", {}).get("success", False)
        r_ok = rules_items.get(rec["id"])
        if r_ok is None:
            continue
        if ft_ok and not r_ok:
            b += 1
        elif r_ok and not ft_ok:
            c += 1
        elif ft_ok:
            both += 1
        else:
            neither += 1
    return {"ft_only": b, "rules_only": c, "both": both, "neither": neither,
            "p_mcnemar": round(mcnemar_exact(b, c), 6)}


def load_cells(eval_dir: Path) -> dict[tuple[str, str, str], dict]:
    cells = {}
    for path in sorted(eval_dir.glob("*__*__*.json")):
        cell = json.loads(path.read_text(encoding="utf-8"))
        cells[(cell["subroutine"], cell["size"], cell["condition"])] = cell
    return cells


def build_floor(cells: dict, rules: dict, gpu_usd_hr: float,
                data_dir: Path | None = None) -> dict:
    """Per-subroutine size curves, verdicts, and cost-per-success."""
    out: dict[str, dict] = {}
    for name in sorted(REGISTRY):
        rules_items = (rules_per_item(data_dir, name)
                       if data_dir and (data_dir / name / "test.jsonl").is_file()
                       else None)
        sizes: dict[str, dict] = {}
        for size in SIZE_ORDER:
            entry: dict = {}
            for cond in ("ft", "base"):
                cell = cells.get((name, size, cond))
                if cell is None:
                    continue
                n = cell["n"]
                k = round(cell["success_retry"] * n)
                lo, hi = wilson_ci(k, n)
                secs = cell["usage"]["wall_seconds"]
                succ_per_item_cost = (
                    (secs / n) * gpu_usd_hr / 3600 / cell["success_retry"]
                    if cell["success_retry"] > 0 else None)
                entry[cond] = {
                    "success": cell["success"],
                    "success_retry": cell["success_retry"],
                    "schema_valid": cell["schema_valid"],
                    "ci95": [round(lo, 4), round(hi, 4)],
                    "n": n,
                    "gpu_seconds": secs,
                    "latency_s": round(secs / n, 4),
                    "cost_per_success_usd": (round(succ_per_item_cost, 8)
                                             if succ_per_item_cost else None),
                    "tokens_out": cell["usage"]["tokens_out"],
                }
                if cond == "ft" and rules_items is not None:
                    entry[cond]["vs_rules"] = paired_vs_rules(cell, rules_items)
            if entry:
                sizes[size] = entry

        rules_succ = rules.get(name, {}).get("success", 0.0)
        floor = None
        for size in SIZE_ORDER:
            ft = sizes.get(size, {}).get("ft")
            if ft and ft["success_retry"] >= RELIABILITY_BAR and \
               ft["success_retry"] >= rules_succ + RULES_MARGIN:
                floor = size
                break
        best_size, best = None, -1.0
        for size in SIZE_ORDER:
            ft = sizes.get(size, {}).get("ft")
            if ft and ft["success_retry"] > best:
                best, best_size = ft["success_retry"], size
        rules_sufficient = rules_succ >= RELIABILITY_BAR
        if floor is None:
            verdict = ("rules suffice" if rules_sufficient
                       else f"unsolved (best {best:.2f} @ {best_size})")
        else:
            params = MODEL_GRID[floor]["params_m"]
            verdict = (f"works at {params}M" if params <= 500
                       else f"needs {params}M")
            if rules_sufficient:
                verdict += " (rules also suffice)"
        out[name] = {
            "description": REGISTRY[name].description,
            "rules_success": rules_succ,
            "rules_sufficient": rules_sufficient,
            "sizes": sizes,
            "floor": floor,
            "floor_params_m": MODEL_GRID[floor]["params_m"] if floor else None,
            "best_size": best_size,
            "best_success_retry": best,
            "verdict": verdict,
        }
    return out


def render_markdown(floor: dict) -> str:
    head = ("| Subroutine | Rules | " +
            " | ".join(f"{MODEL_GRID[s]['params_m']}M FT" for s in SIZE_ORDER) +
            " | Floor | Verdict |")
    sep = "|" + "---|" * (len(SIZE_ORDER) + 4)
    lines = [head, sep]
    for name, row in floor.items():
        cells = []
        for size in SIZE_ORDER:
            ft = row["sizes"].get(size, {}).get("ft")
            cells.append(f"{ft['success_retry']:.3f}" if ft else "—")
        floor_label = (f"{row['floor_params_m']}M" if row["floor"] else "—")
        lines.append(f"| {name} | {row['rules_success']:.3f} | " +
                     " | ".join(cells) +
                     f" | {floor_label} | {row['verdict']} |")
    return "\n".join(lines)


def make_figures(floor: dict, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # Success vs parameter count, one line per subroutine (ft, with retry).
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = [MODEL_GRID[s]["params_m"] for s in SIZE_ORDER]
    for name, row in floor.items():
        ys = [row["sizes"].get(s, {}).get("ft", {}).get("success_retry")
              for s in SIZE_ORDER]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                label=name, linewidth=1.6, markersize=4)
    ax.axhline(RELIABILITY_BAR, color="gray", linestyle="--", linewidth=1,
               label=f"reliability bar ({RELIABILITY_BAR:.2f})")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x}M" for x in xs])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("specialist parameters")
    ax.set_ylabel("success (1 retry) on held-out repos")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.set_title("Parameter floor: fine-tuned specialist success vs size")
    fig.tight_layout()
    p = out_dir / "parameter_floor.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    written.append(str(p))

    # FT-vs-base gap at each size (what fine-tuning buys, averaged).
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for cond, style in (("ft", "-o"), ("base", "--s")):
        ys = []
        for s in SIZE_ORDER:
            vals = [row["sizes"].get(s, {}).get(cond, {}).get("success_retry")
                    for row in floor.values()]
            vals = [v for v in vals if v is not None]
            ys.append(sum(vals) / len(vals) if vals else None)
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        ax.plot([p_[0] for p_ in pts], [p_[1] for p_ in pts], style,
                label=f"{cond} (mean over subroutines)")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x}M" for x in xs])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("parameters")
    ax.set_ylabel("mean success (1 retry)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.set_title("Fine-tuning is the floor-setter: FT vs base instruct")
    fig.tight_layout()
    p = out_dir / "ft_vs_base.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    written.append(str(p))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu-usd-hr", type=float, default=3.19)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()

    cells = load_cells(Path(args.eval))
    summary = json.loads((Path(args.data) / "datagen_summary.json")
                         .read_text(encoding="utf-8"))
    rules = {k: v["rules_test"] for k, v in summary.items()}
    floor = build_floor(cells, rules, args.gpu_usd_hr, data_dir=Path(args.data))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parameter_floor.json").write_text(
        json.dumps(floor, indent=2), encoding="utf-8")
    md = render_markdown(floor)
    (out_dir / "parameter_floor.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    if args.figures:
        for p in make_figures(floor, out_dir / "figures"):
            print(f"[fig] {p}")


if __name__ == "__main__":
    main()
