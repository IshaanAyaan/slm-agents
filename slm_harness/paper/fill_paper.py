"""Render the white paper from live metrics JSON.

Reads the headline (4B) and optional ablation (1.7B) metrics produced by the real run
and writes a complete white-paper markdown with the actual numbers substituted in.
Any value that is not yet available is rendered as an explicit ``[PENDING RUN]`` token,
so an un-filled paper is never mistaken for a finished one.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

TEMPLATE = Path(__file__).with_name("white_paper.template.md")


def _g(d: dict, *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _num(x: Any, fmt: str = "{:.3f}", pending: str = "[PENDING RUN]") -> str:
    if x is None or (isinstance(x, float) and math.isinf(x)):
        return pending
    try:
        return fmt.format(x)
    except (ValueError, TypeError):
        return pending


def _cond(metrics: dict, cid: str, field: str, fmt: str = "{:.3f}") -> str:
    return _num(_g(metrics, "conditions", cid, field), fmt)


def render(headline: dict | None, ablation: dict | None, figures_dir: str) -> str:
    tpl = TEMPLATE.read_text(encoding="utf-8")
    m = headline or {}
    claim = _g(m, "primary_claim", default={}) or {}
    attr_s = _g(m, "attribution", "success_rate", default={}) or {}
    attr_c = _g(m, "attribution", "cost_per_success", default={}) or {}
    be = _g(m, "break_even", default={}) or {}

    subs = {
        "{{FIGDIR}}": figures_dir,
        "{{STATUS}}": "FINAL (real self-hosted run)" if headline else "DRAFT — [PENDING RUN]",
        # headline conditions
        **{f"{{{{SR_{c}}}}}": _cond(m, c, "success_rate") for c in
           ["C1", "C2", "C3", "C4", "C5", "C6"]},
        **{f"{{{{CPS_{c}}}}}": _cond(m, c, "cost_per_success", "{:.6f}") for c in
           ["C1", "C2", "C3", "C4", "C5", "C6"]},
        **{f"{{{{CPA_{c}}}}}": _cond(m, c, "cost_per_attempt", "{:.6f}") for c in
           ["C1", "C2", "C3", "C4", "C5", "C6"]},
        # claim
        "{{COST_REDUCTION_PCT}}": _num(claim.get("cost_reduction_pct"), "{:.1f}"),
        "{{SUCCESS_DROP_PP}}": _num(claim.get("success_drop_pp"), "{:.1f}"),
        "{{CLAIM_HOLDS}}": str(claim.get("claim_holds", "[PENDING RUN]")),
        # attribution (success)
        "{{HARNESS_EFFECT_SR}}": _num(attr_s.get("harness_effect_small"), "{:+.3f}"),
        "{{FINETUNE_EFFECT_SR}}": _num(attr_s.get("finetune_effect"), "{:+.3f}"),
        "{{ADDITIVE_PRED_SR}}": _num(attr_s.get("additive_pred_c5"), "{:.3f}"),
        "{{INTERACTION_SR}}": _num(attr_s.get("interaction"), "{:+.3f}"),
        "{{SUPERADD_SR}}": str(attr_s.get("superadditive", "[PENDING RUN]")),
        # attribution (cost)
        "{{HARNESS_EFFECT_CPS}}": _num(attr_c.get("harness_effect_small"), "{:+.6f}"),
        "{{FINETUNE_EFFECT_CPS}}": _num(attr_c.get("finetune_effect"), "{:+.6f}"),
        "{{INTERACTION_CPS}}": _num(attr_c.get("interaction"), "{:+.6f}"),
        "{{SUPERADD_CPS}}": str(attr_c.get("superadditive", "[PENDING RUN]")),
        # break-even
        "{{FT_COST}}": _num(be.get("finetune_cost_usd"), "{:.2f}"),
        "{{SAVINGS_PER_TASK}}": _num(be.get("savings_per_task_usd"), "{:.6f}"),
        "{{BREAKEVEN_TASKS}}": _num(be.get("breakeven_tasks"), "{:.0f}"),
        # ablation
        "{{SR_C5_17B}}": _cond(ablation or {}, "C5", "success_rate") if ablation else "[PENDING RUN]",
        "{{CPS_C5_17B}}": _cond(ablation or {}, "C5", "cost_per_success", "{:.6f}") if ablation else "[PENDING RUN]",
    }
    out = tpl
    for k, v in subs.items():
        out = out.replace(k, v)
    return out


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--headline", default=None, help="metrics_4b.json")
    p.add_argument("--ablation", default=None, help="metrics_17b.json")
    p.add_argument("--figures", default="../results/real/figures_4b")
    p.add_argument("--out", default="slm_harness/paper/white_paper.md")
    args = p.parse_args()

    headline = json.loads(Path(args.headline).read_text()) if args.headline and Path(args.headline).is_file() else None
    ablation = json.loads(Path(args.ablation).read_text()) if args.ablation and Path(args.ablation).is_file() else None
    text = render(headline, ablation, args.figures)
    Path(args.out).write_text(text, encoding="utf-8")
    status = "with REAL numbers" if headline else "as DRAFT (no metrics yet)"
    print(f"wrote {args.out} {status}")


if __name__ == "__main__":
    main()
