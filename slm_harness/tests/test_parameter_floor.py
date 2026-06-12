"""Parameter-floor aggregation: verdicts, Wilson CIs, markdown rendering."""

from __future__ import annotations

from slm_harness.evals.parameter_floor import (
    SIZE_ORDER,
    build_floor,
    render_markdown,
    wilson_ci,
)
from slm_harness.subroutines.registry import REGISTRY


def _cell(name, size, cond, success, valid=1.0, n=200):
    return {
        "subroutine": name, "size": size, "condition": cond, "n": n,
        "schema_valid": valid, "success": success, "success_retry": success,
        "usage": {"tokens_in": 1000, "tokens_out": 500, "wall_seconds": 20.0},
    }


def test_wilson_ci_sane():
    lo, hi = wilson_ci(180, 200)
    assert 0.84 < lo < 0.90 < hi < 0.94
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_floor_picks_smallest_passing_size():
    name = sorted(REGISTRY)[0]
    cells = {}
    for size, s in zip(SIZE_ORDER, [0.50, 0.93, 0.97, 0.99]):
        cells[(name, size, "ft")] = _cell(name, size, "ft", s)
        cells[(name, size, "base")] = _cell(name, size, "base", 0.1)
    rules = {name: {"success": 0.40}}
    floor = build_floor(cells, rules, gpu_usd_hr=3.19)
    assert floor[name]["floor"] == SIZE_ORDER[1]
    assert floor[name]["floor_params_m"] == 362
    assert floor[name]["verdict"].startswith("works at 362M")
    md = render_markdown(floor)
    assert name in md and "362M" in md


def test_rules_suffice_verdict():
    name = sorted(REGISTRY)[0]
    cells = {(name, SIZE_ORDER[0], "ft"): _cell(name, SIZE_ORDER[0], "ft", 0.91)}
    rules = {name: {"success": 0.95}}
    floor = build_floor(cells, rules, gpu_usd_hr=3.19)
    # Model never beats rules by the margin, but rules clear the bar alone.
    assert floor[name]["floor"] is None
    assert floor[name]["verdict"] == "rules suffice"


def test_unsolved_verdict():
    name = sorted(REGISTRY)[0]
    cells = {(name, s, "ft"): _cell(name, s, "ft", 0.4) for s in SIZE_ORDER}
    floor = build_floor(cells, {name: {"success": 0.2}}, gpu_usd_hr=3.19)
    assert floor[name]["floor"] is None
    assert floor[name]["verdict"].startswith("unsolved")
