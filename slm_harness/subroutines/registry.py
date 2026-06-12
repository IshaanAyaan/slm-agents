"""Registry of all Phase-2 subroutines plus the model-size grid."""

from __future__ import annotations

from slm_harness.subroutines import (
    action_router,
    evidence_judge,
    json_repair,
    path_normalizer,
    read_span_selector,
    search_hit_ranker,
    search_query_gen,
    trace_localizer,
)
from slm_harness.subroutines.base import Subroutine

REGISTRY: dict[str, Subroutine] = {
    s.name: s
    for s in [
        json_repair.JsonRepair(),
        action_router.ActionRouter(),
        path_normalizer.PathNormalizer(),
        search_query_gen.SearchQueryGen(),
        search_hit_ranker.SearchHitRanker(),
        read_span_selector.ReadSpanSelector(),
        evidence_judge.EvidenceJudge(),
        trace_localizer.TraceLocalizer(),
    ]
}

GENERATORS = {
    "json_repair": json_repair.generate,
    "action_router": action_router.generate,
    "path_normalizer": path_normalizer.generate,
    "search_query_gen": search_query_gen.generate,
    "search_hit_ranker": search_hit_ranker.generate,
    "read_span_selector": read_span_selector.generate,
    "evidence_judge": evidence_judge.generate,
    "trace_localizer": trace_localizer.generate,
}

#: Phase-2 model grid. Sub-500M is the target; 1.5B is the upper-bound control.
MODEL_GRID: dict[str, dict] = {
    "smollm2-135m": {"hf_id": "HuggingFaceTB/SmolLM2-135M-Instruct", "params_m": 135},
    "smollm2-360m": {"hf_id": "HuggingFaceTB/SmolLM2-360M-Instruct", "params_m": 362},
    "qwen2.5-0.5b": {"hf_id": "Qwen/Qwen2.5-0.5B-Instruct", "params_m": 494},
    "qwen2.5-1.5b": {"hf_id": "Qwen/Qwen2.5-1.5B-Instruct", "params_m": 1544},
}


def get(name: str) -> Subroutine:
    """Look up one subroutine by registry key."""
    if name not in REGISTRY:
        raise KeyError(f"unknown subroutine: {name} (known: {sorted(REGISTRY)})")
    return REGISTRY[name]
