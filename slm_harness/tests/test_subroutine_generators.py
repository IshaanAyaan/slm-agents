"""Generators produce schema-complete, oracle-consistent, reproducible examples."""

from __future__ import annotations

import json

import pytest

from slm_harness.subroutines.registry import GENERATORS, REGISTRY

N = 6


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_generator_contract(name, mini_index):
    rows = GENERATORS[name]([mini_index], "train", N, seed=0)
    assert len(rows) == N
    ids = {r["id"] for r in rows}
    assert len(ids) == N, "ids must be unique"
    for r in rows:
        assert r["subroutine"] == name
        assert r["split"] == "train"
        assert isinstance(r["input"], dict) and r["input"]
        assert isinstance(r["target"], dict)
        assert "repo" in r["meta"]
        json.dumps(r)  # JSONL-serializable


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_generator_deterministic(name, mini_index):
    a = GENERATORS[name]([mini_index], "val", 3, seed=7)
    b = GENERATORS[name]([mini_index], "val", 3, seed=7)
    assert a == b


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_gold_passes_own_verifier(name, mini_index):
    """The rendered gold target must round-trip through parse + verify."""
    sub = REGISTRY[name]
    for ex in GENERATORS[name]([mini_index], "test", N, seed=1):
        gold_raw = sub.render_target(ex)
        payload, err = sub.parse_output(gold_raw)
        assert payload is not None, f"{ex['id']}: gold did not parse: {err}"
        assert sub.verify(ex, payload), f"{ex['id']}: gold failed verification"


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_chat_render(name, mini_index):
    sub = REGISTRY[name]
    ex = GENERATORS[name]([mini_index], "train", 1, seed=2)[0]
    chat = sub.to_chat(ex)
    roles = [m["role"] for m in chat["messages"]]
    assert roles == ["system", "user", "assistant"]
    json.loads(chat["messages"][2]["content"])  # assistant turn is strict JSON
