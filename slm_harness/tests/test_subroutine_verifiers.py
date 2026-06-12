"""Verifiers reject wrong answers, parsers reject malformed output, rules run."""

from __future__ import annotations

import pytest

from slm_harness.subroutines.registry import GENERATORS, REGISTRY


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_parser_rejects_garbage(name):
    sub = REGISTRY[name]
    for bad in ("", "not json", '{"unexpected": 1}', "[1,2,3]"):
        payload, err = sub.parse_output(bad)
        assert payload is None
        assert err


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_parser_tolerates_fences(name, mini_index):
    sub = REGISTRY[name]
    ex = GENERATORS[name]([mini_index], "train", 1, seed=3)[0]
    fenced = "```json\n" + sub.render_target(ex) + "\n```"
    payload, err = sub.parse_output(fenced)
    assert payload is not None, err
    assert sub.verify(ex, payload)


def _perturb(name: str, target: dict) -> dict:
    """A wrong-but-schema-valid output for each subroutine."""
    wrong = dict(target)
    if name == "json_repair":
        wrong = {"action": "ESCALATE", "reason": "wrong on purpose"}
    elif name == "action_router":
        wrong["next_action"] = "ESCALATE" if target["next_action"] != "ESCALATE" else "SEARCH"
    elif name == "path_normalizer":
        wrong["path"] = "definitely/not/a/real_file.py"
    elif name == "search_query_gen":
        wrong["pattern"] = "zzz_no_such_token_zzz"
    elif name == "search_hit_ranker":
        wrong["choice"] = target["choice"] + 1
    elif name == "read_span_selector":
        wrong = {"start": target["start"] + 30, "end": target["end"] + 60}
    elif name == "evidence_judge":
        wrong = ({"decision": "continue"} if target["decision"] == "answer"
                 else {"decision": "answer", "path": "nope/wrong.py"})
    elif name == "trace_localizer":
        wrong = dict(target, line=target["line"] + 5)
    return wrong


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_verifier_rejects_wrong_output(name, mini_index):
    sub = REGISTRY[name]
    ex = GENERATORS[name]([mini_index], "test", 1, seed=4)[0]
    assert not sub.verify(ex, _perturb(name, ex["target"]))


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_rules_baseline_runs(name, mini_index):
    """Rules must run on every example and either abstain or emit verifiable output."""
    sub = REGISTRY[name]
    rows = GENERATORS[name]([mini_index], "test", 4, seed=5)
    for ex in rows:
        out = sub.rules_baseline(ex)
        if out is not None:
            sub.verify(ex, out)  # must not raise


def test_json_repair_corruptions_recoverable(mini_index):
    """Every corruption layer combination keeps the oracle target reachable."""
    sub = REGISTRY["json_repair"]
    rows = GENERATORS["json_repair"]([mini_index], "train", 40, seed=6)
    for ex in rows:
        gold_payload, err = sub.parse_output(sub.render_target(ex))
        assert gold_payload is not None, err
        assert sub.verify(ex, gold_payload)
