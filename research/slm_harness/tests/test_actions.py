"""JSON action parsing for the custom harness."""

from __future__ import annotations

import json

from research.slm_harness.schemas.actions import (
    AnswerAction,
    EscalateAction,
    ReadAction,
    SearchAction,
    parse_action,
)


def test_parse_each_action_type() -> None:
    a, err = parse_action('{"action":"SEARCH","pattern":"class Foo"}')
    assert err == "" and isinstance(a, SearchAction) and a.file_glob == "**/*"

    a, err = parse_action('{"action":"READ","path":"a/b.py","offset":10,"limit":50}')
    assert err == "" and isinstance(a, ReadAction) and a.offset == 10

    a, err = parse_action(
        json.dumps(
            {
                "action": "ANSWER",
                "files": ["a.py"],
                "evidence": [{"path": "a.py", "line_start": 1, "line_end": 3}],
                "confidence": 0.8,
            }
        )
    )
    assert err == "" and isinstance(a, AnswerAction)

    a, err = parse_action('{"action":"ESCALATE","reason":"stuck"}')
    assert err == "" and isinstance(a, EscalateAction)


def test_parse_tolerates_code_fences_and_prose() -> None:
    a, err = parse_action('```json\n{"action":"SEARCH","pattern":"x"}\n```')
    assert err == "" and isinstance(a, SearchAction)
    a, err = parse_action('Sure! Here you go: {"action":"SEARCH","pattern":"x"}')
    assert err == "" and isinstance(a, SearchAction)


def test_parse_failures_are_descriptive() -> None:
    for raw, fragment in [
        ("", "empty"),
        ("let me think about this", "no JSON object"),
        ('{"action":"SEARCH","pattern":', "no JSON object"),
        ('{"pattern":"x"}', "missing required field 'action'"),
        ('{"action":"FLY","speed":1}', "schema error"),
        ('{"action":"READ"}', "schema error"),
        ('{"action":"ANSWER","files":[]}', "schema error"),
    ]:
        a, err = parse_action(raw)
        assert a is None and fragment in err, (raw, err)


def test_confidence_bounds() -> None:
    a, err = parse_action('{"action":"ANSWER","files":["a.py"],"confidence":1.5}')
    assert a is None and "confidence" in err
